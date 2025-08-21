import numpy as np
import cv2
import rasterio
from scipy.ndimage import gaussian_gradient_magnitude, laplace
from skimage.filters import rank
from skimage.morphology import disk
import triangle as tr
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

import geopandas as gpd
from shapely.geometry import LineString, Point



def save_edges_to_vector_file(binary_image, transform, output_file, file_format="ESRI Shapefile"):
    """
    Save binary edge image as vector (lines) in GeoJSON or Shapefile.

    :param binary_image: np.uint8 edge map (0/1 or 0/255)
    :param transform: rasterio Affine transform from original TIFF
    :param output_file: path to output file (with .geojson or .shp extension)
    :param file_format: "GeoJSON" or "ESRI Shapefile"
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    geometries = []

    for contour in contours:
        coords = []
        for point in contour:
            x_pix, y_pix = point[0]
            x_geo, y_geo = transform * (x_pix, y_pix)  # Pixel to geo coords
            coords.append((x_geo, y_geo))

        if len(coords) > 1:
            geometries.append(LineString(coords))

    gdf = gpd.GeoDataFrame(geometry=geometries)
    gdf.set_crs(epsg=4326, inplace=True)  # You can set your specific CRS if known

    gdf.to_file(output_file, driver="GeoJSON" if file_format == "GeoJSON" else "ESRI Shapefile")
    print(f"Saved vector edges to {output_file}")

def save_points_to_shapefile(binary_image, transform, output_path):
    coords = np.column_stack(np.where(binary_image > 0))
    geometries = [Point(transform * (x, y)) for y, x in coords]
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    gdf.to_file(output_path, driver="ESRI Shapefile")
    print(f"Saved point shapefile: {output_path}")

# Function to read TIFF files
def read_tiff(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
        return data


# Function to normalize the data for better processing
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize to [0, 1]


# Edge detection methods
def sobel_gradient_detection(data, threshold):
    grad_x = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return (gradient_magnitude > threshold).astype(np.uint8)


def gaussian_gradient_detection(data, sigma, threshold):
    gradient = gaussian_gradient_magnitude(data, sigma=sigma)
    return (gradient > threshold).astype(np.uint8)


def canny_edge_detection(data, lower_threshold, upper_threshold):
    data_scaled = (data * 255).astype(np.uint8)
    return cv2.Canny(data_scaled, lower_threshold, upper_threshold)


def laplacian_detection(data, threshold):
    laplace_result = np.abs(laplace(data))
    return (laplace_result > threshold).astype(np.uint8)


def local_variance_detection(data, radius, threshold):
    data_scaled = (data * 255).astype(np.uint8)
    local_variance = rank.entropy(data_scaled, disk(radius))
    return (local_variance > threshold).astype(np.uint8)


def save_vector_points_to_obj(binary_image, data, output_file):
    """Edge points → OBJ, perfectly aligned with full_dem_mesh.obj"""
    import numpy as np
    pts = np.column_stack(np.where(binary_image > 0))

    verts = []
    for y, x in pts:
        z_scaled = data[y, x] / 30          # ❶ scale only, no minus sign
        verts.append((x, -y, z_scaled))     # ❷ flip Y, keep Z positive

    min_z = min(v[2] for v in verts)        # ❸ same ground shift

    with open(output_file, "w") as f:
        f.write("# edge points aligned with DEM\n")
        for X, Yneg, Z in verts:            # Yneg = -y
            f.write(f"v {X} {Z - min_z} {-Yneg}\n")  # X  (height)  Y



def save_edges_as_obj(binary_image, data, output_file, original_vertex_lookup=None):
    """Edge contours → OBJ lines, aligned with full_dem_mesh.obj"""
    import cv2
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    verts, lines, vmap = [], [], {}

    for c in contours:
        seq = []
        for pt in c:
            x, y = pt[0]
            z_scaled = data[y, x] / 30                      # ❶ scale only
            key = (x, -y, z_scaled)                        # ❷ flip Y, keep Z+

            if key not in vmap:
                vmap[key] = len(verts)
                verts.append(key)
            seq.append(vmap[key])
        for i in range(len(seq) - 1):
            lines.append((seq[i], seq[i+1]))

    min_z = min(v[2] for v in verts)                       # ❸ same ground shift

    with open(output_file, "w") as f:
        f.write("# edge lines aligned with DEM\n")
        for X, Yneg, Z in verts:
            f.write(f"v {X} {Z - min_z} {-Yneg}\n")        # X  (height)  Y
        for a, b in lines:
            f.write(f"l {a+1} {b+1}\n")






# Main function
def process_tiff(file_path):
    data = read_tiff(file_path)
    normalized_data = normalize_data(data)
    return data, normalized_data


def load_obj_vertices_only(file_path):
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
    return vertices

def load_edge_obj(file_path):
    vertices = []
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
            elif line.startswith('l '):
                v1, v2 = map(int, line.strip().split()[1:3])
                lines.append((v1 - 1, v2 - 1))
    return vertices, lines

def match_constraint_indices(edge_vertices, original_vertices):
    original_2d = [(round(x, 3), round(y, 3)) for x, y, z in original_vertices]
    tree = KDTree(original_2d)
    matched_indices = []
    for x, y, z in edge_vertices:
        idx = tree.query((round(x, 3), round(y, 3)))[1]
        matched_indices.append(idx)
    return matched_indices


def build_segments_from_matched(lines, matched_indices):
    return [(matched_indices[i1], matched_indices[i2]) for i1, i2 in lines]

def save_obj_with_faces(vertices, triangles, output_path):
    with open(output_path, 'w') as f:
        for x, y, z in vertices:
            f.write(f"v {x} {y} {z}\n")
        for t in triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

import rasterio

def save_tiff_as_mesh_obj(tiff_path, output_obj):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)  # Read first band
        height, width = data.shape

        vertices = []
        faces = []
        USE_METERS = False  # ← Set to False if you prefer to keep pixels and scale Z instead

        # Build vertices
        for y in range(height):
            for x in range(width):
                z = data[y, x]
                if USE_METERS:
                    X = x * 30
                    Y = y * 30
                else:
                    X = x
                    Y = y
                    z = z / 30

                vertices.append((X, -Y, z))

        # Build faces (2 triangles per cell)
        def idx(x, y):
            return y * width + x

        for y in range(height - 1):
            for x in range(width - 1):
                a = idx(x, y)
                b = idx(x + 1, y)
                c = idx(x, y + 1)
                d = idx(x + 1, y + 1)

                # Triangle 1: A-B-D
                faces.append((a, b, d))
                # Triangle 2: A-D-C
                faces.append((a, d, c))

        # Calculate minimum Z value
        min_z = min(v[2] for v in vertices)

        # Write OBJ
        with open(output_obj, 'w') as f:
            for v in vertices:
                # Blender fix: adjust Z so minimum is at 0
                f.write(f"v {v[0]} {v[2] - min_z} {-v[1]}\n")
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Saved mesh OBJ to: {output_obj}")


if __name__ == '__main__':


    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay
    import triangle as tr

    # Step 1: Create 10x10 grid and apply jitter
    n = 10
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xv, yv = np.meshgrid(x, y)
    points = np.vstack([xv.ravel(), yv.ravel()]).T

    # Apply small random noise to simulate "floating" point cloud
    np.random.seed(42)
    jitter_strength = 0.02
    points += jitter_strength * np.random.randn(*points.shape)
    # points =np.array([v[:2] for v in load_obj_vertices_only("C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture1_full/vectors_25_04/obj/gaussian_edges.obj")])
    def get_tri_edges(triangles):
        edge_set = set()
        for tri in triangles:
            a, b, c = tri
            for u, v in [(a,b), (b,c), (c,a)]:
                edge = tuple(sorted((u,v)))
                edge_set.add(edge)
        return list(edge_set)

    def find_nearest_point_idx(pt, points):
        return np.argmin(np.linalg.norm(points - pt, axis=1))

    def get_constraint_indices(points):
        constraint_coords = [
            ([0.1, 0.1], [0.8, 0.2]),
            ([0.2, 0.8], [0.9, 0.8]),
            ([0.3, 0.2], [0.7, 0.6]),
            ([0.75, 0.25], [0.75, 0.75])
        ]
        constraints = []
        for p1, p2 in constraint_coords:
            idx1 = find_nearest_point_idx(np.array(p1), points)
            idx2 = find_nearest_point_idx(np.array(p2), points)
            if idx1 != idx2:
                constraints.append([idx1, idx2])
        return constraints

    def get_bounding_square_segments(points):
        corners = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ]
        corner_indices = [find_nearest_point_idx(c, points) for c in corners]
        return [
            [corner_indices[0], corner_indices[1]],
            [corner_indices[1], corner_indices[2]],
            [corner_indices[2], corner_indices[3]],
            [corner_indices[3], corner_indices[0]]
        ]

    def plot_edges(ax, points, edges, title, constraint_edges=None):
        for a, b in edges:
            ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color='blue', linewidth=0.8)
        ax.scatter(points[:, 0], points[:, 1], color='black', s=8)
        if constraint_edges is not None and len(constraint_edges) > 0:
            for a, b in constraint_edges:
                ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color='red', linestyle='--', linewidth=2)
        ax.set_title(title)
        ax.set_aspect('equal')

    # Unconstrained Delaunay
    delaunay = Delaunay(points)
    edges_delaunay = get_tri_edges(delaunay.simplices)

    # Constrained triangulation input
    internal_constraints = get_constraint_indices(points)
    boundary_segments = get_bounding_square_segments(points)
    all_constraints = boundary_segments + internal_constraints

    A = dict(vertices=points, segments=np.array(all_constraints))
    t = tr.triangulate(A, 'p')

    if 'triangles' not in t:
        raise RuntimeError("❌ Triangulation failed with floating points.")

    edges_constrained = get_tri_edges(t['triangles'])
    # Perform constrained Delaunay triangulation
    original_vertices = load_obj_vertices_only("C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture1_full/full_dem_mesh.obj")
    vertices_2d = [(x, y) for x, y, z in original_vertices]

    tri_input = {
        'vertices': vertices_2d,
        'segments': all_constraints
    }

    result = tr.triangulate(tri_input, 'pq')

    if 'triangles' not in result or len(result['triangles']) == 0:
        raise RuntimeError("❌ Triangulation failed or produced no triangles.")

    triangles = result['triangles']


    def save_triangles_as_obj_faces(triangles, output_path):
        """
        Save triangle indices as OBJ 'f' lines (no vertices).
        Assumes triangle indices refer to the original vertex list elsewhere.
        """
        with open(output_path, 'w') as f:
            for tri in triangles:
                # OBJ is 1-based index
                f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")
    def save_complete_obj(vertices, triangles, output_path):
        with open(output_path, 'w') as f:
            for x, y, z in vertices:
                f.write(f"v {x} {y} {z}\n")
            for t in triangles:
                f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


    save_complete_obj(original_vertices, result['triangles'], "full_triangulated.obj")


    def load_obj_vertices_only(path):
        vertices = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z = line.strip().split()
                    vertices.append((float(x), float(y), float(z)))
        return vertices


    def load_obj_faces_only(path):
        faces = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("f "):
                    parts = line.strip().split()
                    face = tuple(int(p) - 1 for p in parts[1:4])  # OBJ is 1-based
                    faces.append(face)
        return faces


    def save_obj_with_faces(vertices, faces, output_path):
        with open(output_path, 'w') as f:
            for x, y, z in vertices:
                f.write(f"v {x} {y} {z}\n")
            for a, b, c in faces:
                f.write(f"f {a + 1} {b + 1} {c + 1}\n")  # back to 1-based for OBJ


    # save_obj_with_faces(vertices, faces, output_path)

    # result = tr.triangulate(tri_input, 'pq')
    # if 'triangles' not in result or len(result['triangles']) == 0:
    #     raise RuntimeError("❌ No triangles were produced.")
    #
    # save_triangles_as_obj_faces(
    #     result['triangles'],
    #     "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture1_full/vectors_25_04/obj/triangles_only.obj"
    # )
    #
    # faces = load_obj_faces_only("triangles_only.obj")  # from saved file
    # vertices = load_obj_vertices_only("full_dem_mesh.obj")
    # save_obj_with_faces(vertices, faces, "final_triangulated_fullmesh.obj")
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_edges(axes[0], points, edges_delaunay, "Delaunay (Floating Grid)")
    plot_edges(axes[1], points, edges_constrained, "Constrained Delaunay", constraint_edges=internal_constraints)

    plt.tight_layout()
    plt.show()













    # tiff_file = "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/output_hh.tif"


    # save_tiff_as_mesh_obj(tiff_file, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/full_dem_mesh.obj")


    # # # Read and process the TIFF file
    # data, normalized_data = process_tiff(tiff_file)
    # #
    # # # Apply detection methods
    # sobel_edges = sobel_gradient_detection(normalized_data, threshold=0.15)
    # gaussian_edges = gaussian_gradient_detection(normalized_data, sigma=3, threshold=0.01)
    # canny_edges = canny_edge_detection(normalized_data, lower_threshold=25, upper_threshold=75)
    # laplacian_edges = laplacian_detection(normalized_data, threshold=0.01)
    # local_variance_edges = local_variance_detection(normalized_data, radius=100, threshold=0.01)



    # with rasterio.open(tiff_file) as src:
    #     transform = src.transform

    # save_points_to_shapefile(sobel_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/sobel_points.shp")
    # save_points_to_shapefile(gaussian_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/gaussian_points.shp")
    # save_points_to_shapefile(canny_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/canny_points.shp")
    # save_points_to_shapefile(laplacian_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/laplacian_points.shp")
    # save_points_to_shapefile(local_variance_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/local_variance_points.shp")

    # save_edges_to_vector_file(sobel_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/sobel_lines.shp")
    # save_edges_to_vector_file(gaussian_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/gaussian_lines.shp")
    # save_edges_to_vector_file(canny_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/canny_lines.shp")
    # save_edges_to_vector_file(laplacian_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/laplacian_lines.shp")
    # save_edges_to_vector_file(local_variance_edges, transform, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/vec/local_variance_lines.shp")

    # save_vector_points_to_obj(sobel_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/canay_edges.obj")
    # save_vector_points_to_obj(gaussian_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/gaussian_edges.obj")
    # save_vector_points_to_obj(canny_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/canny_edges.obj")
    # save_vector_points_to_obj(laplacian_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/laplacian_edges.obj")
    # save_vector_points_to_obj(local_variance_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/local_variance_edges.obj")



    # save_edges_as_obj(sobel_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/sobel_edges_lines.obj")
    # save_edges_as_obj(gaussian_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/gaussian_edges_lines.obj")
    # save_edges_as_obj(canny_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/canny_edges_lines.obj")
    # save_edges_as_obj(laplacian_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/laplacian_edges_lines.obj")
    # save_edges_as_obj(local_variance_edges, data, "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture2_full/vectors_25_04/obj/local_variance_edges_lines.obj")






    # # save_vector_points_to_obj(normalized_data,data, "original.obj")#check if modifications were good
    # edge_teqniques = ["sobel_edges_lines.obj","gaussian_edges_lines.obj","canny_edges_lines.obj","laplacian_edges_lines.obj","local_variance_edges_lines.obj"]
    # for edge_teq in edge_teqniques:
    #     print("ffffffffffffffffffffffffffffffffff")
    #     original_obj_path = "original.obj"
    #     edge_obj_path = edge_teq
    #     output_obj_path = "triangulated_with_"+edge_teq+"_pqVersion.obj"

    #     print("Loading original and edge obj files...")
    #     original_vertices = load_obj_vertices_only(original_obj_path)

    #     edge_vertices, edge_lines = load_edge_obj(edge_obj_path)

    #     print("Matching constraint edges to original vertices...")
    #     matched_indices = match_constraint_indices(edge_vertices, original_vertices)
    #     segments = build_segments_from_matched(edge_lines, matched_indices)

    #     print("Running constrained Delaunay triangulation...")
    #     vertices_2d = [(x, y) for x, y, z in original_vertices]
    #     tri_input = {
    #         'vertices': vertices_2d,
    #         'segments': segments
    #     }

    #     result = tr.triangulate(tri_input, 'pq')
    #     triangles = result.get('triangles', [])
    #     if len(triangles) == 0:
    #         print(f"Triangulation failed or produced no triangles for {edge_teq}.")
    #         continue
    #     print(f" Generated {len(triangles)} triangles.")




    #     triangles = result['triangles']
    #     if len(triangles) < len(original_vertices) * 2:
    #         print(
    #             " Warning: Triangulated faces seem fewer than expected. Consider increasing point density or checking constraints.")

    #     save_obj_with_faces(original_vertices, triangles, output_obj_path)
    #     print(f"Saved triangulated mesh to: {output_obj_path}")



