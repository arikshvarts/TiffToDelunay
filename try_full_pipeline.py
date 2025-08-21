import numpy as np
import triangle as tr
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay


# --- Utility: Load full 3D vertices from .obj file ---
def load_obj_vertices_only(file_path):
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, z, y))#changed because y represent height
    return vertices

# --- Utility: Load edge constraints from another .obj (lines as edges) ---
def get_constraints_from_edge_obj(obj_path, points_2d):
    from scipy.spatial import cKDTree

    tree = cKDTree(points_2d)
    constraints = []

    with open(obj_path) as f:
        for line in f:
            if line.startswith('l '):
                parts = line.strip().split()[1:]
                if len(parts) != 2:
                    continue
                i1, i2 = int(parts[0]) - 1, int(parts[1]) - 1
                constraints.append((i1, i2))
    return constraints

# --- Utility: Save new .obj with original 3D vertices and triangulated faces ---
def save_obj_with_new_faces(original_vertices, new_faces, save_path):
    """
    Save mesh with vertices formatted as: v x z -y
    Applies Blender Z-normalization fix by subtracting min(z)
    """
    original_vertices = np.array(original_vertices)
    min_z = np.min(original_vertices[:, 1])  # Z is at index 1 (after you swapped Y/Z earlier)

    with open(save_path, 'w') as f:
        for v in original_vertices:
            x, z, y = v  # original format was (x, z, y)
            f.write(f"v {x} {z - min_z} {-y}\n")  # match save_tiff_as_mesh_obj convention
        for tri in new_faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

    print(f"✅ Saved mesh (Blender friendly) to: {save_path}")


def get_convex_hull_constraints(points):
    hull = ConvexHull(points)
    edges = []
    for simplex in hull.simplices:
        edges.append([simplex[0], simplex[1]])
    return edges


def get_tri_edges(triangles):
    edge_set = set()
    for tri in triangles:
        a, b, c = tri
        for u, v in [(a,b), (b,c), (c,a)]:
            edge = tuple(sorted((u,v)))
            edge_set.add(edge)
    return list(edge_set)
# --- Main Pipeline ---

# Step 1: Load DEM points with height
points = np.array([v[:2] for v in load_obj_vertices_only(
    "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture1_full/dem_points.obj")])


def get_tri_edges(triangles):
    edge_set = set()
    for tri in triangles:
        a, b, c = tri
        for u, v in [(a, b), (b, c), (c, a)]:
            edge = tuple(sorted((u, v)))
            edge_set.add(edge)
    return list(edge_set)


def find_nearest_point_idx(pt, points):
    return np.argmin(np.linalg.norm(points - pt, axis=1))


def plot_edges(ax, points, edges, title, constraint_edges=None):
    for a, b in edges:
        ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color='blue', linewidth=0.8)
    ax.scatter(points[:, 0], points[:, 1], color='black', s=8)
    if constraint_edges is not None and len(constraint_edges) > 0:
        for a, b in constraint_edges:
            ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color='red', linestyle='--',
                    linewidth=2)
    ax.set_title(title)
    ax.set_aspect('equal')


# Unconstrained Delaunay
delaunay = Delaunay(points)
edges_delaunay = get_tri_edges(delaunay.simplices)
from scipy.spatial import ConvexHull


def get_convex_hull_constraints(points):
    hull = ConvexHull(points)
    edges = []
    for simplex in hull.simplices:
        edges.append([simplex[0], simplex[1]])
    return edges




# Constrained triangulation input
internal_constraints = get_constraints_from_edge_obj(
    "C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture1_full/vectors_generated/gaussian_lines.obj",
    points
)
boundary_segments = get_convex_hull_constraints(points)  # optional but useful
all_constraints = boundary_segments + internal_constraints

A = dict(vertices=points, segments=np.array(all_constraints))
t = tr.triangulate(A, 'p')

if 'triangles' not in t:
    raise RuntimeError("❌ Triangulation failed with floating points.")

edges_constrained = get_tri_edges(t['triangles'])


# Step 5: Save final .obj with original vertices + new triangles
save_obj_with_new_faces(
    original_vertices=np.array(
        load_obj_vertices_only("C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture1_full/dem_points.obj")),
    # MUST include Z
    new_faces=t['triangles'],
    save_path="dem_retriangulated.obj"
)
