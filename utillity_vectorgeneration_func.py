import os
import numpy as np
import rasterio
import cv2
from scipy.ndimage import gaussian_gradient_magnitude, laplace
from skimage.filters import rank
from skimage.morphology import disk
import geopandas as gpd
from shapely.geometry import LineString, Point

### --- Edge Detection Methods --- ###
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

### --- Save Functions --- ###
def save_points_to_shapefile(binary_image, transform, output_path):
    coords = np.column_stack(np.where(binary_image > 0))
    geometries = [Point(transform * (x, y)) for y, x in coords]
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    gdf.to_file(output_path, driver="ESRI Shapefile")
    print(f"✅ Saved point shapefile: {output_path}")

def save_edges_to_vector_file(binary_image, transform, output_file):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    geometries = []
    for contour in contours:
        coords = []
        for point in contour:
            x_pix, y_pix = point[0]
            x_geo, y_geo = transform * (x_pix, y_pix)
            coords.append((x_geo, y_geo))
        if len(coords) > 1:
            geometries.append(LineString(coords))
    gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    gdf.to_file(output_file, driver="ESRI Shapefile")
    print(f"✅ Saved line shapefile: {output_file}")

def save_vector_points_to_obj(binary_image, data, output_file):
    pts = np.column_stack(np.where(binary_image > 0))
    verts = []
    for y, x in pts:
        z_scaled = data[y, x] / 30
        verts.append((x, -y, z_scaled))
    min_z = min(v[2] for v in verts)
    with open(output_file, "w") as f:
        f.write("# edge points aligned with DEM\n")
        for X, Yneg, Z in verts:
            f.write(f"v {X} {Z - min_z} {-Yneg}\n")
    print(f"✅ Saved points OBJ: {output_file}")

def save_edges_as_obj(binary_image, data, output_file):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    verts, lines, vmap = [], [], {}
    for c in contours:
        seq = []
        for pt in c:
            x, y = pt[0]
            z_scaled = data[y, x] / 30
            key = (x, -y, z_scaled)
            if key not in vmap:
                vmap[key] = len(verts)
                verts.append(key)
            seq.append(vmap[key])
        for i in range(len(seq) - 1):
            lines.append((seq[i], seq[i+1]))
    min_z = min(v[2] for v in verts)
    with open(output_file, "w") as f:
        f.write("# edge lines aligned with DEM\n")
        for X, Yneg, Z in verts:
            f.write(f"v {X} {Z - min_z} {-Yneg}\n")
        for a, b in lines:
            f.write(f"l {a+1} {b+1}\n")
    print(f"✅ Saved lines OBJ: {output_file}")

### --- Pipeline --- ###
def generate_all_vector_outputs(tiff_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        transform = src.transform

    normalized = normalize_data(data)

    methods = {
        "sobel": sobel_gradient_detection(normalized, threshold=0.15),
        "gaussian": gaussian_gradient_detection(normalized, sigma=3, threshold=0.01),
        "canny": canny_edge_detection(normalized, 25, 75),
        "laplacian": laplacian_detection(normalized, threshold=0.01),
        "local_variance": local_variance_detection(normalized, radius=5, threshold=0.01)
    }

    for name, binary in methods.items():
        base = os.path.join(output_dir, name)
        save_points_to_shapefile(binary, transform, base + "_points.shp")
        save_edges_to_vector_file(binary, transform, base + "_lines.shp")
        save_vector_points_to_obj(binary, data, base + "_points.obj")
        save_edges_as_obj(binary, data, base + "_lines.obj")

### --- Run Example --- ###
if __name__ == "__main__":
    generate_all_vector_outputs(
        tiff_path="C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture1_full/output_hh.tif",
        output_dir="C:/Users/ariks/uni/ResearchStudent/SAR_Imageas_02_01/Picture1_full/vectors_generated"
    )
