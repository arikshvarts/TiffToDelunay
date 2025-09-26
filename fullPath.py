# tiff_constrained_obj_main.py
# -----------------------------------------------------------
# TIFF -> edges -> constrained Delaunay (Triangle) -> OBJ
# -----------------------------------------------------------
# Dependencies:
#   pip install numpy rasterio opencv-python scipy scikit-image triangle
#
# How to use:
#   1) Edit the paths in main() (TIFF_IN, OBJ_OUT, optional EDGES_OUT).
#   2) Adjust params: METHOD, STRIDE, etc.
#   3) Run:  python tiff_constrained_obj_main.py
#
# Output OBJ uses vertices as "v X Z -Y" (nice in Blender), with Z ground-shifted.
# If you set SAVE_EDGE_LINES=True, you'll also get an edges OBJ with "l" segments.
#
# Notes:
# - STRIDE controls density (bigger = fewer verts, faster).
# - METHOD can be "canny", "sobel", "gaussian", "laplace", "entropy".
# - To use geo-referenced XY instead of pixel XY, set GEO_XY=True.
# - Triangle flags: we use 'pq' for PSLG + quality (min angle). You can add area, e.g., 'pq30' or 'pqa100'.

from pathlib import Path
import numpy as np
import rasterio
import cv2
from scipy.ndimage import gaussian_gradient_magnitude, laplace
from skimage.filters import rank
from skimage.morphology import disk
import triangle as tr

# ------------- Edge detection helpers -------------
def _normalize(data: np.ndarray) -> np.ndarray:
    d = data.astype(np.float32)
    mn, mx = float(np.min(d)), float(np.max(d))
    if mx <= mn:
        return np.zeros_like(d, dtype=np.float32)
    return (d - mn) / (mx - mn)

def edges_canny(norm01: np.ndarray, low: int = 25, high: int = 75) -> np.ndarray:
    u8 = (norm01 * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.Canny(u8, low, high)

def edges_sobel(norm01: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    gx = cv2.Sobel(norm01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm01, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return (mag > threshold).astype(np.uint8) * 255

def edges_gaussian(norm01: np.ndarray, sigma: float = 3.0, threshold: float = 0.01) -> np.ndarray:
    g = gaussian_gradient_magnitude(norm01, sigma=sigma)
    return (g > threshold).astype(np.uint8) * 255

def edges_laplace(norm01: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    lp = np.abs(laplace(norm01))
    return (lp > threshold).astype(np.uint8) * 255

def edges_entropy(norm01: np.ndarray, radius: int = 5, threshold: float = 0.01) -> np.ndarray:
    u8 = (norm01 * 255.0).clip(0, 255).astype(np.uint8)
    ent = rank.entropy(u8, disk(radius)).astype(np.float32)
    ent /= (float(np.max(ent)) + 1e-8)
    return (ent > threshold).astype(np.uint8) * 255

def detect_edges(hmap: np.ndarray, method: str, params: dict) -> np.ndarray:
    n = _normalize(hmap)
    if method == "canny":
        return edges_canny(n, params.get("canny_low", 25), params.get("canny_high", 75))
    if method == "sobel":
        return edges_sobel(n, params.get("sobel_th", 0.15))
    if method == "gaussian":
        return edges_gaussian(n, params.get("sigma", 3.0), params.get("edge_th", 0.01))
    if method == "laplace":
        return edges_laplace(n, params.get("edge_th", 0.01))
    if method == "entropy":
        return edges_entropy(n, params.get("entropy_radius", 5), params.get("edge_th", 0.01))
    raise ValueError(f"Unknown method '{method}'")

# ------------- Contours & grid helpers -------------
def contours_from_binary(binary: np.ndarray, approx_eps: float = 0.0):
    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines = []
    for c in cnts:
        cc = c
        if approx_eps > 0:
            cc = cv2.approxPolyDP(c, epsilon=approx_eps, closed=False)
        pts = [tuple(pt[0]) for pt in cc]
        if len(pts) >= 2:
            polylines.append(pts)
    return polylines

def build_stride_grid(width: int, height: int, stride: int):
    nx = (width  + stride - 1) // stride
    ny = (height + stride - 1) // stride
    xs = np.clip(np.arange(nx) * stride, 0, width  - 1)
    ys = np.clip(np.arange(ny) * stride, 0, height - 1)
    return nx, ny, xs, ys

def _grid_index(i: int, j: int, nx: int) -> int:
    return j * nx + i

def segments_from_polylines(polylines, nx, ny, stride: int):
    segs = set()
    for pts in polylines:
        prev = None
        for (x, y) in pts:
            i = int(round(x / float(stride)));  i = max(0, min(i, nx - 1))
            j = int(round(y / float(stride)));  j = max(0, min(j, ny - 1))
            cur = _grid_index(i, j, nx)
            if prev is not None and prev != cur:
                a, b = (prev, cur) if prev < cur else (cur, prev)
                segs.add((a, b))
            prev = cur
    return list(segs)

def boundary_segments(nx, ny):
    segs = []
    def idx(i, j): return _grid_index(i, j, nx)
    # top/bottom
    for i in range(nx - 1):
        segs.append((idx(i, 0), idx(i + 1, 0)))
        segs.append((idx(i, ny - 1), idx(i + 1, ny - 1)))
    # left/right
    for j in range(ny - 1):
        segs.append((idx(0, j), idx(0, j + 1)))
        segs.append((idx(nx - 1, j), idx(nx - 1, j + 1)))
    return segs

# ------------- Triangulation & export -------------
def triangulate_with_constraints(vertices_2d: np.ndarray, segments: np.ndarray, flags: str = "pq"):
    """Run Triangle on PSLG. flags: 'p' (PSLG) + 'q' (quality)."""
    A = dict(vertices=vertices_2d.astype(np.float64), segments=segments.astype(np.int32))
    res = tr.triangulate(A, flags)
    tris = res.get('triangles', None)
    if tris is None or len(tris) == 0:
        raise RuntimeError("Triangulation failed or returned no triangles.")
    return tris.astype(np.int32)

def write_obj(vertices_xyz: np.ndarray, faces: np.ndarray, out_path: Path, ground_shift: bool = True):
    """
    Save OBJ as:
      v X Z -Y
      f i j k
    Z is ground-shifted to min(Z)=0 (nicer in Blender).
    """
    X = vertices_xyz[:, 0]
    Y = vertices_xyz[:, 1]  # pixel Y
    Z = vertices_xyz[:, 2]  # height (scaled)
    if ground_shift:
        Z = Z - float(np.min(Z))
    with open(out_path, "w", encoding="utf-8") as f:
        for x, y, z in zip(X, Y, Z):
            f.write(f"v {x:.6f} {z:.6f} {-y:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")

def write_edges_obj(vertices_xyz: np.ndarray, segs: list[tuple[int,int]], out_path: Path, ground_shift: bool = True):
    X = vertices_xyz[:, 0]
    Y = vertices_xyz[:, 1]
    Z = vertices_xyz[:, 2]
    if ground_shift:
        Z = Z - float(np.min(Z))
    with open(out_path, "w", encoding="utf-8") as f:
        for x, y, z in zip(X, Y, Z):
            f.write(f"v {x:.6f} {z:.6f} {-y:.6f}\n")
        for a, b in segs:
            f.write(f"l {int(a)+1} {int(b)+1}\n")

# ------------- Pipeline -------------
def tiff_to_constrained_obj(
    tiff_path: Path,
    out_obj: Path,
    *,
    method: str = "canny",
    stride: int = 4,
    z_div: float = 30.0,
    approx_eps: float = 0.0,
    canny_low: int = 25,
    canny_high: int = 75,
    sobel_th: float = 0.15,
    sigma: float = 3.0,
    edge_th: float = 0.01,
    entropy_radius: int = 5,
    triangle_flags: str = "pq",
    save_edge_lines: Path | None = None,
    geo_xy: bool = False,
):
    # 1) Read first band as height (DEM)
    with rasterio.open(tiff_path) as src:
        hmap = src.read(1).astype(np.float32)
        H, W = hmap.shape
        transform = src.transform  # affine

    # 2) Edges
    params = dict(
        canny_low=canny_low, canny_high=canny_high,
        sobel_th=sobel_th, sigma=sigma, edge_th=edge_th, entropy_radius=entropy_radius
    )
    binary = detect_edges(hmap, method=method, params=params)

    # 3) Contours -> polylines (pixel coordinates)
    polylines = contours_from_binary(binary, approx_eps=approx_eps)

    # 4) Build a regular grid (downsampled by STRIDE)
    nx, ny, xs, ys = build_stride_grid(W, H, stride)
    Xs, Ys = np.meshgrid(xs, ys)  # shapes (ny, nx)

    # 5) Vertex 2D coords (Triangle’s plane). Either pixel XY or georeferenced XY.
    if geo_xy:
        # map (col=x, row=y) through affine
        cols = Xs.ravel().astype(int)
        rows = Ys.ravel().astype(int)
        coords = [transform * (int(c), int(r)) for c, r in zip(cols, rows)]
        vertices_2d = np.array([(cx, cy) for (cx, cy) in coords], dtype=np.float64)
    else:
        # pixel plane
        vertices_2d = np.stack([Xs.ravel(), Ys.ravel()], axis=1).astype(np.float64)

    # 6) Constraints: snapped polylines + outer boundary
    segs = segments_from_polylines(polylines, nx, ny, stride)
    segs += boundary_segments(nx, ny)
    segs = np.array(segs, dtype=np.int32)

    # 7) Constrained triangulation
    faces = triangulate_with_constraints(vertices_2d, segs, flags=triangle_flags)

    # 8) Build 3D vertices (X, Y, Zscaled)
    Zs = hmap[Ys, Xs].astype(np.float32) / float(z_div)
    vertices_xyz = np.stack([
        Xs.ravel().astype(np.float32),     # X (pixels or geo-x)
        Ys.ravel().astype(np.float32),     # Y (pixels or geo-y)
        Zs.ravel().astype(np.float32)      # Z (height / z_div)
    ], axis=1)

    # 9) Write OBJ
    write_obj(vertices_xyz, faces, out_obj, ground_shift=True)

    # 10) Optional: save edge lines OBJ
    if save_edge_lines is not None:
        write_edges_obj(vertices_xyz, segs.tolist(), save_edge_lines, ground_shift=True)

# ------------- Main (edit here) -------------
def main():
    # === Edit these paths ===
    TIFF_IN  = Path(r"/Picture1_full/output_hh.tif")
    OBJ_OUT  = Path(r"/Picture1_full/objFull.obj")
    EDGES_OUT = Path(r"/Picture1_full/objEdges.obj")  # or None
    SAVE_EDGE_LINES = True  # set to False if you don't want a lines OBJ

    # === Adjust params ===
    METHOD = "canny"          # "canny" | "sobel" | "gaussian" | "laplace" | "entropy"
    STRIDE = 4                # ↑ for lighter mesh, ↓ for denser
    Z_DIV  = 30.0             # divide heights for nicer Z scale
    APPROX = 0.0              # Douglas-Peucker simplification epsilon (px)
    CANNY  = (40, 120)        # used if METHOD="canny"
    SOBEL_TH = 0.15
    SIGMA    = 3.0
    EDGE_TH  = 0.01
    ENT_RAD  = 5
    TRI_FLAGS = "pq"          # Triangle flags: 'p' (PSLG) + 'q' (quality). Add 'a<number>' for max area.
    GEO_XY   = False          # True => write XY in georeferenced coordinates

    EDGES_PATH = EDGES_OUT if SAVE_EDGE_LINES else None

    tiff_to_constrained_obj(
        tiff_path=TIFF_IN,
        out_obj=OBJ_OUT,
        method=METHOD,
        stride=STRIDE,
        z_div=Z_DIV,
        approx_eps=APPROX,
        canny_low=CANNY[0], canny_high=CANNY[1],
        sobel_th=SOBEL_TH,
        sigma=SIGMA,
        edge_th=EDGE_TH,
        entropy_radius=ENT_RAD,
        triangle_flags=TRI_FLAGS,
        save_edge_lines=EDGES_PATH,
        geo_xy=GEO_XY,
    )
    print(f"[OK] Wrote OBJ → {OBJ_OUT}")
    if EDGES_PATH:
        print(f"[OK] Wrote constraint lines OBJ → {EDGES_PATH}")

if __name__ == "__main__":
    main()
