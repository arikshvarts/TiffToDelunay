
"""
tiff_to_constrained_obj.py

A single-file pipeline that:
1) Loads a GeoTIFF (1st band as height/DEM).
2) Detects edges (choose method).
3) Converts edge contours into constraint segments on a downsampled regular grid.
4) Runs constrained Delaunay triangulation (Triangle library) on that grid.
5) Writes an OBJ with vertices (X, Z, Y) using the TIFF heights and faces from the triangulation.

Default coordinates are in pixel units (X, Y). Z is scaled by 1/Z_DIV to look good in 3D tools.
By default we also "ground shift" so min(Z) == 0 for easier viewing in Blender.

USAGE (examples):
  python tiff_to_constrained_obj.py --tiff /path/to/dem.tif --out /path/to/mesh.obj
  python tiff_to_constrained_obj.py --tiff dem.tif --out mesh.obj --method canny --canny 40 120 --stride 4
  python tiff_to_constrained_obj.py --tiff dem.tif --out mesh.obj --method gaussian --sigma 3.0 --edge-th 0.01 --stride 8 --z-div 30

Dependencies:
  pip install numpy rasterio opencv-python scipy scikit-image triangle

Notes:
  • For large TIFFs, increase --stride (e.g., 8, 12, 16) to keep the grid manageable.
  • If you want georeferenced X/Y instead of pixel coordinates, set --geo-xy.
  • To also dump the edge lines as an OBJ for inspection, pass --save-edge-lines edges.obj
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import cv2
from scipy.ndimage import gaussian_gradient_magnitude, laplace
from skimage.filters import rank
from skimage.morphology import disk
import triangle as tr


# ----------------------------
# Edge detection helpers
# ----------------------------

def normalize_data(data: np.ndarray) -> np.ndarray:
    d = data.astype(np.float32)
    mn, mx = np.min(d), np.max(d)
    if mx <= mn:
        return np.zeros_like(d)
    return (d - mn) / (mx - mn)


def edges_sobel(norm: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return (mag > threshold).astype(np.uint8) * 255


def edges_gaussian(norm: np.ndarray, sigma: float = 3.0, threshold: float = 0.01) -> np.ndarray:
    g = gaussian_gradient_magnitude(norm, sigma=sigma)
    return (g > threshold).astype(np.uint8) * 255


def edges_canny(norm: np.ndarray, low: int = 25, high: int = 75) -> np.ndarray:
    u8 = (norm * 255.0).clip(0,255).astype(np.uint8)
    return cv2.Canny(u8, low, high)


def edges_laplace(norm: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    lp = np.abs(laplace(norm))
    return (lp > threshold).astype(np.uint8) * 255


def edges_entropy(norm: np.ndarray, radius: int = 5, threshold: float = 0.01) -> np.ndarray:
    u8 = (norm * 255.0).clip(0,255).astype(np.uint8)
    ent = rank.entropy(u8, disk(radius)).astype(np.float32)
    ent /= (np.max(ent) + 1e-8)
    return (ent > threshold).astype(np.uint8) * 255


# ----------------------------
# Core pipeline
# ----------------------------

def contours_from_binary(binary: np.ndarray, approx_eps: float = 0.0):
    """Return list of polylines (each is a list of (x,y) pixel coords)."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines = []
    for c in cnts:
        if approx_eps > 0:
            c = cv2.approxPolyDP(c, epsilon=approx_eps, closed=False)
        pts = [tuple(pt[0]) for pt in c]
        if len(pts) >= 2:
            polylines.append(pts)
    return polylines


def build_stride_grid(width: int, height: int, stride: int):
    nx = (width  + stride - 1) // stride
    ny = (height + stride - 1) // stride

    # For each grid point (i, j), the pixel sample is (x=i*stride, y=j*stride) clamped to bounds
    xs = np.clip(np.arange(nx) * stride, 0, width  - 1)
    ys = np.clip(np.arange(ny) * stride, 0, height - 1)
    return nx, ny, xs, ys


def grid_index(i: int, j: int, nx: int) -> int:
    return j * nx + i


def segments_from_polylines(polylines, nx, ny, xs, ys, stride: int):
    """
    Convert polylines in pixel coords into segment indices on the stride grid:
    - map pixel (x,y) -> nearest grid (i,j) = (round(x/stride), round(y/stride))
    - add edges between consecutive points (i1,j1)->(i2,j2) if distinct
    """
    segs = set()
    for pts in polylines:
        prev_idx = None
        for (x, y) in pts:
            i = int(round(x / float(stride)))
            j = int(round(y / float(stride)))
            i = max(0, min(i, len(xs) - 1))
            j = max(0, min(j, len(ys) - 1))
            cur = grid_index(i, j, nx)
            if prev_idx is not None and prev_idx != cur:
                a, b = (prev_idx, cur) if prev_idx < cur else (cur, prev_idx)
                segs.add((a, b))
            prev_idx = cur
    return list(segs)


def boundary_segments(nx, ny):
    """Outer rectangle segments to keep hull closed."""
    def idx(i, j): return grid_index(i, j, nx)

    segs = []
    # top and bottom
    for i in range(nx - 1):
        segs.append((idx(i, 0), idx(i + 1, 0)))
        segs.append((idx(i, ny - 1), idx(i + 1, ny - 1)))
    # left and right
    for j in range(ny - 1):
        segs.append((idx(0, j), idx(0, j + 1)))
        segs.append((idx(nx - 1, j), idx(nx - 1, j + 1)))
    return segs


def triangulate_with_constraints(vertices_2d: np.ndarray, segments: np.ndarray, quality: bool = True):
    """Run Triangle on given 2D vertices and PSLG segments."""
    A = dict(vertices=vertices_2d.astype(np.float64), segments=np.array(segments, dtype=np.int32))
    flags = "p" + ("q" if quality else "")
    result = tr.triangulate(A, flags)
    if 'triangles' not in result or result['triangles'] is None or len(result['triangles']) == 0:
        raise RuntimeError("Triangulation failed or returned no triangles.")
    return result['triangles']


def write_obj(vertices_xyz: np.ndarray, faces: np.ndarray, out_path: Path, ground_shift: bool = True):
    """
    Save OBJ with 'v X Z Y' layout:
      v x z y
      f i j k
    and by default set min(Z) = 0 (ground shift) and flip Y sign for Blender friendliness.
    vertices_xyz is array of shape (N, 3) with (X, Y, Zpix) where Y is pixel Y.
    """
    X = vertices_xyz[:, 0]
    Y = vertices_xyz[:, 1]
    Z = vertices_xyz[:, 2]

    if ground_shift:
        Z = Z - np.min(Z)

    with open(out_path, "w") as f:
        for x, y, z in zip(X, Y, Z):
            f.write(f"v {x:.6f} {z:.6f} {-y:.6f}\n")  # v X Z -Y
        for a, b, c in faces:
            f.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")


def run_pipeline(
    tiff_path: Path,
    out_obj: Path,
    method: str = "canny",
    stride: int = 4,
    z_div: float = 30.0,
    approx_eps: float = 0.0,
    canny_low: int = 25,
    canny_high: int = 75,
    sobel_th: float = 0.15,
    gauss_sigma: float = 3.0,
    edge_th: float = 0.01,
    entropy_radius: int = 5,
    save_edge_lines: Path | None = None,
    geo_xy: bool = False,
):
    # 1) Read TIFF
    with rasterio.open(tiff_path) as src:
        hmap = src.read(1).astype(np.float32)
        height, width = hmap.shape
        transform = src.transform

    # 2) Edges
    norm = normalize_data(hmap)

    if method == "canny":
        binary = edges_canny(norm, canny_low, canny_high)
    elif method == "sobel":
        binary = edges_sobel(norm, sobel_th)
    elif method == "gaussian":
        binary = edges_gaussian(norm, gauss_sigma, edge_th)
    elif method == "laplace":
        binary = edges_laplace(norm, edge_th)
    elif method == "entropy":
        binary = edges_entropy(norm, entropy_radius, edge_th)
    else:
        raise ValueError(f"Unknown --method '{method}'")

    # 3) Contours → polylines
    polylines = contours_from_binary(binary, approx_eps=approx_eps)

    # 4) Build stride grid and map constraints
    nx, ny, xs, ys = build_stride_grid(width, height, stride)

    # 5) 2D vertices for Triangle
    if geo_xy:
        # Use georeferenced XY via raster transform
        Xs, Ys = np.meshgrid(xs, ys)
        # Affine transform is (col=x, row=y)
        # Flatten
        Xf = Xs.ravel()
        Yf = Ys.ravel()
        coords = [transform * (int(xc), int(yc)) for xc, yc in zip(Xf, Yf)]
        vertices_2d = np.array([(cx, cy) for (cx, cy) in coords], dtype=np.float64)
    else:
        # Use pixel coordinates
        Xs, Ys = np.meshgrid(xs, ys)
        vertices_2d = np.stack([Xs.ravel(), Ys.ravel()], axis=1).astype(np.float64)

    # 6) Segments: from polylines + boundary
    segs = segments_from_polylines(polylines, nx, ny, xs, ys, stride)
    segs += boundary_segments(nx, ny)

    # 7) Triangulate (constrained)
    faces = triangulate_with_constraints(vertices_2d, np.array(segs, dtype=np.int32), quality=True)

    # 8) Build 3D vertices (X, Y, Z_scaled)
    #    Z comes from the original TIFF at the grid sample locations.
    Zs = hmap[Ys, Xs].astype(np.float32) / float(z_div)
    vertices_xyz = np.stack([Xs.ravel().astype(np.float32),
                             Ys.ravel().astype(np.float32),
                             Zs.ravel().astype(np.float32)], axis=1)

    # 9) Save OBJ
    write_obj(vertices_xyz, faces, out_obj, ground_shift=True)

    # 10) Optionally save edge lines OBJ for inspection
    if save_edge_lines is not None:
        # Build a lightweight edge-lines OBJ using the same vertices list.
        save_edge_lines = Path(save_edge_lines)
        with open(save_edge_lines, "w") as f:
            for x, y, z in vertices_xyz:
                f.write(f"v {x:.6f} {z:.6f} {-y:.6f}\n")
            for a, b in segs:
                f.write(f"l {int(a)+1} {int(b)+1}\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="TIFF → edges → constrained Delaunay → OBJ")
    p.add_argument("--tiff", required=True, type=Path, help="Path to GeoTIFF (DEM) file (first band used)")
    p.add_argument("--out",  required=True, type=Path, help="Output OBJ path")
    p.add_argument("--method", choices=["canny","sobel","gaussian","laplace","entropy"], default="canny")
    p.add_argument("--stride", type=int, default=4, help="Grid stride (>=1). Larger = fewer verts.")
    p.add_argument("--z-div", type=float, default=30.0, help="Divide raw height by this factor for nicer Z scale")
    p.add_argument("--approx-eps", type=float, default=0.0, help="Douglas-Peucker epsilon (in pixels) to simplify contours")
    # Edge params
    p.add_argument("--canny", nargs=2, type=int, metavar=("LOW","HIGH"), default=None,
                   help="Canny thresholds (if --method canny). Example: --canny 25 75")
    p.add_argument("--sobel-th", type=float, default=0.15, help="Sobel magnitude threshold (if --method sobel)")
    p.add_argument("--sigma", type=float, default=3.0, help="Gaussian gradient sigma (if --method gaussian)")
    p.add_argument("--edge-th", type=float, default=0.01, help="Threshold for gaussian/laplace/entropy (normalized)")
    p.add_argument("--entropy-radius", type=int, default=5, help="Entropy radius (if --method entropy)")
    # Extras
    p.add_argument("--save-edge-lines", type=Path, default=None, help="Also export edges as OBJ 'l' elements")
    p.add_argument("--geo-xy", action="store_true", help="Write vertices in georeferenced XY using raster transform")

    args = p.parse_args(argv)

    canny_low, canny_high = (25, 75)
    if args.canny is not None:
        canny_low, canny_high = args.canny

    run_pipeline(
        tiff_path=args.tiff,
        out_obj=args.out,
        method=args.method,
        stride=max(1, args.stride),
        z_div=args.z_div,
        approx_eps=max(0.0, args.approx_eps),
        canny_low=canny_low,
        canny_high=canny_high,
        sobel_th=args.sobel_th,
        gauss_sigma=args.sigma,
        edge_th=args.edge_th,
        entropy_radius=args.entropy_radius,
        save_edge_lines=args.save_edge_lines,
        geo_xy=args.geo_xy,
    )


if __name__ == "__main__":
    sys.exit(main())
