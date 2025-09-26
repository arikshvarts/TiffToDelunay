# TiffToDelunay — TIFF to Constrained Delaunay OBJ

This project converts **GeoTIFF digital elevation models (DEM)** into constrained Delaunay triangulated surface meshes (OBJ). It supports two workflows:

1. **Integrated TIFF Flow (single input)**  
   - Input: a GeoTIFF.  
   - Process: edge detection (Canny, Sobel, Gaussian, Laplace, Entropy), grid generation, constraint snapping, constrained Delaunay triangulation.  
   - Output: OBJ mesh (Blender-friendly).  

2. **External Vectors Flow (DEM points + Edges)**  
   - Input: a DEM points OBJ and an edges OBJ.  
   - Process: snap edge vertices to DEM grid via KD-Tree, add boundary, constrained Delaunay triangulation.  
   - Output: OBJ mesh (Blender-friendly).  

Both flows save OBJ meshes in the format `v X Z -Y` with ground-shifted Z (minimum = 0).

---

## Installation

### pip
```bash
pip install numpy rasterio opencv-python scipy scikit-image triangle
```

### conda (recommended for Windows)
```bash
conda install -c conda-forge numpy rasterio scipy scikit-image triangle
pip install opencv-python
```

---

## Data

You can download elevation GeoTIFFs from OpenTopography:  
https://opentopography.org/

Ensure that the TIFF's first band contains height values. If georeferenced XY output is required, confirm CRS consistency.

---

## Coordinate and OBJ Convention

Vertices are written as:

```
v X Z -Y
```

- **X** = pixel column or georeferenced X  
- **Y** = pixel row or georeferenced Y  
- **Z** = height (scaled and shifted so minimum = 0)  
- `-Y` ensures correct upright orientation in Blender or MeshLab.

---

## Flow A: Integrated TIFF

**Script:** `tiff_to_constrained_obj.py`

### Steps
1. Read GeoTIFF band 1 as height.  
2. Detect edges.  
3. Build a grid at stride resolution.  
4. Snap edges to the grid as constraint segments.  
5. Add outer rectangle.  
6. Run Triangle for constrained Delaunay.  
7. Save OBJ. Optionally save edges as an OBJ with `l` segments.

### Example (Windows CMD)
```bash
python tiff_to_constrained_obj.py ^
  --tiff "C:\path\to\dem.tif" ^
  --out  "C:\path\to\mesh.obj" ^
  --method canny --canny 40 120 ^
  --stride 4 ^
  --save-edge-lines "C:\path\to\edges.obj"
```

### Example (Linux/Mac)
```bash
python tiff_to_constrained_obj.py   --tiff /path/to/dem.tif   --out  /path/to/mesh.obj   --method canny --canny 40 120   --stride 4   --save-edge-lines /path/to/edges.obj
```

### Parameters

| Parameter | Effect | Notes |
|-----------|--------|-------|
| `--method` | Edge detection method | `canny`, `sobel`, `gaussian`, `laplace`, `entropy` |
| `--canny LOW HIGH` | Thresholds for Canny | Lower = more edges, higher = fewer edges |
| `--sigma`, `--edge-th` | Gaussian and threshold | Adjust detail level |
| `--stride` | Grid resolution | Lower stride = denser mesh |
| `--z-div` | Z scale divisor | Controls vertical exaggeration |
| `--approx-eps` | Simplifies contours | Reduce noisy zigzags |
| `--geo-xy` | Use georeferenced XY | Optional |
| `--save-edge-lines` | Export edges OBJ | For inspection |

**Triangle flags:**  
- `'pq'` — basic quality triangulation.  
- `'pq30'` — enforces minimum angle of 30°.  
- `'pqa20'` — caps triangle area.  
- Combine as needed (e.g. `'pq30a20'`).

---

## Flow B: External Vectors

**Script:** `dem_and_edges_to_constrained_obj.py`

### Steps
1. Load DEM points OBJ.  
2. Load edges OBJ.  
3. KD-Tree snap edge endpoints to DEM points.  
4. Add rectangular boundary.  
5. Run Triangle with constraints.  
6. Save OBJ.

This flow is recommended when curated GIS/CAD edges are available.

---

## Validation

Use `check_obj_summary.py` to verify results:

```bash
python check_obj_summary.py mesh.obj
```

Outputs:
- Vertex, face, line counts  
- Bounding box extents  
- Triangle area distribution  
- Zero/near-zero area triangles

Healthy meshes should have consistent bounds, nonzero face counts, and minimal degenerate triangles.

---

## Project Structure

```
TiffToDelunay/
├─ README.md
├─ tiff_to_constrained_obj.py
├─ dem_and_edges_to_constrained_obj.py
├─ check_obj_summary.py
├─ examples/
│  ├─ sample.tif
│  ├─ sample_mesh.obj
│  └─ sample_edges.obj
└─ requirements.txt
```

---

## Recommendations

- Start with Flow A for simplicity.  
- Use Flow B only when curated vectors are required.  
- Adjust `stride` and Triangle flags for balance between mesh quality and speed.  
- Validate outputs using the checker script.  
- View results in Blender or MeshLab for confirmation.

---

## Troubleshooting

- **ModuleNotFoundError: rasterio** → install via pip or conda.  
- **Empty mesh** → thresholds too strict or stride too large.  
- **Triangulation errors** → relax flags, simplify edges, or check boundary constraints.  
- **Flat appearance** → adjust `--z-div`.  
- **Constraints invisible** → lower stride, enforce `'pq30'` and area limits.  
- **Misaligned edges (Flow B)** → ensure DEM and edges are in the same coordinate system.

---

## License and Attribution

- DEM data: subject to license/terms at [OpenTopography](https://opentopography.org/).  
- Code: add appropriate license (MIT/Apache/BSD).

---
