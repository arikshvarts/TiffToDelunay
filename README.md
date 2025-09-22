# SAR Image Triangulation and Edge Detection

This project focuses on processing SAR (Synthetic Aperture Radar) Tiff files, applying geographical adjustments, and converting them into a triangulated mesh. A key aspect of this process involves incorporating constraints derived from edge detection procedures to refine the mesh triangulation.

## Project Structure

- `dem2mesh/`: Contains the C++ project for converting Digital Elevation Models (DEM) to meshes.
- `utillity_vectorgeneration_func.py`: Python script related to vector generation.
- `TiffWork.py`: Python script for Tiff file processing.
- `try_full_pipeline.py`: Python script to run the full processing pipeline.
- `old/`: Directory containing older or experimental output files.
- `Picture1_full/`, `Picture2_full/`: Directories containing various output pictures, potentially including intermediate and final results.
- `rasters_COP30/`, `rasters_COP90/`: Directories likely holding input or processed raster data at different resolutions.
- `Reg_Pics/`: Contains registration related pictures.
- `delunay explanation.jpg`: An image providing an explanation related to Delaunay triangulation.
- `cmd_working_line.txt`: This file contains the command-line arguments needed to run the C++ `dem2mesh` procedure for converting TIFF files to OBJ meshes. **This file needs to be modified per input image** to specify the correct input and output paths, and other parameters specific to the image being processed.
- `סטטוס התקדמות 25.pptx`: A PowerPoint presentation detailing the project's progress and status.

