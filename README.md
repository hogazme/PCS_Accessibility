# Accessibility Analysis (Albany–Schenectady Region)

This repository contains the code and data exported from a Code Ocean capsule.
It analyzes accessibility metrics for Census Block Groups (CBGs) using POI and
public charging station data.

## Structure
- `code/`: Python scripts for computing accessibility.
- `data/`: Input datasets (CBG boundaries, POIs, PCS, mappings).
- `environment/`: Dockerfile for reproducible setup.
- `metadata/`: Metadata from Code Ocean export (optional).

## Reproducibility
To build and run the analysis inside a Docker container:

```bash
docker build -t accessibility .
docker run --rm -v $(pwd):/workspace accessibility python code/accessibility.py
