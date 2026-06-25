# generate_weights.py
"""
Generation file for weights for catchments on different model grids

Generates catchment area-fraction weight files (one per catchment per model)
that are used by catchment_tools.find_weight_file() during the analysis

Usage — run from the command line with --dataset:

    conda activate storm_hans

    python .../generate_weights.py --dataset era5_0.25
    python .../generate_weights.py --dataset gfdl_spear
    python .../generate_weights.py --dataset cesm2_le

All output files are written to cfg.WEIGHTS_DIR  (postprocessed/weights/)

Output filename convention (must match catchment_tools.find_weight_file()):
    weights_catchment_<slug>_<dataset>_<resolution>.nc   (for ERA5, SMILE models have no resolution)

Sections Overview
--------
A  Shared utilities   — dissolve GeoJSON, reproject, compute area-fraction
                        weights in metric space, save to NetCDF, inner loop
B  ERA5 0.25x0.25     — reads actual grid from one ERA5 raw file
C  GFDL-SPEAR-MED-LE  — reads grid from one SMILE raw file (lat/lon dims)
D  CESM2-LE           — reads grid from one SMILE raw file (lat/lon dims)

Methodology:
  1. Dissolve all GeoJSON features → single Shapely polygon in EPSG:4326
  2. Reproject polygon to a metric CRS (EPSG:25833 -> Norway adapted)
  3. For every candidate grid cell:
       a. Build cell box in EPSG:4326
       b. Reproject cell to the same metric CRS
       c. weight = area(catchment ∩ cell) / area(cell)   [both in m²]
  Area computation is always in metric space!  This is the only way to get correct area fractions on a lat/lon grid at high lattitudes
"""

#Import standard libraries
import sys
import re
import argparse
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from shapely.prepared import prep
from pyproj import Transformer

#Define Helper files
HELPER_DIR = Path("/nird/home/lbal/internship_storm_hans/helper")
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

import config_paths as cfg
from data_era5 import find_era5_files


# ── Catchment Origin (shared by all sections) 
GEOJSON_FILES = {
    "nevina_bergheim":  "catchment_nve_nevina_bergheim.geojson",
    "nevina_honnefoss": "catchment_nve_nevina_hønnefoss.geojson",
    "nevina_losna":     "catchment_nve_nevina_losna.geojson",
    "regine_drammen":   "catchment_nve_regine_drammen.geojson",
    "regine_glomma":    "catchment_nve_regine_glomma.geojson",}

GFDL_SPEAR_DIR = cfg.GFDL_SPEAR_DIR
CESM2_LE_DIR   = cfg.CESM2_LE_DIR

# ═════════════════════════════════════════════════════════════════════════════
# SECTION A — Shared utilities
# All three weight-generation sections call only the functions defined here.
# ═════════════════════════════════════════════════════════════════════════════

#Define how to transform Geojson file to metric polygon files
def _dissolve_geojson(geojson_path: Path):
    """
    Read a GeoJSON catchment file and dissolve all features into a single
    Shapely geometry (Polygon or MultiPolygon) in EPSG:4326 (lon/lat)
    """
    gdf = gpd.read_file(str(geojson_path))
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    try:
        return gdf.geometry.union_all()         # geopandas >= 0.14
    except AttributeError:
        return gdf.geometry.unary_union         # geopandas < 0.14


#Define how to build a transformer function for reprojection
def _build_projector(dst_epsg: str = "EPSG:25833"):
    """
    Return a Shapely-compatible transform function EPSG:4326 → dst_epsg
    Default EPSG:25833 = UTM zone 33N, appropriate for Norway
    """
    return Transformer.from_crs("EPSG:4326", dst_epsg, always_xy=True).transform


#Define how to compute area-fraction weights for one catchment on a lat/lon grid
def build_weights(geojson_path: Path,
                  lat_grid: np.ndarray,
                  lon_grid: np.ndarray,
                  dst_epsg: str = "EPSG:25833") -> np.ndarray:
    """
    Compute area-fraction catchment weights on any regular lat/lon grid

    Parameters
    ----------
    Returns
    -------
    np.ndarray, shape (n_lat, n_lon), dtype float32
        Values in [0, 1].  0 = no overlap; 1 = cell fully inside catchment
    """
    # Step 1 — dissolve catchment to one polygon in lon/lat
    poly_ll = _dissolve_geojson(geojson_path)

    # Step 2 — reproject catchment to metric CRS once (area computation)
    proj_fn  = _build_projector(dst_epsg)
    poly_m   = shapely_transform(proj_fn, poly_ll)
    if not poly_m.is_valid:
        poly_m = poly_m.buffer(0)           # repair self-intersections from reprojection
    poly_prep = prep(poly_m)                # prepared geometry for fast intersects checks

    # Step 3 — grid cell half-widths
    dlat = float(np.abs(np.diff(lat_grid)).mean())
    dlon = float(np.abs(np.diff(lon_grid)).mean())

    weights = np.zeros((len(lat_grid), len(lon_grid)), dtype=np.float32)

    # Step 4 — bounding-box pre-filter (in lon/lat space, cheap)
    minx, miny, maxx, maxy = poly_ll.bounds
    lat_idx = np.where((lat_grid >= miny - dlat) & (lat_grid <= maxy + dlat))[0]
    lon_idx = np.where((lon_grid >= minx - dlon) & (lon_grid <= maxx + dlon))[0]

    # Step 5 — per-cell area fraction in metric space (m² / m²)
    for i in lat_idx:
        lat = float(lat_grid[i])
        for j in lon_idx:
            lon = float(lon_grid[j])

            # Build cell in lon/lat, reproject to metric CRS
            cell_ll = box(lon - dlon / 2, lat - dlat / 2,
                          lon + dlon / 2, lat + dlat / 2)
            cell_m  = shapely_transform(proj_fn, cell_ll)

            if not poly_prep.intersects(cell_m):
                continue

            inter = poly_m.intersection(cell_m)
            if (not inter.is_empty) and (cell_m.area > 0):
                weights[i, j] = float(inter.area / cell_m.area)  # both m²

    return weights


#Define how to save weights to NetCDF appropriately (with metadata and dimension names)
def save_weight_nc(weights: np.ndarray,
                   lat_grid: np.ndarray,
                   lon_grid: np.ndarray,
                   out_path: Path,
                   lat_dim: str = "latitude",
                   lon_dim: str = "longitude") -> None:
    """
    Write catchment weight DataArray to NetCDF

    lat_dim / lon_dim must match the dimension names used in the raw model files so that catchment_tools.align_weights_to_precip() can align them without reindexing:
        ERA5         → lat_dim="latitude",  lon_dim="longitude"
        SMILE models → lat_dim="lat",       lon_dim="lon"
    """
    da = xr.DataArray(
        weights,
        coords={lat_dim: lat_grid, lon_dim: lon_grid},
        dims=[lat_dim, lon_dim],
        name="catchment_weight",
        attrs={"long_name": "Fractional model-grid-cell area inside catchment", "units": "1",},)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"catchment_weight": da}).to_netcdf(str(out_path))
    print(f"  Saved → {out_path.name}")


#Define how to run the loop
def _run_weight_loop(dataset: str,
                     resolution: str,
                     lat_grid: np.ndarray,
                     lon_grid: np.ndarray,
                     out_dir: Path,
                     lat_dim: str = "latitude",
                     lon_dim: str = "longitude") -> None:
    """
    Inner per-catchment loop shared by all three weight-generation sections

    Iterates over GEOJSON_FILES, calls build_weights() and save_weight_nc()
    for each catchment. Skips files that already exist on disk

    Output filename pattern (must match catchment_tools.find_weight_file()):
        weights_catchment_<slug>_<dataset>_<resolution>.nc   (if resolution != "")
        weights_catchment_<slug>_<dataset>.nc                 (if resolution == "")
    """
    res_part = f"_{resolution}" if resolution else ""
    print(f"  Grid: {len(lat_grid)} lats × {len(lon_grid)} lons")
    print()

    for slug, geojson_name in GEOJSON_FILES.items():
        out_path = out_dir / f"weights_catchment_{slug}_{dataset}{res_part}.nc"

        if out_path.exists():
            print(f"[skip] Already exists: {out_path.name}")
            continue

        geojson_path = cfg.GEOJSON_DIR / geojson_name
        if not geojson_path.exists():
            print(f"[WARNING] GeoJSON not found for '{slug}':")
            print(f"  Expected : {geojson_path}")
            print(f"  Check    : ls {cfg.GEOJSON_DIR}")
            continue

        print(f"Processing {slug} ...")
        weights = build_weights(geojson_path, lat_grid, lon_grid)
        nonzero = int((weights > 0).sum())
        print(f"  Non-zero cells: {nonzero}")
        if nonzero == 0:
            print("  ERROR: no overlap found — check GeoJSON CRS and coordinates.")
            continue

        save_weight_nc(weights, lat_grid, lon_grid, out_path,
                       lat_dim=lat_dim, lon_dim=lon_dim)
        print()

    print(f"Done. Weight files written to:  {out_dir}")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION B — ERA5 0.25° weight generation
# ═════════════════════════════════════════════════════════════════════════════

def run_era5_025() -> None:
    """
    Generate ERA5 0.25° catchment weight files

    Grid is read from the first raw ERA5 annual file -> takes automatically what is in there

    Output : cfg.WEIGHTS_DIR / weights_catchment_<slug>_era5_0.25x0.25.nc
    Dim names : latitude, longitude  (standard ERA5 names)
    """
    DATASET, RESOLUTION = "era5", "0.25x0.25"

    print("─" * 60)
    print(f"ERA5 {RESOLUTION} weight generation")
    print("─" * 60)
    print("Reading grid from ERA5 raw files ...")

    era5_files = find_era5_files(cfg.ERA5_RAW_DIR, RESOLUTION)
    ds_ref     = xr.open_dataset(str(era5_files[0]))
    lat_grid   = ds_ref["latitude"].values
    lon_grid   = ds_ref["longitude"].values
    ds_ref.close()

    _run_weight_loop(DATASET, RESOLUTION, lat_grid, lon_grid, cfg.WEIGHTS_DIR, lat_dim="latitude", lon_dim="longitude")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION C — GFDL-SPEAR-MED-LE weight generation
# ═════════════════════════════════════════════════════════════════════════════

#Define how to find files for Smile model
def _find_one_smile_file(directory: Path, pattern: str) -> Path:
    """
    Return the first file in directory whose name matches the regex pattern
    Raises FileNotFoundError if none found
    """
    rx = re.compile(pattern)
    for f in sorted(directory.iterdir()):
        if rx.match(f.name):
            return f
    raise FileNotFoundError(
        f"No files matching '{pattern}' found in:\n  {directory}")


#Define how to run the loop for GFDL-SPEAR-MED-LE
def run_gfdl_spear() -> None:
    """
    Generate GFDL-SPEAR-MED-LE catchment weight files

    Output : cfg.WEIGHTS_DIR / weights_catchment_<slug>_gfdl_spear_med_le.nc
    Dim names : lat, lon
    """
    DATASET, RESOLUTION = "gfdl_spear_med_le", ""   # no resolution suffix; model name is unique

    print("─" * 60)
    print("GFDL-SPEAR-MED-LE weight generation")
    print("─" * 60)
    print(f"Reading grid from: {GFDL_SPEAR_DIR}")

    ref_file = _find_one_smile_file(
        GFDL_SPEAR_DIR,
        r"^gfdl_spear_med_le\.tp24\.\d{2}\.\d{8}-\d{8}\.nc$",)
    print(f"  Reference file : {ref_file.name}")

    ds_ref   = xr.open_dataset(str(ref_file))
    lat_grid = ds_ref["lat"].values
    lon_grid = ds_ref["lon"].values
    ds_ref.close()

    _run_weight_loop(DATASET, RESOLUTION, lat_grid, lon_grid,
                     cfg.WEIGHTS_DIR,
                     lat_dim="lat", lon_dim="lon")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION D — CESM2-LE weight generation
# ═════════════════════════════════════════════════════════════════════════════

#Define how to run the loop for CESM2-LE — same file pattern and grid structure as GFDL-SPEAR-MED-LE, just different directory and dataset name
def run_cesm2_le() -> None:
    """
    Generate CESM2-LE catchment weight files
    """
    DATASET, RESOLUTION = "cesm2_le", ""    # no resolution suffix; model name is unique

    print("─" * 60)
    print("CESM2-LE weight generation")
    print("─" * 60)
    print(f"Reading grid from: {CESM2_LE_DIR}")

    ref_file = _find_one_smile_file(CESM2_LE_DIR, r"^cesm2-le\.[A-Za-z0-9_]+\.\d{3}\.\d{8}-\d{8}\.nc$",)
    print(f"  Reference file : {ref_file.name}")

    ds_ref   = xr.open_dataset(str(ref_file))
    lat_grid = ds_ref["lat"].values
    lon_grid = ds_ref["lon"].values
    ds_ref.close()

    _run_weight_loop(DATASET, RESOLUTION, lat_grid, lon_grid, cfg.WEIGHTS_DIR, lat_dim="lat", lon_dim="lon")

# ═════════════════════════════════════════════════════════════════════════════
# Registry + CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

REGISTRY = {"era5_0.25":  run_era5_025, "gfdl_spear": run_gfdl_spear, "cesm2_le":   run_cesm2_le,}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate catchment area-fraction weight files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Available datasets:\n"
            + "\n".join(f"  {k}" for k in REGISTRY)),)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(REGISTRY.keys()),
        help="Which model/resolution to generate weights for.",)
    REGISTRY[parser.parse_args().dataset]()