# generate_0.25_weights.py
"""
Generate ERA5 0.25° catchment weight files from GeoJSON boundaries.

Run ONCE before the 0.25x0.25 analysis sections in the notebook:
    conda activate storm_hans
    python /nird/home/lbal/internship_storm_hans/helper/generate_0.25_weights.py

Produces for each catchment:
    weights_catchment_<slug>_era5_0.25x0.25.nc
in cfg.WEIGHTS_025_DIR.
"""

import sys
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box

HELPER_DIR = Path("/nird/home/lbal/internship_storm_hans/helper")
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

import config_paths as cfg
from data_era5 import find_era5_files

# ── Slug → GeoJSON filename ────────────────────────────────────────────────────
# Exact filenames as seen on disk in cfg.CATCHMENT_RAW_DIR
GEOJSON_FILES = {
    "nevina_bergheim":  "catchment_nve_nevina_bergheim.geojson",
    "nevina_honnefoss": "catchment_nve_nevina_hønnefoss.geojson",
    "nevina_losna":     "catchment_nve_nevina_losna.geojson",
    "regine_drammen":   "catchment_nve_regine_drammen.geojson",
    "regine_glomma":    "catchment_nve_regine_glomma.geojson",
}

DATASET    = "era5"
RESOLUTION = "0.25x0.25"


def build_weights(geojson_path: Path,
                  lat_grid: np.ndarray,
                  lon_grid: np.ndarray) -> np.ndarray:
    """
    For every ERA5 0.25° cell, compute the fraction of its area that lies
    inside the catchment polygon. Returns float32 array (n_lat, n_lon).
    """
    gdf = gpd.read_file(str(geojson_path))
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    # geopandas >= 0.14: union_all(); older versions: unary_union
    try:
        poly = gdf.geometry.union_all()
    except AttributeError:
        poly = gdf.geometry.unary_union

    dlat = float(np.abs(np.diff(lat_grid)).mean())
    dlon = float(np.abs(np.diff(lon_grid)).mean())
    cell_area = dlat * dlon   # degrees² — units cancel in the ratio

    weights = np.zeros((len(lat_grid), len(lon_grid)), dtype=np.float32)

    # Bounding-box pre-filter: only iterate over cells near the catchment
    minx, miny, maxx, maxy = poly.bounds
    lat_idx = np.where((lat_grid >= miny - dlat) & (lat_grid <= maxy + dlat))[0]
    lon_idx = np.where((lon_grid >= minx - dlon) & (lon_grid <= maxx + dlon))[0]

    for i in lat_idx:
        lat = float(lat_grid[i])
        for j in lon_idx:
            lon = float(lon_grid[j])
            cell = box(lon - dlon / 2, lat - dlat / 2,
                       lon + dlon / 2, lat + dlat / 2)
            if poly.intersects(cell):
                overlap = poly.intersection(cell).area
                weights[i, j] = float(overlap / cell_area)

    return weights


def save_weight_nc(weights: np.ndarray,
                   lat_grid: np.ndarray,
                   lon_grid: np.ndarray,
                   out_path: Path) -> None:
    da = xr.DataArray(
        weights,
        coords={"latitude": lat_grid, "longitude": lon_grid},
        dims=["latitude", "longitude"],
        name="catchment_weight",
        attrs={"long_name": "Fractional ERA5-cell area inside catchment",
               "units": "1"},
    )
    ds = xr.Dataset({"catchment_weight": da})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(out_path))
    print(f"  Saved → {out_path.name}")


# ── Load the actual ERA5 0.25° grid from one raw file ─────────────────────────
print("Reading ERA5 0.25° grid coordinates from raw files ...")
era5_files = find_era5_files(cfg.ERA5_RAW_DIR, RESOLUTION)
ds_ref     = xr.open_dataset(str(era5_files[0]))
lat_grid   = ds_ref["latitude"].values
lon_grid   = ds_ref["longitude"].values
ds_ref.close()
print(f"  Grid: {len(lat_grid)} lats × {len(lon_grid)} lons")
print()

# ── Generate one weight file per catchment ────────────────────────────────────
for slug, geojson_name in GEOJSON_FILES.items():

    # Output filename must match what find_weight_file() in catchment_tools.py expects:
    # weights_catchment_<slug>_era5_0.25x0.25.nc
    out_path = cfg.WEIGHTS_025_DIR / f"weights_catchment_{slug}_era5_0.25x0.25.nc"

    if out_path.exists():
        print(f"[skip] Already exists: {out_path.name}")
        continue

    geojson_path = cfg.GEOJSON_DIR / geojson_name
    if not geojson_path.exists():
        print(f"[WARNING] GeoJSON not found for '{slug}':")
        print(f"  Expected: {geojson_path}")
        print(f"  Check spelling against: ls {cfg.GEOJSON_DIR}")
        continue

    print(f"Processing {slug} ...")
    weights = build_weights(geojson_path, lat_grid, lon_grid)
    nonzero = int((weights > 0).sum())
    print(f"  Non-zero cells: {nonzero}")
    if nonzero == 0:
        print(f"  ERROR: no overlap found — check GeoJSON CRS and coordinates.")
        continue

    save_weight_nc(weights, lat_grid, lon_grid, out_path)
    print()

print("Done. All weight files written to:")
print(f"  {cfg.WEIGHTS_025_DIR}")