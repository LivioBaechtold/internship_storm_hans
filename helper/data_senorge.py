# data_senorge.py
"""
seNorge file discovery and loading.

Key differences from ERA5:
- No resolution suffix in filenames (one grid only: 1 x 1 km, UTM Zone 33)
- Spatial dimensions are Y and X (projected meters), not latitude/longitude
- Variable name: rr
- Units: kg/m² == mm (1:1, no conversion needed)
- Fill value: -999.99 (masked on load)
"""

import re
import xarray as xr
import numpy as np
from pathlib import Path
from pyproj import Transformer


def find_senorge_files(senorge_dir: Path) -> list[Path]:
    """
    Return chronologically sorted seNorge annual NetCDF files.
    Expected filename pattern: rr_<year>.nc   e.g. rr_1957.nc
    """
    pattern = re.compile(r"^rr_(\d{4})\.nc$")
    matched = []
    for f in senorge_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            matched.append((int(m.group(1)), f))

    if not matched:
        raise FileNotFoundError(
            f"No seNorge files found in:\n  {senorge_dir}\n"
            f"Expected filenames like: rr_1957.nc"
        )

    matched.sort(key=lambda x: x[0])
    return [f for _, f in matched]


def load_senorge_precipitation(senorge_files: list[Path]) -> xr.DataArray:
    """
    Open and concatenate annual seNorge files into a single lazy DataArray.

    - Masks the fill value -999.99
    - Units kg/m² are equivalent to mm — no numeric conversion needed
    - Spatial dims remain Y (northing) and X (easting) in projected meters
    - Verifies that the time axis is strictly increasing

    Returns
    -------
    xr.DataArray
        Dimensions: (time, Y, X), units: mm.
        Loaded lazily via Dask.
    """
    print(f"  Opening {len(senorge_files)} seNorge files (lazy) ...")
    ds = xr.open_mfdataset(
        [str(f) for f in senorge_files],
        combine="by_coords",
        coords="minimal",
        compat="override",
        chunks={"time": 90, "Y": 256, "X": 256},)

    da = ds["rr"]

    # Mask the explicit fill value before any computation
    da = da.where(da != -999.99)

    da.attrs["units"] = "mm"
    da.name = "rr_mm"

    # Validate time axis
    times = da["time"].values
    if not np.all(np.diff(times.astype("int64")) > 0):
        raise ValueError(
            "seNorge time axis is not strictly increasing after concatenation.\n"
            "Check for duplicate or overlapping annual files."
        )

    return da


def get_year_range_senorge(senorge_files: list[Path]) -> tuple[int, int]:
    """Extract start_year and end_year from the sorted seNorge file list."""
    pattern = re.compile(r"^rr_(\d{4})\.nc$")
    years = [int(pattern.match(f.name).group(1)) for f in senorge_files]
    return min(years), max(years)

# Additional helper for SMILE figure paths 

def latlon_extent_to_utm_bbox(
    extent: tuple,
    epsg_dst: str = "EPSG:32633",
    n_edge: int = 80,
    pad_m: float = 25_000.0,) -> tuple:
    """
    Convert a (west, east, south, north) lat/lon extent to a UTM bounding box
    suitable for cropping seNorge data.

    Samples many points along the boundary and adds a padding margin so that
    reprojection distortion at high latitudes does not leave white gaps.

    Returns
    -------
    (xmin, xmax, ymin, ymax) in the target projected CRS (metres).
    """
    west, east, south, north = extent
    xs = np.concatenate([
        np.linspace(west, east, n_edge),
        np.full(n_edge, east),
        np.linspace(east, west, n_edge),
        np.full(n_edge, west),
    ])
    ys = np.concatenate([
        np.full(n_edge, south),
        np.linspace(south, north, n_edge),
        np.full(n_edge, north),
        np.linspace(north, south, n_edge),
    ])
    transformer = Transformer.from_crs("EPSG:4326", epsg_dst, always_xy=True)
    x_utm, y_utm = transformer.transform(xs, ys)
    return (
        float(np.nanmin(x_utm) - pad_m),
        float(np.nanmax(x_utm) + pad_m),
        float(np.nanmin(y_utm) - pad_m),
        float(np.nanmax(y_utm) + pad_m),)

# ── Overall-precipitation spatial-cache builders ───────────────────────────────

def save_senorge_overall(
    senorge_dir: Path,
    out_path_fn,
    extent: tuple,
    force: bool = False,
) -> None:
    """
    Build and save the spatially-cropped daily seNorge precipitation cache.

    Parameters
    ----------
    out_path_fn : callable(dataset, resolution, start_year, end_year) -> Path
                  e.g. cfg.overall_precip_path  (will be called with dataset="senorge", resolution="")
    extent      : (west, east, south, north) in degrees
    """
    from catchment_tools import save_spatial_netcdf
    files = find_senorge_files(senorge_dir)
    start_year, end_year = get_year_range_senorge(files)
    out_path = out_path_fn("senorge", "", start_year, end_year)

    if (not force) and out_path.exists():
        print(f"  [skip] SeNorge cache exists: {out_path.name}")
        return

    print(f"  Building SeNorge DAILY cache ({start_year}–{end_year}) ...")
    xmin, xmax, ymin, ymax = latlon_extent_to_utm_bbox(extent)

    ds = xr.open_mfdataset(
        [str(f) for f in files], combine="by_coords", coords="minimal",
        compat="override", mask_and_scale=False,
        chunks={"time": 31, "Y": 256, "X": 256})
    try:
        # seNorge Y (northing) is stored DESCENDING and X (easting) ASCENDING.
        # Order each slice to match its coordinate direction, else .sel returns
        # an empty range (Y=0) and the cache is silently built empty.
        yv = ds["Y"].values
        xv = ds["X"].values
        y_slice = slice(ymin, ymax) if (yv.size < 2 or yv[0] <= yv[-1]) else slice(ymax, ymin)
        x_slice = slice(xmin, xmax) if (xv.size < 2 or xv[0] <= xv[-1]) else slice(xmax, xmin)
        da = ds["rr"].sel(X=x_slice, Y=y_slice).astype("float32")
        da = da.where(da != np.float32(-999.99)).sortby("time")
        save_spatial_netcdf(da, out_path, "rr_mm",
                            time_chunk=31, y_chunk=256, x_chunk=256)
    finally:
        ds.close()


def compute_senorge_annual_median_2d(
    start_year: int,
    end_year: int,
    senorge_dir: Path,
    cache_path_fn,
    open_cache_fn,
) -> "xr.DataArray":
    """Annual median daily precipitation (mm/year) from seNorge overall cache."""
    files = find_senorge_files(senorge_dir)
    avail_s, avail_e = get_year_range_senorge(files)
    cache = cache_path_fn("senorge", "", avail_s, avail_e)
    if not cache.exists():
        raise FileNotFoundError(f"SeNorge overall cache not found.\n{cache}")
    print(f"  Loading SeNorge from cache ({start_year}–{end_year}) ...")
    da = open_cache_fn(cache, start_year, end_year)
    out = da.groupby("time.year").sum(dim="time").median(dim="year")
    out.name = "annmedian_precip"
    out.attrs.update({"units": "mm/year",
                      "start_year": start_year, "end_year": end_year})
    return out.compute()


def compute_senorge_2day_median_2d(
    start_year: int,
    end_year: int,
    senorge_dir: Path,
    cache_path_fn,
    open_cache_fn,
    rolling_fn,
    subset_fn,
) -> "xr.DataArray":
    """
    Median of 2-day rolling sums (mm) from seNorge overall 1-day cache.
    rolling_fn : catchment_tools.rolling_accumulation
    subset_fn  : catchment_tools.subset_time_series_by_year
    """
    files = find_senorge_files(senorge_dir)
    avail_s, avail_e = get_year_range_senorge(files)
    cache = cache_path_fn("senorge", "", avail_s, avail_e)
    if not cache.exists():
        raise FileNotFoundError(f"SeNorge overall cache not found.\n{cache}")
    print(f"  Loading SeNorge (2-day) from cache ({start_year}–{end_year}) ...")
    da = open_cache_fn(cache, start_year - 1, end_year)
    da_2day = rolling_fn(da, window_days=2)
    return subset_fn(da_2day, start_year, end_year).median(dim="time").compute()
