# data_era5.py
"""
ERA5 file discovery, loading, unit conversion, and optional full-grid caching.
"""

import re
import xarray as xr
import numpy as np
from pathlib import Path


def find_era5_files(era5_dir: Path, resolution: str) -> list[Path]:
    """
    Return a chronologically sorted list of ERA5 annual NetCDF files
    for the requested resolution.

    Expected filename pattern:  tp24_<resolution>_<year>.nc
    Example:                    tp24_0.5x0.5_1941.nc
    """
    pattern = re.compile(rf"^tp24_{re.escape(resolution)}_(\d{{4}})\.nc$")
    matched = []
    for f in era5_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            matched.append((int(m.group(1)), f))

    if not matched:
        raise FileNotFoundError(
            f"No ERA5 files found in:\n  {era5_dir}\n"
            f"for resolution '{resolution}'.\n"
            f"Expected filenames like: tp24_{resolution}_1941.nc"
        )

    matched.sort(key=lambda x: x[0])
    return [f for _, f in matched]


def load_era5_precipitation(era5_files: list[Path]) -> xr.DataArray:
    """
    Open and concatenate annual ERA5 files into a single lazy DataArray.
    - Converts precipitation from meters to millimeters.
    - Verifies that the time axis is strictly increasing (no duplicates).

    Returns
    -------
    xr.DataArray
        Dimensions: (time, latitude, longitude), units: mm.
        Values are loaded lazily via Dask — no RAM issue even for 80 years.
    """
    print(f"  Opening {len(era5_files)} ERA5 files (lazy) ...")
    ds = xr.open_mfdataset([str(f) for f in era5_files], combine="by_coords", coords="minimal", compat="override", chunks={"time": 365},)

    # If an unnecessary singleton ensemble/member dimension exists, remove it
    if "number" in ds.dims and ds.sizes["number"] == 1:
        ds = ds.isel(number=0, drop=True)

    da = ds["tp24"] * 1000.0    # meters → millimeters
    da.attrs["units"] = "mm"
    da.name = "tp24_mm"

    # Validate time axis (check before any computation)
    times = da["time"].values
    if not np.all(np.diff(times.astype("int64")) > 0):
        raise ValueError(
            "ERA5 time axis is not strictly increasing after concatenation.\n"
            "Check for duplicate or overlapping annual files.")
    return da


def get_year_range(era5_files: list[Path], resolution: str) -> tuple[int, int]:
    """
    Extract start_year and end_year from the sorted list of ERA5 files.
    Relies on the filename pattern tp24_<resolution>_<year>.nc
    """
    pattern = re.compile(rf"^tp24_{re.escape(resolution)}_(\d{{4}})\.nc$")
    years = [int(pattern.match(f.name).group(1)) for f in era5_files]
    return min(years), max(years)