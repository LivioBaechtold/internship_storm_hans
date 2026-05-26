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