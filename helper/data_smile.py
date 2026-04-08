# data_smile.py
"""
SMILE Climate Model large-ensemble file discovery and loading.

Supported datasets
------------------
  cesm2_le          : cesm2-le.tp24.<mmm>.<YYYYMMDD>-<YYYYMMDD>.nc
  gfdl_spear_med_le : gfdl_spear_med_le.tp24.<mm>.<YYYYMMDD>-<YYYYMMDD>.nc

"""

#Load important libraries
import re
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


# ── Filename patterns ──────────────────────────────────────────────────────
# Capture groups:  (1) member_id   (2) file_start_yyyymmdd   (3) file_end_yyyymmdd
_FILE_PATTERNS: dict[str, re.Pattern] = {
    "cesm2_le": re.compile(
        r"^cesm2-le\.tp24\.(\d{3})\.(\d{8})-(\d{8})\.nc$"
    ),
    "gfdl_spear_med_le": re.compile(
        r"^gfdl_spear_med_le\.tp24\.(\d{2})\.(\d{8})-(\d{8})\.nc$"
    ),}


# Define Ensemble Member discovery ────────────────────────────────────────────────────────
def find_smile_members(model_dir: Path, dataset: str) -> list[str]:
    """Return a sorted list of all unique member IDs present in model_dir"""
    pat = _FILE_PATTERNS[dataset]
    members: set[str] = set()
    for f in model_dir.iterdir():
        m = pat.match(f.name)
        if m:
            members.add(m.group(1))
    if not members:
        raise FileNotFoundError(
            f"No {dataset} files found in:\n  {model_dir}\n"
            f"Expected filename pattern: {pat.pattern}")
    return sorted(members)


# Define File discovery for one member ─────────────────────────────────────────────────
def find_smile_files_for_member(
    model_dir: Path,
    dataset: str,
    member_id: str,
    start_year: int | None = None,
    end_year:   int | None = None,
) -> list[Path]:
    """
    Return the chronologically sorted file list for one ensemble member
    """
    pat = _FILE_PATTERNS[dataset]
    matched: list[tuple[str, str, Path]] = []

    for f in model_dir.iterdir():
        m = pat.match(f.name)
        if not m or m.group(1) != member_id:
            continue

        file_start = m.group(2)  # YYYYMMDD
        file_end   = m.group(3)  # YYYYMMDD

        file_start_year = int(file_start[:4])
        file_end_year   = int(file_end[:4])

        if start_year is not None and file_end_year < start_year:
            continue
        if end_year is not None and file_start_year > end_year:
            continue

        matched.append((file_start, file_end, f))

    if not matched:
        raise FileNotFoundError(
            f"No {dataset} files found for member {member_id} in:\n  {model_dir}")

    matched.sort(key=lambda x: (x[0], x[1], x[2].name))
    return [f for _, _, f in matched]

# Define SMILE tp24 unit conversion ─────────────────────────────────────────────────
def _convert_smile_tp24_to_mm(da: xr.DataArray, unit_mode: str = "auto") -> xr.DataArray:
    """
    Convert raw SMILE tp24 to daily precipitation amount in mm

    unit_mode
    ---------
    "auto"       : infer from da.attrs["units"]
    "m_to_mm"    : raw tp24 is a daily total in metres
    "already_mm" : raw tp24 is already a daily total in mm or kg m-2
    """
    unit_mode = unit_mode.lower()
    if unit_mode not in {"auto", "m_to_mm", "already_mm"}:
        raise ValueError(
            "unit_mode must be one of: 'auto', 'm_to_mm', 'already_mm'")

    units_raw = str(da.attrs.get("units", "")).strip()
    u = units_raw.lower().replace(" ", "")

    if unit_mode == "m_to_mm":
        factor = 1000.0
        note = "forced m_to_mm"

    elif unit_mode == "already_mm":
        factor = 1.0
        note = "forced already_mm"

    else:
        # Auto-detect from metadata
        if any(tok in u for tok in [
            "kgm-2", "kg/m2", "kgm^-2", "kgm**-2",
            "mm", "millimetre", "millimeter"]):
            factor = 1.0
            note = "auto: already mm or kg m-2"

        elif u in {"m", "m/day", "md-1", "mday-1", "metre", "meter"} or u.startswith("m/"):
            factor = 1000.0
            note = "auto: m_to_mm"

        else:
            raise ValueError(
                "Could not infer SMILE tp24 units from metadata. "
                f"Found units={units_raw!r}. "
                "Inspect one raw file and then set "
                "cfg.SMILE_CONFIG[dataset]['tp24_unit_mode'] to "
                "'m_to_mm' or 'already_mm'.")

    out = da * factor
    out.attrs = dict(da.attrs)
    out.attrs["raw_units"] = units_raw if units_raw else "missing"
    out.attrs["units_conversion"] = note
    out.attrs["units"] = "mm"
    out.name = "tp24_mm"

    print(f"  [units] tp24 raw units={units_raw!r} -> mm ({note})")
    return out

# Define Loading for Climatology Model ───────────────────────────────────────────────────────
def load_smile_precipitation(
    files:      list[Path],
    start_year: int | None = None,
    end_year:   int | None = None,
    unit_mode:  str = "auto",) -> xr.DataArray:
    """
    Open and concatenate SMILE block files

    - Converts tp24 from metres to millimetres
    - Optionally clips to [start_year, end_year]
    - Sorts by time
    - Removes duplicate timestamps at file boundaries
    - Verifies that the final time axis is strictly increasing

    Returns
    -------
    xr.DataArray
        Dimensions: (time, lat, lon), units: mm
    """
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        coords="minimal",
        compat="override",
        chunks={"time": 365},)

    # Drop singleton member/ensemble dimension if any
    for dim_name in ("member", "ensemble", "ens", "number"):
        if dim_name in ds.dims and ds.sizes[dim_name] == 1:
            ds = ds.isel({dim_name: 0}, drop=True)

    da = _convert_smile_tp24_to_mm(ds["tp24"], unit_mode=unit_mode)

    # Year-range clip
    if start_year is not None or end_year is not None:
        yr = da["time"].dt.year
        mask = xr.ones_like(yr, dtype=bool)
        if start_year is not None:
            mask = mask & (yr >= start_year)
        if end_year is not None:
            mask = mask & (yr <= end_year)
        da = da.isel(time=mask.values)

    # Enforce chronological order after concatenation
    da = da.sortby("time")

    # Remove duplicate timestamps that can occur at block boundaries
    time_index = pd.Index(da["time"].values)
    dup_mask = time_index.duplicated(keep="first")
    n_dup = int(dup_mask.sum())

    if n_dup > 0:
        print(f"  [time] Removing {n_dup} duplicate timestamp(s) at SMILE block boundaries ...")
        da = da.isel(time=~dup_mask)

    # Final validation
    times = da["time"].values
    if len(times) > 1:
        bad_idx = next(
            (
                i for i, (t_prev, t_curr)
                in enumerate(zip(times[:-1], times[1:]))
                if not (t_curr > t_prev)),
            None,)
        if bad_idx is not None:
            raise ValueError(
                "SMILE time axis is still not strictly increasing after sorting "
                "and de-duplicating.\n"
                f"Problem around: {times[bad_idx]!r} -> {times[bad_idx + 1]!r}\n"
                "Check the raw block files for overlapping or misnamed date ranges.")
    return da


# Define Year-range helper ──────────────────────────────────────────────────────────
def get_year_range_smile(model_dir: Path, dataset: str) -> tuple[int, int]:
    """
    Return the full (start_year, end_year) covered by the dataset on disk
    Uses the first member found; all members share the same time span
    """
    pat = _FILE_PATTERNS[dataset]
    members = find_smile_members(model_dir, dataset)
    first = members[0]
    years: list[int] = []

    for f in model_dir.iterdir():
        m = pat.match(f.name)
        if m and m.group(1) == first:
            years += [int(m.group(2)[:4]), int(m.group(3)[:4])]

    if not years:
        raise FileNotFoundError(
            f"No files found for member {first} in {model_dir}")

    return min(years), max(years)