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
    # Variable token is matched generically (PRECT, SWE, SM, ...) so the same
    # discovery works for every single-variable CESM2-LE directory.
    # Groups stay (1)=member (2)=start_yyyymmdd (3)=end_yyyymmdd.
    "cesm2_le": re.compile(
        r"^cesm2-le\.[A-Za-z0-9_]+\.(\d{3})\.(\d{8})-(\d{8})\.nc$"
    ),
    "gfdl_spear_med_le": re.compile(
        r"^gfdl_spear_med_le\.tp24\.(\d{2})\.(\d{8})-(\d{8})\.nc$"
    ),}


# Define Ensemble Member discovery ────────────────────────────────────────────────────────
_SMILE_VAR_TOKEN = re.compile(
    r"^(?:cesm2-le|gfdl_spear_med_le)\.([A-Za-z0-9_]+)\.\d+\.\d{8}-\d{8}\.nc$")

def find_smile_members(model_dir: Path, dataset: str) -> list[str]:
    """Return a sorted list of all unique member IDs present in model_dir.

    For cesm2_le the variable token (PRECT, SWE, SM, ...) is matched generically,
    so the same discovery works for any single-variable CESM2-LE directory.
    A directory must hold exactly ONE variable; a mixed folder (e.g. the legacy
    tp24 directory that also contains PRECT files) raises a clear error.
    """
    pat = _FILE_PATTERNS[dataset]
    members: set[str] = set()
    variables: set[str] = set()
    for f in model_dir.iterdir():
        m = pat.match(f.name)
        if m:
            members.add(m.group(1))
            vm = _SMILE_VAR_TOKEN.match(f.name)
            if vm:
                variables.add(vm.group(1))
    if not members:
        raise FileNotFoundError(
            f"No {dataset} files found in:\n  {model_dir}\n"
            f"Expected filename pattern: {pat.pattern}")
    if len(variables) > 1:
        raise ValueError(
            f"{model_dir} contains multiple variables {sorted(variables)}; "
            "point to a single-variable directory (e.g. .../PRECT, .../SWE, .../SM).")
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

    # CESM2-LE precip files now use 'PRECT'; GFDL still uses 'tp24'. Pick whichever
    # is present (both are daily totals in mm/day → already_mm).
    _precip_var = next((v for v in ("PRECT", "tp24") if v in ds.data_vars), None)
    if _precip_var is None:
        raise KeyError(
            "No recognised SMILE precipitation variable (expected 'PRECT' or "
            f"'tp24'); found {list(ds.data_vars)} in the opened files.")
    da = _convert_smile_tp24_to_mm(ds[_precip_var], unit_mode=unit_mode)

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


# ── Overall-precipitation spatial-cache builders ───────────────────────────────

def save_smile_overall(
    dataset: str,
    model_dir: Path,
    out_path_fn,
    force: bool = False,
    unit_mode: str = "auto",
) -> None:
    """
    Build and save per-member daily SMILE precipitation for the full available period.

    Parameters
    ----------
    out_path_fn : callable(member_id, avail_start, avail_end) -> Path
        e.g. functools.partial(cfg.overall_precip_member_path, dataset)
    """
    from catchment_tools import save_spatial_netcdf
    members = find_smile_members(model_dir, dataset)
    avail_start, avail_end = get_year_range_smile(model_dir, dataset)
    print(f"  {dataset}: {len(members)} members, {avail_start}–{avail_end} ...")
    for i, mid in enumerate(members, start=1):
        out_path = out_path_fn(mid, avail_start, avail_end)
        if (not force) and out_path.exists():
            print(f"    [skip] member {mid} ({i}/{len(members)})")
            continue
        print(f"    [build] member {mid} ({i}/{len(members)}) ...")
        files = find_smile_files_for_member(model_dir, dataset, mid,
                                            avail_start, avail_end)
        da = load_smile_precipitation(files, avail_start, avail_end,
                                      unit_mode=unit_mode)
        save_spatial_netcdf(da.astype("float32"), out_path, "tp24_mm",
                            time_chunk=365, y_chunk=64, x_chunk=64)


def load_cesm2_le_field(
    files: list[Path],
    variable: str,
    start_year: int | None = None,
    end_year:   int | None = None,
) -> xr.DataArray:
    """
    Open and concatenate CESM2-LE block files for a state variable ('SWE','SM').

    No unit conversion (values kept in native kg/m²). Same robustness as
    `load_smile_precipitation`: optional year clip, chronological sort, removal
    of duplicate block-boundary timestamps.
    """
    ds = xr.open_mfdataset(
        [str(f) for f in files], combine="by_coords", coords="minimal",
        compat="override", chunks={"time": 365})
    for dim_name in ("member", "ensemble", "ens", "number"):
        if dim_name in ds.dims and ds.sizes[dim_name] == 1:
            ds = ds.isel({dim_name: 0}, drop=True)
    if variable not in ds:
        raise KeyError(
            f"Variable {variable!r} not found in CESM2-LE files; "
            f"found {list(ds.data_vars)}")
    da = ds[variable]

    if start_year is not None or end_year is not None:
        yr = da["time"].dt.year
        mask = xr.ones_like(yr, dtype=bool)
        if start_year is not None:
            mask = mask & (yr >= start_year)
        if end_year is not None:
            mask = mask & (yr <= end_year)
        da = da.isel(time=mask.values)

    da = da.sortby("time")
    dup_mask = pd.Index(da["time"].values).duplicated(keep="first")
    if int(dup_mask.sum()) > 0:
        print(f"  [time] Removing {int(dup_mask.sum())} duplicate timestamp(s) ...")
        da = da.isel(time=~dup_mask)

    da.attrs["units"] = str(ds[variable].attrs.get("units", "kg/m2"))
    da.name = variable
    return da


def save_cesm2_le_field_overall(
    model_dir: Path,
    out_path_fn,            # callable(member_id, avail_start, avail_end) -> Path
    variable: str,          # raw variable in the files, e.g. 'SWE' or 'SM'
    cache_var: str,         # name stored in the cache, e.g. 'swe' or 'soil_moisture'
    units: str = "kg/m2",
    force: bool = False,
) -> None:
    """
    Build/save per-member daily CESM2-LE state-field caches (SWE / soil moisture)
    over the full available period — the per-member counterpart of
    `save_smile_overall`, but without mm conversion.
    """
    from catchment_tools import save_spatial_netcdf
    members = find_smile_members(model_dir, "cesm2_le")
    avail_start, avail_end = get_year_range_smile(model_dir, "cesm2_le")
    print(f"  cesm2_le/{cache_var}: {len(members)} members, {avail_start}–{avail_end} ...")
    for i, mid in enumerate(members, start=1):
        out_path = out_path_fn(mid, avail_start, avail_end)
        if (not force) and out_path.exists():
            print(f"    [skip] member {mid} ({i}/{len(members)})")
            continue
        print(f"    [build] member {mid} ({i}/{len(members)}) ...")
        files = find_smile_files_for_member(model_dir, "cesm2_le", mid,
                                            avail_start, avail_end)
        da = load_cesm2_le_field(files, variable, avail_start, avail_end)
        save_spatial_netcdf(da.astype("float32"), out_path, cache_var,
                            time_chunk=365, y_chunk=64, x_chunk=64, units=units)

def save_cesm2_le_field_diff_overall(
    model_dir: Path,
    in_path_fn,             # callable(member_id, avail_start, avail_end) -> Path (1-day cache)
    out_path_fn,            # callable(member_id, avail_start, avail_end) -> Path (N-day ΔSWE cache)
    cache_var: str,         # variable name stored in the cache, e.g. 'swe'
    diff_fn,                # callable(da, window_days) -> xr.DataArray, e.g. rolling_melt
    open_cache_fn,          # callable(path, start_year, end_year) -> xr.DataArray
    window_days: int,
    units: str = "kg/m2",
    force: bool = False,
) -> None:
    """
    Build/save per-member daily CESM2-LE N-day snowmelt caches from the raw 1-day
    field caches. For each member it opens the 1-day cache, applies
    ``diff_fn(da, window_days)`` (max(0, −ΔSWE): SWE decrease → positive snowmelt,
    SWE gain → 0) over the FULL available record, and writes it to the N-day cache path.
    """
    from catchment_tools import save_spatial_netcdf
    members = find_smile_members(model_dir, "cesm2_le")
    avail_start, avail_end = get_year_range_smile(model_dir, "cesm2_le")
    print(f"  cesm2_le/{cache_var}: {len(members)} members, "
          f"{window_days}-day ΔSWE, {avail_start}–{avail_end} ...")
    for i, mid in enumerate(members, start=1):
        out_path = out_path_fn(mid, avail_start, avail_end)
        if (not force) and out_path.exists():
            print(f"    [skip] member {mid} ({i}/{len(members)})")
            continue
        in_path = in_path_fn(mid, avail_start, avail_end)
        if not in_path.exists():
            raise FileNotFoundError(
                f"Raw 1-day cache missing: {in_path}\n"
                "    Build the 1-day SWE caches first (SWE & soil-moisture daily-cache cell).")
        print(f"    [build] member {mid} ({i}/{len(members)}) ...")
        da = open_cache_fn(in_path, avail_start, avail_end)
        da_diff = diff_fn(da, window_days=window_days)
        save_spatial_netcdf(da_diff.astype("float32"), out_path, cache_var,
                            time_chunk=365, y_chunk=64, x_chunk=64, units=units)

def compute_cesm2_le_annual_median_2d(
    start_year: int,
    end_year: int,
    model_dir: Path,
    member_cache_path_fn,
    median_cache_path: Path,
    open_cache_fn,
    force_recompute: bool = False,
) -> "xr.DataArray":
    """
    100-member median of annual-total daily precipitation from CESM2-LE overall caches.

    Parameters
    ----------
    member_cache_path_fn : callable(member_id, avail_s, avail_e) -> Path
    open_cache_fn        : callable(path, start_year, end_year) -> xr.DataArray
    median_cache_path    : Path  — where to save/load the result
    """
    members = find_smile_members(model_dir, "cesm2_le")
    avail_s, avail_e = get_year_range_smile(model_dir, "cesm2_le")

    if (not force_recompute) and median_cache_path.exists():
        print(f"  [cache] CESM2-LE annual median ← {median_cache_path.name}")
        with xr.open_dataset(str(median_cache_path)) as ds:
            return ds["annmedian_precip"].load()

    print(f"  Computing CESM2-LE {len(members)}-member annual median "
          f"({start_year}–{end_year}) — pooling all member × year totals ...")
    all_annual_arrs = []
    lat_coord = lon_coord = None
    for i, mid in enumerate(members):
        cache = member_cache_path_fn(mid, avail_s, avail_e)
        if not cache.exists():
            raise FileNotFoundError(f"Per-member cache missing: {cache}")
        da = open_cache_fn(cache, start_year, end_year)
        annual = da.groupby("time.year").sum(dim="time").load()  # [year, lat, lon]
        all_annual_arrs.append(annual.values)
        if lat_coord is None:
            lat_coord = annual["lat"]
            lon_coord = annual["lon"]
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(members)} members processed ...")

    # True global median: pool all n_members × n_years annual totals per pixel
    all_pooled = np.concatenate(all_annual_arrs, axis=0)   # [n_members*n_years, lat, lon]
    median = xr.DataArray(np.nanmedian(all_pooled, axis=0).astype("float32"),
        dims=["lat", "lon"], coords={"lat": lat_coord, "lon": lon_coord},)

    median.name = "annmedian_precip"
    median.attrs.update({"units": "mm/year",
                         "start_year": start_year, "end_year": end_year})
    median_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"annmedian_precip": median}).to_netcdf(
        str(median_cache_path),
        encoding={"annmedian_precip": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {median_cache_path.name}")
    return median


def compute_cesm2_le_window_median_2d(
    start_year: int,
    end_year: int,
    model_dir: Path,
    member_cache_path_fn,
    median_cache_path: Path,
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> "xr.DataArray":
    """
    100-member median of the 2-day rolling precipitation from CESM2-LE caches.

    Parameters
    ----------
    rolling_fn : callable(da, window_days) -> xr.DataArray  — e.g. rolling_accumulation
    subset_fn  : callable(da, start_year, end_year) -> xr.DataArray
    """
    members = find_smile_members(model_dir, "cesm2_le")
    avail_s, avail_e = get_year_range_smile(model_dir, "cesm2_le")

    if (not force_recompute) and median_cache_path.exists():
        print(f"  [cache] CESM2-LE 2-day median ← {median_cache_path.name}")
        with xr.open_dataset(str(median_cache_path)) as ds:
            return ds["twodaymedian_precip"].load()

    print(f"  Computing CESM2-LE {len(members)}-member 2-day median "
          f"({start_year}–{end_year}) ...")
    member_medians = []
    for i, mid in enumerate(members):
        cache = member_cache_path_fn(mid, avail_s, avail_e)
        if not cache.exists():
            raise FileNotFoundError(f"Per-member cache missing: {cache}")
        da = open_cache_fn(cache, start_year - 1, end_year)
        da_2day = rolling_fn(da, window_days=window_days)
        da_2day = subset_fn(da_2day, start_year, end_year)
        member_medians.append(da_2day.median(dim="time"))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(members)} members processed ...")

    stacked = xr.concat(member_medians, dim="member")
    median = stacked.median(dim="member").compute()
    median.name = "twodaymedian_precip"
    median.attrs.update({"units": "mm",
                         "start_year": start_year, "end_year": end_year})
    median_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twodaymedian_precip": median}).to_netcdf(
        str(median_cache_path),
        encoding={"twodaymedian_precip": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {median_cache_path.name}")
    return median


def compute_cesm2_le_window_p90_2d(
    start_year: int,
    end_year: int,
    model_dir: "Path",
    member_cache_path_fn,
    p90_cache_path: "Path",
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> "xr.DataArray":
    """
    100-member median of the 90th percentile of 2-day rolling precipitation.

    Per member: compute 2-day rolling sums, take 90th percentile over time.
    Across members: take the median of the 100 per-member p90 fields.
    """
    members = find_smile_members(model_dir, "cesm2_le")
    avail_s, avail_e = get_year_range_smile(model_dir, "cesm2_le")

    if (not force_recompute) and p90_cache_path.exists():
        print(f"  [cache] CESM2-LE 2-day p90 ← {p90_cache_path.name}")
        with xr.open_dataset(str(p90_cache_path)) as ds:
            var = "twoday_p90_precip" if "twoday_p90_precip" in ds else "twodayp90_precip"
            return ds[var].load()


    print(f"  Computing CESM2-LE {len(members)}-member 2-day p90 "
          f"({start_year}–{end_year}) ...")
    member_p90s = []
    for i, mid in enumerate(members):
        cache = member_cache_path_fn(mid, avail_s, avail_e)
        if not cache.exists():
            raise FileNotFoundError(f"Per-member cache missing: {cache}")
        da = open_cache_fn(cache, start_year - 1, end_year)
        da_2day = rolling_fn(da, window_days=window_days)
        da_2day = subset_fn(da_2day, start_year, end_year)
        p90 = da_2day.quantile(0.90, dim="time")
        member_p90s.append(p90.drop_vars("quantile", errors="ignore"))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(members)} members processed ...")

    stacked = xr.concat(member_p90s, dim="member")
    result = stacked.median(dim="member").compute()
    result.name = "twoday_p90_precip"
    result.attrs.update({"units": "mm",
                         "start_year": start_year, "end_year": end_year})
    p90_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_p90_precip": result}).to_netcdf(
        str(p90_cache_path),
        encoding={"twoday_p90_precip": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {p90_cache_path.name}")
    return result


def compute_cesm2_le_window_global_median_2d(
    start_year: int,
    end_year: int,
    model_dir: "Path",
    member_cache_path_fn,
    global_median_cache_path: "Path",
    per_member_medians_cache_path: "Path",
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> "xr.DataArray":
    """
    True global median and per-member medians of 2-day rolling sums (CESM2-LE).

    Computes the median over ALL (n_members × n_timesteps) values per pixel —
    not a median-of-medians. Also saves per-member median fields for significance
    testing.

    Saves two cache files:
      global_median_cache_path      : [lat, lon] — global median
      per_member_medians_cache_path : [member, lat, lon] — one field per member
    """
    members = find_smile_members(model_dir, "cesm2_le")
    avail_s, avail_e = get_year_range_smile(model_dir, "cesm2_le")

    if (not force_recompute) and global_median_cache_path.exists() and per_member_medians_cache_path.exists():
        print(f"  [cache] CESM2-LE global 2-day median ← {global_median_cache_path.name}")
        with xr.open_dataset(str(global_median_cache_path)) as ds:
            return ds["twoday_global_median"].load()

    print(f"  Computing CESM2-LE {len(members)}-member 2-day GLOBAL median "
          f"({start_year}–{end_year}) — this reads all members ...")

    per_member_arrs = []   # list of [T, lat, lon] numpy arrays after subset
    per_member_medians = []  # list of [lat, lon] numpy arrays (one per member)

    for i, mid in enumerate(members):
        cache = member_cache_path_fn(mid, avail_s, avail_e)
        if not cache.exists():
            raise FileNotFoundError(f"Per-member cache missing: {cache}")
        da = open_cache_fn(cache, start_year - 1, end_year)
        da_2day = rolling_fn(da, window_days=window_days)
        da_sub = subset_fn(da_2day, start_year, end_year).load()
        arr = da_sub.values                         # [T, lat, lon]
        per_member_arrs.append(arr)
        per_member_medians.append(np.nanmedian(arr, axis=0))  # [lat, lon]
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(members)} members loaded ...")

    # ── Global median over all (member × time) values per pixel ───────────────
    all_values = np.concatenate(per_member_arrs, axis=0)  # [100*T, lat, lon]
    global_median_arr = np.nanmedian(all_values, axis=0)   # [lat, lon]

    # Build output DataArrays using coordinate info from the last loaded member
    lat_coord = da_sub["lat"]
    lon_coord = da_sub["lon"]

    global_median = xr.DataArray(
        global_median_arr.astype("float32"),
        dims=["lat", "lon"],
        coords={"lat": lat_coord, "lon": lon_coord},
        name="twoday_global_median",
        attrs={"units": "mm", "start_year": start_year, "end_year": end_year,
               "description": "True global median of 2-day rolling sums across all members × time"},
    )

    stacked_medians = xr.DataArray(
        np.stack(per_member_medians, axis=0).astype("float32"),  # [100, lat, lon]
        dims=["member", "lat", "lon"],
        coords={"member": members, "lat": lat_coord, "lon": lon_coord},
        name="twoday_per_member_median",
        attrs={"units": "mm", "start_year": start_year, "end_year": end_year,
               "description": "Per-member median of 2-day rolling sums [n_members, lat, lon]"},
    )

    global_median_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_global_median": global_median}).to_netcdf(
        str(global_median_cache_path),
        encoding={"twoday_global_median": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {global_median_cache_path.name}")

    per_member_medians_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_per_member_median": stacked_medians}).to_netcdf(
        str(per_member_medians_cache_path),
        encoding={"twoday_per_member_median": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {per_member_medians_cache_path.name}")

    return global_median


def compute_cesm2_le_window_per_member_p90_2d(
    start_year: int,
    end_year: int,
    model_dir: "Path",
    member_cache_path_fn,
    per_member_p90_cache_path: "Path",
    global_p90_cache_path: "Path",
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> tuple["xr.DataArray", "xr.DataArray"]:
    """
    Compute per-member 90th-percentile fields and the global 90th percentile.

    Returns
    -------
    (global_p90, per_member_p90_da)
      global_p90        : [lat, lon] — true 90th pctl of all member×time values
      per_member_p90_da : [member, lat, lon] — one p90 field per member (for significance)
    """
    members = find_smile_members(model_dir, "cesm2_le")
    avail_s, avail_e = get_year_range_smile(model_dir, "cesm2_le")

    if (not force_recompute) and per_member_p90_cache_path.exists() and global_p90_cache_path.exists():
        print(f"  [cache] CESM2-LE per-member p90 ← {per_member_p90_cache_path.name}")
        with xr.open_dataset(str(per_member_p90_cache_path)) as ds:
            per_member_da = ds["twoday_per_member_p90"].load()
        with xr.open_dataset(str(global_p90_cache_path)) as ds:
            var = "twoday_global_p90" if "twoday_global_p90" in ds else "twoday_p90_precip"
            global_da = ds[var].load()
        return global_da, per_member_da

    print(f"  Computing CESM2-LE {len(members)}-member 2-day per-member p90 "
          f"({start_year}–{end_year}) ...")

    per_member_arrs = []
    per_member_p90s = []

    for i, mid in enumerate(members):
        cache = member_cache_path_fn(mid, avail_s, avail_e)
        if not cache.exists():
            raise FileNotFoundError(f"Per-member cache missing: {cache}")
        da = open_cache_fn(cache, start_year - 1, end_year)
        da_2day = rolling_fn(da, window_days=window_days)
        da_sub = subset_fn(da_2day, start_year, end_year).load()
        arr = da_sub.values
        per_member_arrs.append(arr)
        per_member_p90s.append(np.nanpercentile(arr, 90, axis=0))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(members)} members processed ...")

    all_values = np.concatenate(per_member_arrs, axis=0)
    global_p90_arr = np.nanpercentile(all_values, 90, axis=0)

    lat_coord = da_sub["lat"]
    lon_coord = da_sub["lon"]

    global_p90 = xr.DataArray(
        global_p90_arr.astype("float32"),
        dims=["lat", "lon"],
        coords={"lat": lat_coord, "lon": lon_coord},
        name="twoday_global_p90",
        attrs={"units": "mm", "start_year": start_year, "end_year": end_year},
    )
    stacked_p90 = xr.DataArray(
        np.stack(per_member_p90s, axis=0).astype("float32"),
        dims=["member", "lat", "lon"],
        coords={"member": members, "lat": lat_coord, "lon": lon_coord},
        name="twoday_per_member_p90",
        attrs={"units": "mm", "start_year": start_year, "end_year": end_year},
    )

    global_p90_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_global_p90": global_p90}).to_netcdf(
        str(global_p90_cache_path),
        encoding={"twoday_global_p90": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {global_p90_cache_path.name}")

    per_member_p90_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_per_member_p90": stacked_p90}).to_netcdf(
        str(per_member_p90_cache_path),
        encoding={"twoday_per_member_p90": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {per_member_p90_cache_path.name}")

    return global_p90, stacked_p90


def compute_significance_masks(
    da_era5: "xr.DataArray",
    per_member_da: "xr.DataArray",
    lower_pctl: float,
    upper_pctl: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Simple 2-sided percentile-rank significance test (no FDR correction).

    For each pixel, a cell is marked significant if ERA5 falls strictly outside
    the [lower_pctl, upper_pctl] range of the CESM2-LE ensemble by direct member
    count: sig if n_greater >= threshold (ERA5 ≤ lower pctile) or n_less >=
    threshold (ERA5 ≥ upper pctile), where threshold = round(upper_pctl/100 *
    n_members).

    Parameters
    ----------
    da_era5       : [lat, lon] — ERA5 interpolated value (median or p90)
        per_member_da : [member, lat, lon] — per-member CESM2-LE values
                    (n_members fields; 90 for SWE/soil moisture, 100 for precip)
    lower_pctl    : lower percentile threshold (e.g. 5.0); used only as label —
                    threshold is derived from upper_pctl (lower_pctl = 100 - upper_pctl)
    upper_pctl    : upper percentile threshold (e.g. 95.0 → threshold = 95 members)

    Returns
    -------
    (sig_cesm_higher, sig_era5_higher) : tuple of 2-D bool arrays [lat, lon]
      sig_cesm_higher : True where ERA5 ≤ lower_pctl of ensemble (CESM2-LE > ERA5)
      sig_era5_higher : True where ERA5 ≥ upper_pctl of ensemble (ERA5 > CESM2-LE)
    """
    if per_member_da.shape[1:] != da_era5.shape:
        per_member_da = per_member_da.sel(lat=da_era5["lat"], lon=da_era5["lon"])

    members_arr = per_member_da.values          # [n_members, lat, lon]
    era5_arr    = da_era5.values                # [lat, lon]
    n_members   = members_arr.shape[0]

    # Per-pixel counts (strict comparisons; ties contribute to neither)
    n_greater = np.sum(members_arr > era5_arr[np.newaxis], axis=0)   # [lat, lon]
    n_less    = np.sum(members_arr < era5_arr[np.newaxis], axis=0)   # [lat, lon]

    # threshold: e.g. upper_pctl=95 → 95 members must exceed/be below ERA5
    threshold = round(upper_pctl / 100.0 * n_members)

    sig_cesm_higher = n_greater >= threshold   # ERA5 ≤ lower_pctl of ensemble
    sig_era5_higher = n_less    >= threshold   # ERA5 ≥ upper_pctl of ensemble

    # NaN propagation: ocean/masked pixels → not significant
    nan_mask = ~np.isfinite(era5_arr)
    sig_cesm_higher[nan_mask] = False
    sig_era5_higher[nan_mask] = False

    return sig_cesm_higher, sig_era5_higher

    # ── Seasonal helpers ───────────────────────────────────────────────────────────
_SEASON_MONTHS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

def _season_time_mask(time_values, season: str) -> "np.ndarray":
    """Boolean array selecting timesteps whose month belongs to season."""
    # pd.to_datetime cannot handle cftime.DatetimeNoLeap (used by CESM2-LE's
    # noleap calendar). Both cftime and numpy datetime64 expose .month directly,
    # so extract month element-wise when the standard conversion fails.
    try:
        months = pd.to_datetime(time_values).month.values
    except TypeError:
        months = np.array([t.month for t in time_values], dtype=np.int8)
    return np.isin(months, _SEASON_MONTHS[season.upper()])


def compute_cesm2_le_window_seasonal_global_median_2d(
    season: str,
    start_year: int,
    end_year: int,
    model_dir: "Path",
    member_cache_path_fn,
    global_median_cache_path: "Path",
    per_member_medians_cache_path: "Path",
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> "tuple[xr.DataArray, xr.DataArray]":
    """
    Seasonal global median and per-member medians of 2-day rolling sums (CESM2-LE).

    For each member the 2-day rolling sums are computed over the full annual period,
    then only timesteps in `season`'s months are kept. The global median pools all
    n_members × n_season_days values per pixel. Per-member medians (for significance
    testing) are stored separately.

    Returns
    -------
    (global_median [lat, lon], per_member_medians [n_members, lat, lon])
    """
    members = find_smile_members(model_dir, "cesm2_le")
    avail_s, avail_e = get_year_range_smile(model_dir, "cesm2_le")

    if (not force_recompute) and (global_median_cache_path.exists()
                                   and per_member_medians_cache_path.exists()):
        print(f"  [cache] CESM2-LE seasonal {season} global median "
              f"← {global_median_cache_path.name}")
        with xr.open_dataset(str(global_median_cache_path)) as _ds:
            global_median = _ds["twoday_seasonal_global_median"].load()
        with xr.open_dataset(str(per_member_medians_cache_path)) as _ds:
            per_member_da = _ds["twoday_seasonal_per_member_median"].load()
        return global_median, per_member_da

    print(f"  Computing CESM2-LE seasonal {season} {len(members)}-member 2-day global median "
          f"({start_year}–{end_year}) ...")

    per_member_season_arrs: list[np.ndarray] = []
    per_member_medians:      list[np.ndarray] = []
    lat_coord = lon_coord = None

    for i, mid in enumerate(members):
        cache = member_cache_path_fn(mid, avail_s, avail_e)
        if not cache.exists():
            raise FileNotFoundError(f"Per-member cache missing: {cache}")
        da = open_cache_fn(cache, start_year - 1, end_year)
        da_2day = rolling_fn(da, window_days=window_days)
        da_sub = subset_fn(da_2day, start_year, end_year).load()
        mask = _season_time_mask(da_sub["time"].values, season)
        arr = da_sub.values[mask]                         # [T_season, lat, lon]
        per_member_season_arrs.append(arr)
        per_member_medians.append(np.nanmedian(arr, axis=0))   # [lat, lon]
        if lat_coord is None:
            lat_coord = da_sub["lat"]
            lon_coord = da_sub["lon"]
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(members)} members processed ...")

    all_values = np.concatenate(per_member_season_arrs, axis=0)  # [n_members*T_s, lat, lon]
    global_median_arr = np.nanmedian(all_values, axis=0)

    global_median = xr.DataArray(
        global_median_arr.astype("float32"),
        dims=["lat", "lon"],
        coords={"lat": lat_coord, "lon": lon_coord},
        name="twoday_seasonal_global_median",
        attrs={"units": "mm", "season": season,
               "start_year": start_year, "end_year": end_year,
               "description": f"True global median of seasonal {season} 2-day rolling sums "
                               "across all members × season-days"},
    )
    stacked_medians = xr.DataArray(
        np.stack(per_member_medians, axis=0).astype("float32"),  # [n_members, lat, lon]
        dims=["member", "lat", "lon"],
        coords={"member": members, "lat": lat_coord, "lon": lon_coord},
        name="twoday_seasonal_per_member_median",
        attrs={"units": "mm", "season": season,
               "start_year": start_year, "end_year": end_year,
               "description": f"Per-member median of seasonal {season} 2-day rolling sums"},
    )

    global_median_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_seasonal_global_median": global_median}).to_netcdf(
        str(global_median_cache_path),
        encoding={"twoday_seasonal_global_median": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {global_median_cache_path.name}")

    per_member_medians_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_seasonal_per_member_median": stacked_medians}).to_netcdf(
        str(per_member_medians_cache_path),
        encoding={"twoday_seasonal_per_member_median": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {per_member_medians_cache_path.name}")

    return global_median, stacked_medians


def compute_cesm2_le_window_seasonal_per_member_p90_2d(
    season: str,
    start_year: int,
    end_year: int,
    model_dir: "Path",
    member_cache_path_fn,
    per_member_p90_cache_path: "Path",
    global_p90_cache_path: "Path",
    open_cache_fn,
    rolling_fn,
    subset_fn,
    window_days: int = 2,
    force_recompute: bool = False,
) -> "tuple[xr.DataArray, xr.DataArray]":
    """
    Seasonal per-member 90th-pctile fields and global 90th pctile (CESM2-LE).

    Only timesteps in `season`'s months are used. The global p90 pools all
    n_members × n_season_days values per pixel.

    Returns
    -------
    (global_p90 [lat, lon], per_member_p90 [n_members, lat, lon])
    """
    members = find_smile_members(model_dir, "cesm2_le")
    avail_s, avail_e = get_year_range_smile(model_dir, "cesm2_le")

    if (not force_recompute) and (per_member_p90_cache_path.exists()
                                   and global_p90_cache_path.exists()):
        print(f"  [cache] CESM2-LE seasonal {season} per-member p90 "
              f"← {per_member_p90_cache_path.name}")
        with xr.open_dataset(str(per_member_p90_cache_path)) as _ds:
            per_member_da = _ds["twoday_seasonal_per_member_p90"].load()
        with xr.open_dataset(str(global_p90_cache_path)) as _ds:
            global_da = _ds["twoday_seasonal_global_p90"].load()
        return global_da, per_member_da

    print(f"  Computing CESM2-LE seasonal {season} {len(members)}-member 2-day p90 "
          f"({start_year}–{end_year}) ...")

    per_member_season_arrs: list[np.ndarray] = []
    per_member_p90s:         list[np.ndarray] = []
    lat_coord = lon_coord = None

    for i, mid in enumerate(members):
        cache = member_cache_path_fn(mid, avail_s, avail_e)
        if not cache.exists():
            raise FileNotFoundError(f"Per-member cache missing: {cache}")
        da = open_cache_fn(cache, start_year - 1, end_year)
        da_2day = rolling_fn(da, window_days=window_days)
        da_sub = subset_fn(da_2day, start_year, end_year).load()
        mask = _season_time_mask(da_sub["time"].values, season)
        arr = da_sub.values[mask]
        per_member_season_arrs.append(arr)
        per_member_p90s.append(np.nanpercentile(arr, 90, axis=0))
        if lat_coord is None:
            lat_coord = da_sub["lat"]
            lon_coord = da_sub["lon"]
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(members)} members processed ...")

    all_values = np.concatenate(per_member_season_arrs, axis=0)
    global_p90_arr = np.nanpercentile(all_values, 90, axis=0)

    global_p90 = xr.DataArray(
        global_p90_arr.astype("float32"),
        dims=["lat", "lon"],
        coords={"lat": lat_coord, "lon": lon_coord},
        name="twoday_seasonal_global_p90",
        attrs={"units": "mm", "season": season,
               "start_year": start_year, "end_year": end_year},
    )
    stacked_p90 = xr.DataArray(
        np.stack(per_member_p90s, axis=0).astype("float32"),
        dims=["member", "lat", "lon"],
        coords={"member": members, "lat": lat_coord, "lon": lon_coord},
        name="twoday_seasonal_per_member_p90",
        attrs={"units": "mm", "season": season,
               "start_year": start_year, "end_year": end_year},
    )

    global_p90_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_seasonal_global_p90": global_p90}).to_netcdf(
        str(global_p90_cache_path),
        encoding={"twoday_seasonal_global_p90": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {global_p90_cache_path.name}")

    per_member_p90_cache_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"twoday_seasonal_per_member_p90": stacked_p90}).to_netcdf(
        str(per_member_p90_cache_path),
        encoding={"twoday_seasonal_per_member_p90": {"zlib": True, "complevel": 4}})
    print(f"  [saved] {per_member_p90_cache_path.name}")

    return global_p90, stacked_p90
