# config_paths.py
"""
Path-building functions and directory basis
"""

from pathlib import Path

# ── Fixed base directories (never change these) ────────────────────────────────
ERA5_RAW_DIR       = Path("/nird/datapeak/NS9873K/etdu/raw/era5/continuous-format/europe/daily/tp24/")
CATCHMENT_RAW_DIR  = Path("/nird/datalake/NS9873K/etdu/raw/nve/")
SENORGE_RAW_DIR    = Path("/nird/datapeak/NS9873K/DATA/senorge/rr/")   # reserved for future use
GEOJSON_DIR        = CATCHMENT_RAW_DIR     # adjust if GeoJSONs live elsewhere

FIGURES_DIR           = Path("/nird/datalake/NS9873K/lbal/figures/")
FIGURES_DIR_SECONDARY = Path("/nird/home/lbal/internship_storm_hans/figures/")
POSTPROC_DIR          = Path("/nird/datalake/NS9873K/lbal/postprocessed/")
WEIGHTS_DIR       = Path("/nird/datalake/NS9873K/lbal/postprocessed/weights")


# ────────────── Define Climatology Models ─────────────────────
# ── SMILE model raw data paths ────────────────────────────────
_SMILE_BASE       = Path("/nird/datalake/NS9873K/etdu/raw/smile")
CESM2_LE_DIR      = _SMILE_BASE / "cesm2_le"            / "scandinavia" / "tp24"
GFDL_SPEAR_DIR    = _SMILE_BASE / "gfdl_spear_med_le"   / "scandinavia" / "tp24"

# ── SMILE analysis defaults ────────────────────────────────────────────────────
# default_start / default_end : 40-year present-climate window centred near 2023.
# ref_resolution              : ERA5 resolution whose catchment values are used
#                               as Storm Hans benchmarks (both models ≈ 1°,
#                               closest to ERA5 0.5°).
SMILE_CONFIG: dict[str, dict] = {
    "cesm2_le": {
        "model_dir":      CESM2_LE_DIR,
        "n_members":      100,
        "member_digits":  3,
        "default_start":  1995,
        "default_end":    2034,
        "ref_dataset":    "era5",
        "ref_resolution": "0.5x0.5",
        "description":    "CESM2 Large Ensemble",
        "figure_label":   "CESM2-LE/1°x1°",
        "tp24_unit_mode": "auto",
    },
    "gfdl_spear_med_le": {
        "model_dir":      GFDL_SPEAR_DIR,
        "n_members":      30,
        "member_digits":  2,
        "default_start":  2001,
        "default_end":    2040,
        "ref_dataset":    "era5",
        "ref_resolution": "0.5x0.5",
        "description":    "GFDL-SPEAR-MED Large Ensemble",
        "figure_label":   "GFDL SPEAR-LE/1°x1°",
        "tp24_unit_mode": "auto",},}

# ── Catchment registry ─────────────────────────────────────────────────────────
# Keys   = slug used in filenames and caches
# Values = human-readable title used in figure titles
CATCHMENTS = {
    "nevina_bergheim":  "Nevina Bergheim",
    "nevina_honnefoss": "Nevina Hønnefoss",
    "nevina_losna":     "Nevina Losna",
    "regine_drammen":   "Regine Drammen",
    "regine_glomma":    "Regine Glomma",}

# ── Storm Hans event settings ──────────────────────────────────────────────────
# These do not depend on dataset/resolution and stay here centrally.
HANS_DATE        = "2023-08-08"   # reference date for Storm Hans; not used for return-period event selection
HANS_SEARCH_YEAR = 2023           # the event is defined as the annual maximum within this year

# ── Path-building functions ────────────────────────────────────────────────────
# All functions below take dataset and resolution as arguments so the notebook
# controls which data is used without editing this file.

def res_tag(dataset: str, resolution: str) -> str:
    """
    Return a filesystem-safe tag for a dataset+resolution combination
    Examples:
        era5, 0.5x0.5  → 'era5_0.5x0.5'
        senorge, ''    → 'senorge'
    """
    if resolution:
        return f"{dataset}_{resolution}"
    return dataset

def acc_tag(window_days: int) -> str:
    """Return a filename-safe accumulation label, e.g. '1day' or '2day'."""
    return f"{window_days}day"

def postproc_dir(dataset: str) -> Path:
    """Dataset-level subdirectory for postprocessed grid NetCDF cache files."""
    return POSTPROC_DIR / dataset

def postproc_filename(dataset: str, resolution: str, window_days: int,
                      catchment_slug: str, start_year: int, end_year: int) -> str:
    """Filename for a postprocessed grid-level NetCDF cache file."""
    return (
        f"post_processed_{res_tag(dataset, resolution)}_"
        f"{acc_tag(window_days)}_{catchment_slug}_{start_year}-{end_year}.nc")

def catchment_postproc_path(dataset: str, resolution: str, window_days: int,
                             catchment_slug: str, start_year: int, end_year: int) -> Path:
    """Full path for a postprocessed grid-level NetCDF cache file."""
    return postproc_dir(dataset) / postproc_filename(
        dataset, resolution, window_days, catchment_slug, start_year, end_year)

def figure_filename(dataset: str, resolution: str, window_days: int,
                    catchment_slug: str, start_year: int, end_year: int) -> str:
    """PDF filename — no dailyprecip segment, no double underscores for Senorge."""
    return (
        f"timeseries_returnperiod_hans_{res_tag(dataset, resolution)}_"
        f"{acc_tag(window_days)}_{catchment_slug}_{start_year}-{end_year}.pdf")

def figure_paths(dataset: str, resolution: str, window_days: int,
                 catchment_slug: str, start_year: int, end_year: int,
                 fig_subdir: str) -> list:
    """Return PDF save paths for both the primary and secondary figure roots."""
    fname = figure_filename(dataset, resolution, window_days,
                            catchment_slug, start_year, end_year)
    return [
        FIGURES_DIR           / fig_subdir / fname,
        FIGURES_DIR_SECONDARY / fig_subdir / fname,]


# Define SMILE-specific path builders ───────────────────────────────────────────────
def smile_member_postproc_path(
    dataset:        str,
    window_days:    int,
    member_id:      str,
    catchment_slug: str,
    start_year:     int,
    end_year:       int,
) -> Path:
    """Cache path for one ensemble member's 1-D catchment-mean time series."""
    fname = (
        f"post_processed_{dataset}_{acc_tag(window_days)}_"
        f"{catchment_slug}_member{member_id}_{start_year}-{end_year}.nc"
    )
    return POSTPROC_DIR / dataset / f"{start_year}-{end_year}" / fname


def smile_yearmax_stats_path(
    dataset:        str,
    window_days:    int,
    catchment_slug: str,
    start_year:     int,
    end_year:       int,
) -> Path:
    """
    Cache path for SMILE yearly-max statistics used by the single-panel
    return-period plots.
    """
    fname = (
        f"yearmax_stats_{dataset}_{acc_tag(window_days)}_"
        f"{catchment_slug}_{start_year}-{end_year}.nc"
    )
    return POSTPROC_DIR / dataset / f"{start_year}-{end_year}" / fname


def smile_figure_paths(
    dataset:        str,
    window_days:    int,
    catchment_slug: str,
    start_year:     int,
    end_year:       int,
    fig_subdir:     str,
    reference_mode: str,
) -> list:
    """
    Return PDF save paths for both figure roots (SMILE figures).

    reference_mode:
        "ref_precip"       -> ERA5 Hans precipitation as reference
        "ref_returnperiod" -> ERA5 Hans return period as reference
    """
    fname = (
        f"timeseries_returnperiod_hans_{dataset}_{acc_tag(window_days)}_"
        f"{catchment_slug}_{start_year}-{end_year}_{reference_mode}.pdf"
    )
    return [
        FIGURES_DIR           / fig_subdir / fname,
        FIGURES_DIR_SECONDARY / fig_subdir / fname,
    ]