# config_paths.py
"""
Path-building functions and fixed base directories.

IMPORTANT: This file defines *where things live*, not *which dataset to use*.
Dataset and resolution are passed in from the notebook at runtime.
All path-building functions take (dataset, resolution) as arguments.
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
WEIGHTS_025_DIR       = Path("/nird/datalake/NS9873K/lbal/postprocessed/0.25_weights")

# ── Catchment registry ─────────────────────────────────────────────────────────
# Keys   = slug used in filenames and caches
# Values = human-readable title used in figure titles
CATCHMENTS = {
    "nevina_bergheim":  "Nevina Bergheim",
    "nevina_honnefoss": "Nevina Hønnefoss",
    "nevina_losna":     "Nevina Losna",
    "regine_drammen":   "Regine Drammen",
    "regine_glomma":    "Regine Glomma",
}

# ── Storm Hans event settings ──────────────────────────────────────────────────
# These do not depend on dataset/resolution and stay here centrally.
HANS_DATE        = "2023-08-08"   # reference date for Storm Hans; not used for return-period event selection
HANS_SEARCH_YEAR = 2023           # the event is defined as the annual maximum within this year

# ── Path-building functions ────────────────────────────────────────────────────
# All functions below take dataset and resolution as arguments so the notebook
# controls which data is used without editing this file.

def res_tag(dataset: str, resolution: str) -> str:
    """
    Return a filesystem-safe tag for a dataset+resolution combination.
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
        f"{acc_tag(window_days)}_{catchment_slug}_{start_year}-{end_year}.nc"
    )

def catchment_postproc_path(dataset: str, resolution: str, window_days: int,
                             catchment_slug: str, start_year: int, end_year: int) -> Path:
    """Full path for a postprocessed grid-level NetCDF cache file."""
    return postproc_dir(dataset) / postproc_filename(
        dataset, resolution, window_days, catchment_slug, start_year, end_year
    )

def figure_filename(dataset: str, resolution: str, window_days: int,
                    catchment_slug: str, start_year: int, end_year: int) -> str:
    """PDF filename — no dailyprecip segment, no double underscores for Senorge."""
    return (
        f"timeseries_returnperiod_hans_{res_tag(dataset, resolution)}_"
        f"{acc_tag(window_days)}_{catchment_slug}_{start_year}-{end_year}.pdf"
    )

def figure_paths(dataset: str, resolution: str, window_days: int,
                 catchment_slug: str, start_year: int, end_year: int,
                 fig_subdir: str) -> list:
    """Return PDF save paths for both the primary and secondary figure roots."""
    fname = figure_filename(dataset, resolution, window_days,
                            catchment_slug, start_year, end_year)
    return [
        FIGURES_DIR           / fig_subdir / fname,
        FIGURES_DIR_SECONDARY / fig_subdir / fname,]
