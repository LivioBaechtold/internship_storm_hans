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

FIGURES_DIR        = Path("/nird/datalake/NS9873K/lbal/figures/")
POSTPROC_DIR       = Path("/nird/datalake/NS9873K/lbal/postprocessed/")

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

def postproc_dir(dataset: str, resolution: str) -> Path:
    """Subdirectory for cached catchment time-series NetCDF files."""
    return POSTPROC_DIR / res_tag(dataset, resolution)

def catchment_nc_path(dataset: str, resolution: str, catchment_slug: str) -> Path:
    """Full path for a cached catchment time-series NetCDF file."""
    return postproc_dir(dataset, resolution) / (
        f"tp_catchment_{catchment_slug}_{res_tag(dataset, resolution)}.nc")

def figure_path(dataset: str, resolution: str, catchment_slug: str,
                start_year: int, end_year: int,
                window_days: int = 2,
                fig_subdir: str = "timeseries_return_hans") -> Path:
    """Full path for an output PDF, including accumulation window in filename."""
    fname = (
        f"timeseries_returnperiod_hans_{dataset}_{resolution}_"
        f"{acc_tag(window_days)}_{catchment_slug}_dailyprecip_{start_year}-{end_year}.pdf"
    )
    return FIGURES_DIR / fig_subdir / fname