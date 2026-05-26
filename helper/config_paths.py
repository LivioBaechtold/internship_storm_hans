# config_paths.py
"""
Path-building functions and directory basis
"""

from pathlib import Path

# ── Fixed base directories (never change these) ────────────────────────────────
ERA5_RAW_DIR       = Path("/nird/datapeak/NS9873K/etdu/raw/era5/continuous-format/daily/europe/tp24/")
CATCHMENT_RAW_DIR  = Path("/nird/datalake/NS9873K/etdu/raw/nve/")
SENORGE_RAW_DIR    = Path("/nird/datapeak/NS9873K/DATA/senorge/rr/")   # reserved for future use
ERA5_INTERPOLATED_DIR = Path("/nird/datalake/NS9873K/etdu/raw/era5/scandinavia/tp")
GEOJSON_DIR        = CATCHMENT_RAW_DIR     # adjust if GeoJSONs live elsewhere

FIGURES_DIR           = Path("/nird/datalake/NS9873K/lbal/figures/")
FIGURES_DIR_SECONDARY = Path("/nird/home/lbal/internship_storm_hans/figures/")
POSTPROC_DIR          = Path("/nird/datalake/NS9873K/lbal/postprocessed/")
WEIGHTS_DIR       = Path("/nird/datalake/NS9873K/lbal/postprocessed/weights")
OVERALL_PRECIP_EXTENT: tuple = (3.0, 16.0, 56.5, 66.0)  # (west, east, south, north)


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
        "figure_label":   "CESM2-LE / 0.942° × 1.25°",
        "tp24_unit_mode": "already_mm",
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
    """Full path for a postprocessed catchment-averaged NetCDF cache file."""
    return (postproc_dir(dataset) / "catchment_averaged" / postproc_filename(
        dataset, resolution, window_days, catchment_slug, start_year, end_year))

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
def smile_member_postproc_path(dataset, window_days, member_id, catchment_slug, start_year, end_year):
    fname = (f"post_processed_{dataset}_{acc_tag(window_days)}_{catchment_slug}_member{member_id}_{start_year}-{end_year}.nc")
    return POSTPROC_DIR / dataset / "catchment_averaged" / fname


def smile_yearmax_stats_path(dataset, window_days, catchment_slug, start_year, end_year):
    fname = (f"yearmax_stats_{dataset}_{acc_tag(window_days)}_{catchment_slug}_{start_year}-{end_year}.nc")
    return POSTPROC_DIR / dataset / "catchment_averaged" / fname

def smile_reference_tag(reference_dataset: str, reference_resolution: str) -> str:
    """
    Short, stable tag used in SMILE PDF filenames.

    Examples
    --------
    era5 + 0.5x0.5   -> era5_0.5
    era5 + 0.25x0.25 -> era5_0.25
    senorge + ""     -> senorge
    """
    if reference_dataset == "senorge":
        return "senorge"

    if reference_dataset == "era5":
        if reference_resolution == "0.5x0.5":
            return "era5_0.5"
        if reference_resolution == "0.25x0.25":
            return "era5_0.25"

    return res_tag(reference_dataset, reference_resolution)


def smile_figure_paths(
    dataset:              str,
    window_days:          int,
    catchment_slug:       str,
    start_year:           int,
    end_year:             int,
    fig_subdir:           str,
    reference_dataset:    str,
    reference_resolution: str,
) -> list:
    """
    Return PDF save paths for both figure roots (SMILE figures),
    with the actual reference dataset encoded in the filename.
    """
    ref_tag = smile_reference_tag(reference_dataset, reference_resolution)

    fname = (
        f"timeseries_returnperiod_hans_{dataset}_{acc_tag(window_days)}_"
        f"{catchment_slug}_{start_year}-{end_year}_ref_{ref_tag}.pdf"
    )
    return [
        FIGURES_DIR           / fig_subdir / fname,
        FIGURES_DIR_SECONDARY / fig_subdir / fname,
    ]

# ── Overall-precipitation (spatial 2-D) path builders ─────────────────────────
# ERA5 / SeNorge  : one file per dataset/resolution covering full available period.
# SMILE models    : one file per member covering full available period.

def overall_precip_path(dataset: str, resolution: str,
                        start_year: int, end_year: int) -> Path:
    """
    Cache path for the spatially-cropped overall daily precipitation field.
    All datasets (including SeNorge) now store daily values.

    Examples
    --------
    era5, 0.5x0.5,  1941, 2024 → overall_precipitation/post_processed_era5_0.5x0.5_1day_1941-2024.nc
    senorge, '',    1957, 2024 → overall_precipitation/post_processed_senorge_1day_1957-2024.nc
    """
    if dataset == "senorge":
        fname = f"post_processed_senorge_1day_{start_year}-{end_year}.nc"
    else:
        tag = res_tag(dataset, resolution)
        fname = f"post_processed_{tag}_1day_{start_year}-{end_year}.nc"
    return POSTPROC_DIR / dataset / "overall_precipitation" / fname


def overall_precip_member_path(dataset: str, member_id: str,
                                start_year: int, end_year: int) -> Path:
    """
    Cache path for one SMILE member's spatially-cropped daily precipitation field.

    Example
    -------
    cesm2_le, '001', 1920, 2034
        → overall_precipitation/post_processed_cesm2_le_1day_member001_1920-2034.nc
    """
    fname = f"post_processed_{dataset}_1day_member{member_id}_{start_year}-{end_year}.nc"
    return POSTPROC_DIR / dataset / "overall_precipitation" / fname

# ── Overview-map figure path builders ─────────────────────────────────────────

def precip_map_figure_paths(fig_subdir: str, fname: str) -> list:
    """
    Return [primary_path, secondary_path] for any overview precipitation map figure.

    Example
    -------
    precip_map_figure_paths("precip_maps_hans", "annualmean_precip_1995-2024.pdf")
    """
    return [
        FIGURES_DIR           / fig_subdir / fname,
        FIGURES_DIR_SECONDARY / fig_subdir / fname,
    ]


def annmedian_precip_paths(fig_subdir: str, start_year: int, end_year: int) -> list:
    """Paths for the 4-panel annual median precipitation figure."""
    return precip_map_figure_paths(
        fig_subdir, f"annualmedian_precip_{start_year}-{end_year}.pdf"
    )

def twodaymedian_precip_paths(fig_subdir: str, start_year: int, end_year: int) -> list:
    """Paths for the 2-panel 2-day median precipitation figure."""
    return precip_map_figure_paths(
        fig_subdir, f"2daymedian_precip_{start_year}-{end_year}.pdf"
    )

def twodaymedian_3panel_paths(fig_subdir: str, start_year: int, end_year: int) -> list:
    """Paths for the 3-panel 2-day median vs ERA5-interpolated + difference figure."""
    return precip_map_figure_paths(
        fig_subdir, f"2daymedian_3panel_interpera5_{start_year}-{end_year}.pdf"
    )

def twodaymedian_diff_paths(fig_subdir: str, start_year: int, end_year: int) -> list:
    """Paths for the single-panel 2-day median difference figure."""
    return precip_map_figure_paths(
        fig_subdir, f"2daymedian_diff_interpera5_{start_year}-{end_year}.pdf"
    )

def twodayp90_3panel_paths(fig_subdir: str, start_year: int, end_year: int) -> list:
    """Paths for the 3-panel 2-day 90th-percentile figure."""
    return precip_map_figure_paths(
        fig_subdir, f"2dayp90_3panel_interpera5_{start_year}-{end_year}.pdf"
    )

def twodayp90_diff_paths(fig_subdir: str, start_year: int, end_year: int) -> list:
    """Paths for the single-panel 2-day 90th-percentile difference figure."""
    return precip_map_figure_paths(
        fig_subdir, f"2dayp90_diff_interpera5_{start_year}-{end_year}.pdf"
    )

def cesm2_annmedian_cache_path(start_year: int, end_year: int) -> Path:
    """Cache path for the CESM2-LE 100-member median of the annual-total spatial field."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"annmedian_median_cesm2le_{start_year}-{end_year}.nc")

def cesm2_2day_annmedian_cache_path(start_year: int, end_year: int) -> Path:
    """Cache path for the CESM2-LE 100-member median 2-day median spatial field."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"annmedian_2day_median_cesm2le_{start_year}-{end_year}.nc")

def cesm2_2day_p90_cache_path(start_year: int, end_year: int) -> Path:
    """Cache path for the CESM2-LE 100-member median 2-day 90th-percentile spatial field."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"p90_2day_median_cesm2le_{start_year}-{end_year}.nc")