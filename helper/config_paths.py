# config_paths.py
"""
Path-building functions and directory basis
"""

from pathlib import Path

# ── Fixed base directories 
ERA5_RAW_DIR       = Path("/nird/datapeak/NS9873K/etdu/raw/era5/continuous-format/daily/europe/tp24/")
CATCHMENT_RAW_DIR  = Path("/nird/datalake/NS9873K/etdu/raw/nve/")
SENORGE_RAW_DIR    = Path("/nird/datapeak/NS9873K/DATA/senorge/rr/")   
ERA5_INTERPOLATED_BASE = Path("/nird/datalake/NS9873K/etdu/raw/era5/scandinavia")
ERA5_INTERPOLATED_DIR  = ERA5_INTERPOLATED_BASE / "tp"   # kept for backward-compat; prefer ERA5_INTERPOLATED_BASE / "tp" in new code
ERA5_INTERPOLATED_SWE_DIR = ERA5_INTERPOLATED_BASE / "sd" # ERA5 interpolated snow depth
ERA5_INTERPOLATED_SWVL_DIR = ERA5_INTERPOLATED_BASE / "swvl"   # ERA5 interpolated soil moisture
GEOJSON_DIR        = CATCHMENT_RAW_DIR     # adjust if GeoJSONs live elsewhere

FIGURES_DIR           = Path("/nird/datalake/NS9873K/lbal/figures/")
FIGURES_DIR_SECONDARY = Path("/nird/home/lbal/internship_storm_hans/figures/")
POSTPROC_DIR          = Path("/nird/datalake/NS9873K/lbal/postprocessed/")
WEIGHTS_DIR       = Path("/nird/datalake/NS9873K/lbal/postprocessed/weights")
OVERALL_PRECIP_EXTENT: tuple = (3.0, 16.0, 56.5, 66.0)  # (west, east, south, north)


# ────────────── Define Climatology Models ─────────────────────
# ── SMILE model raw data paths ────────────────────────────────
_SMILE_BASE         = Path("/nird/datalake/NS9873K/etdu/raw/smile")
CESM2_LE_BASE       = _SMILE_BASE / "cesm2_le"           / "scandinavia"
CESM2_LE_DIR        = CESM2_LE_BASE / "PRECT"            # CESM2-LE daily precipitation (variable PRECT, mm/day; identical values to the retired tp24 folder)
CESM2_LE_SWE_DIR    = CESM2_LE_BASE / "SWE"
CESM2_LE_SM_DIR     = CESM2_LE_BASE / "SM"
GFDL_SPEAR_DIR      = _SMILE_BASE / "gfdl_spear_med_le"  / "scandinavia" / "tp24"


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


# ── Compound-analysis catchment drammen + glomma
COMPOUND_CATCHMENTS: dict[str, str] = {
    "regine_drammen":        "Regine Drammen",
    "regine_glomma":         "Regine Glomma",
    "regine_drammen_glomma": "Regine Drammen + Glomma",}


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

def cesm2_catchment_field_path(window_days: int, catchment_slug: str,
                               variable: str, start_year: int, end_year: int) -> Path:
    """Catchment-averaged CESM2-LE compound-series cache — ONE file per
    (window, catchment, variable) holding ALL common members [member, time].
    variable ∈ {'precipitation', 'soil_moisture', 'snowmelt'}.

    Example: post_processed_cesm2_le_2day_regine_glomma_soil_moisture_1920-2034.nc
    """
    fname = (f"post_processed_cesm2_le_{acc_tag(window_days)}_{catchment_slug}_"
             f"{variable}_{start_year}-{end_year}.nc")
    return POSTPROC_DIR / "cesm2_le" / "catchment_averaged" / fname


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

def median_precip_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int) -> list:
    """Paths for the 2-panel N-day median precipitation figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_precip_{start_year}-{end_year}.pdf")

def median_precip_diff_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int) -> list:
    """Paths for the 3-panel N-day median vs ERA5-interpolated + difference figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_precip_diff_{start_year}-{end_year}.pdf")

def median_precip_diffonly_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int) -> list:
    """Paths for the single-panel N-day median difference figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_precip_diffonly_{start_year}-{end_year}.pdf")

def p90_precip_diff_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int) -> list:
    """Paths for the 3-panel 90th-percentile N-day median difference figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_90pctl_precip_diff_{start_year}-{end_year}.pdf")

def p90_precip_diffonly_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int) -> list:
    """Paths for the single-panel 90th-percentile N-day median difference figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_90pctl_precip_diffonly_{start_year}-{end_year}.pdf")

def median_precip_sig_diff_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int, pctl_tag: str) -> list:
    """Paths for significance-hatched 3-panel N-day median diff figure. pctl_tag e.g. '5_95pctl'."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_precip_{pctl_tag}_diff_{start_year}-{end_year}.pdf")

def p90_precip_sig_diff_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int, pctl_tag: str) -> list:
    """Paths for significance-hatched 3-panel 90th-pctile diff figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_90pctl_precip_{pctl_tag}_diff_{start_year}-{end_year}.pdf")

def median_precip_sig_diffonly_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int, pctl_tag: str) -> list:
    """Paths for significance-hatched single-panel N-day median diffonly figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_precip_{pctl_tag}_diffonly_{start_year}-{end_year}.pdf")

def p90_precip_sig_diffonly_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int, pctl_tag: str) -> list:
    """Paths for significance-hatched single-panel 90th-pctile diffonly figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_90pctl_precip_{pctl_tag}_diffonly_{start_year}-{end_year}.pdf")

def cesm2_annmedian_cache_path(start_year: int, end_year: int) -> Path:
    """Cache path for the CESM2-LE 100-member median of annual-total precipitation."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"annmedian_median_cesm2le_{start_year}-{end_year}.nc")

# Cesm2-le and Era5 Interpolated Paths
def cesm2_window_annmedian_cache_path(window_days: int, start_year: int, end_year: int) -> Path:
    """Cache path for the CESM2-LE 100-member median N-day median spatial field."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"annmedian_{acc_tag(window_days)}_median_cesm2le_{start_year}-{end_year}.nc")

def cesm2_window_p90_cache_path(window_days: int, start_year: int, end_year: int) -> Path:
    """Cache path for the CESM2-LE 100-member median N-day 90th-percentile spatial field."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"p90_{acc_tag(window_days)}_median_cesm2le_{start_year}-{end_year}.nc")

def cesm2_window_global_median_cache_path(window_days: int, start_year: int, end_year: int) -> Path:
    """Cache path for the CESM2-LE true global median of N-day rolling sums (all members×time)."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"cesm2_{acc_tag(window_days)}_global_median_cesm2le_{start_year}-{end_year}.nc")

def cesm2_window_per_member_medians_cache_path(window_days: int, start_year: int, end_year: int) -> Path:
    """Cache path for stacked per-member N-day medians [n_members, lat, lon]."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"cesm2_{acc_tag(window_days)}_per_member_medians_cesm2le_{start_year}-{end_year}.nc")

def cesm2_window_per_member_p90_cache_path(window_days: int, start_year: int, end_year: int) -> Path:
    """Cache path for stacked per-member N-day 90th-percentile fields [n_members, lat, lon]."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"cesm2_{acc_tag(window_days)}_per_member_p90_cesm2le_{start_year}-{end_year}.nc")


# ── Seasonal analysis constants ────────────────────────────────────────────────
# SEASONS_ORDER holds only the four standard seasons — it drives the 4×3
# seasonal figure and the diagnostic loops, both of which require exactly 4.
# Extra custom windows below are addressable by key but stay out of that list.
SEASONS_ORDER: list[str] = ["DJF", "MAM", "JJA", "SON"]

SEASON_LABELS: dict[str, str] = {
    "DJF":    "Winter (DJF)",
    "MAM":    "Spring (MAM)",
    "JJA":    "Summer (JJA)",
    "SON":    "Autumn (SON)",
    "MAMJ":   "Spring (MAMJ)",   # extended spring incl. June
}
SEASON_MONTHS: dict[str, list[int]] = {
    "DJF":    [12, 1, 2],
    "MAM":    [3, 4, 5],
    "JJA":    [6, 7, 8],
    "SON":    [9, 10, 11],
    "MAMJ":   [3, 4, 5, 6],
}


# ── Seasonal analysis cache paths ─────────────────────────────────────────────
def era5_interp_window_seasonal_median_cache_path(window_days: int, season: str, start_year: int, end_year: int) -> Path:
    """Cache for ERA5-interpolated seasonal N-day median [lat, lon]."""
    return (POSTPROC_DIR / "era5_interpolated" / "overall_precipitation" /
            f"{acc_tag(window_days)}_seasonal_{season}_median_era5interp_{start_year}-{end_year}.nc")

def era5_interp_window_seasonal_p90_cache_path(window_days: int, season: str, start_year: int, end_year: int) -> Path:
    """Cache for ERA5-interpolated seasonal N-day 90th-percentile [lat, lon]."""
    return (POSTPROC_DIR / "era5_interpolated" / "overall_precipitation" /
            f"{acc_tag(window_days)}_seasonal_{season}_p90_era5interp_{start_year}-{end_year}.nc")

def cesm2_window_seasonal_global_median_cache_path(window_days: int, season: str, start_year: int, end_year: int) -> Path:
    """Cache for CESM2-LE true seasonal global median [lat, lon]."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"cesm2_{acc_tag(window_days)}_seasonal_{season}_global_median_{start_year}-{end_year}.nc")

def cesm2_window_seasonal_per_member_medians_cache_path(window_days: int, season: str, start_year: int, end_year: int) -> Path:
    """Cache for CESM2-LE per-member seasonal N-day median [n_members, lat, lon]."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"cesm2_{acc_tag(window_days)}_seasonal_{season}_per_member_medians_{start_year}-{end_year}.nc")

def cesm2_window_seasonal_global_p90_cache_path(window_days: int, season: str, start_year: int, end_year: int) -> Path:
    """Cache for CESM2-LE true seasonal global 90th-percentile [lat, lon]."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"cesm2_{acc_tag(window_days)}_seasonal_{season}_global_p90_{start_year}-{end_year}.nc")

def cesm2_window_seasonal_per_member_p90_cache_path(window_days: int, season: str, start_year: int, end_year: int) -> Path:
    """Cache for CESM2-LE per-member seasonal N-day 90th-pctile [n_members, lat, lon]."""
    return (POSTPROC_DIR / "cesm2_le" / "overall_precipitation" /
            f"cesm2_{acc_tag(window_days)}_seasonal_{season}_per_member_p90_{start_year}-{end_year}.nc")


# ── Seasonal significance figure paths ────────────────────────────────────────
def median_seasonal_precip_sig_diff_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int, pctl_tag: str) -> list:
    """Paths for 4×3 seasonal significance-hatched N-day median diff figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_seasonal_precip_{pctl_tag}_diff_{start_year}-{end_year}.pdf")

def p90_seasonal_precip_sig_diff_paths(fig_subdir: str, window_days: int, start_year: int, end_year: int, pctl_tag: str) -> list:
    """Paths for 4×3 seasonal significance-hatched 90th-pctile diff figure."""
    return precip_map_figure_paths(
        fig_subdir, f"{acc_tag(window_days)}median_seasonal_90pctl_precip_{pctl_tag}_diff_{start_year}-{end_year}.pdf")


def p90_single_season_precip_sig_diff_paths(fig_subdir: str, window_days: int, season: str,
                                            start_year: int, end_year: int, pctl_tag: str) -> list:
    """Paths for the 3-panel single-season significance-hatched 90th-pctile diff figure.

    Same 1x3 layout as `p90_precip_sig_diff_paths`, but restricted to one season.
    `pctl_tag` must keep the '{lo}_{hi}pctl' form (e.g. '2_98pctl') so that
    `plot_window_interp_3panel` picks the same colorbar offset as the annual figure.
    """
    return precip_map_figure_paths(
        fig_subdir,
        f"{acc_tag(window_days)}median_{season}_90pctl_precip_{pctl_tag}_diff_{start_year}-{end_year}.pdf")


# Era5-Interpolated Annual Caches Paths
def era5_interp_window_median_cache_path(window_days: int, start_year: int, end_year: int) -> Path:
    """Cache path for the ERA5-interpolated N-day median spatial field."""
    return (POSTPROC_DIR / "era5_interpolated" / "overall_precipitation" /
            f"{acc_tag(window_days)}_median_era5interp_{start_year}-{end_year}.nc")

def era5_interp_window_p90_cache_path(window_days: int, start_year: int, end_year: int) -> Path:
    """Cache path for the ERA5-interpolated N-day 90th-percentile spatial field."""
    return (POSTPROC_DIR / "era5_interpolated" / "overall_precipitation" /
            f"{acc_tag(window_days)}_p90_era5interp_{start_year}-{end_year}.nc")


# ── SWE / soil-moisture cache paths (compound flood-risk analysis) ────────────
# Mirrors the precip cache layout, but parameterised by `kind` so two functions
# cover both variables and every cache type (instead of one function each).
#   model ∈ {"cesm2_le", "era5_interpolated"}
#   kind  ∈ {"swe", "soil_moisture"}   — postprocessed sub-folder

def field_daily_cache_path(model: str, kind: str, start_year: int, end_year: int,
                           member_id: str | None = None, window_days: int = 1) -> Path:
    """Per-member (CESM2-LE) or overall (ERA5-interp) DAILY field cache path —
    counterpart of overall_precip_member_path / overall_precip_path for SWE & SM.

    window_days=1 → the raw daily field (SWE / soil moisture), i.e. the existing
    `…_1day_…` caches. window_days>=2 → the pre-computed N-day snowmelt cache
    (SWE only, max(0, −ΔSWE): positive melt, gains→0) built by
    save_*_field_diff_overall in load_data_store_postprocessed.ipynb; the label in
    the filename tracks window_days (1day/2day/3day/4day)."""

    tag = acc_tag(window_days)
    if member_id is not None:
        fname = f"post_processed_{model}_{kind}_{tag}_member{member_id}_{start_year}-{end_year}.nc"
    else:
        fname = f"post_processed_{model}_{kind}_{tag}_{start_year}-{end_year}.nc"
    return POSTPROC_DIR / model / kind / fname


def field_window_cache_path(model: str, kind: str, which: str, window_days: int,
                            start_year: int, end_year: int, season: str = "") -> Path:
    """Derived N-day cache path (global/per-member median or p90; optional season).
    which (cesm2_le): global_median | per_member_medians | global_p90 | per_member_p90
    which (era5_interpolated): median | p90
    season: "" → annual, else one of DJF/MAM/JJA/SON."""
    seas = f"seasonal_{season}_" if season else ""
    if model == "cesm2_le":
        fname = f"cesm2_{acc_tag(window_days)}_{seas}{which}_{kind}_{start_year}-{end_year}.nc"
    else:
        fname = f"{acc_tag(window_days)}_{seas}{which}_era5interp_{kind}_{start_year}-{end_year}.nc"
    return POSTPROC_DIR / model / kind / fname


# ── Compound-extreme frequency-evolution output location + filenames ──────────
# Sub-folder of the compound-analysis output
COMPOUND_FREQ_FIG_SUBDIR: str = "compound_flood_risk_output/frequency_evolution"

def compound_freq_figure_paths(fname: str) -> list:
    """Both figure roots for one frequency-evolution output (PDF, CSV or JSON).

    Example
    -------
    compound_freq_figure_paths("internal_variability_trend_2day_….pdf")
        -> [FIGURES_DIR           / COMPOUND_FREQ_FIG_SUBDIR / "internal_variability_trend_2day_….pdf",
            FIGURES_DIR_SECONDARY / COMPOUND_FREQ_FIG_SUBDIR / "internal_variability_trend_2day_….pdf"]
    """
    return precip_map_figure_paths(COMPOUND_FREQ_FIG_SUBDIR, fname)


# ── Frequency-evolution filenames ─────────────────────────────────────────────
def compound_freq_stem(figtype: str, window_days: int, x_variable: str, y_variable: str,
                       catchment_slug: str, start_year: int, end_year: int,
                       roll_years: int, threshold: float, norm_ref: tuple,
                       season: str = "all", members="all") -> str:
    """Shared filename stem for the frequency-evolution figures and data outputs.

    figtype : 'internal_variability_trend' (Figure 1) | 'signal_to_noise' (Figure 2)

    Every selection that changes the result appears in the stem, so two runs can
    never overwrite each other. `norm_ref` is the FROZEN reference window that
    fixes max(x)/max(y) and therefore the position of the threshold line, so two
    runs differing only in the reference must not share a filename. The
    `_{season}` and `_members{spec}` segments are appended ONLY when they differ
    from the defaults, so the standard case stays clean. The threshold keeps its
    decimal point (0.8, 0.85, 1.0).

    Example
    -------
    signal_to_noise_2day_precipitation_snowmelt_regine_drammen_glomma_1920-2034_7year_thr0.75_ref1995-2024_MAMJ
    """
    thr = f"{float(threshold):.10g}"
    if "." not in thr:
        thr += ".0"
    stem = (f"{figtype}_{acc_tag(window_days)}_{x_variable}_{y_variable}_"
            f"{catchment_slug}_{start_year}-{end_year}_{roll_years}year_thr{thr}_"
            f"ref{int(norm_ref[0])}-{int(norm_ref[1])}")
    if str(season).lower() != "all":
        stem += f"_{season}"
    if not (isinstance(members, str) and members.strip().lower() == "all"):
        stem += "_members" + (members.strip().lower().replace(",", "-").replace(" ", "")
                              if isinstance(members, str)
                              else "-".join(f"{int(m):03d}" for m in members))
    return stem


# ── GeoJSON filenames (canonical definition — imported by notebooks) ───────────
GEOJSON_FILES: dict[str, str] = {
    "nevina_bergheim":  "catchment_nve_nevina_bergheim.geojson",
    "nevina_honnefoss": "catchment_nve_nevina_hønnefoss.geojson",
    "nevina_losna":     "catchment_nve_nevina_losna.geojson",
    "regine_drammen":   "catchment_nve_regine_drammen.geojson",
    "regine_glomma":    "catchment_nve_regine_glomma.geojson",
}

# ── Model display settings (moved here from plot_style.py to break circular import) ──
MODEL_COLORS: dict[str, str] = {
    "senorge":           "#D73027",
    "era5_0.25":         "#F28E2B",
    "era5_0.5":          "#E3B505",
    "gfdl_spear_med_le": "#4DAF4A",
    "cesm2_le":          "#2C7BB6",
}
MODEL_LABELS: dict[str, str] = {
    "senorge":           "SeNorge / 1 km",
    "era5_0.25":         "ERA5 / 0.25°",
    "era5_0.5":          "ERA5 / 0.5°",
    "gfdl_spear_med_le": "GFDL-SPEAR / 0.5° x 0.625°",
    "cesm2_le":          "CESM2-LE / 0.94° x 1.25°",
}
MODEL_ORDER: list[str] = [
    "senorge", "era5_0.25", "era5_0.5", "gfdl_spear_med_le", "cesm2_le"
]