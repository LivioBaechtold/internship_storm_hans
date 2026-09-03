# Compound Flood Risk and Precipitation Extremes in Southern Norway

This repository contains the code and the supporting documentation of my internship project on
precipitation extremes and compound flood drivers in southern Norway, using Storm Hans
(August 2023) as the reference event.

The goal of the project is twofold. First, to put the Storm Hans precipitation into a
statistical context by estimating its return period in reanalysis data and in two single-model
initial-condition large ensembles (SMILEs). Second, to move from precipitation alone to the
compound perspective — the joint occurrence of heavy precipitation with snowmelt or with wet
soils — and to ask how the frequency of such compound situations evolves over 1920–2034 in the
CESM2 Large Ensemble, and when that signal emerges from internal variability.

The repository is organised so that it can be seen how the analysis was done and, given access
to the ERA5 / seNorge / SMILE datasets, reproduce every figure of the report.

---

## 1. Analysis concept & naming conventions

### 1.1 The two strands of the analysis

The project consists of two strands that share the same catchment-averaging machinery:

- **Univariate precipitation extremes** — catchment-averaged precipitation time series,
  annual maxima, GEV fits and return periods, plus a comparison of the ensembles against the
  reanalyses (Q-Q plots, distribution plots, percentile tables) and against gridded
  climatology maps.

- **Compound extremes** — two catchment-averaged quantities in the same N-day window
  (precipitation with snowmelt, or precipitation with soil moisture), their joint
  distribution, an absolute compound threshold, and the evolution of the exceedance frequency
  in centred rolling windows across the 90 CESM2-LE members.

### 1.2 Accumulation windows

Every quantity is defined over an **N-day window**, set once per notebook by `WINDOW_DAYS`.
The window operator depends on the variable:

- **Precipitation** — rolling **sum** over the window (`rolling_accumulation`).
- **Snowmelt** — the SWE **decrease** over the window,
  `max(0, −(SWE(t) − SWE(t−(N−1))))` (`rolling_melt`). Only decreases count, and they are
  stored as a positive melt magnitude, so SWE gains map to zero. This keeps the 90th-percentile
  analysis meaningful, because high-melt days then sit at the top of the distribution.
- **Soil moisture** — rolling **mean** over the window (`rolling_mean`).

This is reflected in the file names: `1day`, `2day`, `3day`, … produced by `cfg.acc_tag`.
Most of the analysis was run with `WINDOW_DAYS = 2`.

### 1.3 Seasons

A season is always a **named key** of `cfg.SEASON_MONTHS`, never an ad-hoc month range. Next to
the four standard seasons `DJF`, `MAM`, `JJA`, `SON` there is the custom four-month spring
window **`MAMJ`** (March–June), which is the season used for the compound snowmelt analysis,
because the Norwegian melt season extends well into June.

### 1.4 Catchments

Five NVE catchments in southern Norway are used, referred to by their slug throughout the code
and the filenames:

| Slug | Title | Source |
|---|---|---|
| `nevina_bergheim` | Nevina Bergheim | NVE NEVINA |
| `nevina_honnefoss` | Nevina Hønnefoss | NVE NEVINA |
| `nevina_losna` | Nevina Losna | NVE NEVINA |
| `regine_drammen` | Regine Drammen | NVE REGINE |
| `regine_glomma` | Regine Glomma | NVE REGINE |

The compound analysis additionally uses the **dissolved union** `regine_drammen_glomma`
(Drammen ∪ Glomma). Its two GeoJSONs are merged with `unary_union` into one polygon *before*
the per-cell area fractions are computed, so the shared internal border is never double-counted
and cells fully inside the union get weight 1.

### 1.5 Filenames

Figures and caches carry their full selection in the name, so two runs cannot overwrite each
other. The general shape is

```
{quantity}_{window}_{variables}_{catchment}_{start}-{end}[_{season}][_thr{value}].pdf
```

For example:

- `2daymedian_90pctl_snowmelt_2_98pctl_diff_1995-2024.pdf`
- `joint_distribution_2day_precipitation_snowmelt_regine_drammen_glomma_1995-2024_MAMJ_thr0.7.pdf`
- `internal_variability_trend_2day_precipitation_snowmelt_regine_drammen_glomma_1920-2034_10year_thr0.9_ref1995-2024_MAMJ.pdf`

In the last case `_ref1995-2024` is the **frozen reference window** that fixes the position of
the compound threshold line, so two runs that differ only in the reference must not share a
filename.

---

## 2. Repository layout

The repository holds two things: the importable Python modules in `helper/` and the
orchestration notebooks in `code/`. All reusable logic lives in `helper/`; the notebooks only
set parameters and call into it.

### 2.1 `helper/`

All importable Python modules. The separation of responsibilities is strict: paths live only in
`config_paths.py`, figure code lives only in `plot_style.py`, and statistics live in
`return_period.py` and `catchment_tools.py`.

#### 2.1.1 Configuration

- **`config_paths.py`**
  Paths and constants only — no plotting, no data loading, no statistics. Holds the raw-data
  and output directories, the catchment registries (`CATCHMENTS`, `COMPOUND_CATCHMENTS`,
  `GEOJSON_FILES`), the ensemble metadata (`SMILE_CONFIG`), the season definitions
  (`SEASON_MONTHS`, `SEASON_LABELS`, `SEASONS_ORDER`), the model colours and labels
  (`MODEL_COLORS`, `MODEL_LABELS`, `MODEL_ORDER`), and every path builder in the project.
  The model colours live here rather than in `plot_style.py` to break a circular import with
  `catchment_tools.py`.
  Central helpers: `res_tag`, `acc_tag`, `catchment_postproc_path`, `overall_precip_path`,
  `field_daily_cache_path`, `field_window_cache_path`, `compound_freq_stem` and the
  `*_paths` functions that return both figure roots.

#### 2.1.2 Data access

Three modules with the same shape: file discovery, lazy loading with unit conversion, and
spatial-cache builders that write one cropped daily NetCDF per dataset (or per ensemble
member).

- **`data_era5.py`**
  ERA5 discovery (`find_era5_files`), loading and metre→mm conversion
  (`load_era5_precipitation`), day-level time selection (`select_time_range_by_day`,
  `pick_year_file`, `coord_slice`) and the cache builders `save_era5_overall` and
  `save_era5_interpolated_overall`. It also holds the ERA5-interpolated SWE and soil-moisture
  builders (`save_era5_interpolated_field_overall`,
  `save_era5_interpolated_field_diff_overall`) and the map computations
  `compute_era5_annual_median_2d`, `compute_era5_window_median_2d`,
  `compute_era5_interpolated_window_median_2d` / `_p90_2d` and their seasonal counterparts.

- **`data_senorge.py`**
  The same structure for seNorge, plus the coordinate handling of the native 1 km UTM-33 grid
  (`latlon_extent_to_utm_bbox`). The fill value −999.99 is masked in
  `load_senorge_precipitation`. Cache builder `save_senorge_overall`; map computations
  `compute_senorge_annual_median_2d`, `compute_senorge_2day_median_2d`.

- **`data_smile.py`**
  Ensemble member discovery (`find_smile_members`, `find_smile_files_for_member`), loading with
  automatic unit detection (`load_smile_precipitation`, `_convert_smile_tp24_to_mm`) and the
  per-member cache builders `save_smile_overall`, `save_cesm2_le_field_overall` and
  `save_cesm2_le_field_diff_overall`. It also holds the ensemble map computations
  (`compute_cesm2_le_window_global_median_2d`, `compute_cesm2_le_window_per_member_p90_2d`
  and the seasonal versions) and the significance test `compute_significance_masks`.

#### 2.1.3 Analysis

- **`catchment_tools.py`**
  The core of the project. It covers the catchment averaging (`find_weight_file`,
  `load_weights`, `align_weights_to_precip`, `crop_to_weight_bbox`, `compute_catchment_mean`),
  the cache I/O (`save_spatial_netcdf`, `open_precip_cache`, `open_field_cache`), the window
  operators (`rolling_accumulation`, `rolling_melt`, `rolling_mean`, `rolling_identity`,
  `rolling_change`), the CESM2-LE compound-series builder
  (`save_cesm2_le_catchment_field_series`, `common_cesm2_le_members`,
  `load_cesm2_le_catchment_field_series`), the season handling (`resolve_season_months`,
  `subset_season`, `season_tag`, `season_label`), the compound threshold statistics
  (`compound_threshold_stats`), the full frequency-evolution pipeline
  (`validate_frequency_evolution_config`, `load_compound_pair`,
  `freeze_normalisation_maxima`, `annual_exceedance_counts`, `rolling_window_counts`,
  `ensemble_frequency_statistics`, `grouped_percentile`,
  `run_compound_frequency_evolution`), and the two high-level return-period orchestrators
  `run_all` (reanalysis) and `run_all_smile` (ensembles).

- **`return_period.py`**
  Pure statistics, no I/O: `get_annual_maxima`, `weibull_plotting_positions`, `fit_gev`,
  `gev_return_level`, `get_event_annual_max`, `estimate_return_period`.

#### 2.1.4 Plotting

- **`plot_style.py`**
  Every Matplotlib and Cartopy figure of the project. No data loading and no statistics happen
  here. It holds the projection and colormap constants (`MAP_PROJ`, `PRECIP_CMAP`,
  `PRECIP_DIV_CMAP`, `DOY_CMAP`, `WEIGHT_CMAP`) and the figure functions:
  `make_figure` and `make_smile_return_period_figure` (return periods),
  `make_distribution_figure` and `make_qq_figure` (evaluation),
  `plot_precip_map` and `plot_single_catchment_weight_map` (event and weight maps),
  `plot_annual_median_4panel`, `plot_window_median_2panel`, `plot_window_interp_3panel`,
  `plot_window_interp_diffonly`, `plot_window_interp_diffonly_sig` and
  `plot_window_interp_seasonal_4row_3col` (climatology comparison maps),
  and for the compound part `make_joint_distribution_figure`,
  `plot_internal_variability_trend` and `plot_signal_to_noise_ratio`.

#### 2.1.5 Run-once scripts and tests

- **`generate_weights.py`**
  Generates the catchment area-fraction weight NetCDFs, one per catchment × dataset. Entry
  points `run_era5_025`, `run_gfdl_spear`, `run_cesm2_le` (also runnable from the command line
  via `--dataset`), built on `build_weights`, `save_weight_nc` and `_run_weight_loop`.
  `COMBINED_CATCHMENTS` defines the union catchments, whose GeoJSONs are dissolved with
  `_dissolve_geojson_union` before the fractions are computed. Existing weight files are
  skipped, so it is safe to re-run.

- **`test_grouped_percentile.py`**
  Reference unit test for `catchment_tools.grouped_percentile`, locked against a real
  2002–2011 CESM2-LE window (90 members, L = 10). Run with
  `python helper/test_grouped_percentile.py`. It checks the reference percentile values, that
  the function does not silently fall back to `np.percentile`, that empty bins are skipped,
  that the half-width zero bin keeps low percentiles non-negative, and that the ordering never
  inverts.

- **`prec_seq.txt`, `prec_div.txt`**
  The IPCC sequential and diverging precipitation colormaps as 256-row RGB tables, loaded at
  import time by `plot_style.py`.

### 2.2 `code/`

The notebooks are orchestration only: imports, a parameter block at the top, and calls into
`helper/`. No reusable function is defined in a notebook.

#### 2.2.1 Data preparation

- **`load_data_store_postprocessed.ipynb`**
  The single entry point that builds **all** postprocessed caches consumed by the other
  notebooks. It produces no figures. Run it once, and again whenever the raw data or the grids
  change; every step skips what already exists unless the corresponding `FORCE_*` flag is set.
  It builds, in this order: the catchment weights, the daily precipitation caches for ERA5
  (0.5° and 0.25°), seNorge and both SMILEs, the ERA5-interpolated precipitation cache, the
  daily SWE and soil-moisture caches for CESM2-LE and ERA5-interpolated, the N-day snowmelt
  caches, and finally the CESM2-LE catchment-averaged compound series.

#### 2.2.2 Return-period analysis

- **`analysis_return_hans.ipynb`**
  Return-period analysis of Storm Hans. The first half runs the reanalyses: pick a dataset with
  `DATASET_KEY` (`era5_0.5`, `era5_0.25`, `senorge`) and a window with `WINDOW_DAYS`, verify
  every path and cache with the configuration-check cell, then call `run_all`, which loops the
  five catchments and produces the two-panel time-series-plus-return-period figure for each.
  The second half runs the ensembles: `SMILE_RUN_TABLE` lists the
  (dataset, window, period) combinations, and `run_all_smile` loops the members, pools their
  annual maxima and produces the ensemble return-period figures.

#### 2.2.3 Model evaluation

- **`climate_model_evaluation.ipynb`**
  Evaluates CESM2-LE and GFDL-SPEAR against ERA5 (both resolutions) and seNorge over
  1985–2024, per catchment and for the 1-day and 2-day windows. It loads the catchment series
  with `load_annual_maxima_per_catchment` and `load_daily_values_per_catchment` and produces
  distribution figures (`make_distribution_figure`), Q-Q plots (`make_qq_figure`), percentile
  mapping tables (`build_percentile_mapping_table`) and distribution summary tables
  (`build_distribution_summary_table`). The tables are written as CSV next to the figures.

#### 2.2.4 Precipitation maps

- **`create_precip_maps_hans.ipynb`**
  All spatial precipitation figures. Its first part produces the **Storm Hans event maps**:
  `load_event_field` reads the year file of the selected dataset, crops it, applies the rolling
  window and takes the pixel-wise maximum over the 7–9 August 2023 envelope, and
  `plot_precip_map` draws the six dataset × window combinations. The second part produces the
  **catchment weight maps** for every dataset × catchment pair. The third and largest part is
  the **climatology comparison** over 1995–2024: the four-panel annual median map, the two-panel
  N-day median (CESM2-LE vs ERA5 0.5°), the three-panel and difference-only N-day median and
  90th-percentile comparisons against ERA5 interpolated onto the CESM2-LE grid, the same
  comparisons with per-pixel significance hatching at the 5/95 and 2/98 percentile levels, the
  4×3 seasonal version of those, and a single-season (`MAMJ`) figure in the annual layout.
  Several diagnostic cells print per-pixel tables used to calibrate the colorbar ranges; they
  produce no PDF.

#### 2.2.5 Compound analysis

- **`compound_flood_risk_analysis.ipynb`**
  The compound part, in two halves.

  The **first half** repeats the map methodology of `create_precip_maps_hans.ipynb` for
  snowmelt and soil moisture instead of precipitation: N-day median, 90th percentile,
  significance-hatched differences at 5/95 and 2/98, the seasonal 4×3 breakdown and the
  single-season figure. Units are kg/m². The per-variable configuration lives in one
  `VARIABLES` dictionary that supplies the source directory, the window operator, the cache
  opener and all labels, so the same helper functions serve both variables unchanged. The
  annual-median and two-panel overview figures are left out, because they would need seNorge
  and native-grid ERA5, which do not exist for these variables.

  The **second half** is the compound analysis proper: the joint distribution of two
  catchment-averaged window quantities (plain, and with the absolute threshold line), the
  rolling-window evolution of the compound exceedance frequency with its internal-variability
  band, and the signal-to-noise ratio of that frequency.

### 2.3 `figures/`

Generated figures, one sub-folder per notebook:

- `timeseries_return_hans/` — return-period and time-series figures.
- `climate_model_evaluation/` — distribution and Q-Q figures plus the CSV tables.
- `precip_maps_hans/` — Storm Hans event maps, weight maps and precipitation climatology maps.
- `compound_flood_risk_output/` — snowmelt and soil-moisture maps and the joint distributions.
- `compound_flood_risk_output/frequency_evolution/` — the rolling-window frequency and
  signal-to-noise figures and their CSV/JSON outputs.

Every figure function writes to **two** roots: the repository folder above and a mirror under
`FIGURES_DIR` on the data lake. A few of the largest seasonal PDFs are excluded from the
repository in `.gitignore`.

### 2.4 Large external data (not in this repo)

The raw datasets and the postprocessed caches live outside the repository and are not tracked:

```
/nird/datapeak/NS9873K/etdu/raw/era5/…/tp24/     ERA5 daily precipitation (0.5°, 0.25°)
/nird/datapeak/NS9873K/DATA/senorge/rr/          seNorge daily precipitation (1 km)
/nird/datalake/NS9873K/etdu/raw/era5/scandinavia/
    tp/ sd/ swvl/                                ERA5 regridded to the CESM2-LE grid
/nird/datalake/NS9873K/etdu/raw/smile/
    cesm2_le/scandinavia/PRECT/ SWE/ SM/         CESM2-LE (100 / 90 / 90 members)
    gfdl_spear_med_le/scandinavia/tp24/          GFDL-SPEAR-MED-LE (30 members)
/nird/datalake/NS9873K/etdu/raw/nve/             catchment GeoJSONs
/nird/datalake/NS9873K/lbal/postprocessed/       all caches built by the preparation notebook
/nird/datalake/NS9873K/lbal/figures/             mirror of figures/
```

The postprocessed tree is organised as:

```
postprocessed/
├── era5/                    overall_precipitation/  catchment_averaged/
├── senorge/                 overall_precipitation/  catchment_averaged/
├── era5_interpolated/       overall_precipitation/  swe/  soil_moisture/
├── cesm2_le/                overall_precipitation/  catchment_averaged/
│                            swe/  soil_moisture/
├── gfdl_spear_med_le/       overall_precipitation/  catchment_averaged/
├── weights/                 catchment weight NetCDFs
└── old_gold/                archived earlier versions of the caches
```

`overall_precipitation/` holds the cropped daily spatial caches (one file per dataset, or one
per ensemble member), `swe/` and `soil_moisture/` the daily state-variable caches and the
derived N-day snowmelt caches, and `catchment_averaged/` the catchment time series — including
the `[member, time]` CESM2-LE compound series under
`cesm2_le/catchment_averaged/`.

---

## 3. Datasets

| Key | Dataset | Grid | Variable | Raw unit | Members |
|---|---|---|---|---|---|
| `era5_0.5` | ERA5 | 0.5° lat/lon | `tp24` | m | — |
| `era5_0.25` | ERA5 | 0.25° lat/lon | `tp24` | m | — |
| `senorge` | seNorge | 1 km UTM-33 | `rr` | mm | — |
| `era5_interpolated` | ERA5 on the CESM2-LE grid | ~1° | `tp`, `sd`, `swvl` | m, kg/m² | — |
| `cesm2_le` | CESM2 Large Ensemble | 0.94° × 1.25° | `PRECT`, `SWE`, `SM` | mm/day, kg/m² | 100 / 90 / 90 |
| `gfdl_spear_med_le` | GFDL-SPEAR-MED-LE | 0.5° × 0.625° | `tp24` | m or mm | 30 |

ERA5 is converted from metres to mm inside `data_era5.py`; seNorge is already in mm and its
fill value −999.99 is masked; the SMILE unit is auto-detected from the metadata unless
`SMILE_CONFIG[...]["tp24_unit_mode"]` overrides it. All spatial caches are cropped to
`OVERALL_PRECIP_EXTENT = (3.0°E, 16.0°E, 56.5°N, 66.0°N)`.

Only **90 of the 100** CESM2-LE members carry SWE and soil-moisture output — the odd members
001–019 are missing — so every compound quantity is computed on the intersection returned by
`common_cesm2_le_members()`, and the ensemble size is detected automatically rather than
assumed.

Analysis periods: **1985–2024** for the model evaluation, **1995–2024** for the climatology
maps and for the frozen compound reference, and the full CESM2-LE record **1920–2034** for the
frequency evolution.

---

## 4. Methods

### 4.1 Catchment averaging

For every catchment × dataset pair, `generate_weights.py` computes the **area fraction** of
each grid cell that falls inside the catchment polygon and stores it as a weight NetCDF. The
polygons are projected to EPSG:25833 before the intersection, so the fractions are computed on
an equal-area footing rather than in degrees.

A catchment series is then obtained by aligning the weight field onto the data grid
(`align_weights_to_precip`), cropping the data to the bounding box of the non-zero weights
(`crop_to_weight_bbox`, so no full-domain array is ever loaded), taking the weighted spatial
mean (`compute_catchment_mean`), and finally applying the N-day window operator. Doing the
spatial average *before* the window operator means the N-day quantity describes the catchment
as a whole, not a single grid cell.

### 4.2 Return periods

Annual maxima are extracted from the catchment series (`get_annual_maxima`), a GEV
distribution is fitted by maximum likelihood (`fit_gev`), and return levels are obtained from
its quantile function (`gev_return_level`). The empirical positions are added with the Weibull
formula (`weibull_plotting_positions`). The Storm Hans return period is estimated as
`T = 1 / (1 − CDF(x))` (`estimate_return_period`), where the event value is the annual maximum
of `HANS_SEARCH_YEAR = 2023` rather than the value on a fixed date, so the event is captured
even when the catchment maximum falls a day off the nominal date.

For the ensembles the same fit is applied to the **pooled** annual maxima of all members
(`pool_member_annual_maxima`), which extends the effective sample from a few decades to a few
thousand member-years and stabilises the tail.

### 4.3 Model evaluation

The ensembles are compared against the reanalyses at the catchment level, separately for annual
maxima and for daily values, and for the 1-day and 2-day windows. Three views are produced:
kernel-density plus boxplot per model (`make_distribution_figure`), quantile-quantile plots of
each ensemble against each reanalysis (`make_qq_figure`), and two tables — a percentile mapping
that reads off the ensemble value at the reanalysis percentiles
(`build_percentile_mapping_table`) and a summary of mean, standard deviation and quantiles
(`build_distribution_summary_table`).

### 4.4 Climatology comparison maps and significance

CESM2-LE is compared against ERA5 regridded onto the CESM2-LE grid, so both fields sit on the
same cells and the difference is meaningful pixel by pixel. Two statistics are mapped: the
**N-day median** and the **N-day 90th percentile**, each as an ensemble field and as a
percentage difference against ERA5.

The ensemble median is the **true pooled median** over all members × time steps
(`compute_cesm2_le_window_global_median_2d`), not a median of per-member medians. Alongside it
the per-member fields are stored, because they are what the significance test needs.

Significance is assessed with a **percentile-rank test** without FDR correction
(`compute_significance_masks`): a pixel is flagged when ERA5 falls below the lower percentile
(or above the upper percentile) of the 90-or-100-member distribution at that pixel. Two levels
are used, 5/95 and 2/98, which correspond to two-sided p = 0.10 and p = 0.04. The mask is drawn
as `//` where CESM2-LE is higher and `\\` where ERA5 is higher, rendered as individual cell
rectangles (`_add_pixel_hatch_overlay`) to avoid the half-pixel artefacts that `contourf`-based
hatching produces on a coarse grid.

### 4.5 Compound severity and joint distribution

For a catchment, a window length and an ordered pair of variables, every (member, date) pair
gives one point in the joint distribution (`make_joint_distribution_figure`). Over the full
year the points are coloured by day of year with a circular month wheel; when the record is cut
to a single season the colour scale would only re-display the selection that has already been
made, so all points are drawn in one neutral grey and the wheel is dropped.

The compound criterion is an **additive normalised severity**

```
s = x / max(x) + y / max(y)  ≥  threshold
```

(`compound_threshold_stats`), which is a straight line in the (x, y) plane running from
`(0, threshold·max(y))` to `(threshold·max(x), 0)`. Both variables are normalised by their own
sample maximum, so a value of `s` says how far a day is from the joint corner of the
distribution, independently of the physical units of the two variables.

### 4.6 Frequency evolution and signal-to-noise

The frequency analysis (`run_compound_frequency_evolution`) asks how often the criterion of
Section 4.5 is met, and how that changes over 1920–2034.

The decisive methodological point is that `max(x)` and `max(y)` are **frozen**
(`freeze_normalisation_maxima`): they are the sample maxima of a fixed reference window
(`FE_NORM_REF`, by default 1995–2024), restricted to the reference member pool and to the
selected season. The criterion is therefore one fixed line in the (x, y) plane for every
rolling window, and a drifting denominator cannot masquerade as a trend.

The exceedances are then counted per member and calendar year
(`annual_exceedance_counts`), summed over **centred rolling windows** of `FE_ROLL_YEARS`
(`rolling_window_counts`, centre = `start + (L−1)/2`), and turned into per-window ensemble
statistics (`ensemble_frequency_statistics`). Two figures are drawn from these:

- **Figure 1, internal variability** (`plot_internal_variability_trend`) — the ensemble
  **mean** rate as the central line, with the 25–75 % interquartile band and either the
  min/max member envelope or the 2.5–97.5 % envelope. The mean is used throughout rather than
  a median, because its resolution is 1/(M·L) whereas any rank statistic is limited to 1/L.
- **Figure 2, signal-to-noise** (`plot_signal_to_noise_ratio`) — the same mean divided by the
  standard deviation across members, which is unitless and says when the forced change exceeds
  the ensemble spread.

Two limitations are accepted deliberately and stated in the notebook: there is **no
declustering**, so one storm exceeding on several consecutive days counts several times, and
the rolling windows **overlap**, so consecutive points are not independent and the curve acts
as a display-only low-pass filter.

### 4.7 Grouped percentiles

A member's rate is an integer exceedance count divided by the window length L, so it can only
take multiples of the **rate quantum 1/L**. `np.percentile` lands on that same grid, which
turns the percentile curves into a staircase and can collapse the interquartile band to zero
width, even though the underlying mean is perfectly smooth.

`grouped_percentile` treats the sample as **binned** and interpolates *through* the tied
blocks, using the actual counts so that empty bins are skipped, with a half-width zero bin
`[0, w/2)` that keeps low percentiles non-negative. On a 15-year window this lifts the number
of distinct p25 values from 8 to 82. Which flavour is used is a one-line switch
(`FE_SPREAD_METHOD`), and `print_frequency_evolution_summary` prints the distinct-p25 count
under both flavours, so the effect stays visible without a separate figure. The function is
pinned by `helper/test_grouped_percentile.py`.

---

## 5. Reproducing the analysis

### 5.1 Order of execution

1. **`code/load_data_store_postprocessed.ipynb`** — build every cache. This is the only
   notebook that touches the raw data; everything downstream reads the caches. It takes by far
   the longest and only has to be run once.
2. **`code/analysis_return_hans.ipynb`** — return periods, per reanalysis dataset and for the
   two ensembles.
3. **`code/climate_model_evaluation.ipynb`** — distribution, Q-Q and table comparison.
4. **`code/create_precip_maps_hans.ipynb`** — event maps, weight maps and the precipitation
   climatology comparison.
5. **`code/compound_flood_risk_analysis.ipynb`** — snowmelt and soil-moisture maps, joint
   distributions, frequency evolution and signal-to-noise.

Steps 2–5 are independent of each other and can be run in any order.

### 5.2 Where to change what

Each notebook has one parameter block at the top; nothing below it needs editing.

- **Window length** — `WINDOW_DAYS` in the map and compound notebooks, `WINDOW_DAYS_SWE` and
  `WINDOW_DAYS_COMPOUND` in the preparation notebook. Only the 2-day compound series currently
  exist on disk; selecting another window means re-running step 7 of the preparation notebook
  first, which needs no raw reload because its inputs are complete.
- **Analysis period** — `MAP_START` / `MAP_END` for the maps, `EVAL_START_YEAR` /
  `EVAL_END_YEAR` for the evaluation, `FE_START` / `FE_END` for the frequency evolution.
- **Season** — `SPRING`, `JD_SEASON` and `FE_SEASON`, each taking a key of `cfg.SEASON_MONTHS`
  or `"all"`.
- **Compound selection** — the `JD_*` block for the joint distribution and the `FE_*` block for
  the frequency evolution: catchment, variable pair, members, threshold, rolling-window length
  and step, and the frozen reference window.
- **Rebuilding a cache** — the `FORCE_*` and `RECOMPUTE` flags, all `False` by default.

If a required cache is missing, the helper functions raise an error that names the exact
notebook cell and the exact setting that builds it, rather than failing somewhere deep inside
the computation.

### 5.3 Environment

Python 3.11 with the standard geoscience stack: `numpy`, `pandas`, `xarray`, `dask`, `scipy`,
`netCDF4`, `matplotlib`, `cartopy`, `geopandas`, `shapely` and `pyproj`. The notebooks set
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 and run Dask with the
synchronous scheduler, which is what the shared login nodes expect.

---

## 6. Further documentation

- **`Code_Overview.md`** — the function-level reference: every constant and function of every
  helper module with its signature and a one-line description, plus a cell-by-cell listing of
  each notebook. Use it to find where something is implemented; use this README to understand
  what it does and why.
