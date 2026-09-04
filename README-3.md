# Climate Change vs. Internal Variability of Compound Flood Drivers in Norway

This README gives an overview of the data and code structure from the analysis of this particular project.

The goal of the project is to first of all evaluate the return period of Storm Hans and on the other hand analyze
the compound precipitation + snowmelt compound flood driver and to ask how the frequency of such compound situations evolves over
1920–2034 in the CESM2 Large Ensemble climate model, and how the forced signal compares with internal
variability.

The README is organised so that it can be seen how the analysis was done and given access
to the ERA5 / seNorge / CESM2-LE / GFDL-SPEAR datasets, reproduce every figure of the report.

Following documents give additional valuable insight into the methods and code structure, where
especially **Code_Overview explains every function and variable defined**:

| Document | Answers |
|---|---|
| **Internship report** (PDF) | Especially methods and results plus discussion sections |
| **`Code_Overview.md`** | explains every constant and function of every module as a one-line description |
| **this README** | the conventions the code follows as well as layout of code structure |

---

## 1. Analysis concept & naming conventions

### 1.1 The two parts of the analysis

The project consists of two parts that have the same catchment-averaging machinery
(report Section 4.2):

- **Univariate precipitation extremes** — catchment-averaged precipitation series, annual
  maxima, GEV fits and return periods (report Section 4.4), the comparison of the ensembles
  against the reanalyses (Section 4.5) and the gridded climatology and significance maps
  (Section 4.6).

- **Compound extremes** — two catchment-averaged quantities in the same N-day window
  (precipitation with snowmelt, or precipitation with soil moisture), their joint distribution
  and the additive severity threshold (Section 4.7), and the evolution of the exceedance
  frequency and its signal-to-noise ratio in centred rolling windows across the 90 CESM2-LE
  members (Section 4.8).

### 1.2 Accumulation windows

Every quantity is defined over an **N-day window**, set once per notebook by `WINDOW_DAYS`.
The window operator depends on the variable:

| Variable | Operator | Function | Report |
|---|---|---|---|
| Precipitation | rolling sum | `rolling_accumulation` | Eq. 1 |
| Snowmelt | SWE decrease, `max(0, −(SWE(t) − SWE(t−(N−1))))` | `rolling_melt` | Eq. 2 |
| Soil moisture | rolling mean | `rolling_mean` | Eq. 3 |

SWE (Snow-water Equivalent) decreases count and is stored as a positive melt magnitude, so accumulation
maps to zero (no values below zero present) and high-melt days sit at the top of the distribution. 
Rolling operators are always applied to the whole time-period before any season etc. is evaluated, so every window is
complete. The window is in every filename as the name `1day`, `2day`, `3day`, … (`cfg.acc_tag`);
the analysis in the report was run with `WINDOW_DAYS = 2`.

### 1.3 Seasons

Next to the four standard seasons `DJF`, `MAM`, `JJA`, `SON` there is the extended spring season
window **`MAMJ`** (March–June) for the analysis with the snowmelt compound driver. A window spanning a season boundary
is assigned to the season of its closing day.

### 1.4 Catchments

Five NVE catchments in southern Norway are used, referred to in the code and the filenames:

| Name | Title | Source |
|---|---|---|
| `nevina_bergheim` | Nevina Bergheim | NVE NEVINA |
| `nevina_honnefoss` | Nevina Hønnefoss | NVE NEVINA |
| `nevina_losna` | Nevina Losna | NVE NEVINA |
| `regine_drammen` | Regine Drammen | NVE REGINE |
| `regine_glomma` | Regine Glomma | NVE REGINE |

The compound analysis additionally uses the **combination** `regine_drammen_glomma`
(Drammen + Glomma). Their two GeoJSONs are merged with `unary_union` into one polygon *before*
the per-cell area fractions are computed -> the shared border is never double-counted


### 1.5 Filenames

Figures and caches have their full selection in the name, generally built up like this:

```
{quantity}_{window}_{variables}_{catchment}_{start}-{end}[_{season}][_thr{value}][_ref{start}-{end}].pdf
```

For example:

- `2daymedian_90pctl_snowmelt_2_98pctl_diff_1995-2024.pdf`
- `joint_distribution_2day_precipitation_snowmelt_regine_drammen_glomma_1995-2024_MAMJ_thr0.7.pdf`
- `internal_variability_trend_2day_precipitation_snowmelt_regine_drammen_glomma_1920-2034_10year_thr0.9_ref1995-2024_MAMJ.pdf`

`_ref1995-2024` is the **reference window** that fixes the position of the compound threshold line.

---

## 2. Repository structure/layout

The repository is separated in two parts: **the Python module-imports in `helper/` and the
code-execution notebooks in `code/`**. All reusable logic is in `helper/`, the notebooks only
set parameters and call the functions inside `helper/`, no reusable function is defined in a notebook.

**Important**: the raw data and all postprocessed caches live on the NIRD project area
`NS9873K` and are **not** part of this repository (see Section 2.4). Without access to that
project the notebooks can be read but not executed. 

### 2.1 `helper/`

All importable Python modules. They are strictly separated by theme and topic: paths are only
in `config_paths.py`, figure code in `plot_style.py`, return-period build-ups in
`return_period.py` and catchment-based calculations in `catchment_tools.py`. Every function of every 
module is listed in `Code_Overview.md`, Sections 1–9.

- **`config_paths.py`** — paths and constants only: raw-data and output directories, the
  catchment and ensemble registries, the season definitions, the model colours and labels, and
  every path builder in the project.
- **`data_era5.py`, `data_senorge.py`, `data_smile.py`** — one module per data source, all
  for file discovery, unit conversion, spatial-cache builders that write one NetCDF
  per dataset (or per ensemble member), and the gridded median / 90th-percentile computations.
  `data_smile.py` also holds the per-member statistics and the percentile-rank significance test.
- **`catchment_tools.py`** — catchment averaging, the window operators of Section 1.2, the CESM2-LE
  compound-series builder, the season handling, the compound threshold statistics, the complete
  frequency-evolution, the two return-period commands `run_all`(reanalyses) and `run_all_smile` (ensembles).
- **`return_period.py`** —  return period calculations: annual maxima, Weibull plotting positions,
  the GEV fit, return levels and the return-period estimate.
- **`plot_style.py`** — Matplotlib and Cartopy figure of the project, plus the projection
  and colormap constants. No data loading and no statistics happen here.
- **`generate_weights.py`** — run-once .py-file that computes the per-cell area fraction of each
  catchment on each model grid and writes one weight NetCDF per catchment × dataset.
- **`test_grouped_percentile.py`** — reference unit test for `grouped_percentile`, tested
  against 2002–2011 CESM2-LE window (90 members, L = 10). Run with
  `python helper/test_grouped_percentile.py` from the repository root.
- **`prec_seq.txt`, `prec_div.txt`** — the IPCC sequential and diverging precipitation
  colormaps as 256-row RGB tables which are used inside `plot_style.py`.

### 2.2 `code/`

Five code-execution notebooks listed here. Their cell-by-cell contents are in
`Code_Overview.md`, Section 10. The figures they produce for the report are mapped in
Section 4 below.

- **`load_data_store_postprocessed.ipynb`** — builds **all** postprocessed caches
  (catchment weights, daily precipitation caches for ERA5, seNorge and SMILEs,
  the ERA5-interpolated caches, the daily SWE and soil-moisture caches, the N-day snowmelt
  caches and the CESM2-LE catchment-averaged compound series. It is the only notebook that
  takes the raw data and it produces no figures.
- **`analysis_return_hans.ipynb`** — return-period analysis of Storm Hans, first for the
  reanalyses (one dataset at a time via `DATASET_KEY`) and then
  for the two SMILEs  (members pooled before the GEV fit).
- **`climate_model_evaluation.ipynb`** — evaluation of CESM2-LE and GFDL-SPEAR against ERA5
  and seNorge over 1985–2024 per catchment and window. Analyzing distribution figures, Q-Q plots and the
  percentile-mapping.
- **`create_precip_maps_hans.ipynb`** — all spatial precipitation figures that give an overview of
  spatial distribution of precipitation in SMILE. Includes the Storm Hans event maps, the catchment
  weight maps, and the 1995–2024 climatology comparison against ERA5 regridded onto the CESM2-LE grid,
  including the significance-hatched annual, seasonal and single-season (`MAMJ`) versions.
- **`compound_flood_risk_analysis.ipynb`** — the compound analysis part. The first half repeats the map
  methodology for snowmelt and soil moisture instead of precipitation (units kg/m², one
  `VARIABLES` dictionary configuring both). The second half is the compound analysis proper:
  the joint distribution with the severity threshold line, the rolling-window evolution of the
  exceedance frequency with its internal-variability band, and the signal-to-noise ratio.

### 2.3 `figures/`

Generated figures in separate folders (mostly separated by notebooks)

- `timeseries_return_hans/` — return-period and time-series figures.
- `climate_model_evaluation/` — distribution and Q-Q figures plus the CSV tables.
- `precip_maps_hans/` — Storm Hans event maps, weight maps and precipitation climatology maps.
- `compound_flood_risk_output/` — snowmelt and soil-moisture maps and the joint distributions.
- `compound_flood_risk_output/frequency_evolution/` — the rolling-window frequency and
  signal-to-noise figures together with their `{stem}_ensemble.csv` and `{stem}_metadata.json`.

Every figure function writes to **two** roots: the repository folder above and a mirror under
`FIGURES_DIR` on the data lake. A few of the largest seasonal PDFs are excluded from the
repository in `.gitignore`.

### 2.4 Large external data (not in this repo)

The raw datasets and the postprocessed caches live outside the repository and are not tracked.
The dataset keys in the first column are the ones used in the notebook parameter blocks
(`DATASET_KEY`, `SMILE_RUN_TABLE`) and in the cache filenames:

| Key | Dataset | Grid | Variables | Members | Raw directory |
|---|---|---|---|---|---|
| `era5_0.5` | ERA5 | 0.5° | `tp24` (m) | — | `/nird/datapeak/NS9873K/etdu/raw/era5/…/tp24/` |
| `era5_0.25` | ERA5 | 0.25° | `tp24` (m) | — | `/nird/datapeak/NS9873K/etdu/raw/era5/…/tp24/` |
| `senorge` | seNorge | 1 km UTM-33 | `rr` (mm) | — | `/nird/datapeak/NS9873K/DATA/senorge/rr/` |
| `era5_interpolated` | ERA5 on the CESM2-LE grid | 0.94° × 1.25° | `tp`, `sd`, `swvl` | — | `/nird/datalake/NS9873K/etdu/raw/era5/scandinavia/` |
| `cesm2_le` | CESM2 Large Ensemble | 0.94° × 1.25° | `PRECT`, `SWE`, `SM` | 100 / 90 / 90 | `/nird/datalake/NS9873K/etdu/raw/smile/cesm2_le/scandinavia/` |
| `gfdl_spear_med_le` | GFDL-SPEAR-MED-LE | 0.5° × 0.625° | `tp24` | 30 | `/nird/datalake/NS9873K/etdu/raw/smile/gfdl_spear_med_le/scandinavia/` |

Catchment GeoJSONs are in `/nird/datalake/NS9873K/etdu/raw/nve/`, the caches in
`/nird/datalake/NS9873K/lbal/postprocessed/` and the figure mirror in
`/nird/datalake/NS9873K/lbal/figures/`.

ERA5 is converted from metres to mm on load; seNorge is already in mm and its fill value
−999.99 is masked; the SMILE unit is auto-detected from the metadata. All spatial caches are
cropped to `OVERALL_PRECIP_EXTENT = (3.0°E, 16.0°E, 56.5°N, 66.0°N)`, the domain of report
Section 4.1. Only **90 of the 100** CESM2-LE members carry SWE and soil-moisture output — the
odd members 001–019 are missing — so every compound quantity is computed on the intersection
returned by `common_cesm2_le_members()` and the ensemble size is detected rather than assumed.

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
derived N-day snowmelt caches, and `catchment_averaged/` the catchment time series, including
the `[member, time]` CESM2-LE compound series.

**Alternative: obtaining the data independently.** ERA5 can be downloaded from the Copernicus
Climate Data Store, seNorge is distributed by MET Norway, and the two large ensembles are
published by NCAR (CESM2-LE) and GFDL (SPEAR). Note that the files used here are not the raw
archive versions: they are cropped to the Scandinavian domain, and ERA5 additionally exists in
a version regridded onto the CESM2-LE grid. Reproducing them from the public sources therefore
requires the cropping and regridding step before
`code/load_data_store_postprocessed.ipynb` can be run.

---

## 3. Reproducing the analysis

### 3.1 Order of execution

1. **`code/load_data_store_postprocessed.ipynb`** — build every cache. This is the only
   notebook that touches the raw data; everything downstream reads the caches. It takes by far
   the longest and only has to be run once. Every step skips what already exists unless the
   corresponding `FORCE_*` flag is set.
2. **`code/analysis_return_hans.ipynb`** — return periods, per reanalysis dataset and for the
   two ensembles.
3. **`code/climate_model_evaluation.ipynb`** — distribution, Q-Q and table comparison.
4. **`code/create_precip_maps_hans.ipynb`** — event maps, weight maps and the precipitation
   climatology comparison.
5. **`code/compound_flood_risk_analysis.ipynb`** — snowmelt and soil-moisture maps, joint
   distributions, frequency evolution and signal-to-noise.

Steps 2–5 are independent of each other and can be run in any order.

### 3.2 Where to change what

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
  the frequency evolution. The `FE_*` block is the code counterpart of Table 2 of the report:
  catchment, variable pair, members, threshold, rolling-window length and step, season, and the
  frozen reference window `FE_NORM_REF`.
- **Spread estimator** — `FE_SPREAD_METHOD`, `percentile_grouped` (default, report Eq. 8) or
  `percentile_empirical`. `print_frequency_evolution_summary` reports the number of distinct
  p25 values under both, which is the diagnostic for the staircase artefact.
- **Rebuilding a cache** — the `FORCE_*` and `RECOMPUTE` flags, all `False` by default.

If a required cache is missing, the helper functions raise an error that names the exact
notebook cell and the exact setting that builds it, rather than failing somewhere deep inside
the computation.

### 3.3 Environment

Python 3.11 with the standard geoscience stack: `numpy`, `pandas`, `xarray`, `dask`, `scipy`,
`netCDF4`, `matplotlib`, `cartopy`, `geopandas`, `shapely` and `pyproj`. The notebooks set
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 and run Dask with the
synchronous scheduler, which is what the shared login nodes expect.

---
