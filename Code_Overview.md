# Code Overview

This is a reference for the modules in `helper/` and a cell-level reference for the
notebooks in `code/`. For the description of *what* the analysis does and in which order
things have to be run, see  the `README.md`.


---

## 1. `helper/config_paths.py`

Paths and constants are here, no plotting, data loading

### 1.1 Raw-data directories

- `ERA5_RAW_DIR` — raw ERA5 annual NetCDF files (`tp24`)
- `SENORGE_RAW_DIR` — raw seNorge annual NetCDF files (`rr`)
- `ERA5_INTERPOLATED_BASE` — ERA5 regridded onto the CESM2-LE grid; sub-paths
  `ERA5_INTERPOLATED_DIR` (`tp`), `ERA5_INTERPOLATED_SWE_DIR` (`sd`),
  `ERA5_INTERPOLATED_SWVL_DIR` (`swvl`)
- `CESM2_LE_BASE` with sub-paths `CESM2_LE_DIR` (`PRECT`, mm/day), `CESM2_LE_SWE_DIR` (`SWE`),
  `CESM2_LE_SM_DIR` (`SM`)
- `GFDL_SPEAR_DIR` — GFDL-SPEAR-MED-LE precipitation
- `CATCHMENT_RAW_DIR` / `GEOJSON_DIR` — NVE catchment GeoJSONs and the legacy weight files

### 1.2 Output directories

- `FIGURES_DIR`, `FIGURES_DIR_SECONDARY` — every figure is written to both
- `POSTPROC_DIR`, `WEIGHTS_DIR` — postprocessed caches and catchment weight files
- `OVERALL_PRECIP_EXTENT` — `(W, E, S, N)` crop applied to all spatial caches

### 1.3 Entries and registries

- `SMILE_CONFIG` — per-model metadata: `model_dir`, `n_members`, `member_digits`,
  `default_start/end`, `ref_dataset`, `ref_resolution`, `figure_label`, `tp24_unit_mode`
- `CATCHMENTS` — the five study catchments, slug → title
- `COMPOUND_CATCHMENTS` — catchments of the compound analysis: `regine_drammen`,
  `regine_glomma` and the dissolved union `regine_drammen_glomma`. Kept apart from
  `CATCHMENTS` so the return-period loops stay untouched
- `GEOJSON_FILES` — slug → GeoJSON filename
- `MODEL_COLORS`, `MODEL_LABELS`, `MODEL_ORDER` — colours/labels/order of the models in all
  evaluation figures. They live here rather than in `plot_style.py` to break a circular
  import with `catchment_tools`
- `HANS_DATE`, `HANS_SEARCH_YEAR` — Storm Hans reference date and event year

### 1.4 Seasons

- `SEASONS_ORDER` — `["DJF", "MAM", "JJA", "SON"]`. Drives the 4×3 seasonal figure, which
  needs exactly four entries, so custom windows do not belong here
- `SEASON_MONTHS` — season key → month numbers, including the four-month spring window
  `"MAMJ"` = Mar–Jun. Imported by `data_era5` and `data_smile`. A season is always a named
  key; there is no ad-hoc `(m1, m2)` selector
- `SEASON_LABELS` — season key → plot label, e.g. `"MAMJ"` → `"Spring (MAMJ)"`
- `rate_unit_label(months=None)` — `'events per year'` for all twelve months, otherwise
  `'events per season'`. Used by both `catchment_tools` (console output) and `plot_style`
  (y-axis label). Only the label changes; the value is the same, since each year
  contributes one season

### 1.5 Filename helpers

- `res_tag(dataset, resolution)` — filesystem-safe tag, e.g. `era5_0.5x0.5`
- `acc_tag(window_days)` — `"1day"`, `"2day"`, `"3day"`, …
- `postproc_dir(dataset)`, `postproc_filename(...)`, `figure_filename(...)` — low-level
  builders the `*_path` / `*_paths` helpers below compose
- `smile_reference_tag(reference_dataset, reference_resolution)` — tag for a SMILE
  reference series used in the Storm Hans figures

All N-day path builders (`median_precip_*`, `p90_precip_*`, `*_seasonal_*`,
`cesm2_window_*`, `era5_interp_window_*`, `field_window_cache_path`) take `window_days` and
embed `acc_tag(window_days)` in the name. For `window_days=2` the names are identical to
the older `2day` names, so existing caches and figures stay valid.

### 1.6 Cache paths

- `catchment_postproc_path(dataset, resolution, window_days, slug, start, end)` — catchment-
  averaged time series
- `overall_precip_path(dataset, resolution, start, end)` — spatial daily cache
  (ERA5 / seNorge / ERA5-interpolated)
- `overall_precip_member_path(dataset, member_id, start, end)` — one SMILE member
- `smile_member_postproc_path(...)`, `smile_yearmax_stats_path(...)` — SMILE per-member
  catchment cache and the pooled annual-maxima statistics
- `cesm2_catchment_field_path(window_days, slug, variable, start, end)` — one file per
  (window, catchment, variable) holding all common members as `[member, time]`;
  variable is `precipitation`, `soil_moisture` or `snowmelt`
- `era5_interp_window_median_cache_path(...)` / `..._p90_cache_path(...)`
- `cesm2_annmedian_cache_path(...)`, `cesm2_window_annmedian_cache_path(...)`,
  `cesm2_window_p90_cache_path(...)`, `cesm2_window_global_median_cache_path(...)`,
  `cesm2_window_per_member_medians_cache_path(...)`,
  `cesm2_window_per_member_p90_cache_path(...)`
- Seasonal counterparts: `era5_interp_window_seasonal_median_cache_path(...)`,
  `era5_interp_window_seasonal_p90_cache_path(...)`,
  `cesm2_window_seasonal_global_median_cache_path(...)`,
  `cesm2_window_seasonal_per_member_medians_cache_path(...)`,
  `cesm2_window_seasonal_global_p90_cache_path(...)`,
  `cesm2_window_seasonal_per_member_p90_cache_path(...)`
- `field_daily_cache_path(model, kind, start, end, member_id=None, window_days=1)` — daily
  SWE / soil-moisture cache. `window_days=1` gives the raw `…_1day_…` cache;
  `window_days>=2` gives the pre-computed N-day snowmelt cache `…_{N}day_…`
  (SWE only, stored as `max(0, −ΔSWE)`, so gains map to zero)
- `field_window_cache_path(model, kind, which, window_days, start, end, season="")` —
  derived median / p90 cache, global or per member, with an optional season

### 1.7 Figure paths

- `figure_paths(...)` — both roots for the `timeseries_return_hans` PDFs
- `smile_figure_paths(...)` — both roots for the SMILE return-period PDFs
- `precip_map_figure_paths(fig_subdir, fname)` — both roots for any map PDF; the other map
  helpers wrap it
- `annmedian_precip_paths(...)` — 4-panel annual median map
- `median_precip_paths(fig_subdir, window_days, start, end)` — 2-panel N-day median
- `median_precip_diff_paths(...)`, `median_precip_diffonly_paths(...)` — 3-panel and
  single-panel N-day median difference
- `p90_precip_diff_paths(...)`, `p90_precip_diffonly_paths(...)` — same for the 90th
  percentile
- `median_precip_sig_diff_paths(..., pctl_tag)`, `p90_precip_sig_diff_paths(..., pctl_tag)`
  and their `*_sig_diffonly_paths` counterparts — significance-hatched versions
- `median_seasonal_precip_sig_diff_paths(...)`, `p90_seasonal_precip_sig_diff_paths(...)` —
  4×3 seasonal significance figures
- `p90_single_season_precip_sig_diff_paths(fig_subdir, window_days, season, start, end, pctl_tag)`
  — the single-season 1×3 figure. `pctl_tag` must keep the `{lo}_{hi}pctl` form so that
  `plot_window_interp_3panel` picks the same colorbar offset as the annual figure
- `COMPOUND_FREQ_FIG_SUBDIR` and `compound_freq_figure_paths(fname)` — output folder
  `compound_flood_risk_output/frequency_evolution` for the rolling-window figures and their
  CSV/JSON
- `compound_freq_stem(figtype, window_days, x_variable, y_variable, slug, start, end, roll_years, threshold, norm_ref, season, members)`
  — shared filename stem. `figtype` is `internal_variability_trend` or `signal_to_noise`.
  The threshold keeps its decimal point (`thr0.8`, `thr1.0`) and
  is followed by `_ref{start}-{end}`, because the frozen reference window fixes the position
  of the threshold line. `_{season}` and `_members{spec}` are appended only when they differ
  from the defaults

---

## 2. `helper/data_era5.py`

ERA5 file discovery, loading and spatial-cache builders.

- `find_era5_files(era5_dir, resolution)` — sorted annual files for one resolution
- `load_era5_precipitation(era5_files)` — open the annual files lazily and convert `tp24`
  from metres to mm. Every `save_era5_*_overall` builder goes through this
- `get_year_range(era5_files, resolution)` — `(start_year, end_year)`
- `day_start(dt)` — normalise a timestamp to midnight
- `select_time_range_by_day(da, start, end)` / `select_single_time_by_day(da, day)` —
  day-level selection, robust to sub-daily timestamps
- `coord_slice(coord, lower, upper)` — a `slice()` that works on ascending and descending axes
- `pick_year_file(files, year)` — the single file covering one year
- `save_era5_overall(resolution, era5_dir, out_path_fn, extent, force)` — build the
  full-period cropped daily cache
- `compute_era5_annual_median_2d(...)` — annual median map from that cache
- `compute_era5_window_median_2d(...)` — N-day rolling median map
- `find_era5_interpolated_files(era5_interp_dir)` — sorted ERA5-on-CESM2-grid files. The
  variable prefix is matched generically (`tp` / `sd` / `swvl`), so one function serves
  precipitation, SWE and soil moisture
- `get_year_range_era5_interp(files)` — year range of that file list
- `save_era5_interpolated_overall(era5_interp_dir, out_path_fn, extent, force)` — the
  interpolated precipitation cache
- `save_era5_interpolated_field_overall(era5_interp_dir, out_path_fn, variable, cache_var, units, extent, force)`
  — the interpolated daily SWE / soil-moisture cache, no metre→mm conversion
- `save_era5_interpolated_field_diff_overall(..., diff_fn, open_cache_fn, window_days, ...)` —
  the N-day snowmelt cache built from the 1-day cache via `diff_fn`
- `compute_era5_interpolated_window_median_2d(...)` / `..._p90_2d(...)` — N-day median and
  90th percentile, cached. The p90 reader accepts both the current
  `twoday_p90_precip` and the legacy `twodayp90_precip` variable name
- `_SEASON_MONTHS` — alias of `cfg.SEASON_MONTHS`, so a new season key is picked up here
  automatically
- `_filter_season_da(da, season)` — keep the time steps whose month is in the season
- `compute_era5_interpolated_window_seasonal_median_2d(season, ...)` /
  `..._seasonal_p90_2d(season, ...)` — seasonal versions, cached

---

## 3. `helper/data_senorge.py`

Same structure as `data_era5.py` and handling of the seNorge grid.

- `find_senorge_files(senorge_dir)` — sorted annual files
- `get_year_range_senorge(files)` — year range
- `load_senorge_precipitation(senorge_files)` — open and concatenate `rr`, masking the fill
  value −999.99
- `latlon_extent_to_utm_bbox(extent)` — `(W, E, S, N)` in degrees → UTM-33 bounding box in
  metres
- `save_senorge_overall(senorge_dir, out_path_fn, extent, force)` — full-period cropped
  daily cache
- `compute_senorge_annual_median_2d(...)` — annual median map
- `compute_senorge_2day_median_2d(...)` — 2-day rolling median map

---

## 4. `helper/data_smile.py`

SMILE (single-model initialized large-ensemble) file discovery, loading and cache builders. Covers `cesm2_le` and
`gfdl_spear_med_le`.

- `_FILE_PATTERNS`, `_SMILE_VAR_TOKEN` — per-model filename patterns and the variable token
  inside them; a new model or variable is registered here
- `find_smile_members(model_dir, dataset)` — sorted unique member IDs. For CESM2-LE the
  variable token (PRECT / SWE / SM) is matched generically, and a directory holding mixed
  variables raises an error
- `find_smile_files_for_member(model_dir, dataset, member_id, ...)` — chronological files of
  one member
- `_convert_smile_tp24_to_mm(da, unit_mode)` — m→mm conversion with auto-detection from the
  metadata
- `load_smile_precipitation(files, start, end, unit_mode)` — open, convert, deduplicate and
  sort one member's precipitation
- `get_year_range_smile(model_dir, dataset)` — full year range on disk
- `save_smile_overall(dataset, model_dir, out_path_fn, force, unit_mode)` — per-member daily
  precipitation caches
- `load_cesm2_le_field(files, variable, start, end)` — open a CESM2-LE state variable
  (`SWE` / `SM`) without unit conversion, with the same dedup/sort robustness
- `save_cesm2_le_field_overall(model_dir, out_path_fn, variable, cache_var, units, force)` —
  per-member daily SWE / soil-moisture caches in kg/m²
- `save_cesm2_le_field_diff_overall(..., diff_fn, open_cache_fn, window_days, ...)` —
  per-member N-day snowmelt caches built from the 1-day caches
- `compute_cesm2_le_annual_median_2d(...)` — pooled median annual map: all
  members × years per pixel, then `np.nanmedian`
- `compute_cesm2_le_window_median_2d(...)` — 100-member median N-day rolling map
- `compute_cesm2_le_window_p90_2d(...)` — median of the per-member 90th percentiles
- `compute_cesm2_le_window_global_median_2d(...)` — global median over all members × time
  per pixel; also writes the per-member medians `[n_members, lat, lon]` needed for the
  significance test
- `compute_cesm2_le_window_per_member_p90_2d(...)` — per-member 90th-percentile fields plus
  the global p90; writes two caches
- `compute_significance_masks(da_era5, per_member_da, lower_pctl, upper_pctl)` — percentile-
  rank test without FDR correction. A pixel is flagged when the number of members above
  (or below) ERA5 reaches `round(upper_pctl/100 · n_members)`. NaN pixels are False.
  Returns `(sig_cesm_higher, sig_era5_higher)`
- `_SEASON_MONTHS`, `_season_time_mask(time_values, season)` — season alias and a boolean
  time mask that handles both `numpy.datetime64` and `cftime.DatetimeNoLeap` axes
- `compute_cesm2_le_window_seasonal_global_median_2d(season, ...)` /
  `..._seasonal_per_member_p90_2d(season, ...)` — seasonal versions, two caches each

---

## 5. `helper/catchment_tools.py`

Catchment averaging and all non-plotting data preparation.

### 5.1 Weights and spatial averaging

- `find_weight_file(dataset, resolution, catchment_slug, weight_dir=None)` — locate the
  area-fraction weight NetCDF
- `load_weights(weight_path)` — open it as a DataArray
- `_spatial_dims(da)` — detect the `(y_dim, x_dim)` name pair
- `align_weights_to_precip(precip_da, weights)` — reindex the weight field onto the data
  grid; raises when the grids cannot be matched
- `compute_catchment_mean(precip_da, weights)` — the area-fraction weighted spatial mean
- `crop_to_weight_bbox(da, weights, pad_cells)` — crop to the bounding box of the non-zero
  weights before averaging, so no full-domain array is loaded
- `crop_weight_field_to_nonzero_bbox(da, pad_cells)` — the same crop applied to a weight
  field itself
- `_mean_step(coord, fallback)` / `get_plot_extent_and_crs(da)` — infer a tight map extent
  and the Cartopy CRS from a weight field
- `load_catchments(geojson_files, geojson_dir=None)` — all catchment GeoDataFrames in
  EPSG:4326

### 5.2 Cache and time handling

- `save_postproc_dataset(ds, out_path)` / `load_postproc_dataset(nc_path)` — compressed
  Dataset write / read
- `_chunksizes_for(da, time_chunk, y_chunk, x_chunk)` — chunk sizes aligned to `da.dims`
- `save_spatial_netcdf(da, out_path, var_name, ..., units="mm")` — float32 compressed
  chunked write, used by every `save_*_overall`. Pass `units="kg/m2"` for SWE and soil
  moisture
- `open_precip_cache(cache_path, start_year, end_year)` — lazily open a precipitation cache
  (`tp24_mm` / `rr_mm`) and subset the years
- `open_field_cache(cache_path, var_name, start_year, end_year)` — the same for an explicitly
  named SWE / soil-moisture variable
- `_infer_year_range_from_cache(...)`, `_validate_requested_years(...)`,
  `get_cached_year_range(dataset, resolution, window_days)` — read the year range out of the
  cache filenames and validate a requested range against it
- `subset_time_series_by_year(da, start, end)` — year clip for any DataArray

### 5.3 (Rolling) Window operators

- `rolling_accumulation(da, window_days)` — rolling sum, works for 1-D series and 2-D fields
- `rolling_change(da, window_days, var_name)` — trailing last-minus-first change,
  `da(t) − da(t−(N−1))`. Retained but currently unused
- `rolling_melt(da, window_days, var_name)` — snowmelt magnitude
  `max(0, −(da(t) − da(t−(N−1))))`, so only SWE decreases count and gains map to zero. Used
  to pre-build the daily N-day snowmelt caches
- `rolling_mean(da, window_days, var_name)` — trailing rolling mean, the soil-moisture
  counterpart
- `rolling_identity(da, window_days, var_name)` — pass-through for fields whose window
  quantity is already on disk (the N-day snowmelt cache); returns `da` unchanged

### 5.4 CESM2-LE compound series

- `common_cesm2_le_members()` — members present in all three CESM2-LE variable directories.
  PRECT ∩ SM ∩ SWE gives 90; SM and SWE lack the odd members 001–019
- `_CESM2_COMPOUND_SPECS` — per-variable spec (source dir, variable name, window operator,
  units) for precipitation, soil moisture and snowmelt
- `_cesm2_compound_window_op(variable, da_daily, window_days)` — dispatch one variable to its
  window operator, so the N-day definition lives in one place
- `save_cesm2_le_catchment_field_series(variable, window_days, catchment_slug, force)` —
  build one `[member, time]` cache over the full record: per member weighted catchment mean,
  then the window operator (precipitation → sum, soil moisture → mean,
  snowmelt → `max(0, −ΔSWE)`)
- `_check_series_reasonableness(da, variable)` — range / sign / all-NaN check after a build;
  prints a warning instead of failing
- `parse_member_selection(selection, available)` — normalise `"all"`, `"1-30"` or
  `[3, 4, 27]`; raises listing the unavailable members
- `load_cesm2_le_catchment_field_series(variable, window_days, slug, start, end, members)` —
  load and subset a compound series. Missing selections raise an error naming the notebook
  cell that builds them
- `require_compound_series(catchment_slug, variables, window_days)` — check that the caches
  exist; the error names the `WINDOW_DAYS_COMPOUND` / `COMPOUND_SLUGS` /
  `COMPOUND_VARIABLES` settings in `load_data_store_postprocessed.ipynb`

### 5.5 Seasons

- `resolve_season_months(season)` — month list for a selector: `"all"` → `None`, otherwise a
  key of `cfg.SEASON_MONTHS`. Anything else raises, naming the valid keys
- `season_tag(season)` / `season_label(season)` — filename tag and legend label
- `subset_season(da, season)` — cut a `[member, time]` series down to the months of a
  season. An N-day window belongs to the season of its closing day, matching the mask
  applied inside `run_compound_frequency_evolution`. Raises when nothing is left

### 5.6 Compound thresholds and frequency evolution

- `compound_threshold_stats(x_vals, y_vals, threshold, x_max=None, y_max=None)` — severity
  `s = x/x_max + y/y_max`, the exceedance mask, the counts and the two axis intercepts of
  the criterion line. Pure array maths
- `FREQ_SPREAD_KINDS` — which spread elements Figure 1 draws: `iqr` (blue 25–75 % band),
  `minmax` and `p025p975`. The latter two share one style and are mutually exclusive
- `FREQ_SPREAD_METHODS`, `_FREQ_SPREAD_COLUMNS` — how those numbers are obtained. Two
  percentile-based options: `percentile_empirical` (`np.percentile`; because a member rate
  is an integer count / L these land on multiples of the rate quantum 1/L, so the curves
  form a staircase and the IQR band can collapse to zero width) and `percentile_grouped`
  (default; interpolated, staircase removed). `print_frequency_evolution_summary` reports
  how many distinct p25 values each flavour produces, which is the diagnostic for the
  staircase
- `frequency_spread_columns(spread_method)` — the ensemble-table column names for one
  method. `central` is always `f_mean`; there is no median column
- `FREQ_PERCENTILES` — `(2.5, 25.0, 75.0, 97.5)`, driving both the empirical and the grouped
  columns. No 50th percentile is computed, and no median is drawn anywhere
- `grouped_percentile(values, q, bin_width, zero_half_bin=True)` — percentiles of a
  quantised sample, interpolated through the tied blocks as
  `lower_edge(b) + (target − F)/f · width(b)`, with `F` and `f` from the actual counts
  (`np.unique`), so empty bins are skipped. Bins are centred except for a half-width zero bin
  `[0, w/2)`, which keeps low percentiles non-negative. `bin_width` must be `1/roll_years`.
  Reference case in `helper/test_grouped_percentile.py`
- `validate_frequency_evolution_config(config)` — validate and normalise the `FE_*` block and
  return a copy with `season_months`, `season_tag`, `record_start/end`, `spread_method`,
  `rate_quantum` (= 1/L) and the resolved cache paths. Checks the mutually exclusive
  `minmax` / `p025p975` rule and rejects `spread_method='std'` and `FE_SPREAD_SHOW=('std',)`
- `load_compound_pair(catchment_slug, combo, window_days)` — full-record `[member, time]`
  pair for the two `FE_COMBO` variables, inner-aligned and loaded into memory
- `freeze_normalisation_maxima(da_x, da_y, ref_years, ref_members, season_months)` — the
  frozen `x_max` / `y_max`: sample maxima of `FE_NORM_REF`, restricted to the reference
  member pool and, when set, to the season months, so the criterion line comes from the same
  population it is applied to
- `annual_exceedance_counts(candidate, exceed, years_of_time)` — collapse the
  `[member, time]` masks to per-calendar-year counts `[member, year]`; cftime/no-leap safe,
  so a season selection simply leaves the out-of-season days at zero
- `rolling_window_counts(years, k_ary, roll_years, roll_step)` — counts per member in every
  complete centred rolling window; returns `(starts, centres, counts)` with
  centre = `start + (L−1)/2`. Half-years are kept, the ends are not padded
- `ensemble_frequency_statistics(starts, centres, counts, roll_years)` — per window:
  `f_mean` (ensemble mean, identical to the pooled rate ΣK/(M·L), resolution 1/(M·L)),
  `sigma` across members, `min` / `max`, `rate_quantum`, the empirical
  `p025`/`p25`/`p75`/`p975` and the grouped `p025_grouped`/`p25_grouped`/`p75_grouped`/`p975_grouped`,
  plus `signal_to_noise` = `f_mean / sigma`. Asserts that every member rate is an integer
  multiple of 1/L
- `run_compound_frequency_evolution(config)` — the orchestration: load pair → freeze maxima →
  severity → season and candidate mask → exceedances → annual counts → rolling windows →
  ensemble statistics. Returns `{config, ensemble, diagnostics}`
- `print_frequency_evolution_summary(result)` — console summary: frozen maxima and the season
  they were taken over, the physical threshold line, candidate and exceedance days, first and
  last window, mean σ, the IQR of the selected method, the rate quantum with the
  distinct-p25 count empirical vs grouped, the S/N range and a low-count warning
- `write_frequency_evolution_outputs(result, stem, out_paths_fn)` — write
  `{stem}_ensemble.csv` and `{stem}_metadata.json` next to the PDFs

### 5.7 Return-period and evaluation tables

- `pool_member_annual_maxima(member_annual_maxima)` — pool the per-member annual-maxima
  Series of an ensemble into one Series for the GEV fit
- `_load_or_build_smile_annual_maxima_for_period(...)` — per-member annual maxima for one
  SMILE period; reads the cache or builds and saves it first
- `SMILE_REFERENCE_SPECS`, `_load_smile_hans_references(...)` — which SMILE member and period
  serve as the Storm Hans reference in each figure, and the loader for those series
- `run_all(dataset, resolution, window_days, ...)` — the main loop over the five catchments
  for one reanalysis dataset: cache → catchment mean → rolling accumulation → figures
- `run_all_smile(dataset, model_dir, start_year, end_year, window_days, ...)` — the SMILE
  counterpart: loop members, pool the annual maxima, produce the ensemble return-period and
  distribution figures
- `load_annual_maxima_per_catchment(window_days, start, end)` — annual maxima for all
  catchments and models as a nested dict
- `load_daily_values_per_catchment(window_days, start, end)` — the same for the daily values
- `build_percentile_mapping_table(climate_key, climate_data, refs, percentiles)` — percentile
  comparison SMILE vs reanalysis
- `build_distribution_summary_table(annual_maxima)` — mean, std and quantiles per model

---

## 6. `helper/return_period.py`

Return Period statistics

- `get_annual_maxima(da)` — annual maxima of a daily catchment series as a Series indexed by
  year
- `weibull_plotting_positions(annual_max)` — empirical return periods, Weibull formula
- `fit_gev(annual_max)` — GEV fit via scipy MLE; returns `(c, loc, scale)`
- `gev_return_level(c, loc, scale, return_periods)` — GEV quantiles for an array of return
  periods
- `get_event_annual_max(da, search_year)` — `(value, date)` of the annual maximum in the
  event year
- `estimate_return_period(event_value, c, loc, scale)` — `T = 1 / (1 − CDF(x))`

---

## 7. `helper/plot_style.py`

All Matplotlib and Cartopy figure code.

### 7.1 Constants

- `fig_dpi` (= 150) — the output-resolution lever; every `savefig` passes it. Lowering it
  shrinks the PDFs while the catchment vectors stay crisp. The map functions call
  `ax.set_rasterization_zorder(4)`, so coastlines, ocean, land and pcolormesh become raster
  at this DPI while the catchment outlines and labels stay vector
- `MAP_PROJ` — Lambert Conformal centred on Scandinavia
- `DATA_CRS_LATLON` (PlateCarree), `DATA_CRS_SENORGE` (UTM zone 33), `OCEAN_COLOR`
- `PRECIP_CMAP`, `PRECIP_DIV_CMAP` — IPCC sequential and diverging colormaps loaded from
  `prec_seq.txt` / `prec_div.txt` by `_load_ipcc_prec_seq` / `_load_ipcc_prec_div`
- `WEIGHT_CMAP` — viridis, for the weight-fraction maps
- `MODEL_COLORS`, `MODEL_LABELS`, `MODEL_ORDER` — re-exported from `config_paths`
- `DOY_CMAP`, `DOY_MAX` — cyclic day-of-year colormap (`hsv`) and its upper bound (366)
- `JOINT_VAR_TITLES`, `JOINT_VAR_AXIS_LABELS`, `JOINT_VAR_FORMULA_NAMES` — title fragments,
  axis labels with units, and the short names used inside the mathtext threshold formula
- `THRESHOLD_LINE_KW` — the style dict shared by the threshold line and its legend handle
- `JOINT_SEASON_POINT_COLOR`, `JOINT_SEASON_POINT_ALPHA`, `JOINT_NOWHEEL_LEGEND_ANCHOR` —
  neutral slate grey (`#6B7C8C`, α 0.30) for a season-restricted joint distribution, where
  colouring by day of year carries no information, plus the legend anchor that replaces the
  dropped month wheel
- `COMPOUND_VAR_DISPLAY_NAMES` — full display names for the frequency-evolution titles
- `FREQ_AXIS_TEMPLATE` and `freq_axis_label(season_months)` — the Figure 1 y-axis,
  `Compound Extremes ({unit})`, with the unit from `cfg.rate_unit_label`
- `SN_AXIS_LABEL` — the Figure 2 y-axis, `Signal-to-noise-ratio (unitless)`
- `FREQ_MAIN_COLOR`, `FREQ_BAND_COLOR`, `FREQ_ENV_COLOR`, `FREQ_FIG_DPI` — the
  frequency-evolution palette; the mean line reuses `MODEL_COLORS["cesm2_le"]`
- `FREQ_BAND_LABELS`, `FREQ_ENV_LABELS` — legend wording per `spread_method`, naming which
  flavour of percentile is on screen
- `FREQ_FIGSIZE`, `FREQ_AXES_BOX`, `FREQ_LEG_TOP`, `FREQ_LEG_LEFT`, `FREQ_SEL_LEFT`,
  `FREQ_SEL_INDENT`, `FREQ_LEG_FS`, `FREQ_SEL_FS`, `FREQ_LINE_GAP` — named geometry of the
  two-column legend block below the axes, so the figure functions hold no magic numbers

### 7.2 Return-period and evaluation figures

- `make_figure(da, catchment_title, ...)` — 2-panel time series + return period for one
  catchment
- `make_smile_return_period_figure(...)` — the SMILE return-period figure
- `make_distribution_figure(annual_maxima, window_days, ...)` — density plus boxplot for all
  models
- `make_qq_figure(climate_key, climate_data, reanalysis, ...)` — Q-Q plot SMILE vs
  reanalysis

### 7.3 Map helpers

- `draw_catchments(ax, catchments, data_crs, ...)` — plain catchment outlines
- `draw_catchments_numbered(ax, catchments, data_crs, catchment_numbers, ...)` — outlines with
  circled numbers
- `_finite_max_abs(values, fallback=1.0)` — `max(|finite values|)` or the fallback. Guards the
  diverging-norm scaling against an all-NaN difference field
- `round_up_nice(value)` — round up to a clean colorbar bound
- `colorbar_label(window_days)`, `title_text(combo)`, `make_colorbar_ticks(vmax)`,
  `compute_vmax_by_window(event_fields)` — labels, ticks and the fixed colorbar maxima for
  the Storm Hans event maps
- `plot_precip_map(combo, da_evt, catchments, vmax, out_paths, ...)` — one Storm Hans event
  map
- `plot_single_catchment_weight_map(combo, catchment_slug, catchment_title, da_w, catchment_gdf, out_paths)`
  — one weight-fraction map
- `_plot_annmedian_panel(ax, da, dataset_type, ...)` — one panel of an annual median map;
  returns the mesh for the shared colorbar
- `_plot_diff_panel(ax, da, panel_title, catchments, norm, ...)` — one diverging difference
  panel
- `_add_pixel_hatch_overlay(ax, lons, lats, mask, hatch, transform, ...)` — draw the hatch
  over every `True` cell as individual `Rectangle` patches via a `PatchCollection`, which
  avoids the half-pixel artefacts of `contourf`-based hatching
- `plot_annual_median_4panel(da_cesm2, da_senorge, da_era5_05, da_era5_025, ...)` — 2×2
  annual median map
- `plot_window_median_2panel(da_cesm2, da_era5, ...)` — 2-panel N-day median
- `plot_window_interp_3panel(da_cesm2, da_era5_interp, da_diff, ..., sig_cesm_higher, sig_era5_higher, sig_legend_text)`
  — CESM2-LE / ERA5-interp / difference with two colorbars and optional significance
  hatching, plus a combined catchment and significance legend
- `plot_window_interp_diffonly(da_diff, ...)` — the single-panel difference map
- `plot_window_interp_diffonly_sig(da_diff, ..., sig_cesm_higher, sig_era5_higher, sig_legend_text)`
  — the same with significance hatching
- `plot_window_interp_seasonal_4row_3col(seasonal_data, catchments, ...)` — 4 rows
  (DJF/MAM/JJA/SON) × 3 columns, season labels vertical on the left, column labels on the top
  row, colorbars and legend under the bottom row, one shared diverging norm

### 7.4 Compound figures

- `add_month_color_wheel(fig, rect)` — circular Jan–Dec day-of-year legend, January at the
  top, running clockwise
- `threshold_formula_mathtext(x_variable, y_variable, threshold)` — the mathtext
  `x/max(x) + y/max(y) ≥ threshold`, used by both the legend and the selection lines
- `add_absolute_threshold_legend(fig, x_variable, y_variable, threshold, anchor)` — the
  threshold legend entry under the month wheel
- `make_joint_distribution_figure(x_vals, y_vals, doy_vals, x_variable, y_variable, window_days, start_year, end_year, catchment_title, n_members, out_paths, threshold=None, x_norm_max=None, y_norm_max=None, season_label=None, single_season=False)`
  — scatter of two catchment-averaged window quantities, one point per (member, date), with
  a rasterized point layer. With `single_season=False` the points are coloured by day of year
  and the month wheel is drawn. With `single_season=True` the cloud is already restricted to
  one season, so all points get `JOINT_SEASON_POINT_COLOR` and the wheel is dropped. Passing
  `threshold` plus `x_norm_max` / `y_norm_max` adds the dashed criterion line and its legend
- `frequency_selection_lines(x_variable, y_variable, threshold, window_days, roll_years, n_members, norm_ref, season_label, extra=None)`
  — the right-hand legend column of the frequency-evolution figures as (bold key, value)
  pairs. `Reference:` shows the years only; the season line is dropped when the season is
  `all`
- `_frequency_figure(fig_title, y_label=None)` / `_finish_frequency_figure(fig, handles, labels, selection_lines, out_paths)`
  — the shared axes skeleton and the two-column legend block plus PDF save
- `plot_internal_variability_trend(window_centres, f_mean, band_lo, band_hi, f_min, f_max, ..., roll_years, spread_show, spread_method, p025, p975, season_months, selection_lines, out_paths)`
  — Figure 1, `internal_variability_trend_*.pdf`. The ensemble mean is the central line.
  `spread_show` adds the blue IQR band and either the min/max or the 2.5/97.5 % envelope;
  `spread_method` selects the flavour of the percentile columns and the legend wording;
  `season_months` sets the y-axis unit
- `plot_signal_to_noise_ratio(window_centres, s_to_n, x_variable, y_variable, catchment_title, start_year, end_year, selection_lines, out_paths)`
  — Figure 2, `signal_to_noise_*.pdf`; one line of `f_mean / sigma` per rolling window with
  the same legend block as Figure 1

---

## 8. `helper/generate_weights.py`

Weight generation just run once. Computes the per-cell area
fraction of each catchment on each model grid and writes one NetCDF per
catchment × dataset

- `run_era5_025()`, `run_gfdl_spear()`, `run_cesm2_le()` — the three entry points, also
  reachable from the command line via `--dataset`; `REGISTRY` maps the names to them
- `build_weights(geojson_path, ...)`, `save_weight_nc(weights, ...)`, `_run_weight_loop(dataset, ...)`
  — the shared machinery
- `_dissolve_geojson(path)` / `_dissolve_geojson_union(paths)`, `_build_projector(dst_epsg)`,
  `_find_one_smile_file(directory, pattern)` — geometry and grid helpers
- `COMBINED_CATCHMENTS` — the union catchments, currently `regine_drammen_glomma` =
  Drammen ∪ Glomma. Their GeoJSONs are dissolved into one polygon with `unary_union` before
  the area fractions are computed, so the shared border is not double-counted and cells fully
  inside the union get weight 1

Existing weight files are skipped, so the functions are safe to re-run. 

---

## 9. `helper/test_grouped_percentile.py`

Reference unit test for `catchment_tools.grouped_percentile` with grouped percentiles. Run it with
`python helper/test_grouped_percentile.py` from the repository root.
It locks the function against the 2002–2011 CESM2-LE window: 90 members, L = 10, rate quantum w = 0.1, value counts
`{0.0: 19, 0.1: 30, 0.2: 19, 0.3: 9, 0.4: 6, 0.5: 4, 0.6: 0, 0.7: 2, 0.8: 0, 0.9: 1}`.

- `test_reference_sample_is_well_formed` — 90 members, mean exactly 0.18
- `test_grouped_percentile_reference_values` — p2.5 = 0.0059, p25 = 0.0617, p50 = 0.1367,
  p75 = 0.2474, p97.5 = 0.6875 (tolerance 1e-3). p50 is tested as a property of the function,
  although the analysis never computes a median
- `test_grouped_percentile_accepts_array_q` — scalar and array `q` agree, shape preserved
- `test_does_not_fall_back_to_np_percentile` — `np.percentile` still returns the staircase and
  the grouped values differ from it
- `test_zero_bin_is_half_width` — the half-width zero bin keeps p2.5 ≥ 0, while
  `zero_half_bin=False` returns a negative rate
- `test_empty_bins_are_skipped` — p97.5 lands in the 0.7 bin only when the empty 0.6 and 0.8
  bins are skipped
- `test_unsorted_input` — shuffled and sorted input agree
- `test_degenerate_single_value` — with all members identical the result stays inside that
  value's bin
- `test_bin_width_is_not_hardcoded` — L = 15 behaves like L = 10 under rescaling
- `test_ordering_never_inverts` — p2.5 ≤ p25 ≤ p50 ≤ p75 ≤ p97.5 and all values ≥ 0

---

## 10. Notebooks (`code/`)

The notebooks contain the code execution commands only: imports, parameter blocks and calls into
`helper/`.

### 10.1 `load_data_store_postprocessed.ipynb`

Builds every postprocessed cache the analysis notebooks read. Run once, and again after the
raw data or the grids change. Produces no figures.

1. Setup — thread environment variables, `sys.path`, `import config_paths as cfg`
2. Catchment weights — `generate_weights.run_era5_025` / `run_gfdl_spear` / `run_cesm2_le`.
   The ERA5 0.5° and seNorge weight files come from an older pipeline and are assumed to
   exist
3. Overall precipitation caches — `save_era5_overall` (0.5° and 0.25°), `save_senorge_overall`,
   `save_smile_overall` for both SMILE models
4. ERA5-interpolated precipitation cache — `save_era5_interpolated_overall`
5. SWE and soil-moisture daily caches — `save_cesm2_le_field_overall` per member and
   `save_era5_interpolated_field_overall`, for `kind ∈ {swe, soil_moisture}` over the full
   record
6. N-day snowmelt caches — `WINDOW_DAYS_SWE` (2/3/4) →
   `save_cesm2_le_field_diff_overall` and `save_era5_interpolated_field_diff_overall` with
   `rolling_melt`. Saved as `…_swe_{N}day_…` next to the raw 1-day caches; run once per
   window
7. CESM2-LE catchment compound series — `WINDOW_DAYS_COMPOUND` →
   `save_cesm2_le_catchment_field_series` for each `cfg.COMPOUND_CATCHMENTS` slug ×
   {precipitation, soil_moisture, snowmelt}, giving full-record `[member, time]` caches over
   the 90 common members. Currently only the 2-day series exist on disk. This is the cell
   that both compound cells of `compound_flood_risk_analysis.ipynb` name in their error
   messages. Its inputs (steps 3 and 5) are complete for all members, so a new window needs
   no raw reload

### 10.2 `analysis_return_hans.ipynb`

Return-period analysis of Storm Hans for the reanalysis datasets and for the two SMILEs.
Saves Figures to `figures/timeseries_return_hans/`.

1. Setup and dataset selection — `DATASET_KEY` (`era5_0.5` / `era5_0.25` / `senorge`),
   `WINDOW_DAYS`, the optional year range and `FORCE_RECOMPUTE`
2. Configuration check — resolves every path, reports the available year range from the raw
   files (or from the cache filenames when the raw data is not reachable) and lists which
   catchment caches already exist
3. Run — `catchment_tools.run_all` over the five catchments
4. SMILE run table — `SMILE_RUN_TABLE` lists (dataset, window, start, end) combinations
5. SMILE run — `catchment_tools.run_all_smile` for each entry

### 10.3 `climate_model_evaluation.ipynb`

Compares CESM2-LE and GFDL-SPEAR against ERA5 and seNorge over certain period (1985–2024). Saves Figures to
`figures/climate_model_evaluation/`.

1. Setup — paths and the figure directories
2. Data loading — `load_annual_maxima_per_catchment` and `load_daily_values_per_catchment`
   for the 1-day and 2-day windows
3. Distribution figures — `make_distribution_figure` per catchment × window × data type →
   `{data_type}_distribution_{N}day_{slug}_{start}-{end}.pdf`
4. Q-Q plots — `make_qq_figure` per catchment × window × data type × model
5. Percentile mapping tables — `build_percentile_mapping_table`, written as CSV
6. Distribution summary tables — `build_distribution_summary_table`, written as CSV

### 10.4 `create_precip_maps_hans.ipynb`

Storm Hans event maps and the precipitation climatology comparison maps. The accumulation
window of all climatology maps is set once by `WINDOW_DAYS`, with `WIN` / `WLABEL` derived
from it for filenames and titles. Writes to `figures/precip_maps_hans/`.

1. Setup — constants (`MAP_EXTENT`, `MAP_START`, `MAP_END`, `ANN_VMAX`, `TWODAY_VMAX`,
   `CATCHMENT_NUMBERS`, `COMBINATIONS`), the notebook-local helpers `combo_key` and
   `build_output_paths`, and the dependency callables `_open`, `_roll`, `_sub`
2. Event loader — `load_event_field`, which reads one year file, crops it, applies the
   rolling window and takes the pixel-wise maximum over the 7–9 Aug 2023 envelope
3. Event maps — `run_all_precip_maps` calls `plot_precip_map` for all six
   dataset × window combinations
4. Weight maps — `WEIGHT_COMBINATIONS` loops `cfg.CATCHMENTS` ∪ `cfg.COMPOUND_CATCHMENTS`
   and calls `plot_single_catchment_weight_map`. The Drammen ∪ Glomma outline is built here
   with `unary_union`; dataset × catchment pairs without a weight file are skipped
5. 4-panel annual median — `compute_era5_annual_median_2d`,
   `compute_senorge_annual_median_2d`, `compute_cesm2_le_annual_median_2d`,
   `plot_annual_median_4panel`
6. 2-panel N-day median — `compute_era5_window_median_2d`,
   `compute_cesm2_le_window_global_median_2d`, `plot_window_median_2panel`
7. N-day median difference vs ERA5-interpolated —
   `compute_era5_interpolated_window_median_2d`, `plot_window_interp_3panel`,
   `plot_window_interp_diffonly`
8. 90th-percentile difference — `compute_cesm2_le_window_per_member_p90_2d`,
   `compute_era5_interpolated_window_p90_2d`, same two plot functions
9. Significance-hatched median difference — loads the per-member median cache, calls
   `compute_significance_masks` at 5/95 and 2/98, plots via `plot_window_interp_3panel`
10. Significance-hatched 90th-percentile difference — the same with the per-member p90 cache
11. Diagnostic pixel table — CESM2-LE vs ERA5-interp values, no PDF
12. Seasonal computation — the four seasonal `compute_*` functions for all four seasons,
    stored in `SEASONAL_MEDIAN_DATA` / `SEASONAL_P90_DATA`
13. Seasonal significance plots — `_build_seasonal_list` calls `compute_significance_masks`
    per season, then `plot_window_interp_seasonal_4row_3col` four times (5/95 and 2/98, median
    and p90)
14. Spring-only 3-panel significance — self-contained; computes the seasonal p90 fields for
    `SPRING` (currently `"MAMJ"`), then `compute_significance_masks` at 2/98 and
    `plot_window_interp_3panel` → `{N}daymedian_MAMJ_90pctl_precip_2_98pctl_diff_*.pdf`. The
    layout matches the annual figure; only the diverging colorbar range differs. Any
    `cfg.SEASON_MONTHS` key works, and the first run for a new key builds its caches from the
    per-member daily caches
15. Seasonal diagnostic — per-season maxima and pixel tables for colorbar calibration, no PDF
16. Catchment-3 diagnostic — grid-cell values by season for Losna, median and 90th percentile

### 10.5 `compound_flood_risk_analysis.ipynb`

The compound flood drivers analysis part.

The first half repeats the map methodology of `create_precip_maps_hans.ipynb` for
**snowmelt** (the N-day SWE decrease, stored as `max(0, −ΔSWE)`) and **soil moisture**
(N-day rolling mean): N-day median, 90th percentile, percentile-rank significance at 5/95 and
2/98, and the seasonal breakdown. Units are kg/m². Only 90 of the 100 CESM2-LE members carry
SWE and SM output, so the ensemble size is detected automatically. The `annualmedian_*` and
2-panel overview figures are left out, because they would need seNorge and native ERA5, which
do not exist for these variables. The filename prefix is
`FSTEM = f"{acc_tag(WINDOW_DAYS)}median"`.

The second half is the compound analysis joint distribution of two catchment-
averaged window quantities and the evolution of the compound exceedance frequency.

1. Setup — `WINDOW_DAYS` with the derived `WLABEL` / `FSTEM`, the `VARIABLES` configuration
   (per-variable roller: `rolling_identity` for snowmelt, since the difference is
   pre-computed on disk, and `rolling_mean` for soil moisture; per-variable `daily_window`:
   SWE → `WINDOW_DAYS`, soil moisture → 1), the `_open_field` / `figp` helpers and the
   `RECOMPUTE` flag
2. Cache guard — `_require_swe_window_caches(WINDOW_DAYS)` fails early with a pointer to
   `load_data_store_postprocessed.ipynb` when the N-day snowmelt cache is missing. Soil
   moisture rolls its 1-day cache on the fly, so only SWE is checked
3. Seasonal cache builder — the seasonal median and p90 computations for both variables
4. N-day median and 90th-percentile 3-panel and diff-only maps
5. Significance-hatched differences at 5/95 and 2/98
6. Diagnostic pixel tables, no PDF
7. Seasonal 4×3 significance plots
8. Spring-only 3-panel significance — standalone, `SPRING = "MAMJ"`, variables chosen with
   `SPRING_VARS` (default `["snowmelt"]`)
9. Seasonal diagnostic for colorbar calibration, no PDF
10. Catchment-3 (Losna) per-member strip plots → `diagnostic_catchment3_*`
11. Joint distribution — the `JD_*` selection block (`JD_CATCHMENT`, `JD_COMBO`,
    `JD_WINDOW_DAYS`, `JD_START` / `JD_END`, `JD_MEMBERS`, `JD_SEASON`) →
    `load_cesm2_le_catchment_field_series` → `subset_season` →
    `make_joint_distribution_figure`. `JD_SEASON = "all"` keeps the day-of-year colours and
    the month wheel; any season sets `single_season=True`, so the points are drawn in one
    neutral grey and the wheel is dropped →
    `joint_distribution_{N}day_{var1}_{var2}_{catchment}_{start}-{end}[_{season}].pdf`
12. Joint distribution with the threshold line — reuses the arrays and the whole `JD_*`
    selection of the previous cell and adds `JD_THRESHOLD`. `compound_threshold_stats`
    supplies the `x_max` / `y_max` denominators for both the printed statistics and the drawn
    line → `joint_distribution_…_thr{JD_THRESHOLD}.pdf`
13. Frequency evolution, selection and computation — the `FE_*` block (catchment, combo,
    window, period, members, threshold, rolling-window length and step, season, the frozen
    `FE_NORM_REF` / `FE_NORM_REF_MEM`, and the figure options `FE_SPREAD_SHOW`,
    `FE_SPREAD_METHOD`, `FE_SAVE_CSV`) → `run_compound_frequency_evolution` →
    `print_frequency_evolution_summary` → optionally `write_frequency_evolution_outputs`. It
    reuses the same caches, window operators, member parser and severity definition as items
    11–12; the difference is that `max(x)` and `max(y)` are frozen on `FE_NORM_REF` and on
    `FE_SEASON`, so the criterion line comes from exactly the months the exceedances are
    counted in. Two filename stems are built: `_fe_stem1` and `_fe_stem2`.
    Known limitations: no declustering, so an event is an exceedance day and one storm can
    count up to `FE_WINDOW_DAYS` times; and the rolling windows overlap, so consecutive
    points are not independent and the curve acts as a display-only low-pass filter
14. Figure 1 — `plot_internal_variability_trend`, *"Compound ({X} and {Y}) Hazard Frequency
    for {catchment} from {start}-{end}"*. The cell looks the columns up with
    `frequency_spread_columns(_fe_c["spread_method"])` rather than hardcoding them
15. Figure 2 — `plot_signal_to_noise_ratio`, *"Signal-to-noise-ratio Compound ({X} and {Y})
    Hazard for {catchment} from {start}-{end}"*. It reuses the `FE_SELECTION` legend block
    built in the Figure 1 cell, so both legends are identical

Cache paths come from `cfg.field_daily_cache_path` and `cfg.field_window_cache_path`. Map
figure paths reuse `cfg.precip_map_figure_paths`, and the frequency-evolution outputs use
`cfg.compound_freq_stem` and `cfg.compound_freq_figure_paths`. 

**Frequency-evolution data outputs** (written when `FE_SAVE_CSV = True`, next to the PDFs in
`figures/compound_flood_risk_output/frequency_evolution/`):

- `{stem}_ensemble.csv` — window_start / end / centre, n_members, n_events_total,
  mean_events_per_member, f_mean, sigma, min, max, rate_quantum, p25, p75, p025, p975,
  p025_grouped, p25_grouped, p75_grouped, p975_grouped, signal_to_noise
- `{stem}_metadata.json` — the full configuration including `spread_method` and
  `rate_quantum`, the frozen maxima, the physical threshold, the diagnostics, the package
  versions and a timestamp

Both carry the same stem as the PDFs and also windows. The rate unit follows the
season (`cfg.rate_unit_label`): `events per year` for an all-months run, `events per season`
for any subset. The value is the same either way, since each year contributes one season, and
it is never converted to a per-decade rate, so it stays comparable when `FE_ROLL_YEARS`
changes.

