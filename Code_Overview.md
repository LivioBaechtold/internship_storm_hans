# Code Overview — Storm Hans Precipitation & Climate-Model Evaluation

Last updated after refactoring: 27. May 2026

---

## Helper Python files (`helper/`)

---

### `config_paths.py` — paths and constants only

No plotting, no data loading, no statistical logic. Everything else imports from here.

| Name | Kind | Purpose |
|---|---|---|
| `ERA5_RAW_DIR` | constant | Raw ERA5 annual NetCDF files |
| `SENORGE_RAW_DIR` | constant | Raw seNorge annual NetCDF files |
| `ERA5_INTERPOLATED_BASE` | constant | Root dir for ERA5 interpolated to CESM2-LE grid (`/scandinavia/`) |
| `ERA5_INTERPOLATED_DIR` | constant | ERA5 interpolated precipitation sub-path (`BASE / "tp"`) |
| `ERA5_INTERPOLATED_SWE_DIR` | constant | ERA5 interpolated SWE sub-path (`BASE / "sd"`) |
| `CESM2_LE_BASE` | constant | Root dir for CESM2-LE scandinavia data |
| `CESM2_LE_DIR` | constant | CESM2-LE precipitation sub-path (`BASE / "tp24"`) |
| `CESM2_LE_DIR` | constant | CESM2-LE precipitation sub-path (`BASE / "PRECT"`; variable `PRECT`, mm/day — identical values to the retired `tp24` folder) |
| `CESM2_LE_SM_DIR` | constant | CESM2-LE soil-moisture sub-path (`BASE / "SM"`) |
| `ERA5_INTERPOLATED_SWVL_DIR` | constant | ERA5 interpolated soil-moisture sub-path (`BASE / "swvl"`) |
| `field_daily_cache_path(model, kind, start, end, member_id=None, window_days=1)` | function | Per-member (CESM2-LE) / overall (ERA5-interp) daily SWE/soil-moisture cache path. `window_days=1` → raw `…_1day_…`; `window_days>=2` → pre-computed N-day snowmelt cache `…_{N}day_…` (SWE only, max(0,−ΔSWE): positive melt, gains→0) |
| `field_window_cache_path(model, kind, which, start, end, season="")` | function | Derived N-day median/p90 cache path (global or per-member; optional season) for SWE/soil moisture |
| `GFDL_SPEAR_DIR` | constant | GFDL-SPEAR raw data |
| `FIGURES_DIR`, `FIGURES_DIR_SECONDARY` | constant | Two output roots for all figures |
| `POSTPROC_DIR`, `WEIGHTS_DIR` | constant | Postprocessed cache and weight files |
| `OVERALL_PRECIP_EXTENT` | constant | (W, E, S, N) crop for all spatial caches |
| `SMILE_CONFIG` | dict | Per-model metadata: dir, n_members, unit_mode, ref_dataset |
| `CATCHMENTS` | dict | slug → human title for the 5 study catchments |
| `COMPOUND_CATCHMENTS` | dict | slug → title for the compound joint-distribution catchments (`regine_drammen`, `regine_glomma`, `regine_drammen_glomma` = dissolved union; kept separate from `CATCHMENTS` so main loops are unchanged) |
| `HANS_DATE`, `HANS_SEARCH_YEAR` | constant | Storm Hans event reference |
| `GEOJSON_FILES` | dict | slug → GeoJSON filename (single canonical definition) |
| `MODEL_COLORS`, `MODEL_LABELS`, `MODEL_ORDER` | dict/list | Consistent colours/labels/order across all evaluation figures. Defined here to avoid a circular import between `catchment_tools` and `plot_style`. |
| `res_tag(dataset, resolution)` | function | Filesystem-safe `era5_0.5x0.5` style tag |
| `acc_tag(window_days)` | function | `"1day"` / `"2day"` / `"3day"` … filename label |
| *(N-day path builders)* | note | All renamed figure-path / cache-path builders (`median_precip_*`, `p90_precip_*`, `*_seasonal_*`, `cesm2_window_*`, `era5_interp_window_*`, `field_window_cache_path`) take a `window_days` argument and embed `acc_tag(window_days)` in the filename (e.g. `3daymedian_…`). For `window_days=2` the filenames are identical to the legacy `2day` names, so existing caches/figures remain valid. |
| `postproc_dir(dataset)` | function | Dataset subdirectory for grid-level caches |
| `catchment_postproc_path(...)` | function | Full path for a catchment-averaged NetCDF cache |
| `figure_paths(...)` | function | Both figure-root paths for `timeseries_return_hans` PDFs |
| `smile_member_postproc_path(...)` | function | SMILE per-member catchment cache path |
| `smile_yearmax_stats_path(...)` | function | SMILE pooled annual-max stats cache path |
| `cesm2_catchment_field_path(window_days, slug, variable, start, end)` | function | Catchment-averaged CESM2-LE compound-series cache — ONE file per (window, catchment, variable) with ALL common members `[member, time]`; variable ∈ {precipitation, soil_moisture, snowmelt} |
| `smile_figure_paths(...)` | function | Both figure-root paths for SMILE return-period PDFs |
| `overall_precip_path(dataset, resolution, start, end)` | function | Spatial daily-precip cache path (ERA5 / seNorge) |
| `overall_precip_member_path(dataset, member_id, start, end)` | function | Spatial daily-precip cache path (one SMILE member) |
| `precip_map_figure_paths(fig_subdir, fname)` | function | Both figure-root paths for any precipitation map PDF |
| `annmedian_precip_paths(fig_subdir, start, end)` | function | Paths for the 4-panel annual median map |
| `median_precip_paths(fig_subdir, start, end)` | function | Paths for 2-panel N-day median figure (CESM2-LE vs ERA5 0.5°) — `median_precip_*.pdf` |
| `median_precip_diff_paths(...)` | function | Paths for 3-panel N-day median diff (CESM2-LE vs ERA5-interp) — `median_precip_diff_*.pdf` |
| `median_precip_diffonly_paths(...)` | function | Paths for single-panel N-day median diff — `median_precip_diffonly_*.pdf` |
| `median_precip_sig_diff_paths(fig_subdir, start, end, pctl_tag)` | function | Paths for significance-hatched 3-panel N-day median diff — `median_precip_{pctl_tag}_diff_*.pdf` |
| `p90_precip_diff_paths(...)` | function | Paths for 3-panel 90th-pctile N-day diff — `median_90pctl_precip_diff_*.pdf` |
| `p90_precip_diffonly_paths(...)` | function | Paths for single-panel 90th-pctile diff — `median_90pctl_precip_diffonly_*.pdf` |
| `p90_precip_sig_diff_paths(fig_subdir, start, end, pctl_tag)` | function | Paths for significance-hatched 3-panel 90th-pctile diff — `median_90pctl_precip_{pctl_tag}_diff_*.pdf` |
| `era5_interp_window_median_cache_path(start, end)` | function | ERA5-interpolated N-day median cache |
| `era5_interp_window_p90_cache_path(start, end)` | function | ERA5-interpolated N-day 90th-pctile cache |
| `cesm2_annmedian_cache_path(start, end)` | function | CESM2-LE 100-member annual-median cache |
| `cesm2_window_annmedian_cache_path(window_days, start, end)` | function | CESM2-LE 100-member N-day-median cache |
| `cesm2_window_p90_cache_path(start, end)` | function | CESM2-LE 100-member N-day 90th-pct cache (global p90 reused by `compute_cesm2_le_window_per_member_p90_2d`) |
| `cesm2_window_global_median_cache_path(start, end)` | function | Cache path for the CESM2-LE true global N-day median (all members × time) |
| `cesm2_window_per_member_medians_cache_path(start, end)` | function | Cache path for stacked per-member N-day medians `[n_members, lat, lon]` |
| `cesm2_window_per_member_p90_cache_path(start, end)` | function | Cache path for stacked per-member N-day 90th-pctile fields `[n_members, lat, lon]` |
| `SEASONS_ORDER`, `SEASON_LABELS`, `SEASON_MONTHS` | constants | Season ordering, display labels, and month lists for DJF/MAM/JJA/SON |
| `era5_interp_window_seasonal_median_cache_path(season, start, end)` | function | Cache for ERA5-interp seasonal N-day median |
| `era5_interp_window_seasonal_p90_cache_path(season, start, end)` | function | Cache for ERA5-interp seasonal N-day 90th-pctile |
| `cesm2_window_seasonal_global_median_cache_path(season, start, end)` | function | Cache for CESM2-LE seasonal global median `[lat, lon]` |
| `cesm2_window_seasonal_per_member_medians_cache_path(season, start, end)` | function | Cache for CESM2-LE per-member seasonal medians `[n_members, lat, lon]` |
| `cesm2_window_seasonal_global_p90_cache_path(season, start, end)` | function | Cache for CESM2-LE seasonal global 90th-pctile `[lat, lon]` |
| `cesm2_window_seasonal_per_member_p90_cache_path(season, start, end)` | function | Cache for CESM2-LE per-member seasonal 90th-pctile `[n_members, lat, lon]` |
| `median_seasonal_precip_sig_diff_paths(fig_subdir, start, end, pctl_tag)` | function | Paths for 4×3 seasonal significance-hatched N-day median diff — `{N}daymedian_seasonal_precip_{pctl_tag}_diff_*.pdf` |
| `p90_seasonal_precip_sig_diff_paths(fig_subdir, start, end, pctl_tag)` | function | Paths for 4×3 seasonal significance-hatched 90th-pctile diff — `{N}daymedian_seasonal_90pctl_precip_{pctl_tag}_diff_*.pdf` |


---

### `data_era5.py` — ERA5 file discovery, loading, and spatial-cache builders

| Name | Kind | Purpose |
|---|---|---|
| `find_era5_files(era5_dir, resolution)` | function | Sorted list of all ERA5 annual files for a resolution |
| `get_year_range(files, resolution)` | function | `(start_year, end_year)` from a file list |
| `day_start(timestamp)` | function | Normalise a timestamp to midnight |
| `select_time_range_by_day(da, start, end)` | function | Day-level time selection (robust to sub-daily timestamps) |
| `select_single_time_by_day(da, day)` | function | Select one day from a DataArray |
| `coord_slice(coord, lo, hi)` | function | `slice()` that works for both ascending and descending axes |
| `pick_year_file(files, year)` | function | Return the single ERA5 file covering a given year |
| `save_era5_overall(resolution, era5_dir, out_path_fn, extent, force)` | function | Build and save the full-period spatially-cropped ERA5 daily cache |
| `compute_era5_annual_median_2d(resolution, start, end, era5_dir, cache_path_fn, open_cache_fn)` | function | Annual median map from ERA5 overall cache |
| `compute_era5_window_median_2d(resolution, start, end, ...)` | function | N-day rolling median map from ERA5 overall cache |
| `find_era5_interpolated_files(era5_interp_dir)` | function | Sorted list of ERA5-interpolated-to-CESM2-LE annual files; variable prefix matched generically (`tp`/`sd`/`swvl`) so one function serves precip, SWE and soil moisture |
| `get_year_range_era5_interp(files)` | function | `(start_year, end_year)` from ERA5-interpolated file list |
| `save_era5_interpolated_field_overall(era5_interp_dir, out_path_fn, variable, cache_var, units, extent, force)` | function | Build/save ERA5-interpolated daily SWE (`sd`) / soil-moisture (`swvl`) cache; no metre→mm conversion |
| `save_era5_interpolated_field_diff_overall(era5_interp_dir, in_path_fn, out_path_fn, cache_var, diff_fn, open_cache_fn, window_days, units, force)` | function | Build the ERA5-interp daily N-day snowmelt cache from the 1-day cache via `diff_fn` (max(0,−ΔSWE): positive melt, gains→0) over the full record |
| `compute_era5_interpolated_window_median_2d(start, end, ...)` | function | N-day rolling median from ERA5-interpolated cache |
| `compute_era5_interpolated_window_p90_2d(start, end, ..., p90_cache_path, ...)` | function | 90th percentile of N-day rolling sums from ERA5-interpolated; caches result; reads cache by trying both `twoday_p90_precip` and legacy `twodayp90_precip` variable names |
| `_SEASON_MONTHS` | constant | Module-level dict mapping season abbreviation → month list |
| `_filter_season_da(da, season)` | function | Select time steps in a DataArray whose month belongs to the given season |
| `compute_era5_interpolated_window_seasonal_median_2d(season, start, end, ...)` | function | Seasonal median of N-day rolling sums from ERA5-interpolated cache; caches result |
| `compute_era5_interpolated_window_seasonal_p90_2d(season, start, end, ...)` | function | Seasonal 90th percentile of N-day rolling sums from ERA5-interpolated cache; caches result |


---

### `data_senorge.py` — seNorge file discovery, loading, and spatial-cache builders

| Name | Kind | Purpose |
|---|---|---|
| `find_senorge_files(senorge_dir)` | function | Sorted list of all seNorge annual files |
| `get_year_range_senorge(files)` | function | `(start_year, end_year)` from a file list |
| `load_senorge_precipitation(files, ...)` | function | Open and mask seNorge `rr` variable |
| `latlon_extent_to_utm_bbox(extent)` | function | Convert (W, E, S, N) degrees to UTM-33 bounding box in metres |
| `save_senorge_overall(senorge_dir, out_path_fn, extent, force)` | function | Build and save the full-period spatially-cropped seNorge daily cache |
| `compute_senorge_annual_median_2d(start, end, senorge_dir, cache_path_fn, open_cache_fn)` | function | Annual median map from seNorge overall cache |
| `compute_senorge_2day_median_2d(start, end, ...)` | function | 2-day rolling median map from seNorge overall cache |

---

### `data_smile.py` — SMILE model file discovery, loading, and spatial-cache builders

Supports `cesm2_le` and `gfdl_spear_med_le`.

| Name | Kind | Purpose |
|---|---|---|
| `find_smile_members(model_dir, dataset)` | function | Sorted list of unique member IDs; for cesm2_le the variable token (PRECT/SWE/SM) is matched generically and a mixed-variable directory raises an error |
| `find_smile_files_for_member(model_dir, dataset, member_id, ...)` | function | Chronological file list for one ensemble member |
| `_convert_smile_tp24_to_mm(da, unit_mode)` | function | m→mm conversion with auto-detection from metadata |
| `load_smile_precipitation(files, start, end, unit_mode)` | function | Open, convert (PRECT or tp24 → mm), deduplicate, sort one member's precipitation |
| `get_year_range_smile(model_dir, dataset)` | function | Full `(start_year, end_year)` available on disk |
| `load_cesm2_le_field(files, variable, start, end)` | function | Open/concatenate a CESM2-LE state variable (`SWE`/`SM`), no unit conversion; same sort/dedup robustness as `load_smile_precipitation` |
| `save_cesm2_le_field_overall(model_dir, out_path_fn, variable, cache_var, units, force)` | function | Build per-member daily SWE/soil-moisture caches (counterpart of `save_smile_overall`, kg/m²) |
| `save_cesm2_le_field_diff_overall(model_dir, in_path_fn, out_path_fn, cache_var, diff_fn, open_cache_fn, window_days, units, force)` | function | Build per-member CESM2-LE daily N-day snowmelt caches from the 1-day caches via `diff_fn` (max(0,−ΔSWE): positive melt, gains→0) over the full record |
| `compute_cesm2_le_annual_median_2d(start, end, model_dir, member_cache_path_fn, median_cache_path, open_cache_fn, ...)` | function | True pooled median annual map: pools all n\_members × n\_years annual totals per pixel then takes `np.nanmedian`; caches result |
| `compute_cesm2_le_window_median_2d(start, end, ..., rolling_fn, subset_fn, ...)` | function | 100-member median N-day rolling map; caches result |
| `compute_cesm2_le_window_p90_2d(start, end, ..., p90_cache_path, rolling_fn, subset_fn, ...)` | function | 100-member median of per-member 90th-pctile N-day sums; caches result; reads cache by trying both `twoday_p90_precip` and legacy `twodayp90_precip` variable names |
| `compute_cesm2_le_window_global_median_2d(start, end, model_dir, ..., global_median_cache_path, per_member_medians_cache_path, ...)` | function | True global median of N-day rolling sums (all members × time) per pixel; simultaneously saves per-member medians `[n_members, lat, lon]` for significance testing |
| `compute_cesm2_le_window_per_member_p90_2d(start, end, model_dir, ..., per_member_p90_cache_path, global_p90_cache_path, ...)` | function | Per-member 90th-pctile N-day fields `[n_members, lat, lon]` plus global p90; saves two caches |
| `compute_significance_masks(da_era5, per_member_da, lower_pctl, upper_pctl)` | function | Direct percentile-rank significance test (no FDR): marks a pixel significant if `n_greater >= round(upper_pctl/100 * n_members)` (ERA5 ≤ lower pctile → `sig_cesm_higher`) or `n_less >= threshold` (ERA5 ≥ upper pctile → `sig_era5_higher`); NaN pixels set to False; returns `(sig_cesm_higher, sig_era5_higher)` bool arrays |
| `_SEASON_MONTHS` | constant | Module-level dict mapping season abbreviation → month list |
| `_season_time_mask(time_values, season)` | Returns boolean mask selecting timesteps in the given season's months; handles both `numpy.datetime64` and `cftime.DatetimeNoLeap` time axes | private helper |
| `compute_cesm2_le_window_seasonal_global_median_2d(season, start, end, ...)` | function | Seasonal true global median + per-member medians of N-day rolling sums; saves two caches |
| `compute_cesm2_le_window_seasonal_per_member_p90_2d(season, start, end, ...)` | function | Seasonal per-member 90th-pctile fields + global 90th-pctile; saves two caches |


---

### `catchment_tools.py` — catchment processing and non-plotting data preparation

| Name | Kind | Purpose |
|---|---|---|
| `find_weight_file(dataset, resolution, slug)` | function | Locate the area-fraction weight NetCDF for one catchment |
| `load_weights(path)` | function | Open and return a weight DataArray |
| `_spatial_dims(da)` | function | Detect `(y_dim, x_dim)` name pair from DataArray dimensions |
| `weighted_catchment_mean(da, weights)` | function | Area-fraction weighted spatial mean over a catchment |
| `rolling_accumulation(da, window_days)` | function | Rolling sum over the time dimension — works for 1-D series and 2-D spatial fields |
| `subset_time_series_by_year(da, start, end)` | function | Year-range clip for any DataArray |
| `_chunksizes_for(da, time_chunk, y_chunk, x_chunk)` | function | NetCDF chunksizes aligned to da.dims order |
| `save_spatial_netcdf(da, out_path, var_name, ..., units="mm")` | function | Write a float32 compressed chunked NetCDF — used by all `save_*_overall` functions; `units` defaults to `mm` (pass `kg/m2` for SWE/soil moisture) |
| `open_precip_cache(cache_path, start_year, end_year)` | function | Open a spatial precipitation cache lazily (`tp24_mm`/`rr_mm`); year-subsets it |
| `open_field_cache(cache_path, var_name, start_year, end_year)` | function | Generic counterpart of `open_precip_cache` for an explicitly named SWE/soil-moisture variable |
| `rolling_change(da, window_days, var_name)` | function | Trailing 'last minus first' change over the time window — `da(t) − da(t−(window−1))`; SWE counterpart of `rolling_accumulation` (units inherited, not forced to mm) |
| `rolling_melt(da, window_days, var_name)` | function | Snowmelt magnitude over the trailing window — `max(0, −(da(t) − da(t−(window−1))))`; only SWE decreases count, as a positive melt flux (gain → 0). Used to pre-build the daily N-day snowmelt caches in load_data_store_postprocessed.ipynb |
| `rolling_identity(da, window_days, var_name)` | function | Pass-through 'rolling' op for fields whose window quantity is already stored on disk (N-day ΔSWE cache); returns `da` unchanged, `window_days` ignored |
| `rolling_mean(da, window_days, var_name)` | function | Trailing rolling mean over the window — mean of `da` over `[t−(window−1), t]`; soil-moisture counterpart of `rolling_accumulation` (units inherited, not forced to mm) |
| `common_cesm2_le_members()` | function | Member IDs present in all three CESM2-LE variable dirs (PRECT ∩ SM ∩ SWE = 90; SM/SWE lack odd members 001–019) |
| `save_cesm2_le_catchment_field_series(variable, window_days, slug, force)` | function | Build/save ONE `[member, time]` compound-series cache over the full record: per member weighted catchment mean → window op (precip → SUM, SM → MEAN, snowmelt → max(0,−ΔSWE)) |
| `parse_member_selection(selection, available)` | function | Normalise `"all"` / `"1-30"` / `[3, 4, 27]` member selections; raises with the full list of unavailable members |
| `load_cesm2_le_catchment_field_series(variable, window_days, slug, start, end, members)` | function | Load + subset a compound-series cache; informative errors name the exact notebook cell that builds/fixes a missing selection |
| `crop_weight_field_to_nonzero_bbox(da, pad_cells)` | function | Crop a weight DataArray to the bounding box of its active cells |
| `get_plot_extent_and_crs(da)` | function | Infer tight map extent and Cartopy CRS from a weight DataArray |
| `load_postproc_dataset(nc_path)` | function | Open a postprocessed grid-level Dataset |
| `build_catchment_cache(...)` | function | Build and save the catchment-averaged rolling-accumulation NetCDF cache |
| `load_catchment_cache(...)` | function | Load a catchment-averaged cache |
| `load_annual_maxima_per_catchment(window_days, start, end)` | function | Load annual maxima for all catchments and all models into a nested dict |
| `load_daily_values_per_catchment(window_days, start, end)` | function | Load all daily values for all catchments and models |
| `build_percentile_mapping_table(climate_key, climate_data, refs, percentiles)` | function | Percentile comparison table: SMILE vs reanalysis |
| `build_distribution_summary_table(annual_maxima)` | function | Summary statistics (mean, std, quantiles) for each model |
| `get_cached_year_range(dataset, resolution, window_days)` | function | Public wrapper: infer available year range from existing cache files |
| `load_catchments(geojson_files, geojson_dir)` | function | Load all catchment GeoDataFrames from GeoJSON into EPSG:4326 |

---

### `return_period.py` — pure statistical functions, no I/O

| Name | Kind | Purpose |
|---|---|---|
| `get_annual_maxima(da, start, end)` | function | Extract annual maximum values from a time series |
| `weibull_plotting_positions(n)` | function | Empirical return periods using the Weibull formula |
| `fit_gev(data)` | function | Fit GEV distribution (L-moments or MLE) |
| `gev_return_level(params, return_period)` | function | Return level for a given return period |
| `get_event_annual_max(da, event_year)` | function | Annual maximum value for the event year |
| `estimate_return_period(value, params)` | function | Estimated return period for an observed value |

---

### `plot_style.py` — all Matplotlib/Cartopy figure code

No raw data loading. No statistical computations.

**Module-level constants (available as `from plot_style import ...`):**

| Name | Kind | Purpose |
|---|---|---|
| `FIG_DPI` | constant | Single output-resolution lever (`= 150`). Every `savefig` in this module passes `dpi=FIG_DPI`; lowering it shrinks PDF file sizes (GitHub-friendly) while catchment vectors stay crisp |
| `figure.dpi` (rcParams) | constant | Set to `FIG_DPI` (150) — resolution of rasterized map content; all map functions use `ax.set_rasterization_zorder(4)` to rasterize coastlines/OCEAN/LAND/pcolormesh at this DPI, keeping catchment outlines/labels as vectors |
use `ax.set_rasterization_zorder(4)` to convert coastlines/OCEAN/LAND/pcolormesh into compact raster bitmaps at this DPI, keeping catchment vectors crisp |
| `MAP_PROJ` | constant | Lambert Conformal CRS centred on Scandinavia |
| `DATA_CRS_LATLON` | constant | PlateCarree (lat/lon) CRS |
| `DATA_CRS_SENORGE` | constant | UTM zone 33 CRS (seNorge native grid) |
| `OCEAN_COLOR` | constant | Light blue ocean background |
| `PRECIP_CMAP` | constant | IPCC sequential precipitation colormap (loaded from `prec_seq.txt`) |
| `WEIGHT_CMAP` | constant | Viridis colormap for weight fraction maps |
| `MODEL_COLORS`, `MODEL_LABELS`, `MODEL_ORDER` | imported from `config_paths` | Re-exported for convenience |
| `PRECIP_DIV_CMAP` | constant | IPCC diverging precipitation colormap (loaded from `prec_div.txt`) |
| `DOY_CMAP`, `DOY_MAX` | constant | Cyclic day-of-year colormap (`hsv`, Blöschl et al. 2017 wheel) and normalisation bound (366) |
| `JOINT_VAR_TITLES`, `JOINT_VAR_AXIS_LABELS` | dict | Title fragments and axis labels (with units) for the joint-distribution variables |


**Functions:**

| Name | Purpose |
|---|---|
| `make_figure(da, catchment_title, ...)` | 2-panel timeseries + return-period figure (Storm Hans analysis) |
| `make_smile_return_period_figure(...)` | SMILE return-period figure |
| `make_distribution_figure(annual_maxima, window_days, ...)` | Distribution density + boxplot for all models |
| `make_qq_figure(climate_key, climate_data, reanalysis, ...)` | Q-Q plot: SMILE vs reanalysis |
| `draw_catchments_numbered(ax, catchments, data_crs, catchment_numbers, ...)` | Draw catchment outlines with circled numbers on any map axes |
| `_load_ipcc_prec_seq(txt_path)` | Load IPCC precipitation colormap from 256-row RGB text file |
| `_load_ipcc_prec_div(txt_path)` | Load IPCC precipitation diverging colormap from 256-row RGB text file |
| `_finite_max_abs(values, fallback=1.0)` | Return max(\|finite values\|), or `fallback` if none are finite. Guards the diverging-norm scaling in `plot_window_interp_3panel` / `_diffonly` / `_diffonly_sig` against the `nanmax` "zero-size array to reduction operation fmax" crash when a difference field is entirely NaN (e.g. grid coords don't align, or all-zero reference field) |
| `round_up_nice(value)` | Round a value up to a visually clean colorbar upper bound |
| `colorbar_label(window_days)` | Standard colorbar label string for event precipitation maps |
| `title_text(combo)` | Standard figure title for Storm Hans event maps |
| `make_colorbar_ticks(vmax)` | Evenly-spaced tick array for precipitation colorbars |
| `compute_vmax_by_window(event_fields)` | Fixed colorbar maxima for Storm Hans 1-day and 2-day maps |
| `plot_precip_map(combo, da_evt, catchments, vmax, out_paths, ...)` | Single-panel Storm Hans event precipitation map |
| `plot_single_catchment_weight_map(combo, slug, title, da_w, gdf, out_paths)` | One weight-fraction map for a single catchment |
| `add_month_color_wheel(fig, rect)` | Circular Jan–Dec day-of-year colour legend (polar inset ring; Jan at top, clockwise Jan→Apr→Jul→Oct) |
| `make_joint_distribution_figure(x_vals, y_vals, doy_vals, x_variable, y_variable, window_days, start, end, catchment_title, n_members, out_paths)` | Joint-distribution scatter of two catchment-averaged window quantities — one point per (member, date), coloured by day of year, month-wheel legend, rasterized point layer |
| `_plot_annmedian_panel(ax, da, dataset_type, ...)` | Draw one panel in an annual median map; returns mesh for shared colorbar |
| `plot_window_median_2panel(da_cesm2, da_era5, ...)` | 2-panel annual median of N-day rolling precipitation; catchment legend centered at colorbar height |
| `plot_window_interp_3panel(da_cesm2, da_era5_interp, da_diff, ..., fig_title, seq_cbar_label, div_cbar_label, sig_cesm_higher, sig_era5_higher, sig_legend_text)` | 3-panel: CESM2-LE \| ERA5-interp \| difference — two colorbars; optional per-pixel significance hatching on ERA5 panel (uses `_add_pixel_hatch_overlay` for clean whole-cell `//` and `\\` patterns); combined catchment + significance legend between the two colorbars |
| `_add_pixel_hatch_overlay(ax, lons, lats, mask, hatch, transform, ...)` | Draw hatch pattern over every `True` cell in mask as individual `Rectangle` patches via `PatchCollection`; avoids the half-pixel artifacts of `contourf`-based hatching |
| `plot_annual_median_4panel(da_cesm2, da_senorge, da_era5_05, da_era5_025, ...)` | 2×2 panel annual median precipitation map; left-column panels anchored "W" and right-column anchored "E" to spread panels outward; `hspace=0.16`, `wspace=0.05`; colorbar at x=0.15, legend at x=0.72 |
| `_plot_diff_panel(ax, da, panel_title, catchments, norm, ...)` | Draw one difference panel using diverging colormap |
| `plot_window_interp_diffonly` | Single-panel diverging difference map (CESM2-LE – ERA5 interp). Catchment legend placed inside panel at bottom-right (`ax.transAxes`). Saves to `out_paths` and closes figure. | `da_diff`, `catchments`, `start_year`, `end_year`, `out_paths`, `fig_title`, `div_cbar_label`, `catchment_numbers`, `catchment_legend_text`, `label_overrides`, `annmedian_extent` |
| `plot_window_interp_seasonal_4row_3col(seasonal_data, catchments, ...)` | 4×3-panel seasonal significance plot: 4 rows (DJF/MAM/JJA/SON) × 3 columns (CESM2-LE \| ERA5-interp \| diff); season labels bold-vertical on left; column labels bold on top row only; colorbars + legend under bottom row only; shared diverging norm across all seasons |



---

### `generate_weights.py` — run-once script

### `generate_weights.py` — weight-generation module

Generates catchment area-fraction weight NetCDF files. Exposes `run_era5_025`,
`run_gfdl_spear`, `run_cesm2_le` plus shared helpers (`build_weights`,
`save_weight_nc`, `_run_weight_loop`, `_dissolve_geojson_union`). Now imported
and invoked from `load_data_store_postprocessed.ipynb` (still runnable as a CLI
via `--dataset`). `COMBINED_CATCHMENTS` defines union catchments
(`regine_drammen_glomma` = Drammen ∪ Glomma): their GeoJSONs are dissolved into
ONE polygon with `unary_union` before the per-cell area fractions are computed,
so the overlapping border is never double-counted (cells fully inside the union
get weight 1). Existing weight files are skipped; only re-run if catchment
boundaries or grid definitions change.


---

## Notebook files (`code/`)

---

### `load_data_store_postprocessed.ipynb`

**Purpose:** Single entry point that builds **all** postprocessed caches consumed
by the analysis notebooks. Run once (or after raw data / grids change) before
`create_precip_maps_hans.ipynb` and `compound_flood_risk_analysis.ipynb`.
**Produces:** no figures — only caches under `postprocessed/`.

**Structure (orchestration only; all logic lives in helper modules):**
1. Setup — env vars, sys.path, `import config_paths as cfg`
2. Catchment weights — `generate_weights.run_era5_025` / `run_gfdl_spear` / `run_cesm2_le` (existing files skipped; era5 0.5x0.5 + seNorge weights assumed pre-existing)
3. Overall precipitation caches — `save_era5_overall` (0.5 + 0.25), `save_senorge_overall`, `save_smile_overall` (both SMILE models)
4. ERA5-interpolated overall precipitation cache — `save_era5_interpolated_overall`
5. SWE & soil-moisture daily caches — `save_cesm2_le_field_overall` (per-member) + `save_era5_interpolated_field_overall` for `kind ∈ {swe, soil_moisture}` over the full record
6. N-day snowmelt caches — `WINDOW_DAYS_SWE` selector (2/3/4) → `save_cesm2_le_field_diff_overall` + `save_era5_interpolated_field_diff_overall` with `rolling_melt` (max(0,−ΔSWE): positive melt, gains→0); saved as `…_swe_{N}day_…` next to the raw 1-day caches. Run once per window
7. CESM2-LE catchment compound series — `WINDOW_DAYS_COMPOUND` selector (2/3/4/…) → `save_cesm2_le_catchment_field_series` for each `cfg.COMPOUND_CATCHMENTS` slug × {precipitation, soil_moisture, snowmelt}: full-record `[member, time]` caches (90 common members) at `cesm2_le/catchment_averaged/post_processed_cesm2_le_{N}day_{slug}_{variable}_1920-2034.nc`. Run once per window


`FORCE_RECOMPUTE_*` flags default to `False`; set `True` to rebuild.

---


### `analysis_return_hans.ipynb`

**Purpose:** Return-period analysis and SMILE return-period analysis for Storm Hans.
**Produces:** `timeseries_return_hans/` PDF figures.
**Structure:** Setup → parameter settings → calls to `catchment_tools`, `return_period`, `plot_style`.
**No reusable function definitions** — those all live in the helper files.

---

### `climate_model_evaluation.ipynb`

**Purpose:** Compare CESM2-LE and GFDL-SPEAR against ERA5 / seNorge reanalysis.
**Produces:** `climate_model_evaluation/` figures:
- QQ plots per catchment × model × window_days × data_type
- `daily_distribution` and `annual_max_distribution` PDFs

**Structure:**
1. Setup cell — env vars, sys.path, `import config_paths`, mkdir
2. Data-loading cell — `load_annual_maxima_per_catchment`, `load_daily_values_per_catchment`
3. Distribution figures loop — `make_distribution_figure`
4. QQ plots loop — `make_qq_figure`
5. Statistical tables — `build_percentile_mapping_table`, `build_distribution_summary_table`

---

### `create_precip_maps_hans.ipynb`

**Purpose:** Spatial precipitation maps for Storm Hans and annual mean / N-day comparison maps (window set by `WINDOW_DAYS`).
**Produces:** `precip_maps_hans/` figures.

**Structure:**
1. Setup cell — imports from all helpers; constants (`MAP_EXTENT`, `MAP_START`, `MAP_END`, `ANN_VMAX`, `TWODAY_VMAX`, `CATCHMENT_NUMBERS`, `COMBINATIONS`, etc.); **window selector `WINDOW_DAYS` (1/2/3/4) with derived `WIN`/`WLABEL` driving all climatology-map filenames and titles**; small notebook-specific helpers (`combo_key`, `build_output_paths`); dependency callables `_open`, `_roll`, `_sub`
2. Cache-builder pointer cell — overall precipitation caches are now built in `load_data_store_postprocessed.ipynb`; this cell is a no-op pointer
3. Event data loader cell — defines `load_event_field` (notebook-local, uses EVENT_DATE/ENVELOPE scope)
4. Event maps cell — defines `run_all_precip_maps` orchestration; calls `plot_precip_map` from `plot_style`
5. Run event maps cell — `run_all_precip_maps()`
6. Weight maps cell — `WEIGHT_COMBINATIONS` config; loops `cfg.CATCHMENTS` ∪ `cfg.COMPOUND_CATCHMENTS` (union outline built by dissolving Drammen+Glomma with `unary_union`; dataset×catchment combos without a weight file are skipped); calls `plot_single_catchment_weight_map` from `plot_style`
7. Run 4-panel cell — calls `compute_era5_annual_median_2d`, `compute_senorge_annual_median_2d`, `compute_cesm2_le_annual_median_2d`, `plot_annual_median_4panel`
8. Run N-day cell — calls `compute_era5_window_median_2d`, `compute_cesm2_le_window_global_median_2d` (true global median, not median-of-medians), `plot_window_median_2panel`
9. ERA5-interpolated N-day median difference cell — sets `ERA5_INTERP_TP_DIR` (overall cache built in `load_data_store_postprocessed.ipynb`), `compute_era5_interpolated_window_median_2d`, `plot_window_interp_3panel`, `plot_window_interp_diffonly`
10. 90th-percentile difference cell — `compute_cesm2_le_window_per_member_p90_2d` (saves global p90 and per-member p90 cache), `compute_era5_interpolated_window_p90_2d`, `plot_window_interp_3panel`, `plot_window_interp_diffonly`
11. Significance-hatched median diff cell — loads `cesm2_window_per_member_medians` cache, calls `compute_significance_masks` at 5/95 and 2/98 pctile, plots via `plot_window_interp_3panel` with sig overlays
12. Significance-hatched 90th-pctile diff cell — loads `cesm2_window_per_member_p90` cache, calls `compute_significance_masks` at 5/95 and 2/98 pctile, plots via `plot_window_interp_3panel` with sig overlays
13. Seasonal data computation cell — `compute_era5_interpolated_window_seasonal_median_2d`, `compute_era5_interpolated_window_seasonal_p90_2d`, `compute_cesm2_le_window_seasonal_global_median_2d`, `compute_cesm2_le_window_seasonal_per_member_p90_2d` for all 4 seasons; stores results in `SEASONAL_MEDIAN_DATA` / `SEASONAL_P90_DATA` dicts keyed by season
14. Seasonal significance plot cell — `_build_seasonal_list` helper calls `compute_significance_masks` per season then calls `plot_window_interp_seasonal_4row_3col` four times (5/95 median, 2/98 median, 5/95 p90, 2/98 p90) → four PDFs in `precip_maps_hans/`


---

### `compound_flood_risk_analysis.ipynb`

**Purpose:** Reproduce the CESM2-LE vs ERA5-interpolated comparison maps from
`create_precip_maps_hans.ipynb` for **Snowmelt** (signed N-day SWE change,
ΔSWE = SWE(t) − SWE(t−(N−1))) and **Soil Moisture** (N-day rolling mean), with the
window N set by `WINDOW_DAYS`. Same methodology: N-day median, 90th percentile,
percentile-rank significance testing (5/95, 2/98), seasonal breakdown, catchment-3
diagnostics. Units kg/m². Only 90 of the 100 CESM2-LE members carry SWE/SM output →
ensemble size auto-detected (n=90).
**Produces:** `compound_flood_risk_output/` figures (18 PDFs per variable). The
filename temporal prefix is the window-derived `FSTEM = f"{acc_tag(WINDOW_DAYS)}median"`
(e.g. `2daymedian_…`), shared by both variables.
Set: `{fstem}_{var}_diff/diffonly`, `{fstem}_90pctl_{var}_diff/diffonly`,
significance `{fstem}_{var}_{5_95pctl|2_98pctl}_diff/diffonly` (and `…_90pctl_…`),
seasonal `{fstem}_seasonal_{var}_…_diff`, and `diagnostic_catchment3_…_{var}`
strip plots. The `annualmedian_*` and 2-panel overview figures are intentionally
omitted (they need SeNorge/native-ERA5, absent for SWE/SM).

**Structure (orchestration only; loops over the two variables):**
1. Setup — `WINDOW_DAYS` selector (1/2/3/4) with derived `WLABEL`/`FSTEM`; `VARIABLES` config (per-variable roller: `rolling_identity` for snowmelt — the ΔSWE difference is pre-computed on disk — and `rolling_mean` for soil moisture; per-variable `daily_window`: SWE→`WINDOW_DAYS`, soil moisture→1), `_open_field`/`figp` helpers, `RECOMPUTE` flag
2. SWE ΔSWE window-cache availability guard — `_require_swe_window_caches(WINDOW_DAYS)` raises a clear "build it in load_data_store_postprocessed.ipynb" error if the selected N-day ΔSWE cache is missing (soil moisture rolls the 1-day raw cache on the fly, so only SWE is checked)
3. N-day median + 90th-pctile 3-panel diff + diffonly — `compute_cesm2_le_window_global_median_2d`, `compute_cesm2_le_window_per_member_p90_2d`, `compute_era5_interpolated_window_median_2d`/`_p90_2d`, `plot_window_interp_3panel`/`_diffonly`
4. Significance-hatched diff (5/95, 2/98) — `compute_significance_masks`, `plot_window_interp_3panel`/`_diffonly_sig`
5. Diagnostic pixel tables (no PDF)
6. Seasonal data computation — `compute_cesm2_le_window_seasonal_global_median_2d`/`per_member_p90_2d`, `compute_era5_interpolated_window_seasonal_median_2d`/`_p90_2d`
7. Seasonal significance 4×3 plots — `plot_window_interp_seasonal_4row_3col`
8. Seasonal diagnostic (vmax calibration, no PDF)
9. Catchment-3 (Losna) per-member strip plots → `diagnostic_catchment3_*` PDFs
10. Joint-distribution scatter — selection block (`JD_CATCHMENT`, `JD_COMBO`, `JD_WINDOW_DAYS`, `JD_START`/`JD_END`, `JD_MEMBERS`) → `load_cesm2_le_catchment_field_series` (validated, with build-it-here error messages) → `make_joint_distribution_figure` (day-of-year colours + month wheel) → `joint_distribution_{N}day_{var1}_{var2}_{catchment}_{start}-{end}.pdf` in `compound_flood_risk_output/`

Cache paths come from `cfg.field_daily_cache_path` / `cfg.field_window_cache_path`;
figure paths reuse `cfg.precip_map_figure_paths`. All heavy compute/plot helpers
are reused unchanged via variable-agnostic discovery + injected cache-openers
(`open_field_cache`) and rollers. Snowmelt now reads the PRE-COMPUTED daily N-day snowmelt cache (`max(0,−ΔSWE)`: positive melt, gains→0, built with `rolling_melt` in load_data_store_postprocessed.ipynb) via `rolling_identity` (no re-differencing), while soil moisture reads its raw 1-day cache and applies `rolling_mean` on the fly (`rolling_change` retained but unused). The `daily_window` per variable selects the daily cache label (SWE→`{N}day` snowmelt, soil moisture→`1day`). The window N is set once by `WINDOW_DAYS`; the shared filename prefix `FSTEM = f"{acc_tag(WINDOW_DAYS)}median"` (e.g. `2daymedian`) is threaded with `window_days=WINDOW_DAYS` into every compute/cache-path call.
