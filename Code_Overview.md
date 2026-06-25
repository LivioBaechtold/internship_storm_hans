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
| `field_daily_cache_path(model, kind, start, end, member_id=None)` | function | Per-member (CESM2-LE) / overall (ERA5-interp) daily SWE/soil-moisture cache path (`kind ∈ {swe, soil_moisture}`) |
| `field_2day_cache_path(model, kind, which, start, end, season="")` | function | Derived 2-day median/p90 cache path (global or per-member; optional season) for SWE/soil moisture |
| `GFDL_SPEAR_DIR` | constant | GFDL-SPEAR raw data |
| `FIGURES_DIR`, `FIGURES_DIR_SECONDARY` | constant | Two output roots for all figures |
| `POSTPROC_DIR`, `WEIGHTS_DIR` | constant | Postprocessed cache and weight files |
| `OVERALL_PRECIP_EXTENT` | constant | (W, E, S, N) crop for all spatial caches |
| `SMILE_CONFIG` | dict | Per-model metadata: dir, n_members, unit_mode, ref_dataset |
| `CATCHMENTS` | dict | slug → human title for the 5 study catchments |
| `HANS_DATE`, `HANS_SEARCH_YEAR` | constant | Storm Hans event reference |
| `GEOJSON_FILES` | dict | slug → GeoJSON filename (single canonical definition) |
| `MODEL_COLORS`, `MODEL_LABELS`, `MODEL_ORDER` | dict/list | Consistent colours/labels/order across all evaluation figures. Defined here to avoid a circular import between `catchment_tools` and `plot_style`. |
| `res_tag(dataset, resolution)` | function | Filesystem-safe `era5_0.5x0.5` style tag |
| `acc_tag(window_days)` | function | `"1day"` / `"2day"` label |
| `postproc_dir(dataset)` | function | Dataset subdirectory for grid-level caches |
| `catchment_postproc_path(...)` | function | Full path for a catchment-averaged NetCDF cache |
| `figure_paths(...)` | function | Both figure-root paths for `timeseries_return_hans` PDFs |
| `smile_member_postproc_path(...)` | function | SMILE per-member catchment cache path |
| `smile_yearmax_stats_path(...)` | function | SMILE pooled annual-max stats cache path |
| `smile_figure_paths(...)` | function | Both figure-root paths for SMILE return-period PDFs |
| `overall_precip_path(dataset, resolution, start, end)` | function | Spatial daily-precip cache path (ERA5 / seNorge) |
| `overall_precip_member_path(dataset, member_id, start, end)` | function | Spatial daily-precip cache path (one SMILE member) |
| `precip_map_figure_paths(fig_subdir, fname)` | function | Both figure-root paths for any precipitation map PDF |
| `annmedian_precip_paths(fig_subdir, start, end)` | function | Paths for the 4-panel annual median map |
| `twodaymedian_precip_paths(fig_subdir, start, end)` | function | Paths for 2-panel 2-day median figure (CESM2-LE vs ERA5 0.5°) — `2daymedian_precip_*.pdf` |
| `twodaymedian_precip_diff_paths(...)` | function | Paths for 3-panel 2-day median diff (CESM2-LE vs ERA5-interp) — `2daymedian_precip_diff_*.pdf` |
| `twodaymedian_precip_diffonly_paths(...)` | function | Paths for single-panel 2-day median diff — `2daymedian_precip_diffonly_*.pdf` |
| `twodaymedian_precip_sig_diff_paths(fig_subdir, start, end, pctl_tag)` | function | Paths for significance-hatched 3-panel 2-day median diff — `2daymedian_precip_{pctl_tag}_diff_*.pdf` |
| `twodaymedian_90pctl_diff_paths(...)` | function | Paths for 3-panel 90th-pctile 2-day diff — `2daymedian_90pctl_precip_diff_*.pdf` |
| `twodaymedian_90pctl_diffonly_paths(...)` | function | Paths for single-panel 90th-pctile diff — `2daymedian_90pctl_precip_diffonly_*.pdf` |
| `twodaymedian_90pctl_precip_sig_diff_paths(fig_subdir, start, end, pctl_tag)` | function | Paths for significance-hatched 3-panel 90th-pctile diff — `2daymedian_90pctl_precip_{pctl_tag}_diff_*.pdf` |
| `era5_interp_2day_median_cache_path(start, end)` | function | ERA5-interpolated 2-day median cache |
| `era5_interp_2day_p90_cache_path(start, end)` | function | ERA5-interpolated 2-day 90th-pctile cache |
| `cesm2_annmedian_cache_path(start, end)` | function | CESM2-LE 100-member annual-median cache |
| `cesm2_2day_annmedian_cache_path(start, end)` | function | CESM2-LE 100-member 2-day-median cache |
| `cesm2_2day_p90_cache_path(start, end)` | function | CESM2-LE 100-member 2-day 90th-pct cache (global p90 reused by `compute_cesm2_le_2day_per_member_p90_2d`) |
| `cesm2_2day_global_median_cache_path(start, end)` | function | Cache path for the CESM2-LE true global 2-day median (all members × time) |
| `cesm2_2day_per_member_medians_cache_path(start, end)` | function | Cache path for stacked per-member 2-day medians `[n_members, lat, lon]` |
| `cesm2_2day_per_member_p90_cache_path(start, end)` | function | Cache path for stacked per-member 2-day 90th-pctile fields `[n_members, lat, lon]` |
| `SEASONS_ORDER`, `SEASON_LABELS`, `SEASON_MONTHS` | constants | Season ordering, display labels, and month lists for DJF/MAM/JJA/SON |
| `era5_interp_2day_seasonal_median_cache_path(season, start, end)` | function | Cache for ERA5-interp seasonal 2-day median |
| `era5_interp_2day_seasonal_p90_cache_path(season, start, end)` | function | Cache for ERA5-interp seasonal 2-day 90th-pctile |
| `cesm2_2day_seasonal_global_median_cache_path(season, start, end)` | function | Cache for CESM2-LE seasonal global median `[lat, lon]` |
| `cesm2_2day_seasonal_per_member_medians_cache_path(season, start, end)` | function | Cache for CESM2-LE per-member seasonal medians `[n_members, lat, lon]` |
| `cesm2_2day_seasonal_global_p90_cache_path(season, start, end)` | function | Cache for CESM2-LE seasonal global 90th-pctile `[lat, lon]` |
| `cesm2_2day_seasonal_per_member_p90_cache_path(season, start, end)` | function | Cache for CESM2-LE per-member seasonal 90th-pctile `[n_members, lat, lon]` |
| `twodaymedian_seasonal_precip_sig_diff_paths(fig_subdir, start, end, pctl_tag)` | function | Paths for 4×3 seasonal significance-hatched 2-day median diff — `2daymedian_seasonal_precip_{pctl_tag}_diff_*.pdf` |
| `twodaymedian_seasonal_90pctl_precip_sig_diff_paths(fig_subdir, start, end, pctl_tag)` | function | Paths for 4×3 seasonal significance-hatched 90th-pctile diff — `2daymedian_seasonal_90pctl_precip_{pctl_tag}_diff_*.pdf` |


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
| `compute_era5_2day_median_2d(resolution, start, end, ...)` | function | 2-day rolling median map from ERA5 overall cache |
| `find_era5_interpolated_files(era5_interp_dir)` | function | Sorted list of ERA5-interpolated-to-CESM2-LE annual files; variable prefix matched generically (`tp`/`sd`/`swvl`) so one function serves precip, SWE and soil moisture |
| `get_year_range_era5_interp(files)` | function | `(start_year, end_year)` from ERA5-interpolated file list |
| `save_era5_interpolated_field_overall(era5_interp_dir, out_path_fn, variable, cache_var, units, extent, force)` | function | Build/save ERA5-interpolated daily SWE (`sd`) / soil-moisture (`swvl`) cache; no metre→mm conversion |
| `compute_era5_interpolated_2day_median_2d(start, end, ...)` | function | 2-day rolling median from ERA5-interpolated cache |
| `compute_era5_interpolated_2day_p90_2d(start, end, ..., p90_cache_path, ...)` | function | 90th percentile of 2-day rolling sums from ERA5-interpolated; caches result; reads cache by trying both `twoday_p90_precip` and legacy `twodayp90_precip` variable names |
| `_SEASON_MONTHS` | constant | Module-level dict mapping season abbreviation → month list |
| `_filter_season_da(da, season)` | function | Select time steps in a DataArray whose month belongs to the given season |
| `compute_era5_interpolated_2day_seasonal_median_2d(season, start, end, ...)` | function | Seasonal median of 2-day rolling sums from ERA5-interpolated cache; caches result |
| `compute_era5_interpolated_2day_seasonal_p90_2d(season, start, end, ...)` | function | Seasonal 90th percentile of 2-day rolling sums from ERA5-interpolated cache; caches result |


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
| `compute_cesm2_le_annual_median_2d(start, end, model_dir, member_cache_path_fn, median_cache_path, open_cache_fn, ...)` | function | True pooled median annual map: pools all n\_members × n\_years annual totals per pixel then takes `np.nanmedian`; caches result |
| `compute_cesm2_le_2day_median_2d(start, end, ..., rolling_fn, subset_fn, ...)` | function | 100-member median 2-day rolling map; caches result |
| `compute_cesm2_le_2day_p90_2d(start, end, ..., p90_cache_path, rolling_fn, subset_fn, ...)` | function | 100-member median of per-member 90th-pctile 2-day sums; caches result; reads cache by trying both `twoday_p90_precip` and legacy `twodayp90_precip` variable names |
| `compute_cesm2_le_2day_global_median_2d(start, end, model_dir, ..., global_median_cache_path, per_member_medians_cache_path, ...)` | function | True global median of 2-day rolling sums (all members × time) per pixel; simultaneously saves per-member medians `[n_members, lat, lon]` for significance testing |
| `compute_cesm2_le_2day_per_member_p90_2d(start, end, model_dir, ..., per_member_p90_cache_path, global_p90_cache_path, ...)` | function | Per-member 90th-pctile 2-day fields `[n_members, lat, lon]` plus global p90; saves two caches |
| `compute_significance_masks(da_era5, per_member_da, lower_pctl, upper_pctl)` | function | Direct percentile-rank significance test (no FDR): marks a pixel significant if `n_greater >= round(upper_pctl/100 * n_members)` (ERA5 ≤ lower pctile → `sig_cesm_higher`) or `n_less >= threshold` (ERA5 ≥ upper pctile → `sig_era5_higher`); NaN pixels set to False; returns `(sig_cesm_higher, sig_era5_higher)` bool arrays |
| `_SEASON_MONTHS` | constant | Module-level dict mapping season abbreviation → month list |
| `_season_time_mask(time_values, season)` | Returns boolean mask selecting timesteps in the given season's months; handles both `numpy.datetime64` and `cftime.DatetimeNoLeap` time axes | private helper |
| `compute_cesm2_le_2day_seasonal_global_median_2d(season, start, end, ...)` | function | Seasonal true global median + per-member medians of 2-day rolling sums; saves two caches |
| `compute_cesm2_le_2day_seasonal_per_member_p90_2d(season, start, end, ...)` | function | Seasonal per-member 90th-pctile fields + global 90th-pctile; saves two caches |


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
| `rolling_melt(da, window_days, var_name)` | function | Snowmelt magnitude over the trailing window — `max(0, −(da(t) − da(t−(window−1))))`; only SWE decreases count, as a positive melt flux (accumulation → 0); used for the snowmelt / compound-flood maps instead of the signed `rolling_change` |
| `rolling_mean(da, window_days, var_name)` | function | Trailing rolling mean over the window — mean of `da` over `[t−(window−1), t]`; soil-moisture counterpart of `rolling_accumulation` (units inherited, not forced to mm) |
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
| `_finite_max_abs(values, fallback=1.0)` | Return max(\|finite values\|), or `fallback` if none are finite. Guards the diverging-norm scaling in `plot_2day_interp_3panel` / `_diffonly` / `_diffonly_sig` against the `nanmax` "zero-size array to reduction operation fmax" crash when a difference field is entirely NaN (e.g. grid coords don't align, or all-zero reference field) |
| `round_up_nice(value)` | Round a value up to a visually clean colorbar upper bound |
| `colorbar_label(window_days)` | Standard colorbar label string for event precipitation maps |
| `title_text(combo)` | Standard figure title for Storm Hans event maps |
| `make_colorbar_ticks(vmax)` | Evenly-spaced tick array for precipitation colorbars |
| `compute_vmax_by_window(event_fields)` | Fixed colorbar maxima for Storm Hans 1-day and 2-day maps |
| `plot_precip_map(combo, da_evt, catchments, vmax, out_paths, ...)` | Single-panel Storm Hans event precipitation map |
| `plot_single_catchment_weight_map(combo, slug, title, da_w, gdf, out_paths)` | One weight-fraction map for a single catchment |
| `_plot_annmedian_panel(ax, da, dataset_type, ...)` | Draw one panel in an annual median map; returns mesh for shared colorbar |
| `plot_2day_median_2panel(da_cesm2, da_era5, ...)` | 2-panel annual median of 2-day rolling precipitation; catchment legend centered at colorbar height |
| `plot_2day_interp_3panel(da_cesm2, da_era5_interp, da_diff, ..., fig_title, seq_cbar_label, div_cbar_label, sig_cesm_higher, sig_era5_higher, sig_legend_text)` | 3-panel: CESM2-LE \| ERA5-interp \| difference — two colorbars; optional per-pixel significance hatching on ERA5 panel (uses `_add_pixel_hatch_overlay` for clean whole-cell `//` and `\\` patterns); combined catchment + significance legend between the two colorbars |
| `_add_pixel_hatch_overlay(ax, lons, lats, mask, hatch, transform, ...)` | Draw hatch pattern over every `True` cell in mask as individual `Rectangle` patches via `PatchCollection`; avoids the half-pixel artifacts of `contourf`-based hatching |
| `plot_annual_median_4panel(da_cesm2, da_senorge, da_era5_05, da_era5_025, ...)` | 2×2 panel annual median precipitation map; left-column panels anchored "W" and right-column anchored "E" to spread panels outward; `hspace=0.16`, `wspace=0.05`; colorbar at x=0.15, legend at x=0.72 |
| `_plot_diff_panel(ax, da, panel_title, catchments, norm, ...)` | Draw one difference panel using diverging colormap |
| `plot_2day_interp_diffonly` | Single-panel diverging difference map (CESM2-LE – ERA5 interp). Catchment legend placed inside panel at bottom-right (`ax.transAxes`). Saves to `out_paths` and closes figure. | `da_diff`, `catchments`, `start_year`, `end_year`, `out_paths`, `fig_title`, `div_cbar_label`, `catchment_numbers`, `catchment_legend_text`, `label_overrides`, `annmedian_extent` |
| `plot_2day_interp_seasonal_4row_3col(seasonal_data, catchments, ...)` | 4×3-panel seasonal significance plot: 4 rows (DJF/MAM/JJA/SON) × 3 columns (CESM2-LE \| ERA5-interp \| diff); season labels bold-vertical on left; column labels bold on top row only; colorbars + legend under bottom row only; shared diverging norm across all seasons |



---

### `generate_weights.py` — run-once script

Generates catchment area-fraction weight NetCDF files. Not imported by any notebook.
Only needs to be re-run if catchment boundaries or grid definitions change.

---

## Notebook files (`code/`)

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

**Purpose:** Spatial precipitation maps for Storm Hans and annual mean / 2-day comparison maps.
**Produces:** `precip_maps_hans/` figures.

**Structure:**
1. Setup cell — imports from all helpers; constants (`MAP_EXTENT`, `MAP_START`, `MAP_END`, `ANN_VMAX`, `TWODAY_VMAX`, `CATCHMENT_NUMBERS`, `COMBINATIONS`, etc.); small notebook-specific helpers (`combo_key`, `build_output_paths`); dependency callables `_open`, `_roll`, `_sub`
2. Cache builder cell — calls `save_era5_overall`, `save_senorge_overall`, `save_smile_overall`
3. Event data loader cell — defines `load_event_field` (notebook-local, uses EVENT_DATE/ENVELOPE scope)
4. Event maps cell — defines `run_all_precip_maps` orchestration; calls `plot_precip_map` from `plot_style`
5. Run event maps cell — `run_all_precip_maps()`
6. Weight maps cell — `WEIGHT_COMBINATIONS` config; calls `plot_single_catchment_weight_map` from `plot_style`
7. Run 4-panel cell — calls `compute_era5_annual_median_2d`, `compute_senorge_annual_median_2d`, `compute_cesm2_le_annual_median_2d`, `plot_annual_median_4panel`
8. Run 2-day cell — calls `compute_era5_2day_median_2d`, `compute_cesm2_le_2day_global_median_2d` (true global median, not median-of-medians), `plot_2day_median_2panel`
9. ERA5-interpolated cache + 2-day median difference cell — `save_era5_interpolated_overall`, `compute_era5_interpolated_2day_median_2d`, `plot_2day_interp_3panel`, `plot_2day_interp_diffonly`
10. 90th-percentile difference cell — `compute_cesm2_le_2day_per_member_p90_2d` (saves global p90 and per-member p90 cache), `compute_era5_interpolated_2day_p90_2d`, `plot_2day_interp_3panel`, `plot_2day_interp_diffonly`
11. Significance-hatched median diff cell — loads `cesm2_2day_per_member_medians` cache, calls `compute_significance_masks` at 5/95 and 2/98 pctile, plots via `plot_2day_interp_3panel` with sig overlays
12. Significance-hatched 90th-pctile diff cell — loads `cesm2_2day_per_member_p90` cache, calls `compute_significance_masks` at 5/95 and 2/98 pctile, plots via `plot_2day_interp_3panel` with sig overlays
13. Seasonal data computation cell — `compute_era5_interpolated_2day_seasonal_median_2d`, `compute_era5_interpolated_2day_seasonal_p90_2d`, `compute_cesm2_le_2day_seasonal_global_median_2d`, `compute_cesm2_le_2day_seasonal_per_member_p90_2d` for all 4 seasons; stores results in `SEASONAL_MEDIAN_DATA` / `SEASONAL_P90_DATA` dicts keyed by season
14. Seasonal significance plot cell — `_build_seasonal_list` helper calls `compute_significance_masks` per season then calls `plot_2day_interp_seasonal_4row_3col` four times (5/95 median, 2/98 median, 5/95 p90, 2/98 p90) → four PDFs in `precip_maps_hans/`


---

### `compound_flood_risk_analysis.ipynb`

**Purpose:** Reproduce the CESM2-LE vs ERA5-interpolated comparison maps from
`create_precip_maps_hans.ipynb` for **Snowmelt** (signed 24-h SWE change,
ΔSWE = SWE(t) − SWE(t−1), one value per date) and **Soil Moisture** (2-day rolling
mean). Same methodology: 2-day median, 90th percentile, percentile-rank significance
testing (5/95, 2/98), seasonal breakdown, catchment-3 diagnostics. Units kg/m². Only
90 of the 100 CESM2-LE members carry SWE/SM output → ensemble size auto-detected (n=90).
**Produces:** `compound_flood_risk_output/` figures (18 PDFs per variable). The
filename temporal prefix is per-variable via `VARIABLES[...]["fstem"]`: soil
moisture → `2daymedian_…` (2-day mean), snowmelt → `dailymedian_…` (24-h ΔSWE).
Set: `{fstem}_{var}_diff/diffonly`, `{fstem}_90pctl_{var}_diff/diffonly`,
significance `{fstem}_{var}_{5_95pctl|2_98pctl}_diff/diffonly` (and `…_90pctl_…`),
seasonal `{fstem}_seasonal_{var}_…_diff`, and `diagnostic_catchment3_…_{var}`
strip plots. The `annualmedian_*` and 2-panel overview figures are intentionally
omitted (they need SeNorge/native-ERA5, absent for SWE/SM).

**Structure (orchestration only; loops over the two variables):**
1. Setup — `VARIABLES` config, `_roll_change`/`_roll_identity`/`_open_field`/`figp` helpers, `RECOMPUTE` flag
2. One-time daily cache builder — `save_cesm2_le_field_overall`, `save_era5_interpolated_field_overall` → `cesm2_le/{swe,soil_moisture}`, `era5_interpolated/{swe,soil_moisture}`
3. 2-day median + 90th-pctile 3-panel diff + diffonly — `compute_cesm2_le_2day_global_median_2d`, `compute_cesm2_le_2day_per_member_p90_2d`, `compute_era5_interpolated_2day_median_2d`/`_p90_2d`, `plot_2day_interp_3panel`/`_diffonly`
4. Significance-hatched diff (5/95, 2/98) — `compute_significance_masks`, `plot_2day_interp_3panel`/`_diffonly_sig`
5. Diagnostic pixel tables (no PDF)
6. Seasonal data computation — `compute_cesm2_le_2day_seasonal_global_median_2d`/`per_member_p90_2d`, `compute_era5_interpolated_2day_seasonal_median_2d`/`_p90_2d`
7. Seasonal significance 4×3 plots — `plot_2day_interp_seasonal_4row_3col`
8. Seasonal diagnostic (vmax calibration, no PDF)
9. Catchment-3 (Losna) per-member strip plots → `diagnostic_catchment3_*` PDFs

Cache paths come from `cfg.field_daily_cache_path` / `cfg.field_2day_cache_path`;
figure paths reuse `cfg.precip_map_figure_paths`. All heavy compute/plot helpers
are reused unchanged via variable-agnostic discovery + injected cache-openers
(`open_field_cache`) and rollers (`rolling_change` = signed ΔSWE = SWE(t)−SWE(t−1) for snowmelt, `rolling_mean` for soil moisture; `rolling_melt` = max(0,−ΔSWE) retained but unused). Each variable also carries an `fstem` filename prefix (`dailymedian` for snowmelt, `2daymedian` for soil moisture).
