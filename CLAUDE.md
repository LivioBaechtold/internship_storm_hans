# CLAUDE.md — Project Instructions for Storm Hans Precipitation & Climate-Model Analysis

> **Read this file first, every time, before touching any code.**  
> This file defines who I am, what this project does, how the code is structured, and exactly how you must respond to every request.

---

## 1. Who I Am and What This Project Is

I am a **master's student in climate science** working on an analysis of **Storm Hans (August 2023)** — an extreme precipitation event in southern Norway. My work lives at:

```
/nird/home/lbal/internship_storm_hans/
```

The project analyses precipitation extremes using:
- **Reanalysis datasets**: ERA5 (0.5° and 0.25°) and seNorge (1 km Norwegian grid)
- **Climate model large ensembles (SMILE)**: CESM2-LE and GFDL-SPEAR-MED-LE
- **Five study catchments** in southern Norway (Bergheim, Hønnefoss, Losna, Drammen, Glomma)
- **Methods**: catchment-averaged precipitation time series, GEV extreme-value fitting, return-period estimation, distribution comparisons

My **typical requests** to you will be: create new analysis code, update existing code to add/change plots or statistics, fix bugs, or add new datasets. I will describe *what I want to see* — you must figure out *where* and *how* to implement it within the existing file structure, and briefly desribe beforehand what the respective change does (but really briefly, just shortly explain why there specifically and what it does).

---

## 2. Absolute Rules — Read Before Doing Anything

### 2.1 Never Directly Edit Files
**You must never modify files yourself.** Instead, for every change, provide:
1. The **target filename** (e.g. `helper/plot_style.py`)
2. The **exact location** — either:
   - Line numbers: `Lines 45–67` (delete these), or
   - Section header: `# ── Statistical helpers ──` (insert after this line)
3. The **code to delete** (copy-pastable block, clearly marked)
4. The **code to insert** (copy-pastable block, clearly marked)

Always use this format:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE: helper/plot_style.py
LOCATION: After line 134 / after `# ── colorbar helpers ──`
ACTION: INSERT the following block
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[exact Python code block here]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE: helper/plot_style.py
LOCATION: Lines 200–215
ACTION: DELETE this block and REPLACE with the following
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DELETE:
[exact existing code]

INSERT:
[new code]
```

If a change touches **multiple files**, show each file in its own clearly labelled block.

### 2.2 Always Respect the File Structure
The project has a **strict logical separation** of responsibilities across files. Before placing any code, check where it belongs according to Section 4 of this document. Never put plotting logic in `catchment_tools.py`, never put statistical logic in `plot_style.py`, never hard-code paths outside `config_paths.py`.

### 2.3 Always Update Code_Overview.md
At the **end of every response**, after all code changes, provide a dedicated section titled:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED UPDATES TO: Code_Overview.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

This must list every new or changed function/constant and which table row to add, modify, or delete. Important is that you also here give me the sections to delete and to newly insert in Copy-Pastable Code-form as it stands here within the .md-file (as Code-form).

### 2.4 Check Postprocessed Data First
Before writing any data-loading code, check what is already cached at:

```
/nird/datalake/NS9873K/lbal/postprocessed/
```

The standard sub-structure is:
```
postprocessed/
├── era5/          # ERA5 spatial caches (.nc, named with resolution + year range)
├── senorge/       # seNorge spatial caches
├── smile/         # SMILE per-member spatial caches
│   ├── cesm2_le/
│   └── gfdl_spear_med_le/
├── weights/       # catchment weight NetCDFs (one per catchment × dataset)
└── catchments/    # catchment-averaged time-series caches (one per catchment × dataset × window)
```

Naming conventions (from `config_paths.py`):
- Spatial daily cache: `{dataset}_{resolution}_{start}_{end}.nc` (ERA5) or `senorge_{start}_{end}.nc`
- Catchment cache: `{dataset}_{res_tag}_2day_{slug}_{start}_{end}.nc`
- Weight file: `weights_catchment_{slug}_{dataset}_{resolution}.nc`

**Decision logic you must follow:**

| Situation | What to do |
|---|---|
| Cache exists, correct variable name, correct period | Read directly from cache — do NOT reload raw data |
| Cache exists but need derived quantity (e.g. 2-day rolling from daily cache) | Load daily cache, compute rolling sum on the fly |
| Cache exists for a superset period | Open cache, subset by year using `open_precip_cache()` |
| Cache missing entirely | Provide a **one-time setup cell** to build it (see Section 5), then use cache in the main analysis |

### 2.5 Handle Images as Interpretation Hints
When I attach an image of a current (wrong or incomplete) plot output, treat it as visual context to understand what currently exists and what needs to change. Do not invent data — use it to understand layout, axis labels, color ranges, panel arrangement, or missing elements.

---

## 3. Project Directory Layout

```
/nird/home/lbal/internship_storm_hans/
├── helper/                        ← All importable Python modules
│   ├── config_paths.py
│   ├── data_era5.py
│   ├── data_senorge.py
│   ├── data_smile.py
│   ├── catchment_tools.py
│   ├── return_period.py
│   ├── plot_style.py
│   ├── generate_weights.py        ← run-once script, not imported
│   ├── prec_seq.txt               ← IPCC sequential colormap (256 RGB rows)
│   └── prec_div.txt               ← IPCC diverging colormap (256 RGB rows)
├── code/                          ← Jupyter notebooks (no reusable functions here)
│   ├── analysis_return_hans.ipynb
│   ├── climate_model_evaluation.ipynb
│   └── create_precip_maps_hans.ipynb
├── catchments/                    ← GeoJSON boundary files for each catchment
├── Code_Overview.md               ← Living documentation of all files and functions
└── CLAUDE.md                      ← This file
```

Raw data locations (read-only, never write here):
```
/nird/datalake/NS9873K/lbal/
├── ERA5/                          ← Raw ERA5 annual NetCDF files
├── seNorge/                       ← Raw seNorge annual NetCDF files
├── CESM2-LE/scandinavia/          ← CESM2-LE model output
└── GFDL-SPEAR/                    ← GFDL-SPEAR model output
```

---

## 4. File Responsibilities — Where Every Type of Code Lives

This is the single most important rule about code placement. Study this before writing anything.

### `config_paths.py` — constants and path helpers ONLY
- All `Path` constants, `CATCHMENTS` dict, `SMILE_CONFIG`, `MODEL_COLORS/LABELS/ORDER`
- All path-construction functions (`res_tag`, `postproc_dir`, `catchment_postproc_path`, figure path functions, etc.)
- **Nothing else.** No imports of numpy/xarray beyond what path functions absolutely need.

### `data_era5.py` — ERA5 file I/O and spatial-cache builders
- File discovery (`find_era5_files`, `find_era5_interpolated_files`)
- Raw data loading and unit conversion
- Spatial cache builders (`save_era5_overall`, `save_era5_interpolated_overall`)
- Spatial median/percentile computation from cache (`compute_era5_*`)

### `data_senorge.py` — seNorge file I/O and spatial-cache builders
- Same structure as `data_era5.py` but for seNorge
- UTM coordinate handling (`latlon_extent_to_utm_bbox`)

### `data_smile.py` — SMILE model I/O and spatial-cache builders
- Member discovery, per-member loading, unit conversion
- Spatial cache builders and ensemble-median computations

### `catchment_tools.py` — catchment averaging and non-plotting data preparation
- Weight file operations (`find_weight_file`, `load_weights`, `weighted_catchment_mean`)
- Rolling accumulation, time subsetting, cache I/O
- High-level orchestration: `run_all`, `run_all_smile`
- Per-catchment data loaders for evaluation: `load_annual_maxima_per_catchment`, `load_daily_values_per_catchment`
- Statistical table builders: `build_percentile_mapping_table`, `build_distribution_summary_table`

### `return_period.py` — pure statistics, no I/O
- GEV fitting, Weibull plotting positions, return-level computation
- `get_annual_maxima`, `fit_gev`, `gev_return_level`, `estimate_return_period`

### `plot_style.py` — ALL Matplotlib/Cartopy figure code
- Every function that creates, populates, or saves a figure goes here
- Module-level constants: `MAP_PROJ`, colormaps loaded from `.txt` files, `OCEAN_COLOR`
- Re-exports `MODEL_COLORS`, `MODEL_LABELS`, `MODEL_ORDER` from `config_paths`
- **No raw data loading, no statistical fitting**

### Notebooks (`code/*.ipynb`) — orchestration only, no reusable logic
- Import from helper files, set parameters, call functions
- Local helper closures (`_open`, `_roll`, `_sub`) for passing callables into functions
- **Never define a reusable function inside a notebook** — it belongs in a helper file
- Notebook cells must have short, clear `# %% [Cell title]` headers

---

## 5. Postprocessed Data Setup Pattern

Whenever a new dataset or derived quantity is needed that does not yet have a cache, provide a **standalone one-time setup cell** at the top of the relevant notebook. Use this template:

```python
# %% [One-time cache builder — run once, then comment out]
# Run this cell ONCE to build the postprocessed cache.
# After it completes, comment out or skip this cell on future runs.

from data_<module> import save_<dataset>_overall
import config_paths as cfg

save_<dataset>_overall(
    <raw_dir>           = cfg.<RAW_DIR>,
    out_path_fn         = cfg.overall_precip_path,
    extent              = cfg.OVERALL_PRECIP_EXTENT,
    force               = False,   # set True to overwrite
)
print("Cache built successfully.")
```

After showing the setup cell, write the main analysis code to **always load from cache**, never from raw files, so subsequent runs are fast.

---

## 6. Coding Standards

### Style
- Python 3.11, numpy/xarray/scipy/matplotlib/cartopy stack
- Type hints on all function signatures
- Docstrings: one-line summary + Parameters / Returns sections for non-trivial functions
- Section headers: `# ── Section name ───────────...` (em-dash style, consistent with existing code)
- Use `Path` objects throughout — no raw strings for file paths

### Imports
- Standard library first, then third-party, then local (`config_paths`, `data_*`, `catchment_tools`, etc.)
- Local imports always use the module name (e.g. `import config_paths as cfg`), never star-imports

### Xarray / data conventions
- Lazy loading via Dask wherever possible; call `.compute()` only at the last moment
- Time subsetting via `open_precip_cache(path, start_year, end_year)` — never slice raw `.nc` manually in notebooks
- Rolling accumulation via `catchment_tools.rolling_accumulation(da, window_days)` — do not re-implement
- Unit for precipitation: always **mm** (or mm/day). seNorge is already mm. ERA5 raw is in metres — `data_era5.py` handles conversion.

### Figures
- All figures saved to **both** `cfg.FIGURES_DIR` and `cfg.FIGURES_DIR_SECONDARY` (use the `out_paths` list pattern already established)
- Colormaps: sequential precipitation → `PRECIP_CMAP`; diverging anomaly → `PRECIP_DIV_CMAP`; weights → `WEIGHT_CMAP`
- Consistent model colours/labels: always use `cfg.MODEL_COLORS` and `cfg.MODEL_LABELS`
- Figure filenames: follow existing naming conventions — `{data_type}_{descriptor}_{window_days}day_{slug}_{start}-{end}.pdf`
- Never hardcode figure DPI, font size, or colorbar position as magic numbers — use named variables with a comment

---

## 7. The Five Catchments

| Slug | Human title | GeoJSON file |
|---|---|---|
| `nevina_bergheim` | Bergheim | `catchment_nve_nevina_bergheim.geojson` |
| `nevina_honnefoss` | Hønnefoss | `catchment_nve_nevina_hønnefoss.geojson` |
| `nevina_losna` | Losna | `catchment_nve_nevina_losna.geojson` |
| `regine_drammen` | Drammen | `catchment_nve_regine_drammen.geojson` |
| `regine_glomma` | Glomma | `catchment_nve_regine_glomma.geojson` |

Iterate catchments as: `for slug, title in cfg.CATCHMENTS.items():`

---

## 8. Dataset Reference

| Key | Dataset | Native grid | Precip variable | Units (raw) | Notes |
|---|---|---|---|---|---|
| `era5_0.5` | ERA5 | 0.5° lat/lon | `tp` | m | Multiply ×1000 for mm |
| `era5_0.25` | ERA5 | 0.25° lat/lon | `tp` | m | Multiply ×1000 for mm |
| `senorge` | seNorge | 1 km UTM-33 | `rr` | mm (=kg/m²) | Fill value −999.99 |
| `cesm2_le` | CESM2-LE | ~1° lat/lon | `tp24` | m or mm | Auto-detected via `unit_mode` |
| `gfdl_spear_med_le` | GFDL-SPEAR-MED-LE | ~1° lat/lon | `tp24` | m or mm | Auto-detected via `unit_mode` |

SMILE config is in `cfg.SMILE_CONFIG[dataset_key]` which contains: `model_dir`, `n_members`, `unit_mode`, `ref_dataset`.

---

## 9. How to Respond to My Requests — Step-by-Step Protocol

When I describe a desired change or new feature, follow this sequence **every time**:

**Step 1 — Understand the request**  
If I attach an image of a plot, use it as visual context. Identify what is currently wrong or missing. If my description is ambiguous on one critical point, ask exactly one clarifying question before proceeding.

**Step 2 — Plan across files**  
Map each piece of work to the correct file using Section 4. State your plan in one short paragraph before any code.

**Step 3 — Check postprocessed data**  
State explicitly whether the required data is expected to already be cached. If uncertain, write the code to check at runtime with an informative error message if the cache is missing, plus a separate one-time setup cell.

**Step 4 — Provide all code changes**  
For each file that needs a change, use the block format from Section 2.1. Cover all files — if a new function is added to `plot_style.py` and called from a notebook, show both the function addition AND the notebook cell that calls it.

**Step 5 — Update Code_Overview.md**  
Always end with the required Code_Overview.md update block (Section 2.3).

---

## 10. Common Patterns to Reuse

### Loading a catchment cache (already-averaged time series)
```python
from catchment_tools import load_catchment_cache
da = load_catchment_cache(dataset, resolution, window_days, slug, start_year, end_year)
```

### Spatial cache → catchment average
```python
from catchment_tools import open_precip_cache, find_weight_file, load_weights, weighted_catchment_mean, rolling_accumulation
cache_path = cfg.overall_precip_path(dataset, resolution, start_year, end_year)
da_spatial = open_precip_cache(cache_path, start_year, end_year)
da_rolled  = rolling_accumulation(da_spatial, window_days)
wf         = find_weight_file(dataset, resolution, slug)
weights    = load_weights(wf)
da_catchment = weighted_catchment_mean(da_rolled, weights)
```

### Saving a figure to both output directories
```python
out_paths = cfg.figure_paths(dataset, resolution, window_days, slug, start_year, end_year)
# or for custom names:
out_paths = [cfg.FIGURES_DIR / fig_subdir / fname,
             cfg.FIGURES_DIR_SECONDARY / fig_subdir / fname]
make_my_figure(..., out_paths=out_paths)
```

### SMILE member loop
```python
from data_smile import find_smile_members
members = find_smile_members(cfg.SMILE_CONFIG[dataset]["model_dir"], dataset)
for member_id in members:
    cache = cfg.overall_precip_member_path(dataset, member_id, start_year, end_year)
    ...
```

---

## 11. What Good Responses Look Like

A complete, correct response to a typical request will have this shape:

```
## Plan
[1–3 sentences: what files change and why]

## Postprocessed data check
[Explicit statement: cache expected at X. Already exists / needs one-time build.]

## One-time setup cell (if needed)
FILE: code/<notebook>.ipynb
[cell code]

## Code changes

━━━ FILE: helper/plot_style.py ━━━
LOCATION: ...
ACTION: INSERT / DELETE+REPLACE
[code block]

━━━ FILE: code/climate_model_evaluation.ipynb ━━━
LOCATION: Cell N — "Distribution figures loop"
ACTION: REPLACE cell content
[cell code]

## Required updates to Code_Overview.md

| File | Table | Action | New row content |
|---|---|---|---|
| `plot_style.py` | Functions | ADD | `make_my_new_figure(...)` \| ... |
```

---

*This file is the source of truth for how Claude should assist me. When in doubt, re-read Section 2 and Section 4.*
