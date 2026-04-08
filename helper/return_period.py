# return_period.py
"""
GEV extreme-value fitting, Weibull plotting positions, return-period
estimation, and two-panel figure generation.
"""

import numpy as np
import pandas as pd
from scipy.stats import genextreme
import xarray as xr


# ── Statistical helpers ────────────────────────────────────────────────────────

def get_annual_maxima(da: xr.DataArray) -> pd.Series:
    """
    Derive the annual maxima series from a daily catchment time series
    Returns a pandas Series indexed by year
    """
    ts = da.to_series().dropna()
    return ts.groupby(ts.index.year).max().rename_axis("year")


def weibull_plotting_positions(annual_max: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Weibull plotting positions for annual maxima

    Largest event gets rank 1.
    Return period:
        T_i = (n + 1) / rank_i
    """
    vals = np.sort(annual_max.values)[::-1]   # descending
    n = len(vals)
    ranks = np.arange(1, n + 1)
    T = (n + 1) / ranks
    return vals, T


def fit_gev(annual_max: pd.Series) -> tuple[float, float, float]:
    """
    Fit a GEV distribution to annual maxima using scipy MLE
    Returns (shape c, location loc, scale scale)
    """
    values = np.asarray(annual_max.values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 10:
        raise ValueError("Too few finite annual maxima for a stable GEV fit.")

    if np.nanmax(values) <= 0:
        raise ValueError(
            "Annual maxima are non-positive. "
            "Check upstream precipitation calculation before fitting the GEV.")

    c, loc, scale = genextreme.fit(values)

    if (not np.isfinite(c)) or (not np.isfinite(loc)) or (not np.isfinite(scale)) or scale <= 0:
        raise ValueError("GEV fit returned invalid parameters.")

    return float(c), float(loc), float(scale)


def gev_return_level(c: float, loc: float, scale: float,
                     return_periods: np.ndarray) -> np.ndarray:
    """Return GEV quantiles for an array of return periods."""
    return genextreme.ppf(1.0 - 1.0 / return_periods, c, loc=loc, scale=scale)


def get_event_annual_max(da: xr.DataArray, search_year: int) -> tuple[float, pd.Timestamp]:
    """
    Use the annual maximum inside search_year as the event
    Returns (event_value, event_date_of_max)
    """
    ts = da.to_series().dropna()
    ts_year = ts[ts.index.year == search_year]

    if ts_year.empty:
        raise ValueError(f"No data found for event year {search_year}.")

    event_date = pd.Timestamp(ts_year.idxmax())
    event_value = float(ts_year.loc[event_date])
    return event_value, event_date

def estimate_return_period(event_value: float,
                           c: float, loc: float, scale: float) -> float:
    """
    Estimate the return period of a given event value from the fitted GEV
    T = 1 / P(X > x) = 1 / (1 − CDF(x))
    """
    exceedance = 1.0 - genextreme.cdf(event_value, c, loc=loc, scale=scale)
    return np.inf if exceedance <= 0 else 1.0 / exceedance

# Define Ensemble helpers ───────────────────────────────────────────────────────────
def combine_member_annual_maxima(
    member_annual_maxima: list[pd.Series],) -> pd.Series:
    """
    Build one annual-max series for a SMILE dataset using your new definition:

    For each calendar year:
        take the maximum across ALL ensemble members and ALL days in that year

    Input
    -----
    member_annual_maxima: list of pd.Series
        One Series per member, indexed by year, values = annual maximum of that member

    Returns
    -------
    pd.Series
        Indexed by year
        Length = number of years in the selected window, not n_members * n_years
    """
    if not member_annual_maxima:
        raise ValueError("member_annual_maxima is empty.")

    combined = pd.concat(member_annual_maxima, axis=1)
    combined = combined.sort_index()

    annual_max_yearly = combined.max(axis=1, skipna=True).dropna()
    annual_max_yearly.index.name = "year"
    annual_max_yearly.name = "annual_max_yearly"

    return annual_max_yearly