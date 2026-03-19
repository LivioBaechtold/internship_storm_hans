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
    Derive the annual maxima series from a daily catchment time series.
    Returns a pandas Series indexed by year.
    """
    ts = da.to_series().dropna()
    return ts.groupby(ts.index.year).max().rename_axis("year")


def weibull_plotting_positions(annual_max: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Weibull plotting positions for annual maxima.

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
    Fit a GEV distribution to annual maxima using scipy MLE.
    Returns (shape c, location loc, scale scale).
    """
    return genextreme.fit(annual_max.values)


def gev_return_level(c: float, loc: float, scale: float,
                     return_periods: np.ndarray) -> np.ndarray:
    """Return GEV quantiles for an array of return periods."""
    return genextreme.ppf(1.0 - 1.0 / return_periods, c, loc=loc, scale=scale)


def get_event_value(da: xr.DataArray, event_date: str) -> tuple[float, pd.Timestamp]:
    """
    Return the time-series value at the exact event date.
    """
    ts = da.to_series().dropna()
    event_date = pd.Timestamp(event_date)

    if event_date not in ts.index:
        raise ValueError(
            f"Event date {event_date.date()} not found in the time series.\n"
            f"Available range: {ts.index.min().date()}–{ts.index.max().date()}"
        )

    return float(ts.loc[event_date]), event_date

def get_event_annual_max(da: xr.DataArray, search_year: int) -> tuple[float, pd.Timestamp]:
    """
    Use the annual maximum inside search_year as the event.
    Returns (event_value, event_date_of_max).
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
    Estimate the return period of a given event value from the fitted GEV.
    T = 1 / P(X > x) = 1 / (1 − CDF(x))
    """
    exceedance = 1.0 - genextreme.cdf(event_value, c, loc=loc, scale=scale)
    return np.inf if exceedance <= 0 else 1.0 / exceedance
