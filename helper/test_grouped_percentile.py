# test_grouped_percentile.py
"""Reference unit test for catchment_tools.grouped_percentile.

Run from the repository root (or from helper/) with

    python helper/test_grouped_percentile.py

It also collects under pytest if that is ever installed. The reference case is
the 2002-2011 window of the CESM2-LE compound-extreme analysis: 90 members,
L = 10 years, rate quantum w = 1/L = 0.1.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catchment_tools import grouped_percentile   # noqa: E402

# ── Reference window 2002-2011: members per attainable rate value ─────────────
# 0.6 and 0.8 are EMPTY on purpose — the function must skip them via the actual
# counts, not via bin-index arithmetic.
REF_COUNTS: dict[float, int] = {
    0.0: 19, 0.1: 30, 0.2: 19, 0.3: 9, 0.4: 6,
    0.5: 4,  0.6: 0,  0.7: 2,  0.8: 0, 0.9: 1,
}
REF_BIN_WIDTH = 0.1     # = 1 / L with L = 10 years
REF_N         = 90
TOL           = 1e-3

# Grouped percentiles expected for the reference window.
REF_GROUPED: dict[float, float] = {
    2.5: 0.0059, 25.0: 0.0617, 50.0: 0.1367, 75.0: 0.2474, 97.5: 0.6875,
}
# What np.percentile returns on the SAME data — every value is a multiple of the
# quantum, which is exactly the staircase the grouped version removes.
REF_EMPIRICAL: dict[float, float] = {25.0: 0.1, 50.0: 0.1, 75.0: 0.2}


def _reference_sample(seed: int = 12345) -> np.ndarray:
    """The 90 member rates of the reference window, deliberately UNSORTED."""
    values = np.concatenate([np.full(n, v) for v, n in REF_COUNTS.items() if n])
    np.random.default_rng(seed).shuffle(values)
    return values


def test_reference_sample_is_well_formed() -> None:
    v = _reference_sample()
    assert v.size == REF_N, f"expected {REF_N} members, got {v.size}"
    assert abs(float(v.mean()) - 0.18) < 1e-12, (
        f"ensemble mean must be exactly 0.18, got {v.mean()!r} — data ingestion is wrong")


def test_grouped_percentile_reference_values() -> None:
    v = _reference_sample()
    for q, expected in REF_GROUPED.items():
        got = grouped_percentile(v, q, REF_BIN_WIDTH)
        assert abs(got - expected) < TOL, f"p{q}: got {got:.6f}, expected {expected}"


def test_grouped_percentile_accepts_array_q() -> None:
    v = _reference_sample()
    qs  = np.array(sorted(REF_GROUPED))
    got = grouped_percentile(v, qs, REF_BIN_WIDTH)
    assert got.shape == qs.shape
    assert np.allclose(got, [REF_GROUPED[q] for q in qs], atol=TOL)


def test_does_not_fall_back_to_np_percentile() -> None:
    """np.percentile must still give the staircase, and grouped must differ from it."""
    v = _reference_sample()
    for q, expected in REF_EMPIRICAL.items():
        emp = float(np.percentile(v, q))
        assert abs(emp - expected) < 1e-12, f"np.percentile p{q}: {emp} != {expected}"
        grp = grouped_percentile(v, q, REF_BIN_WIDTH)
        assert abs(grp - emp) > TOL, (
            f"p{q}: grouped ({grp:.6f}) equals the empirical value ({emp}) — "
            "grouped_percentile has silently fallen back to np.percentile")


def test_zero_bin_is_half_width() -> None:
    """The half-width zero bin keeps low percentiles non-negative."""
    v = _reference_sample()
    assert grouped_percentile(v, 2.5, REF_BIN_WIDTH) >= 0.0
    naive = grouped_percentile(v, 2.5, REF_BIN_WIDTH, zero_half_bin=False)
    assert naive < 0.0, ("the naive symmetric zero bin is supposed to return a "
                         "negative rate — that is why zero_half_bin defaults to True")


def test_empty_bins_are_skipped() -> None:
    """p97.5 falls in the 0.7 bin only if the empty 0.6 and 0.8 bins are skipped."""
    v = _reference_sample()
    got = grouped_percentile(v, 97.5, REF_BIN_WIDTH)
    assert abs(got - 0.6875) < TOL, f"p97.5 = {got:.6f}: empty bins mishandled"


def test_unsorted_input() -> None:
    v = _reference_sample(seed=1)
    w = np.sort(_reference_sample(seed=2))
    assert np.allclose(grouped_percentile(v, [25.0, 75.0], REF_BIN_WIDTH),
                       grouped_percentile(w, [25.0, 75.0], REF_BIN_WIDTH))


def test_degenerate_single_value() -> None:
    """All members identical: the result stays inside that value's own bin."""
    for value, lo, hi in ((0.0, 0.0, 0.05), (0.3, 0.25, 0.35)):
        v = np.full(REF_N, value)
        got = grouped_percentile(v, [0.0, 25.0, 50.0, 100.0], REF_BIN_WIDTH)
        assert np.all(got >= lo - 1e-12) and np.all(got <= hi + 1e-12), \
            f"all-{value} sample gave {got}, outside [{lo}, {hi}]"
        assert np.all(np.diff(got) >= -1e-12), f"ordering inverted: {got}"


def test_bin_width_is_not_hardcoded() -> None:
    """L = 15 (w = 1/15) must be handled exactly like L = 10."""
    w = 1.0 / 15.0
    v = np.concatenate([np.full(n, k * w) for k, n in enumerate([19, 30, 19, 9, 6, 4])])
    got = grouped_percentile(v, [2.5, 25.0, 50.0, 75.0, 97.5], w)
    assert np.all(got >= 0.0) and np.all(np.diff(got) > 0.0)
    # The same relative position inside the sample, just on the 1/15 grid.
    assert abs(got[2] / (grouped_percentile(v * 1.5, 50.0, 0.1) / 1.5) - 1.0) < 1e-9


def test_ordering_never_inverts() -> None:
    v = _reference_sample()
    got = grouped_percentile(v, [2.5, 25.0, 50.0, 75.0, 97.5], REF_BIN_WIDTH)
    assert np.all(np.diff(got) >= 0.0), f"percentiles out of order: {got}"
    assert np.all(got >= 0.0), f"negative percentile: {got}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    sys.exit(1 if failed else 0)
