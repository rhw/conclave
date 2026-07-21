"""Point-estimate candidates + upstream tier scoring for the TS2 bake-off.

Pure numpy: NO qp / RAIL / conclave.metrics import (login-node-safe, like
``conclave.realdata``). PDF inputs are 2-D arrays ``(n_obj, n_grid)`` on a shared
``grid`` (1-D also accepted for a single object); rows need not be normalized —
each statistic normalizes internally where it needs to. See
``docs/specs/2026-07-18-pointest-bakeoff.md`` for the candidate definitions.

As of the 2026-07 C3 revision this module is DEPLOYMENT code, not just
evaluation code: ``prior_shift_mode`` is called directly by
``conclave.submission.priorshift_zmode`` (config ``pointest="priorshift"``,
shipped as TS2's delivered ``ancil["zmode"]``) and by the Sherlock revision
driver ``scripts/revise_ts2_pointest.py``, both via the identical code path.
It is shipped inside the submitted package (not left behind as an
evaluation-only script).
"""
import numpy as np

# Upstream point-metric tier thresholds, copied from
# LSSTDESC/pz_data_challenge src/pz_data_challenge/scoring.py. Each metric maps to
# a list of [lo, hi] ranges ordered TIGHTEST -> LOOSEST; a value v passes tier k
# iff range[0] <= v <= range[1] (upstream's inclusive-bounds convention). Points =
# 3/2/1 for the tightest tier passed (index 0/1/2), 0 if it passes none.
_TIER_THRESHOLDS = {
    "mean": [[-0.01, 0.01], [-0.025, 0.025], [-0.05, 0.05]],
    "std": [[0.0, 0.025], [0.0, 0.05], [0.0, 0.10]],
    "abs_outlier_rate": [[0.0, 0.025], [0.0, 0.1], [0.0, 0.2]],
    "outlier": [[0.0, 0.2], [0.0, 0.4], [0.0, 0.6]],
}
_TIER_POINTS = (3, 2, 1)  # points for tier index 0, 1, 2


def z_mode(pdf, grid):
    """Grid argmax (parity with qp ``.mode()`` to grid resolution)."""
    pdf = np.asarray(pdf, float)
    grid = np.asarray(grid, float)
    return grid[np.argmax(pdf, axis=-1)]


def z_mean(pdf, grid):
    """Normalized first moment: int z p(z) dz / int p(z) dz (per row)."""
    pdf = np.asarray(pdf, float)
    grid = np.asarray(grid, float)
    num = np.trapezoid(pdf * grid, grid, axis=-1)
    den = np.trapezoid(pdf, grid, axis=-1)
    return num / den


def _cumtrapz(pdf, grid):
    """Cumulative trapezoid along the last axis, leading 0 (pure numpy)."""
    dz = np.diff(grid)
    seg = 0.5 * (pdf[..., 1:] + pdf[..., :-1]) * dz
    lead = np.zeros(pdf.shape[:-1] + (1,))
    return np.concatenate([lead, np.cumsum(seg, axis=-1)], axis=-1)


def z_median(pdf, grid):
    """z at CDF = 0.5, linear-interpolated on the (cumulative-trapezoid) CDF."""
    pdf = np.asarray(pdf, float)
    grid = np.asarray(grid, float)
    cdf = _cumtrapz(pdf, grid)
    cdf = cdf / cdf[..., -1:]
    if pdf.ndim == 1:
        return np.interp(0.5, cdf, grid)
    return np.array([np.interp(0.5, c, grid) for c in cdf])


def _gaussian_smooth(y, grid, sigma_z):
    """Convolve a 1-D grid array with a Gaussian kernel of width sigma_z (z-units)."""
    dz = float(np.mean(np.diff(grid)))
    half = int(np.ceil(4.0 * sigma_z / dz))
    offsets = np.arange(-half, half + 1) * dz
    kernel = np.exp(-0.5 * (offsets / sigma_z) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def _unit(y):
    s = y.sum()
    return y / s if s > 0 else y


def prior_shift_mode(pdf, grid, z_labeled, mix_weights, sigma_z=0.05, rmax=5.0):
    """Spec C3: mode of ``pdf * r(z)``, r = clip(N_hat_test / N_train, 1/rmax, rmax).

    N_hat_test = per-object ``mix_weights``-weighted mean of the eval PDF rows
    (label-free); N_train = grid histogram of ``z_labeled`` (spec-z). Both are
    Gaussian-smoothed on the grid with width ``sigma_z`` and normalized to unit
    sum before the ratio, so identical train/test distributions give r == 1.
    """
    pdf = np.asarray(pdf, float)
    was_1d = pdf.ndim == 1
    pdf = np.atleast_2d(pdf)
    grid = np.asarray(grid, float)
    z_labeled = np.asarray(z_labeled, float)
    w = np.asarray(mix_weights, float)
    w = w / w.sum()

    n_test = np.tensordot(w, pdf, axes=(0, 0))          # (n_grid,)
    edges = 0.5 * (grid[:-1] + grid[1:])
    idx = np.clip(np.searchsorted(edges, z_labeled), 0, len(grid) - 1)
    n_train = np.zeros(len(grid))
    np.add.at(n_train, idx, 1.0)

    n_test = _unit(_gaussian_smooth(n_test, grid, sigma_z))
    n_train = _unit(_gaussian_smooth(n_train, grid, sigma_z))

    with np.errstate(divide="ignore", invalid="ignore"):
        r = n_test / n_train
    r = np.where(np.isfinite(r), r, rmax)
    r = np.clip(r, 1.0 / rmax, rmax)

    out = z_mode(pdf * r, grid)
    return out[0] if was_1d else out


def switched(base, alt, mask):
    """Elementwise switch: ``alt`` where ``mask`` else ``base`` (serves C4/C4b/C5)."""
    return np.where(mask, np.asarray(alt), np.asarray(base))


def tier_points(metric_name, value):
    """Points {0..3} for one point-metric value under the upstream tier ladder."""
    ranges = _TIER_THRESHOLDS[metric_name]
    for i, (lo, hi) in enumerate(ranges):          # tightest -> loosest
        if lo <= value <= hi:
            return _TIER_POINTS[i]
    return 0


def point_tier_table(row):
    """Per-metric tier points for a score row; only the four scored point metrics."""
    return {m: tier_points(m, float(row[m])) for m in _TIER_THRESHOLDS if m in row}
