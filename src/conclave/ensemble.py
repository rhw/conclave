# src/pzdc/ensemble.py
"""Member combiner: common-grid resample, equal-weight + convex-QP weights.

CDE loss is a convex quadratic in the weight simplex (spec §2.1):
    L(w) = wᵀ A w − 2 bᵀ w,   A_jk = ⟨∫ p_j p_k dz⟩_i,   b_k = ⟨p_k(z_true,i)⟩_i
A is PSD ⇒ minimized on {w ≥ 0, Σw = 1} by a small QP (SLSQP).
"""
import numpy as np
import qp
from scipy.optimize import minimize

Z_GRID = np.linspace(0.0, 3.0, 301)


def _trapz(pdf, z_grid, axis=-1):
    """np.trapezoid (NumPy ≥ 2.0) with np.trapz fallback for older versions.
    Returns result with the integration axis removed (no keepdims support)."""
    fn = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return fn(pdf, z_grid, axis=axis)


def _renorm(pdf, z_grid):
    """Clip negatives and renormalize each row to unit trapezoid integral.

    Rows that are non-finite anywhere or have a non-positive/non-finite integral are
    replaced with a uniform density over the grid. This is the choke point that keeps a
    single degenerate MEMBER (e.g. a normalizing flow that diverges on noisy 1yr data and
    emits NaN p(z)) from poisoning the whole weighted combine — the bad member contributes
    "no information" (uniform) instead of NaN, and convex-QP weighting down-weights it.
    """
    pdf = np.clip(np.asarray(pdf, dtype=float), 0.0, None)   # NaN survives clip
    z_grid = np.asarray(z_grid, dtype=float)
    area = _trapz(pdf, z_grid, axis=-1)
    bad = ~np.isfinite(area) | (area <= 0) | ~np.isfinite(pdf).all(axis=-1)
    if np.any(bad):
        pdf = pdf.copy()
        pdf[bad] = 1.0 / (z_grid[-1] - z_grid[0])           # uniform fallback row(s)
        area = _trapz(pdf, z_grid, axis=-1)
    area = np.where(area > 0, area, 1.0)[..., np.newaxis]
    return pdf / area


def to_common_grid(ensembles, z_grid=Z_GRID):
    """Evaluate, clip, renormalize each member on z_grid → (K, npdf, ngrid)."""
    z_grid = np.asarray(z_grid, dtype=float)
    mats = []
    npdf = None
    for ens in ensembles:
        p = np.asarray(ens.pdf(z_grid))
        if npdf is None:
            npdf = p.shape[0]
        elif p.shape[0] != npdf:
            raise ValueError(f"member npdf mismatch: {p.shape[0]} != {npdf}")
        mats.append(_renorm(p, z_grid))
    return np.stack(mats, axis=0)


def equal_weight(member_pdfs):
    """Uniform mean over the member axis of a common-grid (K, npdf, ngrid) array."""
    return np.asarray(member_pdfs, dtype=float).mean(axis=0)


def combine(member_pdfs, weights, z_grid=Z_GRID, ancil=None):
    """Weighted member sum → renormalized qp.interp Ensemble.

    ``ancil`` (optional dict of per-object arrays, e.g. ``{"feat_mag_i_lsst": ...}``)
    is attached to the result so feature-aware recalibrators (``magbinned_pit``,
    ``calpit``) can read photometry features off the combined ensemble. Member PDFs
    lose their ancil in ``to_common_grid``, so the caller supplies it here.
    """
    member_pdfs = np.asarray(member_pdfs, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1, 1, 1)
    mixed = (w * member_pdfs).sum(axis=0)        # (npdf, ngrid)
    mixed = _renorm(mixed, np.asarray(z_grid, dtype=float))
    ens = qp.Ensemble(qp.interp, data={"xvals": np.asarray(z_grid), "yvals": mixed})
    if ancil:
        ens.set_ancil({k: np.asarray(v) for k, v in ancil.items()})
    return ens


def _cde_at_truth(member_pdfs, z_true, z_grid):
    """b_k contributions per object: p_k(z_true,i) by linear interp on the grid.

    Loops over the (small) member axis only; vectorized over objects via a
    shared interpolation index (np.interp's per-object form would need a 1-D
    fp, but each object has its own PDF row, so we interpolate manually).
    """
    K, npdf, ngrid = member_pdfs.shape
    z_true = np.asarray(z_true, dtype=float)
    # Locate each object's z_true between grid nodes (shared across members).
    idx = np.clip(np.searchsorted(z_grid, z_true) - 1, 0, ngrid - 2)  # (npdf,)
    z0 = z_grid[idx]
    z1 = z_grid[idx + 1]
    frac = np.clip((z_true - z0) / (z1 - z0), 0.0, 1.0)               # (npdf,)
    rows = np.arange(npdf)
    out = np.empty((K, npdf))
    for k in range(K):
        y0 = member_pdfs[k, rows, idx]
        y1 = member_pdfs[k, rows, idx + 1]
        out[k] = y0 + frac * (y1 - y0)
    return out


def optimal_weights(member_pdfs_calib, z_true_calib, z_grid=Z_GRID):
    """Minimize CDE loss wᵀAw − 2bᵀw on the simplex {w ≥ 0, Σw = 1} via SLSQP.

    Parameters
    ----------
    member_pdfs_calib : (K, npdf, ngrid) array
        Common-grid PDFs on the calibration split (from to_common_grid).
    z_true_calib : (npdf,) array
        True redshifts for the calibration objects.
    z_grid : 1-D array, optional
        The shared redshift grid.

    Returns
    -------
    w : (K,) array
        Non-negative weights summing to 1.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    P = np.asarray(member_pdfs_calib, dtype=float)   # (K, npdf, ngrid)
    K = P.shape[0]
    z_true = np.asarray(z_true_calib, dtype=float)

    # A_jk = mean_i ∫ p_j p_k dz   (Gram matrix averaged over objects)
    A = np.empty((K, K))
    for j in range(K):
        for k in range(j, K):
            prod = _trapz(P[j] * P[k], z_grid, axis=-1)   # (npdf,)
            A[j, k] = A[k, j] = prod.mean()

    # b_k = mean_i p_k(z_true,i)
    b = _cde_at_truth(P, z_true, z_grid).mean(axis=1)    # (K,)

    def obj(w):
        return w @ A @ w - 2.0 * b @ w

    def grad(w):
        return 2.0 * A @ w - 2.0 * b

    w0 = np.full(K, 1.0 / K)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bnds, constraints=cons)
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def member_disagreement(stacked, grid=Z_GRID, full=False):
    """Per-object disagreement among member PDFs — a free reliability flag.

    ``stacked`` is the (K, npdf, ngrid) common-grid member-CDE array from
    ``to_common_grid``.  Computed from members already built, so zero extra training.
    Two interpretable components:

      z_spread : std across the K members of each member's mean redshift
                 E[z]_k = ∫ z p_k dz (smoother than the mode), in redshift units —
                 the headline "members disagree by Δz".
      tv       : mean pairwise total-variation distance 0.5 ∫|p_i − p_j| dz over the
                 K(K−1)/2 member pairs, in [0, 1] — distributional, catches bimodal
                 disagreement the mean-spread misses.

    Returns the primary ``z_spread`` (npdf,) by default; with ``full=True`` returns
    ``{"z_spread": ..., "tv": ...}``.  K=1 (no pairs) → zeros.  numpy only (RAIL-free).
    """
    grid = np.asarray(grid, dtype=float)
    P = np.asarray(stacked, dtype=float)                       # (K, npdf, ngrid)
    K, npdf, _ = P.shape
    if K == 1:
        z = np.zeros(npdf)
        return {"z_spread": z, "tv": z.copy()} if full else z
    zmean = _trapz(P * grid[None, None, :], grid, axis=-1)     # (K, npdf)
    z_spread = zmean.std(axis=0)                               # population std over members
    if not full:
        return z_spread
    tv_sum = np.zeros(npdf)
    npairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            tv_sum += 0.5 * _trapz(np.abs(P[i] - P[j]), grid, axis=-1)
            npairs += 1
    return {"z_spread": z_spread, "tv": tv_sum / npairs}


def member_corr_matrix(member_pdfs, z_true, z_grid=Z_GRID):
    """Pearson correlation matrix of per-member redshift residuals.

    Uses the grid expectation zmean_k = ∫ z p_k dz as the redshift point
    estimate, then forms residuals r_k = zmean_k − z_true and returns the
    (K, K) correlation matrix.  Off-diagonal entries < 1 indicate that
    members are decorrelated, predicting ensemble payoff (spec §3.2).

    Parameters
    ----------
    member_pdfs : (K, npdf, ngrid) array
        Common-grid PDFs (from to_common_grid).
    z_true : (npdf,) array
        True redshifts.
    z_grid : 1-D array, optional
        The shared redshift grid.

    Returns
    -------
    C : (K, K) array
        Pearson correlation matrix with unit diagonal.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    P = np.asarray(member_pdfs, dtype=float)
    z_true = np.asarray(z_true, dtype=float)
    # zmean_k = ∫ z p_k dz, shape (K, npdf); PDFs already normalized
    zmean = _trapz(P * z_grid[None, None, :], z_grid, axis=-1)   # (K, npdf)
    resid = zmean - z_true[None, :]
    return np.corrcoef(resid)
