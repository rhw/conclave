"""Post-hoc recalibrators. Phase 1: identity. Fit and apply must be disjoint."""
import numpy as np
import qp
from scipy.interpolate import interp1d
import qp.parameterizations.quant.quant as _qp_quant

# calpit (0.1.2) predates NumPy 2.0 and uses np.Inf / np.trapz, both removed in NumPy 2.x.
# Shim them at import so calpit's internal CDE->PIT (np.trapz) and early-stopping (np.Inf) work.
if not hasattr(np, "Inf"):
    np.Inf = np.inf            # type: ignore[attr-defined]
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid    # type: ignore[attr-defined]


class IdentityRecalibrator:
    def __init__(self):
        self._fit_idx = set()   # always guard; empty set is disjoint with everything

    def fit(self, ensemble, z_true, fit_idx) -> None:
        self._fit_idx = set(np.asarray(fit_idx).tolist())

    def apply(self, ensemble, apply_idx):
        apply_idx = np.asarray(apply_idx)
        assert self._fit_idx.isdisjoint(apply_idx.tolist()), "recal leakage: fit/apply overlap"
        return ensemble[apply_idx]


class GlobalPITRecalibrator:
    """Closed-form global PIT recalibrator.

    Fits the empirical CDF of PIT values on a training split, then uses its
    inverse to remap the quantile levels of each PDF in the apply split so
    that the marginal PIT distribution is closer to uniform.

    qp quant-API notes (verified on qp 1.1.2 / Sherlock):
    - ``qp.convert(ens, "quant", quants=...)`` pads 0 and 1 at both ends,
      so 101 input quant values → 103 stored.
    - ``q.dist.quants`` returns the padded array (shape (103,)).
    - ``q.dist.locs`` returns the padded locs array (shape (npdf, 103)).
    - To rebuild: pass ``quants=inner[1:-1], locs=padded_locs[:,1:-1]``
      so the constructor pads once and gets the right shape.
    """

    def __init__(self):
        self._fit_idx = set()
        self._F_pit = None

    def fit(self, ensemble, z_true, fit_idx) -> None:
        fit_idx = np.asarray(fit_idx)
        self._fit_idx = set(fit_idx.tolist())
        zt = np.asarray(z_true)[fit_idx]
        pit = np.asarray(ensemble[fit_idx].cdf(zt.reshape(-1, 1))).squeeze()
        xs = np.sort(np.clip(pit, 0, 1))
        ys = np.linspace(0, 1, len(xs))
        self._F_pit = interp1d(xs, ys, bounds_error=False, fill_value=(0.0, 1.0))

    def apply(self, ensemble, apply_idx):
        apply_idx = np.asarray(apply_idx)
        assert self._fit_idx.isdisjoint(apply_idx.tolist()), \
            "recal leakage: fit/apply overlap"
        sub = ensemble[apply_idx]
        # Evaluate the original CDF on a dense uniform grid.
        # Using CDF + remapping is more robust than working in quantile space
        # because qp's quant interpolator requires strictly-monotone locs per
        # object, which FlexZBoost PDFs sometimes violate near z=0.
        y_grid = np.linspace(0.0, 3.0, 301)
        # cdf_orig: shape (npdf, 301) — CDF evaluated at each grid point
        cdf_orig = np.clip(np.asarray(sub.cdf(y_grid)), 0.0, 1.0)
        # Remap: new_cdf_i(z) = F_pit(cdf_orig_i(z))
        cdf_new = np.clip(self._F_pit(cdf_orig), 0.0, 1.0)
        # Recover PDF as derivative of the remapped CDF (finite differences)
        # and clamp negatives to zero.
        dz = y_grid[1] - y_grid[0]
        pdf_new = np.diff(cdf_new, axis=1) / dz          # shape (npdf, 300)
        pdf_new = np.clip(pdf_new, 0.0, None)
        # Align grid: use midpoints so xvals and yvals have the same length
        x_mid = 0.5 * (y_grid[:-1] + y_grid[1:])         # shape (300,)
        return qp.Ensemble(qp.interp,
                           data={"xvals": x_mid, "yvals": pdf_new})


class MagBinnedPITRecalibrator:
    """Magnitude-binned global-PIT: a closed-form empirical-PIT-CDF remap fit
    *separately* within quantile bins of a photometry feature (default i-band mag).

    Middle ground between ``GlobalPITRecalibrator`` (one global remap — fixes only the
    *marginal* PIT) and ``CalPITRecalibrator`` (feature-conditional MLP — overfits the
    small calib set in our setup, worsening PIT-KS ~3×; see 2026-06-22/30 probes). Each
    bin gets its own GlobalPIT-style remap, so the correction varies with magnitude
    without any neural net to overfit. Objects with a NaN feature, or in a bin that had
    too few fit objects, fall back to the pooled global remap.

    Bin edges are quantiles of the feature on the *fit* (calib) set only; the same edges
    are reused at apply. Leakage-guarded like the other recalibrators.
    """

    def __init__(self, feature_key="feat_mag_i_lsst", n_bins=4, min_per_bin=200,
                 y_grid=None):
        self.feature_key = feature_key
        self.n_bins = int(n_bins)
        self.min_per_bin = int(min_per_bin)
        self.y_grid = (np.linspace(0.0, 3.0, 301)
                       if y_grid is None else np.asarray(y_grid))
        self._fit_idx = set()
        self._edges = None          # inner bin edges (length n_bins-1)
        self._F_bins = []           # per-bin interp1d remaps
        self._F_global = None       # pooled fallback remap

    def _feat(self, ensemble, idx):
        return np.asarray(ensemble.ancil[self.feature_key], dtype=float)[idx]

    @staticmethod
    def _fit_pit_cdf(pit):
        xs = np.sort(np.clip(pit, 0.0, 1.0))
        ys = np.linspace(0.0, 1.0, len(xs))
        return interp1d(xs, ys, bounds_error=False, fill_value=(0.0, 1.0))

    def fit(self, ensemble, z_true, fit_idx) -> None:
        fit_idx = np.asarray(fit_idx)
        self._fit_idx = set(fit_idx.tolist())
        zt = np.asarray(z_true)[fit_idx]
        pit = np.asarray(ensemble[fit_idx].cdf(zt.reshape(-1, 1))).squeeze()
        feat = self._feat(ensemble, fit_idx)
        finite = np.isfinite(feat)

        # Pooled global remap (used as fallback for NaN-feature / sparse bins).
        self._F_global = self._fit_pit_cdf(pit)

        # Quantile bin edges on finite features only.
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]   # inner edges
        self._edges = (np.quantile(feat[finite], qs) if finite.any()
                       else np.array([]))
        binid = np.digitize(feat, self._edges)              # 0..n_bins-1 for finite
        self._F_bins = []
        for b in range(self.n_bins):
            sel = finite & (binid == b)
            self._F_bins.append(self._fit_pit_cdf(pit[sel])
                                if sel.sum() >= self.min_per_bin else self._F_global)

    def apply(self, ensemble, apply_idx):
        apply_idx = np.asarray(apply_idx)
        assert self._fit_idx.isdisjoint(apply_idx.tolist()), \
            "recal leakage: fit/apply overlap"
        sub = ensemble[apply_idx]
        y_grid = np.asarray(self.y_grid, dtype=float)
        cdf_orig = np.clip(np.asarray(sub.cdf(y_grid)), 0.0, 1.0)   # (n, ngrid)
        feat = self._feat(ensemble, apply_idx)
        binid = np.digitize(feat, self._edges)
        cdf_new = np.empty_like(cdf_orig)
        # Apply each bin's remap to its objects in one vectorized interp call.
        finite = np.isfinite(feat)
        for b in range(self.n_bins):
            rows = np.where(finite & (binid == b))[0]
            if len(rows):
                cdf_new[rows] = self._F_bins[b](cdf_orig[rows])
        fallback = np.where(~finite)[0]
        if len(fallback):
            cdf_new[fallback] = self._F_global(cdf_orig[fallback])
        cdf_new = np.clip(cdf_new, 0.0, 1.0)
        dz = y_grid[1] - y_grid[0]
        pdf_new = np.clip(np.diff(cdf_new, axis=1) / dz, 0.0, None)
        x_mid = 0.5 * (y_grid[:-1] + y_grid[1:])
        return qp.Ensemble(qp.interp, data={"xvals": x_mid, "yvals": pdf_new})


class CalPITRecalibrator:
    """Instance-wise PIT recalibrator using the calpit (PyTorch) package.

    Learns a *local* PIT correction that depends on per-object photometry
    features stored in ``ensemble.ancil["feat_<col>"]``.  The correction
    is parameterised by calpit's built-in MLP which maps
    ``[coverage, *features] -> sigmoid`` and is fitted via early-stopping BCE.

    NaN handling (required for u-band non-detections):
    - In ``fit``: per-feature medians are computed over the fit set (ignoring
      NaN) and stored.  NaNs are replaced with medians before standardisation.
    - In ``apply``: the stored medians are used for the same imputation step.
    """

    def __init__(self, feature_keys, seed=0, y_grid=None,
                 hidden_layers=None, n_epochs=200, patience=15, oversample=1):
        self.feature_keys = list(feature_keys)
        self.seed = seed
        self.y_grid = (np.linspace(0.0, 3.0, 301)
                       if y_grid is None else np.asarray(y_grid))
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 64]
        self.n_epochs = n_epochs
        self.patience = patience
        # Coverage oversampling: each calib object contributes `oversample` random-coverage
        # training pairs per epoch. oversample=1 (calpit default) badly under-samples the PIT
        # CDF for a flexible MLP; raising it is the lever for the under-trained-map failure.
        self.oversample = oversample

        self._fit_idx = set()
        self._calpit = None
        # NaN imputation + standardisation parameters (set during fit)
        self._medians = None   # shape (n_features,)
        self._mu = None        # shape (n_features,)
        self._sd = None        # shape (n_features,)

    def _features(self, ensemble, idx):
        """Extract, impute NaN, and standardise feature matrix for *idx*."""
        cols = [np.asarray(ensemble.ancil[k], dtype=float)[idx]
                for k in self.feature_keys]
        X = np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)

        # Impute NaN with stored medians (must be set before calling apply)
        if self._medians is not None:
            nan_mask = np.isnan(X)
            if nan_mask.any():
                X = X.copy()
                for j in range(X.shape[1]):
                    col_nan = nan_mask[:, j]
                    if col_nan.any():
                        X[col_nan, j] = self._medians[j]

        # Standardise with stored mu/sd
        if self._mu is not None:
            X = (X - self._mu) / self._sd

        return X

    def fit(self, ensemble, z_true, fit_idx) -> None:
        # calpit uses np.Inf (removed in NumPy 2.0) — shim it before importing
        if not hasattr(np, "Inf"):
            np.Inf = np.inf  # type: ignore[attr-defined]
        from calpit import CalPit  # noqa: heavy import, deferred

        fit_idx = np.asarray(fit_idx)
        self._fit_idx = set(fit_idx.tolist())

        # --- raw feature extraction (no imputation/scaling yet) ---
        cols = [np.asarray(ensemble.ancil[k], dtype=float)[fit_idx]
                for k in self.feature_keys]
        X_raw = (np.column_stack(cols) if len(cols) > 1
                 else cols[0].reshape(-1, 1))

        # Compute and store per-feature medians (ignoring NaN)
        self._medians = np.nanmedian(X_raw, axis=0)

        # Impute NaN with medians
        nan_mask = np.isnan(X_raw)
        if nan_mask.any():
            X_raw = X_raw.copy()
            for j in range(X_raw.shape[1]):
                col_nan = nan_mask[:, j]
                if col_nan.any():
                    X_raw[col_nan, j] = self._medians[j]

        # Compute and store standardisation parameters
        self._mu = X_raw.mean(axis=0)
        self._sd = X_raw.std(axis=0) + 1e-8
        X_scaled = (X_raw - self._mu) / self._sd

        # Fit on the CDE evaluated on the SAME y_grid that apply()/transform() use, and let
        # calpit compute PIT internally. Passing a precomputed analytic pit_calib instead
        # mismatches transform()'s numerical-grid PIT basis and mis-corrects (it WORSENED
        # over-confident PITs in the 2026-06-22 probe); cde_calib+y_calib+y_grid is consistent.
        zt = np.asarray(z_true)[fit_idx]
        cde_calib = np.asarray(ensemble[fit_idx].pdf(self.y_grid))

        # The PIT-recalibration map is a CDF (coverage -> coverage), so it MUST be monotonic
        # in coverage. calpit's "mlp" is a plain non-monotonic net whose non-monotonic "CDF"
        # has a garbage derivative and WORSENS calibration (probes 2026-06-22). MonotonicNN
        # (UMNN) integrates a positive integrand over its FIRST input column — and calpit feeds
        # [coverage, *features] with coverage first — giving a valid monotonic map; sigmoid=True
        # bounds the output to [0, 1].
        import torch
        from calpit.nn import MonotonicNN
        torch.manual_seed(self.seed)
        n_features = X_scaled.shape[1]
        model = MonotonicNN(n_features + 1, self.hidden_layers, sigmoid=True)
        self._calpit = CalPit(model=model)
        # Use a fixed temp checkpoint path to avoid clobbering concurrent runs
        import tempfile, os
        checkpt = os.path.join(tempfile.gettempdir(),
                               f"calpit_checkpoint_{self.seed}.pt")
        self._calpit.fit(
            X_scaled,
            y_calib=zt,
            cde_calib=cde_calib,
            y_grid=self.y_grid,
            n_epochs=self.n_epochs,
            patience=self.patience,
            oversample=self.oversample,
            seed=self.seed,
            trace_func=lambda *a, **kw: None,   # silence training output
            checkpt_path=checkpt,
        )

    def apply(self, ensemble, apply_idx):
        apply_idx = np.asarray(apply_idx)
        assert self._fit_idx.isdisjoint(apply_idx.tolist()), \
            "recal leakage: fit/apply overlap"

        X_scaled = self._features(ensemble, apply_idx)

        # Get (N, n_grid) CDE array from the ensemble
        cde_test = np.asarray(ensemble[apply_idx].pdf(self.y_grid))

        # calpit.transform returns (N, n_grid) calibrated CDEs. NOTE (2026-06-22): the
        # recalibrated CDF F_new=r(F_old|x) need not span [0,1] (UMNN+sigmoid leaves r(0),r(1)
        # un-pinned), but qp.interp renormalises the PDF, which absorbs that affine offset — so
        # endpoint pinning is a no-op here. The operative failure is that the MLP-learned map r
        # is a poor estimate of the PIT-CDF (deviates from identity ~3x more than the true mild
        # miscalibration), so it over-corrects. See docs/journal/2026-06-22-calpit-rootcause.
        cde_new = self._calpit.transform(X_scaled, cde_test, self.y_grid)
        cde_new = np.clip(cde_new, 0.0, None)

        return qp.Ensemble(
            qp.interp,
            data={"xvals": self.y_grid, "yvals": cde_new},
        )


RECALIBRATORS = {
    "none": IdentityRecalibrator,
    "global_pit": GlobalPITRecalibrator,
    "magbinned_pit": MagBinnedPITRecalibrator,
    "calpit": CalPITRecalibrator,
}
