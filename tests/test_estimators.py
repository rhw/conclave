import numpy as np
import pytest
import qp
from conclave import data, estimators
from conclave.experiment import ESTIMATORS


# ---------------------------------------------------------------------------
# Unit tests for _sanitize_grid_pdfs (pure numpy/qp, no RAIL needed)
# ---------------------------------------------------------------------------

def _make_synthetic_interp_ens():
    """Build a 4-row qp.interp ensemble for sanitize testing.

    Row 0: good Gaussian-ish bump (should pass through unchanged in shape).
    Row 1: has a NaN → bad row, must be replaced with uniform.
    Row 2: all zeros → integral=0 → bad row, must be replaced with uniform.
    Row 3: good flat-top (should pass through unchanged in shape).
    """
    grid = np.linspace(0.0, 3.0, 301)
    ngrid = len(grid)
    yvals = np.zeros((4, ngrid), dtype=float)

    # Row 0: Gaussian bump centered at z=0.5
    yvals[0] = np.exp(-0.5 * ((grid - 0.5) / 0.1) ** 2)
    yvals[0] /= np.trapezoid(yvals[0], grid)  # normalize

    # Row 1: put a NaN in the middle → bad
    yvals[1] = np.exp(-0.5 * ((grid - 1.0) / 0.1) ** 2)
    yvals[1, 150] = np.nan

    # Row 2: all zeros → integral = 0 → bad
    yvals[2] = 0.0

    # Row 3: flat-top between 0.5 and 1.5
    mask = (grid >= 0.5) & (grid <= 1.5)
    yvals[3, mask] = 1.0
    yvals[3] /= np.trapezoid(yvals[3], grid)  # normalize

    ens = qp.Ensemble(qp.interp, data={"xvals": grid, "yvals": yvals})
    return ens, grid


def test_sanitize_grid_pdfs_fixes_bad_rows():
    """_sanitize_grid_pdfs must:
    - Replace NaN rows and zero-integral rows with uniform fallback.
    - Leave good rows (0 and 3) with the same PDF shape.
    - Ensure every row integrates to ≈1 after sanitize.
    - Return finite values for all rows.
    """
    from conclave.estimators import _sanitize_grid_pdfs

    ens, grid = _make_synthetic_interp_ens()
    san = _sanitize_grid_pdfs(ens, grid=grid)

    pdf_mat = np.asarray(san.pdf(grid))  # shape (4, 301)
    assert pdf_mat.shape == (4, 301), f"unexpected shape {pdf_mat.shape}"

    # All finite
    assert np.all(np.isfinite(pdf_mat)), "sanitized PDFs contain non-finite values"

    # All rows integrate to ≈1
    integrals = np.trapezoid(pdf_mat, grid, axis=1)
    np.testing.assert_allclose(integrals, 1.0, atol=1e-6,
                               err_msg="not all rows integrate to 1 after sanitize")

    # Good rows (0 and 3) are unchanged in shape (correlation > 0.9999 with original)
    orig_yvals = np.asarray(ens.pdf(grid))  # original (rows 1,2 are bad)
    for row in (0, 3):
        corr = np.corrcoef(pdf_mat[row], orig_yvals[row])[0, 1]
        assert corr > 0.9999, f"row {row} shape changed after sanitize (corr={corr:.6f})"

    # Bad rows (1 and 2) are replaced with uniform
    uniform_val = 1.0 / (3.0 - 0.0)  # 1/(zmax-zmin)
    for row in (1, 2):
        # uniform: all yvals equal (constant density over [0,3])
        row_pdf = pdf_mat[row]
        # After clip+renorm the uniform fallback stays uniform; check flatness
        std_over_mean = np.std(row_pdf) / np.mean(row_pdf)
        assert std_over_mean < 1e-6, (
            f"row {row} expected uniform fallback but got std/mean={std_over_mean:.3e}"
        )

PUBLIC = "/oak/stanford/orgs/kipac/users/risahw/pz_data_challenge/repo/public"

def _subsample(d, n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d["redshift"]), size=n, replace=False)
    return {k: v[idx] for k, v in d.items()}

def test_flexzboost_roundtrip_lsst():
    full = data.load_catalog(data.data_path("cardinal", "training", "10yr", PUBLIC))
    tr = _subsample(full, 2000, seed=1)
    ev = _subsample(full, 500, seed=2)
    model = estimators.train_flexzboost(tr, band_set="lsst", seed=7, name="t_inform")
    ens = estimators.estimate_flexzboost(model, ev, band_set="lsst", name="t_est")
    assert isinstance(ens, qp.Ensemble)
    assert ens.npdf == 500
    zmode = np.asarray(ens.ancil["zmode"]).squeeze()
    assert zmode.shape == (500,)
    assert np.all(np.isfinite(zmode))
    assert np.array_equal(np.asarray(ens.ancil["object_id"]).astype(int),
                          ev["object_id"].astype(int))


def test_flexzboost_ancil_has_features():
    full = data.load_catalog(data.data_path("cardinal", "training", "10yr", PUBLIC))
    tr = _subsample(full, 2000, seed=1); ev = _subsample(full, 400, seed=2)
    model = estimators.train_flexzboost(tr, band_set="lsst", seed=7, name="t_inf_feat")
    ens = estimators.estimate_flexzboost(model, ev, band_set="lsst", name="t_est_feat")
    from conclave import bands
    for col in bands.band_columns("lsst"):
        assert f"feat_{col}" in ens.ancil
        assert np.asarray(ens.ancil[f"feat_{col}"]).shape == (400,)


def test_gpz_roundtrip_lsst():
    full = data.load_catalog(data.data_path("cardinal", "training", "10yr", PUBLIC))
    tr = _subsample(full, 2000, seed=1); ev = _subsample(full, 400, seed=2)
    model = estimators.train_gpz(tr, band_set="lsst", seed=7, name="g_inf")
    ens = estimators.estimate_gpz(model, ev, band_set="lsst", name="g_est")
    assert isinstance(ens, qp.Ensemble) and ens.npdf == 400
    zmode = np.asarray(ens.ancil["zmode"]).squeeze()
    assert zmode.shape == (400,) and np.all(np.isfinite(zmode))
    assert "feat_mag_i_lsst" in ens.ancil


def test_pzflow_roundtrip_lsst():
    full = data.load_catalog(data.data_path("cardinal", "training", "10yr", PUBLIC))
    tr = _subsample(full, 2000, seed=1); ev = _subsample(full, 400, seed=2)
    model = estimators.train_pzflow(tr, band_set="lsst", seed=7, name="f_inf")
    ens = estimators.estimate_pzflow(model, ev, band_set="lsst", name="f_est")
    import qp, numpy as np
    assert isinstance(ens, qp.Ensemble) and ens.npdf == 400
    zmode = np.asarray(ens.ancil["zmode"]).squeeze()
    assert zmode.shape == (400,) and np.all(np.isfinite(zmode))
    grid = np.linspace(0.0, 3.0, 301)
    pdfs = np.asarray(ens.pdf(grid))
    integrals = np.trapezoid(pdfs, grid, axis=1) if hasattr(np, "trapezoid") else np.trapz(pdfs, grid, axis=1)
    assert np.median(integrals) > 0.9   # PDFs are valid normalized densities, not degenerate


@pytest.mark.parametrize("key", [
    "bpz", "cmnn", "knn", "tpz", "trainz", "dnf",
    pytest.param(
        "randomforest",
        marks=pytest.mark.xfail(
            reason="RandomForestEstimator absent from installed RAIL: "
                   "random_forest module only has RandomForestClassifier "
                   "(tomographic-bin classifier, no PDF output). "
                   "Raises NotImplementedError by design.",
            raises=NotImplementedError,
            strict=True,
        ),
    ),
])
def test_new_estimator_roundtrip_lsst(key):
    """Parametrized smoke test for the four new Phase-3 estimator wrappers.

    Informs on 500 training objects and estimates on 200 test objects using
    the lsst band set. Checks: result is a qp.Ensemble with the right npdf,
    the bulk of zmode is finite, object_id matches, and the bulk of PDFs
    integrate to ~1 (mean fraction integrating >0.9 is itself >0.9 — guarding
    against the zero-integral NaN-nondetect failure mode).
    """
    N_TRAIN, N_TEST = 500, 200
    train_fn, est_fn = ESTIMATORS[key]
    full = data.load_catalog(data.data_path("cardinal", "training", "10yr", PUBLIC))
    tr = _subsample(full, N_TRAIN, seed=10)
    ev = _subsample(full, N_TEST, seed=11)
    model = train_fn(tr, band_set="lsst", seed=42, name=f"{key}_inf")
    ens = est_fn(model, ev, band_set="lsst", name=f"{key}_est")
    assert isinstance(ens, qp.Ensemble)
    assert ens.npdf == N_TEST
    zmode = np.asarray(ens.ancil["zmode"]).squeeze()
    assert zmode.shape == (N_TEST,)
    # BPZ non-finite PDFs (~1% of objects) are sanitized by _sanitize_grid_pdfs in estimate_bpz;
    # all other estimators produce finite PDFs natively — require all-finite zmode.
    assert np.all(np.isfinite(zmode)), f"{key}: non-finite zmode found (sanitize or zero-integral failure?)"
    assert np.array_equal(
        np.asarray(ens.ancil["object_id"]).astype(int),
        ev["object_id"].astype(int),
    )
    grid = np.linspace(0.0, 3.0, 301)
    pdfs = np.asarray(ens.pdf(grid))
    integrals = np.trapezoid(pdfs, grid, axis=1) if hasattr(np, "trapezoid") else np.trapz(pdfs, grid, axis=1)
    bulk_ok = np.mean(integrals > 0.9)
    assert bulk_ok > 0.9, f"{key}: only {bulk_ok:.1%} of PDFs integrate >0.9 (zero-integral failure?)"
