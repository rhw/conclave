import numpy as np
import pytest
import qp
from conclave import metrics


def _scattered_ensemble(z_true, pred_sigma, err_scale, seed):
    """Gaussian PDFs whose MEANS are scattered around z_true by err_scale,
    with predicted width pred_sigma. Calibrated iff pred_sigma == err_scale:
    then PIT = Phi(-noise/pred_sigma) with noise~N(0,err_scale) is Uniform(0,1).
    Over-confident (pred_sigma << err_scale) -> PIT piles near 0 and 1."""
    rng = np.random.default_rng(seed)
    means = z_true + rng.normal(0.0, err_scale, size=len(z_true))
    ens = qp.Ensemble(qp.stats.norm,
                      data=dict(loc=means.reshape(-1, 1),
                                scale=np.full((len(z_true), 1), pred_sigma)))
    ens.set_ancil({"zmode": means})   # point estimate = PDF mean
    return ens


def test_score_keys_and_finiteness():
    rng = np.random.default_rng(0)
    z = rng.uniform(0.1, 1.5, 3000)
    ens = _scattered_ensemble(z, pred_sigma=0.02, err_scale=0.02, seed=1)  # calibrated
    s = metrics.score(ens, z, z_grid=np.linspace(0.0, 3.0, 301))
    for k in ["mean", "std", "outlier", "CDELoss", "PIT_ks", "PIT_rmse", "PIT_kld"]:
        assert k in s and np.isfinite(s[k])
    assert abs(s["mean"]) < 0.01 and s["std"] < 0.05 and s["outlier"] < 0.02
    assert s["PIT_ks"] < 0.1     # calibrated -> PIT near uniform


def test_overconfident_raises_pit_ks():
    rng = np.random.default_rng(1)
    z = rng.uniform(0.1, 1.5, 3000)
    err = 0.05
    calibrated = _scattered_ensemble(z, pred_sigma=err,  err_scale=err, seed=2)
    overconf   = _scattered_ensemble(z, pred_sigma=0.01, err_scale=err, seed=2)
    ks_cal = metrics.score(calibrated, z)["PIT_ks"]
    ks_over = metrics.score(overconf, z)["PIT_ks"]
    assert ks_over > ks_cal      # too-narrow PDFs are less calibrated
    assert ks_cal < 0.1          # calibrated really is near-uniform


def test_score_arrays_matches_score():
    """score_arrays (the post-hoc per-bin path) reproduces score's point + PIT
    metrics exactly from arrays, and CDELoss from per-object grid terms."""
    grid = np.linspace(0.0, 3.0, 301)
    rng = np.random.default_rng(3)
    z_true = rng.uniform(0.2, 1.5, 400)
    ens = _scattered_ensemble(z_true, pred_sigma=0.08, err_scale=0.08, seed=4)
    full = metrics.score(ens, z_true, z_grid=grid)
    pit = metrics._pit_values(ens, z_true)
    p = np.asarray(ens.pdf(grid))
    cde = (np.trapezoid(p * p, grid, axis=1)
           - 2.0 * np.asarray(ens.pdf(z_true.reshape(-1, 1))).squeeze())
    arr = metrics.score_arrays(np.asarray(ens.ancil["zmode"]), z_true, pit=pit, cde=cde)
    for k in ("mean", "std", "outlier", "abs_outlier_rate", "PIT_ks", "PIT_rmse", "PIT_kld"):
        assert np.isclose(arr[k], full[k], rtol=0, atol=1e-12), k
    assert np.isclose(arr["CDELoss"], full["CDELoss"], rtol=0.05, atol=0.05)
    assert arr["n_eval"] == 400


def test_score_arrays_official_basket():
    rng = np.random.default_rng(3)
    n = 4000
    zt = rng.uniform(0.05, 2.5, n)
    zm = zt + rng.normal(0, 0.02, n) * (1 + zt)
    pit = rng.beta(1.3, 1.0, n)          # deliberately non-uniform
    cde = rng.normal(-2.0, 0.3, n)

    base = metrics.score_arrays(zm, zt, pit=pit, cde=cde)
    off = metrics.score_arrays(zm, zt, pit=pit, cde=cde, official=True)

    # default output unchanged: no official keys, and shared keys identical
    for k in ("CvM", "ksamp", "PIT_outlier"):
        assert k not in base
        assert k in off
    for k in base:
        assert base[k] == off[k]

    # official values match _pit_official on the clipped pit
    expect = metrics._pit_official(np.clip(pit, 0.0, 1.0))
    for k, v in expect.items():
        assert off[k] == pytest.approx(v, nan_ok=True)

    # non-uniform PIT must register: CvM well above the uniform-sample scale
    uni = metrics.score_arrays(zm, zt, pit=rng.uniform(0, 1, n), official=True)
    assert off["CvM"] > 5 * max(uni["CvM"], 1e-3)
