import numpy as np
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
