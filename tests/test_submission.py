"""Tests for conclave.submission — Config, _combine, train/infer, run_taskset_1_*.

Light tests (no RAIL): monkeypatch _load_catalog + ESTIMATORS.
Heavy test (@slurm): real FlexZBoost+PZFlow on Sherlock Slurm only.
"""
import numpy as np
import qp
import pytest
from conclave import submission


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ens(n, loc0=0.5):
    """Small qp.interp Ensemble for unit tests."""
    grid = np.linspace(0.0, 3.0, 301)
    locs = np.linspace(loc0, loc0 + 0.5, n).reshape(-1, 1)
    yvals = np.exp(-0.5 * ((grid[None, :] - locs) / 0.1) ** 2)
    yvals /= np.trapezoid(yvals, grid, axis=1).reshape(-1, 1)
    e = qp.Ensemble(qp.interp, data={"xvals": grid, "yvals": yvals})
    e.set_ancil({"zmode": locs.squeeze(),
                 "object_id": np.arange(n).astype(int)})
    return e


def _install_stub_estimators(submission_mod, monkeypatch, n_train, n_test):
    """Monkeypatch _load_catalog + ESTIMATORS so tests run without RAIL."""
    rng = np.random.default_rng(1)

    def _fake_catalog(path):
        n = n_test if "test" in path else n_train
        return {"object_id": np.arange(n).astype(float),
                "redshift": rng.uniform(0.1, 1.5, n),
                "mag_i_lsst": rng.uniform(20, 24, n)}

    def _pair(est_name):
        def _t(train_dict, band_set, seed, name):
            return ("M", est_name)

        def _e(model, data_dict, band_set, name):
            m = len(data_dict["object_id"])
            grid = submission_mod.Z_GRID
            locs = data_dict.get("redshift", np.full(m, 0.6)).reshape(-1, 1)
            y = np.exp(-0.5 * ((grid[None, :] - locs) / 0.12) ** 2)
            y /= np.trapezoid(y, grid, axis=1).reshape(-1, 1)
            e = qp.Ensemble(qp.interp, data={"xvals": grid, "yvals": y})
            e.set_ancil({"zmode": locs.squeeze(),
                         "object_id": np.asarray(data_dict["object_id"]).astype(int)})
            return e
        return (_t, _e)

    monkeypatch.setattr(submission_mod, "_load_catalog", _fake_catalog)
    monkeypatch.setattr(
        submission_mod, "ESTIMATORS",
        {n: _pair(n) for n in ["flexzboost", "pzflow", "gpz"]}, raising=False)


# ---------------------------------------------------------------------------
# Task-1 tests: Config + _combine
# ---------------------------------------------------------------------------

def test_default_config_values():
    """DEFAULT_CONFIG matches the winning combination from Phase-3 ensemble."""
    c = submission.DEFAULT_CONFIG
    assert c.members == ["pzflow", "gpz", "flexzboost"]
    assert c.band_set == "lsst_roman"
    assert c.weights == "optimal"
    assert c.recal == "global_pit"


def test_combine_equal_weight_is_mean_and_normalized():
    grid = np.linspace(0.0, 3.0, 301)
    a, b = _ens(50, 0.4), _ens(50, 0.9)
    out = submission._combine([a, b], "equal")
    assert isinstance(out, qp.Ensemble)
    assert out.npdf == 50
    mean_pdf = 0.5 * (np.asarray(a.pdf(grid)) + np.asarray(b.pdf(grid)))
    np.testing.assert_allclose(np.asarray(out.pdf(grid)), mean_pdf, atol=1e-6)
    integ = np.trapezoid(np.asarray(out.pdf(grid)), grid, axis=1)
    assert np.allclose(integ, 1.0, atol=1e-3)


def test_combine_array_weights():
    grid = np.linspace(0.0, 3.0, 301)
    a, b = _ens(20, 0.4), _ens(20, 0.9)
    out = submission._combine([a, b], np.array([0.75, 0.25]))
    expect = 0.75 * np.asarray(a.pdf(grid)) + 0.25 * np.asarray(b.pdf(grid))
    np.testing.assert_allclose(np.asarray(out.pdf(grid)), expect, atol=1e-6)


def test_combine_single_member_passthrough():
    a = _ens(10)
    out = submission._combine([a], "equal")
    assert out.npdf == 10


# ---------------------------------------------------------------------------
# Task-2 tests: train/infer (light, RAIL-free)
# ---------------------------------------------------------------------------

def test_infer_reattaches_object_id_and_applies_recal(monkeypatch):
    """Light, RAIL-free: stub the estimator layer so we exercise train/infer
    plumbing (combine -> recal -> object_id ancil) with equal weights."""
    n_train, n_test = 400, 60
    rng = np.random.default_rng(0)

    def _fake_catalog(path):
        n = n_test if "test" in path else n_train
        return {"object_id": np.arange(n).astype(float),
                "redshift": rng.uniform(0.1, 1.5, n),
                "mag_i_lsst": rng.uniform(20, 24, n)}

    def _fake_train(estimator_name):
        def _t(train_dict, band_set, seed, name):
            return ("MODEL", estimator_name)
        return _t

    def _fake_est(estimator_name):
        def _e(model, data_dict, band_set, name):
            m = len(data_dict["object_id"])
            grid = submission.Z_GRID
            locs = (data_dict.get("redshift", np.full(m, 0.6))
                    + 0.02).reshape(-1, 1)
            y = np.exp(-0.5 * ((grid[None, :] - locs) / 0.1) ** 2)
            y /= np.trapezoid(y, grid, axis=1).reshape(-1, 1)
            e = qp.Ensemble(qp.interp, data={"xvals": grid, "yvals": y})
            e.set_ancil({"zmode": locs.squeeze(),
                         "object_id": np.asarray(data_dict["object_id"]).astype(int)})
            return e
        return _e

    monkeypatch.setattr(submission, "_load_catalog", _fake_catalog)
    monkeypatch.setattr(
        submission, "ESTIMATORS",
        {m: (_fake_train(m), _fake_est(m)) for m in ["flexzboost", "pzflow"]},
        raising=False)

    cfg = submission.Config(members=["flexzboost", "pzflow"],
                            band_set="lsst", weights="equal", recal="global_pit")
    bundle = submission.train_submission_model("train.hdf5", cfg)
    assert set(["models", "weights", "recal", "config"]).issubset(bundle.keys())
    ens = submission.infer(bundle, "test.hdf5")
    assert isinstance(ens, qp.Ensemble)
    assert ens.npdf == n_test
    assert "object_id" in (ens.ancil or {})
    assert np.asarray(ens.ancil["object_id"]).dtype.kind in "iu"


def test_optimal_weights_frozen_in_bundle(monkeypatch):
    """With weights='optimal' and 2 stub members, bundle['weights'] must be a
    length-2 simplex vector (>=0, sums to ~1), and infer must use it (not refit).
    The returned ensemble has npdf == n_test and int object_id ancil."""
    n_train, n_test = 400, 60
    _install_stub_estimators(submission, monkeypatch, n_train=n_train, n_test=n_test)

    cfg = submission.Config(members=["flexzboost", "pzflow"],
                            band_set="lsst", weights="optimal", recal="none")
    bundle = submission.train_submission_model("train.hdf5", cfg)

    # Frozen weights must be a simplex vector
    w = bundle["weights"]
    assert isinstance(w, np.ndarray), f"expected ndarray, got {type(w)}"
    assert w.shape == (2,), f"expected shape (2,), got {w.shape}"
    assert np.all(w >= 0), f"weights must be non-negative: {w}"
    assert abs(w.sum() - 1.0) < 1e-6, f"weights must sum to 1: {w}"

    # infer uses the frozen weights (not refit) and returns the right shape
    ens = submission.infer(bundle, "test.hdf5")
    assert ens.npdf == n_test
    assert "object_id" in (ens.ancil or {})
    assert np.asarray(ens.ancil["object_id"]).dtype.kind in "iu"


def test_run_training_and_estimation_writes_readable_qp(tmp_path, monkeypatch):
    _install_stub_estimators(submission, monkeypatch, n_train=300, n_test=40)
    out = str(tmp_path / "estimate_cardinal_10yr.hdf5")
    cfg = submission.Config(members=["flexzboost"], band_set="lsst",
                            weights="equal", recal="none")
    submission.run_taskset_1_training_and_estimation(
        "train.hdf5", "test.hdf5", out, config=cfg)
    import os
    assert os.path.exists(out)
    e = qp.read(out)
    assert e.npdf == 40
    assert "object_id" in (e.ancil or {})


# ---------------------------------------------------------------------------
# Heavy test: real RAIL estimators on Sherlock Slurm
# ---------------------------------------------------------------------------

PUBLIC = "/oak/stanford/orgs/kipac/users/risahw/pz_data_challenge/repo/public"


@pytest.mark.slurm
def test_run_training_and_estimation_real_rail():
    """Heavy: real FlexZBoost+PZFlow. RUN ON SHERLOCK SLURM ONLY (login-node SIGKILL)."""
    import os
    import tempfile
    import tables_io
    from conclave import submission, data

    tr = data.data_path("cardinal", "training", "10yr", PUBLIC)
    full = data.load_catalog(tr)
    idx = np.random.default_rng(0).choice(len(full["redshift"]), 800, replace=False)
    sub = {k: v[idx] for k, v in full.items()}
    td = tempfile.mkdtemp()
    trp = os.path.join(td, "train.hdf5")
    tep = os.path.join(td, "test.hdf5")
    tables_io.write(sub, trp)
    tables_io.write({k: v for k, v in sub.items() if k != "redshift"}, tep)
    out = os.path.join(td, "out.hdf5")
    cfg = submission.Config(members=["flexzboost", "pzflow"],
                            band_set="lsst", weights="equal", recal="global_pit")
    submission.run_taskset_1_training_and_estimation(trp, tep, out, config=cfg)
    e = qp.read(out)
    assert e.npdf == 800 and "object_id" in (e.ancil or {})
    grid = np.linspace(0, 3, 301)
    integ = np.trapezoid(np.asarray(e.pdf(grid)), grid, axis=1)
    assert np.median(integ) > 0.9
