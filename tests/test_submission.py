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


def _install_stub_estimators(submission_mod, monkeypatch, n_train, n_test,
                             test_has_redshift=True, test_mag_lo=20.0, test_mag_hi=24.0,
                             test_color_shift=0.0):
    """Monkeypatch _load_catalog + ESTIMATORS so tests run without RAIL.

    test_has_redshift=False mimics the REAL blind test-file schema (no
    ``redshift`` column) — the TS2 tests use it. test_mag_{lo,hi} set the test
    i-mag range independently of training; set them fainter than the training
    max to exercise the beyond-calib-support regime. All lsst_roman band
    columns are present (mag_i + correlated noise) so deficit-conditioned
    recals can build adjacent-band colors; test_color_shift displaces every
    other test band by that amount, pushing test COLORS out of the labeled
    support (positive label-free deficit)."""

    _BANDS = ([f"mag_{b}_lsst" for b in "ugrizy"]
              + ["mag_Y_roman", "mag_J_roman", "mag_H_roman"])

    def _fake_catalog(path):
        is_test = "test" in path
        n = n_test if is_test else n_train
        rng = np.random.default_rng(1)          # per-call: same path -> same catalog
        mlo, mhi = (test_mag_lo, test_mag_hi) if is_test else (20.0, 24.0)
        cat = {"object_id": np.arange(n).astype(float),
               "redshift": rng.uniform(0.1, 1.5, n),
               "mag_i_lsst": rng.uniform(mlo, mhi, n)}
        for c in _BANDS:
            if c != "mag_i_lsst":
                cat[c] = cat["mag_i_lsst"] + rng.normal(0.0, 0.3, n)
        if is_test and test_color_shift:
            for j, c in enumerate(_BANDS):
                if j % 2 == 0:                  # every other band -> colors shift
                    cat[c] = cat[c] + test_color_shift
        if is_test and not test_has_redshift:
            del cat["redshift"]
        return cat

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


# ---------------------------------------------------------------------------
# TS2 tests: TS2_CONFIG + magbinned recal path (light, RAIL-free)
# ---------------------------------------------------------------------------

def test_ts2_config_values():
    """TS2_CONFIG = the combination settled by the depth-matched rehearsal
    (job 33575936): deficitbinned_pit, the pre-registered winner."""
    c = submission.TS2_CONFIG
    assert c.members == ["pzflow", "gpz", "flexzboost"]
    assert c.band_set == "lsst_roman"
    assert c.weights == "optimal"
    assert c.recal == "deficitbinned_pit"


# The pre-swap TS2 combination — kept as explicit regression coverage for the
# magbinned submission path (class remains in the roster).
_MAGBINNED_CFG = submission.Config(members=["pzflow", "gpz", "flexzboost"],
                                   band_set="lsst_roman", weights="optimal",
                                   recal="magbinned_pit")


def test_ts2_train_infer_magbinned_no_test_redshift(monkeypatch):
    """End-to-end TS2 path: magbinned recal fit (needs feat ancil on the calib
    ensemble) + apply on a blind test file with NO redshift column. n_train=5000
    -> calib 1000 -> 250/bin, above min_per_bin, so real per-bin remaps are
    exercised (not just the pooled fallback). Would crash with
    TypeError/KeyError before the feat-ancil plumbing."""
    n_test = 60
    _install_stub_estimators(submission, monkeypatch, n_train=5000, n_test=n_test,
                             test_has_redshift=False)
    bundle = submission.train_submission_model("train.hdf5", _MAGBINNED_CFG)
    assert bundle["config"].recal == _MAGBINNED_CFG.recal
    assert bundle["recal"]._edges is not None          # per-bin remaps really fit
    out = submission.infer(bundle, "test_blind.hdf5")
    assert out.npdf == n_test
    grid = submission.Z_GRID
    pdfs = np.asarray(out.pdf(grid))
    assert np.isfinite(pdfs).all()
    assert np.allclose(np.trapezoid(pdfs, grid, axis=1), 1.0, atol=1e-2)
    assert np.array_equal(np.asarray(out.ancil["object_id"]), np.arange(n_test))
    assert "zmode" in out.ancil


def test_ts2_infer_beyond_calib_support_faintest_bin(monkeypatch):
    """The 63%-of-test deployment regime: test objects FAINTER than every calib
    object (i in [25,26], calib maxes ~24) must digitize into the magbinned recal's
    faintest bin and yield finite, normalized PDFs — never NaN or an out-of-range
    error (docstring's 'by design'). Verified with the review's F2 case."""
    n_test = 50
    _install_stub_estimators(submission, monkeypatch, n_train=5000, n_test=n_test,
                             test_has_redshift=False, test_mag_lo=25.0, test_mag_hi=26.0)
    bundle = submission.train_submission_model("train.hdf5", _MAGBINNED_CFG)
    top_edge = float(bundle["recal"]._edges[-1])
    assert top_edge < 25.0                     # all test objects are beyond the top edge
    out = submission.infer(bundle, "test_blind.hdf5")
    pdfs = np.asarray(out.pdf(submission.Z_GRID))
    assert out.npdf == n_test
    assert np.isfinite(pdfs).all()
    assert np.allclose(np.trapezoid(pdfs, submission.Z_GRID, axis=1), 1.0, atol=1e-2)


def test_ts2_estimation_only_pickle_roundtrip(tmp_path, monkeypatch):
    """run_taskset_2_estimation_only: pickled bundle -> qp file, blind schema."""
    import pickle
    _install_stub_estimators(submission, monkeypatch, n_train=5000, n_test=40,
                             test_has_redshift=False)
    bundle = submission.train_submission_model("train.hdf5", _MAGBINNED_CFG)
    mf = tmp_path / "bundle.pkl"
    with open(mf, "wb") as fh:
        pickle.dump(bundle, fh)
    of = tmp_path / "pz_ts2.hdf5"
    submission.run_taskset_2_estimation_only(str(mf), "test_blind.hdf5", str(of))
    back = qp.read(str(of))
    assert back.npdf == 40


def test_ts2_run_training_and_estimation_writes_readable_qp(tmp_path, monkeypatch):
    _install_stub_estimators(submission, monkeypatch, n_train=5000, n_test=40,
                             test_has_redshift=False)
    of = tmp_path / "pz_ts2.hdf5"
    submission.run_taskset_2_training_and_estimation("train.hdf5", "test_blind.hdf5",
                                                     str(of), config=_MAGBINNED_CFG)
    back = qp.read(str(of))
    assert back.npdf == 40
    assert np.isfinite(np.asarray(back.pdf(submission.Z_GRID))).all()


def test_ts1_global_pit_ignores_feat_ancil(monkeypatch):
    """Regression: the new feat ancil is inert for the TS1 global_pit path — the
    trained bundle's recal remap and inferred PDFs are unchanged by its presence."""
    _install_stub_estimators(submission, monkeypatch, n_train=400, n_test=30)
    bundle = submission.train_submission_model("train.hdf5", submission.DEFAULT_CONFIG)
    out = submission.infer(bundle, "test.hdf5")
    monkeypatch.setattr(submission, "_feat_ancil", lambda cat, band_set: {})
    bundle2 = submission.train_submission_model("train.hdf5", submission.DEFAULT_CONFIG)
    out2 = submission.infer(bundle2, "test.hdf5")
    np.testing.assert_allclose(np.asarray(out.pdf(submission.Z_GRID)),
                               np.asarray(out2.pdf(submission.Z_GRID)), atol=1e-12)


# ---------------------------------------------------------------------------
# TS2 deficitbinned tests (recal swap, rehearsal job 33575936)
# ---------------------------------------------------------------------------

def test_ts2_train_infer_deficitbinned_no_test_redshift(monkeypatch):
    """End-to-end TS2 path with the deficit-conditioned recal: the bundle must
    carry a frozen deficit_ref (train-slice band photometry) and infer must
    compute feat_deficit for the blind test file (NO redshift column) from it,
    yielding finite normalized PDFs. Would crash with KeyError('feat_deficit')
    before the plumbing."""
    n_test = 60
    _install_stub_estimators(submission, monkeypatch, n_train=5000, n_test=n_test,
                             test_has_redshift=False)
    bundle = submission.train_submission_model("train.hdf5", submission.TS2_CONFIG)
    assert bundle["config"].recal == "deficitbinned_pit"
    ref = bundle["deficit_ref"]
    assert ref is not None
    from conclave import bands
    assert set(ref) == set(bands.band_columns("lsst_roman"))
    assert len(ref["mag_i_lsst"]) == 4000          # train slice = 80% of 5000
    assert bundle["recal"]._edges is not None      # deficit bins really fit
    out = submission.infer(bundle, "test_blind.hdf5")
    assert out.npdf == n_test
    grid = submission.Z_GRID
    pdfs = np.asarray(out.pdf(grid))
    assert np.isfinite(pdfs).all()
    assert np.allclose(np.trapezoid(pdfs, grid, axis=1), 1.0, atol=1e-2)
    assert np.array_equal(np.asarray(out.ancil["object_id"]), np.arange(n_test))
    assert "zmode" in out.ancil


def test_ts2_infer_out_of_support_positive_deficit(monkeypatch):
    """The deployment regime the rehearsal measured: test objects whose COLORS
    sit outside the labeled support must get a strictly positive label-free
    deficit from the frozen reference (routing them to the positive-tail bins)
    and still yield finite, normalized PDFs."""
    n_test = 50
    _install_stub_estimators(submission, monkeypatch, n_train=5000, n_test=n_test,
                             test_has_redshift=False, test_color_shift=3.0)
    bundle = submission.train_submission_model("train.hdf5", submission.TS2_CONFIG)
    test_cat = submission._load_catalog("test_blind.hdf5")
    d = submission._deficit_from_ref(bundle["deficit_ref"], test_cat, "lsst_roman")
    assert d.shape == (n_test,)
    assert (d > 0).all()                           # far out of color support
    out = submission.infer(bundle, "test_blind.hdf5")
    pdfs = np.asarray(out.pdf(submission.Z_GRID))
    assert out.npdf == n_test
    assert np.isfinite(pdfs).all()
    assert np.allclose(np.trapezoid(pdfs, submission.Z_GRID, axis=1), 1.0, atol=1e-2)


def test_deficit_from_ref_matches_direct_label_free_deficit(monkeypatch):
    """Consistency guarantee: the merged-catalog helper must reproduce
    challenge.label_free_deficit called directly on a full catalog with the
    same labeled/target index sets (identical standardization, imputation,
    d_ref)."""
    from conclave import challenge
    rng = np.random.default_rng(7)
    n = 300
    cols = ([f"mag_{b}_lsst" for b in "ugrizy"]
            + ["mag_Y_roman", "mag_J_roman", "mag_H_roman"])
    base = rng.uniform(20, 24, n)
    cat = {c: base + rng.normal(0, 0.3, n) for c in cols}
    cat["mag_i_lsst"] = base
    labeled = np.arange(0, 200)
    target = np.arange(200, 300)
    direct = challenge.label_free_deficit(cat, labeled, target,
                                          band_set="lsst_roman")
    ref = {c: np.asarray(cat[c], float)[labeled] for c in cols}
    tgt = {c: np.asarray(cat[c], float)[target] for c in cols}
    via_ref = submission._deficit_from_ref(ref, tgt, "lsst_roman")
    np.testing.assert_allclose(via_ref, direct, atol=1e-12)


def test_ts2_infer_missing_deficit_ref_fails_clearly(monkeypatch):
    """An OLD bundle (pre-swap, no deficit_ref) used with a deficit recal must
    fail with a message naming deficit_ref — not a deep KeyError inside the
    recalibrator."""
    _install_stub_estimators(submission, monkeypatch, n_train=5000, n_test=20,
                             test_has_redshift=False)
    bundle = submission.train_submission_model("train.hdf5", submission.TS2_CONFIG)
    bundle.pop("deficit_ref")
    with pytest.raises(AssertionError, match="deficit_ref"):
        submission.infer(bundle, "test_blind.hdf5")


def test_ts2_deficitbinned_pickle_roundtrip(tmp_path, monkeypatch):
    """run_taskset_2_estimation_only with the NEW config: pickled bundle
    (including deficit_ref arrays) -> qp file, blind schema."""
    import pickle
    from conclave import bands
    _install_stub_estimators(submission, monkeypatch, n_train=5000, n_test=40,
                             test_has_redshift=False)
    bundle = submission.train_submission_model("train.hdf5", submission.TS2_CONFIG)
    mf = tmp_path / "bundle.pkl"
    with open(mf, "wb") as fh:
        pickle.dump(bundle, fh)

    # Verify deficit-specific plumbing survived the pickle roundtrip
    with open(mf, "rb") as fh:
        loaded = pickle.load(fh)
    assert loaded["config"].recal == "deficitbinned_pit"
    assert type(loaded["recal"]).__name__ == "DeficitBinnedPITRecalibrator"
    assert loaded["deficit_ref"] is not None
    assert set(loaded["deficit_ref"]) == set(bands.band_columns("lsst_roman"))
    for col in bands.band_columns("lsst_roman"):
        np.testing.assert_allclose(loaded["deficit_ref"][col], bundle["deficit_ref"][col])

    of = tmp_path / "pz_ts2.hdf5"
    submission.run_taskset_2_estimation_only(str(mf), "test_blind.hdf5", str(of))
    back = qp.read(str(of))
    assert back.npdf == 40
    assert np.isfinite(np.asarray(back.pdf(submission.Z_GRID))).all()


def test_ts1_bundle_has_no_deficit_ref_payload(monkeypatch):
    """Non-deficit recals must not pay the deficit_ref memory cost: the key is
    present (uniform schema) but None for TS1's global_pit."""
    _install_stub_estimators(submission, monkeypatch, n_train=400, n_test=30)
    bundle = submission.train_submission_model("train.hdf5", submission.DEFAULT_CONFIG)
    assert bundle["deficit_ref"] is None
