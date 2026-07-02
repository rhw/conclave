# tests/test_ensemble.py
import numpy as np
import qp
from conclave import ensemble


def _ens(n, loc0=0.2, loc1=1.2, scale=0.05):
    loc = np.linspace(loc0, loc1, n).reshape(-1, 1)
    return qp.Ensemble(qp.stats.norm, data=dict(loc=loc, scale=np.full((n, 1), scale)))


def _trapz(pdf, z_grid, axis=-1):
    fn = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return fn(pdf, z_grid, axis=axis)


def _integral(pdf, z_grid):
    return _trapz(pdf, z_grid, axis=-1)


def test_to_common_grid_shape_and_normalization():
    members = [_ens(10), _ens(10, scale=0.1)]
    stacked = ensemble.to_common_grid(members)
    assert stacked.shape == (2, 10, 301)
    np.testing.assert_allclose(_integral(stacked, ensemble.Z_GRID), 1.0, atol=1e-3)


def test_equal_weight_is_uniform_mean():
    members = [_ens(10), _ens(10, scale=0.1)]
    stacked = ensemble.to_common_grid(members)
    ew = ensemble.equal_weight(stacked)
    np.testing.assert_allclose(ew, stacked.mean(axis=0))
    assert ew.shape == (10, 301)


def test_combine_yields_valid_qp_ensemble():
    members = [_ens(10), _ens(10, scale=0.1)]
    stacked = ensemble.to_common_grid(members)
    w = np.array([0.5, 0.5])
    out = ensemble.combine(stacked, w)
    assert isinstance(out, qp.Ensemble)
    assert out.npdf == 10
    np.testing.assert_allclose(
        _integral(np.asarray(out.pdf(ensemble.Z_GRID)), ensemble.Z_GRID), 1.0, atol=1e-3)


def test_combine_attaches_ancil():
    """combine(ancil=...) attaches the given ancil to the result so feature-aware
    recalibrators (e.g. magbinned_pit) can read feat_* off the combined ensemble."""
    members = [_ens(10), _ens(10, scale=0.1)]
    stacked = ensemble.to_common_grid(members)
    feat = np.linspace(20.0, 24.0, 10)
    out = ensemble.combine(stacked, np.array([0.5, 0.5]),
                           ancil={"feat_mag_i_lsst": feat})
    assert out.ancil is not None
    np.testing.assert_array_equal(out.ancil["feat_mag_i_lsst"], feat)
    # default (no ancil) still returns a bare ensemble
    out2 = ensemble.combine(stacked, np.array([0.5, 0.5]))
    assert out2.ancil is None


def test_combine_equal_weights_matches_equal_weight():
    members = [_ens(10), _ens(10, scale=0.1)]
    stacked = ensemble.to_common_grid(members)
    ew = ensemble.equal_weight(stacked)
    out = ensemble.combine(stacked, np.array([0.5, 0.5]))
    # equal_weight pdf and combine pdf agree after combine's renormalization
    out_pdf = np.asarray(out.pdf(ensemble.Z_GRID))
    ew_norm = ew / _trapz(ew, ensemble.Z_GRID, axis=-1)[..., np.newaxis]
    np.testing.assert_allclose(out_pdf, ew_norm, atol=1e-3)


def test_optimal_weights_on_simplex():
    members = [_ens(200), _ens(200, scale=0.1), _ens(200, scale=0.2)]
    stacked = ensemble.to_common_grid(members)
    z_true = np.linspace(0.2, 1.2, 200)
    w = ensemble.optimal_weights(stacked, z_true)
    assert w.shape == (3,)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= 0)


def test_optimal_weights_recovers_best_member_vs_noise():
    # member 0 is sharp & centered on truth; members 1,2 are broad near-uniform "noise"
    n = 300
    z_true = np.linspace(0.2, 1.2, n)
    good = qp.Ensemble(qp.stats.norm,
                       data=dict(loc=z_true.reshape(-1, 1), scale=np.full((n, 1), 0.03)))
    noise1 = qp.Ensemble(qp.stats.norm,
                         data=dict(loc=np.full((n, 1), 1.5), scale=np.full((n, 1), 0.9)))
    noise2 = qp.Ensemble(qp.stats.norm,
                         data=dict(loc=np.full((n, 1), 1.5), scale=np.full((n, 1), 1.1)))
    stacked = ensemble.to_common_grid([good, noise1, noise2])
    w = ensemble.optimal_weights(stacked, z_true)
    assert w[0] > 0.8                       # nearly all weight on the good member
    assert w[1] < 0.15 and w[2] < 0.15


def test_member_corr_matrix_shape_and_diag():
    members = [_ens(200), _ens(200, scale=0.1)]
    stacked = ensemble.to_common_grid(members)
    z_true = np.linspace(0.2, 1.2, 200)
    C = ensemble.member_corr_matrix(stacked, z_true)
    assert C.shape == (2, 2)
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-6)


def test_run_ensemble_cell_smoke(monkeypatch, tmp_path):
    import numpy as np
    import qp
    from conclave import ensemble_experiment as ee
    from conclave import experiment

    # tiny synthetic catalog written to an hdf5 the loader can read
    import tables_io
    n = 400
    rng = np.random.default_rng(0)
    z = rng.uniform(0.1, 1.5, n)
    cat = {"redshift": z, "object_id": np.arange(n)}
    for c in ["mag_u_lsst", "mag_g_lsst", "mag_r_lsst", "mag_i_lsst",
              "mag_z_lsst", "mag_y_lsst"]:
        cat[c] = z + rng.normal(0, 0.3, n) + 22.0
        cat[f"{c}_err"] = np.full(n, 0.03)
    path = tmp_path / "pz_challenge_taskset_1_cardinal_training_10yr.hdf5"
    tables_io.write(cat, str(path).replace(".hdf5", ""), "hdf5")

    # fake estimator: returns a gaussian centered near truth (train/estimate signature)
    def fake_train(train_dict, band_set, seed, name):
        return {"bias": 0.0}

    def make_fake_est(bias, scale):
        def est(model, data_dict, band_set, name):
            zt = np.asarray(data_dict["redshift"])
            loc = (zt + bias).reshape(-1, 1)
            e = qp.Ensemble(qp.stats.norm,
                            data=dict(loc=loc, scale=np.full((len(zt), 1), scale)))
            e.set_ancil({"zmode": (zt + bias),
                         "object_id": np.asarray(data_dict["object_id"]).astype(int)})
            return e
        return est

    monkeypatch.setitem(experiment.ESTIMATORS, "fakeA", (fake_train, make_fake_est(0.0, 0.05)))
    monkeypatch.setitem(experiment.ESTIMATORS, "fakeB", (fake_train, make_fake_est(0.02, 0.08)))

    row = ee.run_ensemble_cell(
        members=["fakeA", "fakeB"], band_set="lsst", weighting="optimal_weights",
        recal="none", sim="cardinal", scenario="10yr", seed=0, public_dir=str(tmp_path),
        frac_val=0.2, frac_calib=0.2)
    assert set(["CDELoss", "PIT_ks", "std", "n_test"]).issubset(row)
    assert row["members"] == "fakeA+fakeB"
    assert np.isfinite(row["CDELoss"])


def test_run_ensemble_group_smoke(monkeypatch, tmp_path):
    import numpy as np
    import qp
    from conclave import ensemble_experiment as ee
    from conclave import experiment

    # tiny synthetic catalog written to an hdf5 the loader can read
    import tables_io
    n = 400
    rng = np.random.default_rng(0)
    z = rng.uniform(0.1, 1.5, n)
    cat = {"redshift": z, "object_id": np.arange(n)}
    for c in ["mag_u_lsst", "mag_g_lsst", "mag_r_lsst", "mag_i_lsst",
              "mag_z_lsst", "mag_y_lsst"]:
        cat[c] = z + rng.normal(0, 0.3, n) + 22.0
        cat[f"{c}_err"] = np.full(n, 0.03)
    path = tmp_path / "pz_challenge_taskset_1_cardinal_training_10yr.hdf5"
    tables_io.write(cat, str(path).replace(".hdf5", ""), "hdf5")

    def fake_train(train_dict, band_set, seed, name):
        return {"bias": 0.0}

    def make_fake_est(bias, scale):
        def est(model, data_dict, band_set, name):
            zt = np.asarray(data_dict["redshift"])
            loc = (zt + bias).reshape(-1, 1)
            e = qp.Ensemble(qp.stats.norm,
                            data=dict(loc=loc, scale=np.full((len(zt), 1), scale)))
            e.set_ancil({"zmode": (zt + bias),
                         "object_id": np.asarray(data_dict["object_id"]).astype(int)})
            return e
        return est

    monkeypatch.setitem(experiment.ESTIMATORS, "fakeA", (fake_train, make_fake_est(0.0, 0.05)))
    monkeypatch.setitem(experiment.ESTIMATORS, "fakeB", (fake_train, make_fake_est(0.02, 0.08)))
    monkeypatch.setitem(experiment.ESTIMATORS, "fakeC", (fake_train, make_fake_est(0.04, 0.12)))

    rows = ee.run_ensemble_group(
        member_superset=["fakeA", "fakeB", "fakeC"],
        subsets=[["fakeA", "fakeB"], ["fakeA", "fakeB", "fakeC"]],
        band_set="lsst",
        sim="cardinal",
        scenario="10yr",
        seed=0,
        public_dir=str(tmp_path),
    )

    # 2 subsets × 2 weightings × 2 recals = 8 rows
    assert len(rows) == 8, f"expected 8 rows, got {len(rows)}"

    # All CDELoss values are finite
    for row in rows:
        assert np.isfinite(row["CDELoss"]), f"CDELoss not finite: {row}"

    # Exactly one row: fakeA+fakeB, equal_weight, none
    ab_ew_none = [r for r in rows
                  if r["members"] == "fakeA+fakeB"
                  and r["weighting"] == "equal_weight"
                  and r["recal"] == "none"]
    assert len(ab_ew_none) == 1

    # Exactly one row: fakeA+fakeB+fakeC, equal_weight, none
    abc_ew_none = [r for r in rows
                   if r["members"] == "fakeA+fakeB+fakeC"
                   and r["weighting"] == "equal_weight"
                   and r["recal"] == "none"]
    assert len(abc_ew_none) == 1

    # Weight vector lengths match subset size
    for row in rows:
        if row["members"] == "fakeA+fakeB":
            assert len(row["weights"]) == 2, f"expected 2 weights, got {row['weights']}"
        elif row["members"] == "fakeA+fakeB+fakeC":
            assert len(row["weights"]) == 3, f"expected 3 weights, got {row['weights']}"

    # Behavior equivalence: group CDELoss == cell CDELoss for fakeA+fakeB, equal_weight, none
    cell_row = ee.run_ensemble_cell(
        members=["fakeA", "fakeB"], band_set="lsst", weighting="equal_weight",
        recal="none", sim="cardinal", scenario="10yr", seed=0, public_dir=str(tmp_path),
        frac_val=0.2, frac_calib=0.2)
    group_cde = ab_ew_none[0]["CDELoss"]
    assert abs(cell_row["CDELoss"] - group_cde) < 1e-6, (
        f"cell CDELoss {cell_row['CDELoss']} != group CDELoss {group_cde}"
    )
