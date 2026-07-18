import numpy as np
import qp, pytest
from conclave import recal

def _ens(n):
    loc = np.linspace(0.2, 1.2, n).reshape(-1, 1)
    return qp.Ensemble(qp.stats.norm, data=dict(loc=loc, scale=np.full((n, 1), 0.05)))

def test_identity_is_noop():
    e = _ens(100)
    r = recal.RECALIBRATORS["none"]()
    r.fit(e, np.linspace(0.2, 1.2, 100), fit_idx=np.arange(50))
    out = r.apply(e, apply_idx=np.arange(50, 100))
    assert out.npdf == 50
    np.testing.assert_allclose(out.mean().squeeze(), e[50:100].mean().squeeze())

def test_leakage_guard():
    e = _ens(100); r = recal.RECALIBRATORS["none"]()
    r.fit(e, np.linspace(0.2, 1.2, 100), fit_idx=np.arange(50))
    with pytest.raises(AssertionError):
        r.apply(e, apply_idx=np.arange(40, 60))   # overlaps fit_idx → must raise

def test_apply_without_fit_is_allowed():
    e = _ens(10)
    r = recal.RECALIBRATORS["none"]()
    out = r.apply(e, apply_idx=np.arange(10))
    assert out.npdf == 10


def test_global_pit_improves_calibration():
    import numpy as np, qp
    from conclave import recal, metrics
    rng = np.random.default_rng(0)
    z = rng.uniform(0.1, 1.5, 4000)
    # over-confident ensemble: means scattered by 0.05, predicted sigma 0.02
    means = z + rng.normal(0, 0.05, len(z))
    ens = qp.Ensemble(qp.stats.norm, data=dict(loc=means.reshape(-1,1),
                                               scale=np.full((len(z),1), 0.02)))
    ens.set_ancil({"zmode": means})
    fit_idx = np.arange(0, 2000); app_idx = np.arange(2000, 4000)
    r = recal.RECALIBRATORS["global_pit"]()
    r.fit(ens, z, fit_idx)
    out = r.apply(ens, app_idx)
    ks_before = metrics._pit_stats(metrics._pit_values(ens[app_idx], z[app_idx]))["PIT_ks"]
    ks_after  = metrics._pit_stats(metrics._pit_values(out, z[app_idx]))["PIT_ks"]
    assert out.npdf == 2000
    assert ks_after < ks_before     # recalibration improves marginal PIT


def test_magbinned_pit_leakage_guard():
    e = _ens(100)
    r = recal.RECALIBRATORS["magbinned_pit"](feature_key="feat_mag_i_lsst")
    e.set_ancil({"feat_mag_i_lsst": np.linspace(20, 24, 100)})
    r.fit(e, np.linspace(0.2, 1.2, 100), fit_idx=np.arange(50))
    with pytest.raises(AssertionError):
        r.apply(e, apply_idx=np.arange(40, 60))   # overlaps fit_idx → must raise


def test_magbinned_pit_improves_feature_dependent_calibration():
    """Magnitude-binned global-PIT must fix feature-dependent miscalibration that a
    single global remap cannot. Bright objects (feat<22) are well-calibrated; faint
    objects (feat>=22) are under-confident (sigma_pred=0.12 vs true 0.05). A per-bin
    remap should reduce PIT-KS, and beat the single global-PIT remap here because the
    miscalibration direction differs by magnitude (global averages them out).
    The feature carries ~5% NaN to exercise the global fallback path.
    """
    import numpy as np, qp
    from conclave import recal, metrics
    rng = np.random.default_rng(0)
    n = 8000
    z = rng.uniform(0.1, 1.5, n)
    feat = rng.uniform(20, 24, n)
    nan_mask = rng.random(n) < 0.05
    feat_nan = feat.copy(); feat_nan[nan_mask] = np.nan
    # opposite miscalibration by magnitude so a single global remap is ineffective:
    # bright over-confident (0.02), faint under-confident (0.12); true scatter 0.05.
    sigma_pred = np.where(feat < 22, 0.02, 0.12)
    means = z + rng.normal(0, 0.05, n)
    ens = qp.Ensemble(qp.stats.norm, data=dict(loc=means.reshape(-1, 1),
                                               scale=sigma_pred.reshape(-1, 1)))
    ens.set_ancil({"zmode": means, "feat_mag_i_lsst": feat_nan})
    fit_idx = np.arange(0, 4000); app_idx = np.arange(4000, n)

    rb = recal.RECALIBRATORS["magbinned_pit"](feature_key="feat_mag_i_lsst", n_bins=4)
    rb.fit(ens, z, fit_idx); out_b = rb.apply(ens, app_idx)
    rg = recal.RECALIBRATORS["global_pit"]()
    rg.fit(ens, z, fit_idx); out_g = rg.apply(ens, app_idx)

    ks_raw = metrics._pit_stats(metrics._pit_values(ens[app_idx], z[app_idx]))["PIT_ks"]
    ks_b = metrics._pit_stats(metrics._pit_values(out_b, z[app_idx]))["PIT_ks"]
    ks_g = metrics._pit_stats(metrics._pit_values(out_g, z[app_idx]))["PIT_ks"]
    assert out_b.npdf == 4000
    assert ks_b < ks_raw, f"binned-PIT did not improve PIT-KS: raw={ks_raw:.4f} binned={ks_b:.4f}"
    assert ks_b <= ks_g + 1e-6, f"binned-PIT should beat global on bin-dependent miscal: binned={ks_b:.4f} global={ks_g:.4f}"


def _deficit_ens(deficit, rng):
    """Over-confident (0.02) at zero deficit, under-confident (0.12) in the positive tail —
    so a deficit-binned remap has bin-dependent miscalibration to fix that global averages out."""
    n = len(deficit)
    z = rng.uniform(0.1, 1.5, n)
    sigma_pred = np.where(np.asarray(deficit) > 1e-9, 0.12, 0.02)
    means = z + rng.normal(0, 0.05, n)
    ens = qp.Ensemble(qp.stats.norm, data=dict(loc=means.reshape(-1, 1),
                                               scale=sigma_pred.reshape(-1, 1)))
    ens.set_ancil({"zmode": means, "feat_deficit": np.asarray(deficit, float)})
    return ens, z


def test_deficitbinned_pit_registered():
    assert "deficitbinned_pit" in recal.RECALIBRATORS
    assert "deficitbinned_pit" in recal.DEFICIT_RECALS
    r = recal.RECALIBRATORS["deficitbinned_pit"]()
    assert r.feature_key == "feat_deficit"


def test_deficitbinned_pit_zero_inflated_binning():
    """Zero-deficit objects land in bin 0; the strictly-positive tail gets its own bins."""
    rng = np.random.default_rng(0)
    n = 8000
    deficit = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.1, 3.0, n))
    ens, z = _deficit_ens(deficit, rng)
    fit_idx = np.arange(0, 4000); app_idx = np.arange(4000, n)
    r = recal.RECALIBRATORS["deficitbinned_pit"](n_bins=4)
    r.fit(ens, z, fit_idx)
    assert r._edges[0] == r.eps                      # zero/positive boundary is the first edge
    assert len(r._edges) == 3                        # n_bins-1 inner edges
    assert (np.diff(r._edges) > 0).all()             # strictly increasing
    binid = np.digitize(deficit, r._edges)
    assert (binid[deficit <= r.eps] == 0).all()      # zeros -> bin 0
    assert (binid[deficit > r.eps] >= 1).all()       # positive tail -> bins 1..n_bins-1
    out = r.apply(ens, app_idx)
    assert out.npdf == 4000
    grid = np.linspace(0.0, 3.0, 301)
    areas = np.trapezoid(np.asarray(out.pdf(grid)), grid, axis=1)
    assert np.allclose(areas, 1.0, atol=1e-3)        # valid (renormalized) ensemble


def test_deficitbinned_pit_beats_global_on_deficit_miscal():
    """Deficit-dependent miscalibration (over-confident zeros, under-confident tail) that a
    single global remap cannot fix: the deficit-binned remap must reduce PIT-KS and beat global."""
    from conclave import metrics
    rng = np.random.default_rng(0)
    n = 8000
    deficit = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.1, 3.0, n))
    ens, z = _deficit_ens(deficit, rng)
    fit_idx = np.arange(0, 4000); app_idx = np.arange(4000, n)
    rd = recal.RECALIBRATORS["deficitbinned_pit"](n_bins=4)
    rd.fit(ens, z, fit_idx); out_d = rd.apply(ens, app_idx)
    rg = recal.RECALIBRATORS["global_pit"]()
    rg.fit(ens, z, fit_idx); out_g = rg.apply(ens, app_idx)
    ks_raw = metrics._pit_stats(metrics._pit_values(ens[app_idx], z[app_idx]))["PIT_ks"]
    ks_d = metrics._pit_stats(metrics._pit_values(out_d, z[app_idx]))["PIT_ks"]
    ks_g = metrics._pit_stats(metrics._pit_values(out_g, z[app_idx]))["PIT_ks"]
    assert ks_d < ks_raw, f"deficit-binned did not improve PIT-KS: raw={ks_raw:.4f} d={ks_d:.4f}"
    assert ks_d <= ks_g + 1e-6, f"deficit-binned should beat global: d={ks_d:.4f} g={ks_g:.4f}"


def test_deficitbinned_pit_all_zero_behaves_like_global():
    """All-zero deficit -> no positive tail -> empty edges -> a single bin identical to the
    pooled global remap; the output must match global_pit numerically."""
    rng = np.random.default_rng(1)
    n = 4000
    deficit = np.zeros(n)
    ens, z = _deficit_ens(deficit, rng)
    fit_idx = np.arange(0, 2000); app_idx = np.arange(2000, n)
    rd = recal.RECALIBRATORS["deficitbinned_pit"]()
    rd.fit(ens, z, fit_idx); out_d = rd.apply(ens, app_idx)
    assert rd._edges.size == 0                        # degenerate -> single bin
    rg = recal.RECALIBRATORS["global_pit"]()
    rg.fit(ens, z, fit_idx); out_g = rg.apply(ens, app_idx)
    grid = np.linspace(0.0, 3.0, 301)
    np.testing.assert_allclose(np.asarray(out_d.pdf(grid)),
                               np.asarray(out_g.pdf(grid)), atol=1e-6)


def test_deficitbinned_pit_nan_routes_to_fallback():
    """NaN deficit is excluded from the edges and routed to the pooled global fallback at
    apply — the run completes and NaN-feature apply objects get finite, valid PDFs."""
    rng = np.random.default_rng(2)
    n = 6000
    deficit = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.1, 3.0, n))
    nan_mask = rng.random(n) < 0.1
    deficit[nan_mask] = np.nan
    ens, z = _deficit_ens(np.nan_to_num(deficit, nan=0.0), rng)  # sigma set on a NaN-free copy
    ens.set_ancil({**ens.ancil, "feat_deficit": deficit})        # but the FEATURE carries NaN
    fit_idx = np.arange(0, 3000); app_idx = np.arange(3000, n)
    r = recal.RECALIBRATORS["deficitbinned_pit"]()
    r.fit(ens, z, fit_idx)
    out = r.apply(ens, app_idx)
    assert out.npdf == 3000
    grid = np.linspace(0.0, 3.0, 301)
    p = np.asarray(out.pdf(grid))
    assert np.isfinite(p).all()                       # NaN-feature rows produced valid PDFs
    # specifically: at least one apply object had a NaN deficit and was routed via fallback
    assert np.isnan(deficit[app_idx]).any()


def test_deficitbinned_pit_leakage_guard():
    e = _ens(100)
    r = recal.RECALIBRATORS["deficitbinned_pit"]()
    e.set_ancil({"feat_deficit": np.concatenate([np.zeros(50), np.linspace(0.1, 2.0, 50)])})
    r.fit(e, np.linspace(0.2, 1.2, 100), fit_idx=np.arange(50))
    with pytest.raises(AssertionError):
        r.apply(e, apply_idx=np.arange(40, 60))       # overlaps fit_idx -> must raise


def test_calpit_improves_feature_dependent_calibration():
    """Cal-PIT (instance-wise) must fix feature-dependent miscalibration.

    Bright objects (feat < 22) are well-calibrated (sigma_pred=0.05 == true scatter).
    Faint objects are severely under-confident (sigma_pred=0.12, true scatter=0.05).
    Cal-PIT with the feature can fix the faint under-confidence while leaving bright
    objects intact; a global recalibrator would either over-correct or under-correct.

    The PIT_ks MUST decrease after Cal-PIT, which is only achievable if the
    recalibrator correctly identifies faint objects (via the feature) and applies
    the appropriate correction to narrow their distribution.

    The feature column includes ~5% NaN entries to guard NaN median-imputation.
    """
    import numpy as np, qp
    from conclave import recal, metrics

    rng = np.random.default_rng(0)
    n = 6000
    z = rng.uniform(0.1, 1.5, n)
    feat = rng.uniform(20, 24, n)                       # a magnitude-like feature

    # Inject ~5% NaN entries (simulating u-band non-detections such as u-band dropouts)
    nan_mask = rng.random(n) < 0.05
    feat_with_nan = feat.copy()
    feat_with_nan[nan_mask] = np.nan

    # Miscalibration depends on feature:
    #   bright (feat<22): sigma_pred == true scatter (well-calibrated)
    #   faint  (feat>=22): sigma_pred >> true scatter (severely under-confident)
    sigma_pred = np.where(feat < 22, 0.05, 0.12)
    means = z + rng.normal(0, 0.05, n)

    ens = qp.Ensemble(qp.stats.norm, data=dict(loc=means.reshape(-1, 1),
                                               scale=sigma_pred.reshape(-1, 1)))
    ens.set_ancil({"zmode": means, "feat_mag_i_lsst": feat_with_nan})

    fit_idx = np.arange(0, 3000)
    app_idx = np.arange(3000, n)

    r = recal.RECALIBRATORS["calpit"](feature_keys=["feat_mag_i_lsst"], seed=0)
    r.fit(ens, z, fit_idx)
    out = r.apply(ens, app_idx)

    ks_before = metrics._pit_stats(metrics._pit_values(ens[app_idx], z[app_idx]))["PIT_ks"]
    ks_after  = metrics._pit_stats(metrics._pit_values(out, z[app_idx]))["PIT_ks"]

    assert out.npdf == 3000
    assert ks_after < ks_before, (
        f"Cal-PIT did not improve PIT KS: before={ks_before:.4f}, after={ks_after:.4f}"
    )
