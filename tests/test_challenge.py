import numpy as np
import h5py
from conclave import challenge


def _cat(n=2000, seed=1, n_nan=60):
    rng = np.random.default_rng(seed)
    mi = rng.uniform(18.0, 24.5, n)
    mi[rng.permutation(n)[:n_nan]] = np.nan
    cat = {"redshift": rng.uniform(0.0, 3.0, n), "object_id": np.arange(n),
           "mag_i_lsst": mi}
    for c in ["mag_g_lsst", "mag_r_lsst", "mag_z_lsst", "mag_y_lsst"]:
        cat[c] = rng.uniform(18, 25, n)
    for s in challenge.SURVEY_FLAGS:
        cat[s] = rng.random(n) < 0.1
    return cat


def test_splits_pairwise_disjoint_and_partition():
    cat = _cat()
    sp = challenge.ts2_submission_splits(cat, seed=0, budgets=(50, 150))
    parts = [sp["base"], sp["deep_pool"], sp["calib"], sp["eval"]]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            assert np.intersect1d(parts[i], parts[j]).size == 0
    # base/deep_pool/calib/eval partition the finite-mag objects exactly
    mi = np.asarray(cat["mag_i_lsst"])
    all_idx = np.sort(np.concatenate(parts))
    assert np.array_equal(all_idx, np.where(np.isfinite(mi))[0])


def test_nan_excluded_everywhere():
    cat = _cat()
    sp = challenge.ts2_submission_splits(cat, seed=0)
    mi = np.asarray(cat["mag_i_lsst"])
    for key in ["base", "deep_pool", "calib", "eval"]:
        assert np.isfinite(mi[sp[key]]).all()
    for idx in sp["budget"].values():
        assert np.isfinite(mi[idx]).all()


def test_budgets_nested_exact_sizes_and_prefix():
    cat = _cat()
    sp = challenge.ts2_submission_splits(cat, seed=0, budgets=(50, 100, 200))
    budget = sp["budget"]
    # pool (~265 here) comfortably exceeds 200, so these are true interior budgets
    assert len(sp["deep_pool"]) > 200
    assert len(budget[50]) == 50 and len(budget[100]) == 100 and len(budget[200]) == 200
    assert set(budget[50].tolist()) <= set(budget[100].tolist())
    assert set(budget[100].tolist()) <= set(budget[200].tolist())
    assert set(budget[200].tolist()) <= set(budget[None].tolist())
    # budget[b] = first b of the (shuffled) deep_pool; None = full pool
    assert np.array_equal(budget[100], np.sort(sp["deep_pool"][:100]))
    assert np.array_equal(budget[None], np.sort(sp["deep_pool"]))


def test_eval_calib_stable_across_budgets_and_bright_cut():
    cat = _cat()
    sp_a = challenge.ts2_submission_splits(cat, seed=0, budgets=(50, 150), bright_cut=22.5)
    sp_b = challenge.ts2_submission_splits(cat, seed=0, budgets=(500, 2000, 5000), bright_cut=22.5)
    sp_c = challenge.ts2_submission_splits(cat, seed=0, budgets=(50,), bright_cut=23.5)
    for other in (sp_b, sp_c):
        assert np.array_equal(sp_a["eval"], other["eval"])
        assert np.array_equal(sp_a["calib"], other["calib"])


def test_base_bright_or_desi_and_pool_faint():
    cat = _cat()
    cut = 22.5
    sp = challenge.ts2_submission_splits(cat, seed=0, bright_cut=cut)
    mi = np.asarray(cat["mag_i_lsst"])
    desi = np.zeros(len(mi), bool)
    for s in challenge.DESI_FLAGS:
        desi |= np.asarray(cat[s]).astype(bool)
    assert np.all((mi[sp["base"]] < cut) | desi[sp["base"]])
    assert np.all(mi[sp["deep_pool"]] >= cut)
    assert not desi[sp["deep_pool"]].any()
    # faint DESI-flagged objects belong to base, not the pool
    faint_desi = np.where(np.isfinite(mi) & (mi >= cut) & desi)[0]
    in_holdout = np.union1d(sp["calib"], sp["eval"])
    assert np.isin(np.setdiff1d(faint_desi, in_holdout), sp["base"]).all()


def test_eval_stratified_over_full_depth():
    cat = _cat()
    sp = challenge.ts2_submission_splits(cat, seed=0)
    mi = np.asarray(cat["mag_i_lsst"])
    for lo, hi in challenge.STRAT_BINS[:-1]:      # last bin (24, inf] is empty at n=2000/18-24.5
        assert np.any((mi[sp["eval"]] > lo) & (mi[sp["eval"]] <= hi))
        assert np.any((mi[sp["calib"]] > lo) & (mi[sp["calib"]] <= hi))


def _write_ts2_file(path, cat):
    with h5py.File(path, "w") as f:
        for k, v in cat.items():
            f.create_dataset(k, data=np.asarray(v))


def test_loader_roundtrip_env_root(tmp_path, monkeypatch):
    cat = _cat(n=100)
    _write_ts2_file(tmp_path / "pz_challenge_taskset_2_cardinal_training_10yr.hdf5", cat)
    monkeypatch.setenv("PZDC_PUBLIC", str(tmp_path))
    out = challenge.load_ts2("cardinal")
    assert set(out) == set(cat)
    for s in challenge.SURVEY_FLAGS:
        assert np.array_equal(np.asarray(out[s]).astype(bool), cat[s])
    assert np.allclose(out["mag_i_lsst"], cat["mag_i_lsst"], equal_nan=True)
    assert np.array_equal(out["object_id"], cat["object_id"])


def test_loader_root_arg_overrides_env(tmp_path, monkeypatch):
    cat = _cat(n=50, seed=2)
    _write_ts2_file(tmp_path / "pz_challenge_taskset_2_flagship_test_1yr.hdf5", cat)
    monkeypatch.setenv("PZDC_PUBLIC", "/nonexistent/elsewhere")
    out = challenge.load_ts2("flagship", scenario="1yr", kind="test", root=str(tmp_path))
    assert set(out) == set(cat)
    assert np.allclose(out["redshift"], cat["redshift"])


# ---------------------------------------------------------------------------
# label_free_deficit (T4 item A)
# ---------------------------------------------------------------------------

_LSST_COLS = ["mag_u_lsst", "mag_g_lsst", "mag_r_lsst",
              "mag_i_lsst", "mag_z_lsst", "mag_y_lsst"]


def _color_cat(n=400, seed=3, shift=None, n_nan=0):
    """Photometry-only catalog (NO redshift key): 6 LSST bands whose adjacent colors are
    ~N(0, 1); `shift` displaces the last `len(shift)` objects in color space."""
    rng = np.random.default_rng(seed)
    base = rng.normal(22.0, 1.0, n)
    cat = {}
    off = np.zeros(n)
    for j, c in enumerate(_LSST_COLS):
        cat[c] = base + off
        off = off + rng.normal(0.0, 0.4, n)               # random adjacent colors
    if shift is not None:
        for j, c in enumerate(_LSST_COLS):
            cat[c][-len(shift):] += np.asarray(shift) * j  # ramp -> big color offsets
    if n_nan:
        cat["mag_u_lsst"][rng.permutation(n)[:n_nan]] = np.nan
    return cat


def test_label_free_deficit_never_touches_redshift():
    # The deficit MUST be label-free: catalog has no redshift key at all.
    cat = _color_cat(n=300)
    assert "redshift" not in cat
    lab = np.arange(200)
    tgt = np.arange(200, 300)
    d = challenge.label_free_deficit(cat, lab, tgt, band_set="lsst", k=5)
    assert d.shape == tgt.shape
    assert np.isfinite(d).all() and (d >= 0).all()


def test_label_free_deficit_in_support_zero_and_outliers_positive():
    n = 400
    n_out = 30
    cat = _color_cat(n=n, shift=np.full(n_out, 5.0))       # last 30 objects far off-locus
    lab = np.arange(250)
    in_tgt = np.arange(250, n - n_out)                     # same locus as labels
    out_tgt = np.arange(n - n_out, n)                      # shifted off-locus
    d_in = challenge.label_free_deficit(cat, lab, in_tgt, band_set="lsst")
    d_out = challenge.label_free_deficit(cat, lab, out_tgt, band_set="lsst")
    assert np.median(d_in) < 0.5                           # in-support: near identity
    assert (d_in == 0).mean() > 0.2                        # many exactly at the 0 floor
    assert (d_out > 1.0).all()                             # off-locus: strongly flagged
    assert np.median(d_out) > 5 * max(np.median(d_in), 0.1)


def test_label_free_deficit_labeled_targets_leave_self_out():
    # A labeled object queried as a target must NOT be flattered by matching itself:
    # its deficit uses the same leave-self-out distance as the d_ref reference, so
    # roughly half the labeled set sits at the 0 floor — not all of it.
    cat = _color_cat(n=300)
    lab = np.arange(300)
    d = challenge.label_free_deficit(cat, lab, lab, band_set="lsst")
    assert d.shape == (300,)
    frac_zero = (d == 0).mean()
    assert 0.3 < frac_zero < 0.9                           # ~half at floor, not all
    assert (d > 0).any()                                   # self-match would force all-zero


def test_label_free_deficit_nan_imputed_with_labeled_medians():
    cat = _color_cat(n=400, n_nan=80)                      # NaN u-band -> NaN u-g color
    lab = np.arange(300)
    tgt = np.arange(300, 400)
    d = challenge.label_free_deficit(cat, lab, tgt, band_set="lsst")
    assert np.isfinite(d).all()


def test_label_free_deficit_batch_independent():
    # Deployability: a target's deficit depends only on its own photometry + the frozen
    # labeled set (labeled-set standardization + imputation), not on its batch.
    cat = _color_cat(n=350, n_nan=40)
    lab = np.arange(250)
    tgt = np.arange(250, 350)
    d_all = challenge.label_free_deficit(cat, lab, tgt, band_set="lsst")
    d_one = challenge.label_free_deficit(cat, lab, tgt[:1], band_set="lsst")
    np.testing.assert_allclose(d_all[:1], d_one)
