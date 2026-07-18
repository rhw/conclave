"""TS2 challenge-data loader + leakage-clean submission splits (spec req 2): load a TS2
training file into a catalog dict (survey flags preserved) and partition it into a bright
pseudo-wide base, a nested-budget deep pool, and held-out calib/eval slices stratified over
the full depth range. RAIL-free (h5py + numpy only)."""
import math
import os

import h5py
import numpy as np

PUBLIC = "/oak/stanford/orgs/kipac/users/risahw/pz_data_challenge/repo/public"
SURVEY_FLAGS = ["DEEP2_LSST", "DESI_BGS", "DESI_ELG_LOP", "DESI_LRG", "VVDSf02", "zCOSMOS"]
DESI_FLAGS = ["DESI_BGS", "DESI_ELG_LOP", "DESI_LRG"]

# Stratification bins for the held-out calib/eval slices: full depth range so eval covers
# bright AND faint. (lo, hi] convention as in realdata.DEFAULT_BINS / crossfield.DEPTH_BINS
# (neither of which spans the full range, hence a local constant).
STRAT_BINS = [(0.0, 21.0), (21.0, 22.0), (22.0, 23.0),
              (23.0, 23.5), (23.5, 24.0), (24.0, math.inf)]


def load_ts2(sim, scenario="10yr", kind="training", root=None):
    """Load a TS2 challenge HDF5 into a catalog dict of numpy arrays (ALL columns, including
    the six survey flags). root defaults to $PZDC_PUBLIC, falling back to the Sherlock path."""
    if root is None:
        root = os.environ.get("PZDC_PUBLIC", PUBLIC)
    path = os.path.join(root, f"pz_challenge_taskset_2_{sim}_{kind}_{scenario}.hdf5")
    with h5py.File(path, "r") as f:
        node = f
        keys = list(node.keys())
        if len(keys) == 1 and isinstance(node[keys[0]], h5py.Group):
            node = node[keys[0]]
        return {k: np.asarray(node[k][()]) for k in node
                if isinstance(node[k], h5py.Dataset)}


def label_free_deficit(cat, labeled_idx, target_idx, band_set="lsst_roman", k=10, seed=0):
    """Label-free per-object support deficit: color-space kNN distance to the labeled set.

    d_j = mean distance of target j to its k nearest LABELED neighbors (leave-self-out
    when j is itself labeled, so calib objects are not flattered by self-matches);
    d_ref = median of the same leave-self-out k-NN distance computed labeled->labeled;
    deficit_j = max(0, d_j / d_ref - 1) — in-support objects sit at 0, which is the
    identity under ``broaden.SupportBroadening``.

    LABEL-FREE by construction: consumes photometry columns (the same adjacent-band
    colors as ``support.color_matrix``) plus WHICH objects are labeled — never any
    redshift. Guarded by a unit test that deletes ``cat["redshift"]``.

    Standardization & NaN policy (deployability decision): colors are standardized with
    the LABELED set's mean/std, and NaN colors (non-detections) are imputed with the
    LABELED set's per-feature medians — for both labeled and target rows. A target's
    deficit therefore depends only on its own photometry and the frozen labeled set,
    never on the target batch it arrives with, so the measure is computable
    object-by-object when real (unlabeled) targets arrive later. This is why
    ``support.color_matrix``/``_standardize`` are not called verbatim: they impute and
    standardize with the statistics of whatever rows they are given, which would tie a
    target's deficit to its batch. The color construction mirrors ``color_matrix``.

    ``seed`` is reserved (the kNN measure is deterministic; kept for API stability with
    future subsampled variants). Returns a float array aligned with ``target_idx``.
    """
    from sklearn.neighbors import NearestNeighbors
    from conclave import bands

    labeled_idx = np.asarray(labeled_idx, dtype=int)
    target_idx = np.asarray(target_idx, dtype=int)
    assert labeled_idx.size >= 2, "label_free_deficit needs >=2 labeled objects"
    cols = bands.band_columns(band_set)

    def colors(idx):
        mags = [np.asarray(cat[c], dtype=float)[idx] for c in cols]
        return np.column_stack([mags[i] - mags[i + 1] for i in range(len(mags) - 1)])

    Xl = colors(labeled_idx)
    Xt = colors(target_idx)
    med = np.nanmedian(Xl, axis=0)                     # labeled-set per-feature medians
    med = np.where(np.isfinite(med), med, 0.0)         # all-NaN labeled column -> 0
    for X in (Xl, Xt):
        for j in range(X.shape[1]):
            m = ~np.isfinite(X[:, j])
            if m.any():
                X[m, j] = med[j]
    mu = Xl.mean(axis=0)
    sd = Xl.std(axis=0)
    sd[sd == 0] = 1.0                                  # as in support._standardize
    Xl = (Xl - mu) / sd
    Xt = (Xt - mu) / sd

    kk = min(int(k), len(labeled_idx) - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xl)
    # Reference spacing: labeled->labeled leave-self-out (drop the 0-distance self column).
    dl, _ = nn.kneighbors(Xl)
    d_ref = float(np.median(dl[:, 1:].mean(axis=1)))
    assert d_ref > 0, "label_free_deficit: degenerate labeled set (zero k-NN spacing)"
    dt, _ = nn.kneighbors(Xt)
    in_labeled = np.isin(target_idx, labeled_idx)
    d = np.where(in_labeled, dt[:, 1:].mean(axis=1), dt[:, :kk].mean(axis=1))
    return np.maximum(0.0, d / d_ref - 1.0)


def ts2_submission_splits(cat, seed=0, budgets=(500, 2000, 5000), bright_cut=22.5,
                          frac_calib=0.15, frac_eval=0.25):
    """Bright pseudo-wide base + nested-budget deep pool + held-out calib/eval slices.

    Returns {"base", "deep_pool", "budget", "calib", "eval"}:
      base      = bright pseudo-wide training set: (i < bright_cut) OR any DESI flag,
                  minus calib/eval members.
      deep_pool = faint remainder (i >= bright_cut, not DESI, not calib/eval), shuffled ONCE.
      budget    = {b: idx}, nested prefixes of deep_pool (first b, sorted); None -> full pool.
      calib     = held-out slice, stratified by i-mag bin (STRAT_BINS) over the full depth.
      eval      = held-out slice, same stratification, disjoint from calib.
    NaN i-mag objects are excluded up front. base/deep_pool/calib/eval pairwise disjoint.
    """
    mi = np.asarray(cat["mag_i_lsst"], float)
    valid = np.isfinite(mi)
    # Holdout RNG stream: calib/eval are drawn FIRST and are the only draws from `rng`, so
    # they never depend on `budgets` or `bright_cut` — those only partition the remainder
    # below, using an independent RNG (the decoupled-RNG lesson, crossfield e90ba6d).
    rng = np.random.default_rng(seed)
    eval_parts, calib_parts = [], []
    for lo, hi in STRAT_BINS:
        in_bin = np.where(valid & (mi > lo) & (mi <= hi))[0]
        perm = rng.permutation(len(in_bin))
        n_eval = int(round(frac_eval * len(in_bin)))
        n_calib = int(round(frac_calib * len(in_bin)))
        eval_parts.append(in_bin[perm[:n_eval]])
        calib_parts.append(in_bin[perm[n_eval:n_eval + n_calib]])
    eval_idx = np.sort(np.concatenate(eval_parts))
    calib_idx = np.sort(np.concatenate(calib_parts))
    held = np.zeros(len(mi), bool)
    held[eval_idx] = True
    held[calib_idx] = True

    desi = np.zeros(len(mi), bool)
    for s in DESI_FLAGS:
        desi |= np.asarray(cat[s]).astype(bool)
    remain = valid & ~held
    base = np.where(remain & ((mi < bright_cut) | desi))[0]
    pool = np.where(remain & (mi >= bright_cut) & ~desi)[0]
    deep_pool = pool[np.random.default_rng(seed + 1).permutation(len(pool))]
    budget = {}
    for b in tuple(budgets) + (None,):
        take = len(deep_pool) if (b is None or b >= len(deep_pool)) else b
        budget[b] = np.sort(deep_pool[:take])

    parts = [base, deep_pool, calib_idx, eval_idx]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            assert np.intersect1d(parts[i], parts[j]).size == 0
    bs = sorted(b for b in budget if b is not None)
    for b_lo, b_hi in zip(bs, bs[1:]):
        assert np.isin(budget[b_lo], budget[b_hi]).all()
    if bs:
        assert np.isin(budget[bs[-1]], budget[None]).all()
    return {"base": base, "deep_pool": deep_pool, "budget": budget,
            "calib": calib_idx, "eval": eval_idx}
