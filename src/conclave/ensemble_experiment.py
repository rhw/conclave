"""Ensemble factorial runners: single cell and group (train-once superset)."""
import numpy as np
from conclave import data, experiment, ensemble, recal as recal_mod, metrics, bands


def _score_subset_combo(stacked, member_names, all_member_names,
                        calib_local, test_local, z_pool,
                        weighting, recal, band_set, sim, scenario, seed,
                        pool_ancil=None) -> dict:
    """Score one (subset × weighting × recal) combo from a pre-computed stacked array.

    Parameters
    ----------
    stacked : (K_all, npool, 301) ndarray
        Common-grid PDFs for ALL members in all_member_names over the calib+test pool.
    member_names : list of str
        The subset of members to combine for this combo.
    all_member_names : list of str
        Ordered list of all members whose PDFs are in stacked (row i = all_member_names[i]).
    calib_local : ndarray of int
        Indices into the pool for the calibration split (used for weight fitting and recal.fit).
    test_local : ndarray of int
        Indices into the pool for the test split (used for recal.apply and scoring).
    z_pool : (npool,) ndarray
        True redshifts for the full calib+test pool.
    weighting : str
        "equal_weight" or "optimal_weights".
    recal : str
        Key into recal_mod.RECALIBRATORS.
    band_set, sim, scenario, seed : metadata scalars
        Passed through into the returned row dict.

    Returns
    -------
    dict
        Row with keys: members, band_set, weighting, recal, sim, scenario, seed,
        n_test, weights, plus all keys from metrics.score().
    """
    # Select the subset rows from the full stacked array
    subset_idx = [all_member_names.index(m) for m in member_names]
    sub_stacked = stacked[subset_idx]   # (K_sub, npool, 301)

    # Fit weights on calib split only
    calib_sub = sub_stacked[:, calib_local]
    if weighting == "equal_weight":
        w = np.full(len(member_names), 1.0 / len(member_names))
    else:
        w = ensemble.optimal_weights(calib_sub, z_pool[calib_local])

    # Combine over the FULL pool (so recal.apply can slice any rows from the result).
    # pool_ancil (feat_* photometry over the pool) is attached so feature-aware
    # recalibrators (magbinned_pit, calpit) can read features off `combined`.
    combined = ensemble.combine(sub_stacked, w, ancil=pool_ancil)

    # Recalibrate: fit on calib only, apply on test only (disjoint — no leakage).
    r = recal_mod.RECALIBRATORS[recal]()
    r.fit(combined, z_pool, fit_idx=calib_local)
    out = r.apply(combined, apply_idx=test_local)

    # Point metrics need ancil["zmode"]
    if out.ancil is None or "zmode" not in (out.ancil or {}):
        zmode = out.mode(grid=np.linspace(0.0, 3.0, 301)).squeeze()
        base = {} if out.ancil is None else dict(out.ancil)
        base["zmode"] = zmode
        out.set_ancil(base)

    test_truth = z_pool[test_local]
    s = metrics.score(out, test_truth)

    row = dict(
        members="+".join(member_names),
        band_set=band_set,
        weighting=weighting,
        recal=recal,
        sim=sim,
        scenario=scenario,
        seed=seed,
        n_test=len(test_local),
        weights=list(map(float, w)),
    )
    row.update(s)
    return row


def run_ensemble_cell(members, band_set, weighting, recal, sim, scenario, seed,
                      public_dir, frac_val=0.2, frac_calib=0.2, subsample=None) -> dict:
    """Train members and score one (weighting × recal) combo.

    Public signature and return schema are unchanged.
    """
    full = data.load_catalog(data.data_path(sim, "training", scenario, public_dir))
    if subsample is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(full["redshift"]), size=subsample, replace=False)
        full = {k: v[idx] for k, v in full.items()}
    z = full["redshift"]
    # 3-way split (mirror experiment.run_cell)
    tr_idx, rest_idx = data.stratified_split(z, frac_val + frac_calib, seed)
    ca_rel, te_rel = data.stratified_split(
        z[rest_idx], frac_calib / (frac_val + frac_calib), seed + 1)
    ca_idx = rest_idx[ca_rel]; te_idx = rest_idx[te_rel]
    tr = {k: v[tr_idx] for k, v in full.items()}
    pool_idx = np.concatenate([ca_idx, te_idx])
    pool = {k: v[pool_idx] for k, v in full.items()}
    calib_local = np.arange(len(ca_idx))
    test_local = np.arange(len(ca_idx), len(pool_idx))

    member_ens = []
    for m in members:
        train_fn, est_fn = experiment.ESTIMATORS[m]
        model = train_fn(tr, band_set=band_set, seed=seed, name=f"inf_{m}_{seed}")
        member_ens.append(est_fn(model, pool, band_set=band_set, name=f"est_{m}_{seed}"))

    stacked = ensemble.to_common_grid(member_ens)   # (K, npool, 301)
    z_pool = pool["redshift"]

    return _score_subset_combo(
        stacked, members, members,
        calib_local, test_local, z_pool,
        weighting, recal, band_set, sim, scenario, seed,
    )


def run_ensemble_group(member_superset, subsets, band_set, sim, scenario, seed,
                       public_dir, weightings=("equal_weight", "optimal_weights"),
                       recals=("none", "global_pit"),
                       frac_val=0.2, frac_calib=0.2, subsample=None) -> list:
    """Train the superset once; evaluate all subset × weighting × recal combos.

    Parameters
    ----------
    member_superset : list of str
        All member estimator names.  Each is trained and estimated exactly once.
    subsets : list of list of str
        Each subset is a list of member names drawn from member_superset.
    band_set : str
    sim : str
    scenario : str
    seed : int
    public_dir : str
    weightings : sequence of str
        Weighting strategies to sweep (default: equal + optimal).
    recals : sequence of str
        Recalibrator keys to sweep (default: none + global_pit).
    frac_val : float
    frac_calib : float
    subsample : int or None

    Returns
    -------
    list of dict
        One row per (subset × weighting × recal) combo.
        len == len(subsets) * len(weightings) * len(recals).
    """
    full = data.load_catalog(data.data_path(sim, "training", scenario, public_dir))
    if subsample is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(full["redshift"]), size=subsample, replace=False)
        full = {k: v[idx] for k, v in full.items()}
    z = full["redshift"]

    # 3-way split (same logic as run_ensemble_cell / experiment.run_cell)
    tr_idx, rest_idx = data.stratified_split(z, frac_val + frac_calib, seed)
    ca_rel, te_rel = data.stratified_split(
        z[rest_idx], frac_calib / (frac_val + frac_calib), seed + 1)
    ca_idx = rest_idx[ca_rel]; te_idx = rest_idx[te_rel]
    tr = {k: v[tr_idx] for k, v in full.items()}
    pool_idx = np.concatenate([ca_idx, te_idx])
    pool = {k: v[pool_idx] for k, v in full.items()}
    calib_local = np.arange(len(ca_idx))
    test_local = np.arange(len(ca_idx), len(pool_idx))

    # Train and estimate EACH superset member ONCE
    member_ens = []
    for m in member_superset:
        train_fn, est_fn = experiment.ESTIMATORS[m]
        model = train_fn(tr, band_set=band_set, seed=seed, name=f"inf_{m}_{seed}")
        member_ens.append(est_fn(model, pool, band_set=band_set, name=f"est_{m}_{seed}"))

    # Build the common-grid stack once → (K_all, npool, 301)
    stacked = ensemble.to_common_grid(member_ens)
    z_pool = pool["redshift"]
    # Photometry features over the pool, for feature-aware recalibrators (magbinned_pit).
    pool_ancil = {f"feat_{c}": pool[c] for c in bands.band_columns(band_set)}

    rows = []
    for subset in subsets:
        for weighting in weightings:
            for recal in recals:
                row = _score_subset_combo(
                    stacked, subset, member_superset,
                    calib_local, test_local, z_pool,
                    weighting, recal, band_set, sim, scenario, seed,
                    pool_ancil=pool_ancil,
                )
                rows.append(row)
    return rows
