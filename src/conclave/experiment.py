"""One factorial cell: data → estimator → recal → metrics → tidy row."""
import numpy as np
from conclave import data, estimators, recal as recal_mod, metrics

ESTIMATORS = {
    "bpz": (estimators.train_bpz, estimators.estimate_bpz),
    "cmnn": (estimators.train_cmnn, estimators.estimate_cmnn),
    "dnf": (estimators.train_dnf, estimators.estimate_dnf),
    "flexzboost": (estimators.train_flexzboost, estimators.estimate_flexzboost),
    "gpz": (estimators.train_gpz, estimators.estimate_gpz),
    "knn": (estimators.train_knn, estimators.estimate_knn),
    "pzflow": (estimators.train_pzflow, estimators.estimate_pzflow),
    "randomforest": (estimators.train_randomforest, estimators.estimate_randomforest),
    "trainz": (estimators.train_trainz, estimators.estimate_trainz),
    "tpz": (estimators.train_tpz, estimators.estimate_tpz),
}


def _make_recal(recal, recal_kwargs):
    cls = recal_mod.RECALIBRATORS[recal]
    return cls(**(recal_kwargs or {}))


def run_cell(estimator, band_set, recal, sim, scenario, seed, public_dir,
             frac_val=0.2, frac_calib=0.0, subsample=None, recal_kwargs=None) -> dict:
    full = data.load_catalog(data.data_path(sim, "training", scenario, public_dir))
    if subsample is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(full["redshift"]), size=subsample, replace=False)
        full = {k: v[idx] for k, v in full.items()}
    train_fn, est_fn = ESTIMATORS[estimator]
    if frac_calib <= 0.0:
        # Phase-1 path (identity recal scored on full val)
        tr_idx, va_idx = data.stratified_split(full["redshift"], frac_val, seed)
        tr = {k: v[tr_idx] for k, v in full.items()}; va = {k: v[va_idx] for k, v in full.items()}
        model = train_fn(tr, band_set=band_set, seed=seed, name=f"inf_{seed}")
        ens = est_fn(model, va, band_set=band_set, name=f"est_{seed}")
        r = _make_recal(recal, recal_kwargs)
        r.fit(ens, va["redshift"], fit_idx=np.array([], dtype=int))
        ens = r.apply(ens, apply_idx=np.arange(ens.npdf)); test_truth = va["redshift"]; n_test = len(va_idx)
    else:
        # 3-way: train / calib / test
        z = full["redshift"]
        tr_idx, rest_idx = data.stratified_split(z, frac_val + frac_calib, seed)
        ca_rel, te_rel = data.stratified_split(z[rest_idx], frac_calib / (frac_val + frac_calib), seed + 1)
        ca_idx = rest_idx[ca_rel]; te_idx = rest_idx[te_rel]
        tr = {k: v[tr_idx] for k, v in full.items()}
        pool_idx = np.concatenate([ca_idx, te_idx])
        pool = {k: v[pool_idx] for k, v in full.items()}
        calib_local = np.arange(len(ca_idx)); test_local = np.arange(len(ca_idx), len(pool_idx))
        model = train_fn(tr, band_set=band_set, seed=seed, name=f"inf_{seed}")
        ens = est_fn(model, pool, band_set=band_set, name=f"est_{seed}")
        r = _make_recal(recal, recal_kwargs)
        r.fit(ens, pool["redshift"], fit_idx=calib_local)
        ens = r.apply(ens, apply_idx=test_local)
        test_truth = pool["redshift"][test_local]; n_test = len(te_idx)
    # Ensure ancil["zmode"] is present for point metrics; some recalibrators
    # (e.g. GlobalPITRecalibrator) return a rebuilt ensemble with no ancil.
    if ens.ancil is None or "zmode" not in (ens.ancil or {}):
        zmode = ens.mode(grid=np.linspace(0.0, 3.0, 301)).squeeze()
        base = {} if ens.ancil is None else dict(ens.ancil)
        base["zmode"] = zmode
        ens.set_ancil(base)
    s = metrics.score(ens, test_truth)
    row = dict(estimator=estimator, band_set=band_set, recal=recal, sim=sim,
               scenario=scenario, seed=seed, n_test=n_test, frac_calib=frac_calib)
    row.update(s); return row
