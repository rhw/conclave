"""Config-agnostic TS1 submission entry points wrapping the pzdc harness.

Exposes the two upstream functions
    run_taskset_1_training_and_estimation(train_file, test_file, output_file, config=DEFAULT_CONFIG)
    run_taskset_1_estimation_only(model_file, test_file, output_file, config=DEFAULT_CONFIG)
plus a (train_submission_model, infer) pair for the pretrained-model deliverable.

DEFAULT_CONFIG is the winning combination locked by the Phase-3 ensemble
experiment: PZFlow + GPz + FlexZBoost on lsst_roman, convex-QP optimal weights,
global-PIT recalibration.

No-leakage recipe (the blind 20k test file has no redshift):
  - train_submission_model splits a calibration slice off the *training* file;
    members are trained on the train slice and estimated on the calib slice.
  - The optimal weights are fit on the calib slice and FROZEN into the bundle.
  - The recalibrator is fit on the combined calib ensemble vs calib truth and
    FROZEN into the bundle.
  - infer reuses the frozen weights + recal on the test set; weights are never
    refit on test. The recal leakage guard is cleared after fit because the test
    ensemble is a different object whose row indices are unrelated to calib rows.

Heavy RAIL estimator imports are deferred into the train/infer indirections so
the Config + combiner logic stays importable and unit-testable off-cluster.
"""
from dataclasses import dataclass
import numpy as np

from conclave import ensemble

Z_GRID = ensemble.Z_GRID


@dataclass
class Config:
    members: list[str]
    band_set: str = "lsst_roman"
    weights: str = "optimal"           # "equal" or "optimal" (convex-QP)
    recal: str = "global_pit"


# The winning combination (Phase-3 ensemble experiment).
DEFAULT_CONFIG = Config(members=["pzflow", "gpz", "flexzboost"],
                        band_set="lsst_roman",
                        weights="optimal",
                        recal="global_pit")


def _combine(ens_list, weights):
    """Weighted average of member CDEs on Z_GRID -> qp.interp Ensemble.

    Reuses conclave.ensemble (to_common_grid + equal_weight/combine), never
    reimplements the combine math.  ``weights`` may be the string "equal", a
    pre-fit weight vector (1-D array summing to ~1), or "optimal" is NOT
    accepted here (optimal weights require calib truth — fit them upstream and
    pass the resulting vector).  Carries the object_id ancil from the first
    member (members share rows/order).
    """
    stacked = ensemble.to_common_grid(ens_list)        # (K, npdf, ngrid)
    k = stacked.shape[0]
    if isinstance(weights, str):
        if weights == "equal":
            w = np.full(k, 1.0 / k)
        else:
            raise ValueError(
                f"_combine accepts 'equal' or a weight vector, not {weights!r}")
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (k,):
            raise ValueError(f"weights shape {w.shape} != ({k},)")
    out = ensemble.combine(stacked, w)
    anc = ens_list[0].ancil or {}
    if "object_id" in anc:
        out.set_ancil({"object_id": np.asarray(anc["object_id"]).astype(int)})
    return out


# Indirections so tests can monkeypatch the heavy layer off-cluster.
def _load_catalog(path):
    from conclave import data
    return data.load_catalog(path)


def _estimators():
    # imported lazily: importing experiment pulls in RAIL estimators
    from conclave import experiment
    return experiment.ESTIMATORS


# Module-level name tests monkeypatch; falls back to the lazy real registry.
ESTIMATORS = None


def _get_estimators():
    return ESTIMATORS if ESTIMATORS is not None else _estimators()


CALIB_FRAC = 0.2  # slice of the training file used to fit weights + recalibrator


def train_submission_model(train_file, config=DEFAULT_CONFIG):
    """Train every member on a train slice; fit weights + recalibrator on a
    held-out calib slice of the *training* file (the blind test set has no z).

    Returns a model bundle ``{models, weights, recal, config}`` where ``weights``
    is the FROZEN weight vector (convex-QP optimal, or equal) and ``recal`` is
    the FROZEN fitted recalibrator.  Both are reused (never refit) by infer.
    """
    from conclave import data, recal as recal_mod
    full = _load_catalog(train_file)
    z = np.asarray(full["redshift"])
    tr_idx, ca_idx = data.stratified_split(z, CALIB_FRAC, seed=0)
    tr = {k: np.asarray(v)[tr_idx] for k, v in full.items()}
    ca = {k: np.asarray(v)[ca_idx] for k, v in full.items()}

    est = _get_estimators()
    models, calib_member_ens = {}, []
    for m in config.members:
        train_fn, est_fn = est[m]
        mh = train_fn(tr, band_set=config.band_set, seed=0, name=f"sub_inf_{m}")
        models[m] = mh
        calib_member_ens.append(
            est_fn(mh, ca, band_set=config.band_set, name=f"sub_ca_{m}"))

    # Fit (or set) the weight vector on the calib slice and FREEZE it.
    stacked_ca = ensemble.to_common_grid(calib_member_ens)
    if config.weights == "optimal":
        w = ensemble.optimal_weights(stacked_ca, ca["redshift"])
    elif config.weights == "equal":
        k = stacked_ca.shape[0]
        w = np.full(k, 1.0 / k)
    else:
        raise ValueError(f"unknown weights spec: {config.weights!r}")
    w = np.asarray(w, dtype=float)

    combined_ca = ensemble.combine(stacked_ca, w)

    # Fit the recalibrator on the combined calib ensemble vs calib truth.
    r = recal_mod.RECALIBRATORS[config.recal]()
    r.fit(combined_ca, ca["redshift"], fit_idx=np.arange(combined_ca.npdf))
    # Clear the leakage guard: the test ensemble is a different object/rows.
    r._fit_idx = set()

    return {"models": models, "weights": w, "recal": r, "config": config}


def infer(model_bundle, test_file):
    """Combine member CDEs on the blind test set using the FROZEN weights, apply
    the FROZEN recal, re-attach int object_id (and zmode) ancil.

    Returns a qp.Ensemble with npdf == n_test.
    """
    config = model_bundle["config"]
    est = _get_estimators()
    test = _load_catalog(test_file)
    oid = np.asarray(test["object_id"]).astype(int)

    member_ens = []
    for m in config.members:
        _, est_fn = est[m]
        member_ens.append(est_fn(model_bundle["models"][m], test,
                                 band_set=config.band_set, name=f"sub_te_{m}"))
    stacked_te = ensemble.to_common_grid(member_ens)
    combined = ensemble.combine(stacked_te, model_bundle["weights"])
    out = model_bundle["recal"].apply(combined, apply_idx=np.arange(combined.npdf))

    # The submission must emit a valid, finite p(z) for EVERY object. Combining +
    # recalibrating can leave a few zero-integral / non-finite rows (e.g. objects
    # where members disagree to ~0 density, then global-PIT -> nan); sanitize them
    # to a broad fallback so the deliverable is always well-formed.
    from conclave.estimators import _sanitize_grid_pdfs
    out = _sanitize_grid_pdfs(out)

    # Re-attach object_id (GlobalPIT/CalPIT rebuild an ancil-less ensemble) and
    # ensure zmode is present for point metrics / downstream.
    base = dict(out.ancil) if out.ancil else {}
    base["object_id"] = oid
    if "zmode" not in base:
        base["zmode"] = out.mode(grid=Z_GRID).squeeze()
    out.set_ancil(base)
    return out


def run_taskset_1_training_and_estimation(train_file, test_file, output_file,
                                          config=DEFAULT_CONFIG):
    bundle = train_submission_model(train_file, config)
    ens = infer(bundle, test_file)
    ens.write_to(output_file)


def run_taskset_1_estimation_only(model_file, test_file, output_file,
                                  config=DEFAULT_CONFIG):
    import pickle
    with open(model_file, "rb") as fh:
        bundle = pickle.load(fh)
    ens = infer(bundle, test_file)
    ens.write_to(output_file)
