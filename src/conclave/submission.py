"""Config-agnostic submission entry points wrapping the pzdc harness.

Exposes the upstream functions
    run_taskset_1_training_and_estimation(train_file, test_file, output_file, config=DEFAULT_CONFIG)
    run_taskset_1_estimation_only(model_file, test_file, output_file, config=DEFAULT_CONFIG)
    run_taskset_2_training_and_estimation(train_file, test_file, output_file, config=TS2_CONFIG)
    run_taskset_2_estimation_only(model_file, test_file, output_file, config=TS2_CONFIG)
plus a (train_submission_model, infer) pair for the pretrained-model deliverable.

DEFAULT_CONFIG is the winning combination locked by the Phase-3 ensemble
experiment: PZFlow + GPz + FlexZBoost on lsst_roman, convex-QP optimal weights,
global-PIT recalibration.

TS2_CONFIG swaps in the support-deficit-binned PIT recalibrator, settled by the
depth-matched rehearsal (job 33575936): train on ALL provided training objects
(importance weighting demonstrably hurts at depth) + per-deficit-bin PIT remap,
where the deficit is the label-free color-space kNN support deficit of
challenge.label_free_deficit. Because the blind test file arrives without the
training photometry, the model bundle carries a frozen ``deficit_ref`` (the
train-slice band columns); infer rebuilds a merged catalog (reference + test)
and calls the same label_free_deficit, so train- and test-time deficits share
one code path. The deliverable partition is stratified_split(CALIB_FRAC=0.2,
seed=0) on the training file — stratified on REDSHIFT, whereas the evidence
arms' calib slices were stratified on I-MAG (challenge.ts2_submission_splits) —
a known, deliberate recipe difference (red-team M2): the TS1-validated
partition is retained, and the rehearsal winner was selected under the harsher
bright-only-calib deployment condition. Test objects out of training support
digitize into the positive-deficit tail bins and receive their remaps — by
design, not an error path.

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

PRIORSHIFT_SIGMA_Z = 0.075   # PI-approved 2026-07-21 (sweep plateau, post-hoc — disclosed)
PRIORSHIFT_RMAX = 3.0


@dataclass
class Config:
    members: list[str]
    band_set: str = "lsst_roman"
    weights: str = "optimal"           # "equal" or "optimal" (convex-QP)
    recal: str = "global_pit"
    pointest: str = "mode"             # "mode" | "priorshift" (C3, TS2 revision 2026-07)


# The winning combination (Phase-3 ensemble experiment).
DEFAULT_CONFIG = Config(members=["pzflow", "gpz", "flexzboost"],
                        band_set="lsst_roman",
                        weights="optimal",
                        recal="global_pit")

# TS2: same ensemble, support-deficit-binned recalibration. Settled by the
# depth-matched rehearsal (job 33575936, journal 2026-07-11-depth-rehearsal):
# deficitbinned_pit beats the previously packaged magbinned_pit in 36/36
# paired per-seed deployment-condition comparisons and wins 5/7 scored
# metrics vs global_pit on both blind-test i-mag mixes (pre-registered rule).
# pointest="priorshift": the delivered zmode is the C3 empirical-Bayes
# prior-shift mode (conclave.pointest.prior_shift_mode), settled by the
# 2026-07-21 point-estimate bake-off (docs/journal/2026-07-21-*).
TS2_CONFIG = Config(members=["pzflow", "gpz", "flexzboost"],
                    band_set="lsst_roman",
                    weights="optimal",
                    recal="deficitbinned_pit",
                    pointest="priorshift")


def priorshift_zmode(pdf, grid, ztrain):
    """Deployment C3: prior-shift mode with UNIFORM weights (the blind test file
    is the exact target mix; N_hat_test is the plain mean of its delivered PDFs)."""
    from conclave import pointest
    w = np.ones(np.atleast_2d(pdf).shape[0])
    return pointest.prior_shift_mode(pdf, grid, ztrain, w,
                                     sigma_z=PRIORSHIFT_SIGMA_Z, rmax=PRIORSHIFT_RMAX)


def _feat_ancil(cat, band_set):
    """feat_<band> ancil columns (same naming as ts2.run_eval) so feature-conditional
    recalibrators (magbinned_pit) can fit/apply; ignored by global_pit. Only columns
    present in the catalog are attached — a recal needing an absent feature fails
    with a clear KeyError at its own fit/apply, not here."""
    from conclave import bands
    return {f"feat_{c}": np.asarray(cat[c])
            for c in bands.band_columns(band_set) if c in cat}


def _deficit_ref(cat, band_set):
    """Frozen labeled-set photometry (band columns only) stored in the model
    bundle so infer can compute the label-free support deficit of blind test
    objects long after the training file is gone."""
    from conclave import bands
    return {c: np.asarray(cat[c], dtype=float) for c in bands.band_columns(band_set)}


def _deficit_from_ref(ref, target_cat, band_set):
    """feat_deficit for target objects against the FROZEN labeled reference.

    Concatenates the stored labeled band columns with the target's and calls
    challenge.label_free_deficit on the merged catalog, so train- and
    test-time deficits share one code path (identical labeled-set
    standardization, NaN imputation, and d_ref by construction — a target's
    deficit depends only on its own photometry and the frozen reference)."""
    from conclave import bands, challenge
    cols = bands.band_columns(band_set)
    n_lab = len(np.asarray(ref[cols[0]]))
    n_tgt = len(np.asarray(target_cat[cols[0]]))
    merged = {c: np.concatenate([np.asarray(ref[c], dtype=float),
                                 np.asarray(target_cat[c], dtype=float)])
              for c in cols}
    return challenge.label_free_deficit(
        merged, labeled_idx=np.arange(n_lab),
        target_idx=np.arange(n_lab, n_lab + n_tgt), band_set=band_set)


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

    # feat ancil is required by feature-conditional recals (magbinned_pit) and
    # inert for global_pit (whose fit reads only PDFs + truth). Deficit recals
    # additionally need feat_deficit (labeled set = the train slice, matching
    # ts2.run_eval's labeled_idx=mtr_idx) and a frozen reference for infer.
    feat_ca = _feat_ancil(ca, config.band_set)
    deficit_ref = None
    if config.recal in recal_mod.DEFICIT_RECALS:
        deficit_ref = _deficit_ref(tr, config.band_set)
        feat_ca["feat_deficit"] = _deficit_from_ref(deficit_ref, ca, config.band_set)
    combined_ca = ensemble.combine(stacked_ca, w, ancil=feat_ca)

    # Fit the recalibrator on the combined calib ensemble vs calib truth.
    r = recal_mod.RECALIBRATORS[config.recal]()
    r.fit(combined_ca, ca["redshift"], fit_idx=np.arange(combined_ca.npdf))
    # Clear the leakage guard: the test ensemble is a different object/rows.
    r._fit_idx = set()

    # Frozen full-training-file redshift column (train UNION calib) for the
    # C3 prior-shift point estimate (pointest="priorshift"). Stored
    # unconditionally (100k float64 ~= 0.8 MB against a 59 MB bundle) so an
    # existing bundle can be upgraded to priorshift deployment without a
    # retrain.
    ztrain = np.asarray(full["redshift"], dtype=float)

    return {"models": models, "weights": w, "recal": r, "config": config,
            "deficit_ref": deficit_ref, "ztrain": ztrain}


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
    feat_te = _feat_ancil(test, config.band_set)
    from conclave import recal as recal_mod
    if config.recal in recal_mod.DEFICIT_RECALS:
        ref = model_bundle.get("deficit_ref")
        assert ref is not None, (
            "deficit-conditioned recal needs 'deficit_ref' in the model bundle "
            "(frozen labeled photometry, stored by train_submission_model since "
            "the 2026-07 deficitbinned swap) — retrain the bundle")
        feat_te["feat_deficit"] = _deficit_from_ref(ref, test, config.band_set)
    combined = ensemble.combine(stacked_te, model_bundle["weights"], ancil=feat_te)
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
    if getattr(config, "pointest", "mode") == "priorshift":
        ztrain = model_bundle.get("ztrain")
        assert ztrain is not None, (
            "pointest='priorshift' needs 'ztrain' in the model bundle (frozen "
            "training spec-z, stored by train_submission_model since the 2026-07 "
            "C3 revision) — retrain or upgrade the bundle")
        base["zmode"] = priorshift_zmode(np.asarray(out.pdf(Z_GRID)), Z_GRID, ztrain)
    elif "zmode" not in base:
        base["zmode"] = out.mode(grid=Z_GRID).squeeze()
    # Free per-object reliability flag from member PDF spread (npdf aligned with the
    # test set / object_id), so the deliverable carries a quality signal.
    base["disagreement"] = ensemble.member_disagreement(stacked_te)
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


def run_taskset_2_training_and_estimation(train_file, test_file, output_file,
                                          config=TS2_CONFIG):
    bundle = train_submission_model(train_file, config)
    ens = infer(bundle, test_file)
    ens.write_to(output_file)


def run_taskset_2_estimation_only(model_file, test_file, output_file,
                                  config=TS2_CONFIG):
    import pickle
    with open(model_file, "rb") as fh:
        bundle = pickle.load(fh)
    ens = infer(bundle, test_file)
    ens.write_to(output_file)
