"""FlexZBoost, GPz, PZFlow, BPZ, CMNN, KNN, TPZ, TrainZ, RandomForest, and DNF wrappers: RAIL inform→estimate → qp.Ensemble with object_id/zmode ancil."""
import numpy as np
import qp
from rail.core import DataStore
from rail.core.data import TableHandle
from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator
from rail.estimation.algos.cmnn import CMNNInformer, CMNNEstimator
from rail.estimation.algos.dnf import DNFInformer, DNFEstimator
# DNFEstimator.__init__ calls self.log.info(...) before RailStage provides a `.log` attribute
# (rail dnf.py:171 -> AttributeError: 'DNFEstimator' has no attribute 'log'). Shim a module
# logger onto the classes so DNF (the dnf_lsst/dnf_roman reference submissions) is runnable.
import logging as _logging
for _dnf_cls in (DNFInformer, DNFEstimator):
    if not isinstance(getattr(_dnf_cls, "log", None), _logging.Logger):
        _dnf_cls.log = _logging.getLogger(_dnf_cls.__name__)
from rail.estimation.algos.flexzboost import FlexZBoostInformer, FlexZBoostEstimator
from rail.estimation.algos.gpz import GPzInformer, GPzEstimator
from rail.estimation.algos.k_nearneigh import KNearNeighInformer, KNearNeighEstimator
from rail.estimation.algos.pzflow_nf import PZFlowInformer, PZFlowEstimator
from rail.estimation.algos.random_forest import RandomForestInformer, RandomForestClassifier
from rail.estimation.algos.train_z import TrainZInformer, TrainZEstimator
from rail.estimation.algos.tpz_lite import TPZliteInformer, TPZliteEstimator
from conclave import bands

# Number of training epochs for PZFlow normalizing-flow training (PZFlow's default ~30
# under-trains). 200 is the CDE-loss/scatter knee (epoch probe 2026-06-21 on 20k: CDELoss
# 100ep=-12.55 -> 200ep=-13.61 -> 400ep=-13.67, i.e. 200->400 is negligible). On the full
# 80k split the flow converges faster than at 20k, so 200 is conservatively converged.
# NOTE: the zero-integral degeneracy was caused by NaN non-detections, not epochs — see
# _impute_nondetect. Future tuning option: include_mag_errors + err_names_dict.
PZFLOW_EPOCHS = 200

# z-grid parameters pinned to match the rest of the harness (metrics uses linspace(0,3,301)).
PZFLOW_ZMIN = 0.0
PZFLOW_ZMAX = 3.0
PZFLOW_NZBINS = 301

# Allow re-use of DataStore keys across multiple train/estimate calls in a session.
# In this RAIL version DataStore.allow_overwrite is a class attribute (not on RailStage).
DataStore.allow_overwrite = True

# Monkey-patch flexcode.XGBoost to default n_jobs=1.
# FlexZBoost's MultiOutputRegressor uses n_jobs=-1 (all cores) by default, spawning
# loky worker processes that get SIGKILL from cgroup OOM on shared Sherlock login nodes.
# Overriding the default to n_jobs=1 forces serial fitting without process spawning.
# This is applied module-level so it takes effect before FlexZBoostInformer.run() fires.
try:
    from flexcode.regression_models import XGBoost as _XGBoost
    _orig_xgb_init = _XGBoost.__init__

    def _xgb_init_serial(self, max_basis, params, *args, **kwargs):
        kwargs.setdefault("n_jobs", 1)
        _orig_xgb_init(self, max_basis, params, *args, **kwargs)

    _XGBoost.__init__ = _xgb_init_serial
except ImportError:
    pass  # flexcode not installed; RAIL will raise when FlexZBoostInformer runs


def make_handle(name: str, data_dict: dict) -> TableHandle:
    """Wrap a column-dict in a RAIL TableHandle (in-memory, no file I/O)."""
    return TableHandle(name, data=data_dict)


def _sanitize_grid_pdfs(ens, grid=None):
    """Replace non-finite or zero-integral PDF rows with a uniform fallback.

    BPZ produces non-finite p(z) rows for ~1% of objects (degenerate template
    fits). These poison CDELoss and PIT metrics (both come out NaN). This helper
    identifies bad rows, substitutes a maximally-uncertain uniform distribution
    (constant density = 1/(zmax-zmin)), clips residual negatives, and renormalizes
    every row to integrate to 1 on the supplied grid.

    Parameters
    ----------
    ens : qp.Ensemble
        Input ensemble (any qp distribution type).
    grid : np.ndarray or None
        Evaluation grid for the PDFs. Defaults to ``np.linspace(0.0, 3.0, 301)``
        to match the rest of the harness.

    Returns
    -------
    qp.Ensemble
        New ``qp.interp`` ensemble on ``grid`` with all rows finite and normalized.
        The original ``ens.ancil`` is preserved; ``ancil["zmode"]`` is recomputed
        from the sanitized PDFs.
    """
    if grid is None:
        grid = np.linspace(0.0, 3.0, 301)
    grid = np.asarray(grid, dtype=float)

    # Evaluate all PDFs on the grid → shape (npdf, ngrid)
    pdf_mat = np.asarray(ens.pdf(grid), dtype=float)

    # Identify bad rows: any non-finite value OR non-positive integral
    row_has_nonfinite = ~np.all(np.isfinite(pdf_mat), axis=1)
    integrals = np.trapezoid(pdf_mat, grid, axis=1)
    row_bad_integral = ~np.isfinite(integrals) | (integrals <= 0.0)
    bad = row_has_nonfinite | row_bad_integral

    if bad.any():
        uniform_density = 1.0 / (grid[-1] - grid[0])
        pdf_mat[bad] = uniform_density

    # Clip residual negatives (can arise from interpolation edge effects)
    np.clip(pdf_mat, 0.0, None, out=pdf_mat)

    # Renormalize every row to integrate to 1
    norms = np.trapezoid(pdf_mat, grid, axis=1)[:, np.newaxis]
    pdf_mat /= norms

    # Build new interp ensemble
    new_ens = qp.Ensemble(qp.interp, data={"xvals": grid, "yvals": pdf_mat})

    # Preserve ancil from original ensemble, recomputing zmode from sanitized PDFs
    ancil = dict(ens.ancil) if ens.ancil is not None else {}
    ancil["zmode"] = new_ens.mode(grid=grid).squeeze()
    new_ens.set_ancil(ancil)

    return new_ens


def _finalize_ensemble(ens, data_dict, band_set):
    """Attach zmode (if missing), object_id, and per-object band magnitudes (feat_*)
    to the ensemble ancil. feat_* are the photometry features Cal-PIT consumes."""
    from conclave import bands
    if ens.ancil is None or "zmode" not in (ens.ancil or {}):
        zmode = ens.mode(grid=np.linspace(0.0, 3.0, 301)).squeeze()
        base = {} if ens.ancil is None else dict(ens.ancil)
        base["zmode"] = zmode
        ens.set_ancil(base)
    ens.ancil["object_id"] = np.asarray(data_dict["object_id"]).astype(int)
    for col in bands.band_columns(band_set):
        ens.ancil[f"feat_{col}"] = np.asarray(data_dict[col])
    return ens


def train_gpz(train_dict, band_set: str, seed: int, name: str):
    """Train a GPz model.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data (must contain redshift + magnitude columns).
    band_set : str
        Band set name (e.g. "lsst"). Applied via bands.apply_band_set which
        mutates RAIL's active catalog config to select the right mag columns.
    seed : int
        Random seed for GPz training.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the trained GPz model.
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", train_dict)
    informer = GPzInformer.make_stage(
        name=name, output_mode="return", seed=seed,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
    )
    return informer.inform(handle)


def estimate_gpz(model_handle, data_dict, band_set: str, name: str):
    """Run GPz photo-z estimation.

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_gpz.
    data_dict : dict
        Column-dict of evaluation data (must contain magnitude columns + object_id).
    band_set : str
        Band set name (e.g. "lsst"). Applied via bands.apply_band_set.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble (Gaussian p(z)) with ancil["zmode"] (float array,
        shape (npdf,)) and ancil["object_id"] (int array, shape (npdf,)).
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", data_dict)
    est = GPzEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
    )
    out = est.estimate(handle)
    return _finalize_ensemble(out.data, data_dict, band_set)


def train_flexzboost(train_dict, band_set: str, seed: int, name: str):
    """Train a FlexZBoost model.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data (must contain redshift + magnitude columns).
    band_set : str
        Band set name (e.g. "lsst"). Applied via bands.apply_band_set which
        mutates RAIL's active catalog config to select the right mag columns.
    seed : int
        Random seed for FlexZBoost training.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the trained FlexZBoost model.
    """
    bands.apply_band_set(band_set)
    handle = make_handle(f"{name}_data", train_dict)
    informer = FlexZBoostInformer.make_stage(
        name=name, output_mode="return", seed=seed,
        hdf5_groupname="", redshift_col="redshift",
    )
    return informer.inform(handle)


def estimate_flexzboost(model_handle, data_dict, band_set: str, name: str):
    """Run FlexZBoost photo-z estimation.

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_flexzboost.
    data_dict : dict
        Column-dict of evaluation data (must contain magnitude columns + object_id).
    band_set : str
        Band set name (e.g. "lsst"). Applied via bands.apply_band_set.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble with ancil["zmode"] (float array, shape (npdf,))
        and ancil["object_id"] (int array, shape (npdf,)).
    """
    bands.apply_band_set(band_set)
    handle = make_handle(f"{name}_data", data_dict)
    est = FlexZBoostEstimator.make_stage(
        name=name, output_mode="return", model=model_handle, hdf5_groupname="",
    )
    out = est.estimate(handle)
    return _finalize_ensemble(out.data, data_dict, band_set)


def _impute_nondetect(data_dict, band_set):
    """Return a shallow copy with NaN magnitudes replaced by per-band 5-sigma limits.

    PZFlow reads ``column_names`` directly and bypasses RAIL's catalog-tag nondetect
    handling, so NaN non-detections (~2% u-band dropouts in TS1) poison the flow and yield
    zero-integral PDFs. Imputing to the limiting magnitude keeps every object while giving
    the flow finite inputs. (Diagnostic 2026-06-21: with-NaN integral=nan; imputed=1.0.)
    """
    out = dict(data_dict)
    for col in bands.band_columns(band_set):
        vals = np.asarray(out[col], dtype=float).copy()
        nan = ~np.isfinite(vals)
        if nan.any():
            vals[nan] = bands.MAG_LIMITS[col]
        out[col] = vals
    return out


def train_pzflow(train_dict, band_set: str, seed: int, name: str):
    """Train a PZFlow normalizing-flow photo-z model.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data (must contain redshift + magnitude columns).
    band_set : str
        Band set name (e.g. "lsst"). Determines the magnitude columns passed to
        PZFlow via ``column_names`` (PZFlow does not use RAIL SHARED_PARAMS).
    seed : int
        Random seed for PZFlow/JAX training.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the trained PZFlow model.
    """
    bands.apply_band_set(band_set)
    cols = bands.band_columns(band_set)
    handle = make_handle(f"{name}_data", _impute_nondetect(train_dict, band_set))
    informer = PZFlowInformer.make_stage(
        name=name, output_mode="return", seed=seed,
        column_names=cols, redshift_col="redshift",
        hdf5_groupname="",
        n_training_epochs=PZFLOW_EPOCHS,
        zmin=PZFLOW_ZMIN, zmax=PZFLOW_ZMAX, nzbins=PZFLOW_NZBINS,
    )
    return informer.inform(handle)


def estimate_pzflow(model_handle, data_dict, band_set: str, name: str):
    """Run PZFlow normalizing-flow photo-z estimation.

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_pzflow.
    data_dict : dict
        Column-dict of evaluation data (must contain magnitude columns + object_id).
    band_set : str
        Band set name (e.g. "lsst"). Determines the magnitude columns passed to
        PZFlow via ``column_names``.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble (interp grid) with ancil["zmode"] (float array,
        shape (npdf,)) and ancil["object_id"] (int array, shape (npdf,)).
    """
    cols = bands.band_columns(band_set)
    est_dict = _impute_nondetect(data_dict, band_set)
    # PZFlow models the JOINT p(mags, z); its estimator's _process_chunk selects the
    # redshift column even at estimate time. On the blind test set (no truth) that
    # KeyErrors, so supply a dummy redshift column — its values are unused because
    # estimate evaluates p(z|mags) on the z-grid, not the input z. (Verified: blind
    # inference fails with "['redshift'] not in index" without this.)
    if "redshift" not in est_dict:
        n_obj = len(np.asarray(est_dict[cols[0]]))
        est_dict = dict(est_dict)
        est_dict["redshift"] = np.zeros(n_obj, dtype=float)
    handle = make_handle(f"{name}_data", est_dict)
    est = PZFlowEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        column_names=cols, hdf5_groupname="",
        zmin=PZFLOW_ZMIN, zmax=PZFLOW_ZMAX, nzbins=PZFLOW_NZBINS,
    )
    out = est.estimate(handle)
    return _finalize_ensemble(out.data, data_dict, band_set)


# ---------------------------------------------------------------------------
# BPZ (Bayesian Photometric Redshifts — SED template fitting)
# Note: BPZliteInformer generates AB template files on first inform; these are
# cached in the env so subsequent calls are fast.
# ---------------------------------------------------------------------------

def train_bpz(train_dict, band_set: str, seed: int, name: str, nt_array=None):
    """Train a BPZ template-fitting model.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data (must contain redshift + magnitude columns).
    band_set : str
        Band set name (e.g. "lsst").
    seed : int
        Random seed passed to BPZliteInformer.
    name : str
        Stage name (used as RAIL DataStore key prefix).
    nt_array : list[int] or None
        Templates-per-type for the prior, e.g. ``[10, 9, 12]`` for the 31-template
        COSMOS set (matches the upstream ``bpz_31temps`` submission). ``None`` keeps
        the BPZ default (8-template CWWSB+starburst).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the trained BPZ model.
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(train_dict, band_set))
    kw = {} if nt_array is None else {"nt_array": list(nt_array)}
    informer = BPZliteInformer.make_stage(
        name=name, output_mode="return", seed=seed,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols, **kw,
    )
    return informer.inform(handle)


def estimate_bpz(model_handle, data_dict, band_set: str, name: str, spectra_file=None):
    """Run BPZ photo-z estimation.

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_bpz.
    data_dict : dict
        Column-dict of evaluation data (must contain magnitude columns + object_id).
    band_set : str
        Band set name (e.g. "lsst").
    name : str
        Stage name (used as RAIL DataStore key prefix).
    spectra_file : str or None
        SED template list, e.g. ``"COSMOS_seds.list"`` for the 31-template set.
        ``None`` keeps the BPZ default (CWWSB+starburst, 8 templates).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble with ancil["zmode"] and ancil["object_id"].
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(data_dict, band_set))
    kw = {} if spectra_file is None else {"spectra_file": spectra_file}
    est = BPZliteEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols, **kw,
    )
    out = est.estimate(handle)
    sanitized = _sanitize_grid_pdfs(out.data)
    return _finalize_ensemble(sanitized, data_dict, band_set)


# ---------------------------------------------------------------------------
# Lephare (SED template fitting; independent second template code, standalone only).
# Filter transmission files live under $LEPHAREDIR/filt (staged on $OAK). LSST = the
# 6-band ouphz_micol total throughputs; Roman Y/J/H = WFI F106/F129/F158. The challenge
# bands map IN ORDER to FILTER_LIST. LEPHAREDIR must be set BEFORE importing rail's
# lephare module (it pins /tmp otherwise), and LephareInformer/Estimator hit the same
# missing-self.log RAIL bug as DNF -> shim a logger. Verified end-to-end (job 31805692).
# ---------------------------------------------------------------------------
_LEPHARE_FILT = {
    "mag_u_lsst": "ouphz_micol/LSST_total_u.pb", "mag_g_lsst": "ouphz_micol/LSST_total_g.pb",
    "mag_r_lsst": "ouphz_micol/LSST_total_r.pb", "mag_i_lsst": "ouphz_micol/LSST_total_i.pb",
    "mag_z_lsst": "ouphz_micol/LSST_total_z.pb", "mag_y_lsst": "ouphz_micol/LSST_total_y.pb",
    "mag_Y_roman": "roman/Roman_WFI.F106.dat", "mag_J_roman": "roman/Roman_WFI.F129.dat",
    "mag_H_roman": "roman/Roman_WFI.F158.dat",
}


def _lephare_setup():
    """Set LEPHAREDIR/WORK (persistent $OAK) then import + logger-shim rail's lephare."""
    import os
    base = os.path.expandvars("$OAK/pz_data_challenge")
    os.environ.setdefault("LEPHAREDIR", f"{base}/lephare_data")
    os.environ.setdefault("LEPHAREWORK", f"{base}/lephare_work")
    from rail.estimation.algos.lephare import LephareInformer, LephareEstimator
    import logging as _logging
    for _c in (LephareInformer, LephareEstimator):
        if not isinstance(getattr(_c, "log", None), _logging.Logger):
            _c.log = _logging.getLogger(_c.__name__)
    return LephareInformer, LephareEstimator


def _lephare_cfg(band_cols, z_step="0.02,0.,3."):
    flist = ",".join(_LEPHARE_FILT[c] for c in band_cols)
    return {
        "lephare.FILTER_LIST": flist,
        "lephare.FILTER_CALIB": ",".join(["0"] * len(band_cols)),
        "lephare.FILTER_FILE": "filter_pzdc",
        "lephare.Z_STEP": z_step,
    }


def train_lephare(train_dict, band_set: str, seed: int, name: str):
    """Train (prepare SED libraries + AUTO_ADAPT offsets) a Lephare model. Heavy prepare
    stage -> run via Slurm (login-node SIGKILL)."""
    LephareInformer, _ = _lephare_setup()
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(train_dict, band_set))
    informer = LephareInformer.make_stage(
        name=name, hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols, ref_band="mag_i_lsst",
        nzbins=301, **_lephare_cfg(band_cols),
    )
    return informer.inform(handle)


def estimate_lephare(model_handle, data_dict, band_set: str, name: str):
    """Run Lephare photo-z estimation -> qp.Ensemble (sanitized, ancil attached)."""
    _, LephareEstimator = _lephare_setup()
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(data_dict, band_set))
    est = LephareEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        hdf5_groupname="", bands=band_cols, err_bands=err_cols, ref_band="mag_i_lsst",
    )
    out = est.estimate(handle)
    sanitized = _sanitize_grid_pdfs(out.data)
    return _finalize_ensemble(sanitized, data_dict, band_set)


# ---------------------------------------------------------------------------
# CMNN (Color–Matched Nearest Neighbours)
# ---------------------------------------------------------------------------

def train_cmnn(train_dict, band_set: str, seed: int, name: str):
    """Train a CMNN color-matched nearest-neighbour model.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data (must contain redshift + magnitude columns).
    band_set : str
        Band set name (e.g. "lsst").
    seed : int
        Random seed passed to CMNNInformer.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the trained CMNN model.
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(train_dict, band_set))
    informer = CMNNInformer.make_stage(
        name=name, output_mode="return", seed=seed,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
    )
    return informer.inform(handle)


def estimate_cmnn(model_handle, data_dict, band_set: str, name: str):
    """Run CMNN photo-z estimation.

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_cmnn.
    data_dict : dict
        Column-dict of evaluation data (must contain magnitude columns + object_id).
    band_set : str
        Band set name (e.g. "lsst").
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble with ancil["zmode"] and ancil["object_id"].
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(data_dict, band_set))
    est = CMNNEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
    )
    out = est.estimate(handle)
    return _finalize_ensemble(out.data, data_dict, band_set)


# ---------------------------------------------------------------------------
# KNN (K-Nearest Neighbours)
# ---------------------------------------------------------------------------

def train_knn(train_dict, band_set: str, seed: int, name: str):
    """Train a KNN photo-z model.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data (must contain redshift + magnitude columns).
    band_set : str
        Band set name (e.g. "lsst").
    seed : int
        Random seed passed to KNearNeighInformer.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the trained KNN model.
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(train_dict, band_set))
    informer = KNearNeighInformer.make_stage(
        name=name, output_mode="return", seed=seed,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
    )
    return informer.inform(handle)


def estimate_knn(model_handle, data_dict, band_set: str, name: str):
    """Run KNN photo-z estimation.

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_knn.
    data_dict : dict
        Column-dict of evaluation data (must contain magnitude columns + object_id).
    band_set : str
        Band set name (e.g. "lsst").
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble with ancil["zmode"] and ancil["object_id"].
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(data_dict, band_set))
    est = KNearNeighEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
    )
    out = est.estimate(handle)
    return _finalize_ensemble(out.data, data_dict, band_set)


# ---------------------------------------------------------------------------
# TPZ (Trees for Photo-Z — random forest regression)
# ---------------------------------------------------------------------------

def train_tpz(train_dict, band_set: str, seed: int, name: str):
    """Train a TPZ random-forest photo-z model.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data (must contain redshift + magnitude columns).
    band_set : str
        Band set name (e.g. "lsst").
    seed : int
        Random seed passed to TPZliteInformer.
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the trained TPZ model.
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(train_dict, band_set))
    informer = TPZliteInformer.make_stage(
        name=name, output_mode="return", seed=seed,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
    )
    return informer.inform(handle)


def estimate_tpz(model_handle, data_dict, band_set: str, name: str):
    """Run TPZ photo-z estimation.

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_tpz.
    data_dict : dict
        Column-dict of evaluation data (must contain magnitude columns + object_id).
    band_set : str
        Band set name (e.g. "lsst").
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble with ancil["zmode"] and ancil["object_id"].
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(data_dict, band_set))
    est = TPZliteEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
    )
    out = est.estimate(handle)
    return _finalize_ensemble(out.data, data_dict, band_set)


# ---------------------------------------------------------------------------
# TrainZ (trivial n(z) baseline — ignores photometry)
# Probe results (2026-06-26): TrainZInformer params: hdf5_groupname, zmin, zmax,
# nzbins, redshift_col. TrainZEstimator params: same + id_col, calc_summary_stats,
# calculated_point_estimates, recompute_point_estimates. No bands/err_bands.
# ---------------------------------------------------------------------------

def train_trainz(train_dict, band_set: str, seed: int, name: str):
    """Train a TrainZ model (trivial n(z) baseline; ignores photometry).

    TrainZ builds the training-set n(z) histogram and returns it as the p(z)
    for every test object — independent of photometry. This is the floor
    reference: any useful photometric estimator should beat it.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data. Only the redshift column is used.
    band_set : str
        Band set name (unused; accepted for API uniformity).
    seed : int
        Random seed (unused; accepted for API uniformity).
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the TrainZ n(z) model.
    """
    # TrainZ ignores bands/photometry entirely; no apply_band_set or imputation needed.
    handle = make_handle(f"{name}_data", train_dict)
    informer = TrainZInformer.make_stage(
        name=name, output_mode="return",
        hdf5_groupname="", redshift_col="redshift",
        zmin=0.0, zmax=3.0, nzbins=301,
    )
    return informer.inform(handle)


def estimate_trainz(model_handle, data_dict, band_set: str, name: str):
    """Run TrainZ estimation (returns training n(z) for every object; photometry unused).

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_trainz.
    data_dict : dict
        Column-dict of evaluation data (must contain object_id for ancil).
    band_set : str
        Band set name (used only for feat_* ancil columns in _finalize_ensemble).
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble (same n(z) for every object) with ancil["zmode"]
        and ancil["object_id"].
    """
    handle = make_handle(f"{name}_data", data_dict)
    est = TrainZEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        hdf5_groupname="", redshift_col="redshift",
        zmin=0.0, zmax=3.0, nzbins=301,
        id_col="object_id",
    )
    out = est.estimate(handle)
    return _finalize_ensemble(out.data, data_dict, band_set)


# ---------------------------------------------------------------------------
# RandomForest (easy_forest reference submission)
# Probe results (2026-06-26): rail.estimation.algos.random_forest does NOT expose
# a RandomForestEstimator class. The module contains RandomForestClassifier (a
# tomographic-bin classifier, entrypoint: classify(), output: TableHandle of bin
# assignments) and RandomForestInformer. There is no PDF Estimator counterpart.
# The brief assumed RandomForestEstimator would exist — it does not in this RAIL
# install. Wrapper raises NotImplementedError with a clear diagnostic message.
# This means configs/phase3_compare.py and the sbatch script must NOT include
# "randomforest"; the test is skipped with a mark.
# ---------------------------------------------------------------------------

def train_randomforest(train_dict, band_set: str, seed: int, name: str):
    """Train step for RandomForest — not available in this RAIL install.

    The rail.estimation.algos.random_forest module in the pzdc conda env only
    exposes RandomForestClassifier (a tomographic-bin classifier) with no
    PDF Estimator counterpart. Call raises NotImplementedError.
    """
    raise NotImplementedError(
        "RandomForestEstimator does not exist in the installed RAIL version "
        "(rail.estimation.algos.random_forest only has RandomForestClassifier, "
        "which produces bin assignments via classify(), not photo-z PDFs). "
        "Cannot implement the easy_forest reference submission via this interface. "
        "Consider substituting with a different RAIL PDF estimator."
    )


def estimate_randomforest(model_handle, data_dict, band_set: str, name: str):
    """Estimate step for RandomForest — not available in this RAIL install. See train_randomforest."""
    raise NotImplementedError(
        "RandomForestEstimator does not exist in the installed RAIL version. "
        "See train_randomforest docstring for details."
    )


# ---------------------------------------------------------------------------
# DNF (Directional Neighbourhood Fitting — dnf_lsst / dnf_roman reference submissions)
# Probe results (2026-06-26): DNFInformer params: hdf5_groupname, bands, err_bands,
# redshift_col, mag_limits, nondetect_val, zmin, zmax, nzbins, and many others.
# DNFEstimator params: same + id_col, chunk_size, calc_summary_stats, selection_mode.
# Both use standard bands/err_bands interface. Outputs qp.Ensemble via estimate().
# ---------------------------------------------------------------------------

def train_dnf(train_dict, band_set: str, seed: int, name: str):
    """Train a DNF (Directional Neighbourhood Fitting) photo-z model.

    Parameters
    ----------
    train_dict : dict
        Column-dict of training data (must contain redshift + magnitude columns).
    band_set : str
        Band set name (e.g. "lsst" or "lsst_roman").
    seed : int
        Random seed (accepted for API uniformity; DNF is deterministic).
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    model_handle : ModelHandle
        RAIL ModelHandle containing the trained DNF model.
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(train_dict, band_set))
    informer = DNFInformer.make_stage(
        name=name, output_mode="return",
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
        zmin=0.0, zmax=3.0, nzbins=301,
    )
    return informer.inform(handle)


def estimate_dnf(model_handle, data_dict, band_set: str, name: str):
    """Run DNF photo-z estimation.

    Parameters
    ----------
    model_handle : ModelHandle
        Trained model handle returned by train_dnf.
    data_dict : dict
        Column-dict of evaluation data (must contain magnitude columns + object_id).
    band_set : str
        Band set name (e.g. "lsst" or "lsst_roman").
    name : str
        Stage name (used as RAIL DataStore key prefix).

    Returns
    -------
    ens : qp.Ensemble
        Photo-z PDF ensemble with ancil["zmode"] and ancil["object_id"].
    """
    bands.apply_band_set(band_set)
    band_cols = bands.band_columns(band_set)
    err_cols = [f"{c}_err" for c in band_cols]
    handle = make_handle(f"{name}_data", _impute_nondetect(data_dict, band_set))
    est = DNFEstimator.make_stage(
        name=name, output_mode="return", model=model_handle,
        hdf5_groupname="", redshift_col="redshift",
        bands=band_cols, err_bands=err_cols,
        zmin=0.0, zmax=3.0, nzbins=301,
        id_col="object_id",
    )
    out = est.estimate(handle)
    sanitized = _sanitize_grid_pdfs(out.data)
    return _finalize_ensemble(sanitized, data_dict, band_set)
