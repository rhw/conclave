"""Per-object diagnostics: dump (z_true, zmode, PIT, mag_i) per validation object
for a cell, so we can resolve *where* a band set helps — binned by true redshift
and by i-magnitude — rather than only the dataset-aggregate scalars in metrics.score.

Mirrors the Phase-1 path of experiment.run_cell (frac_val split, identity recal),
but returns the per-object arrays instead of scoring them to scalars.
"""
import numpy as np
import pandas as pd

from conclave import data, metrics
from conclave.experiment import ESTIMATORS


def per_object_predictions(estimator, band_set, sim, scenario, seed, public_dir,
                           frac_val=0.2) -> pd.DataFrame:
    """Train on the train split, predict the val split, return one row per val object.

    Columns: estimator, band_set, sim, scenario, seed, object_id, z_true, zmode,
    pit, mag_i (the i-band magnitude, the challenge's selection variable).
    """
    full = data.load_catalog(data.data_path(sim, "training", scenario, public_dir))
    train_fn, est_fn = ESTIMATORS[estimator]
    tr_idx, va_idx = data.stratified_split(full["redshift"], frac_val, seed)
    tr = {k: v[tr_idx] for k, v in full.items()}
    va = {k: v[va_idx] for k, v in full.items()}

    model = train_fn(tr, band_set=band_set, seed=seed, name=f"inf_{seed}")
    ens = est_fn(model, va, band_set=band_set, name=f"est_{seed}")

    z_true = np.asarray(va["redshift"], dtype=float)
    zmode = np.asarray(ens.ancil["zmode"], dtype=float).squeeze()
    pit = metrics._pit_values(ens, z_true)
    mag_i = np.asarray(va["mag_i_lsst"], dtype=float)
    object_id = np.asarray(va["object_id"]).astype(np.int64)

    return pd.DataFrame(dict(
        estimator=estimator, band_set=band_set, sim=sim, scenario=scenario,
        seed=seed, object_id=object_id, z_true=z_true, zmode=zmode,
        pit=pit, mag_i=mag_i,
    ))


def run_diag_grid(cells, out_parquet: str) -> pd.DataFrame:
    """Concatenate per-object predictions for a list of cells into one parquet."""
    frames = []
    for c in cells:
        frames.append(per_object_predictions(**c))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out_parquet, index=False)
    return df
