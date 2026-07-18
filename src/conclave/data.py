"""TS1 data loading and deterministic stratified splitting."""
import os
import numpy as np
import tables_io


def data_path(sim: str, label: str, scenario: str, public_dir: str, taskset: int = 1) -> str:
    return os.path.join(public_dir, f"pz_challenge_taskset_{taskset}_{sim}_{label}_{scenario}.hdf5")


def load_catalog(path: str) -> dict[str, np.ndarray]:
    d = tables_io.read(path)
    keys = d.keys() if hasattr(d, "keys") else d
    return {k: np.asarray(d[k]) for k in keys}


def stratified_split(redshift, frac_val: float, seed: int, n_bins: int = 20):
    z = np.asarray(redshift)
    n = len(z)
    rng = np.random.default_rng(seed)
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_id = np.digitize(z, edges[1:-1])
    val_mask = np.zeros(n, dtype=bool)
    for b in np.unique(bin_id):
        idx = np.where(bin_id == b)[0]
        k = int(round(frac_val * len(idx)))
        chosen = rng.choice(idx, size=k, replace=False)
        val_mask[chosen] = True
    all_idx = np.arange(n)
    return all_idx[~val_mask], all_idx[val_mask]
