"""Sweep a list of cell configs into a tidy parquet of results."""
import pandas as pd
from conclave import experiment


def run_grid(cells, out_parquet: str) -> pd.DataFrame:
    rows = []
    for c in cells:
        try:
            rows.append(experiment.run_cell(**c))
        except Exception as e:  # a bad cell must not kill the sweep
            rows.append({**{k: c.get(k) for k in
                         ["estimator", "band_set", "recal", "sim", "scenario", "seed"]},
                         "error": f"{type(e).__name__}: {e}"})
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    return df
