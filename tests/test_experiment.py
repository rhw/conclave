import numpy as np, pandas as pd, os
from conclave import experiment, grid

PUBLIC = "/oak/stanford/orgs/kipac/users/risahw/pz_data_challenge/repo/public"

def test_run_cell_row():
    row = experiment.run_cell(estimator="flexzboost", band_set="lsst", recal="none",
                              sim="cardinal", scenario="10yr", seed=3,
                              public_dir=PUBLIC, subsample=3000)
    for k in ["estimator", "band_set", "recal", "sim", "scenario", "seed",
              "mean", "std", "outlier", "CDELoss", "PIT_ks"]:
        assert k in row
    assert row["estimator"] == "flexzboost" and np.isfinite(row["CDELoss"])

def test_run_cell_3way_global_pit():
    row = experiment.run_cell(estimator="flexzboost", band_set="lsst", recal="global_pit",
                              sim="cardinal", scenario="10yr", seed=3, public_dir=PUBLIC,
                              subsample=3000, frac_val=0.2, frac_calib=0.2)
    assert row["recal"] == "global_pit" and np.isfinite(row["CDELoss"]) and np.isfinite(row["PIT_ks"])

def test_run_cell_gpz():
    row = experiment.run_cell(estimator="gpz", band_set="lsst", recal="none",
                              sim="cardinal", scenario="10yr", seed=3,
                              public_dir=PUBLIC, subsample=2000)
    assert row["estimator"] == "gpz" and np.isfinite(row["CDELoss"])

def test_run_grid_writes_parquet(tmp_path):
    cells = [dict(estimator="flexzboost", band_set="lsst", recal="none",
                  sim="cardinal", scenario="10yr", seed=3, public_dir=PUBLIC, subsample=3000)]
    out = str(tmp_path / "runs.parquet")
    df = grid.run_grid(cells, out)
    assert os.path.exists(out) and len(df) == 1
