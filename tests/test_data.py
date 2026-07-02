import numpy as np
import pytest
from conclave import data

PUBLIC = "/oak/stanford/orgs/kipac/users/risahw/pz_data_challenge/repo/public"

def test_data_path():
    p = data.data_path("cardinal", "training", "10yr", PUBLIC)
    assert p.endswith("pz_challenge_taskset_1_cardinal_training_10yr.hdf5")

def test_load_catalog_columns():
    d = data.load_catalog(data.data_path("cardinal", "training", "10yr", PUBLIC))
    for c in ["object_id", "redshift", "mag_i_lsst", "mag_H_roman"]:
        assert c in d
    assert len(d["redshift"]) == 100000

def test_stratified_split_deterministic_and_disjoint():
    z = np.linspace(0.01, 2.2, 1000)
    a1, b1 = data.stratified_split(z, frac_val=0.2, seed=42)
    a2, b2 = data.stratified_split(z, frac_val=0.2, seed=42)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)        # deterministic
    assert set(a1).isdisjoint(set(b1))                              # disjoint
    assert len(a1) + len(b1) == 1000                                # partition
    assert abs(len(b1) - 200) <= 20                                 # ~20% val
    # stratified: val z-distribution matches train within coarse bins
    assert abs(np.median(z[b1]) - np.median(z[a1])) < 0.15
