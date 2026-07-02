import numpy as np
from conclave import bands

def test_band_sets_registry():
    assert bands.BAND_SETS["lsst"] == "pzdc_lsst"
    assert bands.BAND_SETS["lsst_roman"] == "pzdc_lsst_roman"

def test_band_columns():
    assert bands.band_columns("lsst") == [f"mag_{b}_lsst" for b in "ugrizy"]
    cols = bands.band_columns("lsst_roman")
    assert "mag_H_roman" in cols and len(cols) == 9

def test_apply_band_set_returns_tag():
    assert bands.apply_band_set("lsst_roman") == "pzdc_lsst_roman"
