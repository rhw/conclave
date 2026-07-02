# tests/test_features.py
import numpy as np
from conclave import features, bands


def test_color_names_are_adjacent_pairs():
    names = features.color_names("lsst")
    # 6 lsst bands -> 5 adjacent colors
    assert names == ["color_mag_u_lsst_mag_g_lsst", "color_mag_g_lsst_mag_r_lsst",
                     "color_mag_r_lsst_mag_i_lsst", "color_mag_i_lsst_mag_z_lsst",
                     "color_mag_z_lsst_mag_y_lsst"]


def test_add_colors_math_and_error_propagation():
    d = {
        "mag_u_lsst": np.array([25.0, 24.0]),
        "mag_g_lsst": np.array([24.0, 23.0]),
        "mag_r_lsst": np.array([23.0, 22.0]),
        "mag_i_lsst": np.array([22.0, 21.0]),
        "mag_z_lsst": np.array([21.0, 20.0]),
        "mag_y_lsst": np.array([20.0, 19.0]),
    }
    for c in bands.band_columns("lsst"):
        d[f"{c}_err"] = np.array([0.03, 0.04])
    out = features.add_colors(d, "lsst")
    # color u-g = mag_u - mag_g = 1.0 for both rows
    np.testing.assert_allclose(out["color_mag_u_lsst_mag_g_lsst"], [1.0, 1.0])
    # all band errors are [0.03, 0.04], so color_err = sqrt(e^2 + e^2) = e*sqrt(2)
    expected_err = np.array([0.03, 0.04]) * np.sqrt(2)
    np.testing.assert_allclose(out["color_mag_u_lsst_mag_g_lsst_err"], expected_err)
    # original columns preserved
    np.testing.assert_allclose(out["mag_u_lsst"], d["mag_u_lsst"])


def test_add_colors_nan_policy():
    d = {c: np.array([24.0, np.nan]) for c in bands.band_columns("lsst")}
    for c in bands.band_columns("lsst"):
        d[f"{c}_err"] = np.array([0.03, 0.03])
    out = features.add_colors(d, "lsst")
    col = out["color_mag_u_lsst_mag_g_lsst"]
    err = out["color_mag_u_lsst_mag_g_lsst_err"]
    assert np.isfinite(col[0]) and not np.isfinite(col[1])  # NaN propagates, no imputation
    assert np.isfinite(err[0]) and not np.isfinite(err[1])
