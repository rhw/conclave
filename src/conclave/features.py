# src/pzdc/features.py
"""Shared color feature builder: adjacent-band colors + propagated errors.

NaN/non-detection policy: if either magnitude is non-finite the color (and its
error) is non-finite; imputation is left to downstream estimators (the harness
already median-imputes u-band non-detections via bands.MAG_LIMITS where needed).

TODO (Phase-P / Cal-PIT): wire add_colors into experiment.run_cell via a
``use_colors=True`` flag and set ancil["feat_*"] on each estimator's output so
CalPITRecalibrator can use them in the ensemble sweep.
"""
import numpy as np
from conclave import bands


def color_names(band_set: str) -> list[str]:
    cols = bands.band_columns(band_set)
    return [f"color_{a}_{b}" for a, b in zip(cols[:-1], cols[1:])]


def add_colors(data_dict: dict, band_set: str) -> dict:
    out = dict(data_dict)
    cols = bands.band_columns(band_set)
    for a, b in zip(cols[:-1], cols[1:]):
        ma = np.asarray(data_dict[a], dtype=float)
        mb = np.asarray(data_dict[b], dtype=float)
        color = ma - mb
        out[f"color_{a}_{b}"] = color
        ea = np.asarray(data_dict[f"{a}_err"], dtype=float)
        eb = np.asarray(data_dict[f"{b}_err"], dtype=float)
        color_err = np.sqrt(ea ** 2 + eb ** 2)
        # propagate NaN from magnitudes to error (matches NaN policy: non-finite mag → non-finite color and error)
        color_err[~np.isfinite(color)] = np.nan
        out[f"color_{a}_{b}_err"] = color_err
    return out
