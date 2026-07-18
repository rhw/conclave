"""Logical band-set → RAIL catalog tag, backed by the packaged data/catalogs_pzdc.yaml."""
import os
from rail.utils import catalog_utils

BAND_SETS = {"lsst": "pzdc_lsst", "lsst_roman": "pzdc_lsst_roman", "hsc_grizy": "pzdc_hsc_grizy"}
_LSST = [f"mag_{b}_lsst" for b in "ugrizy"]
_ROMAN = ["mag_Y_roman", "mag_J_roman", "mag_H_roman"]
_HSC_GRIZY = [f"mag_{b}_lsst" for b in "grizy"]

# 5-sigma limiting magnitudes per band column (mirror of data/catalogs_pzdc.yaml).
# Used to impute non-detections (NaN magnitudes) for estimators that bypass RAIL's
# catalog-tag nondetect handling (e.g. PZFlow, which reads column_names directly).
MAG_LIMITS = {
    "mag_u_lsst": 26.4, "mag_g_lsst": 27.8, "mag_r_lsst": 27.1,
    "mag_i_lsst": 26.7, "mag_z_lsst": 25.8, "mag_y_lsst": 24.6,
    "mag_Y_roman": 26.5, "mag_J_roman": 26.5, "mag_H_roman": 26.5,
}


def catalog_yaml_path() -> str:
    # Packaged data — works both editable (src/conclave/catalogs/) and pip-installed
    # (shipped via [tool.setuptools.package-data]). Dir is `catalogs/` not `data/` to
    # avoid colliding with the conclave.data module.
    return os.path.join(os.path.dirname(__file__), "catalogs", "catalogs_pzdc.yaml")


def band_columns(band_set: str) -> list[str]:
    if band_set == "lsst":
        return list(_LSST)
    if band_set == "lsst_roman":
        return _LSST + _ROMAN
    if band_set == "hsc_grizy":
        return list(_HSC_GRIZY)
    raise KeyError(band_set)


_YAML_LOADED = False


def apply_band_set(band_set: str) -> str:
    """Register our catalog tags (once) and activate the one for `band_set`.

    Robust to the bands already being registered by another loader — e.g. the challenge
    harness loads its own tests/catalogs.yaml, which declares the same DC2LSST_* bands.
    In that case BandFactory aborts on the duplicate before our CatalogTags register, so we
    fall back to registering just the tags (the bands already exist). Guarded to load once
    per process (BandFactory dedupes by filename, but not across colliding files)."""
    global _YAML_LOADED
    tag = BAND_SETS[band_set]
    if not _YAML_LOADED:
        try:
            catalog_utils.load_yaml(catalog_yaml_path())          # bands + our tags
        except Exception:
            from rail.utils.catalog_utils import CatalogTagFactory
            CatalogTagFactory.load_yaml(catalog_yaml_path())      # tags only (bands pre-exist)
        _YAML_LOADED = True
    catalog_utils.apply(tag)
    return tag
