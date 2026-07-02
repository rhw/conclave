# conclave

**A committee photo-z ensemble for the LSST-DESC Photometric Redshift Data Challenge (TS1).**

`conclave` combines three complementary photo-z estimators — **PZFlow**, **GPz**, and
**FlexZBoost** — into a single per-object redshift PDF using **convex-QP optimal weights** and a
**global-PIT** recalibration, on LSST 6-band + Roman (Y/J/H) photometry. The optimizer assigns
each member the weight that minimizes the ensemble's conditional-density loss on held-out data,
driving weak members to zero — a committee that selects its own members.

In the DESC challenge's Task Set 1 it beats the best single estimator on every metric in both the
Cardinal and Flagship simulations, and sits deep in the top tier of the challenge's scored
(calibration-heavy) metrics.

## Install

```bash
pip install "conclave @ git+https://github.com/rhw/conclave@ts1-v1"
```

This pulls the RAIL estimator stack (`pz-rail-base`, `-flexzboost`, `-pzflow`, `-gpz-v1`),
`qp-prob`, `tables_io`, `numpy`, and `scipy`.

## Usage

The package exposes the challenge's two Task Set 1 entry points, plus a
`(train_submission_model, infer)` pair for the pretrained-model path:

```python
from conclave.submission import (
    run_taskset_1_training_and_estimation,   # (train_file, test_file, output_file)
    run_taskset_1_estimation_only,           # (model_file, test_file, output_file)
    DEFAULT_CONFIG,                          # PZFlow+GPz+FlexZBoost, optimal weights, global-PIT
)

# train the committee and write a qp p(z) ensemble for the test set
run_taskset_1_training_and_estimation("train.hdf5", "test.hdf5", "estimate.hdf5")
```

The method is config-agnostic via `conclave.submission.Config(members, band_set, weights, recal)`;
`DEFAULT_CONFIG` is the challenge-winning combination.

## Method components

| module | role |
|---|---|
| `conclave.estimators` | RAIL estimator wrappers (PZFlow/GPz/FlexZBoost/…) + non-detection imputation |
| `conclave.ensemble` | common-grid resample, convex-QP `optimal_weights`, weighted `combine` |
| `conclave.recal` | post-hoc recalibrators (`global_pit`, `magbinned_pit`, …) |
| `conclave.submission` | the config-agnostic Task Set 1 entry points |
| `conclave.metrics` | the challenge's scored point + PIT metrics |

## Challenge

LSST-DESC PZ Data Challenge: <https://pz-data-challenge.readthedocs.io> ·
<https://github.com/LSSTDESC/pz_data_challenge>

## License

BSD 3-Clause — see [LICENSE](LICENSE).
