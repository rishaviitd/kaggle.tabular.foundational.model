# Kaggle: Tabular Foundation Models

Predict the percentage of households that can afford housing in each US ZIP
code. The competition score is `0.2 × RMSE_seen + 0.8 × RMSE_new`; lower is
better. The current goal is a score of 3 or less.

See [experiment.md](experiment.md) for results and the direction for further
work, [docs/data.md](docs/data.md) for the original data description, and
[docs/split.md](docs/split.md) for the local split.

## Setup

Use Python 3.14, create a virtual environment, and install dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the official files from the [competition data page](https://www.kaggle.com/competitions/predicting-housing-affordability-in-the-u-s/data)
and place `train.csv`, `test.csv`, and `sample_submission.csv` in `data/`.
These files and generated outputs are intentionally excluded from Git.

## Local evaluation

```sh
python scripts/split_data.py
mkdir -p artifacts
python scripts/evaluate_validation.py --save artifacts/val_base.npz
python scripts/grouped_validation.py --base artifacts/val_base.npz --save artifacts/val_local_predictions.csv
```

TabICL downloads its public pretrained checkpoint on first use. On Apple
Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1` if required. The validation
score has been inspected during development and is not a fresh holdout.

To try the experimental TabFM dev evaluation:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/tabfm_dev.py --context 256 --estimators 2
```

The checkpoint is approximately 6.6 GB. Its license restricts commercial use;
check competition eligibility before using it in a submission.

## Create a submission

```sh
mkdir -p artifacts
python scripts/submission_base.py --save artifacts/test_base.npz
python scripts/create_submission.py --base artifacts/test_base.npz --save artifacts/submission.csv
```

Upload the resulting CSV to Kaggle. The script checks its ZIP and state keys
against the sample submission and keeps predictions between 0 and 100.

## Competition rules

The [rules](https://www.kaggle.com/competitions/predicting-housing-affordability-in-the-u-s/rules)
prohibit sharing competition data and using external datasets. They also
require publicly shared competition code to be shared on the competition's
Kaggle forum or notebooks. Check model licenses before submitting or sharing.
