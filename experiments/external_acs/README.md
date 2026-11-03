# External-data research workflows

These scripts reproduce the retained Zillow/Realtor/ACS experiments. Run from
the repository root after `pip install -e '.[experiments]'` and downloading
the source data as described in [docs/serving.md](../../docs/serving.md).
The competition rules prohibit external data; these submissions are research
results and are not eligible for the competition.

## Local evaluation

```bash
python experiments/external_acs/evaluate_realtor_foundation.py --seed 42
python experiments/external_acs/evaluate_realtor_tuning.py --seed 42
```

The optional second seed is `137`. Each run holds out ZIPs in five folds,
trains on both months for the other ZIPs, and evaluates held-out June labels.
The seen-ZIP part uses May-to-June persistence as a proxy. Seeds reuse the same
ZIPs and selected results are not independent final validation.

## Refit and create submissions

The best public-score single model uses all training rows:

```bash
python experiments/external_acs/predict_realtor.py --home-value-distribution
python experiments/external_acs/create_zillow_submission.py --realtor
```

The three-model blend can be reproduced with:

```bash
python experiments/external_acs/predict_realtor_tuning.py --home-value-distribution
python experiments/external_acs/create_realtor_tuned_submission.py --weights '{"pfn16":0.7,"pfn_arcsine":0.1,"tabicl":0.2}'
```

Predictions and metadata are written to `artifacts/external_acs/`, which is
excluded from Git. Zillow and Realtor.com revise historical data; preserve the
exact downloaded snapshots and source hashes to reproduce the recorded scores.
Attribution: [Realtor.com Economic Research](https://www.realtor.com/research/data/).
