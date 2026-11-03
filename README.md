![Housing Affordability](docs/assets/banner.svg)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-TabPFN%20v2-00A98F)](https://github.com/PriorLabs/TabPFN)
[![Kaggle](https://img.shields.io/badge/Kaggle-public%20score%203.919-20BEFF?logo=kaggle&logoColor=white)](experiment.md)

[Install](#install) · [Serve predictions](#serve-predictions) · [Interpret](#interpret-predictions) · [Experiments](experiment.md) · [Benchmarks](docs/serving.md)

A tabular foundation model for predicting housing affordability by US ZIP code.
The repository includes reproducible evaluation, a persistent inference service,
and model interpretation tools.

The best public Kaggle score was **3.919** using TabPFN v2 with Census, Zillow,
and Realtor.com Economic Research features. The tuned three-model blend scored
**4.012**. Lower is better; the public leaderboard covers only part of the test
set. Both submissions use external data and are not eligible under the
competition rules. See [experiment.md](experiment.md) for the full history.

## Install

Use Python 3.10 or newer with a CUDA-enabled PyTorch environment for GPU runs:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the optional TabICL blend, install `pip install -e '.[experiments]'`.
TabPFN downloads its public checkpoint on first use. Put the competition files
in `data/`, then download the external feature sources using the commands in
[docs/serving.md](docs/serving.md).

## Serve predictions

```bash
housing-affordability-server --cache
```

For experimental request batching, use Python 3.12 or newer, install
`pip install -e '.[batching]'`, and run
`housing-affordability-server --cache --batch-window-ms 20`.

The server accepts `POST /predict` requests containing ZIP-level records. It
uses the latest known target for seen ZIPs and the fitted model for unseen ZIPs.
See [docs/serving.md](docs/serving.md) for the request format, reproducibility
instructions, measured accuracy tradeoff and security limitations.

The same service can be used directly from Python:

```python
import pandas as pd
from housing_affordability.server import HousingService

history = pd.read_csv("data/train.csv", dtype={"Zip": str})
service = HousingService(history, cache=True)
records = pd.read_csv("data/test.csv", dtype={"Zip": str}).head(1).to_dict("records")
predictions = service.predict(records)
```

On a single T4, the cached service measured **0.53 seconds for one unseen ZIP**
and **7.37 seconds for 1,704 ZIPs** in process. These timings exclude HTTP and
network overhead and use three warmed measurements. The cache changes
predictions slightly; full benchmarks are in [docs/benchmarks](docs/benchmarks/).

## Interpret predictions

`housing_affordability.interpret` provides grouped permutation importance on
held-out data and local group sensitivity. In one held-out sample, the model
relied most on Zillow home values and income/rent features. These diagnostics
measure model behavior and are not causal explanations. See
[docs/serving.md](docs/serving.md#interpretability) for details and limitations.

## Repository layout

- `housing_affordability/` — feature construction, model inference, HTTP service and interpretation.
- `experiments/external_acs/` — retained data downloads, local evaluation and submission workflows.
- `benchmarks/` — reproducible single-T4 and cache-quality benchmarks.
- `tests/` — service and interpretation tests.
- `docs/` — problem, data, split and serving documentation.
- `data/` — competition files and local splits.

Run tests with `python -m unittest discover -s tests`. Local evaluation splits
and their limitations are described in [docs/split.md](docs/split.md). See the
[competition rules](https://www.kaggle.com/competitions/predicting-housing-affordability-in-the-u-s/rules)
for data-sharing and external-data restrictions.
