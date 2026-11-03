# Serving and interpretation

The service implements the single 4-estimator TabPFN model from public score
3.919, using the same ACS, Zillow and Realtor.com Economic Research features.
It returns the most recent observed value for ZIP/state pairs in training and
uses TabPFN for unseen ZIPs. The different three-model stack scored 4.012 and is
not implemented by this service.

## Prepare the environment and data

Install the package with `pip install -e .` in a CUDA-capable environment. The model
downloads its public checkpoint on first use. The training files must be in
`data/`. When installed outside the repository, set `HOUSING_AFFORDABILITY_ROOT`
to the repository or data workspace path. Download the external files with:

```sh
python experiments/external_acs/download_census.py
python experiments/external_acs/download_cost_distributions.py
python experiments/external_acs/download_zillow.py
python experiments/external_acs/download_realtor.py
```

The Realtor history download streams about 0.84 GB and saves only May/June
2026. Zillow and Realtor periodically revise historical series; downloader
hashes are stored in `artifacts/external_acs/`. Preserve the exact snapshots
to reproduce a run. The competition's rules prohibit external data, so this is
research code and the external-data submissions are not competition-eligible.

## Run the service

```sh
python -m housing_affordability.server --host 127.0.0.1 --port 8000
```

The process loads the data, prepares the training context and fits once at
startup. POST JSON to `/predict` with `{"records": [<row>, ...]}`. Each row
must include a five-digit string `Zip` and `State`; unseen rows also need the
remaining input fields from the competition test data. The response preserves
order and contains `Zip`, `State`, `prediction` and `seen`. For example,
create a request from the first test row with:

```sh
python -c 'import json,pandas as pd; d=pd.read_csv("data/test.csv",dtype={"Zip":str}); print(json.dumps({"records":json.loads(d.head(1).to_json(orient="records"))}))' > /tmp/housing-request.json
curl -s -H 'Content-Type: application/json' --data-binary @/tmp/housing-request.json http://127.0.0.1:8000/predict
```

The default service is a single-worker, synchronous research server. It has no
authentication, TLS or deployment controls; bind it locally or
put it behind appropriate infrastructure. `--cache` requests TabPFN's KV
cache; check the measured T4 memory and prediction agreement before using it.
For the tested setup, `python -m housing_affordability.server --cache` enables the faster
configuration. The default remains the exact submission-equivalent path.

For experimental concurrent request coalescing, add `--batch-window-ms 80`
to the cached server command. A single worker collects compatible requests,
including arrivals while the GPU is busy. GPU access is serialized; this is
request micro-batching rather than token-level continuous batching. The threaded
research server has no backpressure or production admission controls. To use
the older tabctx coalescer, install `pip install -e '.[batching]'` on Python
3.12 or newer and add `--batch-strategy tabctx`.

## Single-T4 measurements

The [Kaggle end-to-end run](https://www.kaggle.com/code/atomstack001/housing-serving-t4-end-to-end)
used one Tesla T4, 6,830 training ZIPs, 106 features and the 4-estimator model.
Its timings measure `HousingService.predict` in process, including input
feature preparation, but exclude HTTP/network transport and checkpoint download.
The following are medians of three warmed requests:

| Unseen ZIPs per request | Seconds per request | ZIPs/second |
| ---: | ---: | ---: |
| 1 | 32.14 | 0.03 |
| 32 | 31.94 | 1.00 |
| 256 | 32.63 | 7.85 |
| 1,704 | 39.63 | 42.99 |

Data loading, feature preparation and model fit took **14.29 seconds** after
the checkpoint was available. The first unseen request took **29.97 seconds**.
Peak GPU memory was **4.41 GiB allocated** and **13.93 GiB reserved**. A seen
ZIP lookup took **0.002 seconds**. Predictions differed from the saved
submission predictions by at most **1.42e-14**. Batch requests substantially
improve throughput; a one-ZIP latency of about 32 seconds makes this uncached
configuration unsuitable for interactive use. Measurements are from one run
and do not represent HTTP overhead, queueing, concurrent traffic or service SLAs.

The separate [cached service run](https://www.kaggle.com/code/atomstack001/housing-serving-t4-cache)
measured the same in-process path and batch sizes:

| Unseen ZIPs per request | Cached seconds | ZIPs/second |
| ---: | ---: | ---: |
| 1 | 0.531 | 1.88 |
| 32 | 0.577 | 55.46 |
| 256 | 1.409 | 181.75 |
| 1,704 | 7.369 | 231.23 |

Cache startup, including data loading and fitting, took **41.32 seconds**;
the first request took **0.735 seconds**. Peak GPU memory was **6.88 GiB
allocated / 10.03 GiB reserved**. On these runs, caching improved the median
single-ZIP service time about **60×** and the 1,704-ZIP batch time about
**5.4×**. The maximum prediction difference from the submitted uncached model
was **0.0535 percentage points**; these are separate single runs, not a
statistical latency guarantee.

In the [model-only cache comparison](https://www.kaggle.com/code/atomstack001/housing-serving-t4-benchmark),
KV caching reduced median single-row prediction from **30.65 to 0.569 seconds**;
the 1,704-row batch fell from **39.82 to 8.22 seconds**. Cache construction
took **35.32 seconds**, versus **4.40 seconds** for preprocessing-only fit.
The cache run peaked at **6.88 GiB allocated / 10.03 GiB reserved**. It changed
predictions by up to **0.0535 percentage points** across the 1,704 unseen test
ZIPs. These are model-only timings, separate from the service measurements above;
the cache is an approximate numerical match, not bit-for-bit identical.
On a separate [held-out ZIP fold](https://www.kaggle.com/code/atomstack001/housing-serving-cache-quality)
of 1,366 rows, uncached new-ZIP RMSE was **4.621998** and cached RMSE was
**4.622992**. The mean absolute prediction difference was **0.00822** and the
maximum was **0.09922 percentage points**. This is one local fold, not a
guarantee for every input.

## tabctx coalescing experiment

The [tabctx T4 benchmark](https://www.kaggle.com/code/atomstack001/housing-tabctx-t4-benchmark)
tested the tabctx engine and its 5 ms same-context coalescing window using
32 preprocessed, one-ZIP requests. The results are one burst per concurrency
level and exclude feature preparation, HTTP and network overhead:

| Concurrent clients | Serialized requests/second | Coalesced requests/second | Serialized median latency | Coalesced median latency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.80 | 1.69 | 0.556 s | 0.590 s |
| 8 | 1.67 | 13.47 | 4.656 s | 0.592 s |
| 32 | 1.61 | 49.47 | 10.343 s | 0.639 s |

At 32 concurrent clients this synthetic burst achieved about **30.7×** the
throughput of the same tabctx engine with coalescing disabled. This is not a
comparison of production HTTP servers or a sustained-load measurement. For
one client, the coalescing window added overhead. Cached predictions differed
from the uncached submitted model by at most **0.0535 percentage points**
across 1,704 ZIPs; the test did not evaluate Kaggle targets.

The stock tabctx backend does not directly reproduce this project: the test
uses a custom adapter to select our TabPFN v2 checkpoint, four estimators, and
the original categorical feature schema. Its A100-calibrated default memory
estimator projected **20.60 GiB** and rejected the T4 workload, so this test
used a fixed, workload-specific estimate derived from earlier T4 measurements.
It is not safe as a general production admission policy. The run used tabctx
commit `6397d5caad52c3705c664336b77f3fe39f0cbfd0`, peaked at **6.88 GiB allocated /
10.03 GiB reserved**, and took **34.62 seconds** for context fitting. tabctx
is used optionally by the project's HTTP server. To reproduce the core test, install
that commit of tabctx and run `python -m benchmarks.tabctx_t4` after preparing
the data and reference predictions.

The subsequent [HTTP benchmark](https://www.kaggle.com/code/atomstack001/housing-tabctx-http-scaling)
included localhost HTTP, JSON and feature preparation. At 32 concurrent
one-ZIP requests, the synchronous server achieved **1.55 requests/second**
and the 20 ms coalescing server **41.14 requests/second** (averages of two
short bursts). At 256 clients, batching achieved **26.06 requests/second**
with a roughly **9.4-second** average p95 latency; throughput dropped as
contention increased. This is a short synthetic benchmark, not a sustained
production-load test. Across the HTTP runs, the maximum prediction difference
from the uncached submission was 0.0371 percentage points.

The separate [engine-only sweep](https://www.kaggle.com/code/atomstack001/housing-tabctx-t4-scaling)
reached a median **172.5 requests/second** at 256 clients with a 20 ms
window, using preprocessed inputs and excluding HTTP. The two workloads are
not directly comparable. A [model-mode sweep](https://www.kaggle.com/code/atomstack001/housing-tabpfn-t4-mode-sweep)
found forced float16 failed with NaNs, while float32 was much slower than the
default automatic precision. Raw results are in `docs/benchmarks/`.

In a [follow-up HTTP run](https://www.kaggle.com/code/atomstack001/housing-http-queued-batching),
the queued worker with an 80 ms window reached **108.7 requests/second** at
128 concurrent clients, versus **27.3** for the tabctx coalescer with a 20 ms
window in the same run. Average p95 latency was **1.14 seconds** versus
**4.65 seconds**. These figures average two short localhost bursts; they do
not establish sustained performance or a maximum. The two windows differ,
and the longer window can increase latency at low traffic. Across the run,
predictions differed from the uncached submission by at most **0.0467
percentage points**; this is not a labeled accuracy evaluation.

## Interpretability

`housing_affordability.interpret.grouped_permutation_importance` permutes related feature
groups jointly among held-out rows and reports the change in new-ZIP RMSE.
`local_group_sensitivity` replaces each group with sampled background values
and reports prediction changes. Both explain model behavior, not causal effects;
correlated and derived inputs can make the substituted rows unrealistic, and
importance depends on the chosen background and evaluation sample.

For a sample of 256 ZIPs held out from one seed-42 training fold, two grouped
permutations gave the following mean increase in **new-ZIP RMSE** (baseline
RMSE **4.65**):

| Input group | Mean RMSE increase |
| --- | ---: |
| Zillow home values and price/income ratio | 23.76 |
| Income, rent and derived ratios | 12.46 |
| ACS home-value distribution | 6.15 |
| ACS income distribution and income thresholds | 3.06 |
| Other demographics | 0.98 |
| Realtor listing prices | 0.91 |
| Regional target summaries | 0.90 |
| Location | 0.27 |

This is one held-out sample, not the competition weighted metric. Correlated
groups and derived features are not independent, so these values must not be
added up or interpreted as causal effects. Permuting can create combinations
that do not occur naturally. Full output and benchmark details are produced by
`python -m benchmarks.benchmark`.

The raw measurements and grouped-permutation results are in
[`docs/benchmarks/`](benchmarks/). The T4 environment used PyTorch
**2.11.0+cu130** and `tabpfn==9.2.0`. Reproduce the end-to-end runs with
`python -m benchmarks.benchmark_service`, `python -m benchmarks.benchmark_cache`, and
`python -m benchmarks.benchmark_cache_quality` after preparing the data. These
scripts compare against `artifacts/external_acs/foundation_realtor_test.csv`;
recreate that file with `python experiments/external_acs/predict_realtor.py
--home-value-distribution` on a CUDA machine.

Run tests with `python -m unittest discover -s tests`.
