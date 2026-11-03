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

For the measured optimized path, use:

```sh
housing-affordability-server --cache --optimize --batch-window-ms 5
```

A single worker collects compatible requests,
including arrivals while the GPU is busy. GPU access is serialized; this is
request micro-batching rather than token-level continuous batching. The threaded
research server has no backpressure or production admission controls. To use
the older tabctx coalescer, install `pip install -e '.[batching]'` on Python
3.12 or newer and add `--batch-strategy tabctx`.

Queued stdlib serving uses persistent HTTP/1.1 connections and TCP_NODELAY.
An optional ASGI transport is available with `pip install -e '.[serving]'`
and `--http-backend asgi`. It was tested but did not beat stdlib overall in
the final workload; the command above is the recommended measured configuration.

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

## XGBoost CPU comparison

The [same-machine benchmark](https://www.kaggle.com/code/atomstack001/housing-xgboost-cpu-vs-tabpfn-t4)
compared cached TabPFN on a T4 with a 1,000-tree XGBoost model on four CPU
threads, using identical training data, feature construction, and unseen-ZIP
requests. Both HTTP services used the 80 ms queue. Results are averages of two
short localhost bursts, not sustained capacity guarantees:

| Concurrent clients | TabPFN requests/s | XGBoost requests/s |
| ---: | ---: | ---: |
| 1 | 1.36 | 7.42 |
| 32 | 39.74 | 191.37 |
| 128 | 99.85 | 484.17 |
| 256 | 94.30 | 415.96 |

At 128 clients, mean p95 HTTP latency was 1.25 s for TabPFN and 0.23 s for
XGBoost. For a single request, without waiting to batch XGBoost's median HTTP
latency was 0.055 s; TabPFN with an 80 ms window was 0.734 s. Model-only
timings for 1,704 prepared rows were 8.06 s and 0.0238 s respectively; these
exclude shared feature preparation and HTTP but include XGBoost encoding.

On the same 1,366 held-out ZIPs, RMSE against ZIP-average targets was 4.953
for TabPFN and 5.959 for this XGBoost configuration. This is one fold and is
not the competition's weighted metric. The benchmark did not measure monetary
cost. See `benchmarks/compare_xgboost.py` and
[`docs/benchmarks/xgboost_comparison.json`](benchmarks/xgboost_comparison.json).

## Inference optimization experiments

The [T4 attention profile](https://www.kaggle.com/code/atomstack001/housing-t4-attention-profile)
identified repeated K/V head copies and attention itself as the main GPU cost.
`--cache --optimize` keeps the checkpoint, all four estimators, full training
context and features. It folds shared-key query heads into the query sequence
and reuses the fitted input schema. Regional summaries are also fitted once.
The implementation changes only this fitted model, without patching TabPFN
module globals. It targets the pinned `tabpfn==9.2.0` v2 architecture.

In the profile, model-only single-row latency fell from 0.494 to 0.160 seconds
with identical predictions in the tested batches. The separate
[typed-input experiment](https://www.kaggle.com/code/atomstack001/housing-t4-cpu-gpu-overlap)
reduced a roughly 0.146-second path to 0.116 seconds and preserved the cleaned
features and predictions across all 1,704 test ZIPs. These are separate warmed
measurements, not a combined HTTP latency claim.

The [HTTP validation](https://www.kaggle.com/code/atomstack001/housing-t4-optimized-http)
of shared-key attention and cached regional summaries reduced median serial
request latency from about 0.667 to 0.192 seconds with a shorter queue window.
Both cached models scored 4.622991913 on 1,366 held-out June ZIPs, with zero
prediction difference. That run used three trials and four request waves at
each concurrent load, unlike the earlier single-burst measurements.

Experiments retained for reproduction, but excluded from the default model:

- Forced float16 failed with NaNs; float32 was much slower.
- A custom Triton attention prototype was slower than PyTorch. Eight tile
  configurations on representative shapes also failed to beat PyTorch; some
  exceeded T4 shared-memory limits. See `kernel_tuning.json`.
- Compiled blocks initially failed because CUDA graph outputs were overwritten.
  Cloning outputs fixed the exception, but the tested configuration remained
  slower and changed predictions slightly. See `compiled_blocks_safe.json`.
- Removing diagnostic GPU barriers gave no clear gain. It remains an experiment
  in `benchmarks/execution.py`.
- Combining compatible estimator caches helped single-row inference modestly,
  but slowed larger batches and changed some predictions; it is not enabled.
- A naive input-encoding shortcut mishandled missing ZIP prefixes. The corrected
  codec preserves the fitted cleaner's distinction between unknown and missing
  categories; tests cover this case explicitly.

Raw results and profiles are in [`docs/benchmarks/`](benchmarks/). No prediction
memoization or training-context subsampling is used by these optimizations.

## Final single-T4 comparison

This records the original fixed-configuration XGBoost baseline. The newer
[tuned comparison](#tuned-xgboost-comparison) below supersedes it for the README
infographic.

The [final Kaggle run](https://www.kaggle.com/code/atomstack001/housing-t4-final-inference-comparison)
used one Tesla T4 and an Intel Xeon CPU at 2.00 GHz, with four CPU threads.
It tested the shipped model wrapper and HTTP handlers, using aiohttp clients
on the same host. Each concurrency trial used four request waves; the serial
trial used 16 requests. Every reported value below averages three trials.
All requests target unseen ZIPs and run inference; no prediction memoization
or seen-ZIP shortcuts are included.

| Configuration | Single-client median latency | Requests/s at 128 clients | Requests/s at 256 clients |
| --- | ---: | ---: | ---: |
| Previous cached TabPFN, stdlib, 80 ms window | 669 ms | 100.1 | 110.8 |
| Optimized TabPFN, persistent stdlib, 5 ms window | **148 ms** | 136.6 | **156.2** |
| Optimized TabPFN, ASGI, 80 ms window | 282 ms | 149.6 | 151.2 |
| CPU XGBoost, ASGI, 0.1 ms window | 78 ms | 328.2 | 479.4 |
| CPU XGBoost, ASGI, 80 ms window | 159 ms | 625.4 | 730.8 |

The recommended TabPFN configuration reduced single-client latency about
**4.5x** and increased throughput at 256 clients about **41%**. XGBoost remains
faster. The CPU settings that maximize throughput add waiting time at low
traffic; they are not its lowest-latency settings. Both models use the same
raw engineered features; TabPFN also performs its fitted ensemble encoding.
These finite localhost tests are not maximum-capacity or sustained-load claims,
and they do not measure dollar cost.

The optimized TabPFN predictions were identical to the cached baseline across
all 1,704 unseen test ZIPs. Accuracy checks held out entire ZIPs and evaluated
June targets, separately for seeds 42 and 137:

| ZIP split seed | Rows | Cached baseline RMSE | Optimized RMSE | CPU XGBoost RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 1,366 | 4.622992 | 4.622992 | 5.736962 |
| 137 | 1,366 | 5.116784 | 5.116784 | 5.754148 |

The 32-row sample evaluated as individual one-row model calls also matched
exactly on each split. These are new-ZIP RMSE checks, not the competition's
weighted metric. Cached-vs-uncached differences described earlier still apply.
Install benchmark dependencies with `pip install -e '.[serving,benchmarks]'`
and reproduce with `python -m benchmarks.final_inference_comparison` after
preparing the data. Raw trials are saved in
[`final_inference_comparison.json`](benchmarks/final_inference_comparison.json).

## Tuned XGBoost comparison

After 30 XGBoost accuracy configurations, a new
[same-host run, version 2](https://www.kaggle.com/code/atomstack001/housing-tuned-xgboost-vs-tabpfn?scriptVersionId=365251494)
compared the selected 1,462-tree model with optimized four-estimator TabPFN.
Both use the full training data and the same raw engineered features. Hardware
remained one T4 used by TabPFN and a 2.00 GHz Xeon with four CPU threads.

| Configuration | Single-client median latency | Requests/s at 256 clients |
| --- | ---: | ---: |
| Optimized TabPFN, persistent stdlib, 5 ms window | 145 ms | 163.7 |
| Tuned CPU XGBoost, ASGI, 0.1 ms window | 77 ms | 461.5 |
| Tuned CPU XGBoost, ASGI, 20 ms window | 97 ms | 616.8 |
| Tuned CPU XGBoost, ASGI, 80 ms window | 157 ms | 700.6 |

Values average three trials; every timed request predicts one unseen ZIP.
The infographic uses XGBoost's lowest-latency and highest-throughput settings
separately. These finite localhost request waves exclude internet latency and
do not establish maximum or sustained capacity. Refitting the selected CPU
model reproduced all 8,534 submitted predictions within 1.5e-14.

Tuned XGBoost's reserved new-ZIP RMSE was **5.680814 / 5.544762** for seeds
42 / 137, versus TabPFN's **4.622992 / 5.116784**. Its public Kaggle submission
scored **5.225**, versus TabPFN's **3.919**. The public scores use external data;
TabPFN's public result is from the original uncached model. See
[accuracy methodology and submission details](model-comparison.md) and
[raw paired trials](benchmarks/tuned_xgboost_comparison.json).

Reproduce after preparing the data:

```bash
python -m benchmarks.tune_xgboost
python -m benchmarks.tuned_xgboost_comparison --selection xgboost_tuning.json
```

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
