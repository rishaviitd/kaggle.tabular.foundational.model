# Accuracy and inference speed

![Tuned XGBoost CPU versus optimized TabPFN on one T4](assets/model-tradeoff.png)

Tuned XGBoost serves requests faster on CPU. TabPFN retains lower prediction
error on both matched local splits and on the public Kaggle leaderboard.

| Measurement | Tuned XGBoost CPU | Optimized TabPFN v2, one T4 |
| --- | ---: | ---: |
| Single-client HTTP latency | 77 ms | 145 ms |
| HTTP throughput, 256 concurrent clients | 701 requests/s | 164 requests/s |
| Saved model / base checkpoint size | 3.94 MB | 44.39 MB |
| Local unseen-ZIP RMSE, seed 42 | 5.681 | 4.623 |
| Local unseen-ZIP RMSE, seed 137 | 5.545 | 5.117 |
| Public Kaggle score* | 5.225 | 3.919 |

Lower latency and error are better; higher throughput is better. The local
RMSE checks are distinct from the competition's weighted metric. Each local
split holds out entire ZIPs and evaluates 1,366 June targets. Both models use
the same 106 raw engineered features, including external sources. TabPFN also
performs its fitted ensemble preprocessing.

File sizes use decimal MB: the saved XGBoost UBJSON model is **3,938,691 bytes**;
the shared TabPFN base checkpoint is **44,390,977 bytes**. Runtime RAM/VRAM,
preprocessing state, and TabPFN's training context/cache are additional. Four
TabPFN estimators do not mean four separate checkpoint downloads. The model is
**TabPFN v2**, loaded from `Prior-Labs/TabPFN-v2-reg/tabpfn-v2-regressor.ckpt`
using Python package **tabpfn 9.2.0**. XGBoost uses **3.4.1**.

## How XGBoost was tuned

We tested 30 configurations using two inner ZIP splits, with early stopping
at a maximum of 2,500 trees. The 2,469 ZIPs in the union of the seed-42/137
comparison holdouts were excluded from this search. The remaining 4,361 ZIPs
supplied 3,488 training and 873 validation ZIPs per inner split. Regional target
summaries were rebuilt within each split. Selection minimized mean inner RMSE;
comparison holdouts were evaluated after the configuration was frozen.

The search varied depth, learning rate, subsampling, column sampling, child
weight, L1/L2 regularization, categorical handling and target transformation.
Ordinal categories with an arcsine-square-root target won. Native categorical
handling and logit targets did not help in this search. The selected model
uses depth 5, learning rate 0.03, row/column sampling 0.8, L2 regularization 5,
and **1,462 trees**, fixed from the median inner early-stop count. This is a
bounded search, not a proven optimum.

| Reserved split | Original XGBoost RMSE | Tuned XGBoost RMSE |
| --- | ---: | ---: |
| Seed 42 | 5.736962 | 5.680814 |
| Seed 137 | 5.754148 | 5.544762 |

The [tuning run](https://www.kaggle.com/code/atomstack001/housing-xgboost-fair-tuning)
records [all 30 trials](benchmarks/xgboost_tuning.json). Saved held-out predictions
for [seed 42](benchmarks/xgboost_seed42_heldout.csv) and
[seed 137](benchmarks/xgboost_seed137_heldout.csv) reproduce the XGBoost scores.
The TabPFN local scores
come from the earlier [matched validation](benchmarks/final_inference_comparison.json);
these ZIPs were reserved from the new XGBoost search, not from every historical
TabPFN experiment.

## What was measured

The [completed paired benchmark, version 2](https://www.kaggle.com/code/atomstack001/housing-tuned-xgboost-vs-tabpfn?scriptVersionId=365251494)
ran both models on the same Kaggle host: an Intel Xeon at 2.00 GHz with four
CPU threads and one Tesla T4 used by TabPFN. XGBoost explicitly runs on CPU.
Values average three trials; latency is the mean of the single-client trial
medians. Each request predicts one unseen ZIP. Clients run on localhost,
excluding internet round trips. No prediction memoization or seen-ZIP shortcuts
are included in the speed tests.

TabPFN uses four estimators, the full 6,830-ZIP context, cached inference,
the optimized attention/input path, and persistent HTTP with a 5 ms batching
window. XGBoost uses the tuned model described above with ASGI serving. Its
latency number uses a 0.1 ms batching window; its throughput number uses an
80 ms window. These are different CPU configurations, not simultaneous latency
and throughput guarantees. At the same 0.1 ms window, XGBoost measured
461 requests/s at 256 clients; the 80 ms setting had 157 ms single-client latency.

All speed measurements use the full-data fitted models. The refitted XGBoost
server reproduced all 8,534 submitted predictions to within **1.5e-14**. The
tests use finite request waves and do not establish maximum capacity,
sustained production performance or dollar cost. See the
[raw paired trials](benchmarks/tuned_xgboost_comparison.json) and
[serving details](serving.md#tuned-xgboost-comparison).

## Public submissions and tradeoffs

Both models were fitted using all **13,660 labeled rows**, aggregated into
**6,830 training ZIPs**. Submission files contain 6,830 seen ZIPs, predicted
from their latest known target, and 1,704 unseen ZIPs, predicted by the fitted
model. XGBoost submission **57945713** scored **5.225**; TabPFN submission
**57933746** scored **3.919**. These actual public scores are recorded in
[verified submission metadata](benchmarks/comparison_submissions.json).

XGBoost offers lower latency, higher throughput and CPU deployment. TabPFN
offers lower error in these tests, while using a GPU in the measured setup.

*Both submissions use external Census, Zillow and Realtor.com features and
are not eligible under the competition's external-data rules. The public
leaderboard covers only part of the test set. TabPFN's public score uses the
original uncached model: caching introduces small numerical differences, so
3.919 is not a separate leaderboard validation of the optimized server.

## Reproduce

Prepare the competition and external data as described in [serving.md](serving.md),
install benchmark dependencies with `pip install -e '.[serving,benchmarks]'`, then run:

```bash
python -m benchmarks.tune_xgboost
python -m benchmarks.tuned_xgboost_comparison --selection xgboost_tuning.json
```

The first command writes the full-data submission and selection report in the
repository root. The second verifies the refit against that submission and
benchmarks both models on the same CUDA host. The PNG was edited with built-in
image generation; its exact [final prompt](assets/model-tradeoff.prompt.md)
is saved alongside it.
