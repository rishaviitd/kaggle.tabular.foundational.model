> Historical experiment log: obsolete trial scripts were removed during repository cleanup. The retained evaluation and submission workflows are documented in [README.md](README.md) and [experiments/external_acs/README.md](experiments/external_acs/README.md).

# Experiments

Goal: weighted RMSE **≤ 3**. The score is 20% seen-ZIP RMSE and 80% new-ZIP
RMSE. See [docs/split.md](docs/split.md) for the May/June split.

| Approach | Result | Finding |
|---|---|---|
| Mean baseline | Dev weighted RMSE **22.49** | Uses May's target for seen ZIPs and the training mean for new ZIPs. |
| XGBoost + ZIP history | Dev weighted RMSE **10.27** | Large improvement over the mean baseline. |
| LightGBM + ZIP features | Dev **9.58**; validation **9.90** | Better than the tested XGBoost settings. |
| TabICL alone for new ZIPs | Dev new-ZIP RMSE **11.30** | Better than LightGBM's **11.67** on the same group; seen ZIPs were evaluated separately using May's target. |
| LightGBM + TabICL + regional summaries | Dev new-ZIP RMSE **10.72**; validation weighted RMSE **9.0364** | Best checked local approach; regional summaries are cross-fitted by ZIP. |
| Kaggle submission of that blend | Public score **9.711** | One submission made; target of 3 remains unmet. |

The best validation result is **1.2629 RMSE on seen ZIPs** and **10.9798 on
new ZIPs**. Validation scores were inspected during development; use a fresh
grouped split for an independent estimate.

## Additional grouped evaluation on Kaggle (2026-11-01)

To match the official unseen-ZIP setup, five-fold CV held out complete ZIP
groups, averaged May/June features and targets to one training row per ZIP,
and scored June targets for held-out ZIPs. Regional target encodings were
cross-fitted by ZIP. Kaggle used two Tesla T4 GPUs.

| Model | New-ZIP OOF RMSE | Finding |
|---|---:|---|
| LightGBM | 12.3293 | Same aggregated features and folds. |
| TabICL, 4 estimators | 11.4865 | Best single model; a modest but consistent improvement over LightGBM across the five folds. |
| 75% TabICL + 25% LightGBM | 11.3470 | Best OOF blend; blend weight was selected on these OOF predictions, so treat as a tuning estimate. |
| CatBoost | 12.4382 | Did not improve on TabICL. |

Google TabFM was also rerun with 8 estimators, a 256-row context, one row per
ZIP, and regional target encodings. It scored **13.147 RMSE** on one grouped
holdout, so this improved feature setup did not make TabFM competitive.

Using the measured May-to-June seen-ZIP persistence RMSE of **1.2524**, the
blend's approximate weighted score is **9.3281**. This is not a Kaggle
submission score. A single grouped split produced an overly optimistic 7.17
TabICL RMSE; five-fold CV corrected it to 11.49, so do not use the single-split
result. The private Kaggle run is [TabICL grouped ZIP evaluation](https://www.kaggle.com/code/atomstack001/tabicl-housing-grouped-zip-evaluation).

## Modeling-formulation checks (2026-11-01)

The target is highly stable for a ZIP across May and June (correlation
**0.9991**), so new-ZIP generalization is the main bottleneck. A row-level
five-fold ZIP holdout with TabICL scored **12.5958 RMSE** on held-out June
rows. Adding ZIP-safe regional target encodings reduced that to **11.8079**,
but remained behind the earlier aggregated TabICL result (**11.4865**; fold
construction differs, so this is directional rather than a strict paired
comparison). On a matched random five-fold ZIP split, row-level TabICL with
regional encodings scored **11.8079** and aggregated TabICL scored **11.7651**;
averaging the two months made little difference. Row-level LightGBM and
CatBoost scored **12.3667** and **13.2105** on the same random ZIP-fold setup.
Keeping each month as a separate row did not solve the unseen-ZIP gap.

A heteroskedastic LightGBM check weighted ZIP targets by `SampleSize**p`.
The unweighted score was **12.3959**; the best tested weight, `p=0.25`, was
**12.3166**, while stronger weights degraded to **12.40–12.46**. This is too
small and inconsistent to explain the leaderboard gap. A matched TabICL
feature ablation scored **11.7651** with the original categorical geography
features and **12.0637** after dropping City, Metro, and ZIP-prefix categories.
Even though TabICL warns about ordinal encoding high-cardinality categories,
removing them hurt; keep them alongside ZIP-safe regional encodings for now.

The earlier OOF predictions are compressed toward the center: prediction
standard deviation **24.7** versus target standard deviation **28.4**. In the
lowest target decile (mean target **2.0**), mean TabICL prediction was **7.8**;
in the highest decile (mean target **88.9**), it was **77.8**. A global affine
calibration fit on those same OOF predictions only reduced RMSE from **11.49**
to **11.40**, and is not an independent validation. On the matched random
ZIP folds, an arcsine-square-root target transform improved TabICL from
**11.7651** to **11.6111**; all five folds improved. Logit scored **11.7533**.
The approximate weighted score for arcsine is **9.5394**, not a submission
score. The [target-scale run](https://www.kaggle.com/code/atomstack001/tabicl-housing-target-scale-experiment)
completed as version 6 after feature-column ordering and notebook syntax fixes.

Exploratory local LightGBM tests scored May/June **mean targets**, unlike the
June-target evaluation above, so their absolute scores are not directly
comparable. Richer demographic and rent/income interactions improved **12.2501**
to **12.1477**. Cross-fitted numeric-bin/geographic target encodings worsened
this to **12.3878**. A reduced interaction set scored **12.1349** raw,
**12.1274** with arcsine, and **12.3013** with logit. These small changes do
not establish a solution to the larger generalization gap.

The competition rules prohibit outside datasets, APIs, synthetic data, and
proxy datasets for training, validation, and inference. Public pretrained
models are allowed only under the stated cost/license and target-training
conditions; see the [official rules](https://www.kaggle.com/competitions/predicting-housing-affordability-in-the-u-s/rules).

The leaderboard's best public score is **3.905**, which we have not
reproduced. The grouped CV still shows a large gap; the next work should focus
on new-ZIP signal and validation-aligned feature design, not just changing
foundation-model versions. No submission was made from these experiments.
The full-data TabICL/LightGBM candidate is saved locally at
`artifacts/submission_tabicl_agg.csv`; its 8,534 keys and 0–100 prediction
range were checked. It has not been submitted, so its leaderboard score is
unknown. Kaggle run: [TabICL aggregated ZIP competition predictions](https://www.kaggle.com/code/atomstack001/tabicl-aggregated-zip-competition-predictions).

## Alternative modeling: learned similarity and regional functions (2026-11-01)

National income/target correlation on June rows is **-0.201**, whereas the
correlation after subtracting metro means is **+0.221**. This is descriptive
evidence of regional differences, not a causal result. It motivated a Gaussian
process with a learned smooth numeric response and regional effects.

`scripts/similarity_gp.py` learns feature length scales, observation noise and
regional covariance weights using marginal likelihood on 1,000 training ZIPs
(seed 42), then conditions predictions on all fitting ZIPs. Preprocessing and
parameter optimization exclude evaluation labels. Seen ZIPs use May persistence.
The regional-functions variant also learns region-specific response functions,
instead of only intercept adjustments. No external data is used.

| Model | Dev new-ZIP RMSE | Dev weighted RMSE | Validation weighted RMSE |
|---|---:|---:|---:|
| Gaussian process, regional intercepts | 12.0897 | 9.9181 | Not run |
| Gaussian process, regional functions | 11.0245 | 9.0660 | 9.2024 |
| Existing TabICL/LightGBM model | 10.72 (previously recorded) | — | 9.0364 |
| Fixed equal blend of existing model and regional GP | — | — | 8.8977 |

The equal blend's validation new-ZIP RMSE is **10.8064**; its unseen-ZIP error
correlation between components is **0.902**. The blend was inspected on the
already-used validation set and is a tuning result, not fresh confirmation.
It did not meet the original **5–7 local weighted RMSE** submission gate. The
user subsequently requested submission of the current best local candidate.
The equal blend scored **9.775** on Kaggle's public leaderboard (submission
**57861510**, 2026-11-01), versus **9.711** for the previous ensemble: **0.064
worse**, despite the local improvement. Retain the previous ensemble as the
best public-scoring model; the GP blend is only the best observed local candidate.

`python scripts/submit_regional_blend.py` reproduces the submitted file from
the saved original `artifacts/submission.csv` ensemble and a regional GP fit
on all training ZIPs. It uses June targets for seen ZIPs and the equal blend
for 1,704 unseen ZIPs. All 8,534 submission keys, finite predictions and the
0–100 bounds were checked. Output: `artifacts/submission_regional_gp_blend.csv`.
The accompanying JSON records input/output SHA-256 hashes and the blend weight.

Reproduce with `python scripts/similarity_gp.py --regional-functions` for dev
and `python scripts/similarity_gp.py --evaluation val --regional-functions
--output artifacts/regional_gp_val.csv` for validation. The default output is
`artifacts/similarity_gp_dev.csv`; metric JSON files accompany predictions.

Two preliminary matched five-fold June-target tests also failed to provide a
large gain: fixed-scale RBF similarity with regional offsets scored **12.4593**
at its best tested setting versus LightGBM **12.3231**; training-fold regional
relative numeric features scored **12.2532**. Neither is promoted.

## What did not help enough

- CatBoost, extra trees, linear models, and a small neural network did not
  beat the best blend in the tested configurations.
- A separate owner/renter mixture model and similar-ZIP corrections did not
  produce the large improvement needed. Corrections helped slightly on dev.
- Fine-tuning TabICL did not improve on the selected approach.
- TabPFN v2's public regressor checkpoint was evaluated on Kaggle on the
  same five random ZIP folds and regional features as the TabICL target-scale
  experiment. Its unseen-ZIP RMSE was **11.6004**, versus TabICL raw-target
  **11.7651** and arcsine-target **11.6111**. The approximate weighted score
  is **9.5308** using May-to-June seen-ZIP persistence, not a leaderboard
  score. Fold RMSEs: **11.5490, 11.1995, 11.5936, 11.8572, 11.7911**.
  This is a modest improvement over raw TabICL and effectively matches the
  arcsine variant. A fixed equal blend with raw TabICL scored **11.4571**
  on the same pooled folds (weighted proxy **9.4162**), improving every fold;
  error correlation was **0.9243**. These are reused tuning folds, not an
  independent confirmation. Paired OOF predictions are saved under
  `artifacts/tabpfn_v2_grouped_oof.csv` and `artifacts/tabicl_matched_oof.csv`.
  See [TabPFN v2 grouped evaluation](https://www.kaggle.com/code/atomstack001/housing-tabpfn-v2-grouped-evaluation)
  and [matched TabICL OOF run](https://www.kaggle.com/code/atomstack001/housing-tabicl-matched-tabpfn-evaluation)
  and `scripts/tabpfn_grouped_cv.py`. Earlier attempts to access newer
  TabPFN weights required license acceptance and produced no model score.

  Paired repeated ZIP-fold comparison against the previous GP/ensemble blend:
  on seeds 42 and 137, the previous blend scored **9.1925 / 9.1855** weighted
  RMSE, while TabPFN + aggregated TabICL at equal weights scored **9.4162 /
  9.3971**. Thus the TabPFN/TabICL blend did not beat the previous best.
  An exploratory blend of 50% previous blend, 25% TabPFN and 25% aggregated
  TabICL scored **9.1572 / 9.1341** (mean **9.1457**, versus **9.1890**).
  This small gain was selected on these reused folds, so it is a tuning result
  that has not been independently confirmed. At the user's request, the
  full-training blend was submitted on 2026-11-03 and received public score
  **9.713** (submission **57919499**). Previous submissions scored **9.775**
  for the GP blend and **9.711** for the original ensemble; these small public
  differences are not sufficient to establish private-set ranking. The
  full-training TabPFN/TabICL run used all 13,660 labeled rows aggregated to
  6,830 ZIPs. Its 1,704 new-ZIP predictions were blended with 50% of the
  existing GP submission; the 6,830 seen ZIPs retained their June targets.
  Reproduce with `scripts/predict_aggregated_foundation.py` and
  `scripts/submit_foundation_blend.py`; output and SHA-256 metadata are in
  `artifacts/submission_foundation_blend.csv` and its accompanying JSON.
  The second-seed
  predictions come from [Kaggle's matched seed-137 run](https://www.kaggle.com/code/atomstack001/housing-tabpfn-comparison-seed-137).
- A Census coordinate experiment was discarded because competition rules
  prohibit external data. It is not part of the selected model.
- Google's TabFM ran on Kaggle's two T4 GPUs using the local training split.
  With 2 estimators, dev new-ZIP RMSE was **14.75**; with 16 estimators it
  was **13.92**. Both are worse than TabICL's **11.30** and the selected
  blend's **10.72** on new ZIPs, so TabFM did not improve the current model.
  The 16-estimator run took about 3 minutes. Use FP32: FP16 produced NaN
  predictions. Kaggle's MCP did not mount the competition data, so the run
  read the identical training CSV from this repository. The private runs are
  [2 estimators](https://www.kaggle.com/code/atomstack001/tabfm-housing-affordability-dev-evaluation-t4-fp32)
  and [16 estimators](https://www.kaggle.com/code/atomstack001/tabfm-housing-dev-t4-16-estimators).
  The pretrained weights have a non-commercial license; check compatibility
  with competition rules before using them in a submission.

## Direction

### External-data research (not eligible for this competition)

At the user's request, a separate research experiment used 2020–2024 ACS
tables B19001 (income distribution), B25077 (median home value), and B25106
(housing costs as a share of income). These external data are **prohibited
for competition training, validation, and inference**. The experiment and its
trained model must not be used to make a Kaggle submission. See
`experiments/external_acs/README.md` for sources and reproduction.

All 6,830 labeled ZIP identifiers and all 8,534 test identifiers matched a
Census ZCTA identifier; postal ZIP and ZCTA boundaries can still differ. On
two sets of five ZIP-grouped folds, the same LightGBM configuration scored:

| Features | Mean new-ZIP RMSE | Mean weighted proxy |
|---|---:|---:|
| Competition data only | 12.4563 | 10.2155 |
| Plus ACS income distribution and home value | 8.9718 | 7.4279 |
| Plus ACS housing-burden table | 9.0810 | 7.5153 |

An additional matched-fold test added Census B25091 mortgage/cost categories
and 2023 IRS ZIP income sources, separately and together, to the income/home
model. The mean weighted proxies were **7.4279** (existing), **7.4879**
(mortgage), **7.3767** (IRS), and **7.3967** (both). IRS matches 99.78% of
training and test ZIP/state pairs; mortgage ZCTA identifiers match all ZIPs.
The **0.0512** IRS improvement is small: a paired ZIP bootstrap interval for
IRS-minus-baseline was approximately **[-0.126, +0.020]**, excluding retraining
and selection uncertainty. Do not treat the improvement as established.
Sources, OOF predictions, metrics, and reproducible scripts are under
`artifacts/external_acs/` and `experiments/external_acs/`. This entire test
uses external data and is ineligible under the competition rules.

The two seed scores for income/home were **7.4623 / 7.3935**; housing-burden
features did not improve this model. This is a substantial controlled local
improvement from added information, not evidence that those features are
permitted or that 3.9 is achievable. The local research model was fitted on
all 13,660 training rows and saved under `artifacts/external_acs/`. At the
user's explicit request it was submitted on 2026-11-03 with its external-data
use disclosed in the submission description: public score **7.437**, submission
**57919880**. This result is **not eligible under the competition rules**.
June targets and five-year Census survey estimates differ in
time and source from the competition's July test targets.

Follow-up research downloaded ACS income-by-tenure (B25118), gross-rent
distribution (B25063), owner-cost distribution (B25087), and home-value
distribution (B25075). With the same LightGBM folds, home-value distribution
improved weighted RMSE from **7.4623 / 7.3935** to **7.1407 / 7.0119** on
seeds 42 / 137; adding all four distributions was weaker (**7.2622 / 7.1292**).
See `experiments/external_acs/evaluate_cost_distributions.py`.

On Kaggle GPUs, aggregated TabPFN v2 and TabICL were evaluated using ACS
income distribution and median home value. On seed 42 / 137, TabPFN scored
**6.5125 / 6.5211**, TabICL **6.6085 / 6.6445**, and their fixed 50/50 blend
**6.4519 / 6.4664** weighted proxy. The comparable row-level LightGBM baseline
scored **7.4623 / 7.3935**; model recipes and regional features differ, so the
gain cannot be attributed solely to architecture. A paired ZIP bootstrap of
blend-minus-baseline gave **[-1.092, -0.846]**, conditional on these fitted
models and excluding refitting/selection uncertainty.

Adding the B25075 home-value distribution to both foundation models gave
seed-42 / seed-137 weighted proxies **6.1349 / 6.1167** (TabPFN),
**6.4412 / 6.3999** (TabICL), and **6.1123 / 6.0911** (50/50 blend).
A 75/25 TabPFN/TabICL blend scored **6.0784 / 6.0610** (mean **6.0697**),
but its weight was selected on the reused seed-42 folds. Its paired-ZIP
bootstrap difference versus the original ACS LightGBM model was approximately
**[-1.509, -1.211]**, conditional on fitted models and excluding refitting and
model-selection uncertainty.

The 75/25 blend was refit on all **13,660** rows / **6,830** training ZIPs
and submitted with its ACS use disclosed. Kaggle submission **57920504**
scored **5.917** on the public leaderboard, versus **7.437** for the prior
ACS model. Only 30% of test data is public; final performance is unknown.
This submission uses external data and is not eligible under competition
rules. The submission and provenance are saved as
`artifacts/external_acs/submission_foundation_acs_value.csv` and its JSON.
The full-data run is [Housing Foundation Value Full Predictions](https://www.kaggle.com/code/atomstack001/housing-foundation-value-full-predictions).

The previous LightGBM Census model was also scored on held-out July features
using June labels as a proxy: **7.4623** using June features versus **7.4429**
using July features. This is not a true July score; it cannot establish the
effect of unknown July target drift. The held-out labels were excluded from
model and mapping fits. Code: `experiments/external_acs/evaluate_july_proxy.py`.

Validation audit after the second submission: the original submission refit
does use all 13,660 May/June training rows. The old local split matches the
80/20 seen/new row composition and 20/80 score weighting, but not the full
two-month fitting history. June also has 7.0% missing rents versus 0.105% in
July. `scripts/evaluate_geographic.py` now provides matched repeated five-fold
ZIP evaluation of the actual submission recipes (seeds 42 and 137), with
separate temporal persistence and rent-missingness diagnostics. Its LightGBM
baseline completed with weighted scores **10.1846** and **10.2465**; these are
baseline checks, not new scores for the ensemble or GP blend.

A paired bootstrap of the existing validation predictions, sampling roughly
public-leaderboard-sized seen/new groups, gave a 95% interval of **[-0.384,
+0.100]** for blend-minus-old weighted RMSE. This conditional resampling
exercise excludes fitting uncertainty and July shift; it nevertheless shows
that the small leaderboard reversal does not establish that the blend is
worse. Public rank is secondary to paired local evaluation. Previous local
validation has been reused, so all selection results remain tuning estimates.

Focus on **tabular foundation models** and ways to improve their predictions
for unseen ZIPs. Try leakage-safe regional summaries and better representations
of housing costs relative to income. Compare every idea on grouped ZIP splits;
do not use a ZIP's own target to create its features.

### Foundation ensemble and target tuning

On the same seed-42 five ZIP folds and ACS income/home-value features, increasing
TabPFN from 4 to 16 estimators improved its weighted proxy from 6.1349 to
**6.0875**; 4-estimator TabPFN with an arcsine-square-root target scored
**6.0918**. TabICL with 16 estimators scored **6.3997** and with the transformed
target **6.4114**, versus 6.4412 for the original 4-estimators. The old 75/25
TabPFN/TabICL blend scored **6.0784**. Replacing its TabPFN with 16-estimator
TabPFN scored **6.0509**; a 40/35/25 blend of 16-estimator TabPFN,
transformed-target TabPFN, and original TabICL scored **6.0369**. Blend weights
were chosen on these same folds, so this is an exploratory tuning result, not
an independent confirmation. These scores use the May-to-June persistence proxy
for seen ZIPs; July labels remain unavailable.

Run: [Kaggle tuning notebook](https://www.kaggle.com/code/atomstack001/housing-foundation-tuning-experiments).
Script: `experiments/external_acs/evaluate_foundation_tuning.py`.

### Foundation feature representations and additional signals

On the same five ZIP-held-out folds, the existing 4-estimator TabPFN and ACS
features scored **6.1349 / 6.1167** (seeds 42 / 137). Replacing the binned ACS
income and home-value shares with their cumulative distributions improved the
weighted proxy to **6.0039 / 5.9986**. Log-transforming monetary values scored
**6.1519 / 6.1508**; combining the log and cumulative representations scored
**6.0173 / 6.0106**. These are standalone models, with no stacking. The two
seeds partition the same ZIPs, so they are not independent test sets.

Separately adding ACS housing burden (B25106), income by tenure (B25118), or
rent distribution (B25063) to the original features scored **6.1667, 6.1695,
and 6.1572**, respectively, on seed 42, worse than the 6.1349 baseline.
No model was submitted. Scripts: `experiments/external_acs/evaluate_foundation_transforms.py`
and `experiments/external_acs/evaluate_foundation_signals.py`. Notebooks:
[transformations](https://www.kaggle.com/code/atomstack001/housing-foundation-transform-experiments),
[second split](https://www.kaggle.com/code/atomstack001/housing-foundation-transforms-seed137),
[additional signals](https://www.kaggle.com/code/atomstack001/housing-foundation-signal-experiments).

### Cumulative-feature stacked submission

A fixed 60% cumulative-feature TabPFN, 20% 16-estimator original-feature
TabPFN, and 20% original TabICL blend scored **5.9663** weighted proxy on the
reused seed-42 ZIP folds. This weight selection is exploratory; the 16-member
component was not evaluated on seed 137. All components were fitted using all
13,660 labeled rows (6,830 aggregated ZIPs), and seen ZIPs use June persistence.
Kaggle submission **57921346** scored **6.049** publicly, worse than the prior
**5.917**. The public score covers only part of the test set. The submission
description disclosed the ACS data. Code: `predict_stacked.py` and
`create_stacked_submission.py`; predictions are stored in `artifacts/external_acs/`.

### Zillow ZIP home values and joint Census tables

On the same seed-42 five ZIP folds, cumulative-feature TabPFN scored **6.0039**.
Adding Zillow's June 2026 mid-tier home-value history scored **4.6511**;
adding Zillow rents alone scored **6.0173**, and home values plus rents scored
**4.6777**. Adding a home-value-to-income ratio scored **4.6054**, and adding
income-tail features for different fractions of home value scored **4.6016**.
The simpler LightGBM model with price/income scored **5.0286**; its income-tail
version scored **5.0797**. These are local proxies, not Kaggle scores. The Zillow
history is a November 2026 snapshot and may have been revised after June 2026,
so the test is retrospective, not a verified real-time backtest. ACS and Zillow
ZIP identifiers do not necessarily correspond to identical geographic boundaries.

Separately, ACS B25121 (household income by home value), B25122 (household
income by gross rent), and both together scored **6.1669, 6.1213, 6.2321**,
worse than the 6.0039 baseline. Scripts `evaluate_joint_foundation.py`,
`evaluate_zillow_foundation.py`, `evaluate_affordability_mechanism.py`, and
`evaluate_mechanism_boosting.py` reproduce these tests. Data source URLs and
snapshot hashes are recorded in `artifacts/external_acs/sources_zillow.json`
and `sources_joint_distributions.json`. Results use all five held-out ZIP folds.

### Zillow tiers, Realtor listings, and matched XGBoost

Zillow low/middle/high home-value tiers with income-tail features scored
**4.1490 / 4.2381** on seed 42 / 137; the two folds reuse the same ZIPs.
Adding Zillow property-type values scored **4.1308** on seed 42. Adding
[Realtor.com Economic Research listing prices](https://www.realtor.com/research/data/)
to the Zillow tiers scored **4.1006** (all Realtor metrics: **4.1191**).
On the same seed-42 ZIP folds and same price features, XGBoost scored **6.0344**;
this is one XGBoost configuration, not a comprehensive tuning comparison.

The Realtor price model was fit on all **13,660** labeled rows, grouped to
6,830 ZIPs. Submission **57933746** scored **3.919** on the public leaderboard,
improving the prior best **5.917**. Its local proxy is still above the goal of
4; the public score covers only part of the test set. The external-data use
was disclosed in its submission description. Inputs are retrospective snapshots
and may have been revised after the dates represented. Scripts:
`evaluate_realtor_foundation.py`, `evaluate_realtor_xgboost.py`,
`predict_realtor.py`, and `create_zillow_submission.py`.

### Realtor foundation-model tuning

On the same five ZIP folds, three variations were tested on the Zillow/Realtor
feature set. Seed 42 / 137 weighted proxies were **4.0855 / 4.1913** for
16-estimator TabPFN, **4.1049 / 4.2429** for 4-estimator TabPFN with an
arcsine target, and **4.2429 / 4.2934** for 4-estimator TabICL. A 70/10/20
blend respectively scored **4.0685 / 4.1660**. Its weights were chosen on
these reused folds, so results are tuning estimates, not independent validation.
The original 4-estimator TabPFN scored **4.1006** on seed 42. Scripts:
`evaluate_realtor_tuning.py`, `predict_realtor_tuning.py`, and
`create_realtor_tuned_submission.py`.
The matched original baseline on seed 137 scored **4.2242**, so the stack
improved both reused splits versus baseline **4.1006 / 4.2242**. After refitting
all three models on all 13,660 labeled rows, Kaggle submission **57934513**
scored **4.012** publicly, versus **3.919** for the previous submission. The
submission uses external ACS, Zillow and Realtor.com Economic Research data,
which was disclosed in its description; the local gain is a tuning estimate.

### T4 inference optimization and CPU XGBoost comparison

The final same-host test used one Tesla T4, a 2.00 GHz Intel Xeon and four CPU
threads. Shared-key attention avoids repeating the training cache for query
heads. Fitted-schema input encoding avoids repeated dtype inference; regional
summaries are also cached. Persistent HTTP connections, TCP_NODELAY and a
5 ms request queue complete the selected stdlib serving configuration.

Across three trials, single-client median HTTP latency fell from **669 ms to
148 ms**. At 256 clients, throughput rose from **110.8 to 156.2 requests/s**.
The same-host XGBoost ASGI configurations measured **78 ms** at a 0.1 ms window
and **730.8 requests/s** at 256 clients with an 80 ms window. The latter is a
throughput setting, not its lowest-latency configuration. All concurrent trials
used four waves of unseen-ZIP requests; these are finite localhost benchmarks.

The optimized and baseline cached TabPFN predictions matched exactly on 1,704
test ZIPs and two 1,366-ZIP held-out June splits. RMSE remained **4.622992 /
5.116784** for seeds 42 / 137. Matched XGBoost scored **5.736962 / 5.754148**.
These are new-ZIP RMSE checks, distinct from the earlier weighted proxies.

What did not help: the tested Triton kernels were slower or exceeded shared
memory; compiled blocks were slower even after fixing CUDA graph output
reuse; removing diagnostic synchronization had no clear benefit. Pairing
compatible estimator caches improved single-row latency modestly but slowed
larger batches. None is enabled in the selected model. A naive encoding
shortcut mishandled missing ZIP prefixes; the shipped codec reproduces the
fitted cleaner's missing/unknown semantics and has explicit regression tests.

The direction remains tabular foundation models. The selected optimization
retains the four estimators, full training context and existing features.
Reproduce with `python -m benchmarks.final_inference_comparison`; see
[serving details](docs/serving.md#final-single-t4-comparison) and
[raw trials](docs/benchmarks/final_inference_comparison.json).

### Fair XGBoost accuracy tuning

The previous CPU baseline used one fixed configuration. We tested **30**
configurations with early stopping on two inner ZIP splits (seeds 2026/2027).
The 2,469 ZIPs in the union of the published seed-42/137 holdouts were excluded
from tuning entirely. The remaining 4,361 ZIPs supplied 3,488 training and 873
validation ZIPs per inner split. Regional target features were rebuilt within
each split. Selection minimized the mean inner new-ZIP RMSE.

Trials varied depth, learning rate, row/column sampling, child weight, L1/L2
regularization, ordinal versus native categorical handling, and raw/arcsine/logit
targets. The winner used ordinal categories and an arcsine-square-root target
transform, depth 5, learning rate 0.03, sampling 0.8, and L2 regularization 5.
Its two early-stop counts were 1,584 and 1,341; the frozen refit count was
**1,462 trees**. Native categories and logit targets did not improve selection
scores in this search. This is a bounded search, not a proven optimum.

| Reserved June ZIP split | Original XGBoost RMSE | Tuned XGBoost RMSE | Cached TabPFN RMSE |
| --- | ---: | ---: | ---: |
| Seed 42, 1,366 ZIPs | 5.736962 | **5.680814** | 4.622992 |
| Seed 137, 1,366 ZIPs | 5.754148 | **5.544762** | 5.116784 |

Tuning helped XGBoost on both reserved splits; TabPFN retained lower error.
After fitting the selected XGBoost on all **13,660 labeled rows / 6,830 ZIPs**,
submission **57945713** scored **5.225** publicly, versus TabPFN's **3.919**.
Both use the same external Census, Zillow and Realtor.com features. Their
submissions carry the same external-data eligibility limitation. The local
numbers above are new-ZIP RMSE, not the weighted public competition metric.

Reproduce tuning with `python -m benchmarks.tune_xgboost`. The
[completed run](https://www.kaggle.com/code/atomstack001/housing-xgboost-fair-tuning),
[all 30 trials](docs/benchmarks/xgboost_tuning.json), and
[verified submission metadata](docs/benchmarks/comparison_submissions.json)
record selection, full-data training counts and the submission SHA-256.

The final paired run remeasured both full-data models on the same T4/Xeon
host. Optimized TabPFN measured **145 ms / 163.7 requests/s**; tuned XGBoost
measured **77 ms / 700.6 requests/s**, using its 0.1 ms latency and 80 ms
throughput queue settings separately. Throughput uses 256 concurrent clients;
values average three finite localhost trials. XGBoost refit predictions matched
the submitted CSV within 1.5e-14. See the
[paired run](https://www.kaggle.com/code/atomstack001/housing-tuned-xgboost-vs-tabpfn?scriptVersionId=365251494)
and [raw results](docs/benchmarks/tuned_xgboost_comparison.json).

### TabPFN 3 regression, single estimator

Replaced only the checkpoint/estimator configuration in an isolated Kaggle
experiment: default v3 regression checkpoint, `n_estimators=1`, cached inference,
one T4, the same 106 features and the existing two held-out ZIP splits. A v2
single-estimator control separates the version change from ensemble removal.
No v2-specific attention or input-encoding patches were applied to v3.

| Model | Seed 42 RMSE | Seed 137 RMSE | Mean local RMSE |
| --- | ---: | ---: | ---: |
| TabPFN v3, one estimator | 5.220582 | 4.779488 | **5.000035** |
| TabPFN v2, one estimator | 4.696288 | 5.109243 | **4.902766** |
| Existing TabPFN v2, four estimators (previous run) | 4.622992 | 5.116784 | **4.869888** |

V3 won one split and lost the other; its mean error was higher. Removing the
ensemble did not improve mean accuracy. Keep the existing v2 default; do not
replace its public score or README comparison with these unsubmitted results.

The full-data v3 fit used all 13,660 labeled rows / 6,830 ZIPs. With the same
persistent HTTP backend and 5 ms queue, it measured **132.6 ms** single-client
latency and **390.6 requests/s** at 256 clients, averaged over three trials.
Predictions were processed in chunks of at most 128 rows. Earlier optimized
v2-four-estimator timings were 145.0 ms / 163.7 requests/s on the same hardware
class, from a separate run; this is not a same-session speed ablation.

Peak full-data v3 GPU memory was **3.86 GB allocated / 6.27 GB reserved** by
PyTorch. Reported GPU capacity was 15,636,037,632 bytes (T4, nominal 16 GB),
system RAM 33,658,155,008 bytes, and CPU allocation four logical CPUs on an
Intel Xeon at 2.00 GHz. The v3 checkpoint was 233,289,807 bytes. Allocated and
reserved GPU figures are not total device memory usage.

Reproduce with `python -m benchmarks.tabpfn_v3_single`. The
[completed Kaggle run](https://www.kaggle.com/code/atomstack001/housing-tabpfn-v3-single-estimator)
and [raw results](docs/benchmarks/tabpfn_v3_single.json) include local scores,
HTTP trials, hardware capacity and peak GPU memory. Saved held-out predictions
are alongside the report in `docs/benchmarks/`.

### Paired five-fold TabPFN v2 versus v3 accuracy

The follow-up matched the ensemble budget: **four estimators for both models**,
package `tabpfn==9.2.0`, model seed 42, cached inference, one T4 and no custom
v2 attention/input patches. Both models received identical training/query
feature tables per fold; each retained its version-native internal preprocessing.
This compares the released model pipelines, not architecture alone.

Five shuffled ZIP folds (seed 42) evaluated each of the 6,830 ZIPs exactly once
against its June target. All observations of a held-out ZIP were excluded from
training. The shared feature builder refitted regional summaries inside each
fold. No hyperparameters were selected using these results.

| Fold | v2 RMSE | v3 RMSE |
| --- | ---: | ---: |
| 0 | 4.622987 | 4.628972 |
| 1 | 4.465742 | 4.676954 |
| 2 | 4.891314 | 4.661558 |
| 3 | 4.951623 | 4.815942 |
| 4 | 5.209104 | 4.918001 |
| **Pooled out-of-fold RMSE** | **4.835145** | **4.741549** |

V3 reduced overall RMSE by **1.94%** and won three of five folds. This reverses
the earlier single-estimator comparison's mean result: the benefit depends on
the estimator configuration. The gain is modest, and one five-fold assignment
does not establish statistical significance. Neither model was submitted in
this experiment. These five-fold pooled scores are distinct from the README's
earlier two-split mean scores; do not compare them as though the evaluation
set were unchanged.

Reproduce with `python -m benchmarks.tabpfn_accuracy_cv`. The
[completed Kaggle run](https://www.kaggle.com/code/atomstack001/housing-tabpfn-paired-accuracy),
[raw report](docs/benchmarks/tabpfn_accuracy_cv.json), and
[paired out-of-fold predictions](docs/benchmarks/tabpfn_accuracy_cv_oof.csv)
record every fold and checkpoint hashes. Both pooled and fold-specific scores
were independently recomputed from the saved predictions, with ZIP identities,
fold assignments and target labels checked against the original training data.

### TabPFN v3 four-estimator submission

Refitted the evaluated v3 configuration on all **13,660 labeled rows / 6,830
ZIPs**, using the exact checkpoint SHA-256 from the paired accuracy run. The
8,534-row submission used latest-known targets for 6,830 seen ZIPs and v3
predictions for 1,704 unseen ZIPs. No models or predictions were blended.

Kaggle submission **57952579** scored **4.826**, versus the existing v2 best
of **3.919**. The 1.94% pooled local RMSE improvement did not translate into a
public leaderboard improvement. Keep v2 as the best public submission; these
results do not establish why the local/public rankings differ. Both submissions
use external ACS, Zillow and Realtor.com features.

Reproduce the full-data prediction file with
`python -m benchmarks.tabpfn_accuracy_cv --predict-test`. The
[completed prediction notebook](https://www.kaggle.com/code/atomstack001/housing-tabpfn-v3-submission)
and [verified submission record](docs/benchmarks/tabpfn_v3_submission.json)
record the training counts, checkpoint/submission hashes and public score.
