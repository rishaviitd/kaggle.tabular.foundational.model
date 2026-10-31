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

## What did not help enough

- CatBoost, extra trees, linear models, and a small neural network did not
  beat the best blend in the tested configurations.
- A separate owner/renter mixture model and similar-ZIP corrections did not
  produce the large improvement needed. Corrections helped slightly on dev.
- Fine-tuning TabICL did not improve on the selected approach.
- TabPFN could not be evaluated because downloading its weights required
  license acceptance. This is not a performance result.
- A Census coordinate experiment was discarded because competition rules
  prohibit external data. It is not part of the selected model.
- Google's TabFM has not been evaluated yet; its large checkpoint download
  was started. Its weights have a non-commercial license, so check
  compatibility with the competition rules before using it for a submission.

## Direction

Focus on **tabular foundation models** and ways to improve their predictions
for unseen ZIPs. Try leakage-safe regional summaries and better representations
of housing costs relative to income. Compare every idea on grouped ZIP splits;
do not use a ZIP's own target to create its features.
