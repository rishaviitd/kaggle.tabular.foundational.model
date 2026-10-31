# Local train, dev, and validation split

Run `python scripts/split_data.py` from the repository root. Uses only
`data/train.csv`, with random seed 42 and a 20% ZIP holdout.

| Output in `data/splits/` | Month | ZIPs | Rows |
|---|---|---|---:|
| `train_local.csv` | May 2026 | Remaining 80% | 5,464 |
| `dev_local.csv` | June 2026 | 2,732 seen + 683 new | 3,415 |
| `val_local.csv` | June 2026 | 2,732 seen + 683 new | 3,415 |

The 1,366 May rows for held-out ZIPs are deliberately excluded. This keeps
these ZIPs completely unseen during training. June ZIPs are split equally
between dev and validation, separately within seen and new groups. Dev and
validation share no ZIPs. Original files, column names, values, and ZIP leading zeros are
preserved. The supplied July `data/test.csv` remains the competition test set.

Use dev for model selection and tuning; reserve validation for a final check.
Within either set, a ZIP is seen if it appears in `train_local.csv`; otherwise
it is new. Calculate `0.2 * RMSE_seen + 0.8 * RMSE_new` separately for each
set. The 80% score weight for new ZIPs is independent of their 20% row share.

During tuning, fit preprocessing and feature statistics on the local training split only.
Hold dev targets out of fitting when evaluating on dev. Never use validation
targets to fit the model or build features.
Once model selection is complete, retrain using all original May and June
training rows before predicting July.

For the local validation check, the chosen model may be refit using both
local train and dev rows. Dev and validation ZIPs are disjoint, so validation
ZIPs classified as new still have no labels in the refit data.

This is an approximation of the competition: only one earlier month is
available for local training, whereas the final model can train on two.
