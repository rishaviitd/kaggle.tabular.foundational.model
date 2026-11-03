"""Grouped held-out permutation importance for prepared model inputs."""
import numpy as np
import pandas as pd


def feature_groups(columns):
    """Keep correlated encodings together; scores describe reliance, not causality."""
    cols = list(columns)
    groups = {
        'Zillow home values': [c for c in cols if c.startswith('ZILLOW_') or c == 'PRICE_TO_MEDIAN_INCOME'],
        'ACS income distribution and income thresholds': [c for c in cols if c.startswith('ACS_B19001_') or c.startswith('INCOME_TAIL_')],
        'ACS home value distribution': [c for c in cols if c.startswith('ACS_VALUE_DIST_') or c.startswith('ACS_B25077_')],
        'Realtor listing prices': [c for c in cols if c.startswith('REALTOR_')],
        'Regional target summaries': [c for c in cols if c.startswith('Regional')],
        'Location': [c for c in ['State', 'City', 'Metro', 'ZipNumeric', 'ZipPrefix1', 'ZipPrefix2', 'ZipPrefix3'] if c in cols],
        'Income and rent': [c for c in ['AnnualMedianHouseholdIncome', 'RentIndex', 'RentToIncome', 'IncomeToRequired', 'IncomePerPerson', 'RentXrenter'] if c in cols],
    }
    assigned = {c for group in groups.values() for c in group}
    groups['Other demographic features'] = [c for c in cols if c not in assigned]
    return {name: group for name, group in groups.items() if group}


def grouped_permutation_importance(predict, x, y, repeats=3, seed=42):
    """Permute whole feature groups among held-out rows, reporting RMSE change."""
    y = np.asarray(y, dtype=float)
    if len(y) != len(x) or not len(x):
        raise ValueError('x and y must contain the same nonzero number of rows')
    rmse = lambda p: float(np.mean((y - p)**2)**.5)
    baseline = rmse(predict(x))
    rng = np.random.default_rng(seed)
    rows = []
    for group, columns in feature_groups(x.columns).items():
        changes = []
        for _ in range(repeats):
            order = rng.permutation(len(x))
            shuffled = x.copy()
            for column in columns:
                shuffled[column] = x[column].iloc[order].reset_index(drop=True).array
            changes.append(rmse(predict(shuffled)) - baseline)
        rows.append({'group': group, 'columns': len(columns), 'baseline_rmse': baseline,
                     'rmse_increase_mean': float(np.mean(changes)), 'rmse_increase_std': float(np.std(changes))})
    return pd.DataFrame(rows).sort_values('rmse_increase_mean', ascending=False).reset_index(drop=True)


def local_group_sensitivity(predict, row, background, samples=16, seed=42):
    """Describe prediction changes after replacing groups with background values.

    This is a diagnostic, not an additive explanation or a causal effect;
    replacement can create unrealistic combinations of correlated variables.
    """
    if len(row) != 1 or not len(background) or samples < 1:
        raise ValueError('Provide one row and a nonempty background')
    if not row.columns.equals(background.columns):
        raise ValueError('Feature columns must match')
    rng = np.random.default_rng(seed)
    baseline = float(predict(row)[0])
    rows = []
    for name, columns in feature_groups(row.columns).items():
        alternatives = pd.concat([row] * samples, ignore_index=True)
        order = rng.integers(0, len(background), size=samples)
        for col in columns:
            alternatives[col] = background[col].iloc[order].reset_index(drop=True).array
        delta = np.asarray(predict(alternatives)) - baseline
        rows.append({'group': name, 'baseline_prediction': baseline,
                     'mean_change': float(delta.mean()), 'mean_absolute_change': float(np.abs(delta).mean()),
                     'min_change': float(delta.min()), 'max_change': float(delta.max())})
    return pd.DataFrame(rows).sort_values('mean_absolute_change', ascending=False).reset_index(drop=True)
