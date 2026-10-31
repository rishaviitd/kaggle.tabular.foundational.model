# Overview

The dataset contains housing, income, demographic, household, and geographic information summarized by ZIP code.

This competition uses data from May, June, and July of 2026. The goal is to predict:

`AffordabilityPercentageTrue`

The training data contains observations from May and June, while the test data contains observations from July.

This setup is designed to test two different types of generalization:

- Time-based generalization: Can a model use patterns from May and June to predict affordability in July?
- Geographic generalization: Can a model make accurate predictions for ZIP codes it has never seen before?

The training and test datasets contain the same predictor columns. The training dataset additionally contains the target column:

`AffordabilityPercentageTrue`

A month column is included so that participants can identify whether a training observation comes from May or June.

# Training and Test Split

Before the training dataset is created, a portion of ZIP codes is randomly selected as the geographic holdout group.

These ZIP codes are completely removed from both the May and June training data. This means the model never sees any training observation from these ZIP codes.

The remaining ZIP codes are eligible to appear in the May and June training data.

The July test set contains two types of ZIP codes:

## Seen ZIPs

These are ZIP codes that actually appeared in the May or June training data.

For these ZIP codes, the model has information from an earlier month but must predict the July affordability value.

This tests how well the model generalizes forward in time.

## New ZIPs

These are ZIP codes selected for the geographic holdout.

They were completely excluded from both May and June, so the model has never seen an observation from these ZIP codes during training.

This tests how well the relationships learned by the model transfer to new geographic areas.

# Why the Split Is Designed This Way

A model may perform well in July simply because housing characteristics within the same ZIP code tend to remain similar from month to month.

For example, if a model has already seen ZIP code 92122 in May and June, predicting its July value may be easier than predicting a ZIP code it has never encountered.

By evaluating Seen ZIPs and New ZIPs separately, we can better determine whether a model is learning relationships between the housing and demographic features and affordability, or whether it is relying heavily on ZIP-specific patterns that persist over time.


# Files

| File | Description |
|---|---|
| `train.csv` | May and June training observations, including `AffordabilityPercentageTrue` |
| `test.csv` | July observations used for the competition test set |
| `sample_submission.csv` | A sample submission file in the correct format |

# Columns

| Column | Description |
|---|---|
| `Zip` | Five-digit ZIP code represented by the row. The identifier of this dataset. |
| `YearMonth` | The year and month that this row's data is from. |
| `SampleSize` | Number of unique households represented in the ZIP code. |
| `State` | Two-letter state abbreviation. |
| `City` | Main city connected to the ZIP code. |
| `Metro` | Metropolitan area connected to the ZIP code. |
| `RentIndex` | Zillow Observed Rent Index for the ZIP code — the typical monthly asking rent in the area, in USD. Data sourced from https://www.zillow.com/research/data/|
|`AnnualMedianHouseholdIncome` | Median annual income of the households in the ZIP code. |
| `OwnerPercent` | Percentage of households that are known owners. |
| `RenterPercent` | Percentage of households that are known renters. |
| `MedianHomeLengthofResidence` | Median number of years households have lived at their current residence. |
| `AvgNumberofChildren` | Average number of children per household. |
| `AvgNumberofAdults` | Average number of adults per household. |
| `AvgHouseholdSize` | Average number of people living in each household. |
| `AvgGenerationsinHousehold` | Average number of generations living in each household. |
| `YoungAdultinHouseholdPercent` | Percentage of households with at least one young adult. |
| `SeniorAdultinHouseholdPercent` | Percentage of households with at least one senior adult. |
| `AffordabilityPercentageTrue` | Percentage of households that spend 30% or less of their income on housing costs. **The target variable of this dataset.** |

# About the Dataset

This competition dataset was created by combining and summarizing information from multiple data sources at the ZIP-code level. Melissa Data was used as the primary source for household, demographic, address, and geographic information. Additional housing-market information was added from publicly available sources, including Zillow Research.

The original source records were cleaned, grouped, and aggregated to create one row per five-digit ZIP code. The competition files are therefore derived datasets and may not exactly match the format or values shown in the original sources.

Because the data represents a sample of available household and housing information, it may not fully represent every household or geographic area. The dataset should be used to study general patterns and associations rather than to make conclusions about individual households or to assume that a relationship proves causation.

## Geographic Unit

- Each row represents one five-digit ZIP code.
- ZIP codes are stored as text so leading zeros are not removed.
- Some ZIP codes may cross more than one city or county. When this happens, one main city or county is assigned using the same method throughout the dataset.

## Target Calculation

For each household with valid income and housing-cost information:

$$
\text{HousingCostRatio} = \frac{\text{Monthly housing costs}}{\text{Monthly median household income}}
$$

A household is considered housing cost burdened when:

$$
\text{HousingCostRatio} > 0.30
$$

The ZIP-code-level target is then calculated as:

$$
\text{AffordabilityPercentageTrue} = \frac{\text{Non cost-burdened households}}{\text{Households with a valid affordability result}} \times 100
$$

## Missing Values

A missing value means that the information was not available or could not be calculated reliably.

A missing value should **not** automatically be interpreted as:

- Zero
- None
- Not applicable

Participants may decide how to handle missing values in their models.

## Averages, Medians, and Percentages

### Averages

Average columns are calculated using only households with a valid value for that variable. Households with missing or unknown values are excluded.

$$
\text{Average} = \frac{\text{Sum of valid household values}}{\text{Number of households with a valid value}}
$$




### Medians

Median columns are also calculated using only valid values. The valid values are sorted from lowest to highest, and the middle value is selected.


### Percentages

Unless otherwise stated, percentage columns are calculated as:

$$
\text{Percentage} = \frac{\text{Number of households that meet the condition}}{\text{Number of households with a known value}} \times 100
$$

Households with missing or unknown values for that specific variable are excluded from the denominator.


> **Note:** The denominator may differ between percentage columns because each variable may have a different number of missing values. Percentages may also not add up to exactly 100% because of rounding.
