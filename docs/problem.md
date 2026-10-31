## Can You Predict Where the American Dream Still Fits in the Budget?

Housing affordability is one of the defining economic challenges of our time, and it plays out differently block by block, ZIP code by ZIP code. In this competition, you'll build a machine learning model that predicts the percentage of households that can afford housing in a given ZIP code.

This isn't just an abstract regression problem. You'll be working with real geographic, demographic, and economic data, untangling how income, housing costs, and location interact to shape opportunity across the country. Your model could help illuminate patterns that matter to policymakers, urban planners, and everyday families trying to figure out where they can actually afford to live.

Sharpen your feature engineering, test your modeling instincts, and see how your predictions stack up: all while tackling a problem that's as real-world as it gets.

**Have questions about the competition?** Join Melissa's linked Discord server to ask questions about the dataset, rules, submissions, or other competition details under the "Education" tab. You can also use the server to connect with other participants and stay updated throughout the competition.

[Join the Melissa Community Discord Server!](https://discord.gg/92bHmUSCBU)

Note: The Discord server is an informal discussion channel. Nothing posted there modifies the Official Competition Rules, which govern in the event of any conflict.

## Background

Housing affordability affects more than where people live. When a large portion of a household's income goes toward rent, mortgage payments, utilities, and other housing expenses, less money is available for necessities such as food, transportation, healthcare, and education.

This has become a widespread issue across the United States. In 2024, an estimated 43.5 million households, about one-third of all U.S. households, spent more than 30% of their income on housing costs. Nearly half of all renter households were considered housing cost burdened. These challenges affect both renters and homeowners and can look very different depending on the area (Harvard Joint Center for Housing Studies, 2026).

Housing affordability is influenced by many connected factors, including income, rent, home values, household size, age, location, and whether a household rents or owns its home. Because these factors vary across communities, machine learning can help identify patterns that may be difficult to see through individual variables alone.

## The Challenge

In this competition, participants will build a machine-learning model that predicts the percentage of households that can afford housing in each ZIP code.

The dataset provides ZIP-code-level information related to:

- State, city, and metro area
- Rent
- Income
- Population and age
- Owners and renters
- Household size and composition
- Length of residence


Your goal is to use these features to generate accurate predictions for ZIP codes in the hidden test set. Beyond the technical challenge, this competition gives you the chance to dig into a real-world issue where geography, demographics, and economics all collide.

> **A note on interpretation:** This dataset is designed to help identify patterns and trends, not to establish causation. A strong relationship between a feature and housing affordability doesn't necessarily mean that feature causes affordability to rise or fall.

## Goal

The goal is to predict:

**`AffordabilityPercentageTrue`**

for each ZIP code in the test dataset. This column represents the percentage of households in a ZIP code that can afford housing, meaning they are not cost-burdened.

Because the target is a continuous numeric value, this is a **regression competition**.

## Unit of Observation

Each row in the dataset represents one five-digit ZIP code.

The dataset includes a `ZipCode` column that serves as the unique identifier for each row. ZIP codes are stored as text (not numeric) so that codes beginning with a zero, like `02134`, retain all five digits rather than being truncated to four.

# Evaluation Metric

Submissions will be evaluated using [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root_mean_square_deviation), or RMSE.

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

Where:
- *y<sub>i</sub>* is the actual affordability percentage.
- *ŷ<sub>i</sub>* is the predicted affordability percentage.
- *n* is the number of ZIP codes being evaluated.

A lower RMSE score is better. RMSE measures how far the predictions are from the real values, and gives a larger penalty to predictions that are very far off.

## Seen ZIP Score

RMSE is first calculated for July observations belonging to Seen ZIPs.

$$
RMSE_{Seen}
$$

This measures how well the model predicts a future month for ZIP codes that appeared in the May or June training data.

## New ZIP Score

RMSE is also calculated separately for July observations belonging to New ZIPs.

$$
RMSE_{New}
$$

This measures how well the model predicts affordability for ZIP codes that were completely excluded from the May and June training data.

## Final Competition Score

The final competition score gives more weight to new ZIPs:

$$
Final\ Score = 0.2(RMSE_{Seen}) + 0.8(RMSE_{New})
$$

For example, suppose a model receives:
- Seen ZIP RMSE = 2.50
- New ZIP RMSE = 3.50

Its final score would be:

$$
0.2(2.50) + 0.8(3.50) = 3.3
$$

This means a successful model must perform well both on ZIP codes it has previously encountered and on completely new ZIP codes.

# Leaderboards

The test dataset will be split into a public leaderboard and a private leaderboard.

- The public leaderboard will use approximately 30% of the test data.
- The private leaderboard will use the remaining 70%.

The public leaderboard will show participants how their submissions are performing during the competition. Final rankings will be based on the private leaderboard.

# Submission File

For every `ZipCode` in the test dataset, participants must submit a prediction for `AffordabilityPercentageTrue`.

The file should contain a header and have the following format:

ZipCode,State,AffordabilityPercentageTrue  
90210,CA,27  
92101,CA,42  
10001,NY,96  
etc.  

# Submission requirements:

- The file must be a CSV.
- The column names must match the sample submission.
- Every ZIP code + State pairing in the test dataset must appear exactly once.
- The file should not include any additional ZIP codes.
- ZIP codes must remain five digits.
- Predictions must be numeric.
- Predictions should be between 0 and 100.
- Missing or infinite values are not allowed.
- Participants should use sample_submission.csv  as the format for their submissions.

The winning team will receive a **$225 Amazon gift card** + **Melissa swag** (backpack with hat, notebook, and water bottle inside)

*Note: Entry is open as provided in the Foundational Rules. Only legal residents of the United States who are 18 years of age or older are eligible to win or claim prizes.*

## Geographic Columns
Some states, cities metro areas, or ZIP codes may appear more often in the dataset than others.

Participants should not assume that every geographic area is represented equally.

The data should be treated as a sample of the areas and households represented in the dataset and may not perfectly represent the full population of each ZIP code or geographic area.

Patterns found in the data should therefore be interpreted as trends or associations and should not automatically be treated as evidence that one feature causes another.

## Monthly Observations
The same ZIP code may appear more than once in the training data because May and June are separate observations.
Participants should take this into account when validating their models. Randomly splitting individual rows may place the same ZIP code in both the training and validation sets and may make validation performance appear better than performance on genuinely new ZIP codes.

## Household Counts
`SampleSize` shows how many households are represented in the ZIP code.
Percentages based on a small number of households may be less stable than percentages based on a larger number of households.

## Data Source and Interpretation
The dataset should be treated as a sample and may not fully represent every household or geographic area.
The values are intended to show general patterns and trends, but they may contain sampling limitations, missing information, or other inaccuracies.
Participants should not assume that relationships between variables prove causation. A feature may be associated with housing affordability without directly causing it.

## Data Leakage
All predictor columns should be created separately from the hidden test target.
Participants may not use:
- Hidden affordability values
- Another dataset containing the test answers
- Repeated leaderboard submissions to try to reconstruct the target
- Information created directly from the hidden labels


Clark, J. (2022, September 19). Methodology: Zillow Observed Rent Index (ZORI). Zillow Research. https://www.zillow.com/research/methodology-zori-repeat-rent-27092/  
Melissa Data Corporation. (2026). Data sets: Analytics and reference data. Retrieved August 3, 2026, from https://www.melissa.com/data-sets  
Melissa Data Corporation. (2026). Consumer data documentation. Retrieved August 3, 2026, from https://docs.melissa.com/reference-data/consumer-data/consumer-data-index.html  
Whitney, P. (2026, February 4). Housing Unaffordability Soared to New Highs in 2024. Retrieved August 3, 2026, from https://www.jchs.harvard.edu/blog/housing-unaffordability-soared-new-highs-2024  
Zillow Research. (n.d.). Housing data: Zillow Observed Rent Index. Retrieved August 3, 2026, from https://www.zillow.com/research/data/  
