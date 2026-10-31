"""Feature construction shared by local evaluation and submission."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "AffordabilityPercentageTrue"
CAT = ["State", "City", "Metro"]
ZIP_CAT = ["ZipPrefix1", "ZipPrefix2", "ZipPrefix3"]
NUM = [
    "SampleSize", "RentIndex", "AnnualMedianHouseholdIncome", "OwnerPercent",
    "RenterPercent", "MedianHomeLengthofResidence", "AvgNumberofChildren",
    "AvgNumberofAdults", "AvgHouseholdSize", "AvgGenerationsinHousehold",
    "YoungAdultinHouseholdPercent", "SeniorAdultinHouseholdPercent",
]


def features(data):
    x = data[NUM + CAT].copy()
    x["rent_to_income"] = x.RentIndex / (x.AnnualMedianHouseholdIncome / 12)
    x["income_per_person"] = x.AnnualMedianHouseholdIncome / x.AvgHouseholdSize
    x["rent_x_renter"] = x.RentIndex * x.RenterPercent / 100
    x["owner_renter_ratio"] = x.OwnerPercent / (x.RenterPercent + .1)
    x["log_samplesize"] = np.log1p(x.SampleSize)
    for column in CAT:
        x[column] = x[column].fillna("__MISSING__").astype(str)
    zip_code = data.Zip.astype(str).str.zfill(5)
    x["ZipInteger"] = pd.to_numeric(zip_code, errors="coerce")
    for length in (1, 2, 3):
        x[f"ZipPrefix{length}"] = zip_code.str[:length].astype(str)
    return x


def group_columns(data):
    zip_code = data.Zip.str.zfill(5)
    state = data.State.fillna("<missing>")
    city = data.City.fillna("<missing>")
    return pd.DataFrame({
        "zip3": zip_code.str[:3], "state": state,
        "state_city": state + "|" + city,
        "metro": data.Metro.fillna("<missing>"),
    }).reset_index(drop=True)


def encode_groups(reference, targets, query, alpha=10):
    reference = reference.reset_index(drop=True)
    query = query.reset_index(drop=True)
    output = pd.DataFrame(index=query.index)
    global_mean = targets.mean()
    for column in reference:
        grouped = pd.DataFrame({
            "key": reference[column].astype(str), "target": targets,
        }).groupby("key").target.agg(["sum", "count"])
        total = query[column].astype(str).map(grouped["sum"]).fillna(0)
        count = query[column].astype(str).map(grouped["count"]).fillna(0)
        output[column + "_mean"] = (total + alpha * global_mean) / (count + alpha)
        output[column + "_count"] = np.log1p(count)
    return output
