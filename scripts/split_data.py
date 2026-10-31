"""Create local training, development, and validation sets from labeled data."""

import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
HOLDOUT_FRACTION = 0.20


def main():
    with (ROOT / "data/train.csv").open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        rows = list(reader)

    # Read ZIPs as strings to preserve leading zeros. Never use targets to split.
    zips = sorted({row["Zip"] for row in rows})
    rng = random.Random(SEED)
    held_out = set(rng.sample(zips, round(len(zips) * HOLDOUT_FRACTION)))
    seen = sorted(set(zips) - held_out)
    new = sorted(held_out)
    rng.shuffle(seen)
    rng.shuffle(new)
    dev_zips = set(seen[:len(seen) // 2] + new[:len(new) // 2])
    val_zips = set(zips) - dev_zips
    if {row["Month"] for row in rows} != {"5/1/2026", "6/1/2026"}:
        raise ValueError("Expected May and June 2026 observations only.")

    splits = {
        "train_local.csv": [
            row for row in rows
            if row["Month"] == "5/1/2026" and row["Zip"] not in held_out
        ],
        "dev_local.csv": [
            row for row in rows
            if row["Month"] == "6/1/2026" and row["Zip"] in dev_zips
        ],
        "val_local.csv": [
            row for row in rows
            if row["Month"] == "6/1/2026" and row["Zip"] in val_zips
        ],
    }

    train_zips = {row["Zip"] for row in splits["train_local.csv"]}
    actual_dev = {row["Zip"] for row in splits["dev_local.csv"]}
    actual_val = {row["Zip"] for row in splits["val_local.csv"]}
    if (train_zips != set(seen) or actual_dev != dev_zips
            or actual_val != val_zips or actual_dev & actual_val
            or train_zips & held_out):
        raise ValueError("ZIP overlap does not match the intended validation split.")
    for evaluation_zips in (actual_dev, actual_val):
        if not evaluation_zips & train_zips or not evaluation_zips & held_out:
            raise ValueError("Each evaluation set must contain seen and new ZIPs.")
    for name, subset in splits.items():
        keys = {(row["Zip"], row["State"], row["Month"]) for row in subset}
        if not subset or len(keys) != len(subset):
            raise ValueError(f"Empty split or duplicate observation keys: {name}")

    destination = ROOT / "data/splits"
    destination.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        path = destination / name
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(subset)
        temporary.replace(path)
        print(f"{path.relative_to(ROOT)}: {len(subset):,} rows")


if __name__ == "__main__":
    main()
