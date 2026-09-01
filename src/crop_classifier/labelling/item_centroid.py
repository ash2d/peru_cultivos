"""Look up a labelling-campaign item's centroid: item_id -> COD_PREDIO -> lat/lon."""

from __future__ import annotations

import sys

import pandas as pd

from crop_classifier.paths import labels_dir


def centroid(item_id: str) -> tuple[float, float]:
    key = pd.read_csv(labels_dir() / "html" / "item_key.csv")
    match = key.loc[key["item_id"] == item_id, "COD_PREDIO"]
    if match.empty:
        raise KeyError(f"{item_id!r} not found in {labels_dir() / 'html' / 'item_key.csv'}")
    cod_predio = match.iloc[0]

    sample = pd.read_parquet(labels_dir() / "label_sample.parquet",
                              columns=["COD_PREDIO", "centroid_lat", "centroid_lon"])
    row = sample.loc[sample["COD_PREDIO"] == cod_predio]
    if row.empty:
        raise KeyError(f"{cod_predio!r} not found in label_sample.parquet")
    return row["centroid_lat"].iloc[0], row["centroid_lon"].iloc[0]


if __name__ == "__main__":
    item_id = sys.argv[1] if len(sys.argv) > 1 else input("item_id: ").strip()
    lat, lon = centroid(item_id)
    print(f"{item_id}: lat, lon = {lat}, {lon}")
