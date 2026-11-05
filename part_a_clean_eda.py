"""Clean raw shipments and produce Part A evidence tables."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def main():
    OUTPUTS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA / "raw_shipments.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")
    deduped = raw.drop_duplicates().copy()
    service_spellings_before = int(deduped["service_type"].nunique())
    sentinel_rows = int((deduped["parcel_weight_kg"] > 100).sum())

    clean, medians = clean_shipments(raw)
    service_spellings_after = int(clean["service_type"].nunique())
    model_table = build_features(clean, hubs)
    model_table["is_delayed"] = (model_table["delivery_hours"] > model_table["promised_hours"]).astype(int)
    model_table.to_csv(DATA / "modelling_table.csv", index=False)

    hours_q1 = model_table["delivery_hours"].quantile(0.25)
    hours_q3 = model_table["delivery_hours"].quantile(0.75)
    iqr = hours_q3 - hours_q1
    checks = [
        ("raw_rows", len(raw)),
        ("duplicate_rows_removed", len(raw) - len(deduped)),
        ("rows_after_dedup", len(deduped)),
        ("missing_distance_km_before", int(deduped["distance_km"].isna().sum())),
        ("missing_parcel_weight_before", int(deduped["parcel_weight_kg"].isna().sum())),
        ("sentinel_weight_rows", sentinel_rows),
        ("missing_declared_value_before", int(deduped["declared_value"].isna().sum())),
        ("missing_payment_mode_before", int(deduped["payment_mode"].isna().sum())),
        ("service_type_spellings_before", service_spellings_before),
        ("service_type_spellings_after", service_spellings_after),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(model_table)),
        ("nulls_in_features_and_target", int(model_table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
        ("delayed_count", int(model_table["is_delayed"].sum())),
        ("delay_rate", model_table["is_delayed"].mean()),
        ("hours_q1", hours_q1),
        ("hours_q3", hours_q3),
        ("hours_lower_bound", hours_q1 - 1.5 * iqr),
        ("hours_upper_bound", hours_q3 + 1.5 * iqr),
        ("hours_outlier_count", int(((model_table["delivery_hours"] < hours_q1 - 1.5 * iqr) | (model_table["delivery_hours"] > hours_q3 + 1.5 * iqr)).sum())),
        ("declared_value_mean_inr", model_table["declared_value_inr"].mean()),
        ("declared_value_median_inr", model_table["declared_value_inr"].median()),
        ("log_declared_value_mean", np.log(model_table["declared_value_inr"]).mean()),
        ("log_declared_value_median", np.log(model_table["declared_value_inr"]).median()),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).assign(
        value=lambda frame: pd.to_numeric(frame["value"], errors="coerce").round(4).where(
            pd.to_numeric(frame["value"], errors="coerce").notna(), frame["value"]
        )
    ).to_csv(OUTPUTS / "part_a_checks.csv", index=False)

    distance_band = pd.cut(
        model_table["distance_km"], bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"], include_lowest=True,
    )
    groups = []
    for dimension, series in [("service_type", model_table["service_type"]),
                              ("hub_tier", model_table["hub_tier"]),
                              ("distance_band", distance_band)]:
        for group, rows in model_table.groupby(series, observed=True, sort=True):
            groups.append({
                "dimension": dimension, "group": group, "shipments": len(rows),
                "delayed": int(rows["is_delayed"].sum()), "delay_rate": rows["is_delayed"].mean(),
                "mean_hours": rows["delivery_hours"].mean(),
            })
    pd.DataFrame(groups).round(4).to_csv(OUTPUTS / "part_a_delay_by_group.csv", index=False)
    model_table.select_dtypes(include=np.number).corr().rename_axis("column").to_csv(
        OUTPUTS / "part_a_correlations.csv", index=True, index_label="column", float_format="%.4f"
    )


if __name__ == "__main__":
    main()