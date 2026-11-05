"""Score new shipments and write a deterministic operations action list."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

ROOT = Path(__file__).resolve().parent


def main():
    models = ROOT / "models"
    outputs = ROOT / "outputs"
    raw = pd.read_csv(ROOT / "data" / "new_shipments.csv")
    hubs = pd.read_csv(ROOT / "data" / "hubs.csv")
    segments = pd.read_csv(outputs / "hub_segments.csv")[["hub_id", "segment"]]
    feature_order = pd.read_csv(models / "feature_order.csv")["feature"].tolist()
    threshold = float(pd.read_csv(models / "delay_threshold.csv")["threshold"].iloc[0])
    medians_frame = pd.read_csv(models / "impute_values.csv").set_index("column")["median"]
    medians = {column: float(medians_frame[column]) for column in ["distance_km", "parcel_weight_kg", "declared_value_inr"]}

    clean, _ = clean_shipments(raw, medians=medians)
    featured = build_features(clean, hubs)
    model = joblib.load(models / "delay_model.pkl")
    probabilities = np.round(model.predict_proba(featured[feature_order])[:, 1], 4)
    flagged = (probabilities >= threshold).astype(int)
    result = pd.DataFrame({
        "shipment_id": featured["shipment_id"].to_numpy(),
        "hub_id": featured["hub_id"].to_numpy(),
        "delay_probability": probabilities,
        "flagged": flagged,
    }).merge(segments, on="hub_id", how="left", validate="many_to_one", sort=False)
    result["action"] = np.select(
        [(result["flagged"] == 1) & (result["segment"] == "Escalate"), result["flagged"] == 1],
        ["Escalate to hub manager", "Proactive customer call"], default="No action",
    )
    result[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]].to_csv(
        outputs / "new_shipment_actions.csv", index=False
    )


if __name__ == "__main__":
    main()