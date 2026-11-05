"""Fit and save the final classifier, then verify its serialized reload."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"


def main():
    MODELS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
    config = pd.read_csv(OUTPUTS / "part_c_final_config.csv").iloc[0]
    X, y = table[FEATURES], table["is_delayed"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    _, X_test, _, _ = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    max_depth = None if str(config["max_depth"]).lower() == "unlimited" else int(config["max_depth"])
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
        ("model", RandomForestClassifier(
            n_estimators=int(config["n_estimators"]), max_depth=max_depth,
            min_samples_leaf=int(config["min_samples_leaf"]), random_state=42,
        )),
    ]).fit(X_train, y_train)
    joblib.dump(pipeline, MODELS / "delay_model.pkl")
    pd.DataFrame({"threshold": [float(config["threshold"])]}).to_csv(MODELS / "delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv(MODELS / "feature_order.csv", index=False)

    raw = pd.read_csv(ROOT / "data" / "raw_shipments.csv")
    _, medians = clean_shipments(raw)
    pd.DataFrame([{"column": column, "median": value} for column, value in medians.items()]).to_csv(
        MODELS / "impute_values.csv", index=False
    )

    reloaded = joblib.load(MODELS / "delay_model.pkl")
    memory_probabilities = pipeline.predict_proba(X_test)[:, 1]
    reloaded_probabilities = reloaded.predict_proba(X_test)[:, 1]
    check_rows = [
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", np.max(np.abs(memory_probabilities - reloaded_probabilities))),
        ("threshold", float(config["threshold"])),
    ]
    pd.DataFrame(check_rows, columns=["metric", "value"]).round(4).to_csv(
        OUTPUTS / "part_d_reload_check.csv", index=False
    )


if __name__ == "__main__":
    main()