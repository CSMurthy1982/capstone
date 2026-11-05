"""Part B delivery-hours regression experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted), "MSE": mse,
        "RMSE": np.sqrt(mse), "R2": r2_score(actual, predicted),
    }


def main():
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
    X, y = table[FEATURES], table["delivery_hours"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler().fit(X_train)
    train_scaled, test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    checks = [
        ("train_rows", len(X_train)), ("val_rows", len(X_val)), ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", np.abs(train_scaled.mean(axis=0)).max()),
        ("test_scaled_max_abs_mean", np.abs(test_scaled.mean(axis=0)).max()),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).round(4).to_csv(OUTPUTS / "part_b_checks.csv", index=False)

    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_rows = []
    for depth in depths:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))
        val_rmse = np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))
        depth_rows.append({"max_depth": "unlimited" if depth is None else depth, "train_rmse": train_rmse, "val_rmse": val_rmse})
    pd.DataFrame(depth_rows).round(4).to_csv(OUTPUTS / "part_b_tree_depth.csv", index=False)
    best_depth = min(depths, key=lambda depth: next(row["val_rmse"] for row in depth_rows if row["max_depth"] == ("unlimited" if depth is None else depth)))

    models = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    rows = []
    baseline_prediction = np.full(len(y_val), y_train.mean())
    rows.append({"model": "mean_baseline", "split": "val", **regression_metrics(y_val, baseline_prediction)})
    fitted = {}
    for name, model in models.items():
        fitted[name] = model.fit(X_train, y_train)
        rows.append({"model": name, "split": "val", **regression_metrics(y_val, model.predict(X_val))})
    winner = min(models, key=lambda name: next(row["RMSE"] for row in rows if row["model"] == name))
    rows.append({"model": winner, "split": "test", **regression_metrics(y_test, fitted[winner].predict(X_test))})
    pd.DataFrame(rows).round(4).to_csv(OUTPUTS / "part_b_metrics.csv", index=False)

    learning_rows = []
    for rate in [0.01, 0.1, 1.0]:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=rate, random_state=42).fit(X_train, y_train)
        metrics = regression_metrics(y_val, model.predict(X_val))
        learning_rows.append({"learning_rate": rate, "val_rmse": metrics["RMSE"], "val_r2": metrics["R2"]})
    pd.DataFrame(learning_rows).round(4).to_csv(OUTPUTS / "part_b_gb_learning_rate.csv", index=False)

    ranked = sorted(models, key=lambda name: next(row["RMSE"] for row in rows if row["model"] == name))[:2]
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name in ranked:
        scores = cross_val_score(models[name], X_train_val, y_train_val, cv=folds)
        cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
    pd.DataFrame(cv_rows).round(4).to_csv(OUTPUTS / "part_b_cv.csv", index=False)


if __name__ == "__main__":
    main()