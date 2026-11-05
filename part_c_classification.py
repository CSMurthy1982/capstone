"""Part C delay classification, model selection and alert thresholding."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def score_predictions(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
    }


def main():
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
    X, y = table[FEATURES], table["is_delayed"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    model_rows = [{"model": "blind_all_on_time", **score_predictions(y_val, np.zeros(len(y_val), dtype=int))}]
    logistic = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
    model_rows.append({"model": "logistic_regression", **score_predictions(y_val, logistic.predict(X_val))})

    tree_results = []
    for depth in [2, 3, 4, 5, 6, 8, 10]:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42).fit(X_train, y_train)
        tree_results.append((depth, f1_score(y_val, tree.predict(X_val), zero_division=0), tree))
    best_depth, _, best_tree = min(tree_results, key=lambda result: (-result[1], result[0]))
    model_rows.append({"model": f"decision_tree_depth_{best_depth}", **score_predictions(y_val, best_tree.predict(X_val))})

    default_forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    model_rows.append({"model": "random_forest_default", **score_predictions(y_val, default_forest.predict(X_val))})
    pd.DataFrame(model_rows).round(4).to_csv(OUTPUTS / "part_c_models.csv", index=False)

    forest_importance = default_forest.feature_importances_
    selector_all = SelectKBest(score_func=f_classif, k="all").fit(X_train, y_train)
    f_scores = np.nan_to_num(selector_all.scores_, nan=0.0, posinf=0.0, neginf=0.0)
    feature_rows = pd.DataFrame({
        "feature": FEATURES, "f_score": f_scores,
        "f_rank": pd.Series(f_scores).rank(method="min", ascending=False).astype(int),
        "rf_importance": forest_importance, "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
    }).sort_values("f_rank", kind="stable")
    feature_rows.round({"f_score": 4, "rf_importance": 4}).to_csv(OUTPUTS / "part_c_feature_scores.csv", index=False)

    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipeline = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]).fit(X_train, y_train)
        selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipeline.predict(X_val), zero_division=0)})
    pd.DataFrame(selectk_rows).round(4).to_csv(OUTPUTS / "part_c_selectk.csv", index=False)

    selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
    X_train_7 = selector.transform(X_train)
    X_val_7 = selector.transform(X_val)
    parameter_grid = {
        "n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), parameter_grid, cv=5).fit(X_train_7, y_train)
    randomized = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), parameter_grid, n_iter=9, cv=5, random_state=42
    ).fit(X_train_7, y_train)
    default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    default_7 = RandomForestClassifier(random_state=42, **default_params).fit(X_train_7, y_train)
    tuning_candidates = [
        ("grid_search", grid.best_params_, grid.best_score_, grid.best_estimator_),
        ("randomized_search", randomized.best_params_, randomized.best_score_, randomized.best_estimator_),
        ("default_forest_same_7_features", default_params, np.nan, default_7),
    ]
    tuning_rows = []
    for name, params, cv_accuracy, model in tuning_candidates:
        predictions = model.predict(X_val_7)
        is_search = name != "default_forest_same_7_features"
        tuning_rows.append({
            "search": name,
            "candidates": 27 if name == "grid_search" else 9 if name == "randomized_search" else 1,
            "cv_folds": 5 if is_search else 0,
            "total_fits": 135 if name == "grid_search" else 45 if name == "randomized_search" else 1,
            "best_params": json.dumps(params, sort_keys=True),
            "best_cv_accuracy": cv_accuracy,
            "val_f1": f1_score(y_val, predictions, zero_division=0),
        })
    pd.DataFrame(tuning_rows).round(4).to_csv(OUTPUTS / "part_c_tuning.csv", index=False)

    priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
    winner = min(tuning_candidates, key=lambda item: (-next(row["val_f1"] for row in tuning_rows if row["search"] == item[0]), priority[item[0]]))
    chosen_from, chosen_params = winner[0], winner[1]
    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **chosen_params)),
    ]).fit(X_train, y_train)
    validation_probabilities = final_model.predict_proba(X_val)[:, 1]
    threshold_rows = []
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics = score_predictions(y_val, (validation_probabilities >= threshold).astype(int))
        threshold_rows.append({"threshold": threshold, **metrics})
    pd.DataFrame(threshold_rows).round(4).to_csv(OUTPUTS / "part_c_thresholds.csv", index=False)
    qualifying = [row["threshold"] for row in threshold_rows if row["recall"] >= 0.80]
    chosen_threshold = max(qualifying) if qualifying else 0.1

    test_metrics = score_predictions(y_test, (final_model.predict_proba(X_test)[:, 1] >= chosen_threshold).astype(int))
    pd.DataFrame([{"threshold": chosen_threshold, **test_metrics}]).round(4).to_csv(
        OUTPUTS / "part_c_final_test.csv", index=False
    )
    config = {
        "selector_k": 7, "n_estimators": chosen_params["n_estimators"],
        "max_depth": "unlimited" if chosen_params.get("max_depth") is None else chosen_params.get("max_depth"),
        "min_samples_leaf": chosen_params["min_samples_leaf"],
        "threshold": chosen_threshold, "chosen_from": chosen_from,
    }
    pd.DataFrame([config]).to_csv(OUTPUTS / "part_c_final_config.csv", index=False)


if __name__ == "__main__":
    main()