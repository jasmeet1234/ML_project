from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

DATA_2020 = Path("Data/XY_2020.csv")
DATA_2021 = Path("Data/XY_2021.csv")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


TARGET_COLS = ["comp_built_up", "comp_water", "comp_vegetation", "comp_other"]


def load_dataset() -> pd.DataFrame:
    df20 = pd.read_csv(DATA_2020)
    df21 = pd.read_csv(DATA_2021)

    merged = df20.merge(
        df21[["cell_id", *TARGET_COLS]],
        on="cell_id",
        suffixes=("_2020", "_2021"),
        how="inner",
    )

    for col in TARGET_COLS:
        merged[f"delta_{col}"] = merged[f"{col}_2021"] - merged[f"{col}_2020"]
    return merged


def spatial_hash_split(df: pd.DataFrame, holdout_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    import hashlib
    hashed = df["cell_id"].map(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100)
    test_mask = hashed < int(holdout_ratio * 100)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict:
    metrics = {
        "overall": {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        },
        "per_target": {},
    }

    for i, target in enumerate(target_names):
        metrics["per_target"][target] = {
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }

    return metrics


def false_change_rate(y_true_delta: np.ndarray, y_pred_delta: np.ndarray, threshold: float = 0.05) -> dict:
    true_change = np.any(np.abs(y_true_delta) >= threshold, axis=1)
    pred_change = np.any(np.abs(y_pred_delta) >= threshold, axis=1)

    false_positive = np.sum((pred_change == 1) & (true_change == 0))
    false_negative = np.sum((pred_change == 0) & (true_change == 1))

    fp_rate = false_positive / max(np.sum(true_change == 0), 1)
    fn_rate = false_negative / max(np.sum(true_change == 1), 1)
    stability_accuracy = np.mean(pred_change[true_change == 0] == 0) if np.any(true_change == 0) else float("nan")

    return {
        "threshold": threshold,
        "false_positive_rate": float(fp_rate),
        "false_negative_rate": float(fn_rate),
        "stability_accuracy": float(stability_accuracy),
        "change_prevalence": float(np.mean(true_change)),
    }


def main() -> None:
    df = load_dataset()

    feature_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    feature_cols += [f"{c}_2020" for c in TARGET_COLS]

    train_df, test_df = spatial_hash_split(df)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train_comp = train_df[[f"{c}_2021" for c in TARGET_COLS]].to_numpy()
    y_test_comp = test_df[[f"{c}_2021" for c in TARGET_COLS]].to_numpy()

    y_test_delta = test_df[[f"delta_{c}" for c in TARGET_COLS]].to_numpy()

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                MultiOutputRegressor(
                    XGBRegressor(
                        n_estimators=350,
                        max_depth=7,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )

    model.fit(X_train, y_train_comp)
    y_pred_comp = model.predict(X_test)
    y_pred_comp = np.clip(y_pred_comp, 0, 1)
    y_pred_comp = y_pred_comp / np.maximum(y_pred_comp.sum(axis=1, keepdims=True), 1e-9)

    y_pred_delta = y_pred_comp - test_df[[f"{c}_2020" for c in TARGET_COLS]].to_numpy()

    comp_metrics = evaluate(y_test_comp, y_pred_comp, [f"{c}_2021" for c in TARGET_COLS])
    delta_metrics = evaluate(y_test_delta, y_pred_delta, [f"delta_{c}" for c in TARGET_COLS])
    change_metrics = false_change_rate(y_test_delta, y_pred_delta, threshold=0.05)

    summary = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": feature_cols,
        "composition_metrics": comp_metrics,
        "delta_metrics": delta_metrics,
        "change_specific_metrics": change_metrics,
    }

    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    joblib.dump({"model": model, "features": feature_cols}, ARTIFACTS / "landcover_model.joblib")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
