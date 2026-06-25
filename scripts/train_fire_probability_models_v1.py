import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path("data/features/county_month_model_table_v1.parquet")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/model_outputs")

MODEL_BUNDLE_PATH = MODEL_DIR / "fire_probability_models_v1.joblib"
XGBOOST_NATIVE_PATH = MODEL_DIR / "xgboost_fire_probability_v1.json"
METRICS_PATH = OUTPUT_DIR / "fire_probability_metrics_v1.json"
PREDICTIONS_PATH = OUTPUT_DIR / "fire_probability_predictions_v1.parquet"
IMPORTANCE_PATH = OUTPUT_DIR / "xgboost_feature_importance_v1.parquet"

TARGET = "meaningful_fire_activity"
RANDOM_STATE = 42

IDENTIFIER_COLUMNS = [
    "county_geoid",
    "county_name",
    "state_fips",
    "state",
    "year",
    "month",
    "month_start",
    "split",
]

EXCLUDED_PREDICTORS = {
    "county_geoid",
    "county_name",
    "state_fips",
    "year",
    "month",
    "month_start",
    "split",
    "any_fire_activity",
    "meaningful_fire_activity",
    "weather_complete",
}

CATEGORICAL_COLUMNS = ["state"]


def load_model_table() -> pd.DataFrame:
    """Load the model table and retain rows with complete weather."""
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Missing {INPUT_PATH}. Run "
            "`python scripts/build_gridmet_weather_features.py "
            "--start-year 2012 --end-year 2025 --delete-raw` first."
        )

    data = pd.read_parquet(INPUT_PATH)

    required = {
        "county_geoid",
        "state",
        "year",
        "month",
        "month_start",
        "split",
        TARGET,
    }

    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{INPUT_PATH} is missing columns: {missing}")

    data["county_geoid"] = data["county_geoid"].astype(str).str.zfill(5)
    data["year"] = data["year"].astype(int)
    data["month"] = data["month"].astype(int)
    data["month_start"] = pd.to_datetime(data["month_start"], errors="coerce")
    data[TARGET] = data[TARGET].astype(int)

    if data["month_start"].isna().any():
        raise ValueError("Some rows have invalid month_start values.")

    if "weather_complete" in data.columns:
        before = len(data)
        data = data[data["weather_complete"] == 1].copy()
        print(
            f"Kept {len(data):,} of {before:,} rows with complete weather features."
        )

    data = data[
        data["split"].isin(["train", "validation", "test"])
    ].copy()

    duplicate_count = int(
        data.duplicated(["county_geoid", "year", "month"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate county-month rows."
        )

    return data.sort_values(
        ["year", "month", "state", "county_geoid"]
    ).reset_index(drop=True)


def select_predictors(
    data: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Select numeric predictors plus state as a categorical predictor."""
    categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in data.columns
    ]

    numeric = [
        column
        for column in data.columns
        if (
            column not in EXCLUDED_PREDICTORS
            and column not in categorical
            and pd.api.types.is_numeric_dtype(data[column])
        )
    ]

    if not numeric:
        raise ValueError("No numeric predictors were found.")

    return sorted(numeric), categorical


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build one preprocessing object shared by logistic regression and XGBoost."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    transformers = [
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        )
    ]

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def smoothed_rate(
    positives: pd.Series,
    counts: pd.Series,
    prior: pd.Series | float,
    strength: float,
) -> pd.Series:
    """Empirical-Bayes smoothing toward a supplied prior probability."""
    return (
        positives.astype(float)
        + strength * prior
    ) / (
        counts.astype(float)
        + strength
    )


def fit_climatology(train: pd.DataFrame) -> dict:
    """Fit global, month, state-month, and county-month probability baselines."""
    global_rate = float(train[TARGET].mean())

    month = (
        train.groupby("month")[TARGET]
        .agg(["sum", "count"])
        .reset_index()
    )
    month["probability"] = smoothed_rate(
        month["sum"],
        month["count"],
        global_rate,
        strength=20.0,
    )

    state_month = (
        train.groupby(["state", "month"])[TARGET]
        .agg(["sum", "count"])
        .reset_index()
        .merge(
            month[["month", "probability"]].rename(
                columns={"probability": "month_prior"}
            ),
            on="month",
            how="left",
            validate="many_to_one",
        )
    )
    state_month["probability"] = smoothed_rate(
        state_month["sum"],
        state_month["count"],
        state_month["month_prior"],
        strength=15.0,
    )

    county_month = (
        train.groupby(["county_geoid", "state", "month"])[TARGET]
        .agg(["sum", "count"])
        .reset_index()
        .merge(
            state_month[["state", "month", "probability"]].rename(
                columns={"probability": "state_month_prior"}
            ),
            on=["state", "month"],
            how="left",
            validate="many_to_one",
        )
    )
    county_month["probability"] = smoothed_rate(
        county_month["sum"],
        county_month["count"],
        county_month["state_month_prior"],
        strength=8.0,
    )

    return {
        "global_rate": global_rate,
        "month": month[["month", "probability"]],
        "state_month": state_month[
            ["state", "month", "probability"]
        ],
        "county_month": county_month[
            ["county_geoid", "month", "probability"]
        ],
        "smoothing_strengths": {
            "month": 20.0,
            "state_month": 15.0,
            "county_month": 8.0,
        },
    }


def predict_climatology(
    data: pd.DataFrame,
    climatology: dict,
) -> pd.DataFrame:
    """Create hierarchical climatology predictions with fallbacks."""
    output = data[
        ["county_geoid", "state", "month"]
    ].copy()

    output = output.merge(
        climatology["month"].rename(
            columns={"probability": "p_month_climatology"}
        ),
        on="month",
        how="left",
        validate="many_to_one",
    )

    output = output.merge(
        climatology["state_month"].rename(
            columns={"probability": "p_state_month_climatology"}
        ),
        on=["state", "month"],
        how="left",
        validate="many_to_one",
    )

    output = output.merge(
        climatology["county_month"].rename(
            columns={"probability": "p_county_month_climatology"}
        ),
        on=["county_geoid", "month"],
        how="left",
        validate="many_to_one",
    )

    output["p_global_prevalence"] = climatology["global_rate"]

    output["p_month_climatology"] = (
        output["p_month_climatology"]
        .fillna(climatology["global_rate"])
    )

    output["p_state_month_climatology"] = (
        output["p_state_month_climatology"]
        .fillna(output["p_month_climatology"])
    )

    output["p_county_month_climatology"] = (
        output["p_county_month_climatology"]
        .fillna(output["p_state_month_climatology"])
    )

    return output[
        [
            "p_global_prevalence",
            "p_month_climatology",
            "p_state_month_climatology",
            "p_county_month_climatology",
        ]
    ]


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.digitize(
        probability,
        edges[1:-1],
        right=False,
    )

    error = 0.0
    total = len(y_true)

    for index in range(bins):
        mask = bin_index == index
        if not np.any(mask):
            continue

        observed = float(np.mean(y_true[mask]))
        predicted = float(np.mean(probability[mask]))
        error += (
            float(np.sum(mask))
            / total
            * abs(observed - predicted)
        )

    return float(error)


def top_fraction_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    fraction: float,
) -> dict:
    """Measure performance among the highest predicted probabilities."""
    count = max(1, int(math.ceil(len(y_true) * fraction)))
    indexes = np.argsort(-probability, kind="stable")[:count]

    positives_in_top = int(y_true[indexes].sum())
    total_positives = int(y_true.sum())

    return {
        "fraction": fraction,
        "rows": count,
        "precision": float(np.mean(y_true[indexes])),
        "recall": (
            float(positives_in_top / total_positives)
            if total_positives
            else None
        ),
        "lift_over_prevalence": float(
            np.mean(y_true[indexes]) / np.mean(y_true)
        ),
    }


def metric_set(
    y_true: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
) -> dict:
    """Evaluate ranking, probability quality, calibration, and top-risk capture."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(
        np.asarray(probability, dtype=float),
        1e-7,
        1.0 - 1e-7,
    )

    return {
        "rows": int(len(y)),
        "positive_count": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "expected_calibration_error_10_bins": (
            expected_calibration_error(y, p, bins=10)
        ),
        "top_5_percent": top_fraction_metrics(y, p, 0.05),
        "top_10_percent": top_fraction_metrics(y, p, 0.10),
    }


def evaluate_predictions(
    frame: pd.DataFrame,
    probability_columns: list[str],
) -> dict:
    """Calculate metrics and Brier skill relative to global prevalence."""
    results = {}

    global_brier = brier_score_loss(
        frame[TARGET],
        frame["p_global_prevalence"],
    )

    for column in probability_columns:
        metrics = metric_set(
            frame[TARGET],
            frame[column],
        )

        metrics["brier_skill_vs_global"] = float(
            1.0
            - metrics["brier_score"]
            / global_brier
        )

        results[column] = metrics

    return results


def prediction_frame(
    data: pd.DataFrame,
    climatology_predictions: pd.DataFrame,
    logistic_probability: np.ndarray,
    xgboost_probability: np.ndarray,
) -> pd.DataFrame:
    """Combine identifiers, target, and all model probabilities."""
    columns = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in data.columns
    ]

    output = data[columns + [TARGET]].reset_index(drop=True).copy()

    output = pd.concat(
        [
            output,
            climatology_predictions.reset_index(drop=True),
        ],
        axis=1,
    )

    output["p_logistic_regression"] = logistic_probability
    output["p_xgboost"] = xgboost_probability

    return output


def main() -> None:
    data = load_model_table()

    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()

    for name, frame in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if frame.empty:
            raise ValueError(f"The {name} split is empty.")
        print(
            f"{name}: {len(frame):,} rows, "
            f"positive rate = {frame[TARGET].mean():.4f}"
        )

    numeric_columns, categorical_columns = select_predictors(data)
    predictor_columns = numeric_columns + categorical_columns

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    X_train = preprocessor.fit_transform(train[predictor_columns])
    X_validation = preprocessor.transform(
        validation[predictor_columns]
    )
    X_test = preprocessor.transform(test[predictor_columns])

    y_train = train[TARGET].to_numpy()
    y_validation = validation[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    climatology = fit_climatology(train)

    validation_climatology = predict_climatology(
        validation,
        climatology,
    )
    test_climatology = predict_climatology(
        test,
        climatology,
    )

    print("Training logistic regression...")
    logistic = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    logistic.fit(X_train, y_train)

    logistic_validation = logistic.predict_proba(
        X_validation
    )[:, 1]
    logistic_test = logistic.predict_proba(X_test)[:, 1]

    print("Training XGBoost...")
    early_stopping = xgb.callback.EarlyStopping(
        rounds=75,
        metric_name="logloss",
        data_name="validation_0",
        save_best=True,
    )

    xgboost_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        gamma=0.0,
        tree_method="hist",
        eval_metric="logloss",
        callbacks=[early_stopping],
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    xgboost_model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    xgboost_validation = xgboost_model.predict_proba(
        X_validation
    )[:, 1]
    xgboost_test = xgboost_model.predict_proba(X_test)[:, 1]

    validation_predictions = prediction_frame(
        validation,
        validation_climatology,
        logistic_validation,
        xgboost_validation,
    )
    test_predictions = prediction_frame(
        test,
        test_climatology,
        logistic_test,
        xgboost_test,
    )

    predictions = pd.concat(
        [
            validation_predictions,
            test_predictions,
        ],
        ignore_index=True,
    )

    probability_columns = [
        "p_global_prevalence",
        "p_month_climatology",
        "p_state_month_climatology",
        "p_county_month_climatology",
        "p_logistic_regression",
        "p_xgboost",
    ]

    validation_metrics = evaluate_predictions(
        validation_predictions,
        probability_columns,
    )
    test_metrics = evaluate_predictions(
        test_predictions,
        probability_columns,
    )

    candidate_models = [
        "p_county_month_climatology",
        "p_logistic_regression",
        "p_xgboost",
    ]

    selected_model = min(
        candidate_models,
        key=lambda name: validation_metrics[name]["brier_score"],
    )

    transformed_feature_names = (
        preprocessor.get_feature_names_out().tolist()
    )

    importance = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": xgboost_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "target": TARGET,
        "selected_model_by_validation_brier": selected_model,
        "predictor_columns": predictor_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "preprocessor": preprocessor,
        "climatology": climatology,
        "logistic_regression": logistic,
        "xgboost": xgboost_model,
        "split_definition": {
            "train": "2013-2021",
            "validation": "2022",
            "test": "2023-2025",
        },
        "package_versions": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
    }

    joblib.dump(
        model_bundle,
        MODEL_BUNDLE_PATH,
        compress=3,
    )

    xgboost_model.save_model(XGBOOST_NATIVE_PATH)

    predictions.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    importance.to_parquet(
        IMPORTANCE_PATH,
        index=False,
    )

    metrics_output = {
        "input": str(INPUT_PATH),
        "target": TARGET,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(train[TARGET].mean()),
        "validation_positive_rate": float(
            validation[TARGET].mean()
        ),
        "test_positive_rate": float(test[TARGET].mean()),
        "predictor_count_before_encoding": int(
            len(predictor_columns)
        ),
        "transformed_feature_count": int(
            len(transformed_feature_names)
        ),
        "selected_model_by_validation_brier": selected_model,
        "xgboost_best_iteration": (
            int(xgboost_model.best_iteration)
            if hasattr(xgboost_model, "best_iteration")
            else None
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_selection_rule": (
            "Choose among county-month climatology, logistic regression, "
            "and XGBoost using the lowest 2022 validation Brier score. "
            "The 2023-2025 test metrics are not used for selection."
        ),
        "probability_note": (
            "No class weighting is used because the output is intended "
            "to represent occurrence probability. Calibration is evaluated "
            "but not adjusted in this first comparison."
        ),
        "package_versions": model_bundle["package_versions"],
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics_output, file, indent=2)

    print(f"Saved model bundle to {MODEL_BUNDLE_PATH}")
    print(f"Saved native XGBoost model to {XGBOOST_NATIVE_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved feature importance to {IMPORTANCE_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(
        "Selected model by validation Brier score: "
        f"{selected_model}"
    )

    for model_name in candidate_models:
        validation_result = validation_metrics[model_name]
        test_result = test_metrics[model_name]

        print(
            f"{model_name}: "
            f"validation PR-AUC={validation_result['pr_auc']:.4f}, "
            f"validation Brier={validation_result['brier_score']:.4f}; "
            f"test PR-AUC={test_result['pr_auc']:.4f}, "
            f"test Brier={test_result['brier_score']:.4f}"
        )


if __name__ == "__main__":
    main()
