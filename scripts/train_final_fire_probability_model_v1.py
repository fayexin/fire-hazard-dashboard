import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb


try:
    from train_fire_probability_models_v1 import (
        TARGET,
        build_preprocessor,
        select_predictors,
    )
except ImportError:
    from scripts.train_fire_probability_models_v1 import (
        TARGET,
        build_preprocessor,
        select_predictors,
    )


INPUT_PATH = Path(
    "data/features/county_month_model_table_v1.parquet"
)

EVALUATION_METRICS_PATH = Path(
    "data/model_outputs/fire_probability_metrics_v1.json"
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/model_outputs")

MODEL_BUNDLE_PATH = (
    MODEL_DIR / "fire_probability_deployment_v1.joblib"
)

NATIVE_MODEL_PATH = (
    MODEL_DIR / "xgboost_fire_probability_deployment_v1.json"
)

SCORES_PATH = (
    OUTPUT_DIR / "fire_probability_deployment_scores_v1.parquet"
)

LATEST_SCORES_PATH = (
    OUTPUT_DIR / "fire_probability_latest_scores_v1.parquet"
)

IMPORTANCE_PATH = (
    OUTPUT_DIR / "xgboost_deployment_feature_importance_v1.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "fire_probability_deployment_summary_v1.json"
)

TRAIN_START_YEAR = 2013
TRAIN_END_YEAR = 2025
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

PROBABILITY_BINS = [
    -np.inf,
    0.20,
    0.40,
    0.60,
    0.80,
    np.inf,
]

PROBABILITY_LABELS = [
    "Very low",
    "Low",
    "Moderate",
    "High",
    "Very high",
]


def load_full_model_table() -> pd.DataFrame:
    """Load all model-table rows, including post-2025 scoring rows."""
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Missing {INPUT_PATH}. Run the feature-building scripts first."
        )

    data = pd.read_parquet(INPUT_PATH)

    required = {
        "county_geoid",
        "county_name",
        "state",
        "year",
        "month",
        "month_start",
        TARGET,
    }

    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"{INPUT_PATH} is missing required columns: {missing}"
        )

    data["county_geoid"] = (
        data["county_geoid"]
        .astype(str)
        .str.zfill(5)
    )

    if "state_fips" in data.columns:
        data["state_fips"] = (
            data["state_fips"]
            .astype(str)
            .str.zfill(2)
        )

    data["year"] = data["year"].astype(int)
    data["month"] = data["month"].astype(int)
    data["month_start"] = pd.to_datetime(
        data["month_start"],
        errors="coerce",
    )
    data[TARGET] = data[TARGET].astype(int)

    if data["month_start"].isna().any():
        raise ValueError(
            "Some rows have invalid month_start values."
        )

    duplicate_count = int(
        data.duplicated(
            ["county_geoid", "year", "month"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate county-month rows."
        )

    return data.sort_values(
        ["year", "month", "state", "county_geoid"]
    ).reset_index(drop=True)


def load_evaluation_metrics() -> dict:
    """Load the locked held-out evaluation results."""
    if not EVALUATION_METRICS_PATH.exists():
        raise SystemExit(
            f"Missing {EVALUATION_METRICS_PATH}. "
            "Run the evaluation training script first."
        )

    with open(
        EVALUATION_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def choose_tree_count(metrics: dict) -> int:
    """
    Reuse the tree count selected by 2022 early stopping.

    The deployment refit does not tune on 2023-2025.
    """
    best_iteration = metrics.get(
        "xgboost_best_iteration"
    )

    if best_iteration is None:
        return 500

    tree_count = int(best_iteration) + 1

    if tree_count < 1:
        raise ValueError(
            f"Invalid XGBoost tree count: {tree_count}"
        )

    return tree_count


def validate_scoring_features(
    data: pd.DataFrame,
    predictor_columns: list[str],
) -> None:
    """Check that scoring rows contain every trained predictor."""
    missing = [
        column
        for column in predictor_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Scoring table is missing predictors: {missing}"
        )


def probability_band(
    probability: pd.Series,
) -> pd.Series:
    """Create descriptive bands for display, not official danger classes."""
    return pd.cut(
        probability,
        bins=PROBABILITY_BINS,
        labels=PROBABILITY_LABELS,
        ordered=True,
    ).astype(str)


def main() -> None:
    data = load_full_model_table()
    evaluation_metrics = load_evaluation_metrics()

    if "weather_complete" in data.columns:
        scoring_data = data[
            data["weather_complete"] == 1
        ].copy()
    else:
        scoring_data = data.copy()

    training_data = scoring_data[
        scoring_data["year"].between(
            TRAIN_START_YEAR,
            TRAIN_END_YEAR,
        )
    ].copy()

    if training_data.empty:
        raise ValueError(
            "No deployment training rows were found."
        )

    numeric_columns, categorical_columns = (
        select_predictors(training_data)
    )

    predictor_columns = (
        numeric_columns
        + categorical_columns
    )

    validate_scoring_features(
        scoring_data,
        predictor_columns,
    )

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    x_train = preprocessor.fit_transform(
        training_data[predictor_columns]
    )

    y_train = training_data[TARGET].to_numpy()

    tree_count = choose_tree_count(
        evaluation_metrics
    )

    print(
        f"Training deployment XGBoost on "
        f"{len(training_data):,} rows from "
        f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}."
    )

    print(
        f"Using {tree_count:,} trees selected from "
        "the locked 2022 validation procedure."
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=tree_count,
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
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model.fit(
        x_train,
        y_train,
        verbose=False,
    )

    x_score = preprocessor.transform(
        scoring_data[predictor_columns]
    )

    probability = model.predict_proba(
        x_score
    )[:, 1]

    output_columns = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in scoring_data.columns
    ]

    scores = scoring_data[
        output_columns
    ].reset_index(drop=True).copy()

    scores["predicted_probability"] = probability

    scores["probability_band"] = probability_band(
        scores["predicted_probability"]
    )

    scores["score_role"] = np.where(
        scores["year"] <= TRAIN_END_YEAR,
        "deployment_training_period",
        "post_training_score",
    )

    scores["observed_target"] = np.where(
        scores["year"] <= TRAIN_END_YEAR,
        scoring_data[TARGET].to_numpy(),
        np.nan,
    )

    latest_month = scores["month_start"].max()

    latest_scores = scores[
        scores["month_start"] == latest_month
    ].copy()

    latest_scores = latest_scores.sort_values(
        "predicted_probability",
        ascending=False,
    ).reset_index(drop=True)

    transformed_feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    importance = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    locked_test_metrics = (
        evaluation_metrics
        .get("test_metrics", {})
        .get("p_xgboost", {})
    )

    model_bundle = {
        "target": TARGET,
        "training_start_year": TRAIN_START_YEAR,
        "training_end_year": TRAIN_END_YEAR,
        "tree_count": tree_count,
        "predictor_columns": predictor_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "preprocessor": preprocessor,
        "xgboost": model,
        "probability_bins": PROBABILITY_BINS,
        "probability_labels": PROBABILITY_LABELS,
        "locked_test_metrics": locked_test_metrics,
        "package_versions": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
    }

    summary = {
        "target": TARGET,
        "training_period": (
            f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}"
        ),
        "training_rows": int(
            len(training_data)
        ),
        "training_positive_rate": float(
            training_data[TARGET].mean()
        ),
        "tree_count": tree_count,
        "predictor_count_before_encoding": int(
            len(predictor_columns)
        ),
        "transformed_feature_count": int(
            len(transformed_feature_names)
        ),
        "scored_rows": int(
            len(scores)
        ),
        "latest_scored_month": (
            latest_month.strftime("%Y-%m-%d")
        ),
        "latest_scored_counties": int(
            len(latest_scores)
        ),
        "latest_probability_summary": {
            "minimum": float(
                latest_scores[
                    "predicted_probability"
                ].min()
            ),
            "median": float(
                latest_scores[
                    "predicted_probability"
                ].median()
            ),
            "maximum": float(
                latest_scores[
                    "predicted_probability"
                ].max()
            ),
        },
        "locked_evaluation": {
            "evaluation_period": "2023-2025",
            "test_metrics": locked_test_metrics,
            "note": (
                "These metrics come from the earlier model trained on "
                "2013-2021 and selected with 2022 validation. They are "
                "not recomputed after fitting the deployment model on "
                "2013-2025."
            ),
        },
        "score_interpretation": {
            "post_training_score": (
                "A probability generated for a month after the final "
                "deployment training year."
            ),
            "deployment_training_period": (
                "An in-sample deployment-model score. It must not be "
                "used as model evaluation evidence."
            ),
            "probability_band": (
                "A descriptive display category, not an official fire-"
                "danger classification."
            ),
        },
        "package_versions": (
            model_bundle[
                "package_versions"
            ]
        ),
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_bundle,
        MODEL_BUNDLE_PATH,
        compress=3,
    )

    model.save_model(
        NATIVE_MODEL_PATH
    )

    scores.to_parquet(
        SCORES_PATH,
        index=False,
    )

    latest_scores.to_parquet(
        LATEST_SCORES_PATH,
        index=False,
    )

    importance.to_parquet(
        IMPORTANCE_PATH,
        index=False,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"Saved deployment model to "
        f"{MODEL_BUNDLE_PATH}"
    )

    print(
        f"Saved latest scores for "
        f"{latest_month.date()} to "
        f"{LATEST_SCORES_PATH}"
    )

    print(
        f"Latest probability range: "
        f"{summary['latest_probability_summary']}"
    )


if __name__ == "__main__":
    main()
