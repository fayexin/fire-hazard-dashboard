import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


try:
    from train_fire_probability_models_v1 import (
        TARGET,
        build_preprocessor,
        fit_climatology,
        load_model_table,
        metric_set,
        predict_climatology,
        select_predictors,
    )
except ImportError:
    from scripts.train_fire_probability_models_v1 import (
        TARGET,
        build_preprocessor,
        fit_climatology,
        load_model_table,
        metric_set,
        predict_climatology,
        select_predictors,
    )


OUTPUT_DIR = Path("data/model_outputs")

SUMMARY_PATH = (
    OUTPUT_DIR / "fire_probability_ablation_v1.json"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "fire_probability_ablation_predictions_v1.parquet"
)

YEAR_PATH = (
    OUTPUT_DIR / "fire_probability_robustness_by_year_v1.parquet"
)

STATE_PATH = (
    OUTPUT_DIR / "fire_probability_robustness_by_state_v1.parquet"
)

RANDOM_STATE = 42

SEASON_GEOGRAPHY_CANDIDATES = [
    "month_sin",
    "month_cos",
    "months_since_panel_start",
    "county_area_km2",
    "county_centroid_latitude",
    "county_centroid_longitude",
]


def build_feature_groups(
    data: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Build nested feature groups.

    Every group retains the state categorical variable. This makes the
    comparison answer a useful question: what do fire-history and weather
    variables add beyond basic season and geography?
    """
    all_numeric, categorical = select_predictors(data)

    season_geography = [
        column
        for column in SEASON_GEOGRAPHY_CANDIDATES
        if column in all_numeric
    ]

    weather = [
        column
        for column in all_numeric
        if column.startswith("gridmet_")
    ]

    fire_history = [
        column
        for column in all_numeric
        if (
            column not in season_geography
            and column not in weather
        )
    ]

    if not season_geography:
        raise ValueError(
            "No season/geography features were found."
        )

    if not weather:
        raise ValueError(
            "No gridMET weather features were found."
        )

    if not fire_history:
        raise ValueError(
            "No fire-history features were found."
        )

    groups = {
        "xgb_season_geography": sorted(
            season_geography
        ),
        "xgb_plus_fire_history": sorted(
            set(season_geography + fire_history)
        ),
        "xgb_plus_weather": sorted(
            set(season_geography + weather)
        ),
        "xgb_all_features": sorted(
            set(
                season_geography
                + fire_history
                + weather
            )
        ),
    }

    return groups, categorical


def train_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[
    object,
    xgb.XGBClassifier,
    np.ndarray,
    np.ndarray,
]:
    """Train one XGBoost model with the same settings used in v1."""
    predictor_columns = (
        numeric_columns
        + categorical_columns
    )

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    x_train = preprocessor.fit_transform(
        train[predictor_columns]
    )

    x_validation = preprocessor.transform(
        validation[predictor_columns]
    )

    x_test = preprocessor.transform(
        test[predictor_columns]
    )

    y_train = train[TARGET].to_numpy()
    y_validation = validation[TARGET].to_numpy()

    early_stopping = xgb.callback.EarlyStopping(
        rounds=75,
        metric_name="logloss",
        data_name="validation_0",
        save_best=True,
    )

    model = xgb.XGBClassifier(
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

    model.fit(
        x_train,
        y_train,
        eval_set=[
            (x_validation, y_validation)
        ],
        verbose=False,
    )

    validation_probability = model.predict_proba(
        x_validation
    )[:, 1]

    test_probability = model.predict_proba(
        x_test
    )[:, 1]

    return (
        preprocessor,
        model,
        validation_probability,
        test_probability,
    )


def evaluate(
    y_true: pd.Series,
    probability: np.ndarray,
    reference_brier: float,
) -> dict:
    """Evaluate a model and add Brier skill against climatology."""
    metrics = metric_set(
        y_true,
        probability,
    )

    metrics[
        "brier_skill_vs_county_month_climatology"
    ] = float(
        1.0
        - metrics["brier_score"]
        / reference_brier
    )

    return metrics


def safe_group_metrics(
    y_true: pd.Series,
    probability: pd.Series,
) -> dict:
    """Metrics safe for small state or year subsets."""
    y = np.asarray(
        y_true,
        dtype=int,
    )

    p = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        1e-7,
        1.0 - 1e-7,
    )

    unique_classes = np.unique(y)

    return {
        "rows": int(len(y)),
        "positive_count": int(y.sum()),
        "prevalence": float(y.mean()),
        "roc_auc": (
            float(roc_auc_score(y, p))
            if len(unique_classes) == 2
            else None
        ),
        "pr_auc": (
            float(
                average_precision_score(
                    y,
                    p,
                )
            )
            if y.sum() > 0
            else None
        ),
        "brier_score": float(
            brier_score_loss(
                y,
                p,
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),
    }


def robustness_table(
    predictions: pd.DataFrame,
    group_column: str,
    probability_columns: list[str],
) -> pd.DataFrame:
    """Calculate metrics for each model inside each year or state."""
    records = []

    for group_value, group in predictions.groupby(
        group_column,
        sort=True,
    ):
        climatology_metrics = safe_group_metrics(
            group[TARGET],
            group[
                "p_county_month_climatology"
            ],
        )

        reference_brier = climatology_metrics[
            "brier_score"
        ]

        for probability_column in probability_columns:
            metrics = safe_group_metrics(
                group[TARGET],
                group[probability_column],
            )

            records.append(
                {
                    group_column: group_value,
                    "model": probability_column,
                    **metrics,
                    (
                        "brier_skill_vs_"
                        "county_month_climatology"
                    ): (
                        float(
                            1.0
                            - metrics["brier_score"]
                            / reference_brier
                        )
                        if reference_brier > 0
                        else None
                    ),
                }
            )

    return pd.DataFrame.from_records(
        records
    )


def comparison_summary(
    validation_metrics: dict,
    test_metrics: dict,
) -> dict:
    """Summarize whether weather adds value beyond fire history."""
    history_name = "xgb_plus_fire_history"
    all_name = "xgb_all_features"

    validation_history = validation_metrics[
        history_name
    ]

    validation_all = validation_metrics[
        all_name
    ]

    test_history = test_metrics[
        history_name
    ]

    test_all = test_metrics[
        all_name
    ]

    return {
        "weather_beyond_fire_history": {
            "validation_brier_change": float(
                validation_all["brier_score"]
                - validation_history["brier_score"]
            ),
            "validation_pr_auc_change": float(
                validation_all["pr_auc"]
                - validation_history["pr_auc"]
            ),
            "test_brier_change": float(
                test_all["brier_score"]
                - test_history["brier_score"]
            ),
            "test_pr_auc_change": float(
                test_all["pr_auc"]
                - test_history["pr_auc"]
            ),
            "weather_improves_validation_brier": bool(
                validation_all["brier_score"]
                < validation_history["brier_score"]
            ),
            "weather_improves_test_brier": bool(
                test_all["brier_score"]
                < test_history["brier_score"]
            ),
        }
    }


def main() -> None:
    data = load_model_table()

    train = data[
        data["split"] == "train"
    ].copy()

    validation = data[
        data["split"] == "validation"
    ].copy()

    test = data[
        data["split"] == "test"
    ].copy()

    groups, categorical_columns = (
        build_feature_groups(data)
    )

    climatology = fit_climatology(
        train
    )

    validation_climatology = (
        predict_climatology(
            validation,
            climatology,
        )
    )

    test_climatology = (
        predict_climatology(
            test,
            climatology,
        )
    )

    validation_predictions = validation[
        [
            "county_geoid",
            "county_name",
            "state",
            "year",
            "month",
            "month_start",
            "split",
            TARGET,
        ]
    ].reset_index(drop=True).copy()

    test_predictions = test[
        [
            "county_geoid",
            "county_name",
            "state",
            "year",
            "month",
            "month_start",
            "split",
            TARGET,
        ]
    ].reset_index(drop=True).copy()

    validation_predictions[
        "p_county_month_climatology"
    ] = validation_climatology[
        "p_county_month_climatology"
    ].to_numpy()

    test_predictions[
        "p_county_month_climatology"
    ] = test_climatology[
        "p_county_month_climatology"
    ].to_numpy()

    validation_reference_brier = (
        brier_score_loss(
            validation[TARGET],
            validation_predictions[
                "p_county_month_climatology"
            ],
        )
    )

    test_reference_brier = (
        brier_score_loss(
            test[TARGET],
            test_predictions[
                "p_county_month_climatology"
            ],
        )
    )

    validation_metrics = {
        "p_county_month_climatology": evaluate(
            validation[TARGET],
            validation_predictions[
                "p_county_month_climatology"
            ].to_numpy(),
            validation_reference_brier,
        )
    }

    test_metrics = {
        "p_county_month_climatology": evaluate(
            test[TARGET],
            test_predictions[
                "p_county_month_climatology"
            ].to_numpy(),
            test_reference_brier,
        )
    }

    model_metadata = {}

    for model_name, numeric_columns in groups.items():
        print(
            f"Training {model_name} "
            f"with {len(numeric_columns)} numeric "
            f"and {len(categorical_columns)} categorical features..."
        )

        (
            preprocessor,
            model,
            validation_probability,
            test_probability,
        ) = train_xgboost(
            train=train,
            validation=validation,
            test=test,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )

        validation_predictions[
            model_name
        ] = validation_probability

        test_predictions[
            model_name
        ] = test_probability

        validation_metrics[
            model_name
        ] = evaluate(
            validation[TARGET],
            validation_probability,
            validation_reference_brier,
        )

        test_metrics[
            model_name
        ] = evaluate(
            test[TARGET],
            test_probability,
            test_reference_brier,
        )

        model_metadata[
            model_name
        ] = {
            "numeric_feature_count": int(
                len(numeric_columns)
            ),
            "categorical_features": (
                categorical_columns
            ),
            "numeric_features": (
                numeric_columns
            ),
            "transformed_feature_count": int(
                len(
                    preprocessor
                    .get_feature_names_out()
                )
            ),
            "best_iteration": (
                int(model.best_iteration)
                if hasattr(
                    model,
                    "best_iteration",
                )
                else None
            ),
        }

    predictions = pd.concat(
        [
            validation_predictions,
            test_predictions,
        ],
        ignore_index=True,
    )

    probability_columns = [
        "p_county_month_climatology",
        *groups.keys(),
    ]

    by_year = robustness_table(
        test_predictions,
        group_column="year",
        probability_columns=probability_columns,
    )

    by_state = robustness_table(
        test_predictions,
        group_column="state",
        probability_columns=probability_columns,
    )

    summary = {
        "target": TARGET,
        "split_definition": {
            "train": "2013-2021",
            "validation": "2022",
            "test": "2023-2025",
        },
        "comparison_design": (
            "All XGBoost variants use the same train, "
            "validation, test split and hyperparameters. "
            "Feature groups are nested around a common "
            "season/geography baseline."
        ),
        "model_metadata": model_metadata,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "incremental_comparison": (
            comparison_summary(
                validation_metrics,
                test_metrics,
            )
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    by_year.to_parquet(
        YEAR_PATH,
        index=False,
    )

    by_state.to_parquet(
        STATE_PATH,
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

    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {PREDICTIONS_PATH}")
    print(f"Saved {YEAR_PATH}")
    print(f"Saved {STATE_PATH}")

    for model_name in probability_columns:
        result = test_metrics[
            model_name
        ]

        print(
            f"{model_name}: "
            f"test PR-AUC={result['pr_auc']:.4f}, "
            f"test Brier={result['brier_score']:.4f}, "
            "Brier skill vs climatology="
            f"{result['brier_skill_vs_county_month_climatology']:.4f}"
        )

    weather_result = summary[
        "incremental_comparison"
    ]["weather_beyond_fire_history"]

    print(
        "Weather beyond fire history — "
        "test Brier change="
        f"{weather_result['test_brier_change']:.6f}, "
        "test PR-AUC change="
        f"{weather_result['test_pr_auc_change']:.6f}"
    )


if __name__ == "__main__":
    main()
