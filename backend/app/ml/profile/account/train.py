
from pathlib import Path
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import (
    load_dataset,
    prepare_features,
    validate_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_imputer
)

from evaluation import evaluate_model


BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "random_forest.joblib"

METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

REPORT_PATH = ARTIFACT_DIR / "evaluation_report.json"


def main():

    print("\nTRUSTSHIELD AI - RANDOM FOREST TRAINING")

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. Load training data

    print("\nLoading training dataset...")

    df = load_dataset("train.csv")

    X, y = prepare_features(df)

    print("Training dataset shape:", df.shape)

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # 2. Stratified validation split

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))

    # 3. Build model pipeline

    model = Pipeline([
        ("imputer", build_imputer()),

        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 4. Train on training partition

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    # 5. Evaluate validation partition

    print("\nEvaluating validation data...")

    val_predictions = model.predict(X_val)

    validation_results = evaluate_model(
        y_val,
        val_predictions
    )

    # 6. Refit final model on all labelled training data

    print("\nTraining final model on complete training dataset...")

    model.fit(X, y)

    # 7. Evaluate supplied test.csv if labels exist

    test_results = None

    test_df = load_dataset("test.csv")

    if TARGET_COLUMN in test_df.columns:

        print("\nEvaluating supplied test.csv...")

        validate_dataset(test_df)

        X_test, y_test = prepare_features(test_df)

        test_predictions = model.predict(X_test)

        test_results = evaluate_model(
            y_test,
            test_predictions
        )

    else:

        print(
            "\nTest dataset has no target column. "
            "External test evaluation skipped."
        )

    # 8. Save model

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "RandomForestClassifier",
        "target_column": TARGET_COLUMN,
        "class_mapping": {
            "0": "genuine",
            "1": "fake"
        },
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": len(df),
        "validation_rows": len(X_val),
        "random_state": 42,
        "validation_split": 0.20
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    # 9. Save evaluation report

    report = {
        "validation": validation_results,
        "official_test": test_results
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    # 10. Display results

    print("\n========== VALIDATION RESULTS ==========")

    print(json.dumps(
        validation_results,
        indent=4
    ))

    if test_results is not None:

        print("\n========== OFFICIAL TEST RESULTS ==========")

        print(json.dumps(
            test_results,
            indent=4
        ))

    print("\nMODEL TRAINING COMPLETED")

    print("Saved model:", MODEL_PATH)
    print("Saved metadata:", METADATA_PATH)
    print("Saved report:", REPORT_PATH)


if __name__ == "__main__":
    main()
