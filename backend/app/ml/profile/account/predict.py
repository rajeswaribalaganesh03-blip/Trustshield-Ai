
from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "random_forest.joblib"

FEATURE_COLUMNS = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows"
]

LABEL_MAPPING = {
    0: "genuine",
    1: "fake"
}


class AccountDetectionService:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, account_data: dict):

        missing = [
            feature for feature in FEATURE_COLUMNS
            if feature not in account_data
        ]

        if missing:
            raise ValueError(
                f"Missing account features: {missing}"
            )

        # Prepare input in the exact training feature order
        input_data = pd.DataFrame(
            [[account_data[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS
        )

        # Convert values to numeric
        for col in FEATURE_COLUMNS:
            input_data[col] = pd.to_numeric(
                input_data[col],
                errors="coerce"
            )

        # Reject invalid or non-finite input
        if input_data.isna().all(axis=1).iloc[0]:
            raise ValueError("All account features are invalid.")

        import numpy as np

        if np.isinf(input_data.to_numpy(dtype=float)).any():
            raise ValueError("Infinite feature values are not allowed.")

        # Prediction
        prediction = int(self.model.predict(input_data)[0])

        probabilities = self.model.predict_proba(input_data)[0]

        class_probabilities = {
            LABEL_MAPPING[int(label)]: round(float(prob), 4)
            for label, prob in zip(
                self.model.classes_,
                probabilities
            )
        }

        return {
            "module": "profile_account",
            "prediction": LABEL_MAPPING[prediction],
            "label": prediction,
            "confidence": round(
                float(max(probabilities)), 4
            ),
            "class_probabilities": class_probabilities,
            "evidence": account_data,
            "model": "random_forest"
        }


# Reusable prediction function
_service = None


def predict_account(account_data: dict):

    global _service

    if _service is None:
        _service = AccountDetectionService()

    return _service.predict(account_data)


if __name__ == "__main__":

    sample_account = {
        "profile pic": 1,
        "nums/length username": 0.33,
        "fullname words": 1,
        "nums/length fullname": 0.33,
        "name==username": 1,
        "description length": 30,
        "external URL": 0,
        "private": 0,
        "#posts": 35,
        "#followers": 488,
        "#follows": 604
    }

    result = predict_account(sample_account)

    print("\nACCOUNT PREDICTION RESULT")
    print(result)
