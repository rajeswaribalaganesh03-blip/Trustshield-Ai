
from pathlib import Path
import pandas as pd

from sklearn.impute import SimpleImputer


BACKEND_DIR = Path(__file__).resolve().parents[4]
PROJECT_ROOT = BACKEND_DIR.parent

DATASET_DIR = (
    PROJECT_ROOT / "datasets" / "profile" / "account"
)

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

TARGET_COLUMN = "fake"


def load_dataset(filename):

    path = DATASET_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def validate_dataset(df, require_target=True):

    required = FEATURE_COLUMNS.copy()

    if require_target:
        required.append(TARGET_COLUMN)

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def prepare_features(df):

    validate_dataset(df)

    X = df[FEATURE_COLUMNS].copy()

    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(
            X[col], errors="coerce"
        )

    y = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    )

    if y.isna().any():
        raise ValueError("Target column contains missing labels.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 (genuine) and 1 (fake)."
        )

    return X, y.astype(int)


def build_imputer():

    return SimpleImputer(strategy="median")
