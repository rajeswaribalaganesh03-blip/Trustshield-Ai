
from pathlib import Path
import pandas as pd

# Project paths
BACKEND_DIR = Path(__file__).resolve().parents[4]
PROJECT_ROOT = BACKEND_DIR.parent

DATASET_DIR = (
    PROJECT_ROOT / "datasets" / "profile" / "account"
)


def inspect_dataset(filename):

    path = DATASET_DIR / filename

    print("\n" + "=" * 60)
    print(f"FILE: {filename}")
    print("=" * 60)

    if not path.exists():
        print("File not found:", path)
        return

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nLabel distribution:")

    if "fake" in df.columns:
        print(df["fake"].value_counts(dropna=False))
    else:
        print("No 'fake' column found.")

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nFirst 3 records:")
    print(df.head(3).to_string())


def main():

    print("TRUSTSHIELD AI - DATASET SUMMARY")

    print("\nDataset location:", DATASET_DIR)

    inspect_dataset("train.csv")
    inspect_dataset("test.csv")

    print("\nSUMMARY COMPLETED")


if __name__ == "__main__":
    main()
