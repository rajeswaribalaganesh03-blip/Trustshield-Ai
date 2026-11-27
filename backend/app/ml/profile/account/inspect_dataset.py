
from pathlib import Path
import pandas as pd

# Project paths
BACKEND_DIR = Path(__file__).resolve().parents[4]
PROJECT_ROOT = BACKEND_DIR.parent

DATASET_DIR = (
    PROJECT_ROOT / "datasets" / "profile" / "account"
)


def inspect_file(filename):

    file_path = DATASET_DIR / filename

    print("\n" + "=" * 70)
    print(f"INSPECTING FILE: {filename}")
    print("=" * 70)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)

        print("\nDATASET SHAPE:")
        print(df.shape)

        print("\nCOLUMN NAMES:")
        print(df.columns.tolist())

        print("\nDATA TYPES:")
        print(df.dtypes)

        print("\nFIRST FIVE ROWS:")
        print(df.head().to_string())

        print("\nMISSING VALUES:")
        print(df.isnull().sum())

        print("\nDUPLICATE ROWS:")
        print(df.duplicated().sum())

        print("\nUNIQUE VALUES:")

        for col in df.columns:
            print(f"{col}: {df[col].nunique(dropna=False)}")

        print("\nPOTENTIAL TARGET COLUMNS:")

        for col in df.columns:
            if df[col].nunique(dropna=False) <= 20:
                print(f"\nColumn: {col}")
                print(df[col].value_counts(dropna=False))

    except Exception as error:
        print(f"ERROR READING FILE: {error}")


def main():

    print("\nTRUSTSHIELD AI")
    print("INSTAGRAM FAKE ACCOUNT DATASET INSPECTION")

    print("\nBackend directory:", BACKEND_DIR)
    print("Project directory:", PROJECT_ROOT)
    print("Dataset directory:", DATASET_DIR)

    inspect_file("train.csv")
    inspect_file("test.csv")

    print("\nINSPECTION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
