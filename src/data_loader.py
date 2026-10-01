from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"


def load_data():
    """Load the raw energy consumption dataset."""
    df = pd.read_csv(DATA_PATH)

    # Convert Datetime column to datetime type
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    return df


def validate_data(df):
    """Perform basic validation checks on the dataset."""

    print("\n--- Dataset Information ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Time Range ---")
    print(f"Start: {df['Datetime'].min()}")
    print(f"End:   {df['Datetime'].max()}")

    print("\n--- Energy Consumption Statistics ---")
    print(df["AEP_MW"].describe())


if __name__ == "__main__":
    df = load_data()

    print("Dataset loaded successfully!")

    print("\n--- First 5 Rows ---")
    print(df.head())

    validate_data(df)