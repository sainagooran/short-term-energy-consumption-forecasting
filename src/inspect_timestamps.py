
import pandas as pd

from data_loader import load_data


df = load_data()
df = df.sort_values("Datetime").reset_index(drop=True)

# Find timestamps that occur more than once
duplicate_times = (
    df.loc[df["Datetime"].duplicated(keep=False), "Datetime"]
    .drop_duplicates()
)

print("Number of duplicated timestamps:", len(duplicate_times))

for timestamp in duplicate_times:
    print(f"\n--- Around {timestamp} ---")

    start = timestamp - pd.Timedelta(hours=2)
    end = timestamp + pd.Timedelta(hours=2)

    nearby_rows = df.loc[
        df["Datetime"].between(start, end),
        ["Datetime", "AEP_MW"],
    ]

    print(nearby_rows.to_string(index=False))


# Check gaps larger than one hour
time_differences = df["Datetime"].diff()
large_gaps = df.loc[
    time_differences > pd.Timedelta(hours=1),
    "Datetime",
]

print("\nNumber of gaps larger than one hour:", len(large_gaps))

for timestamp in large_gaps.head(10):
    previous_rows = df.loc[df["Datetime"] < timestamp, "Datetime"]

    if not previous_rows.empty:
        previous_timestamp = previous_rows.iloc[-1]
        difference = timestamp - previous_timestamp

        print(
            f"{previous_timestamp} -> {timestamp} "
            f"(gap: {difference})"
        )



# Summarize the time intervals
interval_counts = df["Datetime"].diff().value_counts().sort_index()

print("\n--- Time interval summary ---")
print(interval_counts.to_string())

