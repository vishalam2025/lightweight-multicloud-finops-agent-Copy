from pathlib import Path
import pandas as pd

for filename in ["1.csv", "2.csv", "41.csv"]:
    path = Path("data/raw/bitbrains") / filename

    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()

    timestamps = (
        pd.to_numeric(df["Timestamp [ms]"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    gaps = timestamps.diff().dropna()

    dates = pd.to_datetime(timestamps, unit="s", utc=True)
    daily_counts = dates.dt.floor("D").value_counts().sort_index()

    print(f"\n=== {filename} ===")
    print("Start:", dates.min())
    print("End:", dates.max())

    print("\nMost common sampling gaps in seconds:")
    print(gaps.value_counts().head(5).to_string())

    print("\nDaily sample counts:")
    print(daily_counts.describe().to_string())

    print("\nDays with the most samples:")
    print(daily_counts.nlargest(3).to_string())
