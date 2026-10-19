from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/bitbrains")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

daily_frames = []

for file in files:
    df = pd.read_csv(file, sep=";")
    df.columns = df.columns.str.strip()

    required_columns = ["Timestamp [ms]", "CPU usage [%]"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file.name}: missing columns {missing_columns}"
        )

    timestamps = pd.to_numeric(
        df["Timestamp [ms]"], errors="coerce"
    )
    cpu = pd.to_numeric(
        df["CPU usage [%]"], errors="coerce"
    )

    # These traces use Unix seconds despite the [ms] heading.
    clean = pd.DataFrame({
        "timestamp": pd.to_datetime(
            timestamps,
            unit="s",
            utc=True,
            errors="coerce",
        ),
        "cpu_pct": cpu,
    })

    # Remove missing timestamps and invalid CPU measurements.
    valid = (
        clean["timestamp"].notna()
        & clean["cpu_pct"].between(0, 100)
    )
    clean = clean.loc[valid].copy()

    if clean.empty:
        print(f"Skipped {file.name}: no valid samples")
        continue

    valid_sample_count = len(clean)

    # Average measurements sharing an identical timestamp.
    clean = (
        clean.groupby("timestamp", as_index=False)["cpu_pct"]
        .mean()
        .sort_values("timestamp")
    )

    # Average measurements within each five-minute bin.
    # Bins without measurements remain NaN.
    five_minute = (
        clean.set_index("timestamp")["cpu_pct"]
        .resample("5min")
        .mean()
    )

    # Give each occupied five-minute bin equal weight.
    daily = (
        five_minute.resample("1D")
        .agg(["mean", "count"])
        .rename(columns={
            "mean": "cpu_pct",
            "count": "sample_count",
        })
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )

    # A full UTC day contains 288 five-minute bins.
    # Keep days with at least 90% coverage.
    daily = daily[daily["sample_count"] >= 260].copy()
    daily["resource_id"] = f"bitbrains-{file.stem}"

    print(
        f"{file.name}: {valid_sample_count} valid samples, "
        f"{len(daily)} retained days"
    )

    if not daily.empty:
        daily_frames.append(daily)

if not daily_frames:
    raise ValueError(
        "No days meet the minimum coverage requirement"
    )

result = pd.concat(daily_frames, ignore_index=True)

# Use one shared calendar origin across all VMs.
first_date = result["date"].min()
result["day"] = (result["date"] - first_date).dt.days

result = result[
    ["resource_id", "date", "day", "cpu_pct", "sample_count"]
].sort_values(["resource_id", "day"]).reset_index(drop=True)

output_file = OUTPUT_DIR / "bitbrains_daily_usage.csv"
result.to_csv(output_file, index=False)

print("\n=== Dataset Summary ===")
print("Resources:", result["resource_id"].nunique())
print("Daily rows:", len(result))
print(
    "Date range:",
    result["date"].min(),
    "to",
    result["date"].max(),
)

print("\nRetained days per resource:")
print(result.groupby("resource_id").size().describe())

print("\nOccupied five-minute bins per retained day:")
print(result["sample_count"].describe())

print("\nPreview:")
print(result.head().to_string(index=False))

print("\nSaved:", output_file)
