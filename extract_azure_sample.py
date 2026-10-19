from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/azure")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.glob("*vm_cpu_readings*.csv.gz"))

if not files:
    raise FileNotFoundError("No Azure CPU readings file found")

source_file = files[0]

columns = [
    "timestamp",
    "resource_id",
    "min_cpu",
    "max_cpu",
    "cpu_pct",
]

selected_ids = None
sample_parts = []

total_rows = 0
invalid_rows = 0
first_timestamp = None
last_timestamp = None

# Read in chunks rather than loading the whole file into RAM.
for chunk_number, chunk in enumerate(
    pd.read_csv(
        source_file,
        compression="gzip",
        header=None,
        names=columns,
        dtype={"resource_id": "string"},
        chunksize=200_000,
    ),
    start=1,
):
    total_rows += len(chunk)

    chunk["timestamp"] = pd.to_numeric(
        chunk["timestamp"], errors="coerce"
    )
    chunk["cpu_pct"] = pd.to_numeric(
        chunk["cpu_pct"], errors="coerce"
    )

    valid = (
        chunk["resource_id"].notna()
        & chunk["timestamp"].notna()
        & chunk["timestamp"].ge(0)
        & chunk["timestamp"].mod(1).eq(0)
        & chunk["cpu_pct"].between(0, 100)
    )

    invalid_rows += int((~valid).sum())
    chunk = chunk.loc[valid].copy()

    if chunk.empty:
        continue

    chunk["timestamp"] = chunk["timestamp"].astype("int64")

    # Reproducible starting subset: first 50 valid VM IDs
    # encountered in the source file.
    if selected_ids is None:
        selected_ids = (
            chunk["resource_id"]
            .drop_duplicates()
            .head(50)
            .tolist()
        )

    current_min = int(chunk["timestamp"].min())
    current_max = int(chunk["timestamp"].max())

    first_timestamp = (
        current_min
        if first_timestamp is None
        else min(first_timestamp, current_min)
    )
    last_timestamp = (
        current_max
        if last_timestamp is None
        else max(last_timestamp, current_max)
    )

    subset = chunk[
        chunk["resource_id"].isin(selected_ids)
    ][["resource_id", "timestamp", "cpu_pct"]]

    if not subset.empty:
        sample_parts.append(subset.copy())

    print(
        f"Chunk {chunk_number}: "
        f"{total_rows:,} source rows scanned",
        flush=True,
    )

if not sample_parts:
    raise ValueError("No usable sample records found")

sample = pd.concat(sample_parts, ignore_index=True)

# Combine repeated measurements at an identical VM/timestamp.
sample = sample.groupby(
    ["resource_id", "timestamp"],
    as_index=False,
)["cpu_pct"].mean()

sample.insert(0, "provider", "azure")
sample = sample.sort_values(["resource_id", "timestamp"])

sample.to_csv(
    OUTPUT_DIR / "azure_usage_sample.csv",
    index=False,
)

pd.DataFrame({
    "resource_id": selected_ids,
}).to_csv(
    OUTPUT_DIR / "azure_selected_vm_ids.csv",
    index=False,
)

coverage = sample.groupby("resource_id").agg(
    records=("timestamp", "size"),
    first_second=("timestamp", "min"),
    last_second=("timestamp", "max"),
)

coverage["span_hours"] = (
    coverage["last_second"] - coverage["first_second"]
) / 3600

print("\n=== Source File Coverage ===")
print("File:", source_file.name)
print("Source rows:", total_rows)
print("Invalid rows excluded:", invalid_rows)
print("First second:", first_timestamp)
print("Last second:", last_timestamp)
print(
    "Source span in hours:",
    round((last_timestamp - first_timestamp) / 3600, 2),
)

print("\n=== Selected VM Sample ===")
print("VMs:", sample["resource_id"].nunique())
print("Records:", len(sample))
print(coverage.describe().round(2).to_string())

print("\nSaved: data/processed/azure_usage_sample.csv")
