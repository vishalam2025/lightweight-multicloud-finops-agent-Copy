from pathlib import Path
import csv
import tarfile

import pandas as pd

ARCHIVE = Path("data/raw/alibaba/machine_usage.tar.gz")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = OUTPUT_DIR / "alibaba_usage_sample.csv"
MAX_MACHINES = 50


def main():
    selected_ids = []
    selected_set = set()

    scanned = 0
    invalid = 0
    saved = 0

    first_timestamp = None
    last_timestamp = None
    found = False

    with SAMPLE_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow([
            "provider",
            "resource_id",
            "timestamp",
            "cpu_pct",
        ])

        with tarfile.open(ARCHIVE, mode="r|gz") as tar:
            for member in tar:
                if not (
                    member.isfile()
                    and Path(member.name).name == "machine_usage.csv"
                ):
                    continue

                found = True
                stream = tar.extractfile(member)

                if stream is None:
                    raise ValueError("Could not open machine_usage.csv")

                with stream:
                    # Decode lines directly, avoiding the seekability
                    # issue encountered with Pandas and tar streams.
                    reader = csv.reader(
                        line.decode("utf-8") for line in stream
                    )

                    for row in reader:
                        scanned += 1

                        if scanned % 1_000_000 == 0:
                            output.flush()
                            print(
                                f"Scanned {scanned:,} rows; "
                                f"saved {saved:,} sample readings",
                                flush=True,
                            )

                        if len(row) != 9:
                            invalid += 1
                            continue

                        resource_id = row[0].strip()

                        try:
                            timestamp = float(row[1])
                            cpu_pct = float(row[2])
                        except ValueError:
                            invalid += 1
                            continue

                        # These comparisons also reject NaN/infinity.
                        if not (
                            resource_id
                            and 0 <= timestamp < float("inf")
                            and 0 <= cpu_pct <= 100
                        ):
                            invalid += 1
                            continue

                        first_timestamp = (
                            timestamp
                            if first_timestamp is None
                            else min(first_timestamp, timestamp)
                        )
                        last_timestamp = (
                            timestamp
                            if last_timestamp is None
                            else max(last_timestamp, timestamp)
                        )

                        # Reproducible subset: first 50 valid machine
                        # IDs encountered, without selecting by CPU.
                        if (
                            resource_id not in selected_set
                            and len(selected_ids) < MAX_MACHINES
                        ):
                            selected_ids.append(resource_id)
                            selected_set.add(resource_id)

                        if resource_id in selected_set:
                            writer.writerow([
                                "alibaba",
                                resource_id,
                                timestamp,
                                cpu_pct,
                            ])
                            saved += 1

                break

    if not found:
        raise FileNotFoundError(
            "machine_usage.csv not found inside the archive"
        )

    if saved == 0:
        raise ValueError("No valid sample readings found")

    print("\nScan complete. Combining repeated timestamps...", flush=True)

    sample = pd.read_csv(
        SAMPLE_FILE,
        dtype={"resource_id": "string"},
    )

    sample = sample.groupby(
        ["provider", "resource_id", "timestamp"],
        as_index=False,
    )["cpu_pct"].mean()

    sample = sample.sort_values(
        ["resource_id", "timestamp"]
    ).reset_index(drop=True)

    sample.to_csv(SAMPLE_FILE, index=False)

    pd.DataFrame({
        "resource_id": selected_ids,
    }).to_csv(
        OUTPUT_DIR / "alibaba_selected_machine_ids.csv",
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

    coverage.to_csv(
        OUTPUT_DIR / "alibaba_sample_coverage.csv"
    )

    gaps = (
        sample.groupby("resource_id")["timestamp"]
        .diff()
        .dropna()
    )

    print("\n=== Alibaba Source Coverage ===")
    print("Source rows scanned:", scanned)
    print("Invalid rows excluded:", invalid)
    print("First second:", first_timestamp)
    print("Last second:", last_timestamp)
    print(
        "Source span in hours:",
        round((last_timestamp - first_timestamp) / 3600, 2),
    )

    print("\n=== Selected Machine Sample ===")
    print("Machines:", sample["resource_id"].nunique())
    print("Records after combining timestamps:", len(sample))
    print(coverage.describe().round(2).to_string())

    print("\nMost common sampling gaps in seconds:")
    print(gaps.value_counts().head(10).to_string())

    print("\nPreview:")
    print(sample.head().to_string(index=False))

    print("\nSaved:", SAMPLE_FILE)


if __name__ == "__main__":
    main()
