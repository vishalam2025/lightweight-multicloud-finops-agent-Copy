import pandas as pd

INPUT_FILE = "data/processed/alibaba_usage_sample.csv"

MIN_BINS_PER_HOUR = 324  # 90% of 360 ten-second bins
MIN_MACHINES = 40
WINDOW_HOURS = 5


def main():
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"resource_id": "string"},
    )

    df["hour"] = (df["timestamp"] // 3600).astype(int)
    df["slot"] = (df["timestamp"] // 10).astype(int)

    counts = (
        df.groupby(["resource_id", "hour"])["slot"]
        .nunique()
        .unstack("hour")
    )

    # Preserve gaps as zero occupied bins for coverage checks.
    first_hour = int(df["hour"].min())
    last_hour = int(df["hour"].max())

    counts = counts.reindex(
        columns=range(first_hour, last_hour + 1)
    ).fillna(0)

    print("=== Occupied bins in hours 10–14 ===")
    print(
        counts.reindex(columns=range(10, 15), fill_value=0)
        .describe()
        .round(1)
        .to_string()
    )

    qualified = counts >= MIN_BINS_PER_HOUR

    print("\n=== Machines meeting 90% coverage per hour ===")
    per_hour = qualified.sum().rename("eligible_machines")
    print(per_hour.to_string())

    windows = []

    for start in range(
        first_hour,
        last_hour - WINDOW_HOURS + 2,
    ):
        end = start + WINDOW_HOURS - 1

        eligible = qualified.loc[:, start:end].all(axis=1)

        windows.append({
            "start_hour": start,
            "end_hour": end,
            "eligible_machines": int(eligible.sum()),
        })

    windows = pd.DataFrame(
        windows,
        columns=["start_hour", "end_hour", "eligible_machines"],
    )

    print(
        "\n=== Earliest five-hour windows "
        "with at least 40 machines ==="
    )

    usable = windows[
        windows["eligible_machines"] >= MIN_MACHINES
    ]

    if usable.empty:
        print("No window meets the requirement.")

        if not windows.empty:
            print(
                "Maximum eligible machines:",
                windows["eligible_machines"].max(),
            )
        return

    print(usable.head(10).to_string(index=False))

    # Choose using coverage only, not CPU values or detector scores.
    chosen = usable.iloc[0]
    start = int(chosen["start_hour"])
    end = int(chosen["end_hour"])

    selected_ids = qualified.index[
        qualified.loc[:, start:end].all(axis=1)
    ].tolist()

    print("\n=== Chosen Evaluation Window ===")
    print("Observation hours:", start, "to", start + 3)
    print("Future evaluation hour:", end)
    print("Eligible machines:", len(selected_ids))
    print("Coverage requirement: at least 324 bins per hour")
    print("Selection rule: earliest window with at least 40 machines")


if __name__ == "__main__":
    main()
