from pathlib import Path

import pandas as pd

from detectors import hybrid_detect
from evaluate import score_detector

INPUT_FILE = Path("data/processed/alibaba_usage_sample.csv")
OUTPUT_DIR = Path("results")

OBSERVATION_START = 45
OBSERVATION_END = 48
FUTURE_HOUR = 49

MIN_BINS_PER_HOUR = 324  # 90% of 360 ten-second bins
IDLE_THRESHOLD = 5.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"resource_id": "string"},
    )

    sample_count = df["resource_id"].nunique()

    # This window was selected using coverage, not detector scores.
    df = df[
        df["timestamp"].ge(OBSERVATION_START * 3600)
        & df["timestamp"].lt((FUTURE_HOUR + 1) * 3600)
    ].copy()

    df["hour"] = (df["timestamp"] // 3600).astype(int)
    df["slot"] = (df["timestamp"] // 10).astype(int)

    # Give each occupied ten-second bin equal weight.
    slots = df.groupby(
        ["resource_id", "hour", "slot"],
        as_index=False,
    )["cpu_pct"].mean()

    hourly = slots.groupby(
        ["resource_id", "hour"],
        as_index=False,
    ).agg(
        cpu_pct=("cpu_pct", "mean"),
        sample_count=("slot", "nunique"),
    )

    hourly["coverage_pct"] = hourly["sample_count"] / 360 * 100

    # Missing intervals are never filled with zero CPU.
    hourly = hourly[
        hourly["sample_count"] >= MIN_BINS_PER_HOUR
    ].copy()

    required_hours = set(
        range(OBSERVATION_START, FUTURE_HOUR + 1)
    )

    eligible_ids = [
        resource_id
        for resource_id, group in hourly.groupby("resource_id")
        if required_hours.issubset(set(group["hour"]))
    ]

    if len(eligible_ids) < 2:
        raise ValueError(
            "Need at least two machines with sufficient "
            "coverage in every hour from 45–49"
        )

    selected = hourly[
        hourly["resource_id"].isin(eligible_ids)
    ].copy()

    # Compatibility with the detector interface:
    # 'day' represents trace-hour indices in this experiment.
    observed = selected[
        selected["hour"].between(
            OBSERVATION_START,
            OBSERVATION_END,
        )
    ].rename(columns={"hour": "day"})

    # Future readings are used only for evaluation labels.
    future = selected[
        selected["hour"] == FUTURE_HOUR
    ].copy()

    truth = (
        future.set_index("resource_id")["cpu_pct"]
        .lt(IDLE_THRESHOLD)
        .to_dict()
    )

    # Same shared hybrid function used by Azure and Bitbrains.
    hybrid_results = hybrid_detect(observed, window=1)

    records = []

    for resource_id in eligible_ids:
        result = hybrid_results[resource_id]

        for field in [
            "rule_flag",
            "statistical_flag",
            "ml_flag",
        ]:
            if result[field] is None:
                raise ValueError(
                    f"{resource_id}: missing decision for {field}"
                )

        records.append({
            "provider": "alibaba",
            "resource_type": "machine",
            "resource_id": resource_id,
            **result,
            "future_low_cpu_proxy": bool(truth[resource_id]),
        })

    decisions = pd.DataFrame(records)
    indexed = decisions.set_index("resource_id")

    strategies = {
        "Rule baseline": indexed["rule_flag"].to_dict(),

        "Statistical baseline": (
            indexed["statistical_flag"].to_dict()
        ),

        "ML baseline": indexed["ml_flag"].to_dict(),

        "Hybrid broad review": {
            record["resource_id"]: record["recommendation"]
            in {"review", "priority_review"}
            for record in records
        },

        "Hybrid priority review": {
            record["resource_id"]: record["recommendation"]
            == "priority_review"
            for record in records
        },
    }

    metric_rows = []

    for name, flags in strategies.items():
        metrics = score_detector(flags, truth)
        metrics["strategy"] = name
        metric_rows.append(metrics)

    comparison = pd.DataFrame(
        metric_rows
    ).set_index("strategy")

    selected.insert(0, "provider", "alibaba")
    selected.insert(1, "resource_type", "machine")

    selected = selected.sort_values(
        ["resource_id", "hour"]
    )

    selected.to_csv(
        OUTPUT_DIR / "alibaba_hourly_usage.csv",
        index=False,
    )

    decisions.to_csv(
        OUTPUT_DIR / "alibaba_hybrid_decisions.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "alibaba_detector_comparison.csv"
    )

    print("=== Alibaba Evaluation ===")
    print("Sample machines:", sample_count)
    print("Eligible machines:", len(eligible_ids))
    print("Excluded machines:", sample_count - len(eligible_ids))
    print(
        f"Observation: trace hours "
        f"{OBSERVATION_START}–{OBSERVATION_END}"
    )
    print(f"Future evaluation: trace hour {FUTURE_HOUR}")
    print("Rule/statistical recent window: 1 hour")
    print("ML recent feature window: 3 hours")
    print("Coverage requirement: at least 324 bins per hour")
    print("Future low-CPU machines:", sum(truth.values()))
    print("Other machines:", len(truth) - sum(truth.values()))

    print("\n=== Detector Comparison ===")
    print(comparison.to_string())

    print("\n=== Hybrid Recommendation Counts ===")
    print(
        decisions["recommendation"]
        .value_counts()
        .to_string()
    )

    if len(set(truth.values())) < 2:
        print(
            "\nOnly one proxy-label class is present. "
            "This window cannot assess performance on both classes. "
            "Zero-denominator metrics are displayed as zero."
        )

    print("\nSaved Alibaba results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
