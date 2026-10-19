from pathlib import Path

import pandas as pd

from detectors import hybrid_detect
from evaluate import score_detector

INPUT_FILE = Path("data/processed/azure_usage_sample.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"resource_id": "string"},
    )

    sample_vm_count = df["resource_id"].nunique()

    # Use the first five complete trace hours.
    # Azure timestamps are elapsed seconds, not calendar dates.
    df = df[df["timestamp"].between(0, 17999)].copy()

    df["hour"] = df["timestamp"] // 3600
    df["slot"] = df["timestamp"] // 300

    # Average repeated readings within each five-minute slot.
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

    # Require all 12 five-minute slots in each hour.
    hourly = hourly[hourly["sample_count"] == 12].copy()

    required_hours = set(range(5))

    eligible_ids = [
        resource_id
        for resource_id, group in hourly.groupby("resource_id")
        if required_hours.issubset(set(group["hour"]))
    ]

    if len(eligible_ids) < 2:
        raise ValueError(
            "Need at least two VMs with five complete trace hours"
        )

    selected = hourly[
        hourly["resource_id"].isin(eligible_ids)
    ].copy()

    # Compatibility with the existing detector interface:
    # 'day' contains hourly indices in this Azure experiment.
    observed = selected[
        selected["hour"].between(0, 3)
    ].rename(columns={"hour": "day"})

    # Future data is used only to derive evaluation labels.
    future = selected[selected["hour"] == 4].copy()

    truth = (
        future.set_index("resource_id")["cpu_pct"]
        .lt(5.0)
        .to_dict()
    )

    # Shared hybrid algorithm:
    # Rule/statistical recent window = one hour.
    # ML retains its existing three-observation feature window.
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
            "provider": "azure",
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

    selected.insert(0, "provider", "azure")

    selected.to_csv(
        OUTPUT_DIR / "azure_hourly_usage.csv",
        index=False,
    )

    decisions.to_csv(
        OUTPUT_DIR / "azure_hybrid_decisions.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "azure_detector_comparison.csv"
    )

    print("=== Azure Evaluation ===")
    print("Sample VMs:", sample_vm_count)
    print("Eligible VMs:", len(eligible_ids))
    print("Excluded VMs:", sample_vm_count - len(eligible_ids))
    print("Observation: trace hours 0–3")
    print("Future evaluation: trace hour 4")
    print("Rule/statistical recent window: 1 hour")
    print("ML recent feature window: 3 hours")
    print("Future low-CPU VMs:", sum(truth.values()))
    print("Other VMs:", len(truth) - sum(truth.values()))

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
            "Both classes cannot be assessed."
        )

    print("\nSaved Azure results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
