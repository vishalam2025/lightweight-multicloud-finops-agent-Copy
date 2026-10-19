from pathlib import Path
import pandas as pd

from detectors import (
    rule_based_detect,
    statistical_detect,
    ml_based_detect,
)
from evaluate import score_detector

INPUT_FILE = Path("data/processed/bitbrains_daily_usage.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OBSERVE_DAYS = 21
FUTURE_DAYS = 3
IDLE_THRESHOLD = 5.0

df = pd.read_csv(INPUT_FILE)

required_days = set(range(OBSERVE_DAYS + FUTURE_DAYS))

# Evaluate only VMs with every required calendar day available.
eligible_ids = [
    resource_id
    for resource_id, group in df.groupby("resource_id")
    if required_days.issubset(set(group["day"]))
]

if len(eligible_ids) < 2:
    raise ValueError("Need at least two VMs with complete evaluation data")

selected = df[df["resource_id"].isin(eligible_ids)].copy()

observed = selected[
    selected["day"].between(0, OBSERVE_DAYS - 1)
].copy()

future = selected[
    selected["day"].between(
        OBSERVE_DAYS,
        OBSERVE_DAYS + FUTURE_DAYS - 1,
    )
].copy()

# Proxy label: every future daily average is below 5%.
truth_map = (
    future.groupby("resource_id")["cpu_pct"]
    .apply(lambda values: bool((values < IDLE_THRESHOLD).all()))
    .to_dict()
)

# Only observed data reaches the detectors.
strategies = {
    "Rule-Based": rule_based_detect(
        observed, window=3, threshold=IDLE_THRESHOLD
    ),
    "Statistical": statistical_detect(observed, window=3),
    "ML-Based": ml_based_detect(observed),
}

# Do not silently omit missing detector decisions.
for name, flags in strategies.items():
    missing_ids = set(eligible_ids) - set(flags)

    if missing_ids:
        raise ValueError(
            f"{name} returned no decision for: {sorted(missing_ids)}"
        )

strategies["Combined (Union)"] = {
    resource_id: any(
        flags[resource_id]
        for flags in strategies.values()
    )
    for resource_id in eligible_ids
}

metric_rows = []

for name, flags in strategies.items():
    metrics = score_detector(flags, truth_map)
    metrics["strategy"] = name
    metrics["evaluated_resources"] = len(eligible_ids)
    metric_rows.append(metrics)

comparison = pd.DataFrame(metric_rows).set_index("strategy")

# Save each VM's decisions for inspection and the dashboard.
decisions = pd.DataFrame({
    "resource_id": eligible_ids,
    "future_low_cpu_proxy": [
        truth_map[resource_id] for resource_id in eligible_ids
    ],
})

for name, flags in strategies.items():
    decisions[name] = [
        flags[resource_id] for resource_id in eligible_ids
    ]

comparison.to_csv(OUTPUT_DIR / "bitbrains_detector_comparison.csv")
decisions.to_csv(
    OUTPUT_DIR / "bitbrains_detector_decisions.csv",
    index=False,
)

print("=== Evaluation Setup ===")
print("Total VMs:", df["resource_id"].nunique())
print("Eligible VMs:", len(eligible_ids))
print("Excluded VMs:", df["resource_id"].nunique() - len(eligible_ids))
print("Observation: days 0–20")
print("Future evaluation: days 21–23")
print("Future low-CPU VMs:", sum(truth_map.values()))
print("Other VMs:", len(truth_map) - sum(truth_map.values()))

print("\n=== Detector Comparison ===")
print(comparison.to_string())

if len(set(truth_map.values())) < 2:
    print(
        "\nOnly one proxy-label class is present. "
        "This evaluation cannot assess both classes."
    )

print("\nResults saved in:", OUTPUT_DIR)
