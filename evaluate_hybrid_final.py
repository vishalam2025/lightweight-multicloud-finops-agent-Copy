from pathlib import Path
import pandas as pd

from detectors import hybrid_detect
from evaluate import score_detector

INPUT_FILE = Path("data/processed/bitbrains_daily_usage.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Require complete observation and future evaluation periods.
required_days = set(range(29))

eligible_ids = [
    rid
    for rid, group in df.groupby("resource_id")
    if required_days.issubset(set(group["day"]))
]

if len(eligible_ids) < 2:
    raise ValueError("Need at least two VMs with complete days 0–28")

selected = df[df["resource_id"].isin(eligible_ids)].copy()

# Only these observations enter the hybrid algorithm.
observed = selected[selected["day"].between(0, 25)].copy()

# Reserve these days for evaluation only.
future = selected[selected["day"].between(26, 28)].copy()

truth = (
    future.groupby("resource_id")["cpu_pct"]
    .apply(lambda values: bool((values < 5.0).all()))
    .to_dict()
)

hybrid_results = hybrid_detect(observed)

# All eligible VMs must receive all three detector decisions.
for rid in eligible_ids:
    record = hybrid_results[rid]

    for field in ["rule_flag", "statistical_flag", "ml_flag"]:
        if record[field] is None:
            raise ValueError(f"{rid}: missing decision for {field}")

strategies = {
    "Rule baseline": {
        rid: hybrid_results[rid]["rule_flag"]
        for rid in eligible_ids
    },
    "Statistical baseline": {
        rid: hybrid_results[rid]["statistical_flag"]
        for rid in eligible_ids
    },
    "ML baseline": {
        rid: hybrid_results[rid]["ml_flag"]
        for rid in eligible_ids
    },
    "Hybrid broad review": {
        rid: hybrid_results[rid]["recommendation"]
        in {"review", "priority_review"}
        for rid in eligible_ids
    },
    "Hybrid priority review": {
        rid: hybrid_results[rid]["recommendation"]
        == "priority_review"
        for rid in eligible_ids
    },
}

rows = []

for name, flags in strategies.items():
    metrics = score_detector(flags, truth)
    metrics["strategy"] = name
    metrics["evaluated_resources"] = len(eligible_ids)
    rows.append(metrics)

comparison = pd.DataFrame(rows).set_index("strategy")

decisions = pd.DataFrame([
    {
        "resource_id": rid,
        **hybrid_results[rid],
        "future_low_cpu_proxy": truth[rid],
    }
    for rid in eligible_ids
])

comparison.to_csv(OUTPUT_DIR / "hybrid_final_comparison.csv")
decisions.to_csv(
    OUTPUT_DIR / "hybrid_final_decisions.csv",
    index=False,
)

print("=== Final Evaluation Setup ===")
print("Total VMs:", df["resource_id"].nunique())
print("Eligible VMs:", len(eligible_ids))
print("Excluded VMs:", df["resource_id"].nunique() - len(eligible_ids))
print("Observation: days 0–25")
print("Future evaluation: days 26–28")
print("Future low-CPU VMs:", sum(truth.values()))
print("Other VMs:", len(truth) - sum(truth.values()))

print("\n=== Final Comparison ===")
print(comparison.to_string())

print("\n=== Hybrid Recommendation Counts ===")
print(decisions["recommendation"].value_counts().to_string())

if len(set(truth.values())) < 2:
    print(
        "\nOnly one proxy-label class is present. "
        "Both classes cannot be assessed."
    )

print("\nSaved results in:", OUTPUT_DIR)
