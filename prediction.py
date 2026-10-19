from pathlib import Path
import numpy as np
import pandas as pd


def forecast_cpu(observed, target_days):
    rows = []

    for resource_id, group in observed.groupby("resource_id"):
        group = group.sort_values("day")

        if len(group) < 3:
            continue

        x = group["day"].to_numpy(dtype=float)
        y = group["cpu_pct"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        last_cpu = float(y[-1])

        for day in target_days:
            linear_cpu = np.clip(slope * day + intercept, 0, 100)

            rows.append({
                "resource_id": resource_id,
                "day": int(day),
                "linear_cpu": float(linear_cpu),
                "baseline_cpu": last_cpu,
                "trend_slope": float(slope),
            })

    return pd.DataFrame(rows)


def evaluate_forecasts(predictions, future):
    actual = future[
        ["resource_id", "day", "cpu_pct"]
    ].rename(columns={"cpu_pct": "actual_cpu"})

    scored = predictions.merge(
        actual,
        on=["resource_id", "day"],
        how="left",
        validate="one_to_one",
    )

    if scored.empty or scored["actual_cpu"].isna().any():
        raise ValueError("Missing actual CPU values for forecast evaluation")

    metric_rows = []

    for name, column in [
        ("Linear trend", "linear_cpu"),
        ("Last-value baseline", "baseline_cpu"),
    ]:
        errors = scored[column] - scored["actual_cpu"]

        metric_rows.append({
            "method": name,
            "MAE_percentage_points": float(errors.abs().mean()),
            "RMSE_percentage_points": float(
                np.sqrt((errors ** 2).mean())
            ),
            "evaluated_predictions": len(scored),
        })

    metrics = pd.DataFrame(metric_rows).set_index("method")
    return scored, metrics


def main():
    df = pd.read_csv("data/processed/bitbrains_daily_usage.csv")

    required_days = set(range(29))

    eligible_ids = [
        rid
        for rid, group in df.groupby("resource_id")
        if required_days.issubset(set(group["day"]))
    ]

    if not eligible_ids:
        raise ValueError("No VMs have complete days 0–28")

    selected = df[df["resource_id"].isin(eligible_ids)].copy()

    observed = selected[selected["day"].between(0, 25)].copy()
    future = selected[selected["day"].between(26, 28)].copy()

    predictions = forecast_cpu(observed, target_days=[26, 27, 28])
    scored, metrics = evaluate_forecasts(predictions, future)

    # Use the same persistent low-CPU definition as detection evaluation:
    # all three daily CPU values must be below 5%.
    risk_rows = []

    for rid, group in scored.groupby("resource_id"):
        risk_rows.append({
            "resource_id": rid,
            "linear_predicted_low_cpu": bool(
                (group["linear_cpu"] < 5.0).all()
            ),
            "baseline_predicted_low_cpu": bool(
                (group["baseline_cpu"] < 5.0).all()
            ),
            "actual_future_low_cpu_proxy": bool(
                (group["actual_cpu"] < 5.0).all()
            ),
        })

    risk = pd.DataFrame(risk_rows)

    # Reuse your existing classification scoring function.
    from evaluate import score_detector

    truth = risk.set_index("resource_id")[
        "actual_future_low_cpu_proxy"
    ].to_dict()

    classification_rows = []

    for name, column in [
        ("Linear trend", "linear_predicted_low_cpu"),
        ("Last-value baseline", "baseline_predicted_low_cpu"),
    ]:
        flags = risk.set_index("resource_id")[column].to_dict()
        row = score_detector(flags, truth)
        row["method"] = name
        classification_rows.append(row)

    classification = pd.DataFrame(
        classification_rows
    ).set_index("method")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    scored.to_csv(
        output_dir / "prediction_daily_results.csv",
        index=False,
    )
    metrics.to_csv(output_dir / "prediction_error_metrics.csv")
    risk.to_csv(output_dir / "prediction_resource_risk.csv", index=False)
    classification.to_csv(
        output_dir / "prediction_classification_metrics.csv"
    )

    print("=== Prediction Setup ===")
    print("Eligible VMs:", len(eligible_ids))
    print("Observation: days 0–25")
    print("Forecast: days 26–28")

    print("\n=== CPU Forecast Errors ===")
    print(metrics.round(4).to_string())

    print("\n=== Future Low-CPU Classification ===")
    print(classification.to_string())

    print("\nSaved prediction results in:", output_dir)


if __name__ == "__main__":
    main()
