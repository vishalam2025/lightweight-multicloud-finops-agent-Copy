import numpy as np
from sklearn.ensemble import IsolationForest


def rule_based_detect(df, window=3, threshold=5.0):
    results = {}

    if df.empty:
        return results

    max_day = df["day"].max()
    recent = df[df["day"] > max_day - window]

    grouped = recent.groupby("resource_id")["cpu_pct"].mean()

    for resource_id, avg_cpu in grouped.items():
        results[resource_id] = bool(avg_cpu < threshold)

    return results


def statistical_detect(
    df,
    window=3,
    z_threshold=-1.5,
    absolute_floor=10.0,
):
    results = {}

    if df.empty:
        return results

    max_day = df["day"].max()

    for resource_id, group in df.groupby("resource_id"):
        history = group[
            group["day"] <= max_day - window
        ]["cpu_pct"]

        recent = group[
            group["day"] > max_day - window
        ]["cpu_pct"]

        # No decision when there is insufficient history.
        if len(history) < 3 or len(recent) == 0:
            results[resource_id] = None
            continue

        historical_mean = history.mean()
        historical_std = history.std()

        if not np.isfinite(historical_std) or historical_std == 0:
            historical_std = 1e-6

        z_score = (
            recent.mean() - historical_mean
        ) / historical_std

        results[resource_id] = bool(
            z_score < z_threshold
            and recent.mean() < absolute_floor
        )

    return results


def make_features(series, recent_window=3):
    x = np.arange(len(series))

    slope = np.polyfit(x, series, 1)[0]
    recent_mean = series[-recent_window:].mean()

    return [
        series.mean(),
        series.std(),
        series.min(),
        series.max(),
        slope,
        recent_mean,
    ]


def ml_based_detect(
    df,
    contamination=0.35,
    low_usage_cutoff=15.0,
):
    results = {}
    resource_ids = []
    feature_list = []

    for resource_id, group in df.groupby("resource_id"):
        series = (
            group.sort_values("day")["cpu_pct"]
            .to_numpy(dtype=float)
        )

        if len(series) < 2:
            results[resource_id] = None
            continue

        resource_ids.append(resource_id)
        feature_list.append(make_features(series))

    # Isolation Forest needs multiple resources to compare.
    if len(feature_list) < 2:
        for resource_id in resource_ids:
            results[resource_id] = None
        return results

    features_array = np.array(feature_list)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(features_array)

    for resource_id, prediction, features in zip(
        resource_ids,
        predictions,
        feature_list,
    ):
        recent_mean = features[5]

        results[resource_id] = bool(
            prediction == -1
            and recent_mean < low_usage_cutoff
        )

    return results


def hybrid_detect(df, window=3):
    """
    Run all three detectors and combine their decisions.

    window uses the input's time units:
    days for Bitbrains, hours for the Azure experiment.

    ML retains its existing three-observation feature window.
    """
    rule = rule_based_detect(df, window=window)
    statistical = statistical_detect(df, window=window)
    ml = ml_based_detect(df)

    results = {}

    for resource_id in df["resource_id"].unique():
        r = rule.get(resource_id)
        s = statistical.get(resource_id)
        m = ml.get(resource_id)

        r = None if r is None else bool(r)
        s = None if s is None else bool(s)
        m = None if m is None else bool(m)

        if r is None:
            recommendation = "insufficient_data"
        elif r and (s is True or m is True):
            recommendation = "priority_review"
        elif r:
            recommendation = "review"
        elif s is True or m is True:
            recommendation = "investigate_usage_pattern"
        else:
            recommendation = "no_low_usage_recommendation"

        results[resource_id] = {
            "recommendation": recommendation,
            "rule_flag": r,
            "statistical_flag": s,
            "ml_flag": m,
        }

    return results
