from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FinOps Resource Advisor",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def load_csv(path):
    if not path.exists():
        st.error(
            f"Required file is missing: {path.name}. "
            "Run the pipeline first."
        )
        st.stop()

    return pd.read_csv(path)


def boolean_column(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )


STATUS_LABELS = {
    "review": "Potentially idle — review",
    "priority_review": "Potentially idle — priority review",
    "investigate_usage_pattern": "Unusual usage — investigate",
    "no_low_usage_recommendation": "No low-usage flag",
    "insufficient_data": "Insufficient data",
}


def detector_evidence(decisions, resource_id):
    evidence = decisions[
        decisions["resource_id"] == resource_id
    ][
        ["rule_flag", "statistical_flag", "ml_flag"]
    ].rename(columns={
        "rule_flag": "Low-CPU rule",
        "statistical_flag": "Historical usage drop",
        "ml_flag": "Unusual low-usage behaviour",
    })

    st.dataframe(
        evidence,
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Detector agreement is supporting evidence, "
        "not a confidence probability."
    )


usage = load_csv(DATA_DIR / "bitbrains_daily_usage.csv")
decisions = load_csv(RESULTS_DIR / "hybrid_final_decisions.csv")
risk = load_csv(RESULTS_DIR / "prediction_resource_risk.csv")
forecast = load_csv(RESULTS_DIR / "prediction_daily_results.csv")
benchmark = load_csv(RESULTS_DIR / "container_benchmark_summary.csv")
detection_metrics = load_csv(
    RESULTS_DIR / "hybrid_final_comparison.csv"
)
forecast_metrics = load_csv(
    RESULTS_DIR / "prediction_error_metrics.csv"
)
forecast_classification = load_csv(
    RESULTS_DIR / "prediction_classification_metrics.csv"
)

usage["date"] = pd.to_datetime(usage["date"], utc=True)

for column in ["rule_flag", "statistical_flag", "ml_flag"]:
    decisions[column] = boolean_column(decisions[column])

for column in [
    "linear_predicted_low_cpu",
    "baseline_predicted_low_cpu",
]:
    risk[column] = boolean_column(risk[column])


def date_for_day(day):
    dates = usage.loc[usage["day"] == day, "date"]

    if dates.empty:
        return f"Day {day}"

    return dates.min().strftime("%d %b %Y")


bitbrains_reasons = {
    "review": (
        "Recent three-day average CPU is below 5%. "
        "Neither supporting detector flagged this VM."
    ),
    "priority_review": (
        "Recent three-day average CPU is below 5%, "
        "with statistical or ML supporting evidence."
    ),
    "investigate_usage_pattern": (
        "A supporting detector found a usage change or anomaly, "
        "but the low-CPU rule did not flag this VM."
    ),
    "no_low_usage_recommendation": (
        "None of the three detectors flagged this VM."
    ),
    "insufficient_data": "Not enough data for assessment.",
}

recent_mean = (
    usage[usage["day"].between(23, 25)]
    .groupby("resource_id")["cpu_pct"]
    .mean()
)

status = decisions[
    [
        "resource_id",
        "recommendation",
        "rule_flag",
        "statistical_flag",
        "ml_flag",
    ]
].copy()

status["Recent average CPU (%)"] = (
    status["resource_id"].map(recent_mean)
)
status["Status"] = status["recommendation"].map(STATUS_LABELS)
status["Why?"] = status["recommendation"].map(bitbrains_reasons)

excluded_ids = sorted(
    set(usage["resource_id"]) - set(status["resource_id"])
)

if excluded_ids:
    excluded = pd.DataFrame([
        {
            "resource_id": resource_id,
            "recommendation": "insufficient_data",
            "Status": "Not assessed",
            "Why?": "Missing required days for final evaluation.",
        }
        for resource_id in excluded_ids
    ])

    status = pd.concat([status, excluded], ignore_index=True)


st.title("FinOps Resource Advisor")
st.write(
    "Review low-utilization resources, inspect CPU forecasts, "
    "and choose a tested configuration using runtime and cost assumptions."
)

st.info(
    "This application analyzes published historical traces from "
    "Bitbrains, Microsoft Azure, and Alibaba. "
    "It does not monitor live cloud accounts. "
    "Low CPU is a review signal, not proof that a resource can be stopped."
)

st.caption(
    "Azure contains VM measurements; Alibaba contains machine "
    "measurements. Bitbrains remains a separate dataset source."
)

(
    resource_tab,
    prediction_tab,
    benchmark_tab,
    evaluation_tab,
    provider_tab,
) = st.tabs([
    "Bitbrains Resource Status",
    "Bitbrains Future Prediction",
    "Resource Benchmarking",
    "Bitbrains Evaluation",
    "Azure / Alibaba Analysis",
])


with resource_tab:
    st.subheader("Which Bitbrains VMs need review?")

    st.caption(
        f"Status assessed on {date_for_day(25)}, "
        "using observations through day 25."
    )

    broad_count = decisions["recommendation"].isin(
        ["review", "priority_review"]
    ).sum()

    a, b, c, d = st.columns(4)

    a.metric("Dataset VMs", usage["resource_id"].nunique())
    b.metric("Assessed VMs", len(decisions))
    c.metric("Potentially idle", int(broad_count))
    d.metric(
        "Priority review",
        int(decisions["recommendation"].eq("priority_review").sum()),
    )

    st.write(
        "Potentially idle means average CPU was below 5% over "
        "the last three observed days. Priority review means "
        "another detector also provided supporting evidence."
    )

    status_filter = st.selectbox(
        "Show resources",
        ["All"] + sorted(status["Status"].dropna().unique()),
        key="bitbrains_status_filter",
    )

    visible = status

    if status_filter != "All":
        visible = status[status["Status"] == status_filter]

    display = visible[
        ["resource_id", "Status", "Recent average CPU (%)", "Why?"]
    ].rename(columns={"resource_id": "Resource"})

    st.dataframe(
        display.round(2),
        hide_index=True,
        use_container_width=True,
    )

    resource_id = st.selectbox(
        "Inspect a VM",
        sorted(usage["resource_id"].unique()),
        key="bitbrains_resource",
    )

    record = status[
        status["resource_id"] == resource_id
    ].iloc[0]

    st.write(f"**Status:** {record['Status']}")
    st.write(record["Why?"])

    history = usage[
        (usage["resource_id"] == resource_id)
        & (usage["day"] <= 25)
    ].sort_values("day")

    chart = (
        history.set_index("day")[["cpu_pct"]]
        .reindex(range(26))
        .rename(columns={"cpu_pct": "Average CPU (%)"})
    )
    chart["Low-CPU threshold (%)"] = 5.0

    st.line_chart(chart)
    st.caption(
        f"Day 0 is {date_for_day(0)}. "
        "Future readings are excluded; missing days remain missing."
    )

    if resource_id in set(decisions["resource_id"]):
        with st.expander("See the three detector decisions"):
            detector_evidence(decisions, resource_id)

    st.download_button(
        "Download displayed resource status",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="bitbrains_resource_status.csv",
        mime="text/csv",
        key="bitbrains_status_download",
    )


with prediction_tab:
    st.subheader("Which Bitbrains VMs may remain or become idle?")

    st.caption(
        f"Forecast period: {date_for_day(26)} "
        f"to {date_for_day(28)}."
    )

    st.write(
        "Persistent low usage is predicted when forecast CPU "
        "is below 5% on all three future days. "
        "This includes VMs already using little CPU."
    )

    method = st.radio(
        "Forecast method",
        ["Last observed usage", "Linear trend"],
        horizontal=True,
        key="bitbrains_forecast_method",
    )

    st.caption(
        "Last observed usage repeats the final observed CPU value. "
        "Linear trend extends a straight-line trend from the history. "
        "The last-value baseline performed better on this test."
    )

    if method == "Last observed usage":
        flag_column = "baseline_predicted_low_cpu"
        cpu_column = "baseline_cpu"
    else:
        flag_column = "linear_predicted_low_cpu"
        cpu_column = "linear_cpu"

    future_summary = forecast.groupby("resource_id")[cpu_column].agg(
        ["mean", "max"]
    )

    prediction_table = risk[
        ["resource_id", flag_column]
    ].copy()

    prediction_table["Predicted average CPU (%)"] = (
        prediction_table["resource_id"].map(future_summary["mean"])
    )
    prediction_table["Highest forecast daily CPU (%)"] = (
        prediction_table["resource_id"].map(future_summary["max"])
    )

    current_flags = decisions.set_index("resource_id")["rule_flag"]
    prediction_table["current_low_cpu"] = (
        prediction_table["resource_id"].map(current_flags)
    )

    def explain_prediction(row):
        if row[flag_column]:
            if row["current_low_cpu"]:
                return "Already low usage; predicted to remain low"
            return "Predicted to become persistently low usage"

        return "Not predicted to stay below 5% on all three days"

    prediction_table["Prediction"] = prediction_table.apply(
        explain_prediction,
        axis=1,
    )

    st.metric(
        "Predicted persistent low-usage VMs",
        int(prediction_table[flag_column].sum()),
    )

    only_low = st.checkbox(
        "Show only predicted low-usage VMs",
        value=True,
        key="bitbrains_only_low",
    )

    visible_predictions = prediction_table

    if only_low:
        visible_predictions = prediction_table[
            prediction_table[flag_column]
        ]

    prediction_display = visible_predictions[
        [
            "resource_id",
            "Prediction",
            "Predicted average CPU (%)",
            "Highest forecast daily CPU (%)",
        ]
    ].rename(columns={"resource_id": "Resource"})

    st.dataframe(
        prediction_display.round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download displayed predictions",
        data=prediction_display.to_csv(index=False).encode("utf-8"),
        file_name="bitbrains_future_predictions.csv",
        mime="text/csv",
        key="bitbrains_prediction_download",
    )

    forecast_vm = st.selectbox(
        "Inspect a forecast",
        sorted(forecast["resource_id"].unique()),
        key="bitbrains_forecast_vm",
    )

    vm_forecast = forecast[
        forecast["resource_id"] == forecast_vm
    ].sort_values("day")

    st.line_chart(
        vm_forecast.set_index("day")[[cpu_column]].rename(columns={
            cpu_column: "Forecast CPU (%)",
        })
    )

    with st.expander("Compare with actual future usage"):
        st.line_chart(
            vm_forecast.set_index("day")[
                ["actual_cpu", cpu_column]
            ].rename(columns={
                "actual_cpu": "Actual CPU (%)",
                cpu_column: "Forecast CPU (%)",
            })
        )
        st.caption(
            "Actual future readings are used only for evaluation. "
            "The forecasting methods did not receive them."
        )


with benchmark_tab:
    st.subheader("Which configuration fits your workload?")

    ordered = benchmark.set_index("profile").reindex(
        ["Small", "Medium", "Large"]
    ).copy()

    if ordered["median_sec"].isna().any():
        st.error(
            "Benchmark results must contain Small, Medium, "
            "and Large profiles."
        )
    else:
        fastest = ordered["median_sec"].idxmin()

        st.success(
            f"{fastest} completed the workload fastest: "
            f"{ordered.loc[fastest, 'median_sec']:.3f} seconds."
        )

        st.write(
            "Each configuration ran the same two-worker workload "
            "with a 512 MB memory limit."
        )

        st.subheader("Cost assumptions")
        st.info(
            "These are illustrative scenario inputs, "
            "not AWS/Azure/Alibaba prices or actual charges "
            "for your local containers."
        )

        rate_a, rate_b, rate_c = st.columns(3)

        with rate_a:
            small_rate = st.number_input(
                "Small hourly rate (USD)",
                min_value=0.0001,
                value=0.01,
                step=0.001,
                format="%.4f",
            )

        with rate_b:
            medium_rate = st.number_input(
                "Medium hourly rate (USD)",
                min_value=0.0001,
                value=0.02,
                step=0.001,
                format="%.4f",
            )

        with rate_c:
            large_rate = st.number_input(
                "Large hourly rate (USD)",
                min_value=0.0001,
                value=0.04,
                step=0.001,
                format="%.4f",
            )

        target = st.number_input(
            "Maximum acceptable median runtime (seconds)",
            min_value=0.01,
            value=2.0,
            step=0.1,
            format="%.2f",
        )

        ordered["hourly_rate_usd"] = pd.Series({
            "Small": small_rate,
            "Medium": medium_rate,
            "Large": large_rate,
        })

        ordered["estimated_cost_usd"] = (
            ordered["median_sec"]
            / 3600
            * ordered["hourly_rate_usd"]
        )

        ordered["meets_target"] = ordered["median_sec"] <= target
        eligible = ordered[ordered["meets_target"]]

        if eligible.empty:
            st.warning(
                "No tested configuration meets your median-runtime "
                "target. Increase the target or benchmark "
                "another configuration."
            )
        else:
            best = (
                eligible.sort_values(
                    ["estimated_cost_usd", "median_sec"]
                ).index[0]
            )

            choice = ordered.loc[best]

            st.success(
                f"Recommended under these assumptions: {best}. "
                f"Median runtime: {choice['median_sec']:.3f} seconds. "
                f"Estimated workload cost: "
                f"${choice['estimated_cost_usd']:.8f}."
            )

        cost_table = ordered.reset_index().rename(columns={
            "profile": "Configuration",
            "cpu_limit": "CPU limit",
            "memory_mb": "Memory limit (MB)",
            "median_sec": "Median runtime (seconds)",
            "hourly_rate_usd": "Assumed hourly rate (USD)",
            "estimated_cost_usd": "Estimated workload cost (USD)",
            "meets_target": "Meets runtime target",
        })

        cost_display = cost_table[
            [
                "Configuration",
                "CPU limit",
                "Memory limit (MB)",
                "Median runtime (seconds)",
                "Assumed hourly rate (USD)",
                "Estimated workload cost (USD)",
                "Meets runtime target",
            ]
        ]

        st.dataframe(
            cost_display,
            column_config={
                "Median runtime (seconds)": (
                    st.column_config.NumberColumn(format="%.4f")
                ),
                "Assumed hourly rate (USD)": (
                    st.column_config.NumberColumn(format="$%.4f")
                ),
                "Estimated workload cost (USD)": (
                    st.column_config.NumberColumn(format="$%.8f")
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            ordered[["median_sec"]].rename(columns={
                "median_sec": "Median runtime (seconds)",
            })
        )

        st.caption(
            "Estimated cost = median runtime / 3600 × assumed hourly "
            "rate. Assumes continuous per-second charging; excludes "
            "startup, minimum billing periods, storage, and network "
            "costs. Individual runtimes may exceed the median target."
        )

        st.download_button(
            "Download cost comparison",
            data=cost_display.to_csv(index=False).encode("utf-8"),
            file_name="container_cost_scenario.csv",
            mime="text/csv",
            key="cost_download",
        )

        st.caption(
            "Five measured trials per local Docker profile. "
            "Warm-up and container startup are excluded. "
            "Results do not establish actual cloud instance performance."
        )


with evaluation_tab:
    st.subheader("How well did the Bitbrains methods perform?")

    st.write(
        "Precision: how many flagged VMs met the future low-CPU "
        "definition. Recall: how many future low-CPU VMs were found. "
        "F1 balances precision and recall."
    )

    st.dataframe(
        detection_metrics,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Broad review follows the rule baseline. Priority review "
        "requires another detector's support. If there are no "
        "positive predictions, precision is mathematically undefined "
        "but the scoring code displays zero."
    )

    st.subheader("Forecast error")
    st.write(
        "MAE is average CPU prediction error in percentage points. "
        "Lower MAE and RMSE are better."
    )

    st.dataframe(
        forecast_metrics,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Future low-usage prediction scores")
    st.dataframe(
        forecast_classification,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Final evaluation uses VMs with complete days 0–28. "
        "Observed days: 0–25. Future proxy: CPU below 5% "
        "on every day from 26–28. Results apply to this subset "
        "and period."
    )


with provider_tab:
    st.subheader("Analyze resources from different providers")

    st.write(
        "Both provider datasets use the same hybrid detection "
        "function. Provider data is analyzed separately."
    )

    provider = st.selectbox(
        "Choose provider",
        ["Azure", "Alibaba"],
        key="trace_provider",
    )

    if provider == "Azure":
        prefix = "azure"
        resource_type = "VM"
        observation_start = 0
        observation_end = 3
        future_hour = 4
        coverage_description = (
            "All 12 five-minute readings are required per hour."
        )
    else:
        prefix = "alibaba"
        resource_type = "machine"
        observation_start = 45
        observation_end = 48
        future_hour = 49
        coverage_description = (
            "At least 324 of 360 ten-second bins are required "
            "per hour: 90% coverage. The earliest five-hour "
            "window with at least 40 eligible machines was selected "
            "using coverage alone."
        )

    provider_usage = load_csv(
        RESULTS_DIR / f"{prefix}_hourly_usage.csv"
    )
    provider_decisions = load_csv(
        RESULTS_DIR / f"{prefix}_hybrid_decisions.csv"
    )
    provider_metrics = load_csv(
        RESULTS_DIR / f"{prefix}_detector_comparison.csv"
    )

    for column in ["rule_flag", "statistical_flag", "ml_flag"]:
        provider_decisions[column] = boolean_column(
            provider_decisions[column]
        )

    st.caption(
        f"{provider} {resource_type} traces. Observed hours "
        f"{observation_start}–{observation_end}; future evaluation "
        f"hour {future_hour}. Hours are relative to the trace start, "
        "not today's clock."
    )

    broad = provider_decisions["recommendation"].isin(
        ["review", "priority_review"]
    )

    first, second, third = st.columns(3)

    first.metric(
        f"Assessed {resource_type}s",
        len(provider_decisions),
    )
    second.metric("Potentially idle", int(broad.sum()))
    third.metric(
        "Priority review",
        int(
            provider_decisions["recommendation"]
            .eq("priority_review")
            .sum()
        ),
    )

    st.write(
        "Potentially idle means the last observed hourly average "
        "CPU is below 5%. Priority review means statistical or ML "
        "evidence also supports that flag."
    )

    recent_cpu = (
        provider_usage[
            provider_usage["hour"] == observation_end
        ]
        .set_index("resource_id")["cpu_pct"]
    )

    provider_reasons = {
        "review": (
            "Last observed hourly CPU is below 5%; "
            "no supporting detector flagged the resource."
        ),
        "priority_review": (
            "Last observed hourly CPU is below 5%, "
            "with supporting statistical or ML evidence."
        ),
        "investigate_usage_pattern": (
            "A supporting detector flagged a usage change or anomaly, "
            "but the low-CPU rule did not."
        ),
        "no_low_usage_recommendation": (
            "None of the three detectors flagged the resource."
        ),
        "insufficient_data": "Insufficient data for assessment.",
    }

    provider_table = provider_decisions.copy()
    provider_table["Status"] = (
        provider_table["recommendation"].map(STATUS_LABELS)
    )
    provider_table["Recent CPU (%)"] = (
        provider_table["resource_id"].map(recent_cpu)
    )
    provider_table["Reason"] = (
        provider_table["recommendation"].map(provider_reasons)
    )

    selected_status = st.selectbox(
        "Filter resource status",
        ["All"] + sorted(provider_table["Status"].dropna().unique()),
        key=f"{prefix}_status_filter",
    )

    visible_provider = provider_table

    if selected_status != "All":
        visible_provider = provider_table[
            provider_table["Status"] == selected_status
        ]

    display_table = visible_provider[
        ["resource_id", "Status", "Recent CPU (%)", "Reason"]
    ].rename(columns={"resource_id": "Resource"})

    st.dataframe(
        display_table.round(2),
        hide_index=True,
        use_container_width=True,
    )

    selected_resource = st.selectbox(
        f"Inspect a {resource_type}",
        sorted(provider_decisions["resource_id"].unique()),
        key=f"{prefix}_resource_selector",
    )

    selected_record = provider_table[
        provider_table["resource_id"] == selected_resource
    ].iloc[0]

    st.write(f"**Status:** {selected_record['Status']}")
    st.write(selected_record["Reason"])

    provider_history = provider_usage[
        (provider_usage["resource_id"] == selected_resource)
        & provider_usage["hour"].between(
            observation_start,
            observation_end,
        )
    ].sort_values("hour")

    provider_chart = (
        provider_history.set_index("hour")[["cpu_pct"]]
        .reindex(range(observation_start, observation_end + 1))
        .rename(columns={"cpu_pct": "Average CPU (%)"})
    )
    provider_chart["Low-CPU threshold (%)"] = 5.0

    st.line_chart(provider_chart)
    st.caption(
        "Observed hourly usage only. Future readings are excluded."
    )

    with st.expander("See the three detector decisions"):
        detector_evidence(
            provider_decisions,
            selected_resource,
        )

    st.download_button(
        f"Download displayed {provider} resource status",
        data=display_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{prefix}_resource_status.csv",
        mime="text/csv",
        key=f"{prefix}_status_download",
    )

    st.info(
        "CPU forecasting is currently available for Bitbrains only. "
        "The future hour here evaluates detector decisions; "
        "it is not a forecast generated for this provider."
    )

    with st.expander("Evaluation results and methodology"):
        st.write(coverage_description)
        st.write(
            "Rule/statistical recent window: one hour. "
            "ML recent feature window: three hours. "
            "Future proxy: average CPU below 5% in the next hour."
        )

        st.dataframe(
            provider_metrics,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Broad review follows the rule baseline. Priority review "
            "requires additional detector support. Samples use the "
            "first 50 resource IDs encountered in their source files, "
            "subject to coverage checks; they are not random samples. "
            "Azure and Alibaba also have different coverage rules. "
            "Scores do not establish provider-wide performance and "
            "are not directly comparable to Bitbrains daily scores."
        )
