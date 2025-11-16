"""Streamlit dashboard for the Healthcare Sustainability AI Agent System."""
import os
import tempfile
from typing import Optional

import ast
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import re

# Ensure required directories exist before importing orchestrator (prevents logging errors)
os.makedirs("logs", exist_ok=True)

from main_workflow import SustainabilityAgentOrchestrator

st.set_page_config(
    page_title="Healthcare Sustainability Dashboard",
    page_icon=":leaves:",
    layout="wide",
)

st.title("Healthcare Sustainability AI Agent System")
st.caption("Interactive dashboard powered by the multi-agent workflow")

SAMPLE_DATA_CANDIDATES = [
    "data/multi_hospital_energy.csv",
    "data/hospital_energy.csv",
]


def _resolve_sample_data_path() -> Optional[str]:
    for candidate in SAMPLE_DATA_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


SAMPLE_DATA_PATH = _resolve_sample_data_path()

# Persist workflow results across Streamlit reruns.
if "workflow_results" not in st.session_state:
    st.session_state["workflow_results"] = None


def _write_temp_csv(upload) -> Optional[str]:
    """Write the uploaded file to a temporary CSV so orchestrator can ingest it."""
    if upload is None:
        return None

    upload.seek(0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.write(upload.getbuffer())
    temp_file.close()
    return temp_file.name


def _get_first_existing_column(data_frame: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name found in the provided candidates."""
    for column in candidates:
        if column in data_frame.columns:
            return column
    return None


def _format_metric_value(value: Optional[float], suffix: str = "") -> str:
    """Format numeric values for display in metric cards."""
    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if abs(numeric_value) >= 1000:
        formatted = f"{numeric_value:,.0f}"
    else:
        formatted = f"{numeric_value:,.2f}".rstrip("0").rstrip(".")

    return f"{formatted}{suffix}"


def _format_delta(change: Optional[float]) -> Optional[str]:
    if change is None:
        return None
    try:
        numeric_change = float(change)
    except (TypeError, ValueError):
        return None
    return f"{numeric_change:+.1f}%"


def _derive_analysis_from_dataframe(data_frame: Optional[pd.DataFrame]) -> dict:
    """Generate analytics similar to the workflow outputs for local filtering."""
    if data_frame is None or data_frame.empty:
        return {}

    df = data_frame.copy()

    numeric_candidates = [
        "electricity_kwh",
        "energy_usage_kwh",
        "carbon_emissions_kg",
        "emissions_kgco2",
        "water_gallons",
        "water_usage_liters",
        "operational_hours",
    ]
    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")

    energy_col = _get_first_existing_column(df, ["electricity_kwh", "energy_usage_kwh"])
    carbon_col = _get_first_existing_column(df, ["carbon_emissions_kg", "emissions_kgco2"])
    water_col = _get_first_existing_column(df, ["water_gallons", "water_usage_liters"])

    summary: dict[str, dict[str, float]] = {}
    trends: dict[str, dict[str, float]] = {}
    anomalies: dict[str, dict[str, float]] = {}
    facility_comparison: dict[str, dict[str, float]] = {}
    carbon_intensity: dict[str, dict[str, float] | float] = {}
    efficiency: dict[str, float] = {}

    def _build_summary(series: pd.Series) -> Optional[dict[str, float]]:
        series = series.dropna()
        if series.empty:
            return None
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "total": float(series.sum()),
        }

    if energy_col and energy_col in df.columns:
        summary_stats = _build_summary(df[energy_col])
        if summary_stats:
            summary["electricity_kwh"] = summary_stats

        if "date" in df.columns and len(df) > 1:
            energy_series = df.dropna(subset=[energy_col])[["date", energy_col]]
            if len(energy_series) > 1:
                first = energy_series.iloc[0][energy_col]
                last = energy_series.iloc[-1][energy_col]
                pct_change = ((last - first) / first * 100) if first else 0.0
                trends["electricity_kwh"] = {
                    "direction": "increasing" if pct_change > 0 else "decreasing",
                    "percentage_change": round(pct_change, 2),
                }

            mean = df[energy_col].mean()
            std = df[energy_col].std()
            upper = mean + 2 * std
            lower = mean - 2 * std
            anomaly_mask = (df[energy_col] > upper) | (df[energy_col] < lower)
            anomaly_count = int(anomaly_mask.sum())
            if anomaly_count:
                anomalies["electricity_kwh"] = {
                    "count": anomaly_count,
                    "percentage": round(anomaly_count / len(df) * 100, 2),
                    "threshold_upper": float(upper),
                    "threshold_lower": float(lower),
                }

        if "operational_hours" in df.columns:
            hours_series = pd.to_numeric(df["operational_hours"], errors="coerce")
            ratio = df[energy_col] / hours_series.replace(0, pd.NA)
            ratio = ratio.dropna()
            if not ratio.empty:
                efficiency["energy_per_operating_hour"] = round(float(ratio.mean()), 2)

    if carbon_col and carbon_col in df.columns:
        summary_stats = _build_summary(df[carbon_col])
        if summary_stats:
            summary["carbon_emissions_kg"] = summary_stats

        if "date" in df.columns and len(df) > 1:
            carbon_series = df.dropna(subset=[carbon_col])[["date", carbon_col]]
            if len(carbon_series) > 1:
                first = carbon_series.iloc[0][carbon_col]
                last = carbon_series.iloc[-1][carbon_col]
                pct_change = ((last - first) / first * 100) if first else 0.0
                trends["carbon_emissions_kg"] = {
                    "direction": "increasing" if pct_change > 0 else "decreasing",
                    "percentage_change": round(pct_change, 2),
                }

            mean = df[carbon_col].mean()
            std = df[carbon_col].std()
            upper = mean + 2 * std
            lower = mean - 2 * std
            anomaly_mask = (df[carbon_col] > upper) | (df[carbon_col] < lower)
            anomaly_count = int(anomaly_mask.sum())
            if anomaly_count:
                anomalies["carbon_emissions_kg"] = {
                    "count": anomaly_count,
                    "percentage": round(anomaly_count / len(df) * 100, 2),
                    "threshold_upper": float(upper),
                    "threshold_lower": float(lower),
                }

    if water_col and water_col in df.columns:
        water_series = pd.to_numeric(df[water_col], errors="coerce")
        if water_col == "water_usage_liters":
            water_series = water_series * 0.264172
        water_series = water_series.dropna()
        if not water_series.empty:
            summary_stats = _build_summary(water_series)
            if summary_stats:
                summary["water_gallons"] = summary_stats

            if "date" in df.columns and len(df) > 1:
                water_series_with_date = pd.DataFrame({
                    "date": df["date"],
                    "water_gallons": water_series,
                }).dropna()
                if len(water_series_with_date) > 1:
                    first = water_series_with_date.iloc[0]["water_gallons"]
                    last = water_series_with_date.iloc[-1]["water_gallons"]
                    pct_change = ((last - first) / first * 100) if first else 0.0
                    trends["water_gallons"] = {
                        "direction": "increasing" if pct_change > 0 else "decreasing",
                        "percentage_change": round(pct_change, 2),
                    }

                mean = water_series.mean()
                std = water_series.std()
                upper = mean + 2 * std
                lower = mean - 2 * std
                anomalies["water_gallons"] = {
                    "count": int(((water_series > upper) | (water_series < lower)).sum()),
                    "percentage": round(((water_series > upper) | (water_series < lower)).sum() / len(water_series) * 100, 2),
                    "threshold_upper": float(upper),
                    "threshold_lower": float(lower),
                }

            efficiency["water_usage_per_day"] = round(float(water_series.mean()), 2)

    if "operational_hours" in df.columns:
        hours_series = pd.to_numeric(df["operational_hours"], errors="coerce")
        hours_series = hours_series.dropna()
        if not hours_series.empty:
            efficiency["avg_operational_hours"] = round(float(hours_series.mean()), 1)

    if "facility" in df.columns:
        if energy_col and energy_col in df.columns:
            facility_energy = df.groupby("facility")[energy_col].sum().sort_values(ascending=False)
            facility_comparison["electricity_kwh"] = facility_energy.to_dict()
        if carbon_col and carbon_col in df.columns:
            facility_carbon = df.groupby("facility")[carbon_col].sum().sort_values(ascending=False)
            facility_comparison.setdefault("carbon_emissions_kg", {}).update(facility_carbon.to_dict())

    if energy_col and carbon_col and energy_col in df.columns and carbon_col in df.columns:
        total_energy = df[energy_col].sum()
        total_carbon = df[carbon_col].sum()
        if total_energy:
            carbon_intensity["overall"] = round(float(total_carbon / total_energy), 4)

        if "facility" in df.columns:
            facility_ratio = (
                df.groupby("facility")[[energy_col, carbon_col]]
                .sum()
                .assign(intensity=lambda frame: frame[carbon_col] / frame[energy_col].replace(0, pd.NA))
            )
            facility_ratio = facility_ratio.dropna(subset=["intensity"])
            if not facility_ratio.empty:
                carbon_intensity["by_facility"] = {
                    facility: round(float(value), 4)
                    for facility, value in facility_ratio["intensity"].items()
                }

    return {
        "summary_statistics": summary,
        "trend_analysis": trends,
        "anomaly_detection": anomalies,
        "facility_comparison": facility_comparison,
        "carbon_intensity": carbon_intensity,
        "efficiency_metrics": {k: v for k, v in efficiency.items() if v is not None},
    }


def _coerce_summary_block(summary: str | dict | None) -> tuple[str, Optional[pd.DataFrame]]:
    """Return cleaned summary text and optional structured metrics."""
    if summary is None:
        return "No executive summary available.", None

    if isinstance(summary, dict):
        df = pd.DataFrame(summary).T
        descriptor = ", ".join(str(idx).replace("_", " ").title() for idx in df.index)
        text = f"Summary statistics generated for {descriptor}."
        return text, df

    if isinstance(summary, str):
        trimmed = summary.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                parsed = ast.literal_eval(trimmed)
                if isinstance(parsed, dict):
                    df = pd.DataFrame(parsed).T
                    descriptor = ", ".join(str(idx).replace("_", " ").title() for idx in df.index)
                    text = f"Summary statistics generated for {descriptor}."
                    return text, df
            except Exception:
                pass
        return summary, None

    return str(summary), None


def _render_metric_cards(data_frame: pd.DataFrame, analysis_payload: dict) -> None:
    """Surface key metrics in a compact layout."""
    summary = analysis_payload.get("summary_statistics", {}) if analysis_payload else {}
    trends = analysis_payload.get("trend_analysis", {}) if analysis_payload else {}
    anomalies = analysis_payload.get("anomaly_detection", {}) if analysis_payload else {}
    efficiency = analysis_payload.get("efficiency_metrics", {}) if analysis_payload else {}

    metrics = []

    record_count = len(data_frame) if data_frame is not None else 0
    metrics.append({
        "label": "Records Analyzed",
        "value": f"{record_count:,}"
    })

    energy_summary = summary.get("electricity_kwh", {})
    if energy_summary:
        metrics.append({
            "label": "Energy Consumption",
            "value": _format_metric_value(energy_summary.get("total"), " kWh"),
            "delta": _format_delta(trends.get("electricity_kwh", {}).get("percentage_change"))
        })

    carbon_summary = summary.get("carbon_emissions_kg", {})
    if carbon_summary:
        metrics.append({
            "label": "Carbon Emissions",
            "value": _format_metric_value(carbon_summary.get("total"), " kg"),
            "delta": _format_delta(trends.get("carbon_emissions_kg", {}).get("percentage_change"))
        })

    water_summary = summary.get("water_gallons", {})
    if water_summary:
        metrics.append({
            "label": "Water Usage",
            "value": _format_metric_value(water_summary.get("total"), " gal"),
            "delta": _format_delta(trends.get("water_gallons", {}).get("percentage_change"))
        })

    if anomalies:
        total_anomalies = sum(item.get("count", 0) for item in anomalies.values())
        metrics.append({
            "label": "Anomalies Detected",
            "value": f"{total_anomalies:,}"
        })

    if efficiency:
        intensity = efficiency.get("energy_intensity_kwh_per_sqft")
        if intensity is not None:
            metrics.append({
                "label": "Energy Intensity",
                "value": _format_metric_value(intensity, " kWh/sqft")
            })
        water_usage = efficiency.get("water_usage_per_day")
        if water_usage is not None:
            metrics.append({
                "label": "Avg. Water / Day",
                "value": _format_metric_value(water_usage, " gal")
            })

    if not metrics:
        st.info("No metrics available; run the workflow to populate insights.")
        return

    st.subheader("Key Sustainability Indicators")
    for idx in range(0, len(metrics), 4):
        row = metrics[idx : idx + 4]
        cols = st.columns(len(row))
        for column, metric in zip(cols, row):
            column.metric(metric["label"], metric["value"], metric.get("delta"))


def _render_time_series(data_frame: pd.DataFrame) -> None:
    """Display time-series visualizations for core metrics."""
    if data_frame is None or data_frame.empty or "date" not in data_frame.columns:
        st.info("Time-series charts become available once the dataset includes a date column.")
        return

    time_series = data_frame.copy()
    time_series["date"] = pd.to_datetime(time_series["date"], errors="coerce")
    time_series = time_series.dropna(subset=["date"]).set_index("date").sort_index()

    if time_series.empty:
        st.info("Could not parse any valid timestamps from the dataset.")
        return

    numeric_candidates = [
        "electricity_kwh",
        "energy_usage_kwh",
        "carbon_emissions_kg",
        "emissions_kgco2",
        "water_gallons",
        "water_usage_liters",
    ]
    numeric_columns = [column for column in numeric_candidates if column in time_series.columns]

    if not numeric_columns:
        st.info("No numeric sustainability fields detected for trend visualizations.")
        return

    monthly = time_series[numeric_columns].resample("ME").sum()

    if "energy_usage_kwh" in monthly.columns and "electricity_kwh" not in monthly.columns:
        monthly["electricity_kwh"] = monthly["energy_usage_kwh"]
    if "emissions_kgco2" in monthly.columns and "carbon_emissions_kg" not in monthly.columns:
        monthly["carbon_emissions_kg"] = monthly["emissions_kgco2"]
    if "water_usage_liters" in monthly.columns and "water_gallons" not in monthly.columns:
        monthly["water_gallons"] = monthly["water_usage_liters"] * 0.264172

    columns_to_drop = ["energy_usage_kwh", "emissions_kgco2", "water_usage_liters"]
    monthly = monthly.drop(columns=[column for column in columns_to_drop if column in monthly.columns])

    chart_columns = st.columns(2)
    if "electricity_kwh" in monthly.columns:
        chart_columns[0].subheader("Monthly Energy Consumption")
        chart_columns[0].area_chart(monthly[["electricity_kwh"]])

    if "carbon_emissions_kg" in monthly.columns:
        target_col = chart_columns[1] if len(chart_columns) > 1 else st
        target_col.subheader("Monthly Carbon Emissions")
        target_col.line_chart(monthly[["carbon_emissions_kg"]])

    if "water_gallons" in monthly.columns:
        st.subheader("Monthly Water Usage")
        st.bar_chart(monthly[["water_gallons"]])

    energy_col = _get_first_existing_column(time_series, ["electricity_kwh", "energy_usage_kwh"])
    if "facility" in time_series.columns and energy_col:
        facility_breakdown = (
            time_series.groupby("facility")[energy_col].sum().sort_values(ascending=False)
        )
        if not facility_breakdown.empty:
            st.subheader("Energy Consumption by Facility")
            st.bar_chart(facility_breakdown)


def _render_anomaly_highlights(analysis_payload: dict) -> None:
    anomalies = analysis_payload.get("anomaly_detection", {}) if analysis_payload else {}
    if not anomalies:
        st.caption("No anomaly alerts were triggered in this run.")
        return

    anomaly_df = pd.DataFrame(anomalies).T
    if "count" in anomaly_df.columns:
        chart_source = anomaly_df[["count"]].rename(columns={"count": "events"}).sort_values(
            "events", ascending=True
        )
        st.bar_chart(chart_source)

    sorted_df = anomaly_df
    if "count" in anomaly_df.columns:
        sorted_df = anomaly_df.sort_values("count", ascending=False)

    for metric, row in sorted_df.iterrows():
        threshold_lower = row.get("threshold_lower")
        threshold_upper = row.get("threshold_upper")
        details = []
        if threshold_lower is not None and threshold_upper is not None:
            details.append(
                f"thresholds: {threshold_lower:.0f} – {threshold_upper:.0f}"
            )
        percentage = row.get("percentage")
        if percentage is not None:
            details.append(f"impact: {percentage:.1f}% of records")
        label = ", ".join(details)
        st.caption(f"{metric.replace('_', ' ').title()}: {int(row.get('count', 0))} anomaly flags ({label})")


def _render_operational_profile(data_frame: pd.DataFrame) -> None:
    if data_frame is None or data_frame.empty:
        return

    hvac_col = _get_first_existing_column(data_frame, ["hvac_setting", "hvac_mode"])
    energy_col = _get_first_existing_column(data_frame, ["electricity_kwh", "energy_usage_kwh"])
    hours_col = _get_first_existing_column(data_frame, ["operational_hours"])

    col_left, col_right = st.columns(2)

    if hvac_col:
        hvac_energy = None
        if energy_col:
            hvac_energy = (
                data_frame.groupby(hvac_col)[energy_col].mean().sort_values(ascending=True)
            )
        hvac_counts = data_frame[hvac_col].value_counts().sort_values(ascending=True)
        hvac_chart = pd.DataFrame({
            "occurrences": hvac_counts
        })
        col_left.markdown("**HVAC Strategy Mix**")
        col_left.bar_chart(hvac_chart)
        if hvac_energy is not None and not hvac_energy.empty:
            col_left.caption("Average kWh impact by HVAC mode")
            col_left.bar_chart(hvac_energy.rename("kWh"))

    if hours_col and energy_col:
        scatter_source = data_frame[[hours_col, energy_col]].dropna()
        if not scatter_source.empty:
            scatter_source = scatter_source.rename(
                columns={hours_col: "Operational Hours", energy_col: "Energy kWh"}
            )
            col_right.markdown("**Operational Hours vs Energy Load**")
            col_right.scatter_chart(scatter_source)

    if "date" in data_frame.columns and hours_col:
        temporal = data_frame[["date", hours_col]].dropna()
        temporal["date"] = pd.to_datetime(temporal["date"], errors="coerce")
        temporal = temporal.dropna(subset=["date"]).set_index("date").sort_index()
        if not temporal.empty:
            temporal = temporal.resample("W").mean()
            st.markdown("**Average Operational Hours (Weekly)**")
            st.area_chart(temporal.rename(columns={hours_col: "Avg Hours"}))


def _render_analysis_diagnostics(analysis_payload: dict, data_frame: pd.DataFrame) -> None:
    if data_frame is None or data_frame.empty:
        st.info("No dataset available after applying the current filters.")
        return

    summary_tab, trends_tab, anomalies_tab, efficiency_tab = st.tabs(
        ["Summary", "Trends", "Anomalies", "Efficiency & Correlation"]
    )

    with summary_tab:
        summary = analysis_payload.get("summary_statistics", {})
        if summary:
            summary_df = pd.DataFrame(summary).T
            chart_data = summary_df[[
                column
                for column in ["total", "mean", "max", "min"]
                if column in summary_df.columns
            ]]
            if not chart_data.empty:
                st.bar_chart(chart_data)

            metric_columns = st.columns(len(summary_df.index))
            for column, (metric, stats) in zip(metric_columns, summary_df.iterrows()):
                column.metric(
                    metric.replace("_", " ").title(),
                    _format_metric_value(stats.get("total")),
                    _format_delta(analysis_payload.get("trend_analysis", {}).get(metric, {}).get("percentage_change")),
                )
        else:
            st.caption("Summary statistics unavailable for this slice.")

        energy_col = _get_first_existing_column(data_frame, ["electricity_kwh", "energy_usage_kwh"])
        if energy_col and "facility" in data_frame.columns:
            facility_profile = (
                data_frame.groupby("facility")[energy_col].sum().sort_values(ascending=False)
            )
            if not facility_profile.empty:
                st.markdown("**Energy Share by Facility**")
                st.bar_chart(facility_profile)

    with trends_tab:
        trends = analysis_payload.get("trend_analysis", {})
        if trends:
            trend_columns = st.columns(len(trends))
            for column, (metric, values) in zip(trend_columns, trends.items()):
                column.metric(
                    metric.replace("_", " ").title(),
                    values.get("direction", "stable").title(),
                    _format_delta(values.get("percentage_change")),
                )
        else:
            st.caption("Trend analysis unavailable for this slice.")

        carbon_intensity = analysis_payload.get("carbon_intensity", {})
        facility_intensity = carbon_intensity.get("by_facility", {}) if carbon_intensity else {}
        if facility_intensity:
            st.markdown("**Carbon Intensity by Facility (kg / kWh)**")
            intensity_series = pd.Series(facility_intensity).sort_values(ascending=False)
            st.bar_chart(intensity_series)
        elif carbon_intensity.get("overall"):
            st.metric("Overall Carbon Intensity", f"{carbon_intensity['overall']:.3f} kg/kWh")

        if "date" in data_frame.columns:
            energy_col = _get_first_existing_column(data_frame, ["electricity_kwh", "energy_usage_kwh"])
            carbon_col = _get_first_existing_column(data_frame, ["carbon_emissions_kg", "emissions_kgco2"])
            if energy_col and carbon_col:
                intensity_series = (
                    data_frame.dropna(subset=[energy_col, carbon_col])
                    .assign(
                        carbon_intensity=lambda frame: frame[carbon_col] / frame[energy_col].replace(0, pd.NA)
                    )
                )
                intensity_series["date"] = pd.to_datetime(intensity_series["date"], errors="coerce")
                chart_source = intensity_series.dropna(subset=["carbon_intensity", "date"])
                if not chart_source.empty:
                    chart_source = chart_source.set_index("date")["carbon_intensity"].resample("ME").mean()
                    st.line_chart(chart_source.rename("Carbon Intensity"))

    with anomalies_tab:
        _render_anomaly_highlights(analysis_payload)

    with efficiency_tab:
        efficiency = analysis_payload.get("efficiency_metrics", {})
        if efficiency:
            metric_cols = st.columns(min(3, len(efficiency)))
            for idx, (metric, value) in enumerate(efficiency.items()):
                metric_cols[idx % len(metric_cols)].metric(
                    metric.replace("_", " ").title(), _format_metric_value(value)
                )
        else:
            st.caption("Efficiency insights unavailable for this slice.")

        numeric_cols = data_frame.select_dtypes(include="number")
        if not numeric_cols.empty and len(numeric_cols.columns) > 1:
            corr = numeric_cols.corr().round(3)
            fig, ax = plt.subplots(figsize=(6, 4))
            cax = ax.imshow(corr, cmap="Greens", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
            st.pyplot(fig)


def _render_interventions(interventions: dict) -> None:
    if not interventions:
        st.info("Run the workflow to generate tailored intervention recommendations.")
        return

    recommendations = interventions.get("recommendations", [])
    roadmap = interventions.get("implementation_roadmap", {})

    if recommendations:
        st.subheader("Recommended Actions")
        rec_df = pd.DataFrame(recommendations)

        metrics_row = st.columns(3)
        metrics_row[0].metric("Recommendations", len(rec_df))
        if "category" in rec_df:
            metrics_row[1].metric("Focus Areas", rec_df["category"].nunique())
        if "impact" in rec_df:
            high_impact = (rec_df["impact"].str.lower() == "high").sum()
            metrics_row[2].metric("High-Impact", high_impact)

        charts_container = st.container()
        with charts_container:
            chart_cols = st.columns(2)

            if "impact" in rec_df:
                impact_counts = rec_df["impact"].value_counts().sort_values(ascending=True)
                if not impact_counts.empty:
                    chart_cols[0].markdown("**Impact Mix**")
                    chart_cols[0].bar_chart(impact_counts)

            if "timeframe" in rec_df:
                timeframe_counts = rec_df["timeframe"].value_counts().sort_values(ascending=True)
                if not timeframe_counts.empty:
                    chart_cols[1].markdown("**Timeline Commitments**")
                    chart_cols[1].bar_chart(timeframe_counts)

        st.markdown("**Recommendation Gallery**")
        for rec in rec_df.to_dict(orient="records"):
            card = st.container()
            title = rec.get("title", "Recommendation")
            category = rec.get("category", "N/A")
            impact = rec.get("impact", "N/A")
            cost = rec.get("cost", "N/A")
            timeframe = rec.get("timeframe", "N/A")
            estimated = rec.get("estimated_savings", "N/A")

            card_cols = card.columns([3, 1])
            with card_cols[0]:
                st.markdown(f"### {title}")
                st.write(rec.get("description", ""))
            with card_cols[1]:
                st.metric("Impact", impact)
                st.metric("Cost", cost)
                st.metric("Timeline", timeframe)
                st.caption(f"Savings: {estimated}")

            st.caption(f"Category: {category}")
            st.markdown("---")

    if roadmap:
        st.subheader("Implementation Roadmap")
        phase_counts = {
            phase: len(items)
            for phase, items in roadmap.items()
        }
        if phase_counts:
            st.bar_chart(pd.Series(phase_counts).sort_values(ascending=True))

        for phase, items in roadmap.items():
            st.markdown(f"**{phase}**")
            if items:
                st.markdown("\n".join(f"- {item}" for item in items))
            else:
                st.caption("No actions captured for this phase yet.")


def _render_report(report: dict) -> None:
    if not report:
        st.info("Report artifacts are generated once the workflow completes.")
        return

    raw_executive_summary = report.get("executive_summary")
    executive_summary, summary_df = _coerce_summary_block(raw_executive_summary)
    narrative = report.get("report", "No detailed report available.")
    visualizations = report.get("visualizations", [])

    total_visuals = len([path for path in visualizations if os.path.exists(path)])
    summary_words = len(str(executive_summary).split())
    narrative_words = len(narrative.split())

    metric_cols = st.columns(3)
    metric_cols[0].metric("Visual Assets", total_visuals)
    metric_cols[1].metric("Summary Length", f"{summary_words} words")
    metric_cols[2].metric("Narrative Depth", f"{narrative_words} words")

    st.subheader("Executive Summary")
    if summary_df is not None and not summary_df.empty:
        highlight_metrics = summary_df[[
            column
            for column in ["total", "mean", "max"]
            if column in summary_df.columns
        ]]
        if not highlight_metrics.empty:
            st.markdown("**Top-line Metrics**")
            st.bar_chart(highlight_metrics)

        metric_columns = st.columns(min(4, len(summary_df.index)))
        for idx, (metric, stats) in enumerate(summary_df.iterrows()):
            target_column = metric_columns[idx % len(metric_columns)]
            target_column.metric(
                metric.replace("_", " ").title(),
                _format_metric_value(stats.get("total")),
                _format_delta(stats.get("percentage_change")),
            )

        st.caption("Snapshot derived from latest analysis output.")
    st.success(executive_summary)

    st.subheader("Detailed Narrative")
    sentences = [sentence.strip() for sentence in re.split(r"[\n\.]+", narrative) if sentence.strip()]
    if sentences:
        for sentence in sentences:
            st.markdown(f"- {sentence}")
    else:
        st.write(narrative)

    combined_text = f"{executive_summary} {narrative}".lower()
    tokens = re.findall(r"[a-z]+", combined_text)
    stopwords = {
        "the", "and", "for", "with", "that", "from", "this", "have", "will",
        "into", "within", "while", "which", "their", "overall", "more", "than",
        "clinic", "hospital",
    }
    filtered_tokens = [tok for tok in tokens if tok not in stopwords and len(tok) > 3]
    if filtered_tokens:
        freq_series = pd.Series(filtered_tokens).value_counts().head(10).sort_values(ascending=True)
        if not freq_series.empty:
            st.markdown("**Key Themes in Narrative**")
            st.bar_chart(freq_series)

    if visualizations:
        st.subheader("Generated Visualizations")
        for idx in range(0, len(visualizations), 2):
            row_paths = visualizations[idx : idx + 2]
            columns = st.columns(len(row_paths))
            for column, path in zip(columns, row_paths):
                if os.path.exists(path):
                    column.image(path, width="stretch")
                else:
                    column.warning(f"Visualization not found: {path}")
    else:
        st.caption("No visualization assets were produced in this run.")


def _render_data_room(
    data_frame: pd.DataFrame,
    results: dict,
    history: list,
    total_records: Optional[int] = None,
) -> None:
    st.subheader("Raw Data Preview")
    if total_records is not None and data_frame is not None:
        st.caption(f"Showing {len(data_frame):,} of {total_records:,} records after filters.")
    if data_frame is None or data_frame.empty:
        st.info("No dataset available for preview. Upload data and rerun the workflow.")
    else:
        facility_count = (
            int(data_frame["facility"].nunique())
            if "facility" in data_frame.columns and not data_frame.empty
            else None
        )
        date_range = None
        if "date" in data_frame.columns:
            parsed_dates = pd.to_datetime(data_frame["date"], errors="coerce")
            parsed_dates = parsed_dates.dropna()
            if not parsed_dates.empty:
                date_range = (
                    parsed_dates.min().date(),
                    parsed_dates.max().date(),
                )

        metric_columns = st.columns(3)
        metric_columns[0].metric("Rows Displayed", f"{len(data_frame):,}")
        metric_columns[1].metric(
            "Facilities",
            facility_count if facility_count is not None else "N/A",
        )
        metric_columns[2].metric(
            "Date Coverage",
            f"{date_range[0]:%b %d} – {date_range[1]:%b %d}" if date_range else "N/A",
        )

        st.dataframe(data_frame.head(200))
        csv_bytes = data_frame.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download dataset (CSV)",
            data=csv_bytes,
            file_name="hospital_sustainability_data.csv",
            mime="text/csv",
            width="stretch",
        )

        st.markdown("**Data Distributions**")
        viz_cols = st.columns(2)
        if "facility" in data_frame.columns:
            facility_counts = data_frame["facility"].value_counts().sort_values(ascending=True)
            if not facility_counts.empty:
                viz_cols[0].markdown("Facility Mix")
                viz_cols[0].bar_chart(facility_counts)

        hvac_col = _get_first_existing_column(data_frame, ["hvac_setting", "hvac_mode"])
        if hvac_col:
            hvac_counts = data_frame[hvac_col].value_counts().sort_values(ascending=True)
            if not hvac_counts.empty:
                viz_cols[1].markdown("HVAC Strategy Split")
                viz_cols[1].bar_chart(hvac_counts)

        energy_col = _get_first_existing_column(data_frame, ["electricity_kwh", "energy_usage_kwh"])
        carbon_col = _get_first_existing_column(data_frame, ["carbon_emissions_kg", "emissions_kgco2"])

        numeric_snapshot = data_frame.select_dtypes(include="number")
        if not numeric_snapshot.empty:
            st.markdown("**Numeric Snapshot**")
            summary_table = numeric_snapshot.describe().T
            selected_columns = [
                column for column in ["mean", "std", "min", "max"]
                if column in summary_table.columns
            ]
            if selected_columns:
                st.dataframe(summary_table[selected_columns])

            mean_series = numeric_snapshot.mean().sort_values(ascending=True)
            if not mean_series.empty:
                st.bar_chart(mean_series.to_frame(name="Average"))

        if "date" in data_frame.columns and energy_col:
            trend_df = data_frame[["date", energy_col]].dropna()
            trend_df["date"] = pd.to_datetime(trend_df["date"], errors="coerce")
            trend_df = trend_df.dropna(subset=["date"]).set_index("date").sort_index()
            if not trend_df.empty:
                st.markdown("**Energy Trend (Filtered View)**")
                st.line_chart(trend_df.resample("W").sum().rename(columns={energy_col: "kWh"}))

        if energy_col and carbon_col:
            scatter_df = data_frame[[energy_col, carbon_col]].dropna()
            if not scatter_df.empty:
                scatter_df = scatter_df.rename(
                    columns={
                        energy_col: "Energy (kWh)",
                        carbon_col: "Carbon (kg)",
                    }
                )
                st.markdown("**Energy vs Carbon Correlation**")
                st.scatter_chart(scatter_df)

    st.subheader("Session History")
    if history:
        try:
            with st.status("Workflow Replay", expanded=True) as status:
                for step, entry in enumerate(history, start=1):
                    status.write(f"{step}. {entry}")
                status.update(label="Workflow Replay", state="complete", expanded=False)
        except Exception:
            for entry in history:
                st.markdown(f"- {entry}")
    else:
        st.caption("Session history will display here after running the workflow.")

    st.subheader("Run Metadata")
    st.json(
        {
            "workflow_id": results.get("workflow_id"),
            "status": results.get("status"),
            "records_displayed": len(data_frame) if data_frame is not None else 0,
            "records_total": total_records,
            "timestamp": results.get("analysis", {}).get("timestamp"),
        }
    )


with st.sidebar:
    st.header("Configuration")
    enable_llm = st.checkbox(
        "Enable LLM agents",
        value=os.getenv("ENABLE_LLM", "0") == "1",
        help="When disabled the workflow uses deterministic local fallbacks.",
    )
    provider = st.selectbox(
        "LLM Provider",
        options=["openai", "groq"],
        index=0 if os.getenv("LLM_PROVIDER", "openai").lower() == "openai" else 1,
    )

    os.environ["ENABLE_LLM"] = "1" if enable_llm else "0"
    os.environ["LLM_PROVIDER"] = provider

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload hospital sustainability CSV",
        type=["csv"],
        help="Leave empty to use the bundled sample dataset.",
    )
    use_sample = st.checkbox("Use bundled sample", value=uploaded_file is None)

    if SAMPLE_DATA_PATH is None and use_sample:
        st.warning("Sample dataset unavailable. Please upload a CSV to proceed.")
    elif SAMPLE_DATA_PATH is None:
        st.caption("Sample dataset unavailable in this workspace.")

    run_workflow = st.button("Run Analysis", width="stretch")

if run_workflow:
    data_path = SAMPLE_DATA_PATH if use_sample and SAMPLE_DATA_PATH else None
    temp_path: Optional[str] = None

    if not use_sample and uploaded_file is None:
        st.warning("Please upload a dataset or enable the bundled sample.")
    else:
        if uploaded_file is not None and not use_sample:
            temp_path = _write_temp_csv(uploaded_file)
            data_path = temp_path

        if data_path is None:
            st.error("Unable to determine data source.")
        else:
            with st.spinner("Running sustainability workflow..."):
                orchestrator = SustainabilityAgentOrchestrator()
                results = orchestrator.run_full_workflow(data_path=data_path)
                st.session_state["workflow_results"] = results

        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)

results = st.session_state.get("workflow_results")

if results is None:
    st.info("Configure the options in the sidebar and click *Run Analysis* to generate insights.")
else:
    if results.get("status") == "error":
        st.error(f"Workflow failed: {results.get('message', 'Unknown error')}")
        history = results.get("session_history", [])
        if history:
            st.subheader("Session History")
            for entry in history:
                st.markdown(f"- {entry}")
    else:
        st.success("Workflow completed successfully.")

        collection = results.get("data_collection", {})
        data_dict = collection.get("data", {})
        data_frame = pd.DataFrame(data_dict) if data_dict else pd.DataFrame()
        if not data_frame.empty:
            data_frame = data_frame.convert_dtypes()

        record_count = collection.get("record_count", len(data_frame))
        facility_count = (
            int(data_frame["facility"].nunique())
            if not data_frame.empty and "facility" in data_frame.columns
            else 0
        )

        date_bounds = None
        if not data_frame.empty and "date" in data_frame.columns:
            parsed_dates = pd.to_datetime(data_frame["date"], errors="coerce")
            if not parsed_dates.dropna().empty:
                date_bounds = (
                    parsed_dates.dropna().min().to_pydatetime(),
                    parsed_dates.dropna().max().to_pydatetime(),
                )

        coverage_bits = [f"{record_count:,} records"]
        if facility_count:
            coverage_bits.append(f"{facility_count} facilities")
        if date_bounds:
            coverage_bits.append(
                f"{date_bounds[0]:%b %d, %Y} → {date_bounds[1]:%b %d, %Y}"
            )
        st.caption(" | ".join(coverage_bits))

        base_analysis = results.get("analysis", {}).get("analysis", {}) or {}
        derived_overall = _derive_analysis_from_dataframe(data_frame)
        analysis_payload = base_analysis.copy()
        for key, value in derived_overall.items():
            if key not in analysis_payload or not analysis_payload[key]:
                analysis_payload[key] = value

        filtered_df = data_frame.copy()
        selected_facilities = None
        selected_dates = None
        facility_options: list[str] = []
        active_filters: list[str] = []

        if not filtered_df.empty:
            with st.container():
                st.markdown("### Story Filters")
                filter_cols = st.columns(2)

                if "facility" in filtered_df.columns:
                    facility_options = sorted(filtered_df["facility"].dropna().unique())
                    if facility_options:
                        selected_facilities = filter_cols[0].multiselect(
                            "Facilities",
                            options=facility_options,
                            default=facility_options,
                        )
                        if selected_facilities and set(selected_facilities) != set(facility_options):
                            filtered_df = filtered_df[filtered_df["facility"].isin(selected_facilities)]
                            active_filters.append(f"Facilities ({len(selected_facilities)})")

                if "date" in filtered_df.columns:
                    filtered_df["date"] = pd.to_datetime(filtered_df["date"], errors="coerce")
                    filtered_df = filtered_df.dropna(subset=["date"])
                    if not filtered_df.empty:
                        min_date = filtered_df["date"].min().to_pydatetime()
                        max_date = filtered_df["date"].max().to_pydatetime()
                        if min_date != max_date:
                            selected_dates = filter_cols[1].slider(
                                "Timeline",
                                min_value=min_date,
                                max_value=max_date,
                                value=(min_date, max_date),
                            )
                            filtered_df = filtered_df[
                                (filtered_df["date"] >= selected_dates[0])
                                & (filtered_df["date"] <= selected_dates[1])
                            ]
                            if (
                                selected_dates[0] > min_date
                                or selected_dates[1] < max_date
                            ):
                                active_filters.append(
                                    f"Timeline ({selected_dates[0].date():%b %d} – {selected_dates[1].date():%b %d})"
                                )
                        else:
                            filter_cols[1].caption("Timeline filter activates once multiple dates exist.")

        filtered_analysis = _derive_analysis_from_dataframe(filtered_df)
        for key, value in analysis_payload.items():
            if key not in filtered_analysis or not filtered_analysis[key]:
                filtered_analysis[key] = value

        report_payload = results.get("report", {})
        interventions_payload = results.get("interventions", {})
        history = results.get("session_history", [])

        overview_tab, analysis_tab, interventions_tab, report_tab, data_tab = st.tabs(
            ["Overview", "Analysis", "Interventions", "Report", "Data & History"]
        )

        with overview_tab:
            if filtered_df.empty:
                st.warning("No data matches the selected filters. Adjust the filters to continue the story.")
            else:
                if active_filters:
                    st.caption("Active filters: " + " | ".join(active_filters))

                _render_metric_cards(filtered_df, filtered_analysis)

                st.markdown("### Trends & Utilization")
                _render_time_series(filtered_df)

                st.markdown("### Operational Pulse")
                _render_operational_profile(filtered_df)

                st.markdown("### Alerts & Highlights")
                _render_anomaly_highlights(filtered_analysis)

        with analysis_tab:
            _render_analysis_diagnostics(filtered_analysis, filtered_df)

        with interventions_tab:
            _render_interventions(interventions_payload)

        with report_tab:
            _render_report(report_payload)

        with data_tab:
            _render_data_room(
                filtered_df,
                results,
                history,
                total_records=record_count,
            )

        st.caption(f"Workflow ID: {results.get('workflow_id', 'N/A')}")
