"""Streamlit Dashboard for Multilingual Tourism Review Analytics (Cluster Sentiment Engine).

Features:
- Tab 1: Cluster Overview (Aspect Score Heatmap by Property & Aspect)
- Tab 2: Property Deep-Dive (Aspect Sentiment Trend Lines with CUSUM Changepoints)
- Tab 3: Market vs Property (Competitive Benchmarking with Lead/Lag Flags and Trend Classification)
"""

import pathlib
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cri.benchmark import benchmark_property
from src.cri.changepoint import detect_changepoints
from src.cri.classify import RuleBasedClassifier
from src.cri.generate import generate_synthetic_reviews
from src.cri.normalize import normalize_text

ASPECTS: List[str] = [
    "Food",
    "Room",
    "Staff",
    "Housekeeping",
    "Location",
    "Value for Money",
    "Cleanliness",
    "Booking Experience",
]

# Soft Seaborn-inspired muted color palette for aspect trend lines
MUTED_PALETTE: List[str] = [
    "#4C72B0",  # muted blue
    "#DD8452",  # muted orange
    "#55A868",  # muted green
    "#C44E52",  # muted red
    "#8172B3",  # muted purple
    "#937860",  # muted brown
    "#DA8BC3",  # muted pink
    "#8C8C8C",  # muted grey
]

st.set_page_config(
    page_title="Cluster Sentiment Engine | Analytics Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern premium visual appearance
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .stMetric {
        background-color: rgba(240, 243, 246, 0.5);
        border: 1px solid rgba(220, 224, 230, 0.8);
        border-radius: 8px;
        padding: 12px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.95);
        color: #6c757d;
        text-align: center;
        padding: 8px;
        font-size: 13px;
        border-top: 1px solid #e9ecef;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_process_data() -> pd.DataFrame:
    """Load reviews data (generate if absent) and run normalization + aspect classification pipeline."""
    csv_path = pathlib.Path("data/synthetic/reviews.csv")
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_reviews(n_reviews=5000, seed=42)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    df["review_date"] = pd.to_datetime(df["review_date"])
    df["normalized_text"] = df["review_text"].apply(normalize_text)

    classifier = RuleBasedClassifier()
    aspect_scores: Dict[str, List[float]] = {asp: [] for asp in ASPECTS}

    for text in df["normalized_text"]:
        res = classifier.classify(text)
        for asp in ASPECTS:
            aspect_scores[asp].append(res[asp]["score"])

    for asp in ASPECTS:
        df[asp] = aspect_scores[asp]

    return df


def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🏨 Tourism Review Cluster Sentiment Engine")
    st.markdown(
        "Multilingual aspect-based sentiment analysis, changepoint detection, and competitive market benchmarking."
    )

    df = load_and_process_data()

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    st.sidebar.header("🔍 Dashboard Filters")

    # Cluster Filter
    all_clusters = sorted(df["cluster_id"].unique())
    selected_clusters = st.sidebar.multiselect(
        "Select Clusters",
        options=all_clusters,
        default=all_clusters,
        help="Filter data by hotel clusters",
    )

    # Date Range Filter
    min_date = df["review_date"].min().date()
    max_date = df["review_date"].max().date()
    selected_date_range = st.sidebar.slider(
        "Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
    )

    # Apply Filters
    if not selected_clusters:
        st.warning("Please select at least one cluster in the sidebar.")
        return

    filtered_df = df[
        (df["cluster_id"].isin(selected_clusters))
        & (df["review_date"].dt.date >= selected_date_range[0])
        & (df["review_date"].dt.date <= selected_date_range[1])
    ].copy()

    if filtered_df.empty:
        st.info("No review records found matching the selected filter criteria.")
        return

    # Sidebar summary stats
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Overview")
    st.sidebar.metric("Filtered Reviews", f"{len(filtered_df):,}")
    st.sidebar.metric("Properties", f"{filtered_df['property_id'].nunique()}")

    # ── Tabs Navigation ───────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Tab 1: Cluster Overview",
            "📈 Tab 2: Property Deep-Dive",
            "🎯 Tab 3: Market vs Property",
        ]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: CLUSTER OVERVIEW (HEATMAP)
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Cluster Aspect Sentiment Overview")
        st.markdown(
            "Average aspect sentiment scores by property across selected clusters. "
            "Scores range from **-1.0** (Negative) to **+1.0** (Positive)."
        )

        col1, col2, col3, col4 = st.columns(4)
        mean_sentiment = float(filtered_df[ASPECTS].mean().mean())
        col1.metric("Overall Average Score", f"{mean_sentiment:+.3f}")

        aspect_means = filtered_df[ASPECTS].mean()
        best_aspect = aspect_means.idxmax()
        worst_aspect = aspect_means.idxmin()
        col2.metric("Top Aspect", f"{best_aspect} ({aspect_means[best_aspect]:+.2f})")
        col3.metric("Weakest Aspect", f"{worst_aspect} ({aspect_means[worst_aspect]:+.2f})")
        col4.metric("Active Clusters", f"{len(selected_clusters)} / {len(all_clusters)}")

        st.markdown("---")

        # Prepare heatmap data matrix
        heatmap_df = (
            filtered_df.groupby(["property_id", "cluster_id"])[ASPECTS]
            .mean()
            .reset_index()
            .sort_values(by=["cluster_id", "property_id"])
        )

        y_labels = [
            f"{row['property_id']} ({row['cluster_id']})"
            for _, row in heatmap_df.iterrows()
        ]
        z_matrix = heatmap_df[ASPECTS].values

        # Muted diverging color scheme (soft Seaborn Tealrose style)
        colorscale = [
            [0.0, "#C44E52"],   # Soft Red
            [0.25, "#E69F00"],  # Soft Amber
            [0.5, "#F7F7F7"],   # Soft Muted Neutral
            [0.75, "#56B4E9"],  # Soft Light Blue
            [1.0, "#4C72B0"],   # Soft Deep Blue
        ]

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                x=ASPECTS,
                y=y_labels,
                colorscale=colorscale,
                zmin=-1.0,
                zmax=1.0,
                text=np.round(z_matrix, 2),
                texttemplate="%{text:+.2f}",
                textfont={"size": 12},
                colorbar=dict(title="Sentiment Score", tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0]),
                hovertemplate="Property: %{y}<br>Aspect: %{x}<br>Score: %{z:+.3f}<extra></extra>",
            )
        )

        fig_heatmap.update_layout(
            title="Aspect Sentiment Heatmap by Property",
            xaxis_title="Aspect",
            yaxis_title="Property (Cluster)",
            height=min(600, 100 + len(y_labels) * 35),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: PROPERTY DEEP-DIVE (TREND LINES & CHANGEPOINTS)
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Property Deep-Dive & Changepoint Detection")
        st.markdown(
            "Track monthly sentiment dynamics per aspect for a specific property. "
            "Vertical dashed lines mark automatically detected **CUSUM changepoints**."
        )

        available_props = sorted(filtered_df["property_id"].unique())
        selected_prop = st.selectbox(
            "Select Property to Analyze:",
            options=available_props,
            index=0,
        )

        prop_df = filtered_df[filtered_df["property_id"] == selected_prop].copy()
        prop_cluster = prop_df["cluster_id"].iloc[0] if not prop_df.empty else ""

        st.caption(f"Showing analysis for **{selected_prop}** (Cluster: **{prop_cluster}**)")

        # Aspect filter multiselect
        selected_aspects = st.multiselect(
            "Filter Aspects to Display:",
            options=ASPECTS,
            default=ASPECTS,
        )

        if not selected_aspects:
            st.warning("Please select at least one aspect to view trend lines.")
        else:
            # Aggregate by month
            prop_df["year_month"] = prop_df["review_date"].dt.to_period("M").astype(str)
            monthly_trends = prop_df.groupby("year_month")[selected_aspects].mean()

            fig_trends = go.Figure()
            changepoints_summary: List[Dict[str, Any]] = []

            for idx, aspect in enumerate(selected_aspects):
                color = MUTED_PALETTE[idx % len(MUTED_PALETTE)]
                series = monthly_trends[aspect]

                # Add aspect trend line
                fig_trends.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode="lines+markers",
                        name=aspect,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=6),
                        hovertemplate=f"<b>{aspect}</b><br>Month: %{{x}}<br>Score: %{{y:+.3f}}<extra></extra>",
                    )
                )

                # Detect changepoints for this aspect
                cps = detect_changepoints(series, threshold=1.5)
                for cp in cps:
                    changepoints_summary.append({
                        "Aspect": aspect,
                        "Date": cp["date"],
                        "Direction": cp["direction"],
                        "Magnitude": cp["magnitude"],
                    })
                    # Draw vertical changepoint line
                    line_color = "#C44E52" if cp["direction"] == "down" else "#55A868"
                    fig_trends.add_vline(
                        x=cp["date"],
                        line_width=1.5,
                        line_dash="dash",
                        line_color=line_color,
                        annotation_text=f"{aspect} {cp['direction'].upper()}",
                        annotation_position="top left",
                        annotation_font_size=10,
                        annotation_font_color=line_color,
                    )

            fig_trends.update_layout(
                title=f"Monthly Aspect Sentiment Trends — {selected_prop}",
                xaxis_title="Month",
                yaxis_title="Sentiment Score",
                yaxis=dict(range=[-1.05, 1.05]),
                hovermode="x unified",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=80, b=40),
            )

            st.plotly_chart(fig_trends, use_container_width=True)

            # Display detected changepoints table if any exist
            st.markdown("##### 📍 Detected Changepoints (CUSUM Algorithm)")
            if changepoints_summary:
                cp_df = pd.DataFrame(changepoints_summary)
                st.dataframe(
                    cp_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No significant changepoints detected for the selected property and threshold.")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: MARKET VS PROPERTY (BENCHMARKING)
    # ──────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Market vs Property Competitive Benchmarking")
        st.markdown(
            "Compare property aspect scores against cluster averages to classify performance into "
            "**Lead / Lag / Neutral** flags and distinguish **market-wide** from **property-specific** trends."
        )

        # Run benchmark for all property x aspect combinations in filtered data
        benchmark_rows: List[Dict[str, Any]] = []

        unique_props = filtered_df["property_id"].unique()
        for prop_id in unique_props:
            prop_cluster = filtered_df[filtered_df["property_id"] == prop_id]["cluster_id"].iloc[0]
            for aspect in ASPECTS:
                try:
                    res = benchmark_property(prop_id, prop_cluster, filtered_df, aspect=aspect)
                    benchmark_rows.append(res)
                except Exception:
                    continue

        if benchmark_rows:
            bench_df = pd.DataFrame(benchmark_rows)

            # Metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            lead_count = len(bench_df[bench_df["flag"] == "Lead"])
            lag_count = len(bench_df[bench_df["flag"] == "Lag"])
            prop_spec_count = len(bench_df[bench_df["classification"] == "property-specific"])
            market_wide_count = len(bench_df[bench_df["classification"] == "market-wide"])

            m_col1.metric("Leading Aspects (Flag = Lead)", f"{lead_count}")
            m_col2.metric("Lagging Aspects (Flag = Lag)", f"{lag_count}")
            m_col3.metric("Property-Specific Issues", f"{prop_spec_count}")
            m_col4.metric("Market-Wide Issues", f"{market_wide_count}")

            st.markdown("---")

            # Filters for Tab 3 Table
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                flag_filter = st.multiselect(
                    "Filter Flag:",
                    options=["Lead", "Lag", "Neutral"],
                    default=["Lead", "Lag", "Neutral"],
                )
            with b_col2:
                class_filter = st.multiselect(
                    "Filter Classification:",
                    options=["market-wide", "property-specific"],
                    default=["market-wide", "property-specific"],
                )
            with b_col3:
                aspect_filter = st.multiselect(
                    "Filter Aspect:",
                    options=ASPECTS,
                    default=ASPECTS,
                )

            filtered_bench = bench_df[
                (bench_df["flag"].isin(flag_filter))
                & (bench_df["classification"].isin(class_filter))
                & (bench_df["aspect"].isin(aspect_filter))
            ].sort_values(by=["flag", "gap"], ascending=[True, True])

            # Rename columns for presentation
            display_df = filtered_bench.rename(
                columns={
                    "property_id": "Property",
                    "cluster_id": "Cluster",
                    "aspect": "Aspect",
                    "property_score": "Property Score",
                    "cluster_mean": "Cluster Mean",
                    "gap": "Gap (Prop - Cluster)",
                    "flag": "Performance Flag",
                    "classification": "Trend Classification",
                    "confidence": "Confidence",
                }
            )

            # Styled dataframe display
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Property Score": st.column_config.NumberColumn(format="%+.4f"),
                    "Cluster Mean": st.column_config.NumberColumn(format="%+.4f"),
                    "Gap (Prop - Cluster)": st.column_config.NumberColumn(format="%+.4f"),
                    "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                },
            )
        else:
            st.info("Insufficient data to compute market benchmarks for selected filters.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer'>Data: synthetic | Built by Aditya Kalure</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
