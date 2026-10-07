from typing import Any, Dict
import numpy as np
import pandas as pd


def compute_cluster_means(df: pd.DataFrame, aspect: str) -> pd.Series:
    """Compute the mean sentiment score per cluster for a given aspect.

    Args:
        df: Dataframe containing columns ['cluster_id', aspect].
        aspect: Name of the aspect column to aggregate.

    Returns:
        pd.Series: Mean sentiment score indexed by cluster_id.
    """
    if aspect not in df.columns:
        raise ValueError(f"Aspect column '{aspect}' not found in dataframe.")
    return df.groupby("cluster_id")[aspect].mean()


def benchmark_property(
    property_id: str,
    cluster_id: str,
    aspect_scores: pd.DataFrame,
    aspect: str = "Food",
) -> Dict[str, Any]:
    """Benchmark a property's aspect sentiment against its cluster mean and classify trend origin.

    Args:
        property_id: ID of target property (e.g. 'P1').
        cluster_id: ID of cluster (e.g. 'C1').
        aspect_scores: Dataframe containing columns ['property_id', 'cluster_id', 'review_date', aspect].
        aspect: Aspect column name to analyze.

    Returns:
        Dict[str, Any]:
          {
            "property_id": str,
            "cluster_id": str,
            "aspect": str,
            "property_score": float,
            "cluster_mean": float,
            "gap": float,
            "flag": "Lead" | "Lag" | "Neutral",
            "classification": "market-wide" | "property-specific",
            "confidence": float
          }
    """
    if aspect not in aspect_scores.columns:
        raise ValueError(f"Aspect column '{aspect}' not found in aspect_scores dataframe.")

    # Filter cluster and property data
    cluster_df = aspect_scores[aspect_scores["cluster_id"] == cluster_id].copy()
    if cluster_df.empty:
        raise ValueError(f"No records found for cluster_id: '{cluster_id}'")

    prop_df = cluster_df[cluster_df["property_id"] == property_id].copy()
    if prop_df.empty:
        raise ValueError(f"No records found for property_id: '{property_id}' in cluster '{cluster_id}'")

    prop_score = float(prop_df[aspect].mean())
    cluster_mean = float(cluster_df[aspect].mean())
    gap = prop_score - cluster_mean

    if gap > 0.15:
        flag = "Lead"
    elif gap < -0.15:
        flag = "Lag"
    else:
        flag = "Neutral"

    cluster_df["month"] = pd.to_datetime(cluster_df["review_date"]).dt.to_period("M")
    prop_monthly = prop_df.assign(month=pd.to_datetime(prop_df["review_date"]).dt.to_period("M"))

    cluster_monthly_mean = cluster_df.groupby("month")[aspect].mean()
    prop_monthly_mean = prop_monthly.groupby("month")[aspect].mean()

    # Align months
    common_months = prop_monthly_mean.index.intersection(cluster_monthly_mean.index)
    if len(common_months) >= 2:
        x = np.arange(len(common_months))
        y_prop = prop_monthly_mean.loc[common_months].values
        y_cluster = cluster_monthly_mean.loc[common_months].values

        prop_slope = float(np.polyfit(x, y_prop, 1)[0]) if len(x) >= 2 else 0.0
        cluster_slope = float(np.polyfit(x, y_cluster, 1)[0]) if len(x) >= 2 else 0.0
    else:
        prop_slope = 0.0
        cluster_slope = 0.0

    if prop_slope < -0.01 and cluster_slope < -0.01:
        classification = "market-wide"
    elif prop_slope < -0.01 and cluster_slope >= -0.01:
        classification = "property-specific"
    else:
        classification = "market-wide" if abs(gap) <= 0.15 else "property-specific"

    n_samples = len(prop_df)
    confidence = float(min(1.0, max(0.5, n_samples / 100.0)))

    return {
        "property_id": property_id,
        "cluster_id": cluster_id,
        "aspect": aspect,
        "property_score": round(prop_score, 4),
        "cluster_mean": round(cluster_mean, 4),
        "gap": round(gap, 4),
        "flag": flag,
        "classification": classification,
        "confidence": round(confidence, 2),
    }


def generate_action_report(benchmark_result: Dict[str, Any]) -> str:
    """Generate a plain-text one-page action report for a property benchmark result.

    Args:
        benchmark_result: Dict with keys property_id, cluster_id, aspect, property_score,
            cluster_mean, gap, flag, classification, and optional changepoint.

    Returns:
        str: Formatted plain-text action report readable in terminal (no emojis).
    """
    prop_id = benchmark_result.get("property_id", benchmark_result.get("Property", "N/A"))
    cluster_id = benchmark_result.get("cluster_id", benchmark_result.get("Cluster", "N/A"))
    aspect = benchmark_result.get("aspect", benchmark_result.get("Aspect", "N/A"))

    prop_score = benchmark_result.get("property_score", benchmark_result.get("Current Score", 0.0))
    cluster_mean = benchmark_result.get("cluster_mean", benchmark_result.get("Cluster Mean", 0.0))
    gap = benchmark_result.get("gap", benchmark_result.get("Gap", 0.0))

    flag = benchmark_result.get("flag", benchmark_result.get("Flag", "Neutral"))
    classification = benchmark_result.get("classification", benchmark_result.get("Classification", "property-specific"))

    raw_cp = benchmark_result.get("changepoint", benchmark_result.get("Changepoint", "None detected"))
    if isinstance(raw_cp, dict):
        cp_date = raw_cp.get("date", "N/A")
        cp_dir = raw_cp.get("direction", "")
        changepoint_str = f"{cp_date} ({cp_dir})" if cp_dir else str(cp_date)
    else:
        changepoint_str = str(raw_cp)

    def fmt(val: Any) -> str:
        if isinstance(val, float):
            return f"{val:.1f}" if abs(val) >= 1.0 or val == 0.0 else f"{val:.2f}"
        return str(val)

    if flag == "Lag":
        if classification == "property-specific":
            rec = (
                f"Your {aspect} sentiment has declined steadily and is now significantly below your cluster. "
                f"This is a property-specific issue, not a market trend. Prioritize kitchen staff review and "
                f"recipe consistency audits before next quarter."
                if aspect == "Food" else
                f"Your {aspect} sentiment has declined steadily and is now significantly below your cluster. "
                f"This is a property-specific issue, not a market trend. Prioritize staff review and operational "
                f"quality audits for {aspect} before next quarter."
            )
        else:
            rec = (
                f"Your {aspect} sentiment is lagging alongside the broader cluster trend. "
                f"This indicates a market-wide shift in consumer sentiment. Benchmark regional competitors and "
                f"adjust service standards for {aspect}."
            )
    elif flag == "Lead":
        if classification == "property-specific":
            rec = (
                f"Your {aspect} performance is significantly outperforming cluster peers due to property-specific "
                f"operational strengths. Maintain existing quality controls and highlight {aspect} in marketing campaigns."
            )
        else:
            rec = (
                f"Your {aspect} sentiment leads the cluster amidst positive market-wide demand. "
                f"Continue capitalizing on regional momentum while reinforcing core service standards for {aspect}."
            )
    else:
        rec = (
            f"Your {aspect} sentiment aligns with cluster averages. "
            f"Continue monitoring customer feedback for subtle shift patterns and maintain steady service delivery."
        )

    report = (
        "---\n"
        "ACTION REPORT\n"
        f"Property: {prop_id} (Cluster {cluster_id})\n"
        f"Aspect: {aspect}\n"
        f"Current Score: {fmt(prop_score)} | Cluster Mean: {fmt(cluster_mean)} | Gap: {fmt(gap)}\n"
        f"Flag: {flag}\n"
        f"Classification: {classification}\n"
        f"Changepoint: {changepoint_str}\n\n"
        "Recommended Action:\n"
        f"{rec}\n"
        "---"
    )

    return report


if __name__ == "__main__":
    # Test script on synthetic data
    from src.cri.loader import load_reviews
    from src.cri.classify import RuleBasedClassifier

    print("--- Testing Benchmarker ---")
    reviews_df = load_reviews("data/synthetic/reviews.csv")
    classifier = RuleBasedClassifier()

    # Apply classifier to get Food scores
    food_scores = []
    for text in reviews_df["review_text"]:
        res = classifier.classify(text)
        food_scores.append(res["Food"]["score"])

    reviews_df["Food"] = food_scores

    means = compute_cluster_means(reviews_df, "Food")
    print("Cluster Food Means:\n", means)

    benchmark_p1 = benchmark_property("P1", "C1", reviews_df, aspect="Food")
    print("\nBenchmark P1 (Injected Food Decline):\n", benchmark_p1)

    print("\nAction Report for P1 Food:")
    print(generate_action_report(benchmark_p1))

