from typing import Any, Dict, List
import pandas as pd


def detect_changepoints(
    time_series: pd.Series,
    threshold: float = 1.5,
) -> List[Dict[str, Any]]:
    """Detect changepoints in a sentiment time series using a rolling CUSUM-based approach.

    Algorithm:
      1. Compute rolling mean and rolling std with a window of 4 periods.
      2. Flag any index where |value - rolling_mean| > threshold * rolling_std.
      3. Merge adjacent flagged indices into single changepoint events.
      4. For each merged event, report the peak-deviation date, direction, and magnitude.

    Args:
        time_series: A pd.Series with a DatetimePeriod or DatetimeIndex, values representing
                     aspect sentiment scores over time.
        threshold: Number of rolling std deviations to use as the detection threshold.

    Returns:
        List[Dict[str, Any]]: Detected changepoints, each with:
          - "date": str  — Index label of the peak deviation point.
          - "direction": "up" | "down" — Whether the score spiked upward or downward.
          - "magnitude": float — Absolute deviation from the rolling mean at the changepoint.

    Example:
        >>> s = pd.Series([0.5, 0.5, 0.5, 0.5, -0.8, -0.9, 0.5], index=range(7))
        >>> detect_changepoints(s, threshold=1.5)
        [{"date": 5, "direction": "down", "magnitude": ...}]
    """
    if time_series.empty or len(time_series) < 5:
        return []

    ts = time_series.copy()
    rolling_mean = ts.rolling(window=4, min_periods=2).mean()
    rolling_std = ts.rolling(window=4, min_periods=2).std()

    # Avoid division / comparison issues where std is zero or NaN
    rolling_std = rolling_std.fillna(0)

    # Compute deviations and identify flagged points
    deviation = ts - rolling_mean
    flagged = deviation.abs() > (threshold * rolling_std.clip(lower=1e-6))

    # Merge adjacent flagged indices into contiguous event windows
    changepoints: List[Dict[str, Any]] = []
    in_event = False
    event_indices: List[int] = []

    flagged_positions = list(flagged.values)
    index_labels = list(ts.index)

    for pos, is_flagged in enumerate(flagged_positions):
        if is_flagged:
            event_indices.append(pos)
            in_event = True
        else:
            if in_event and event_indices:
                # Resolve the peak deviation point within this event window
                peak_pos = max(
                    event_indices,
                    key=lambda p: abs(deviation.iloc[p]),
                )
                peak_deviation = float(deviation.iloc[peak_pos])
                changepoints.append({
                    "date": str(index_labels[peak_pos]),
                    "direction": "up" if peak_deviation > 0 else "down",
                    "magnitude": round(abs(peak_deviation), 4),
                })
                event_indices = []
                in_event = False

    # Handle an event that extends to the end of the series
    if in_event and event_indices:
        peak_pos = max(event_indices, key=lambda p: abs(deviation.iloc[p]))
        peak_deviation = float(deviation.iloc[peak_pos])
        changepoints.append({
            "date": str(index_labels[peak_pos]),
            "direction": "up" if peak_deviation > 0 else "down",
            "magnitude": round(abs(peak_deviation), 4),
        })

    return changepoints


if __name__ == "__main__":
    from src.cri.loader import load_reviews
    from src.cri.classify import RuleBasedClassifier

    print("--- Testing Changepoint Detector (P1 Food, monthly) ---")
    reviews_df = load_reviews("data/synthetic/reviews.csv")
    classifier = RuleBasedClassifier()

    food_scores = [classifier.classify(t)["Food"]["score"] for t in reviews_df["review_text"]]
    reviews_df["Food"] = food_scores
    reviews_df["month"] = pd.to_datetime(reviews_df["review_date"]).dt.to_period("M")

    p1_monthly = (
        reviews_df[reviews_df["property_id"] == "P1"]
        .groupby("month")["Food"]
        .mean()
    )

    cps = detect_changepoints(p1_monthly, threshold=1.5)
    if cps:
        for cp in cps:
            print(f"  Changepoint: {cp}")
    else:
        print("  No changepoints detected.")
