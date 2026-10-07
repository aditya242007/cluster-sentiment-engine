"""Integration tests that run the full pipeline on synthetic data and assert
that the three ground-truth injected patterns are recoverable with statistical significance.
"""

import pytest
import pandas as pd
from scipy import stats

from src.cri.generate import generate_synthetic_reviews
from src.cri.normalize import normalize_text
from src.cri.classify import RuleBasedClassifier


@pytest.fixture(scope="module")
def scored_df():
    """Run the full pipeline on 5000 synthetic reviews once per test module.

    The DataFrame includes primary_aspect and polarity from the generator,
    allowing precise filtering of Food/Room/Staff reviews in integration tests.
    """
    df = generate_synthetic_reviews(n_reviews=5000, seed=42)
    classifier = RuleBasedClassifier()

    food_scores, room_scores, staff_scores = [], [], []
    for text in df["review_text"]:
        result = classifier.classify(text)
        food_scores.append(result["Food"]["score"])
        room_scores.append(result["Room"]["score"])
        staff_scores.append(result["Staff"]["score"])

    df["Food_score"] = food_scores
    df["Room_score"] = room_scores
    df["Staff_score"] = staff_scores
    df["month"] = pd.to_datetime(df["review_date"]).dt.month

    return df


# ── Pattern 1: Property P1 Food Decline ──────────────────────────────────────

def test_p1_food_month1_vs_month6_significant_decline(scored_df):
    """P1's early Food-aspect ratings (months 2-3) must be significantly higher than late (months 5-6).

    Uses primary_aspect column to precisely filter Food reviews. Months 2-3 represent
    the high-sentiment early period; months 5-6 represent the injected decline period.
    """
    p1_food = scored_df[(scored_df["property_id"] == "P1") & (scored_df["primary_aspect"] == "Food")]

    early_ratings = p1_food[p1_food["month"].isin([2, 3])]["rating"].values
    late_ratings = p1_food[p1_food["month"].isin([5, 6])]["rating"].values

    assert len(early_ratings) > 0, "No P1 Food reviews found in early months (2-3)."
    assert len(late_ratings) > 0, "No P1 Food reviews found in late months (5-6)."

    _, p_value = stats.ttest_ind(early_ratings, late_ratings, alternative="greater")
    early_mean = early_ratings.mean()
    late_mean = late_ratings.mean()

    assert p_value < 0.05, (
        f"[Pattern 1 FAILED] P1 Food rating early months ({early_mean:.4f}) is not "
        f"significantly greater than late months ({late_mean:.4f}). p-value={p_value:.4f}"
    )


def test_p1_food_mean_month1_greater_than_month6(scored_df):
    """P1's early Food-aspect mean rating (months 2-3) must be greater than late period (months 5-6)."""
    p1_food = scored_df[(scored_df["property_id"] == "P1") & (scored_df["primary_aspect"] == "Food")]
    early_mean = p1_food[p1_food["month"].isin([2, 3])]["rating"].mean()
    late_mean = p1_food[p1_food["month"].isin([5, 6])]["rating"].mean()

    assert early_mean > late_mean, (
        f"[Pattern 1 FAILED] P1 Food early mean ({early_mean:.4f}) is not > late mean ({late_mean:.4f})"
    )


def test_p1_food_steady_decline_months_1_to_6(scored_df):
    """P1's Food-aspect rating must decline from first half (months 1-3) to second half (months 4-6)."""
    p1_food = scored_df[(scored_df["property_id"] == "P1") & (scored_df["primary_aspect"] == "Food")]
    monthly_means = p1_food[p1_food["month"].between(1, 6)].groupby("month")["rating"].mean()

    assert len(monthly_means) >= 4, (
        f"[Pattern 1 FAILED] Not enough monthly Food data for P1 months 1-6: {monthly_means}"
    )

    first_half_mean = monthly_means.iloc[:3].mean()
    second_half_mean = monthly_means.iloc[3:].mean()

    assert first_half_mean > second_half_mean, (
        f"[Pattern 1 FAILED] P1 Food rating first-half mean ({first_half_mean:.4f}) is not "
        f"greater than second-half mean ({second_half_mean:.4f})"
    )


# ── Pattern 2: Cluster C2 Room Winter Dip ────────────────────────────────────

def test_c2_room_winter_lower_than_summer(scored_df):
    """Cluster C2's Room sentiment during winter months (Nov-Feb) must be lower than summer (Jun-Sep)."""
    c2 = scored_df[scored_df["cluster_id"] == "C2"]

    winter_room = c2[c2["month"].isin([11, 12, 1, 2])]["Room_score"].values
    summer_room = c2[c2["month"].isin([6, 7, 8, 9])]["Room_score"].values

    assert len(winter_room) > 0, "No C2 reviews found in winter months."
    assert len(summer_room) > 0, "No C2 reviews found in summer months."

    winter_mean = winter_room.mean()
    summer_mean = summer_room.mean()

    assert winter_mean < summer_mean, (
        f"[Pattern 2 FAILED] C2 Room winter mean ({winter_mean:.4f}) is not "
        f"less than summer mean ({summer_mean:.4f})"
    )


def test_c2_room_winter_vs_summer_significant(scored_df):
    """C2's winter Room dip vs summer must be statistically significant (p < 0.05)."""
    c2 = scored_df[scored_df["cluster_id"] == "C2"]

    winter_room = c2[c2["month"].isin([11, 12, 1, 2])]["Room_score"].values
    summer_room = c2[c2["month"].isin([6, 7, 8, 9])]["Room_score"].values

    _, p_value = stats.ttest_ind(winter_room, summer_room, alternative="less")
    winter_mean = winter_room.mean()
    summer_mean = summer_room.mean()

    assert p_value < 0.05, (
        f"[Pattern 2 FAILED] C2 Room winter ({winter_mean:.4f}) vs summer ({summer_mean:.4f}) "
        f"difference is not significant. p-value={p_value:.4f}"
    )


def test_c2_room_winter_gap_meaningful(scored_df):
    """C2's Room winter mean must be at least 0.1 points below summer mean."""
    c2 = scored_df[scored_df["cluster_id"] == "C2"]
    winter_mean = c2[c2["month"].isin([11, 12, 1, 2])]["Room_score"].mean()
    summer_mean = c2[c2["month"].isin([6, 7, 8, 9])]["Room_score"].mean()

    gap = summer_mean - winter_mean
    assert gap > 0.1, (
        f"[Pattern 2 FAILED] C2 Room winter gap ({gap:.4f}) is too small to be meaningful."
    )


# ── Pattern 3: Property P4 Staff Improvement ─────────────────────────────────

def test_p4_staff_improves_after_month3(scored_df):
    """P4's Staff sentiment mean in months 4+ must be higher than in months 1-3."""
    p4 = scored_df[scored_df["property_id"] == "P4"]

    early_staff = p4[p4["month"].isin([1, 2, 3])]["Staff_score"].values
    late_staff = p4[p4["month"].isin([4, 5, 6, 7, 8, 9, 10, 11, 12])]["Staff_score"].values

    assert len(early_staff) > 0, "No P4 reviews found in months 1-3."
    assert len(late_staff) > 0, "No P4 reviews found in months 4+."

    early_mean = early_staff.mean()
    late_mean = late_staff.mean()

    assert late_mean > early_mean, (
        f"[Pattern 3 FAILED] P4 Staff mean in months 4+ ({late_mean:.4f}) is not "
        f"greater than months 1-3 ({early_mean:.4f})"
    )


def test_p4_staff_improvement_significant(scored_df):
    """P4's Staff improvement after month 3 must be statistically significant (p < 0.05)."""
    p4 = scored_df[scored_df["property_id"] == "P4"]

    early_staff = p4[p4["month"].isin([1, 2, 3])]["Staff_score"].values
    late_staff = p4[p4["month"].isin([4, 5, 6, 7, 8, 9, 10, 11, 12])]["Staff_score"].values

    _, p_value = stats.ttest_ind(late_staff, early_staff, alternative="greater")
    early_mean = early_staff.mean()
    late_mean = late_staff.mean()

    assert p_value < 0.05, (
        f"[Pattern 3 FAILED] P4 Staff improvement months 4+ ({late_mean:.4f}) vs "
        f"months 1-3 ({early_mean:.4f}) is not significant. p-value={p_value:.4f}"
    )
