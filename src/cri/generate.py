"""Synthetic review generator with injected patterns for pipeline validation."""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


ASPECT_TEMPLATES = {
    "Food": {
        "pos": [
            "The food was delicious and fresh",
            "Amazing breakfast, khana bahut accha tha",
            "Great taste, loved the local cuisine",
            "Excellent food quality and variety",
        ],
        "neg": [
            "The food was cold and tasteless",
            "Khana kharab tha, very disappointed",
            "Terrible food quality, stale items",
            "Poor breakfast, bad taste",
        ],
    },
    "Room": {
        "pos": [
            "Spacious and clean room",
            "Room was comfortable and well maintained",
            "Well maintained room, accha tha",
        ],
        "neg": [
            "Room was dirty and cramped",
            "Room ganda tha, very disappointing",
            "Poorly maintained room, needs repair",
        ],
    },
    "Staff": {
        "pos": [
            "Staff was very helpful and polite",
            "Friendly staff, madad kiye",
            "Staff ne bahut help kiya",
        ],
        "neg": [
            "Rude and unhelpful staff",
            "Staff was slow and unfriendly",
            "Staff behavior was disappointing",
        ],
    },
    "Housekeeping": {
        "pos": [
            "Housekeeping was prompt and thorough",
            "Daily cleaning was done well",
        ],
        "neg": [
            "Housekeeping never came during stay",
            "Room was not cleaned properly",
        ],
    },
    "Location": {
        "pos": [
            "Great location near the market",
            "Location was perfect for sightseeing",
        ],
        "neg": [
            "Location was far from everything",
            "Bad location, hard to find",
        ],
    },
    "Value for Money": {
        "pos": [
            "Great value for money",
            "Worth every rupee, paisa vasool",
        ],
        "neg": [
            "Overpriced for what you get",
            "Not worth the money",
        ],
    },
    "Cleanliness": {
        "pos": [
            "Very clean and hygienic, safai acchi thi",
            "Spotlessly clean property",
        ],
        "neg": [
            "Dirty washrooms and stained bedsheets",
            "Property was not clean at all",
        ],
    },
    "Booking Experience": {
        "pos": [
            "Booking was easy and smooth",
            "Check-in was seamless",
        ],
        "neg": [
            "Booking process was confusing",
            "Check-in was slow",
        ],
    },
}


def _pick_review(aspect: str, positive_probability: float) -> str:
    polarity = "pos" if random.random() < positive_probability else "neg"
    return random.choice(ASPECT_TEMPLATES[aspect][polarity])


def generate_synthetic_reviews(n_reviews: int = 5000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    properties = [f"P{i}" for i in range(1, 13)]
    cluster_map = {f"P{i}": f"C{((i - 1) // 4) + 1}" for i in range(1, 13)}

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    total_days = (end_date - start_date).days

    rows = []
    for _ in range(n_reviews):
        property_id = random.choice(properties)
        cluster_id = cluster_map[property_id]
        review_date = start_date + timedelta(days=random.randint(0, total_days))
        month = review_date.month

        aspect = random.choice(list(ASPECT_TEMPLATES.keys()))
        pos_prob = 0.75

        # INJECTION 1: P1 Food declines month 1 to 6, stays low
        if property_id == "P1":
            aspect = "Food"
            pos_prob = max(0.1, 0.9 - (month - 1) * 0.16)

        # INJECTION 2: C2 Room dips in winter (Nov-Feb)
        elif cluster_id == "C2" and random.random() < 0.7:
            aspect = "Room"
            pos_prob = 0.2 if month in (11, 12, 1, 2) else 0.85

        # INJECTION 3: P4 Staff improves after month 3
        elif property_id == "P4":
            aspect = "Staff"
            pos_prob = 0.15 if month <= 3 else 0.9

        review_text = _pick_review(aspect, pos_prob)
        rating = max(1.0, min(5.0, round(pos_prob * 4 + 1 + random.uniform(-0.4, 0.4), 1)))

        rows.append({
            "review_id": f"R{len(rows):05d}",
            "property_id": property_id,
            "cluster_id": cluster_id,
            "review_text": review_text,
            "rating": rating,
            "review_date": review_date,
            "stay_date": review_date,
            "primary_aspect": aspect,
            "polarity": "positive" if random.random() < pos_prob else "negative",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_synthetic_reviews(n_reviews=5000, seed=42)
    df.to_csv("data/synthetic/reviews.csv", index=False)
    print(f"Generated {len(df)} reviews and saved to data/synthetic/reviews.csv")
