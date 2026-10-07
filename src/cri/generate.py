from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


# ---- Aspect review pools -------------------------------------------------

FOOD_REVIEWS_POS = [
    "The food was delicious and fresh",
    "Amazing breakfast, khana bahut accha tha",
    "Great taste, loved the local cuisine",
    "Excellent food quality and variety",
    "Khana lajawab tha, really enjoyed the buffet",
    "Food was amazing, best breakfast spread",
    "Great food quality, dinner was excellent",
    "Tasty breakfast, food was great throughout",
]
FOOD_REVIEWS_NEG = [
    "The food was cold and tasteless",
    "Khana kharab tha, very disappointed",
    "Terrible food quality, stale items",
    "Poor breakfast, bad taste",
    "Food was cold and stale, totally bekar",
    "Worst dinner ever, food was terrible",
    "Cold food and oily, totally disappointing",
    "Stale food served, food was bad throughout",
]

ROOM_REVIEWS_POS = [
    "Spacious room with a breathtaking view",
    "Well maintained room, comfortable bed",
    "Loved the room aesthetics and modern amenities",
    "Super cozy room, peaceful sleep and clean linens",
    "Room was spacious and very comfortable",
    "Great room, comfortable bed and warm heating",
    "Room was clean and cozy throughout the stay",
]
ROOM_REVIEWS_NEG = [
    "Room was freezing cold, heater kharab tha",
    "Extremely cold room during winter, drafty windows and no heating",
    "Damp room with chilled floors, very uncomfortable stay",
    "No proper blanket and room was freezing at night",
    "Room was cold and uncomfortable, terrible experience",
    "Freezing cold room, no heating at all",
    "Room was dirty and damp, very disappointing",
]

STAFF_REVIEWS_POS = [
    "Staff was polite, welcoming, and helpful at every step",
    "Bahut courteous staff, front desk sorted check-in instantly",
    "Excellent service by the team, always smiling and supportive",
    "Staff behavior was top notch, cooperative and quick",
    "Staff was helpful and courteous throughout",
    "Polite staff and excellent hospitality",
    "Warm hospitality, staff was welcoming and cooperative",
]
STAFF_REVIEWS_NEG = [
    "Rude staff with zero manners, koi sunne wala nahi",
    "Staff was indifferent and unhelpful throughout the stay",
    "Poor hospitality, staff was arrogant and delayed everything",
    "Staff didn't care about our requests, worst customer service",
    "Rude staff, unhelpful desk team",
    "Staff was arrogant and uncooperative",
    "Unhelpful staff, terrible hospitality",
]

CLEANLINESS_REVIEWS_POS = [
    "Spotlessly clean property, bathrooms were sparkling tidy",
    "Safai ekdum top quality, fresh towels provided daily",
    "Very clean and hygienic, safai acchi thi",
    "Property was clean throughout, high hygiene standards",
]
CLEANLINESS_REVIEWS_NEG = [
    "Dirty washrooms and stained bedsheets, bilkul ganda",
    "Dusty furniture and dirty floors, safai bilkul nahi thi",
    "Unhygienic surroundings and foul smell in corridors",
    "Stained bedsheets and dirty washrooms, disgusting",
]

LOCATION_REVIEWS_POS = [
    "Convenient location close to scenic spots and mall road",
    "Location bahut acchi hai, easy access to transport",
    "Great location, mountain view was gorgeous",
    "Perfect location, easy access to local attractions",
]
LOCATION_REVIEWS_NEG = [
    "Isolated location, approaching road was broken and steep",
    "Very noisy location with heavy traffic outside",
    "Difficult to reach, location is far from city center",
    "Location was bad, noisy area and far from city",
]

VALUE_REVIEWS_POS = [
    "Total paisa vasool stay, high quality at reasonable price",
    "Affordable rates and premium facilities, great value",
    "Worth every penny, value for money",
    "Great value for money, reasonable price",
]
VALUE_REVIEWS_NEG = [
    "Overpriced property with mediocre service, not worth it",
    "Paisa barbad, heavy tariff for subpar quality",
    "Totally overpriced, hidden charges added at checkout",
    "Not worth the price, overpriced and poor service",
]

HOUSEKEEPING_REVIEWS_POS = [
    "Housekeeping was prompt and attended quickly to requests",
    "Daily room cleaning was meticulous and punctual",
    "Housekeeping service was great, samay par aayi",
    "Prompt housekeeping, room was always clean",
]
HOUSEKEEPING_REVIEWS_NEG = [
    "Housekeeping never turned up despite multiple calls",
    "Delayed housekeeping and dirty glasses left untouched",
    "Housekeeping dhili thi, request karne par bhi koi nahi aaya",
    "Housekeeping was very poor, never came on time",
]

BOOKING_REVIEWS_POS = [
    "Smooth check-in and easy advance booking",
    "No hassle during booking, instant confirmation received",
    "Booking was easy and check-in was very smooth",
    "Fast check-in, booking process was simple",
]
BOOKING_REVIEWS_NEG = [
    "Booking mix-up at reception, waited 2 hours for our room",
    "Discrepancy in reservation details, frustrating experience",
    "Check-in chaos and reservation was not found in system",
    "Booking problem and reservation nahi mila, terrible experience",
]

ASPECT_POOLS = {
    "Food":             (FOOD_REVIEWS_POS,        FOOD_REVIEWS_NEG),
    "Room":             (ROOM_REVIEWS_POS,        ROOM_REVIEWS_NEG),
    "Staff":            (STAFF_REVIEWS_POS,       STAFF_REVIEWS_NEG),
    "Cleanliness":      (CLEANLINESS_REVIEWS_POS, CLEANLINESS_REVIEWS_NEG),
    "Location":         (LOCATION_REVIEWS_POS,    LOCATION_REVIEWS_NEG),
    "Value for Money":  (VALUE_REVIEWS_POS,        VALUE_REVIEWS_NEG),
    "Housekeeping":     (HOUSEKEEPING_REVIEWS_POS, HOUSEKEEPING_REVIEWS_NEG),
    "Booking Experience": (BOOKING_REVIEWS_POS,   BOOKING_REVIEWS_NEG),
}

ALL_ASPECTS: List[str] = list(ASPECT_POOLS.keys())

PROPERTY_CLUSTER_MAP = {
    "P1": "C1", "P2": "C1", "P3": "C1", "P4": "C1",
    "P5": "C2", "P6": "C2", "P7": "C2", "P8": "C2",
    "P9": "C3", "P10": "C3", "P11": "C3", "P12": "C3",
}


def _pos_probability(prop_id: str, cluster_id: str, aspect: str, month: int) -> float:
    """Return the probability of a positive review given injected patterns.

    Three patterns are baked in:
      - P1 Food: declines from 0.85 in month 1 to 0.20 by month 6+.
      - C2 Room: drops to 0.25 in winter (Nov-Feb), 0.75 otherwise.
      - P4 Staff: 0.35 in months 1-3, rises to 0.88 from month 4 onward.
    """
    if prop_id == "P1" and aspect == "Food":
        if month <= 6:
            return max(0.20, 0.85 - (month - 1) * 0.12)
        return 0.20

    if cluster_id == "C2" and aspect == "Room":
        return 0.25 if month in (11, 12, 1, 2) else 0.75

    if prop_id == "P4" and aspect == "Staff":
        return 0.35 if month <= 3 else 0.88

    return 0.70


def generate_synthetic_reviews(n_reviews: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic multilingual tourism reviews with ground-truth injected patterns.

    Args:
        n_reviews: Number of reviews to generate.
        seed: Random seed for deterministic reproducibility.

    Returns:
        pd.DataFrame: Synthetic reviews conforming to the Review schema.
    """
    rng = np.random.default_rng(seed)
    properties = list(PROPERTY_CLUSTER_MAP.keys())
    start_date = datetime(2024, 1, 1)

    records = []
    for i in range(1, n_reviews + 1):
        prop_id = rng.choice(properties)
        cluster_id = PROPERTY_CLUSTER_MAP[prop_id]

        day_offset = int(rng.integers(0, 365))
        review_date = start_date + timedelta(days=day_offset)
        month = review_date.month

        stay_gap = int(rng.integers(0, 8))
        stay_date = review_date - timedelta(days=stay_gap)

        primary_aspect = rng.choice(ALL_ASPECTS)
        pos_prob = _pos_probability(prop_id, cluster_id, primary_aspect, month)
        is_positive = rng.random() < pos_prob
        polarity = "positive" if is_positive else "negative"

        pos_pool, neg_pool = ASPECT_POOLS[primary_aspect]
        pool = pos_pool if is_positive else neg_pool
        review_text = str(rng.choice(pool))

        if is_positive:
            rating = float(rng.choice([4.0, 5.0, 5.0, 4.0, 3.0], p=[0.35, 0.45, 0.10, 0.05, 0.05]))
        else:
            rating = float(rng.choice([1.0, 2.0, 1.0, 2.0, 3.0], p=[0.45, 0.35, 0.10, 0.05, 0.05]))

        records.append({
            "review_id": f"REV-{i:06d}",
            "property_id": prop_id,
            "cluster_id": cluster_id,
            "review_text": review_text,
            "rating": rating,
            "review_date": review_date,
            "stay_date": stay_date,
            "primary_aspect": primary_aspect,
            "polarity": polarity,
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="review_date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "reviews.csv"

    print("Generating synthetic reviews...")
    df_reviews = generate_synthetic_reviews(n_reviews=5000, seed=42)
    df_reviews.to_csv(out_file, index=False)
    print(f"Generated {len(df_reviews)} reviews and saved to {out_file}")
