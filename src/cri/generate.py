from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# Multi-lingual & code-mixed review templates categorized by aspect and polarity
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "Food": {
        "positive": [
            "Khana bahut lajawab aur tasty tha, loved the breakfast buffet!",
            "Excellent dining experience and delicious local food.",
            "खाना बहुत ही स्वादिष्ट था, especially the dinner thali.",
            "Great food and prompt room service for snacks.",
            "Breakfast spread was amazing, shandar quality!",
        ],
        "negative": [
            "Food quality was terrible and cold, bilkul bekar tha.",
            "Worst dinner ever, stale food and slow service.",
            "खाना बिल्कुल ठंडा और बेस्वाद था, total waste of money.",
            "Disappointing food, oily and unhygienic restaurant.",
            "Breakfast mein kuch khas nahi tha, khana kharab tha.",
        ],
    },
    "Room": {
        "positive": [
            "Spacious room with a breathtaking view, bahut cozy tha.",
            "Well maintained room, comfortable bed and warm ambiance.",
            "कमरा बहुत साफ और आरामदायक था, great heating facility.",
            "Loved the room aesthetics and modern amenities.",
            "Super cozy room, peaceful sleep and clean linens.",
        ],
        "negative": [
            "Room was freezing cold, heater kaam nahi kar raha tha.",
            "Extremely cold room during winter, drafty windows and no heating.",
            "कमरे में बहुत ठंड थी, geyser aur heater dono kharab the.",
            "Damp room with chilled floors, very uncomfortable stay.",
            "No proper blanket and room was freezing at night, bura haal tha.",
        ],
    },
    "Staff": {
        "positive": [
            "Staff was polite, welcoming, and helpful at every step.",
            "Bahut courteous staff, front desk sorted check-in instantly.",
            "होटल स्टाफ बहुत मददगार और विनम्र था, warm hospitality.",
            "Excellent service by the team, always smiling and supportive.",
            "Staff behavior was top notch, cooperative and quick.",
        ],
        "negative": [
            "Rude staff with zero manners, koi sunne ko tayar nahi tha.",
            "Staff was indifferent and unhelpful throughout the stay.",
            "स्टाफ का बर्ताव बहुत खराब था, uncooperative desk team.",
            "Poor hospitality, staff was arrogant and delayed everything.",
            "Staff didn't care about our requests, worst customer service.",
        ],
    },
    "Cleanliness": {
        "positive": [
            "Spotlessly clean property, bathrooms were sparkling tidy.",
            "Safai ekdum top quality, fresh towels provided daily.",
            "साफ-सफाई का पूरा ध्यान रखा गया था, very hygienic.",
        ],
        "negative": [
            "Dirty washrooms and stained bedsheets, bilkul ganda.",
            "Dusty furniture and dirty floors, safai bilkul nahi thi.",
            "Unhygienic surroundings and foul smell in corridors.",
        ],
    },
    "Location": {
        "positive": [
            "Convenient location close to scenic spots and mall road.",
            "Location bahut acchi hai, easy access to transport.",
            "होटल की लोकेशन बेहतरीन है, mountain view was gorgeous.",
        ],
        "negative": [
            "Isolated location, approaching road was broken and steep.",
            "Very noisy location with heavy traffic outside.",
            "Difficult to reach, location is far from city center.",
        ],
    },
    "Value for Money": {
        "positive": [
            "Total paisa vasool stay, high quality at reasonable price.",
            "Affordable rates and premium facilities, great value.",
            "कीमत के हिसाब से बहुत बढ़िया सुविधा, worth every penny.",
        ],
        "negative": [
            "Overpriced property with mediocre service, not worth it.",
            "Paisa barbad, heavy tariff for subpar quality.",
            "Totally overpriced, hidden charges added at checkout.",
        ],
    },
    "Housekeeping": {
        "positive": [
            "Housekeeping was prompt and attended quickly to requests.",
            "Daily room cleaning was meticulous and punctual.",
            "हाउसकीपिंग सर्विस समय पर और अच्छी थी।",
        ],
        "negative": [
            "Housekeeping never turned up despite multiple calls.",
            "Delayed housekeeping and dirty glasses left untouched.",
            "हाउसकीपिंग बहुत ढीली थी, request karne par bhi koi nahi aaya.",
        ],
    },
    "Booking Experience": {
        "positive": [
            "Smooth check-in and seamless advance booking.",
            "No hassle during booking, instant confirmation received.",
            "बुकिंग और चेक-इन प्रोसेस बहुत आसान था।",
        ],
        "negative": [
            "Booking mix-up at reception, waited 2 hours for our room.",
            "Discrepancy in reservation details, frustrating experience.",
            "Check-in chaos and reservation was not found in system.",
        ],
    },
}

ALL_ASPECTS: List[str] = list(TEMPLATES.keys())


def generate_synthetic_reviews(n_reviews: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic multilingual tourism reviews with ground-truth injected patterns.

    Clusters and Properties:
      - Cluster C1: P1, P2, P3, P4
      - Cluster C2: P5, P6, P7, P8
      - Cluster C3: P9, P10, P11, P12

    Injected Patterns:
      1. Property P1: Food sentiment declines steadily from month 1 to month 6.
      2. Cluster C2 (P5-P8): Room sentiment dips in winter (Nov, Dec, Jan, Feb).
      3. Property P4: Staff sentiment improves noticeably after month 3.

    Args:
        n_reviews: Number of reviews to generate.
        seed: Random seed for deterministic reproducibility.

    Returns:
        pd.DataFrame: Synthetic reviews conforming to the Review schema.
    """
    rng = np.random.default_rng(seed)

    property_cluster_map = {
        "P1": "C1", "P2": "C1", "P3": "C1", "P4": "C1",
        "P5": "C2", "P6": "C2", "P7": "C2", "P8": "C2",
        "P9": "C3", "P10": "C3", "P11": "C3", "P12": "C3",
    }
    properties = list(property_cluster_map.keys())

    # Start date spanning 12 complete months (Jan 1, 2024 to Dec 31, 2024)
    start_date = datetime(2024, 1, 1)

    records = []
    for i in range(1, n_reviews + 1):
        review_id = f"REV-{i:06d}"
        prop_id = rng.choice(properties)
        cluster_id = property_cluster_map[prop_id]

        # Uniform date selection over 365 days
        day_offset = int(rng.integers(0, 365))
        review_date = start_date + timedelta(days=day_offset)
        month = review_date.month  # 1 to 12

        # Stay date 0 to 7 days before review date
        stay_gap = int(rng.integers(0, 8))
        stay_date = review_date - timedelta(days=stay_gap)

        # Baseline probability of positive sentiment across aspects
        pos_prob = 0.70

        # Select primary aspect of review
        primary_aspect = rng.choice(ALL_ASPECTS)

        # ---------------- Injected Pattern 1: P1 Food Sentiment Decline ----------------
        if prop_id == "P1" and primary_aspect == "Food":
            if month <= 6:
                # Steeper decline month 1 (high positive) down to month 6 (mostly negative)
                pos_prob = max(0.08, 0.90 - (month - 1) * 0.16)
            else:
                pos_prob = 0.15

        # ---------------- Injected Pattern 2: C2 Winter Room Sentiment Dip ------------
        if cluster_id == "C2" and primary_aspect == "Room":
            # Winter months: Nov (11), Dec (12), Jan (1), Feb (2)
            if month in (11, 12, 1, 2):
                pos_prob = 0.12
            else:
                pos_prob = 0.75

        # ---------------- Injected Pattern 3: P4 Staff Improvement --------------------
        if prop_id == "P4" and primary_aspect == "Staff":
            if month <= 3:
                pos_prob = 0.20
            else:
                pos_prob = 0.92

        # Determine sentiment polarity based on adjusted probability
        is_positive = rng.random() < pos_prob
        polarity = "positive" if is_positive else "negative"

        # Select template
        template_pool = TEMPLATES[primary_aspect][polarity]
        review_text = rng.choice(template_pool)

        # Ratings loosely correlated with sentiment (Positive: 4-5, Negative: 1-2 with minor noise)
        if is_positive:
            rating = rng.choice([4.0, 5.0, 5.0, 4.0, 3.0], p=[0.35, 0.45, 0.10, 0.05, 0.05])
        else:
            rating = rng.choice([1.0, 2.0, 1.0, 2.0, 3.0], p=[0.45, 0.35, 0.10, 0.05, 0.05])

        records.append({
            "review_id": review_id,
            "property_id": prop_id,
            "cluster_id": cluster_id,
            "review_text": review_text,
            "rating": float(rating),
            "review_date": review_date,
            "stay_date": stay_date,
            "primary_aspect": primary_aspect,
            "polarity": polarity,
        })

    df = pd.DataFrame(records)
    # Sort chronologically by review_date
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
