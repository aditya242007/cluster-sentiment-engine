from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Review:
    """Represents an ingested individual hotel review."""

    review_id: str
    property_id: str
    cluster_id: str
    review_text: str
    rating: float
    review_date: datetime
    stay_date: Optional[datetime] = None
