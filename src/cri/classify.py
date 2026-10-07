"""Module defining the pluggable Classifier interface and rule-based classifier implementation.

Future transformer-based backends (e.g., fine-tuned mBERT, XLM-RoBERTa) or hybrid model backends
can implement the abstract `Classifier` interface, allowing swapping of aspect-sentiment
engine models without modifying downstream analytics or dashboard components.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import yaml

from src.cri.normalize import normalize_text


class Classifier(ABC):
    """Abstract base class establishing the pluggable contract for aspect classifiers."""

    @abstractmethod
    def classify(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Classify input review text across aspects.

        Args:
            text: Raw or normalized review text.

        Returns:
            Dict keyed by aspect name containing sentiment, score, matched_phrases, review_count.
        """
        pass


class RuleBasedClassifier(Classifier):
    """Explainable rule/lexicon-based aspect sentiment classifier."""

    def __init__(self, aspects_path: str = "src/cri/resources/aspects.yaml"):
        """Load aspect taxonomy and cue phrases from YAML configuration.

        Args:
            aspects_path: Path to the aspects YAML file.

        Raises:
            FileNotFoundError: If aspects_path does not exist.
        """
        path = Path(aspects_path)
        if not path.is_file():
            raise FileNotFoundError(f"Aspects config not found at: {aspects_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.aspects: List[Dict[str, Any]] = data.get("aspects", [])

    def classify(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Classify review text across all aspects using direct phrase matching.

        Matches positive and negative cue phrases from english, hinglish, and devanagari
        sub-keys. Score is (pos - neg) / total, ranging from -1.0 to +1.0.

        Args:
            text: Raw or normalized review text.

        Returns:
            Dict keyed by aspect name with keys: sentiment, score, matched_phrases, review_count.
        """
        normalized = normalize_text(text).lower()
        results: Dict[str, Dict[str, Any]] = {}

        for aspect in self.aspects:
            name = aspect["name"]
            cues = aspect.get("cue_phrases", {})

            pos_matches: List[str] = []
            neg_matches: List[str] = []

            for phrase in cues.get("english_pos", []):
                if phrase.lower() in normalized:
                    pos_matches.append(phrase)
            for phrase in cues.get("english_neg", []):
                if phrase.lower() in normalized:
                    neg_matches.append(phrase)
            for phrase in cues.get("hinglish_pos", []):
                if phrase.lower() in normalized:
                    pos_matches.append(phrase)
            for phrase in cues.get("hinglish_neg", []):
                if phrase.lower() in normalized:
                    neg_matches.append(phrase)
            # Devanagari phrases are matched against original text (not lowercased normalized)
            original_lower = text.lower()
            for phrase in cues.get("devanagari_pos", []):
                if phrase in text:
                    pos_matches.append(phrase)
            for phrase in cues.get("devanagari_neg", []):
                if phrase in text:
                    neg_matches.append(phrase)

            pos = len(pos_matches)
            neg = len(neg_matches)
            total = pos + neg

            if total == 0:
                score = 0.0
                sentiment = "neutral"
            else:
                score = (pos - neg) / total
                sentiment = "positive" if score > 0.15 else ("negative" if score < -0.15 else "neutral")

            results[name] = {
                "sentiment": sentiment,
                "score": score,
                "matched_phrases": pos_matches + [f"-{m}" for m in neg_matches],
                "review_count": total,
            }

        return results


if __name__ == "__main__":
    classifier = RuleBasedClassifier()

    sample_reviews = [
        "The food was excellent and the room was dirty",
        "Khana bahut lajawab aur tasty tha, loved the breakfast buffet! But room was freezing cold.",
        "कमरा बहुत साफ और आरामदायक था, staff was polite and helpful.",
        "Rude staff with zero manners, overpriced property, totally paisa barbad.",
    ]

    print("--- RuleBasedClassifier Test Run ---")
    for idx, review in enumerate(sample_reviews, 1):
        print(f"\nReview {idx}: '{review}'")
        res = classifier.classify(review)
        for aspect_name, details in res.items():
            if details["matched_phrases"]:
                print(
                    f"  - [{aspect_name}] Sentiment: {details['sentiment']} "
                    f"(Score: {details['score']:.2f}) | Matches: {details['matched_phrases']}"
                )
