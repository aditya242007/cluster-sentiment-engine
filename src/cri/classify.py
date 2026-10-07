from pathlib import Path
from typing import Any, Dict, List
import yaml

from src.cri.normalize import normalize_text


class RuleBasedClassifier:
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
        self._compiled_rules: Dict[str, Dict[str, List[str]]] = {}

        # Parse positive and negative cue phrases per aspect across all languages
        for aspect in self.aspects:
            name = aspect["name"]
            pos_phrases: List[str] = []
            neg_phrases: List[str] = []

            for lang, phrases in aspect.get("cue_phrases", {}).items():
                for phrase in phrases:
                    if phrase.startswith("POS:"):
                        clean_p = normalize_text(phrase[4:].strip())
                        if clean_p:
                            pos_phrases.append(clean_p)
                    elif phrase.startswith("NEG:"):
                        clean_p = normalize_text(phrase[4:].strip())
                        if clean_p:
                            neg_phrases.append(clean_p)

            self._compiled_rules[name] = {
                "positive": sorted(set(pos_phrases), key=len, reverse=True),
                "negative": sorted(set(neg_phrases), key=len, reverse=True),
            }

    def classify(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Classify review text across 8 aspects, computing sentiment and matched cue phrases.

        Args:
            text: Raw or normalized review text.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of aspect name to sentiment result:
              {
                "sentiment": "positive" | "negative" | "neutral",
                "score": float,
                "matched_phrases": list of matched cue phrases
              }
        """
        norm_text = normalize_text(text)
        results: Dict[str, Dict[str, Any]] = {}

        for aspect in self.aspects:
            name = aspect["name"]
            pos_cues = self._compiled_rules[name]["positive"]
            neg_cues = self._compiled_rules[name]["negative"]

            matched_pos: List[str] = []
            matched_neg: List[str] = []

            for cue in pos_cues:
                if cue in norm_text:
                    matched_pos.append(cue)

            for cue in neg_cues:
                if cue in norm_text:
                    matched_neg.append(cue)

            pos_count = len(matched_pos)
            neg_count = len(matched_neg)
            total_matches = pos_count + neg_count

            if total_matches == 0:
                score = 0.0
                sentiment = "neutral"
            else:
                score = float((pos_count - neg_count) / max(1, total_matches))
                if score > 0.1:
                    sentiment = "positive"
                elif score < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

            all_matched = matched_pos + matched_neg

            results[name] = {
                "sentiment": sentiment,
                "score": score,
                "matched_phrases": all_matched,
            }

        return results


if __name__ == "__main__":
    classifier = RuleBasedClassifier()

    sample_reviews = [
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
