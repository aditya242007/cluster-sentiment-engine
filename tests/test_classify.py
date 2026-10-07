import pytest
from src.cri.classify import RuleBasedClassifier


@pytest.fixture
def classifier():
    return RuleBasedClassifier()


def test_classify_positive_food(classifier):
    review = "Delicious food and excellent dining experience!"
    results = classifier.classify(review)
    assert results["Food"]["sentiment"] == "positive"
    assert results["Food"]["score"] > 0.1
    assert len(results["Food"]["matched_phrases"]) > 0


def test_classify_negative_room(classifier):
    review = "Freezing cold room with no heating and uncomfortable bed."
    results = classifier.classify(review)
    assert results["Room"]["sentiment"] == "negative"
    assert results["Room"]["score"] < -0.1
    assert "-no heating" in results["Room"]["matched_phrases"]


def test_classify_hinglish_positive(classifier):
    review = "Khana bahut lajawab tha, tasty tha!"
    results = classifier.classify(review)
    assert results["Food"]["sentiment"] == "positive"


def test_classify_hinglish_negative_value(classifier):
    review = "Total paisa barbad, overpriced property for the quality."
    results = classifier.classify(review)
    assert results["Value for Money"]["sentiment"] == "negative"


def test_classify_devanagari_room(classifier):
    review = "कमरा बहुत साफ और आरामदायक था।"
    results = classifier.classify(review)
    assert results["Room"]["sentiment"] == "positive"


def test_classify_no_aspect_keywords(classifier):
    review = "Yesterday was Wednesday and sun was shining brightly."
    results = classifier.classify(review)
    for aspect, details in results.items():
        assert details["sentiment"] == "neutral"
        assert details["score"] == 0.0
        assert details["matched_phrases"] == []


def test_classify_mixed_staff_and_cleanliness(classifier):
    review = "Polite staff but dirty washrooms and stained bedsheets."
    results = classifier.classify(review)
    assert results["Staff"]["sentiment"] == "positive"
    assert results["Cleanliness"]["sentiment"] == "negative"


def test_classify_rude_staff(classifier):
    review = "Rude staff with zero manners."
    results = classifier.classify(review)
    assert results["Staff"]["sentiment"] == "negative"


def test_classify_smooth_booking(classifier):
    review = "Seamless booking and smooth check-in process."
    results = classifier.classify(review)
    assert results["Booking Experience"]["sentiment"] == "positive"


def test_classify_location_scenic(classifier):
    review = "Convenient location close to scenic spots."
    results = classifier.classify(review)
    assert results["Location"]["sentiment"] == "positive"
