import pytest
from src.cri.normalize import normalize_text


def test_normalize_english_basic():
    text = "The room was VERY spacious and clean!"
    result = normalize_text(text)
    assert result == "the room was very spacious and clean!"


def test_normalize_hinglish_transliteration():
    text = "Khana badhiya tha par room me thanda tha"
    result = normalize_text(text)
    assert "food" in result
    assert "great" in result
    assert "cold" in result


def test_normalize_devanagari_preserved():
    text = "खाना बहुत अच्छा था और कमरा साफ था"
    result = normalize_text(text)
    assert "खाना" in result
    assert "अच्छा" in result
    assert "कमरा" in result


def test_normalize_emoji_removal():
    text = "Great view! 😍 Beautiful sunset 🌅🌄"
    result = normalize_text(text)
    assert "😍" not in result
    assert "🌅" not in result
    assert "great view! beautiful sunset" in result


def test_normalize_negation_not_good():
    text = "The service was not good at all"
    result = normalize_text(text)
    assert "bad" in result
    assert "not good" not in result


def test_normalize_negation_not_clean():
    text = "Washroom was not clean"
    result = normalize_text(text)
    assert "dirty" in result
    assert "not clean" not in result


def test_normalize_whitespace_cleanup():
    text = "  Lots   of    extra   spaces \n\t here  "
    result = normalize_text(text)
    assert result == "lots of extra spaces here"


def test_normalize_empty_string():
    assert normalize_text("") == ""


def test_normalize_none_input():
    assert normalize_text(None) == ""


def test_normalize_code_mixed_hinglish_devanagari():
    text = "Khana 🍲 tha lajawab और staff helpful tha"
    result = normalize_text(text)
    assert "food" in result
    assert "delicious" in result
    assert "और" in result
    assert "helpful" in result


def test_normalize_paisa_vasool():
    text = "Total paisa vasool hotel!"
    result = normalize_text(text)
    assert "money" in result
    assert "worth" in result


def test_normalize_numbers_and_punctuation():
    text = "Room #404 was 100% freezing cold!!!"
    result = normalize_text(text)
    assert "room #404 was 100% freezing cold!!!" in result
