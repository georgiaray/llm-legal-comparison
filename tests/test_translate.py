"""Unit tests for utils/translate.py.

detect_language uses langdetect, which ships its language profiles locally and
needs no network access, so it is exercised directly. Anything that would
otherwise depend on NLTK's downloaded 'punkt' data (sent_tokenize) or a live
translation API call is monkeypatched, so this suite runs offline and
deterministically in CI.
"""
import pytest

from utils import translate as translate_module
from utils.translate import (
    chinese_sent_tokenize,
    map_language_code,
    detect_language,
    process_text,
)


class TestChineseSentTokenize:
    def test_splits_on_chinese_punctuation(self):
        text = "这是第一句。这是第二句！这是第三句？"
        sentences = chinese_sent_tokenize(text)
        assert sentences == ["这是第一句。", "这是第二句！", "这是第三句？"]

    def test_filters_very_short_fragments(self):
        text = "AB。这是一句正常长度的句子。"
        sentences = chinese_sent_tokenize(text)
        assert "AB。" not in sentences
        assert "这是一句正常长度的句子。" in sentences

    def test_empty_text(self):
        assert chinese_sent_tokenize("") == []


class TestMapLanguageCode:
    def test_simplified_chinese_default(self):
        assert map_language_code("zh") == "zh-CN"

    def test_case_insensitive_lookup(self):
        assert map_language_code("ZH-TW") == "zh-TW"

    def test_passthrough_for_unmapped_codes(self):
        assert map_language_code("fr") == "fr"
        assert map_language_code("pt") == "pt"


class TestDetectLanguage:
    def test_detects_english(self):
        text = "This is a clearly English sentence used for a unit test."
        assert detect_language(text) == "en"

    def test_detects_french(self):
        text = "Ceci est une phrase clairement en français pour un test unitaire."
        assert detect_language(text) == "fr"

    def test_raises_on_empty_text(self):
        with pytest.raises(ValueError):
            detect_language("")

    def test_raises_on_whitespace_only(self):
        with pytest.raises(ValueError):
            detect_language("   ")


class TestProcessText:
    def test_empty_text_returns_unknown(self):
        processed, lang = process_text("", mode="detect_only")
        assert processed is None
        assert lang == "unknown"

    def test_detect_only_mode_returns_text_unchanged(self):
        text = "This is an English sentence for detection only."
        processed, lang = process_text(text, mode="detect_only")
        assert processed == text
        assert lang == "en"

    def test_auto_mode_leaves_english_text_unchanged(self):
        text = "This is already English, so auto mode should not translate it."
        processed, lang = process_text(text, mode="auto")
        assert processed == text
        assert lang == "en"

    def test_filter_mode_uses_provided_source_lang(self, monkeypatch):
        # Avoid depending on NLTK's downloaded punkt tokenizer data in CI/offline
        # environments: monkeypatch sent_tokenize with a simple splitter, and stub
        # out per-sentence language detection.
        monkeypatch.setattr(
            translate_module, "sent_tokenize", lambda text: text.split(". ")
        )
        monkeypatch.setattr(
            translate_module, "detect", lambda sentence: "en" if "hello" in sentence else "fr"
        )

        text = "hello world. bonjour le monde"
        processed, lang = process_text(text, mode="filter", source_lang="en")
        assert lang == "en"
        assert "hello world" in processed
        assert "bonjour le monde" not in processed

    def test_translate_mode_without_translator_returns_original(self, monkeypatch):
        monkeypatch.setattr(translate_module, "HAS_TRANSLATOR", False)
        text = "Ceci est en français."
        processed, lang = process_text(text, mode="translate", source_lang="fr")
        assert processed == text
        assert lang == "fr"
