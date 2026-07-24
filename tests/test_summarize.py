"""Unit tests for the token-budget math in utils/summarize.py.

tiktoken downloads its BPE vocabulary from a remote blob store on first use,
which makes it unsuitable for offline/sandboxed unit tests. We monkeypatch
tiktoken.encoding_for_model / tiktoken.get_encoding with a trivial
word-count-based fake encoding so the *logic* of these functions (budget
arithmetic, truncation) is verified without any network dependency.
"""
import pytest

from utils import summarize as summarize_module
from utils.summarize import (
    prompt_token_count,
    available_tokens_for_doc,
    truncate_text_to_token_limit,
)


class FakeEncoding:
    """One 'token' per whitespace-separated word, for deterministic testing."""

    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture
def fake_tiktoken(monkeypatch):
    """Replace tiktoken's model/encoding lookups with the fake encoding."""
    fake = FakeEncoding()
    monkeypatch.setattr(summarize_module.tiktoken, "encoding_for_model", lambda name: fake)
    monkeypatch.setattr(summarize_module.tiktoken, "get_encoding", lambda name: fake)
    return fake


@pytest.fixture
def fake_tiktoken_unknown_model(monkeypatch):
    """Simulate tiktoken not recognizing the model name (falls back to get_encoding)."""
    fake = FakeEncoding()

    def raise_key_error(name):
        raise KeyError(name)

    monkeypatch.setattr(summarize_module.tiktoken, "encoding_for_model", raise_key_error)
    monkeypatch.setattr(summarize_module.tiktoken, "get_encoding", lambda name: fake)
    return fake


class TestPromptTokenCount:
    def test_counts_words_as_tokens(self, fake_tiktoken):
        assert prompt_token_count("one two three four") == 4

    def test_empty_prompt(self, fake_tiktoken):
        assert prompt_token_count("") == 0

    def test_falls_back_to_get_encoding_on_unknown_model(self, fake_tiktoken_unknown_model):
        assert prompt_token_count("one two three", encoding="not-a-real-model") == 3


class TestAvailableTokensForDoc:
    def test_subtracts_largest_prompt_overhead_and_margin(self, fake_tiktoken):
        prompts = ["a b c", "a b c d e"]  # 3 and 5 "tokens"
        result = available_tokens_for_doc(prompts, max_tokens=100, safety_margin=10)
        # largest prompt overhead = 5, so 100 - 5 - 10 = 85
        assert result == 85

    def test_empty_prompt_list_has_no_overhead(self, fake_tiktoken):
        result = available_tokens_for_doc([], max_tokens=100, safety_margin=10)
        assert result == 90

    def test_can_go_negative_if_prompts_exceed_budget(self, fake_tiktoken):
        prompts = [" ".join(["word"] * 50)]
        result = available_tokens_for_doc(prompts, max_tokens=40, safety_margin=5)
        assert result == 40 - 50 - 5


class TestTruncateTextToTokenLimit:
    def test_short_text_is_unchanged(self, fake_tiktoken):
        text = "one two three"
        assert truncate_text_to_token_limit(text, max_tokens=10) == text

    def test_long_text_is_truncated_to_exact_limit(self, fake_tiktoken):
        text = " ".join(f"w{i}" for i in range(20))
        truncated = truncate_text_to_token_limit(text, max_tokens=5)
        assert truncated.split() == ["w0", "w1", "w2", "w3", "w4"]

    def test_truncation_at_exact_boundary_is_unchanged(self, fake_tiktoken):
        text = " ".join(f"w{i}" for i in range(5))
        assert truncate_text_to_token_limit(text, max_tokens=5) == text
