"""Unit tests for utils/embed.py: chunk_text and trim_non_content."""
from utils.embed import chunk_text, trim_non_content


class TestChunkText:
    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_text_shorter_than_chunk_size_is_one_chunk(self):
        text = "one two three four five"
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=75)
        assert chunks == [text]

    def test_chunk_count_and_overlap(self):
        # 10 words, chunk_size=4, overlap=2 -> stride of 2 words per chunk
        words = [f"w{i}" for i in range(10)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=4, chunk_overlap=2)

        # Every chunk should have at most chunk_size words
        for chunk in chunks:
            assert len(chunk.split()) <= 4

        # Consecutive chunks should share the overlapping words
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        assert first_words[-2:] == second_words[:2]

        # The chunks should collectively cover every word in order
        assert chunks[0].startswith("w0")
        assert chunks[-1].endswith(words[-1])

    def test_no_infinite_loop_when_overlap_gte_chunk_size(self):
        # chunk_size - chunk_overlap <= 0 must still make forward progress
        text = " ".join(f"w{i}" for i in range(20))
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=10)
        assert len(chunks) > 0
        # Guard against runaway chunking on pathological input
        assert len(chunks) <= 20

    def test_exact_multiple_of_chunk_size(self):
        words = [f"w{i}" for i in range(8)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=4, chunk_overlap=0)
        assert len(chunks) == 2
        assert chunks[0] == "w0 w1 w2 w3"
        assert chunks[1] == "w4 w5 w6 w7"


class TestTrimNonContent:
    def test_strips_skip_to_main_content_header(self):
        text = "Skip to main content\nThe actual law text starts here."
        cleaned = trim_non_content(text)
        assert "Skip to main content" not in cleaned
        assert "The actual law text starts here." in cleaned

    def test_strips_date_modified_footer(self):
        text = "The actual law text.\nDate modified: 2024-01-01"
        cleaned = trim_non_content(text)
        assert "Date modified" not in cleaned
        assert "The actual law text." in cleaned

    def test_strips_canada_ca_header_and_footer_together(self):
        text = (
            "Canada.ca\n"
            "Government of Canada\n"
            "Main content of the legal document follows here, describing the policy in detail.\n"
            "Report a problem or mistake on this page. Date modified: 2024-06-01"
        )
        cleaned = trim_non_content(text)
        assert "Canada.ca" not in cleaned
        assert "Report a problem" not in cleaned
        assert "Main content of the legal document" in cleaned

    def test_plain_text_without_boilerplate_is_preserved(self):
        text = "This document has no navigation chrome at all, just legal text."
        assert trim_non_content(text) == text

    def test_empty_text(self):
        assert trim_non_content("") == ""
