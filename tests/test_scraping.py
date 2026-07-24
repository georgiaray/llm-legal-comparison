"""Unit tests for utils/scraping.py pure functions.

These tests avoid the network entirely: sanitize_filename and is_pdf_response
are pure functions, and download_and_extract / main() are left untested here
since they require live HTTP calls (integration-tested instead by
test_pipeline.py).
"""
from utils.scraping import sanitize_filename, is_pdf_response


class FakeResponse:
    """Minimal stand-in for requests.Response, just enough for is_pdf_response."""

    def __init__(self, content_type: str = ""):
        self.headers = {"Content-Type": content_type}


class TestSanitizeFilename:
    def test_removes_invalid_characters(self):
        assert sanitize_filename('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"

    def test_leaves_normal_filenames_untouched(self):
        assert sanitize_filename("canada_climate_policy_2024") == "canada_climate_policy_2024"

    def test_strips_surrounding_whitespace(self):
        assert sanitize_filename("  spaced name  ") == "spaced name"

    def test_empty_string(self):
        assert sanitize_filename("") == ""


class TestIsPdfResponse:
    def test_pdf_content_type_non_pdf_url(self):
        response = FakeResponse("application/pdf")
        assert is_pdf_response(response, "https://example.com/download?id=123") is True

    def test_pdf_url_html_content_type(self):
        response = FakeResponse("text/html")
        assert is_pdf_response(response, "https://example.com/document.pdf") is True

    def test_pdf_url_with_query_string(self):
        response = FakeResponse("text/html")
        assert is_pdf_response(response, "https://example.com/document.pdf?v=2") is True

    def test_uppercase_pdf_extension(self):
        response = FakeResponse("text/html")
        assert is_pdf_response(response, "https://example.com/document.PDF") is True

    def test_neither_pdf_content_type_nor_url(self):
        response = FakeResponse("text/html; charset=utf-8")
        assert is_pdf_response(response, "https://example.com/page.html") is False

    def test_x_pdf_content_type(self):
        response = FakeResponse("application/x-pdf")
        assert is_pdf_response(response, "https://example.com/download") is True
