import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from clean_trial import clean_html, clean_pdf_pages, format_source_metadata, normalize_plain_text


class HtmlCleaningTests(unittest.TestCase):
    def test_removes_obvious_site_chrome_but_keeps_article_content(self):
        html = """
        <html><head><title>Ethical Requirements - IEEE Author Center Journals</title></head><body>
          <nav>Global navigation</nav>
          <div class="osano-cm-window">Cookie preferences</div>
          <main><!-- Google Tag Manager (noscript) -->
            <a class="skip-link" href="#main">Skip to content</a>
            <h1>Ethical Requirements</h1>
            <p>Authors must disclose conflicts.</p>
            <ul><li>Keep the author guidance.</li></ul>
            <table><tr><th>Type</th><th>Rule</th></tr>
              <tr><td>Data</td><td>Report sources.</td></tr></table>
          </main>
          <footer>Copyright navigation</footer>
        </body></html>
        """
        cleaned, removed = clean_html(html)
        self.assertIn("Ethical Requirements", cleaned)
        self.assertEqual(cleaned.count("# Ethical Requirements"), 1)
        self.assertIn("Authors must disclose conflicts.", cleaned)
        self.assertIn("- Keep the author guidance.", cleaned)
        self.assertIn("Type | Rule", cleaned)
        self.assertIn("Data | Report sources.", cleaned)
        self.assertNotIn("Global navigation", cleaned)
        self.assertNotIn("Google Tag Manager", cleaned)
        self.assertNotIn("Skip to content", cleaned)
        self.assertNotIn("Cookie preferences", cleaned)
        self.assertNotIn("Copyright navigation", cleaned)
        self.assertGreaterEqual(len(removed), 3)


class PdfCleaningTests(unittest.TestCase):
    def test_removes_only_repeated_page_edge_lines_and_standalone_page_numbers(self):
        pages = [
            "IEEE ACCESS\nEligibility rules\nThe main text.\n1",
            "IEEE ACCESS\nEligibility details\nMore main text.\n2",
            "IEEE ACCESS\nExceptions\nAdditional main text.\n3",
        ]
        cleaned, removed = clean_pdf_pages(pages)
        self.assertEqual(cleaned, [
            "Eligibility rules\nThe main text.",
            "Eligibility details\nMore main text.",
            "Exceptions\nAdditional main text.",
        ])
        self.assertIn("IEEE ACCESS", removed)
        self.assertIn("1", removed)

    def test_keeps_a_year_even_when_it_is_near_a_page_edge(self):
        pages = [
            "IEEE ACCESS\nStyle rules\n2024\n1",
            "IEEE ACCESS\nOther rules\n2024\n2",
            "IEEE ACCESS\nMore rules\n2024\n3",
        ]
        cleaned, removed = clean_pdf_pages(pages)
        self.assertTrue(all("2024" in page for page in cleaned))
        self.assertNotIn("2024", removed)


class PlainTextCleaningTests(unittest.TestCase):
    def test_normalizes_whitespace_without_removing_template_examples(self):
        text = "Title\r\n\r\n\r\nSample author\r\n\r\n\r\nExample instructions"
        cleaned = normalize_plain_text(text)
        self.assertEqual(cleaned, "Title\n\nSample author\n\nExample instructions")

    def test_source_manifest_urls_and_times_are_carried_without_guessing(self):
        entry = {
            "source_aliases": [{
                "url": "https://example.org/policy",
                "final_url": "https://example.org/current-policy",
                "retrieved_at_utc": "2026-10-05T12:00:00Z",
                "last_modified_http": "Mon, 05 Oct 2026 10:00:00 GMT",
            }]
        }
        lines = format_source_metadata(entry)
        joined = "\n".join(lines)
        self.assertIn("https://example.org/policy", joined)
        self.assertIn("https://example.org/current-policy", joined)
        self.assertIn("2026-10-05T12:00:00Z", joined)
        self.assertIn("Mon, 05 Oct 2026 10:00:00 GMT", joined)

    def test_missing_source_metadata_is_marked_unknown(self):
        joined = "\n".join(format_source_metadata(None))
        self.assertIn("not recorded", joined)
        self.assertNotIn("2026-10-06", joined)


if __name__ == "__main__":
    unittest.main()
