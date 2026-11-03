import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split_documents import (
    parse_cleaned_markdown,
    _load_embedding_model,
    split_corpus,
    split_html_structurally,
    split_pdf_pages,
)


class EmbeddingConfigurationTests(unittest.TestCase):
    def test_embedding_batch_size_respects_qwen_limit(self):
        config = {
            "EMBEDDING_API_KEY": "test-key",
            "EMBEDDING_BASE_URL": "https://example.org/v1",
            "EMBEDDING_MODEL": "qwen-embedding-test",
        }
        with patch.dict("os.environ", config), patch(
            "split_documents.OpenAIEmbedding"
        ) as embedding_constructor:
            _load_embedding_model()

        self.assertEqual(
            embedding_constructor.call_args.kwargs["embed_batch_size"], 20
        )


class ParseCleanedMarkdownTests(unittest.TestCase):
    def test_extracts_provenance_and_removes_cleaning_header(self):
        text = (
            "# Example policy\n\n"
            "- Source file: `official/policy.pdf`\n"
            "- Type: PDF\n"
            "- SHA-256: `abc123`\n"
            "- Source URL: `https://example.org/policy.pdf`\n"
            "- Source final URL: `https://example.org/policy.pdf`\n"
            "- Source acquisition time: `2026-10-05T00:00:00Z`\n"
            "- Source HTTP Last-Modified: `not recorded`\n"
            "- Cleaning: whitespace normalized\n\n"
            "## PDF page 1\n\nPolicy body.\n"
        )

        metadata, body = parse_cleaned_markdown(text)

        self.assertEqual(metadata["title"], "Example policy")
        self.assertEqual(metadata["file_type"], "PDF")
        self.assertEqual(metadata["source_url"], "https://example.org/policy.pdf")
        self.assertNotIn("Cleaning:", body)
        self.assertIn("Policy body.", body)


class SplitPdfPagesTests(unittest.TestCase):
    def test_keeps_page_numbers_and_page_text_separate(self):
        body = "## PDF page 2\n\nFirst page text.\n\n## PDF page 3\n\nSecond page text."

        pages = split_pdf_pages(body)

        self.assertEqual(pages, [(2, "First page text."), (3, "Second page text.")])


class SplitHtmlStructurallyTests(unittest.TestCase):
    def test_removes_site_chrome_and_attaches_heading_context(self):
        html = (
            "<html><body><nav><p>Navigation noise.</p></nav>"
            "<main><h1>Submission</h1><p>Keep this policy text.</p>"
            "<h2>Eligibility</h2><p>Keep this condition.</p></main>"
            "<footer><p>Footer noise.</p></footer></body></html>"
        )

        sections = split_html_structurally(html, {"title": "Submission"})
        rendered = "\n".join(section.text for section in sections)

        self.assertIn("Keep this policy text.", rendered)
        self.assertIn("Keep this condition.", rendered)
        self.assertNotIn("Navigation noise.", rendered)
        self.assertNotIn("Footer noise.", rendered)
        self.assertTrue(any(section.metadata.get("header_path") == "Submission/Eligibility" for section in sections))

    def test_html_only_corpus_does_not_need_an_embedding_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            source = project_root / "official" / "policy.html"
            source.parent.mkdir()
            source.write_text(
                "<main><h1>Policy</h1><p>Eligible authors must submit.</p></main>",
                encoding="utf-8",
            )
            input_dir = project_root / "cleaned"
            input_dir.mkdir()
            (input_dir / "policy.cleaned.md").write_text(
                "# Policy\n\n"
                "- Source file: `official/policy.html`\n"
                "- Type: HTML\n"
                "- SHA-256: `0123456789abcdef`\n"
                "- Source URL: `https://example.org/policy`\n"
                "- Source final URL: `https://example.org/policy`\n"
                "- Source acquisition time: `2026-10-05`\n"
                "- Source HTTP Last-Modified: `not recorded`\n"
                "- Cleaning: done\n\n"
                "## Old Markdown body\n\nOld body.\n",
                encoding="utf-8",
            )

            chunks, rows = split_corpus(
                input_dir, project_root, embedding=None, allowed_types={"HTML"}
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file_type"], "HTML")
        self.assertIn("Eligible authors must submit.", chunks[0]["text"])
        self.assertEqual(chunks[0]["metadata"]["source_url"], "https://example.org/policy")


if __name__ == "__main__":
    unittest.main()
