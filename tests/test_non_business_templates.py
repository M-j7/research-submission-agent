"""Non-business examples are retrieval evidence, not automatic Agent routing."""

import asyncio
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class NonBusinessTemplateTests(unittest.TestCase):
    def setUp(self):
        self.templates = importlib.import_module("submission_assistant.intent.templates")

    def retrieval_templates(self):
        self.assertTrue(hasattr(self.templates, "load_retrieval_templates"),
                        "retrieval needs the nine business groups plus non-business examples")
        return self.templates.load_retrieval_templates()

    def test_retrieval_extends_business_examples_without_changing_jev_choice(self):
        from submission_assistant.intent import build_request
        library = self.retrieval_templates()
        self.assertEqual(set(library), {f"C{i:02d}" for i in range(1, 11)})
        self.assertGreaterEqual(len(library["C10"]), 20)
        self.assertEqual(sum(len(group) for key, group in library.items() if key != "C10"), 45)
        self.assertEqual(len({text.casefold() for group in library.values() for text in group}),
                         sum(len(group) for group in library.values()))
        self.assertEqual(set(build_request("摘要多少词？")["questions"]["intent"]["criteria"]),
                         {f"C{i:02d}" for i in range(1, 10)})

    def test_bad_nonbusiness_library_is_rejected_instead_of_silently_ignored(self):
        self.retrieval_templates()
        for data in [{}, {"UNKNOWN": ["你好"]}, {"C10": []}, {"C10": "你好"},
                     {"C10": [" "]}, {"C10": ["谢谢", "谢谢"]},
                     {"C10": ["首次投稿需要提交哪些文件和附件？"]}]:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "nonbusiness.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.templates.load_retrieval_templates(non_business_path=path)

    def test_bm25_can_match_unseen_social_capability_and_outside_questions(self):
        from submission_assistant.intent.bm25 import BM25IntentRecognizer
        recognizer = BM25IntentRecognizer()
        for message in ["你好啊，很高兴认识你", "谢谢你的耐心解释", "你可以提供哪些帮助？", "今天北京天气怎么样？"]:
            with self.subTest(message=message):
                output = recognizer.recognize(message).to_dict()
                self.assertIn("C10", output["scores"])
                self.assertGreater(output["scores"]["C10"], 0)
                self.assertEqual(output["selected_intent"], "C10")
                self.assertIsNotNone(output["matched_examples"]["C10"])
                self.assertNotIn("agent_type", output)

    def test_business_question_remains_available_with_nonbusiness_comparators(self):
        from submission_assistant.intent.bm25 import BM25IntentRecognizer
        output = BM25IntentRecognizer().recognize("摘要的字数限制是多少？").to_dict()
        self.assertEqual(output["selected_intent"], "C02")
        self.assertIn("C10", output["scores"])

    def test_embedding_sends_nonbusiness_examples_and_preserves_matching_evidence(self):
        from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings
        calls = []

        def handler(request):
            body = json.loads(request.content)
            calls.extend(body["input"])
            geometry = {"Hello!": [0.0, 1.0], "你好": [0.6, 0.8]}
            rows = [{"index": i, "embedding": geometry.get(text, [1.0, 0.0])}
                    for i, text in enumerate(body["input"])]
            return httpx.Response(200, json={"data": list(reversed(rows))})

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return (await EmbeddingIntentRecognizer(
                    EmbeddingSettings("https://embedding.example.test/v1", "local-secret", "geometry-test"),
                    client=client).recognize("Hello!")).to_dict()

        output = asyncio.run(run())
        self.assertIn("selected_intent", output)
        self.assertEqual(output["selected_intent"], "C10")
        self.assertAlmostEqual(output["scores"]["C10"], 1.0)
        self.assertEqual(output["matched_examples"]["C10"], "Hello!")
        self.assertIn("你好", calls)
        # Known vector geometry tests the real scoring path, not model accuracy.
        self.assertEqual(output["scores"]["C02"], 0.0)

    def test_changing_only_nonbusiness_examples_invalidates_the_disk_cache(self):
        from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings
        self.retrieval_templates()
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body["input"])
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))]})

        async def query(cache, nonbusiness):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return (await EmbeddingIntentRecognizer(
                    EmbeddingSettings("https://embedding.example.test/v1", "local-secret", "cache-test"),
                    client=client, cache_dir=cache, non_business_path=nonbusiness).recognize("你好")).to_dict()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonbusiness.json"
            cache = Path(directory) / "cache"
            data = {"C10": list(self.templates.load_retrieval_templates()["C10"])}
            path.write_text(json.dumps(data), encoding="utf-8")
            first = asyncio.run(query(cache, path))
            reused = asyncio.run(query(cache, path))
            data["C10"][0] = "你好呀，朋友"
            path.write_text(json.dumps(data), encoding="utf-8")
            changed = asyncio.run(query(cache, path))
            self.assertEqual(len(list(cache.glob("*.json"))), 2)
        self.assertFalse(first["template_cache_hit"])
        self.assertTrue(reused["template_cache_hit"])
        self.assertFalse(changed["template_cache_hit"])
        self.assertEqual([len(batch) for batch in requests],
                         [20, 20, 20, 15, 1, 1, 20, 20, 20, 15, 1])

    def test_fusion_exposes_nonbusiness_evidence_without_inventing_jev_probability(self):
        from submission_assistant.intent.ensemble import fuse_results
        from submission_assistant.intent.jev import parse_response
        from submission_assistant.intent.templates import TemplateIntentResult
        from test_jev_intent import response_fixture
        scores = {f"C{i:02d}": 0.0 for i in range(1, 11)}
        scores.update(C02=2.0, C10=6.0)
        matches = {key: None for key in scores}
        matches["C10"] = "你能提供哪些帮助？"
        result = fuse_results(jev=parse_response(response_fixture()),
            bm25=TemplateIntentResult("bm25", "bm25", scores, matches))
        self.assertEqual(result["non_business_evidence"]["bm25"]["score"], 6.0)
        self.assertEqual(result["non_business_evidence"]["bm25"]["example"], "你能提供哪些帮助？")
        self.assertEqual(len(result["scores"]), 9)
        self.assertNotIn("C10", result["sources"]["jev"]["probabilities"])

    def test_cli_preview_and_local_query_expose_the_extended_library_without_keys(self):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "judge_template_intent.py"),
            "--message", "你好", "--dry-run", "--env-file", "no-such-env"],
            capture_output=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        self.assertIn("C10", output["templates"])
        self.assertEqual(output["template_count"], 75)
        self.assertEqual(output["embedding_batch_sizes"], [20, 20, 20, 15])


if __name__ == "__main__":
    unittest.main()
