"""Template retrieval tests; only the external embedding HTTP boundary is fake."""

import asyncio
import importlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class TemplateIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.templates = importlib.import_module("submission_assistant.intent.templates")
            cls.bm25 = importlib.import_module("submission_assistant.intent.bm25")
            cls.embedding = importlib.import_module("submission_assistant.intent.embedding")
        except ModuleNotFoundError:
            cls.templates = cls.bm25 = cls.embedding = None

    def setUp(self):
        self.assertIsNotNone(self.templates, "shared template library and both retrieval paths must exist")

    def test_nine_categories_have_five_distinct_examples_and_jev_uses_one_each(self):
        from submission_assistant.intent import build_request
        examples = self.templates.load_templates()
        self.assertEqual(set(examples), {f"C{i:02d}" for i in range(1, 10)})
        self.assertEqual([len(v) for v in examples.values()], [5] * 9)
        self.assertEqual(len({text for group in examples.values() for text in group}), 45)
        criteria = build_request("摘要多少词？")["questions"]["intent"]["criteria"]
        for key, group in examples.items():
            self.assertIn(group[0], criteria[key])
            self.assertNotIn(group[1], criteria[key])
        self.assertLessEqual(len(json.dumps(criteria, ensure_ascii=False)), 2000)

    def test_invalid_template_catalog_is_rejected(self):
        valid = {k: list(v) for k, v in self.templates.load_templates().items()}
        for change in [lambda d: d.pop("C09"), lambda d: d.update(UNKNOWN=["x"] * 5),
                       lambda d: d["C01"].pop(), lambda d: d["C01"].__setitem__(0, "  "),
                       lambda d: d["C01"].__setitem__(0, d["C02"][0])]:
            data = json.loads(json.dumps(valid))
            change(data)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "templates.json"
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.templates.load_templates(path)

    def test_chinese_bigrams_and_latin_terms_are_tokenized_without_intent_keywords(self):
        self.assertEqual(self.bm25.tokenize("摘要格式 APC"), ["摘要", "要格", "格式", "apc"])

    def test_bm25_matches_paraphrase_and_exposes_all_class_scores_and_evidence(self):
        result = self.bm25.BM25IntentRecognizer().recognize("摘要的字数限制是多少？").to_dict()
        self.assertEqual(result["selected_intent"], "C02")
        self.assertEqual(result["score_type"], "bm25")
        self.assertEqual(len(result["scores"]), 10)
        self.assertGreater(result["scores"]["C02"], 0)
        self.assertIn(result["matched_examples"]["C02"], self.templates.load_templates()["C02"])
        self.assertNotIn("confidence", result)
        self.assertNotIn("probabilities", result)

    def test_bm25_preserves_two_intent_candidates(self):
        result = self.bm25.BM25IntentRecognizer().recognize("摘要字数要求是什么？系统上传文件报错怎么办？").to_dict()
        self.assertGreater(result["scores"]["C02"], 0)
        self.assertGreater(result["scores"]["C05"], 0)

    def test_bm25_no_overlap_does_not_invent_a_category(self):
        result = self.bm25.BM25IntentRecognizer().recognize("zzzxxyyqq").to_dict()
        self.assertIsNone(result["selected_intent"])
        self.assertEqual(result["scores"], {f"C{i:02d}": 0.0 for i in range(1, 11)})
        self.assertTrue(all(v is None for v in result["matched_examples"].values()))

    def test_both_paths_reject_empty_message_before_model_call(self):
        settings = self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", "test-embedding")
        recognizer = self.embedding.EmbeddingIntentRecognizer(settings)
        for message in [" ", None, 42]:
            with self.assertRaises(ValueError):
                self.bm25.BM25IntentRecognizer().recognize(message)
            with self.assertRaises(ValueError):
                asyncio.run(recognizer.recognize(message))

    def test_embedding_env_is_local_only_and_repr_never_contains_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("EMBEDDING_BASE_URL=https://example.test/compatible-mode/v1\nEMBEDDING_API_KEY=local-secret\nEMBEDDING_MODEL=qwen-test\nJEV_API_KEY=unrelated\n", encoding="utf-8")
            settings = self.embedding.EmbeddingSettings.from_env(path, environ={})
        self.assertEqual(settings.endpoint, "https://example.test/compatible-mode/v1/embeddings")
        self.assertEqual(settings.model, "qwen-test")
        self.assertNotIn("local-secret", repr(settings))

    def test_embedding_configuration_rejects_unsafe_urls_and_invalid_fields(self):
        for url, key, model, timeout in [
            ("http://example.test/v1", "key", "model", 30),
            ("https://user:secret@example.test/v1", "key", "model", 30),
            ("https://example.test/v1?key=secret", "key", "model", 30),
            ("https://example.test:99999/v1", "key", "model", 30),
            ("https://example.test/v1", "", "model", 30),
            ("https://example.test/v1", "key", "", 30),
            ("https://example.test/v1", "key", "model", math.nan),
        ]:
            with self.subTest(url=url, model=model, timeout=timeout):
                with self.assertRaises(self.embedding.EmbeddingConfigurationError):
                    self.embedding.EmbeddingSettings(url, key, model, timeout_s=timeout)

    def fixture_handler(self, requests, model="test-embedding"):
        examples = self.templates.load_retrieval_templates()
        category_by_text = {text: index for index, texts in enumerate(examples.values()) for text in texts}
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(str(request.url), "https://example.test/v1/embeddings")
            self.assertEqual(request.headers["Authorization"], "Bearer local-secret")
            self.assertEqual(body["model"], model)
            self.assertEqual(body["encoding_format"], "float")
            self.assertLessEqual(len(body["input"]), 20)
            requests.append(body["input"])
            rows = []
            for index, text in enumerate(body["input"]):
                vector = [0.0] * 10
                # A deliberately known embedding geometry, not a simulated semantic model.
                vector[category_by_text.get(text, 1)] = 2.0
                rows.append({"object": "embedding", "index": index, "embedding": vector})
            return httpx.Response(200, json={"object": "list", "model": model, "data": list(reversed(rows)), "usage": {"prompt_tokens": 10, "total_tokens": 10}})
        return handler

    def test_embedding_reorders_indices_and_batches_then_reuses_memory_templates(self):
        requests = []
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(self.fixture_handler(requests))) as client:
                recognizer = self.embedding.EmbeddingIntentRecognizer(
                    self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", "test-embedding"), client=client)
                first = await recognizer.recognize("摘要词数上限？")
                second = await recognizer.recognize("格式要求？")
                return first.to_dict(), second.to_dict()
        first, second = asyncio.run(run())
        self.assertEqual([len(batch) for batch in requests], [20, 20, 20, 15, 1, 1])
        self.assertEqual(first["selected_intent"], "C02")
        self.assertAlmostEqual(first["scores"]["C02"], 1.0)
        self.assertEqual(first["scores"]["C07"], 0.0)
        self.assertEqual(first["score_type"], "cosine")
        self.assertFalse(first["template_cache_hit"])
        self.assertTrue(second["template_cache_hit"])
        self.assertEqual(first["usage"]["prompt_tokens"], 50)
        self.assertEqual(second["usage"]["prompt_tokens"], 10)

    def test_embedding_disk_cache_reused_across_instances(self):
        requests = []
        async def run(directory):
            async with httpx.AsyncClient(transport=httpx.MockTransport(self.fixture_handler(requests))) as client:
                settings = self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", "test-embedding")
                await self.embedding.EmbeddingIntentRecognizer(settings, client=client, cache_dir=directory).recognize("摘要？")
                result = await self.embedding.EmbeddingIntentRecognizer(settings, client=client, cache_dir=directory).recognize("摘要？")
                return result.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            output = asyncio.run(run(Path(directory)))
            cache_paths = list(Path(directory).glob("*.json"))
            self.assertEqual(len(cache_paths), 1)
            self.assertNotIn("local-secret", cache_paths[0].read_text(encoding="utf-8"))
        self.assertEqual([len(batch) for batch in requests], [20, 20, 20, 15, 1, 1])
        self.assertTrue(output["template_cache_hit"])

    def test_model_and_template_changes_recompute_vectors_instead_of_reusing_stale_cache(self):
        requests = []
        async def query(directory, model, path):
            async with httpx.AsyncClient(transport=httpx.MockTransport(self.fixture_handler(requests, model))) as client:
                return await self.embedding.EmbeddingIntentRecognizer(
                    self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", model),
                    client=client, cache_dir=directory, template_path=path).recognize("摘要？")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            path = Path(directory) / "templates.json"
            data = {key: list(texts) for key, texts in self.templates.load_templates().items()}
            path.write_text(json.dumps(data), encoding="utf-8")
            asyncio.run(query(cache, "test-embedding", path))
            changed_model = asyncio.run(query(cache, "new-embedding", path)).to_dict()
            data["C07"][0] = "发表收费项目有哪些？"
            path.write_text(json.dumps(data), encoding="utf-8")
            changed_templates = asyncio.run(query(cache, "new-embedding", path)).to_dict()
            self.assertEqual(len(list(cache.glob("*.json"))), 3)
        self.assertEqual([len(batch) for batch in requests], [20, 20, 20, 15, 1] * 3)
        self.assertFalse(changed_model["template_cache_hit"])
        self.assertFalse(changed_templates["template_cache_hit"])

    def test_corrupt_cache_is_rebuilt_and_query_dimension_mismatch_is_rejected(self):
        requests = []
        async def run(directory, wrong_dimension=False):
            good_handler = self.fixture_handler(requests)
            def handler(request):
                if wrong_dimension and len(json.loads(request.content)["input"]) == 1:
                    return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})
                return good_handler(request)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await self.embedding.EmbeddingIntentRecognizer(
                    self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", "test-embedding"),
                    client=client, cache_dir=directory).recognize("摘要？")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            asyncio.run(run(cache))
            next(cache.glob("*.json")).write_text("not json", encoding="utf-8")
            result = asyncio.run(run(cache)).to_dict()
            self.assertFalse(result["template_cache_hit"])
            with self.assertRaises(self.embedding.EmbeddingProtocolError):
                asyncio.run(run(cache, True))
        self.assertEqual([len(batch) for batch in requests], [20, 20, 20, 15, 1] * 2)

    def test_concurrent_first_queries_only_initialize_templates_once(self):
        requests = []
        async def run():
            synchronous_handler = self.fixture_handler(requests)
            async def handler(request):
                # Yield during actual template initialization, so the second
                # query attempts initialization while the first holds the lock.
                await asyncio.sleep(0)
                return synchronous_handler(request)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                recognizer = self.embedding.EmbeddingIntentRecognizer(
                    self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", "test-embedding"), client=client)
                return await asyncio.gather(recognizer.recognize("摘要？"), recognizer.recognize("格式？"))
        outputs = asyncio.run(run())
        self.assertEqual([len(batch) for batch in requests], [20, 20, 20, 15, 1, 1])
        self.assertEqual(len(outputs), 2)

    def test_bad_embedding_response_is_rejected_without_hash_fallback(self):
        mutations = [
            (lambda rows: rows.clear(), "数量"),
            (lambda rows: rows[0].update(embedding=[0, 0]), "零向量"),
            (lambda rows: rows[0].update(embedding=[math.nan, 1]), "非法数值"),
            (lambda rows: rows[0].update(embedding=[True, 1]), "非法数值"),
            (lambda rows: rows[1].update(index=0), "index"),
            (lambda rows: rows[0].update(embedding=[1, 0, 0]), "维度"),
        ]
        for mutate, error_fragment in mutations:
            # All malformed-vector fixtures have the full 20 expected rows;
            # otherwise a row-count failure would hide the actual validator.
            rows = [{"object": "embedding", "index": i, "embedding": [1, 0]} for i in range(20)]
            mutate(rows)
            def handler(request):
                return httpx.Response(200, text=json.dumps({"data": rows}))
            async def run():
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    await self.embedding.EmbeddingIntentRecognizer(
                        self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", "test-embedding"), client=client).recognize("摘要？")
            with self.subTest(error=error_fragment):
                with self.assertRaisesRegex(self.embedding.EmbeddingProtocolError, error_fragment):
                    asyncio.run(run())

    def test_dimensions_must_also_match_across_template_batches(self):
        requests = []
        good_handler = self.fixture_handler(requests)
        def handler(request):
            response = good_handler(request)
            if len(requests) == 2:
                body = response.json()
                for row in body["data"]:
                    row["embedding"] = [1, 0]
                return httpx.Response(200, json=body)
            return response
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                await self.embedding.EmbeddingIntentRecognizer(
                    self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", "test-embedding"), client=client).recognize("摘要？")
        with self.assertRaisesRegex(self.embedding.EmbeddingProtocolError, "维度"):
            asyncio.run(run())

    def test_embedding_http_error_and_timeout_do_not_retry_or_leak_body(self):
        for status in [401, 429, 502, None]:
            requests = []
            def handler(request):
                requests.append(request)
                if status is None:
                    raise httpx.ReadTimeout("local-secret", request=request)
                return httpx.Response(status, text="local-secret")
            async def run():
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    await self.embedding.EmbeddingIntentRecognizer(
                        self.embedding.EmbeddingSettings("https://example.test/v1", "local-secret", "test-embedding"), client=client).recognize("摘要？")
            with self.assertRaises(self.embedding.EmbeddingServiceError) as raised:
                asyncio.run(run())
            self.assertEqual(len(requests), 1)
            self.assertNotIn("local-secret", str(raised.exception))

    def test_cli_bm25_requires_no_credentials_and_emits_real_candidates(self):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "judge_template_intent.py"),
            "--method", "bm25", "--message", "摘要的字数限制是多少？", "--env-file", "no-such-env"], capture_output=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["bm25"]["selected_intent"], "C02")

    def test_cli_preview_emits_75_templates_without_credentials_or_model_calls(self):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "judge_template_intent.py"),
            "--method", "both", "--message", "摘要？", "--dry-run", "--env-file", "no-such-env"], capture_output=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        self.assertEqual(output["template_count"], 75)
        self.assertEqual(output["embedding_batch_sizes"], [20, 20, 20, 15])
        self.assertEqual(len(output["templates"]), 10)

    def test_cli_both_keeps_bm25_result_when_embedding_configuration_fails(self):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "judge_template_intent.py"),
            "--method", "both", "--message", "摘要词数有上限吗？", "--env-file", "no-such-env"], capture_output=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 1)
        self.assertTrue(completed.stdout.strip(), "Embedding failure must not discard the independent BM25 output")
        output = json.loads(completed.stdout)
        self.assertEqual(output["bm25"]["selected_intent"], "C02")
        self.assertEqual(output["embedding"]["status"], "error")
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
