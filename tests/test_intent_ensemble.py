"""Pre-routing fusion: real clients and fusion, fake only external HTTP."""

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


class IntentEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.api = importlib.import_module("submission_assistant.intent.ensemble")
        except ModuleNotFoundError:
            cls.api = None

    def setUp(self):
        self.assertIsNotNone(self.api, "three-source scoring must exist before routing")

    def test_fusion_has_hand_computed_contributions_not_mixed_raw_scores(self):
        from submission_assistant.intent.jev import IntentResult
        from submission_assistant.intent.templates import TemplateIntentResult
        zero = {f"C{i:02d}": 0.0 for i in range(1, 10)}
        jev = IntentResult("C02", {**zero, "C02": 0.6, "C05": 0.4}, 0.8, 0.1,
                           "jev-test", {}, "request-1", dialogue_type="BUSINESS")
        matches = {key: None for key in zero}
        embedding = TemplateIntentResult("embedding", "cosine", {**zero, "C02": 0.6, "C05": 0.3, "C07": -0.2}, matches)
        bm25 = TemplateIntentResult("bm25", "bm25", {**zero, "C02": 2, "C05": 6}, matches)
        result = self.api.fuse_results(jev=jev, embedding=embedding, bm25=bm25)
        self.assertAlmostEqual(result["normalized_scores"]["embedding"]["C02"], 2 / 3)
        self.assertAlmostEqual(result["normalized_scores"]["bm25"]["C02"], 0.25)
        self.assertAlmostEqual(result["contributions"]["jev"]["C02"], 0.42)
        self.assertAlmostEqual(result["scores"]["C02"], 0.5783333333333334)
        self.assertAlmostEqual(result["scores"]["C05"], 0.4216666666666667)
        self.assertEqual(result["scores"]["C07"], 0.0)
        self.assertEqual(result["selected_intent"], "C02")
        self.assertEqual(result["clarity_probability"], 0.1)
        self.assertNotIn("needs_clarification", result)
        self.assertNotIn("multi_agent", result)
        self.assertNotIn("confidence", result)
        self.assertNotIn("probabilities", result)

    def test_bm25_only_never_becomes_a_full_confidence_prediction(self):
        from submission_assistant.intent.templates import TemplateIntentResult
        scores = {f"C{i:02d}": float(i == 7) * 10 for i in range(1, 10)}
        result = self.api.fuse_results(bm25=TemplateIntentResult("bm25", "bm25", scores, {key: None for key in scores}),
                                     source_errors={"jev": "unavailable", "embedding": "unavailable"})
        self.assertIsNone(result["selected_intent"])
        self.assertEqual(result["evidence_top_intent"], "C07")
        self.assertTrue(result["candidate_only"])
        self.assertAlmostEqual(result["scores"]["C07"], 0.1)
        self.assertAlmostEqual(result["score_mass"], 0.1)
        self.assertTrue(result["degraded"])
        self.assertIsNone(result["clarity_probability"])

    def test_auxiliary_votes_cannot_override_jev_primary_intent(self):
        from submission_assistant.intent.jev import IntentResult
        from submission_assistant.intent.templates import TemplateIntentResult
        zero = {f"C{i:02d}": 0.0 for i in range(1, 10)}
        matches = {key: "upload error example" if key == "C05" else None for key in zero}
        jev = IntentResult("C02", {**zero, "C02": 0.4, "C05": 0.35, "C07": 0.25},
                           0.5, 0.95, "jev-test", {}, "request-1", dialogue_type="BUSINESS")
        embedding = TemplateIntentResult("embedding", "cosine", {**zero, "C05": 0.9}, matches)
        bm25 = TemplateIntentResult("bm25", "bm25", {**zero, "C05": 10}, matches)
        result = self.api.fuse_results(jev=jev, embedding=embedding, bm25=bm25)
        self.assertGreater(result["scores"]["C05"], result["scores"]["C02"])
        self.assertEqual(result["evidence_top_intent"], "C05")
        self.assertEqual(result["selected_intent"], "C02")
        self.assertEqual(result["selection_source"], "jev")
        self.assertFalse(result["candidate_only"])
        self.assertEqual(result["auxiliary_evidence"]["embedding"]["relation"], "alternative_candidate")
        self.assertEqual(result["auxiliary_evidence"]["bm25"]["relation"], "alternative_candidate")
        self.assertIn("evidence_ranking_differs_from_primary", result["review_flags"])
        self.assertEqual(result["auxiliary_evidence"]["bm25"]["top_matches"][0]["example"], "upload error example")

    def test_matching_auxiliary_evidence_supports_but_does_not_raise_jev_confidence(self):
        from submission_assistant.intent.jev import IntentResult
        from submission_assistant.intent.templates import TemplateIntentResult
        zero = {f"C{i:02d}": 0.0 for i in range(1, 10)}
        jev = IntentResult("C02", {**zero, "C02": 0.6, "C05": 0.4}, 0.51, 0.94,
                           "jev-test", {}, "request-1", dialogue_type="BUSINESS")
        auxiliary = TemplateIntentResult("bm25", "bm25", {**zero, "C02": 8}, {key: None for key in zero})
        result = self.api.fuse_results(jev=jev, bm25=auxiliary)
        self.assertEqual(result["auxiliary_evidence"]["bm25"]["relation"], "supports_primary")
        self.assertEqual(result["sources"]["jev"]["confidence"], 0.51)
        self.assertEqual(result["sources"]["jev"]["probabilities"]["C02"], 0.6)

    def test_no_match_does_not_disagree_with_jev_and_ties_are_not_arbitrary_votes(self):
        from submission_assistant.intent.jev import IntentResult
        from submission_assistant.intent.templates import TemplateIntentResult
        zero = {f"C{i:02d}": 0.0 for i in range(1, 10)}
        jev = IntentResult("C02", {**zero, "C02": 1}, 1, 0.9, "jev-test", {}, "request-1", dialogue_type="BUSINESS")
        matches = {key: None for key in zero}
        no_match = TemplateIntentResult("bm25", "bm25", zero, matches)
        tied = TemplateIntentResult("embedding", "cosine", {**zero, "C02": 0.7, "C05": 0.7}, matches)
        result = self.api.fuse_results(jev=jev, bm25=no_match, embedding=tied)
        self.assertEqual(result["selected_intent"], "C02")
        self.assertEqual(result["auxiliary_evidence"]["bm25"]["relation"], "no_match")
        self.assertEqual(result["auxiliary_evidence"]["embedding"]["relation"], "tied_candidates")
        self.assertEqual(result["auxiliary_evidence"]["embedding"]["top_intents"], ["C02", "C05"])
        self.assertNotIn("bm25_alternative_candidate", result["review_flags"])
        self.assertNotIn("embedding_alternative_candidate", result["review_flags"])

    def test_no_matches_does_not_force_select_first_category(self):
        from submission_assistant.intent.templates import TemplateIntentResult
        zero = {f"C{i:02d}": 0.0 for i in range(1, 10)}
        result = self.api.fuse_results(bm25=TemplateIntentResult("bm25", "bm25", zero, {key: None for key in zero}))
        self.assertIsNone(result["selected_intent"])
        self.assertEqual(result["score_mass"], 0.0)

    def test_nonbusiness_dialogue_cannot_be_reclassified_by_template_scores(self):
        from submission_assistant.intent.jev import parse_response
        from submission_assistant.intent.templates import TemplateIntentResult
        from test_jev_intent import response_fixture
        for kind in ["SOCIAL", "OUT_OF_SCOPE", "UNCERTAIN"]:
            with self.subTest(kind=kind):
                data = response_fixture()
                data["answers"]["dialogue"].update(choice=kind,
                    probabilities={key: float(key == kind) for key in ["BUSINESS", "SOCIAL", "OUT_OF_SCOPE", "UNCERTAIN"]})
                scores = {f"C{i:02d}": float(i == 2) * 10 for i in range(1, 10)}
                result = self.api.fuse_results(jev=parse_response(data),
                    bm25=TemplateIntentResult("bm25", "bm25", scores, {key: None for key in scores}))
                self.assertIsNone(result["selected_intent"])
                self.assertFalse(result["business_intent_applicable"])
                self.assertEqual(result["dialogue_type"], kind)
                self.assertGreater(result["scores"]["C02"], 0)

    def test_unavailable_dialogue_preserves_scores_without_publishing_valid_intent(self):
        from submission_assistant.intent.templates import TemplateIntentResult
        scores = {f"C{i:02d}": float(i == 2) * 10 for i in range(1, 10)}
        result = self.api.fuse_results(bm25=TemplateIntentResult("bm25", "bm25", scores, {key: None for key in scores}),
                                     source_errors={"jev": "unavailable"})
        self.assertIsNone(result["selected_intent"])
        self.assertIsNone(result["business_intent_applicable"])
        self.assertIsNone(result["dialogue_type"])
        self.assertAlmostEqual(result["scores"]["C02"], 0.1)
        self.assertTrue(result["degraded"])

    def test_fusion_rejects_nonfinite_or_incomplete_scores(self):
        from submission_assistant.intent.templates import TemplateIntentResult
        for scores in [{"C02": 2}, {f"C{i:02d}": math.nan for i in range(1, 10)},
                       {f"C{i:02d}": True for i in range(1, 10)}]:
            with self.assertRaises(ValueError):
                self.api.fuse_results(bm25=TemplateIntentResult("bm25", "bm25", scores, {}))

    def run_http(self, *, fail_jev=False, fail_embedding=False, message="摘要多少词？", history=None):
        from submission_assistant.intent.jev import JevIntentRecognizer, JevSettings
        from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings
        calls = []
        events = {}
        async def handler(request):
            body = json.loads(request.content)
            host = request.url.host
            calls.append((host, body))
            events[host].set()
            # Both real client calls must start before either response returns.
            await asyncio.wait_for(events["jev.example.test"].wait(), 1)
            await asyncio.wait_for(events["embedding.example.test"].wait(), 1)
            if host == "jev.example.test":
                if fail_jev:
                    return httpx.Response(502, text="do-not-leak-secret")
                return httpx.Response(200, json={"model": "jev-test", "answers": {
                    key: {'type': 'noul', 'noul': .12 if key == 'clarity' else float(key == 'C02')}
                    for key in body['questions']},
                    "usage": {"input_tokens": 120, "output_tokens": 60}})
            if fail_embedding:
                return httpx.Response(429, text="do-not-leak-secret")
            return httpx.Response(200, json={"data": [{"index": i, "embedding": [1.0, 0.0]}
                                                       for i in range(len(body["input"]))]})
        async def run():
            events.update({key: asyncio.Event() for key in ["jev.example.test", "embedding.example.test"]})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                recognizer = self.api.EnsembleIntentRecognizer(
                    JevIntentRecognizer(JevSettings("https://jev.example.test", "local-secret"), client=client),
                    EmbeddingIntentRecognizer(EmbeddingSettings("https://embedding.example.test/v1", "local-secret", "embedding-test"), client=client))
                return await recognizer.recognize(message, history)
        return asyncio.run(run()), calls

    def test_parallel_paths_preserve_history_usage_and_low_clarity(self):
        history = [{"role": "user", "content": "我准备投稿。"}, {"role": "assistant", "content": "想咨询什么事项？"}]
        result, calls = self.run_http(history=history)
        self.assertEqual(result["selected_intent"], "C02")
        self.assertEqual(result["clarity_probability"], 0.12)
        self.assertEqual(result["sources"]["jev"]["usage"]["input_tokens"], 240)
        jev_body = next(body for host, body in calls if host == "jev.example.test")
        self.assertEqual(jev_body["state"]["history"], history)
        self.assertFalse(result["degraded"])
        self.assertEqual(len(result["scores"]), 9)

    def test_one_service_failure_preserves_other_results_with_explicit_degradation(self):
        for fail_jev, fail_embedding in [(True, False), (False, True), (True, True)]:
            with self.subTest(jev=fail_jev, embedding=fail_embedding):
                result, calls = self.run_http(fail_jev=fail_jev, fail_embedding=fail_embedding)
                self.assertTrue(result["degraded"])
                self.assertIn("bm25", result["sources"])
                self.assertNotIn("do-not-leak-secret", json.dumps(result))
                self.assertEqual(len([host for host, body in calls if host == "jev.example.test"]), 2)
                if fail_jev:
                    self.assertIsNone(result["selected_intent"])
                    self.assertIsNone(result["dialogue_type"])
                    self.assertIsNone(result["clarity_probability"])
                    self.assertIn("HTTP 502", result["source_errors"]["jev"])
                if fail_embedding:
                    self.assertIn("embedding", result["source_errors"])
                    self.assertEqual(result["contributions"]["embedding"]["C02"], 0.0)

    def test_invalid_input_is_rejected_before_either_http_request(self):
        with self.assertRaises(ValueError):
            self.run_http(message=" " )

    def test_custom_retrieval_templates_cannot_silently_differ_from_jev_examples(self):
        from submission_assistant.intent.bm25 import BM25IntentRecognizer
        from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings
        from submission_assistant.intent.jev import JevIntentRecognizer, JevSettings
        from submission_assistant.intent.templates import load_templates
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "templates.json"
            data = {key: list(texts) for key, texts in load_templates().items()}
            data["C07"][0] = "经济困难作者如何申请减免收费？"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "模板"):
                self.api.EnsembleIntentRecognizer(
                    JevIntentRecognizer(JevSettings("https://jev.example.test", "local-secret")),
                    EmbeddingIntentRecognizer(EmbeddingSettings("https://embedding.example.test/v1", "local-secret", "test-model"), template_path=path),
                    bm25=BM25IntentRecognizer(path))

    def test_changed_template_snapshot_is_rejected_before_paid_requests(self):
        from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings
        from submission_assistant.intent.jev import JevIntentRecognizer, JevSettings
        calls = []
        def handler(request):
            calls.append(request)
            raise AssertionError("no request should be issued for mismatched templates")
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                embedding = EmbeddingIntentRecognizer(EmbeddingSettings("https://embedding.example.test/v1", "local-secret", "test-model"), client=client)
                recognizer = self.api.EnsembleIntentRecognizer(
                    JevIntentRecognizer(JevSettings("https://jev.example.test", "local-secret"), client=client), embedding)
                embedding.templates["C07"] = ("changed",) * 5
                await recognizer.recognize("费用？")
        with self.assertRaisesRegex(ValueError, "模板"):
            asyncio.run(run())
        self.assertEqual(calls, [])

    def test_cancellation_stops_both_services_without_returning_degraded_scores(self):
        from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings
        from submission_assistant.intent.jev import JevIntentRecognizer, JevSettings
        started, stopped = set(), set()
        async def run():
            ready, hold = asyncio.Event(), asyncio.Event()
            async def handler(request):
                started.add(request.url.host)
                if len(started) == 2:
                    ready.set()
                try:
                    await hold.wait()
                finally:
                    stopped.add(request.url.host)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                recognizer = self.api.EnsembleIntentRecognizer(
                    JevIntentRecognizer(JevSettings("https://jev.example.test", "local-secret"), client=client),
                    EmbeddingIntentRecognizer(EmbeddingSettings("https://embedding.example.test/v1", "local-secret", "test-model"), client=client))
                task = asyncio.create_task(recognizer.recognize("摘要？"))
                await asyncio.wait_for(ready.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        asyncio.run(run())
        self.assertEqual(started, {"jev.example.test", "embedding.example.test"})
        self.assertEqual(stopped, started)

    def test_ensemble_cli_dry_run_requires_no_keys_and_reports_scoring_contract(self):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "judge_all_intents.py"),
            "--message", "摘要多少词？", "--dry-run", "--env-file", "no-such-env"], capture_output=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        self.assertEqual(output["weights"], {"jev": 0.7, "embedding": 0.2, "bm25": 0.1})
        self.assertEqual(output["template_count"], 75)
        self.assertEqual(output["embedding_batch_sizes"], [20, 20, 20, 15])
        self.assertEqual([len(body['questions']) for body in output['jev_requests']], [6,5])
        self.assertNotIn("api_key", completed.stdout)


if __name__ == "__main__":
    unittest.main()
