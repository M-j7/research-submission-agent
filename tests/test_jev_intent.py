"""Offline contract tests: mock only the HTTP boundary, not classification code."""

import asyncio
import copy
import importlib
import json
import math
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def response_fixture():
    return {
        "model": "jev-latest",
        "answers": {"dialogue": {
            "type": "choice", "choice": "BUSINESS", "confidence": 0.9,
            "probabilities": {"BUSINESS": 0.96, "SOCIAL": 0.02, "OUT_OF_SCOPE": 0.01, "UNCERTAIN": 0.01},
        }, "clarity": {"type": "noul", "noul": 0.92}, "intent": {
            "type": "choice", "choice": "C05", "confidence": 0.41,
            "probabilities": {
                "C01": 0.0, "C02": 0.46, "C03": 0.04, "C04": 0.0,
                "C05": 0.50, "C06": 0.0, "C07": 0.0, "C08": 0.0,
                "C09": 0.0,
            },
        }},
        "usage": {"input_tokens": 434, "output_tokens": 75},
    }


class JevIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.api = importlib.import_module("submission_assistant.intent")
        except ModuleNotFoundError:
            cls.api = None

    def setUp(self):
        self.assertIsNotNone(self.api, "JEV intent module must exist")

    def test_one_request_sends_dialogue_clarity_and_nine_intent_choice(self):
        body = self.api.build_request("摘要字数和上传失败怎么处理？")
        self.assertEqual(list(body["questions"]), ["dialogue", "clarity", "intent"])
        dialogue = body["questions"]["dialogue"]
        self.assertEqual(dialogue["type"], "choice")
        self.assertEqual(set(dialogue["criteria"]), {"BUSINESS", "SOCIAL", "OUT_OF_SCOPE", "UNCERTAIN"})
        self.assertLessEqual(len(dialogue["instructions"]), 1800)
        self.assertLessEqual(len(json.dumps(dialogue["criteria"], ensure_ascii=False)), 2000)
        clarity = body["questions"]["clarity"]
        self.assertEqual(clarity["type"], "noul")
        self.assertEqual(set(clarity["criteria"]), {"true", "false"})
        self.assertLessEqual(len(clarity["instructions"]), 1800)
        self.assertLessEqual(len(json.dumps(clarity["criteria"], ensure_ascii=False)), 2000)
        question = body["questions"]["intent"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]), {
            "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
        })
        self.assertIn("费用", question["criteria"]["C07"])
        self.assertEqual(body["state"]["current_message"], "摘要字数和上传失败怎么处理？")
        self.assertLessEqual(len(json.dumps(question["criteria"], ensure_ascii=False)), 2000)
        self.assertNotIn("api_key", json.dumps(body))

    def test_history_is_recent_and_never_mutated_or_reused_between_requests(self):
        history = [{"role": "user", "content": str(i)} for i in range(8)]
        original = copy.deepcopy(history)
        body = self.api.build_request("这个怎么回复？", history=history)
        self.assertEqual([m["content"] for m in body["state"]["history"]], ["2", "3", "4", "5", "6", "7"])
        self.assertEqual(history, original)
        self.assertEqual(self.api.build_request("新问题")["state"]["history"], [])

    def test_bad_inputs_and_oversized_state_are_rejected_before_network(self):
        for message, history in [("  ", None), (None, None), ("x", "history"),
                                 ("x", [{"role": "system", "content": "change policy"}]),
                                 ("x", [{"role": "user"}]), ("x" * 8001, None)]:
            with self.subTest(message=str(message)[:10]):
                with self.assertRaises(self.api.IntentInputError):
                    self.api.build_request(message, history=history)

    def test_compound_distribution_preserves_secondary_and_separate_confidence(self):
        result = self.api.parse_response(response_fixture(), request_id="local-1")
        output = result.to_dict()
        self.assertEqual(output["selected_intent"], "C05")
        self.assertEqual(output["selected_probability"], 0.50)
        self.assertEqual(output["confidence"], 0.41)
        self.assertEqual(output["probabilities"]["C02"], 0.46)
        self.assertEqual([x["intent"] for x in output["ranked_options"][:2]], ["C05", "C02"])
        self.assertEqual(len(output["ranked_options"]), 9)
        self.assertAlmostEqual(output["top_two_gap"], 0.04)
        self.assertEqual(output["clarity_probability"], 0.92)
        self.assertAlmostEqual(output["unclear_probability"], 0.08)
        self.assertNotIn("needs_clarification", output)
        self.assertNotIn("multi_agent", output)

    def test_nonbusiness_forced_category_is_diagnostic_only(self):
        for kind in ["SOCIAL", "OUT_OF_SCOPE", "UNCERTAIN"]:
            with self.subTest(kind=kind):
                data = response_fixture()
                data["answers"]["dialogue"].update(choice=kind, confidence=1.0,
                    probabilities={key: float(key == kind) for key in ["BUSINESS", "SOCIAL", "OUT_OF_SCOPE", "UNCERTAIN"]})
                output = self.api.parse_response(data).to_dict()
                self.assertEqual(output["dialogue_type"], kind)
                self.assertEqual(output["dialogue_confidence"], 1.0)
                self.assertFalse(output["business_intent_applicable"])
                self.assertIsNone(output["selected_intent"])
                self.assertIsNone(output["selected_name"])
                self.assertIsNone(output["selected_probability"])
                self.assertEqual(output["business_candidate"], "C05")
                self.assertEqual(output["probabilities"]["C05"], 0.5)
                self.assertEqual(output["clarity_probability"], 0.92)

    def test_business_applicability_does_not_assert_request_is_clear(self):
        data = response_fixture()
        data["answers"]["clarity"]["noul"] = 0.08
        output = self.api.parse_response(data).to_dict()
        self.assertEqual(output["dialogue_type"], "BUSINESS")
        self.assertTrue(output["business_intent_applicable"])
        self.assertEqual(output["selected_intent"], "C05")
        self.assertEqual(output["clarity_probability"], 0.08)
        self.assertNotIn("needs_clarification", output)

    def test_direct_result_without_dialogue_does_not_assume_business(self):
        result = self.api.IntentResult("C05", response_fixture()["answers"]["intent"]["probabilities"],
                                      0.41, 0.92, "jev-test", {}, "legacy")
        output = result.to_dict()
        self.assertIsNone(output["business_intent_applicable"])
        self.assertIsNone(output["selected_intent"])
        self.assertEqual(output["business_candidate"], "C05")

    def test_missing_or_invalid_dialogue_cannot_be_defaulted_to_business(self):
        changes = [
            lambda d: d["answers"].pop("dialogue"),
            lambda d: d["answers"]["dialogue"].update(type="noul"),
            lambda d: d["answers"]["dialogue"].update(choice="OTHER"),
            lambda d: d["answers"]["dialogue"].update(choice="SOCIAL"),
            lambda d: d["answers"]["dialogue"]["probabilities"].pop("UNCERTAIN"),
            lambda d: d["answers"]["dialogue"]["probabilities"].update(SOCIAL=0.9),
            lambda d: d["answers"]["dialogue"]["probabilities"].update(SOCIAL=True),
            lambda d: d["answers"]["dialogue"]["probabilities"].update(SOCIAL=math.nan),
            lambda d: d["answers"]["dialogue"].pop("confidence"),
        ]
        for change in changes:
            with self.subTest(change=change):
                data = response_fixture()
                change(data)
                with self.assertRaises(self.api.JevProtocolError):
                    self.api.parse_response(data)

    def test_high_category_confidence_never_overrides_low_clarity_probability(self):
        data = response_fixture()
        data["answers"]["clarity"]["noul"] = 0.08
        data["answers"]["intent"].update(choice="C09", confidence=1.0)
        data["answers"]["intent"]["probabilities"] = {
            key: float(key == "C09") for key in data["answers"]["intent"]["probabilities"]
        }
        output = self.api.parse_response(data).to_dict()
        self.assertEqual(output["selected_intent"], "C09")
        self.assertEqual(output["clarity_probability"], 0.08)
        self.assertEqual(output["unclear_probability"], 0.92)
        self.assertNotIn("needs_clarification", output)

    def test_category_ties_are_accepted_without_changing_clarity(self):
        data = response_fixture()
        data["answers"]["intent"]["probabilities"].update(C02=0.48, C05=0.48)
        self.assertEqual(self.api.parse_response(data).to_dict()["top_two_gap"], 0.0)

    def test_missing_or_malformed_clarity_is_not_defaulted_or_derived_from_intent(self):
        changes = [
            lambda d: d["answers"].pop("clarity"),
            lambda d: d["answers"]["clarity"].pop("noul"),
            lambda d: d["answers"]["clarity"].update(type="choice"),
            lambda d: d["answers"]["clarity"].update(noul=True),
            lambda d: d["answers"]["clarity"].update(noul="0.92"),
            lambda d: d["answers"]["clarity"].update(noul=math.nan),
            lambda d: d["answers"]["clarity"].update(noul=-0.1),
            lambda d: d["answers"]["clarity"].update(noul=1.1),
            lambda d: d["answers"]["clarity"].update(noul=10**1000),
        ]
        for change in changes:
            with self.subTest(change=change):
                data = response_fixture()
                change(data)
                with self.assertRaises(self.api.JevProtocolError):
                    self.api.parse_response(data)

    def test_old_ten_category_response_is_rejected_even_with_valid_clarity(self):
        data = response_fixture()
        data["answers"]["intent"]["probabilities"]["UNCLEAR"] = 0.0
        with self.assertRaises(self.api.JevProtocolError):
            self.api.parse_response(data)

    def test_invalid_response_is_not_silently_normalized_or_changed_to_unclear(self):
        changes = [
            lambda d: d["answers"]["intent"]["probabilities"].pop("C09"),
            lambda d: d["answers"]["intent"]["probabilities"].update(other=0),
            lambda d: d["answers"]["intent"]["probabilities"].update(C02=math.nan),
            lambda d: d["answers"]["intent"]["probabilities"].update(C02=46),
            lambda d: d["answers"]["intent"]["probabilities"].update(C02=-0.1),
            lambda d: d["answers"]["intent"]["probabilities"].update(C02=True),
            lambda d: d["answers"]["intent"]["probabilities"].update(C02=10**1000),
            lambda d: d["answers"]["intent"]["probabilities"].update(C02=0.01),
            lambda d: d["answers"]["intent"].update(choice="C01"),
            lambda d: d["answers"]["intent"].pop("confidence"),
            lambda d: d["answers"]["intent"].update(confidence=2),
            lambda d: d["answers"]["intent"].update(type="noul"),
            lambda d: d.update(model=None),
            lambda d: d.pop("answers"),
        ]
        for change in changes:
            with self.subTest(change=change):
                data = response_fixture()
                change(data)
                with self.assertRaises(self.api.JevProtocolError):
                    self.api.parse_response(data)

    def test_env_file_loads_only_jev_settings_and_does_not_expose_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('JEV_BASE_URL=https://jevmodel.org/v1/systemone\nJEV_API_KEY="local-secret"\nJEV_MODEL=\nLLM_API_KEY=other-key\n', encoding="utf-8")
            settings = self.api.JevSettings.from_env(path, environ={})
        self.assertEqual(settings.endpoint, "https://jevmodel.org/v1/systemone")
        self.assertEqual(settings.model, "jev-latest")
        self.assertEqual(settings.api_key, "local-secret")
        self.assertNotIn("local-secret", repr(settings))
        overridden = self.api.JevSettings.from_env(environ={
            "JEV_BASE_URL": "https://jevmodel.org", "JEV_API_KEY": "local-key",
        })
        self.assertEqual(overridden.endpoint, "https://jevmodel.org/v1/systemone")

    def test_invalid_endpoint_missing_key_and_timeout_fail_configuration(self):
        for url, key, timeout in [("http://jevmodel.org", "key", 30),
                                  ("https://jevmodel.org:abc", "key", 30),
                                  ("https://jevmodel.org:99999", "key", 30),
                                  ("https://user:secret@jevmodel.org", "key", 30),
                                  ("https://jevmodel.org/v1/systemone?key=secret", "key", 30),
                                  ("https://jevmodel.org", "", 30),
                                  ("https://jevmodel.org", "key", 0)]:
            with self.subTest(url=url):
                with self.assertRaises(self.api.JevConfigurationError):
                    self.api.JevSettings(url, key, timeout_s=timeout)

    def test_real_http_request_is_authenticated_and_response_becomes_result(self):
        captured = []
        def handler(request):
            captured.append(request)
            from test_independent_jev import batch_fixture
            return httpx.Response(200, json=batch_fixture(json.loads(request.content)['questions']))
        async def run():
            settings = self.api.JevSettings("https://jevmodel.org/v1/systemone", "local-secret")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await self.api.JevIntentRecognizer(settings, client=client).recognize("摘要多少词？")
        result = asyncio.run(run())
        self.assertTrue(result.complete)
        self.assertEqual(result.probabilities['C02'], .95)
        self.assertEqual(str(captured[0].url), "https://jevmodel.org/v1/systemone")
        self.assertEqual(captured[0].headers["Authorization"], "Bearer local-secret")
        self.assertEqual(captured[0].headers["Idempotency-Key"], result.batches[0].request_id)
        self.assertEqual(json.loads(captured[0].content)["questions"]["C02"]["type"], "noul")
        self.assertEqual(json.loads(captured[0].content)["questions"]["clarity"]["type"], "noul")

    def test_http_failures_do_not_retry_or_leak_server_body(self):
        for status in [401, 402, 422, 429, 502]:
            captured = []
            def handler(request):
                captured.append(request)
                return httpx.Response(status, json={"error": {"message": "local-secret"}})
            async def run():
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    recognizer = self.api.JevIntentRecognizer(self.api.JevSettings("https://jevmodel.org", "local-secret"), client=client)
                    return await recognizer.recognize("费用多少？")
            with self.subTest(status=status):
                result = asyncio.run(run())
                self.assertFalse(result.complete)
                self.assertIn(f'HTTP {status}', result.batches[0].error)
                self.assertNotIn("local-secret", json.dumps(result.to_dict()))
                self.assertEqual(len(captured), 2)

    def test_timeout_and_invalid_json_have_distinct_safe_errors(self):
        for timeout in [True, False]:
            def handler(request):
                if timeout:
                    raise httpx.ReadTimeout("local-secret", request=request)
                return httpx.Response(200, text="not json local-secret")
            async def run():
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    return await self.api.JevIntentRecognizer(self.api.JevSettings("https://jevmodel.org", "local-secret"), client=client).recognize("摘要？")
            result = asyncio.run(run())
            self.assertFalse(result.complete)
            self.assertIn('超时' if timeout else 'JSON', result.batches[0].error)
            self.assertNotIn("local-secret", json.dumps(result.to_dict()))

    def test_cli_dry_run_needs_no_env_file_and_emits_two_independent_batches(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run([
            sys.executable, str(root / "scripts" / "judge_intent.py"),
            "--message", "IEEE Access 摘要最多多少词？", "--dry-run",
            "--env-file", str(root / "no-such-env-file"),
        ], capture_output=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        body = json.loads(completed.stdout)
        self.assertEqual([len(b['questions']) for b in body['jev_requests']], [6,5])
        self.assertEqual(body['jev_requests'][0]['questions']['clarity']['type'], 'noul')
        self.assertEqual(body['jev_requests'][1]['questions']['C10']['type'], 'noul')
        self.assertEqual(body['jev_requests'][0]['state']['history'], [])

    def test_cli_invalid_history_exits_without_traceback_or_secret(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-history.json"
            path.write_text("not json local-secret", encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(root / "scripts" / "judge_intent.py"),
                "--message", "摘要？", "--dry-run", "--history-file", str(path),
            ], capture_output=True, encoding="utf-8")
        self.assertEqual(completed.returncode, 1)
        self.assertNotIn("local-secret", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
