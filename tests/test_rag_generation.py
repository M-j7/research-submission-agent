import asyncio
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from generation_fixture import make_source
from evaluation.rag.generation_data import prepare_records


class GenerationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module('evaluation.rag.generate')
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, 'answer collector is missing')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.record = prepare_records([('A', make_source(self.temp.name), 'B3')])[0]

    def model(self, reply):
        class Model:
            model = 'fixture-model'
            calls = 0
            async def complete(inner, messages, tools, *, allow_tools=True):
                inner.calls += 1
                self.assertNotIn('REFERENCE_ONLY_SENTINEL', json.dumps(messages))
                self.assertEqual(tools, [])
                self.assertFalse(allow_tools)
                if isinstance(reply, BaseException):
                    raise reply
                return reply
        return Model()

    def reply(self, text='Evidence is insufficient.', finish='stop', **message):
        return {'message': {'role': 'assistant', 'content': text, **message},
                'finish_reason': finish, 'usage': {'prompt_tokens': 20, 'completion_tokens': 5}}

    def test_answer_keeps_identity_usage_and_never_sends_reference(self):
        out = asyncio.run(self.api.collect_answer(self.record, self.model(self.reply())))
        self.assertEqual(out['generation_status'], 'ok')
        self.assertEqual(out['response'], 'Evidence is insufficient.')
        self.assertEqual(out['sample_id'], 'A:abstract.zh')
        self.assertEqual(out['reference'], 'REFERENCE_ONLY_SENTINEL 中文答案')
        self.assertTrue(out['usage_complete'])
        self.assertEqual(out['usage']['prompt_tokens'], 20)

    def test_empty_truncated_tool_calls_are_not_marked_successful(self):
        for reply in [self.reply(''), self.reply('partial', 'length'),
                      self.reply('text', tool_calls=[{'id': 'forbidden'}])]:
            with self.subTest(reply=reply):
                out = asyncio.run(self.api.collect_answer(self.record, self.model(reply)))
                self.assertNotEqual(out['generation_status'], 'ok')
                self.assertEqual(out['attempts'], 1)

    def test_retryable_service_failure_is_bounded_and_usage_not_assumed_free(self):
        from submission_assistant.runtime.provider import ModelServiceError
        test = self
        class Model:
            model = 'fixture'
            calls = 0
            async def complete(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ModelServiceError('模型服务请求失败（HTTP 502）。')
                return test.reply()
        model = Model()
        out = asyncio.run(self.api.collect_answer(self.record, model))
        self.assertEqual(out['attempts'], 2)
        self.assertEqual(model.calls, 2)
        self.assertEqual(out['generation_status'], 'ok')
        self.assertFalse(out['usage_complete'])

    def test_permanent_errors_and_cancellation_do_not_retry_or_leak(self):
        from submission_assistant.runtime.provider import ModelServiceError
        out = asyncio.run(self.api.collect_answer(self.record, self.model(ModelServiceError('HTTP 401 private-secret'))))
        self.assertEqual(out['attempts'], 1)
        self.assertEqual(out['generation_status'], 'failed')
        self.assertNotIn('private-secret', json.dumps(out))
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(self.api.collect_answer(self.record, self.model(asyncio.CancelledError())))

    def test_deadline_and_existing_output_are_checked_before_model_calls(self):
        model = self.model(self.reply())
        out = asyncio.run(self.api.collect_answer(self.record, model, deadline_s=0))
        self.assertEqual(out['generation_status'], 'failed')
        self.assertEqual(model.calls, 0)
        with self.assertRaises(FileExistsError):
            self.api.initialize_run(Path(self.temp.name), [self.record], model)
        self.assertEqual(model.calls, 0)


if __name__ == '__main__':
    unittest.main()
