"""Independent multilabel JEV: mock HTTP only, never the classifier/parser."""
import asyncio
import copy
import json
import math
import sys
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from submission_assistant.intent import jev


def batch_fixture(keys, *, clarity=.92, scores=None, model='jev-test'):
    values = {'C02': .95, 'C07': .94, **(scores or {})}
    return {'model': model, 'answers': {
        key: {'type': 'noul', 'noul': clarity if key == 'clarity' else values.get(key, .01)}
        for key in keys}, 'usage': {'input_tokens': 100, 'output_tokens': 20}}


def judgments_fixture(*, clarity=.92, scores=None):
    from submission_assistant.intent.contracts import BatchStatus, IntentJudgments
    values = {f'C{i:02d}': .01 for i in range(1, 11)}
    values.update(scores or {'C02': .95})
    return IntentJudgments(values, clarity, 'fixture', 'controlled-input', [
        BatchStatus('batch_1', 'fixture-1', 'success', {}),
        BatchStatus('batch_2', 'fixture-2', 'success', {})])


class IndependentJevTests(unittest.TestCase):
    def test_two_batches_have_independent_questions_and_shared_history(self):
        self.assertTrue(hasattr(jev, 'build_requests'), 'independent request builder missing')
        history = [m for i in range(6) for m in (
            {'role': 'user', 'content': f'q{i}', 'private': 'not-forwarded'},
            {'role': 'assistant', 'content': f'a{i}', 'private': 'not-forwarded'})]
        original = copy.deepcopy(history)
        bodies = jev.build_requests('摘要要求和费用？', history)
        self.assertEqual([list(b['questions']) for b in bodies], [
            ['clarity', 'C01', 'C02', 'C03', 'C04', 'C05'],
            ['C06', 'C07', 'C08', 'C09', 'C10']])
        self.assertEqual(bodies[0]['state'], bodies[1]['state'])
        self.assertEqual(bodies[0]['state']['history'], [m for i in range(1, 6) for m in (
            {'role': 'user', 'content': f'q{i}'}, {'role': 'assistant', 'content': f'a{i}'})])
        self.assertEqual(history, original)
        for body in bodies:
            for question in body['questions'].values():
                self.assertEqual(question['type'], 'noul')
                self.assertEqual(set(question['criteria']), {'true', 'false'})
                self.assertLessEqual(len(json.dumps(question['criteria'], ensure_ascii=False)), 2000)
                self.assertLessEqual(len(question['instructions']), 1800)

    def test_online_rejects_orphan_and_non_dialogue_history(self):
        for history in (
            [{'role': 'user', 'content': '未完成'}],
            [{'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': 'q'}],
            [{'role': 'user', 'content': 'q'}, {'role': 'tool', 'content': 'private'}],
            [{'role': 'system', 'content': 'private'}] +
                [m for i in range(6) for m in ({'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'})],
        ):
            with self.subTest(history=history), self.assertRaises(jev.IntentInputError):
                jev.build_requests('这个呢？', history)

    def run_http(self, handler):
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await jev.JevIntentRecognizer(jev.JevSettings('https://jev.example.test','local-secret'), client=client).recognize('摘要要求和费用？')
        return asyncio.run(run())

    def test_online_client_preserves_two_high_scores_ids_and_known_usage(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=batch_fixture(json.loads(request.content)['questions']))
        result = self.run_http(handler)
        self.assertTrue(hasattr(result, 'complete'), 'online API still returns legacy choice')
        self.assertTrue(result.complete)
        self.assertEqual(result.probabilities['C02'], .95)
        self.assertEqual(result.probabilities['C07'], .94)
        self.assertGreater(sum(result.probabilities.values()), 1)
        output = result.to_dict()
        self.assertEqual(output['score_type'], 'independent_relevance_probability')
        self.assertNotIn('confidence', output)
        self.assertEqual(output['usage'], {'input_tokens': 200, 'output_tokens': 40})
        self.assertEqual(len(calls), 2)
        self.assertEqual(len({r.headers['Idempotency-Key'] for r in calls}), 2)
        self.assertNotIn('local-secret', json.dumps(output))

    def test_batch_failure_preserves_success_but_cannot_be_complete(self):
        calls = []
        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            if 'C10' in body['questions']:
                return httpx.Response(502, text='local-secret')
            return httpx.Response(200, json=batch_fixture(body['questions']))
        result = self.run_http(handler)
        self.assertTrue(hasattr(result, 'complete'), 'partial statuses missing')
        self.assertFalse(result.complete)
        self.assertEqual(result.probabilities['C02'], .95)
        self.assertNotIn('C07', result.probabilities)
        self.assertEqual([b.status for b in result.batches], ['success','failed'])
        self.assertEqual(result.to_dict()['usage']['input_tokens'], 100)
        self.assertFalse(result.to_dict()['usage_complete'])
        self.assertNotIn('local-secret', json.dumps(result.to_dict()))
        self.assertEqual(len(calls), 2)

    def test_malformed_noul_and_missing_questions_never_default_to_zero(self):
        for bad in [True, math.nan, -.1, 1.1, '0.9', 10**1000, None]:
            with self.subTest(bad=str(bad)[:20]):
                def handler(request):
                    keys = json.loads(request.content)['questions']
                    data = batch_fixture(keys)
                    if 'C02' in keys:
                        data['answers']['C02']['noul'] = bad
                    return httpx.Response(200, text=json.dumps(data))
                # NaN JSON is deliberately malformed at the boundary, never an SDK-generated value.
                result = self.run_http(handler)
                self.assertTrue(hasattr(result, 'complete'), 'strict independent parser missing')
                self.assertFalse(result.complete)
        def missing(request):
            data = batch_fixture(json.loads(request.content)['questions'])
            data['answers'].pop(next(iter(data['answers'])))
            return httpx.Response(200, json=data)
        self.assertFalse(self.run_http(missing).complete)

    def test_model_mismatch_is_not_a_complete_judgment(self):
        def handler(request):
            keys = json.loads(request.content)['questions']
            return httpx.Response(200, json=batch_fixture(keys, model='model-a' if 'clarity' in keys else 'model-b'))
        result = self.run_http(handler)
        self.assertTrue(hasattr(result, 'complete'), 'independent model check missing')
        self.assertFalse(result.complete)

    def test_cancellation_propagates_to_both_batches(self):
        started, stopped = set(), set()
        async def run():
            ready, hold = asyncio.Event(), asyncio.Event()
            async def handler(request):
                key = request.headers['Idempotency-Key']
                started.add(key)
                if len(started) == 2:
                    ready.set()
                try:
                    await hold.wait()
                finally:
                    stopped.add(key)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                task = asyncio.create_task(jev.JevIntentRecognizer(jev.JevSettings('https://jev.example.test','key'), client=client).recognize('摘要？'))
                await asyncio.wait_for(ready.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        asyncio.run(run())
        self.assertEqual(len(started), 2)
        self.assertEqual(stopped, started)


if __name__ == '__main__':
    unittest.main()
