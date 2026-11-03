"""Offline contracts for independently generated routing candidates."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import httpx


def candidates():
    return {'cases': [{'message': f'测试领域{i}的投稿范围是否合适？', 'history': [],
                       'history_kind': 'none', 'expected_intents': ['C01'],
                       'expected_fallback': None, 'reason': '实际询问选刊适配性。'} for i in range(10)]}


class CandidateTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.qwen_cases'), 'candidate generator missing')
        return importlib.import_module('evaluation.agent.qwen_cases')

    def test_candidates_are_not_silently_promoted_to_reviewed_gold(self):
        api = self.api()
        rows = api.parse_candidates(json.dumps(candidates(), ensure_ascii=False), 'single')
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]['id'], 'qwen-single-01')
        self.assertEqual(rows[0]['proposed_intents'], ['C01'])
        self.assertNotIn('expected_intents', rows[0])
        self.assertEqual(rows[0]['review_status'], 'pending')

    def test_unknown_labels_bad_history_and_incomplete_batches_are_rejected(self):
        api = self.api()
        for mutation in ('label', 'history', 'count'):
            body = candidates()
            if mutation == 'label':
                body['cases'][0]['expected_intents'] = ['C99']
            elif mutation == 'history':
                body['cases'][0]['history'] = [{'role': 'assistant', 'content': '孤立回答'}]
            else:
                body['cases'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                api.parse_candidates(json.dumps(body), 'single')

    def test_custom_preparation_does_not_append_old_development_cases(self):
        self.api()
        runner = importlib.import_module('evaluation.agent.routing_ablation')
        case = {'id': 'new-only', 'group': 'synthetic-reviewed', 'message': '这本期刊适合我的研究吗？',
                'history': [], 'history_kind': 'none', 'expected_intents': ['C01'],
                'expected_roles': ['submission_requirements']}
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'run'
            _, plan, manifest = runner.prepare(home, cases=[case], dataset_note='新合成题，未经独立人工审定。')
            saved = json.loads((home / 'cases.json').read_text(encoding='utf-8'))
            self.assertEqual([c['id'] for c in saved], ['new-only'])
            self.assertEqual(len(plan['logical']), 8)
            self.assertEqual(manifest['dataset_note'], '新合成题，未经独立人工审定。')

    def test_review_export_uses_review_labels_and_keeps_disputed_out(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            rows = api.parse_candidates(json.dumps(candidates()), 'single')
            (home / 'candidates.json').write_text(json.dumps(rows), encoding='utf-8')
            decisions = [{'id': r['id'], 'status': 'accepted', 'intents': ['C02'], 'fallback': None,
                          'reason': '独立复核写作诉求'} for r in rows]
            decisions[-1]['status'] = 'disputed'
            (home / 'review-decisions.json').write_text(json.dumps({'decisions': decisions}), encoding='utf-8')
            api.freeze_review(home)
            saved = json.loads((home / 'reviewed-cases.json').read_text(encoding='utf-8'))
            self.assertEqual(len(saved), 9)
            self.assertEqual(saved[0]['expected_intents'], ['C02'])
            self.assertEqual(saved[0]['expected_roles'], ['submission_requirements'])
            self.assertEqual(len(json.loads((home / 'disputed-cases.json').read_text(encoding='utf-8'))), 1)
            self.assertEqual(json.loads((home / 'candidates.json').read_text(encoding='utf-8'))[0]['proposed_intents'], ['C01'])
            with self.assertRaises(FileExistsError):
                api.freeze_review(home)

    def test_preparation_rejects_pending_or_inconsistent_gold_before_any_run(self):
        self.api()
        runner = importlib.import_module('evaluation.agent.routing_ablation')
        rows = self.api().parse_candidates(json.dumps(candidates()), 'single')
        reviewed = {**rows[0], 'review_status': 'assistant_reviewed_synthetic',
                    'expected_intents': ['C01'], 'expected_roles': ['submission_requirements'], 'expected_fallback': None}
        invalid = [rows, [{**reviewed, 'expected_roles': ['publication_fees']}],
                   [{**reviewed, 'expected_roles': ['general'], 'expected_fallback': 'non_business'}]]
        for cases in invalid:
            with tempfile.TemporaryDirectory() as d:
                home = Path(d) / 'run'
                with self.assertRaises(ValueError):
                    runner.prepare(home, cases=cases)
                self.assertFalse(home.exists())


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, home, *, auth=False):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.qwen_cases'), 'candidate generator missing')
        api = importlib.import_module('evaluation.agent.qwen_cases')
        posted = []
        def handle(req):
            body = json.loads(req.content)
            posted.append(body)
            if auth:
                return httpx.Response(402, json={'error': {'code': 'insufficient_balance'}})
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant',
                'content': json.dumps(candidates(), ensure_ascii=False)}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50}})
        client_type = httpx.AsyncClient
        def client(**kw):
            return client_type(**kw, transport=httpx.MockTransport(handle))
        values = {'LLM_BASE_URL': 'https://offline.invalid/v1', 'LLM_MODEL': 'qwen-test', 'LLM_API_KEY': 'offline-key'}
        with patch.object(api, 'dotenv_values', return_value=values), patch.object(api.httpx, 'AsyncClient', side_effect=client):
            await api.generate(home)
        return posted

    async def test_four_generation_calls_are_saved_without_tools_or_gold_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'generated'
            posted = await self.exercise(home)
            self.assertEqual(len(posted), 4)
            self.assertTrue(all('tools' not in p for p in posted))
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['complete'])
            rows = json.loads((home / 'candidates.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 40)
            self.assertTrue(all(r['review_status'] == 'pending' for r in rows))
            self.assertNotIn('offline-key', ''.join(p.read_text(encoding='utf-8') for p in home.glob('*.json')))
            with self.assertRaises(FileExistsError):
                await self.exercise(home)

    async def test_billing_failure_stops_before_second_call_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'generated'
            posted = await self.exercise(home, auth=True)
            self.assertEqual(len(posted), 1)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['phase'], 'stopped')

    async def test_continuation_skips_attempted_groups_including_failed_one(self):
        api = importlib.import_module('evaluation.agent.qwen_cases')
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'generated'
            await self.exercise(home)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            manifest.update(phase='stopped', complete=False, stopped_reason='ModelServiceError')
            (home / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            # Three attempts: two successes and one request with no response.
            telemetry = json.loads((home / 'telemetry.json').read_text(encoding='utf-8'))
            telemetry['events'] = [e for e in telemetry['events'] if e['turn_id'] != 'history'
                                   and not (e['turn_id'] == 'multi' and e['type'] == 'http_result')]
            (home / 'telemetry.json').write_text(json.dumps(telemetry), encoding='utf-8')
            raw = json.loads((home / 'raw-responses.json').read_text(encoding='utf-8'))[:2]
            (home / 'raw-responses.json').write_text(json.dumps(raw), encoding='utf-8')
            rows = json.loads((home / 'candidates.json').read_text(encoding='utf-8'))[:20]
            (home / 'candidates.json').write_text(json.dumps(rows), encoding='utf-8')
            posted = []
            def handle(req):
                posted.append(json.loads(req.content))
                return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant',
                    'content': json.dumps(candidates())}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 50}})
            kind = httpx.AsyncClient
            values = {'LLM_BASE_URL': 'https://offline.invalid/v1', 'LLM_MODEL': 'qwen-test', 'LLM_API_KEY': 'offline-key'}
            with patch.object(api, 'dotenv_values', return_value=values), \
                 patch.object(api.httpx, 'AsyncClient', side_effect=lambda **kw: kind(**kw, transport=httpx.MockTransport(handle))):
                await api.generate(home, continue_unattempted=True)
            self.assertEqual(len(posted), 1)
            self.assertEqual(posted[0]['messages'][1]['content'], api.BRIEFS['history'])
            saved = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['complete'])
            self.assertEqual(saved['phase'], 'finished_with_gaps')
            self.assertEqual(len(json.loads((home / 'candidates.json').read_text(encoding='utf-8'))), 30)

    async def test_billing_body_on_http400_remains_blocked_on_continuation(self):
        api = importlib.import_module('evaluation.agent.qwen_cases')
        kind = httpx.AsyncClient
        posted = []
        def handle(req):
            posted.append(req)
            return httpx.Response(400, json={'error': {'code': 'insufficient_balance'}})
        values = {'LLM_BASE_URL': 'https://offline.invalid/v1', 'LLM_MODEL': 'qwen-test', 'LLM_API_KEY': 'offline-key'}
        with tempfile.TemporaryDirectory() as d, patch.object(api, 'dotenv_values', return_value=values), \
             patch.object(api.httpx, 'AsyncClient', side_effect=lambda **kw: kind(**kw, transport=httpx.MockTransport(handle))):
            home = Path(d) / 'generated'
            await api.generate(home)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest.get('stopped_reason'), 'credential_or_billing')
            with self.assertRaises(ValueError):
                await api.generate(home, continue_unattempted=True)
            self.assertEqual(len(posted), 1)
