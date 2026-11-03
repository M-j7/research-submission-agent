"""Factorial experiment wiring and paired scoring; no network in tests."""
import copy
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


class AblationDesignTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.ablation_design'),
                             'independent B/E/H experiment construction missing')
        return importlib.import_module('evaluation.agent.ablation_design')

    def test_factor_switches_change_only_their_payload_components(self):
        api = self.api()
        history = [{'role': 'user', 'content': '真实旧问题'}, {'role': 'assistant', 'content': '真实旧回答'}]
        original = copy.deepcopy(history)
        bodies = {a.name: api.build_arm_requests('真实当前问题', history, a, model='fixed-model') for a in api.ARMS}
        self.assertEqual(len(bodies), 8)
        # G0->G1: criteria only; G0->G2: instructions only; G0->G3: real history only.
        self.assertEqual(bodies['G0'][0]['questions']['C03']['instructions'], bodies['G1'][0]['questions']['C03']['instructions'])
        self.assertNotEqual(bodies['G0'][0]['questions']['C03']['criteria'], bodies['G1'][0]['questions']['C03']['criteria'])
        self.assertEqual(bodies['G0'][0]['questions']['C03']['criteria'], bodies['G2'][0]['questions']['C03']['criteria'])
        self.assertNotEqual(bodies['G0'][0]['questions']['C03']['instructions'], bodies['G2'][0]['questions']['C03']['instructions'])
        self.assertEqual(bodies['G0'][0]['questions'], bodies['G3'][0]['questions'])
        self.assertEqual(bodies['G0'][0]['state']['history'], [])
        self.assertEqual(bodies['G3'][0]['state']['history'], original)
        for name, batches in bodies.items():
            self.assertEqual([len(b['questions']) for b in batches], [6, 5])
            self.assertEqual(batches[0]['questions']['clarity'], bodies['G0'][0]['questions']['clarity'])
            self.assertEqual(batches[0]['state']['current_message'], '真实当前问题')
            self.assertEqual(set(batches[0]['state']), {'domain', 'current_message', 'history'})
            if name in ('G2', 'G4', 'G6', 'G7'):
                examples = batches[0]['questions']['C03']['instructions'].split('[示例，仅用于说明判断标准]\n')[1]
                self.assertEqual(examples, bodies['G2'][0]['questions']['C03']['instructions'].split('[示例，仅用于说明判断标准]\n')[1])
        self.assertEqual(history, original)

    def test_identical_requests_are_deduplicated_but_eight_logical_arms_remain(self):
        api = self.api()
        c = {'id': 'a', 'message': '自包含问题', 'history': []}
        plan = api.make_plan([c], model='fixed-model')
        self.assertEqual(len(plan['logical']), 8)
        self.assertEqual(len(plan['queries']), 4)
        self.assertEqual(plan['max_http_attempts'], 8)
        with_history = {**c, 'id': 'b', 'history': [{'role': 'user', 'content': '问'}, {'role': 'assistant', 'content': '答'}]}
        plan = api.make_plan([with_history], model='fixed-model')
        self.assertEqual(len(plan['logical']), 8)
        self.assertEqual(len(plan['queries']), 8)

    def test_missing_history_clarification_is_safe_but_not_full_intent_recovery(self):
        api = self.api()
        c = {'id': 'context-test', 'history_kind': 'dependent', 'expected_intents': ['C07'],
             'expected_roles': ['publication_fees']}
        recognition = {'judgments': {'complete': True}}
        route = {'agent_types': ['general'], 'matched_intents': [], 'fallback_reason': 'unclear_request'}
        row = api.score_arm(c, api.ARMS[0], recognition, route)
        self.assertFalse(row['task_exact'])
        self.assertTrue(row['visible_exact'])
        row = api.score_arm(c, api.ARMS[3], recognition, route)
        self.assertFalse(row['visible_exact'])

    def test_paired_effects_use_all_four_contrasts_not_only_full_minus_empty(self):
        self.api()
        metrics = importlib.import_module('evaluation.agent.ablation_metrics')
        rows = [{'case_id': 'one', 'arm': f'G{i}', 'available': True, 'task_exact': i in (1, 4, 5, 7),
                 'visible_exact': True, 'fp': [], 'fn': [], 'tp': ['C03'], 'latency_ms': 1.}
                for i in range(8)]
        effects = metrics.factor_effects(rows)
        self.assertEqual(effects['main']['boundary']['delta_pp'], 100.)
        self.assertEqual(effects['main']['examples']['delta_pp'], 0.)
        self.assertEqual(effects['main']['history']['delta_pp'], 0.)
        self.assertEqual(effects['main']['boundary']['paired_contrasts'], 4)
        self.assertEqual(effects['interactions']['boundary:examples']['delta_pp'], 0.)

    def test_wrong_category_in_same_role_is_not_counted_as_intent_recovery(self):
        api = self.api()
        metrics = importlib.import_module('evaluation.agent.ablation_metrics')
        case = {'expected_intents': ['C02'], 'expected_roles': ['submission_requirements']}
        row = {'case_id': 'same-role', 'arm': 'G0', 'latency_ms': 1.,
               **api.score_arm(case, api.ARMS[0], {'judgments': {'complete': True}},
                   {'matched_intents': ['C03'], 'agent_types': ['submission_requirements'], 'fallback_reason': None})}
        summary = metrics.arm_summary([row])['G0']
        self.assertEqual(summary['task_exact'], 1)
        self.assertEqual(summary.get('intent_exact'), 0)
        self.assertEqual(summary.get('intent_rate'), 0.)


class AblationRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, home, *, auth=False):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.routing_ablation'), 'ablation runner missing')
        api = importlib.import_module('evaluation.agent.routing_ablation')
        bodies = []
        def handle(request):
            body = json.loads(request.content)
            bodies.append(body)
            self.assertEqual(set(body['state']), {'domain', 'current_message', 'history'})
            self.assertNotIn('expected_intents', json.dumps(body))
            if auth:
                return httpx.Response(401, json={'error': {'message': 'offline'}})
            return httpx.Response(200, json={'model': body['model'], 'answers': {
                k: {'type': 'noul', 'noul': .99 if k in ('clarity', 'C03') else .01} for k in body['questions']},
                'usage': {'input_tokens': 10, 'output_tokens': 1}})
        client_type = httpx.AsyncClient
        def client(**kwargs):
            return client_type(**kwargs, transport=httpx.MockTransport(handle))
        case = {'id': 'one', 'group': 'offline', 'message': '附件提交要求', 'history': [], 'history_kind': 'none',
                'expected_intents': ['C03'], 'expected_roles': ['submission_requirements']}
        with patch.object(api, 'load_cases', return_value=[case]), patch.object(api, 'dotenv_values', return_value={'JEV_API_KEY': 'offline'}), \
                patch.object(api.httpx, 'AsyncClient', side_effect=client):
            await api.run(home, live=True)
        return bodies

    async def test_only_unique_queries_are_paid_and_reused_rows_keep_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'run'
            bodies = await self.exercise(home)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            rows = json.loads((home / 'results.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['complete'])
            self.assertEqual(len(bodies), 8)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({r['query_key'] for r in rows}), 4)
            self.assertEqual(sum(r['reused'] for r in rows), 4)
            self.assertTrue(all(r['task_exact'] for r in rows))
            effects = json.loads((home / 'effects.json').read_text(encoding='utf-8'))
            self.assertIn('intent', effects)
            self.assertEqual(effects['intent']['metric'], 'intent_exact')
            telemetry = json.loads((home / 'telemetry.json').read_text(encoding='utf-8'))
            self.assertEqual(telemetry['usage']['input_tokens'], 80)
            self.assertTrue((home / 'report.html').is_file())

    async def test_auth_error_stops_without_next_unique_query_or_complete_flag(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'run'
            bodies = await self.exercise(home, auth=True)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['stopped_reason'], 'credential_or_rate_limit')
            self.assertLessEqual(len(bodies), 2)
