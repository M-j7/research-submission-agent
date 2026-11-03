"""Exercise the real live-probe pipeline with offline HTTP only."""
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


class PromptProbeComparisonTests(unittest.IsolatedAsyncioTestCase):
    def test_completed_comparison_warns_when_candidate_regresses(self):
        api = self.api()
        c = {'id': 'regression', 'group': 'new_probe', 'message': '材料问题', 'history': [],
             'expected_roles': ['submission_requirements']}
        rows = [{'id': 'regression', 'group': 'new_probe', 'variant': v, 'available': True,
                 'exact_route': passed, 'exact_intents': passed,
                 'route': {'agent_types': roles, 'fallback_reason': None}}
                for v, passed, roles in [('baseline', True, ['submission_requirements']),
                                         ('candidate', False, ['submission_requirements', 'editorial_workflow'])]]
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            api.render_report(home, {'complete': True}, rows, [c], {})
            self.assertIn('新版在配对样本上退步，不宜启用', (home / 'report.html').read_text(encoding='utf-8'))

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.prompt_probe'),
                             'fixed comparison runner missing')
        return importlib.import_module('evaluation.agent.prompt_probe')

    async def exercise(self, home, *, auth_error=False):
        api = self.api()
        bodies = []
        def handle(request):
            body = json.loads(request.content)
            bodies.append(body)
            self.assertEqual(set(body['state']), {'domain', 'current_message', 'history'})
            self.assertEqual(body['state']['history'], [])
            self.assertEqual(set(body), {'model', 'state', 'questions'})
            if auth_error:
                return httpx.Response(401, json={'error': {'message': 'fixture'}})
            current = any('[示例，仅用于说明判断标准]' in q['instructions'] for q in body['questions'].values())
            return httpx.Response(200, json={'model': body['model'], 'answers': {
                k: {'type': 'noul', 'noul': .99 if k == 'clarity' or current and k == 'C03' else .01}
                for k in body['questions']}, 'usage': {'input_tokens': 10, 'output_tokens': 1}})
        client_type = httpx.AsyncClient
        def client(**kwargs):
            return client_type(**kwargs, transport=httpx.MockTransport(handle))
        values = {'JEV_API_KEY': 'offline-key'}
        cases = [{'id': 'new-test', 'group': 'new_probe', 'message': '新的附件提交问题', 'history': [],
                  'expected_intents': ['C03'], 'expected_roles': ['submission_requirements']}]
        with patch.object(api, 'build_cases', return_value=cases), patch.object(api, 'dotenv_values', return_value=values), \
                patch.object(api.httpx, 'AsyncClient', side_effect=client):
            await api.run(home)
        return bodies

    async def test_baseline_and_candidate_use_frozen_questions_and_same_real_input(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'run'
            bodies = await self.exercise(home)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            rows = json.loads((home / 'results.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['complete'])
            self.assertEqual(len(bodies), 4)
            self.assertEqual(len(rows), 2)
            self.assertEqual({r['variant']: r['exact_route'] for r in rows}, {'baseline': False, 'candidate': True})
            self.assertEqual(bodies[0]['state'], bodies[2]['state'])
            frozen = json.loads((api_path := ROOT / 'evaluation/agent/prompt-probes/fewshot-v2/baseline-questions.json').read_text(encoding='utf-8'))
            baseline = next(b for b in bodies if '[示例，仅用于说明判断标准]' not in b['questions'].get('C03', {}).get('instructions', '') and 'C03' in b['questions'])
            self.assertEqual(baseline['questions']['C03'], frozen['C03'])
            telemetry = json.loads((home / 'telemetry.json').read_text(encoding='utf-8'))
            self.assertEqual(telemetry['usage']['input_tokens'], 40)
            self.assertTrue((home / 'report.html').is_file())
            with self.assertRaises(FileExistsError):
                await self.exercise(home)

    async def test_auth_failure_stops_before_next_sample_and_never_marks_complete(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'run'
            bodies = await self.exercise(home, auth_error=True)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['stopped_reason'], 'credential_or_rate_limit')
            self.assertLessEqual(len(bodies), 2)
