"""Clarity-only paired payloads, metrics and safe offline execution."""
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


class ClarityProbeTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.clarity_probe'), 'paired clarity probe missing')
        return importlib.import_module('evaluation.agent.clarity_probe')

    def test_only_clarity_question_changes_business_and_history_remain_identical(self):
        api = self.api()
        history = [{'role': 'user', 'content': '问摘要的长度'}, {'role': 'assistant', 'content': '你想确认上限吗？'}]
        old = api.bodies_for('是的', history, 'baseline')
        new = api.bodies_for('是的', history, 'candidate')
        self.assertEqual([len(b['questions']) for b in old], [6, 5])
        self.assertEqual(old[0]['state'], new[0]['state'])
        self.assertEqual(new[0]['state']['history'], history)
        self.assertNotEqual(old[0]['questions']['clarity'], new[0]['questions']['clarity'])
        self.assertEqual(old[1], new[1])
        for key in ('C01', 'C02', 'C03', 'C04', 'C05'):
            self.assertEqual(old[0]['questions'][key], new[0]['questions'][key])
        self.assertEqual(old[0]['model'], new[0]['model'])

    def test_confusion_counts_ignore_unavailable_and_nonbusiness_clarity(self):
        api = self.api()
        rows = [{'variant': 'candidate', 'group': 'probe', 'available': True, 'expected_clarity': True,
                 'clarity_probability': .5, 'exact_intents': False},
                {'variant': 'candidate', 'group': 'probe', 'available': True, 'expected_clarity': False,
                 'clarity_probability': .9, 'exact_intents': False},
                {'variant': 'candidate', 'group': 'probe', 'available': False, 'expected_clarity': True,
                 'clarity_probability': None, 'exact_intents': False},
                {'variant': 'candidate', 'group': 'probe', 'available': True, 'expected_clarity': None,
                 'clarity_probability': .1, 'exact_intents': True}]
        s = api.summarize(rows)['candidate']['probe']
        self.assertEqual(s['false_clarify'], 1)
        self.assertEqual(s['false_pass'], 1)
        self.assertEqual(s['clear_samples'], 1)
        self.assertEqual(s['unclear_samples'], 1)
        self.assertEqual(s['unavailable'], 1)
        self.assertEqual(s['clarity_excluded'], 1)

    def test_correct_clarification_can_keep_a_known_business_category(self):
        api = self.api()
        row = {'variant': 'candidate', 'group': 'probe', 'available': True,
               'expected_clarity': False, 'clarity_probability': .2,
               'exact_intents': False, 'exact_route': True}
        s = api.summarize([row])['candidate']['probe']
        self.assertEqual(s.get('exact_route'), 1)
        self.assertEqual(s['exact_intents'], 0)
        self.assertEqual(s['false_pass'], 0)


class ClarityRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_transport_parser_and_routes_respect_fixed_scope_and_no_replay(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.clarity_probe'), 'paired clarity probe missing')
        api = importlib.import_module('evaluation.agent.clarity_probe')
        calls = []
        def handle(req):
            body = json.loads(req.content)
            calls.append(body)
            self.assertEqual(set(body['state']), {'domain', 'current_message', 'history'})
            self.assertNotIn('expected_clarity', json.dumps(body))
            return httpx.Response(200, json={'model': body['model'], 'answers': {
                k: {'type': 'noul', 'noul': .99 if k in ('clarity', 'C03') else .01} for k in body['questions']},
                'usage': {'input_tokens': 10, 'output_tokens': 1}})
        case = {'id': 'offline', 'group': 'probe', 'message': '附件需要哪种格式？', 'history': [],
                'expected_intents': ['C03'], 'expected_roles': ['submission_requirements'], 'expected_clarity': True}
        client_type = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'run'
            with patch.object(api, 'load_cases', return_value=[case]):
                api.prepare(home)
            with patch.object(api, 'dotenv_values', return_value={'JEV_API_KEY': 'offline'}), \
                 patch.object(api.httpx, 'AsyncClient', side_effect=lambda **kw: client_type(**kw, transport=httpx.MockTransport(handle))):
                await api.run(home)
                with self.assertRaises(FileExistsError):
                    await api.run(home)
            self.assertEqual(len(calls), 4)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['complete'])
            rows = json.loads((home / 'results.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['exact_intents'] for r in rows))
            self.assertTrue((home / 'report.html').is_file())
