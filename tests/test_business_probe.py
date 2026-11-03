"""Background/real-demand contrasts must not erase genuine multi-intent routing."""
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


class BusinessProbeTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.business_probe'), 'business paired probe missing')
        return importlib.import_module('evaluation.agent.business_probe')

    def test_changes_only_four_example_suffixes_not_clarity_history_or_criteria(self):
        api = self.api()
        history = [{'role': 'user', 'content': '想了解材料要求'}, {'role': 'assistant', 'content': '请说明材料。'}]
        old, new = [api.bodies_for('附件提交要求是什么？', history, v) for v in ('baseline', 'candidate')]
        changed = []
        for before, after in zip(old, new):
            self.assertEqual(before['state'], after['state'])
            self.assertEqual(before['model'], after['model'])
            for key in before['questions']:
                x, y = before['questions'][key], after['questions'][key]
                self.assertEqual(x['criteria'], y['criteria'])
                if x != y:
                    changed.append(key)
                    self.assertTrue(y['instructions'].startswith(x['instructions']))
        self.assertEqual(sorted(changed), ['C03', 'C05', 'C07', 'C08'])
        self.assertEqual(old[0]['questions']['clarity'], new[0]['questions']['clarity'])

    def test_business_counts_score_clear_business_not_ambiguous_partial_categories(self):
        api = self.api()
        rows = [{'case_id': 'clear', 'variant': 'candidate', 'group': 'new', 'available': True,
                 'expected_clarity': True, 'clarity_probability': .9, 'exact_route': False, 'exact_intents': False,
                 'route': {'matched_intents': ['C03', 'C05']}},
                {'case_id': 'unclear', 'variant': 'candidate', 'group': 'new', 'available': True,
                 'expected_clarity': False, 'clarity_probability': .2, 'exact_route': True, 'exact_intents': False,
                 'route': {'matched_intents': ['C03']}}]
        cases = [{'id': 'clear', 'expected_intents': ['C03']}, {'id': 'unclear', 'expected_intents': []}]
        s = api.summarize(rows, cases)['candidate']['new']
        self.assertEqual(s['business_scored'], 1)
        self.assertEqual((s['tp'], s['fp'], s['fn']), (1, 1, 0))
        self.assertEqual(s['false_pass'], 0)


class BusinessRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepared_payloads_use_real_shared_runner_and_business_report(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.business_probe'), 'business paired probe missing')
        api = importlib.import_module('evaluation.agent.business_probe')
        posted = []
        def handle(req):
            body = json.loads(req.content)
            posted.append(body)
            self.assertEqual(set(body['state']), {'domain', 'current_message', 'history'})
            self.assertNotIn('expected_clarity', json.dumps(body))
            return httpx.Response(200, json={'model': body['model'], 'answers': {
                k: {'type': 'noul', 'noul': .99 if k in ('clarity', 'C03', 'C05') else .01}
                for k in body['questions']}, 'usage': {'input_tokens': 10, 'output_tokens': 1}})
        case = {'id': 'multi', 'group': 'multi_intent_controls', 'message': '附件大小要求是什么，上传报错又怎么办？',
                'history': [], 'expected_intents': ['C03', 'C05'],
                'expected_roles': ['submission_requirements', 'editorial_workflow'], 'expected_clarity': True}
        client_type = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / 'run'
            with patch.object(api, 'load_cases', return_value=[case]):
                api.prepare(home)
            with patch.object(api.core, 'dotenv_values', return_value={'JEV_API_KEY': 'offline'}), \
                 patch.object(api.core.httpx, 'AsyncClient', side_effect=lambda **kw: client_type(**kw, transport=httpx.MockTransport(handle))):
                await api.run(home)
                with self.assertRaises(FileExistsError):
                    await api.run(home)
            self.assertEqual(len(posted), 4)
            manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(manifest['complete'])
            summary = json.loads((home / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['candidate']['all']['tp'], 2)
            self.assertEqual(summary['candidate']['all']['exact_intents'], 1)
            html = (home / 'report.html').read_text(encoding='utf-8')
            self.assertIn('背景误选', html)
            self.assertIn('FP', html)
