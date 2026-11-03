"""Probe safety and billable-usage recording, using real clients over offline HTTP."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import httpx


class RoutingProbeTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, output, *, misroute_vague=False):
        spec = importlib.util.spec_from_file_location('routing_probe', ROOT / 'scripts/probe_routing_history.py')
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        requests = []
        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            if request.url.path.endswith('/decisions'):
                vague = body['state']['current_message'] == '这个怎么办？'
                return httpx.Response(200, json={'model': 'pplx-decider-flash-v1.2', 'answers': {
                    k: {'type': 'noul', 'noul': (.1 if vague and not misroute_vague else .99)
                        if k == 'clarity' else .99 if k == 'C02' and (not vague or misroute_vague) else .01}
                    for k in body['questions']}, 'usage': {'input_tokens': 10, 'output_tokens': 1}})
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': '请说明具体事项。'},
                'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15}})
        client_type = httpx.AsyncClient
        def client(**kwargs):
            return client_type(**kwargs, transport=httpx.MockTransport(handle))
        values = {'JEV_PROVIDER': 'decisions', 'JEV_BASE_URL': 'https://api.perplexity.ai',
            'JEV_API_KEY': 'test-key', 'LLM_BASE_URL': 'https://llm.example.test/v1',
            'LLM_API_KEY': 'test-key', 'LLM_MODEL': 'qwen-test'}
        # Usage and fallback tests must not depend on private historical runs.
        cases = [{'id': f'fixture-{i}', 'message': '摘要的字数要求是什么？', 'history': [],
                  'expected_roles': ['submission_requirements']} for i in range(6)]
        cases += [
            {'id': 'vague-control', 'message': '这个怎么办？', 'history': [],
             'expected_roles': ['general'], 'expected_fallback': 'unclear_request'},
            {'id': 'nonbusiness-control', 'message': '你好', 'history': [],
             'expected_roles': ['general'], 'expected_fallback': 'non_business'},
        ]
        with patch.object(api, 'dotenv_values', return_value=values), \
             patch.object(api, 'cases', return_value=cases), \
             patch.object(api.httpx, 'AsyncClient', side_effect=client):
            await api.run(output)
        return requests

    async def test_probe_records_both_feedback_calls_actual_usage(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'run'
            await self.exercise(output)
            telemetry = output / 'telemetry.json'
            self.assertTrue(telemetry.is_file(), 'paid-call observations must be saved, not just estimated RunResult tokens')
            data = json.loads(telemetry.read_text(encoding='utf-8'))
            self.assertEqual(data['generation_usage']['input_tokens'], 24)
            self.assertEqual(data['generation_usage']['output_tokens'], 6)
            self.assertEqual(data['generation_usage']['calls'], 2)

    async def test_misrouted_control_cannot_run_specialist_as_general_feedback(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'run'
            requests = await self.exercise(output, misroute_vague=True)
            feedback = json.loads((output / 'feedback.json').read_text(encoding='utf-8'))
            vague = next(r for r in feedback if r['id'] == 'vague-feedback')
            self.assertEqual(vague.get('skipped_reason'), 'control_did_not_route_to_general')
            generations = [r for r in requests if 'messages' in r]
            self.assertEqual(len(generations), 1)
            self.assertFalse(generations[0].get('tools'))
