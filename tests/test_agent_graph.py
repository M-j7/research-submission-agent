"""Real LangGraph, injected recognizer only at external-service boundary."""
import asyncio
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from submission_assistant.agents import AgentRequest
from submission_assistant.intent.ensemble import fuse_results
from submission_assistant.routing import RoutingSettings
from test_independent_jev import judgments_fixture


class AgentGraphTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.graph'), 'graph entry missing')
        self.api = importlib.import_module('submission_assistant.graph')

    def test_real_graph_returns_selected_role_inputs_not_answers(self):
        cases = [({'C02':.95}, .9, 'single', ['submission_requirements']),
                 ({'C02':.95,'C07':.94}, .9, 'multiple', ['submission_requirements','publication_fees']),
                 ({'C10':.99}, .1, 'fallback', ['general']),
                 ({'C02':.9}, .1, 'fallback', ['general'])]
        for scores, clarity, mode, roles in cases:
            with self.subTest(mode=mode,roles=roles):
                class Recognizer:
                    async def recognize(self, message, history=None):
                        self.message, self.history = message, history
                        return fuse_results(jev=judgments_fixture(scores=scores,clarity=clarity))
                recognizer = Recognizer()
                graph = self.api.build_agent_graph(recognizer)
                request = AgentRequest('本轮消息', [{'role':'user','content':'历史'}], 'request')
                output = asyncio.run(graph.ainvoke({'request':request}))
                self.assertEqual(output['routing_decision']['mode'], mode)
                self.assertEqual([a['agent_type'] for a in output['prepared_agents']], roles)
                self.assertEqual(recognizer.message, '本轮消息')
                self.assertEqual(recognizer.history, [{'role':'user','content':'历史'}])
                self.assertNotIn('answer', output)
                self.assertFalse(any(a['executed'] for a in output['prepared_agents']))

    def test_recognizer_error_falls_back_without_leaking_body(self):
        class Broken:
            async def recognize(self, message, history=None):
                raise RuntimeError('local-secret must never leak')
        result = asyncio.run(self.api.build_agent_graph(Broken()).ainvoke({
            'request':AgentRequest('费用？', [], 'request')}))
        self.assertEqual(result['routing_decision']['fallback_reason'], 'recognition_failed')
        self.assertNotIn('local-secret', json.dumps({key:value for key,value in result.items() if key != 'request'}))

    def test_bm25_runtime_failure_cannot_prevent_complete_jev_routing(self):
        import httpx
        from unittest.mock import patch
        from submission_assistant.intent.bm25 import BM25IntentRecognizer
        from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings
        from submission_assistant.intent.ensemble import EnsembleIntentRecognizer
        from submission_assistant.intent.jev import JevIntentRecognizer, JevSettings
        from test_independent_jev import batch_fixture
        calls = []
        def handler(request):
            body = json.loads(request.content)
            if request.url.host == 'jev.example.test':
                calls.append(request.url.host)
                return httpx.Response(200, json=batch_fixture(body['questions'], scores={'C02':.01,'C07':.99}))
            return httpx.Response(200, json={'data': [
                {'index':i,'embedding':[1.0,0.0]} for i in range(len(body['input']))]})
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                bm25 = BM25IntentRecognizer()
                recognizer = EnsembleIntentRecognizer(
                    JevIntentRecognizer(JevSettings('https://jev.example.test','local-secret'),client=client),
                    EmbeddingIntentRecognizer(EmbeddingSettings('https://embedding.example.test/v1','local-secret','test'),client=client), bm25=bm25)
                # Only induce the local dependency failure; graph/JEV/Embedding paths are real.
                with patch.object(bm25, 'recognize', side_effect=RuntimeError('local-secret')):
                    return await self.api.build_agent_graph(recognizer).ainvoke({'request':AgentRequest('发表费用？',[],'request')})
        result = asyncio.run(run())
        self.assertEqual(result['routing_decision']['agent_types'], ['publication_fees'])
        self.assertTrue(result['recognition']['degraded'])
        self.assertEqual(result['recognition']['auxiliary_evidence']['bm25']['relation'], 'unavailable')
        self.assertIn('bm25', result['recognition']['source_errors'])
        self.assertEqual(len(calls), 2)
        self.assertNotIn('local-secret', json.dumps(result['recognition']))

    def test_custom_thresholds_and_registry_reach_selected_agent(self):
        class Recognizer:
            async def recognize(self, message, history=None):
                return fuse_results(jev=judgments_fixture(scores={'C07':.71},clarity=.61))
        graph = self.api.build_agent_graph(Recognizer(), settings=RoutingSettings(.6,.7),
                                           tool_registry={'search_knowledge': object()})
        result = asyncio.run(graph.ainvoke({'request':AgentRequest('费用？', [], 'request')}))
        self.assertEqual(result['routing_decision']['agent_types'], ['publication_fees'])
        self.assertEqual(result['prepared_agents'][0]['registered_tools'], ['search_knowledge'])

    def test_graph_cancellation_is_not_a_fallback(self):
        async def run():
            ready = asyncio.Event()
            class Slow:
                async def recognize(self, message, history=None):
                    ready.set()
                    await asyncio.Event().wait()
            task = asyncio.create_task(self.api.build_agent_graph(Slow()).ainvoke({'request':AgentRequest('费用？', [], 'request')}))
            await asyncio.wait_for(ready.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(run())

    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT/'scripts'/'route_to_agents.py'), *args],
                               capture_output=True, encoding='utf-8')

    def test_cli_offline_and_preview_require_no_keys(self):
        preview = self.cli('--message','摘要要求和费用？','--dry-run','--env-file','no-such-env')
        self.assertEqual(preview.returncode, 0, preview.stderr)
        self.assertEqual([len(b['questions']) for b in json.loads(preview.stdout)['jev_requests']], [6,5])
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)/'judgments.json'
            fixture.write_text(json.dumps(judgments_fixture(scores={'C02':.95,'C07':.94}).to_dict()),encoding='utf-8')
            result = self.cli('--message','摘要要求和费用？','--judgments-file',str(fixture),'--env-file','no-such-env')
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output['source_mode'], 'controlled_judgments')
        self.assertTrue(output['routing_decision']['multi_agent'])
        self.assertEqual(len(output['prepared_agents']), 2)

    def test_cli_demo_is_labelled_controlled_and_covers_six_flows(self):
        result = self.cli('--demo','--env-file','no-such-env')
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output['source_mode'], 'controlled_demo_not_model_test')
        self.assertEqual(len(output['cases']), 6)
        self.assertEqual([case['routing_decision']['mode'] for case in output['cases']],
                         ['single','single','multiple','fallback','fallback','fallback'])

    def test_invalid_fixture_rejected_no_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)/'bad.json'
            fixture.write_text('{"probabilities":{"C02":0.99}}',encoding='utf-8')
            result = self.cli('--message','格式？','--judgments-file',str(fixture))
        self.assertEqual(result.returncode, 1)
        self.assertNotIn('Traceback', result.stderr)
