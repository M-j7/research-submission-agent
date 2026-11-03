"""Real graph/memory checks; external classifier and generator only are controlled."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import httpx
from submission_assistant.intent.jev import JevSettings, JevIntentRecognizer, IntentInputError
from submission_assistant.intent.ensemble import EnsembleIntentRecognizer
from submission_assistant.intent.embedding import EmbeddingSettings, EmbeddingIntentRecognizer
from submission_assistant.runtime.chat import ChatService, build_chat_graph
from submission_assistant.runtime.contracts import Request
from submission_assistant.runtime.memory import MemoryManager


class RoutingHistoryBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_ensemble_input_reaches_neither_paid_service(self):
        def forbidden(request):
            self.fail('oversized input reached a paid client')
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            recognizer = EnsembleIntentRecognizer(
                JevIntentRecognizer(JevSettings('https://api.perplexity.ai', 'fixture', provider='decisions'), client=client),
                EmbeddingIntentRecognizer(EmbeddingSettings('https://embedding.example.test/v1', 'fixture', 'embedding-test'), client=client))
            with self.assertRaises(IntentInputError):
                await recognizer.recognize('大' * 100000)

    async def test_overlapping_completed_turns_remain_paired_for_the_next_route(self):
        with tempfile.TemporaryDirectory() as d:
            memory = MemoryManager(Path(d) / 'memory.sqlite3')
            try:
                a = Request(user_id='u', conv_id='c', turn_id='a', message='问题A')
                b = Request(user_id='u', conv_id='c', turn_id='b', message='问题B')
                memory.begin_turn(a)
                memory.begin_turn(b)
                memory.finish_turn(b, '回答B')
                memory.finish_turn(a, '回答A')
                recent = memory.public_snapshot(b)['recent']
                self.assertEqual([(r['role'], r['content']) for r in recent], [
                    ('user', '问题A'), ('assistant', '回答A'), ('user', '问题B'), ('assistant', '回答B')])
            finally:
                memory.close()

    async def test_ensemble_uses_decisions_limit_before_any_paid_path(self):
        bodies = []
        def handle(request):
            body = json.loads(request.content)
            if request.url.host == 'api.perplexity.ai':
                bodies.append(body)
                return httpx.Response(200, json={'model': 'pplx-decider-flash-v1.2', 'answers': {
                    k: {'type': 'noul', 'noul': .99 if k in ('clarity', 'C03') else .01}
                    for k in body['questions']}})
            return httpx.Response(200, json={'data': [
                {'index': i, 'embedding': [1., 0.]} for i in range(len(body['input']))]})
        history = [m for i in range(5) for m in (
            {'role': 'user', 'content': f'问题{i}'}, {'role': 'assistant', 'content': '资料' * 1500})]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            recognizer = EnsembleIntentRecognizer(
                JevIntentRecognizer(JevSettings('https://api.perplexity.ai', 'fixture', provider='decisions',
                    model='pplx-decider-flash-v1.2'), client=client),
                EmbeddingIntentRecognizer(EmbeddingSettings('https://embedding.example.test/v1', 'fixture', 'embedding-test'), client=client))
            result = await recognizer.recognize('材料要求呢？', history)
        self.assertTrue(result['judgments']['complete'])
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0]['state']['history'], history)

    async def test_general_has_no_tools_or_skill_catalog_and_routes_only_final_public_pairs(self):
        inputs, requests = [], []
        class Model:
            async def complete(self, messages, tools, *, allow_tools=True):
                inputs.append((copy.deepcopy(messages), copy.deepcopy(tools), allow_tools))
                return {'message': {'role': 'assistant', 'content': '请说明想咨询的具体事项。'},
                        'finish_reason': 'stop', 'usage': None}
        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={'model': 'pplx-decider-flash-v1.2', 'answers': {
                k: {'type': 'noul', 'noul': .1 if k == 'clarity' else .01} for k in body['questions']}})
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            skill = home / 'resources/skills/secret-skill/SKILL.md'
            skill.parent.mkdir(parents=True)
            skill.write_text('---\nname: secret-skill\ndescription: PRIVATE_SKILL_CATALOG\n---\nprivate', encoding='utf-8')
            memory = MemoryManager(home / 'memory.sqlite3')
            try:
                for i in range(6):
                    req = Request(user_id='u', conv_id='c', turn_id=f'old{i}', message=f'q{i}')
                    memory.begin_turn(req)
                    memory.create_run(req, f'run{i}', 'submission_requirements', ['C02'])
                    memory.append(req, f'run{i}', 'assistant', {'content': 'PRIVATE_INTERMEDIATE'}, 1)
                    memory.append(req, f'run{i}', 'tool', {'content': 'PRIVATE_TOOL', 'tool_call_id': 'x'}, 1)
                    memory.finish_turn(req, f'a{i}')
                current = Request(user_id='u', conv_id='c', turn_id='new', message='这个怎么办？')
                async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                    service = ChatService(home, memory, JevIntentRecognizer(JevSettings(
                        'https://api.perplexity.ai', 'fixture', provider='decisions', model='pplx-decider-flash-v1.2'), client=client), Model())
                    out = await build_chat_graph(service).ainvoke({'request': current})
                    await service.close()
                self.assertEqual(out['routing_decision']['agent_types'], ['general'])
                self.assertEqual(requests[0]['state']['history'], [m for i in range(1, 6) for m in (
                    {'role': 'user', 'content': f'q{i}'}, {'role': 'assistant', 'content': f'a{i}'})])
                self.assertNotIn('PRIVATE_', json.dumps(requests))
                self.assertEqual(inputs[0][1], [])
                self.assertFalse(inputs[0][2])
                self.assertNotIn('PRIVATE_SKILL_CATALOG', json.dumps(inputs))
                self.assertEqual(out['final_result']['tool_calls_used'], 0)
            finally:
                memory.close()
