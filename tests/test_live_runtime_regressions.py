"""Offline reproduction of production-only proxy and maintenance failures."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from runtime_fixture import write_test_tokenizer

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def local_settings(trust_env, tokenizer_path):
    return {
        'LLM_BASE_URL':'https://generation.test/v1', 'LLM_API_KEY':'test-only', 'LLM_MODEL':'qwen3.8-flash',
        'AGENT_TOKENIZER_PATH':str(tokenizer_path),
        'JEV_PROVIDER':'decisions', 'JEV_BASE_URL':'https://api.perplexity.ai/v1/decisions',
        'JEV_API_KEY':'test-only', 'JEV_MODEL':'pplx-decider-flash-v1.2', 'JEV_TRUST_ENV':str(trust_env).lower(),
        'EMBEDDING_BASE_URL':'https://embedding.test/v1', 'EMBEDDING_API_KEY':'test-only', 'EMBEDDING_MODEL':'test',
        'REDIS_URL':'',
    }


class LiveProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_routing_and_rewrite_judge_honor_proxy_configuration(self):
        from submission_assistant.runtime.factory import create_runtime
        real_client = httpx.AsyncClient
        for enabled in (True, False):
            clients, generation_policies = [], []
            def create_client(**kwargs):
                policy = kwargs.get('trust_env', True)
                def handle(request):
                    if request.url.host == 'api.perplexity.ai':
                        if policy != enabled:
                            raise httpx.ConnectTimeout('offline proxy selection mismatch', request=request)
                        body = json.loads(request.content)
                        return httpx.Response(200, json={'model':'pplx-decider-flash-v1.2', 'answers':{
                            k:{'type':'noul','noul':.99} for k in body['questions']}})
                    generation_policies.append(policy)
                    return httpx.Response(200, json={'choices':[{'message':{'role':'assistant','content':'离线答复'},
                        'finish_reason':'stop'}]})
                client = real_client(transport=httpx.MockTransport(handle), **kwargs)
                clients.append(client)
                return client
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as folder, \
                 patch.dict('os.environ', local_settings(enabled, write_test_tokenizer(Path(folder)))), \
                 patch('submission_assistant.runtime.rag.ExistingKnowledgeBackend'), \
                 patch('submission_assistant.runtime.factory.httpx.AsyncClient', side_effect=create_client):
                bundle = await create_runtime(Path(folder), mode='live')
                try:
                    result = await bundle.service.recognizer.jev.recognize('投稿材料要求是什么？')
                    self.assertTrue(result.complete)
                    self.assertEqual(await bundle.service.rag.judge('摘要要求', ['摘要字数要求']), {'摘要字数要求':True})
                    model = bundle.service.model.for_role('general')
                    await model.complete([{'role':'user','content':'离线请求'}], [])
                    self.assertEqual(generation_policies, [False])  # don't change Qwen's existing direct policy
                finally:
                    await bundle.aclose()
                self.assertTrue(all(client.is_closed for client in clients))

    async def test_partial_factory_failure_closes_all_owned_http_clients(self):
        from submission_assistant.runtime.factory import create_runtime, RuntimeConfigurationError
        real_client = httpx.AsyncClient
        clients = []
        def create_client(**kwargs):
            client = real_client(transport=httpx.MockTransport(lambda req: self.fail('startup must not call APIs')), **kwargs)
            clients.append(client)
            return client
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict('os.environ', local_settings(True, write_test_tokenizer(Path(folder)))), \
             patch('submission_assistant.runtime.factory.httpx.AsyncClient', side_effect=create_client), \
             patch('submission_assistant.runtime.rag.ExistingKnowledgeBackend', side_effect=ValueError('offline missing index')):
            with self.assertRaises(RuntimeConfigurationError):
                await create_runtime(Path(folder), mode='live')
        self.assertTrue(all(client.is_closed for client in clients))


class MaintenanceCommitTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_graph_with_maintenance_commits_and_schedules_on_event_loop(self):
        from submission_assistant.runtime.factory import create_runtime
        from submission_assistant.runtime.chat import build_chat_graph
        from submission_assistant.runtime.contracts import Request
        from submission_assistant.runtime.maintenance import MemoryMaintenance
        seen = []
        class NoPaidOps:
            async def short_summary(self, rows): raise AssertionError('no compression needed for one turn')
            async def profile_changes(self, rows, profile): raise AssertionError('no extraction due for one turn')
        class ObservedMaintenance(MemoryMaintenance):
            async def update(self, req):
                seen.append(asyncio.get_running_loop())
                return await super().update(req)
        with tempfile.TemporaryDirectory() as folder, patch('httpx.AsyncClient.send', side_effect=AssertionError('no external calls')):
            bundle = await create_runtime(ROOT, mode='demo', data_dir=Path(folder))
            bundle.service.maintenance = ObservedMaintenance(bundle.service.memory, NoPaidOps(), profile_every=100)
            try:
                req = Request(user_id='offline', conv_id='c', turn_id='t', message='摘要和费用')
                result = await build_chat_graph(bundle.service).ainvoke({'request':req})
                await bundle.service.close()
                self.assertIn('离线演示', result['answer'])
                self.assertEqual(seen, [asyncio.get_running_loop()])
                self.assertEqual(bundle.service.memory.completed_count('offline'), 1)
            finally:
                await bundle.aclose()

    async def test_web_real_graph_maintenance_does_not_mark_saved_answer_failed(self):
        from submission_assistant.runtime.factory import create_runtime
        from submission_assistant.runtime.maintenance import MemoryMaintenance
        from submission_assistant.web.jobs import JobController
        from submission_assistant.web.store import ConversationStore
        class NoPaidOps:
            async def short_summary(self, rows): raise AssertionError('no external call')
            async def profile_changes(self, rows, profile): raise AssertionError('no external call')
        with tempfile.TemporaryDirectory() as folder:
            bundle = await create_runtime(ROOT, mode='demo', data_dir=Path(folder))
            bundle.service.maintenance = MemoryMaintenance(bundle.service.memory, NoPaidOps(), profile_every=100)
            store = ConversationStore(bundle.service.memory, 'local-user')
            jobs = JobController(bundle, store)
            conv = store.create().id
            try:
                await jobs.submit(conv, 't1', '摘要和费用')
                async for event in jobs.events(conv, 't1'):
                    final = event['turn']
                # A persisted answer can reach SSE before the graph iterator finishes
                # its bookkeeping. Drain the owned job, never cancel it in this test.
                tasks = list(jobs._tasks.values())
                if tasks:
                    await asyncio.gather(*tasks)
                final = jobs.get(conv, 't1')
                self.assertEqual(final.state, 'completed')
                self.assertIsNone(final.error_code)
                self.assertEqual(final.status[0]['phase'], 'completed')
                with bundle.service.memory.transaction() as db:
                    self.assertEqual(db.execute('SELECT state FROM web_turns WHERE turn=?',('t1',)).fetchone()[0], 'completed')
            finally:
                await jobs.aclose()
                await bundle.aclose()
