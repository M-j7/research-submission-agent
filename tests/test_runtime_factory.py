import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from runtime_fixture import write_test_tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class FactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_demo_never_opens_network(self):
        from submission_assistant.runtime.factory import create_runtime
        from submission_assistant.runtime.chat import build_chat_graph
        from submission_assistant.runtime.contracts import Request
        with tempfile.TemporaryDirectory() as folder:
            with patch('httpx.AsyncClient.send', side_effect=AssertionError('network forbidden')):
                bundle = await create_runtime(ROOT, mode='demo', data_dir=Path(folder))
                out = await build_chat_graph(bundle.service).ainvoke({'request': Request(user_id='local-user', conv_id='demo', message='摘要要求？')})
                self.assertIn('离线', out['answer'])
                self.assertEqual(bundle.mode, 'demo')
                await bundle.aclose()
                await bundle.aclose()

    async def test_invalid_live_does_not_become_demo(self):
        from submission_assistant.runtime.factory import create_runtime, RuntimeConfigurationError
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeConfigurationError) as caught:
                await create_runtime(Path(folder), mode='live')
            self.assertNotIn(folder, str(caught.exception))

    async def test_live_factory_uses_stable_profile(self):
        from submission_assistant.runtime.factory import create_runtime
        from submission_assistant.intent.stable_routing_prompts import STABLE_ROUTING_PROMPT_VERSION
        # External transports are not invoked by startup; local index constructor is isolated.
        settings = {'LLM_BASE_URL': 'https://example.test/v1', 'LLM_API_KEY': 'test', 'LLM_MODEL': 'qwen3.8-flash',
                    'JEV_PROVIDER': 'decisions', 'JEV_BASE_URL': 'https://api.perplexity.ai/v1/decisions',
                    'JEV_API_KEY': 'test', 'JEV_MODEL': 'pplx-decider-flash-v1.2',
                    'EMBEDDING_BASE_URL': 'https://example.test/v1', 'EMBEDDING_API_KEY': 'test', 'EMBEDDING_MODEL': 'test'}
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict('os.environ', {**settings, 'AGENT_TOKENIZER_PATH': str(write_test_tokenizer(Path(folder)))}), \
             patch('submission_assistant.runtime.rag.ExistingKnowledgeBackend'):
            bundle = await create_runtime(Path(folder), mode='live', data_dir=Path(folder) / 'live')
            self.assertEqual(bundle.service.recognizer.jev.prompt_version, STABLE_ROUTING_PROMPT_VERSION)
            await bundle.aclose()

    async def test_aclose_orders_shutdown(self):
        from submission_assistant.runtime.factory import RuntimeBundle
        order = []
        class Service:
            async def close(self): order.append('service')
        class Memory:
            def close(self): order.append('memory')
        class Resource:
            def __init__(self, name): self.name = name
            async def aclose(self): order.append(self.name)
        service = Service()
        service.memory = Memory()
        bundle = RuntimeBundle(service, 'demo', 'demo', client=Resource('http'), redis=Resource('redis'))
        await bundle.aclose()
        await bundle.aclose()
        self.assertEqual(order, ['service', 'memory', 'http', 'redis'])
