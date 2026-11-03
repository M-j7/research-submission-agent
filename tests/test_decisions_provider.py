import asyncio
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import httpx
from submission_assistant.intent.jev import JevSettings, JevIntentRecognizer, JevConfigurationError, IntentInputError


class DecisionsProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_decisions_long_five_round_history_reaches_native_http(self):
        settings = self.settings()
        history = [m for i in range(5) for m in (
            {'role': 'user', 'content': f'q{i}'}, {'role': 'assistant', 'content': '答' * 1800})]
        bodies = []
        def handle(request):
            body = json.loads(request.content)
            bodies.append(body)
            return httpx.Response(200, json={'model': settings.model, 'answers': {
                k: {'type': 'noul', 'noul': .9} for k in body['questions']}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await JevIntentRecognizer(settings, client=client).recognize('这些呢？', history)
        self.assertTrue(result.complete)
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0]['state']['history'], history)

    async def test_oversized_decisions_state_is_rejected_before_http(self):
        def forbidden(request):
            self.fail('invalid input must not reach HTTP')
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            with self.assertRaises(IntentInputError):
                await JevIntentRecognizer(self.settings(), client=client).recognize('大' * 100000)

    def settings(self, **overrides):
        values = {'JEV_PROVIDER': 'decisions', 'JEV_BASE_URL': 'https://api.perplexity.ai/v1',
                  'JEV_API_KEY': 'unit-test-secret'}
        values.update(overrides)
        factory = getattr(JevSettings, 'from_values', None)
        self.assertIsNotNone(factory, 'explicit Decisions provider settings missing')
        return factory(values)

    async def test_decisions_uses_native_endpoint_and_keeps_independent_contract(self):
        settings = self.settings()
        requests = []
        def handle(request):
            self.assertEqual(str(request.url), 'https://api.perplexity.ai/v1/decisions')
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={'model': 'pplx-decider-flash-v1.2',
                'answers': {k: {'type': 'noul', 'noul': .9} for k in body['questions']},
                'usage': {'input_tokens': 7, 'output_tokens': 1}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await JevIntentRecognizer(settings, client=client).recognize('IEEE Access版面费是多少？')
        self.assertTrue(result.complete)
        self.assertEqual(sorted(len(r['questions']) for r in requests), [5, 6])
        self.assertEqual(settings.model, 'pplx-decider-flash-v1.2')
        self.assertNotIn('unit-test-secret', repr(settings))

    async def test_provider_url_validation_does_not_accept_router_or_credentials(self):
        for url in ['https://api.perplexity.ai/router/v1', 'https://user:pass@api.perplexity.ai/v1',
                    'http://api.perplexity.ai/v1', 'https://api.perplexity.ai/v1?key=secret']:
            with self.subTest(url=url), self.assertRaises(JevConfigurationError):
                self.settings(JEV_BASE_URL=url)
        with self.assertRaises(JevConfigurationError):
            self.settings(JEV_PROVIDER='unknown')
        self.assertEqual(JevSettings('https://test.example/v1', 'test').endpoint,
                         'https://test.example/v1/systemone')

    async def test_malformed_probability_is_not_reported_as_complete(self):
        settings = self.settings(JEV_TRUST_ENV='false')
        self.assertFalse(settings.trust_env)
        def handle(request):
            body = json.loads(request.content)
            return httpx.Response(200, json={'model': settings.model,
                'answers': {k: {'type': 'noul', 'noul': 1.1} for k in body['questions']}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await JevIntentRecognizer(settings, client=client).recognize('投稿如何操作')
        self.assertFalse(result.complete)
        self.assertTrue(all(b.status == 'failed' for b in result.batches))


if __name__ == '__main__':
    unittest.main()
