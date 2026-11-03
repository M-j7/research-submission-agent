import asyncio
import importlib
import importlib.util
import math
import json
import tempfile
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


class ScoringTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module('evaluation.rag.ragas_score')
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, 'official RAGAS adapter is missing')

    def test_real_zero_is_not_a_judge_failure(self):
        self.assertEqual(self.api.metric_result(0)['status'], 'ok')
        self.assertEqual(self.api.metric_result(0)['value'], 0)
        for value in (None, math.nan, math.inf):
            result = self.api.metric_result(value)
            self.assertEqual(result['status'], 'unavailable')
            self.assertIsNone(result['value'])

    def test_summary_keeps_failed_denominators_and_language_groups(self):
        rows = [{'sample_id': str(i), 'config_id': 'B4_multiquery_no_jev', 'language': 'zh',
                 'metrics': {k: self.api.metric_result(v) for k in self.api.METRICS}}
                for i, v in enumerate([1, 0, None])]
        groups = self.api.summarize(rows)
        for group in ('all', 'zh'):
            actual = groups['B4_multiquery_no_jev'][group]['faithfulness']
            self.assertEqual(actual, {'total': 3, 'valid': 2, 'unavailable': 1, 'mean': .5})

    def test_invalid_generation_and_missing_reference_never_reach_judge(self):
        records = [dict(sample_id='a', config_id='B3', language='zh', generation_status='failed',
                        user_input='q', response='', reference='r', retrieved_contexts=['c']),
                   dict(sample_id='b', config_id='B3', language='en', generation_status='ok',
                        user_input='q', response='a', reference='', retrieved_contexts=['c'])]
        result = asyncio.run(self.api.score_records(records, judge=None, embeddings=None))
        self.assertEqual([r['sample_id'] for r in result], ['a', 'b'])
        self.assertTrue(all(r['metrics']['answer_correctness']['value'] is None for r in result))
        self.assertEqual(result[0]['metrics']['faithfulness']['reason'], 'generation_failed')
        self.assertEqual(result[1]['metrics']['answer_correctness']['reason'], 'missing_reference')

    def test_score_load_rejects_reference_drift_and_keeps_all_ids(self):
        from generation_fixture import make_source
        from evaluation.rag.generation_data import prepare_records
        from evaluation.rag.generate import initialize_run
        from evaluation.rag.dataset import write_jsonl, file_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_source(root)
            records = prepare_records([('B3', source, 'B3')])
            run = root / 'answers'
            class Model:
                model = 'fixture'
            manifest = initialize_run(run, records, Model())
            answers = [{**r, 'response': 'answer', 'generation_status': 'ok'} for r in records]
            write_jsonl(run / 'answers.jsonl', answers)
            manifest.update(complete=True, answers_sha256=file_hash(run / 'answers.jsonl'))
            (run / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            scores = run / 'ragas'
            scores.mkdir()
            rows = [{k: a[k] for k in ('sample_id', 'query_id', 'config_id', 'language', 'input_sha256', 'generation_status')}
                    | {'metrics': {m: self.api.metric_result(0.) for m in self.api.METRICS}} for a in answers]
            write_jsonl(scores / 'scores.jsonl', rows)
            (scores / 'manifest.json').write_text(json.dumps({'complete': True, 'answers_sha256': manifest['answers_sha256'],
                'scores_sha256': file_hash(scores / 'scores.jsonl')}), encoding='utf-8')
            self.assertEqual(len(self.api.load_scores(run, scores)[2]), 2)
            # No input/reference changes are tolerated even if score rows are untouched.
            answers[0]['reference'] = 'altered reference'
            write_jsonl(run / 'answers.jsonl', answers)
            with self.assertRaises(ValueError):
                self.api.load_scores(run, scores)

    @unittest.skipUnless(importlib.util.find_spec('ragas'), 'optional RAGAS environment')
    def test_official_metric_identity_fields_and_no_parser_retries(self):
        metrics, descriptions = self.api.make_metrics(None, None, initialize=False)
        self.assertEqual(set(metrics), set(self.api.METRICS))
        self.assertEqual(type(metrics['faithfulness']).__module__, 'ragas.metrics._faithfulness')
        self.assertEqual(type(metrics['context_precision']).__name__, 'LLMContextPrecisionWithReference')
        self.assertEqual(metrics['answer_correctness'].weights, [.75, .25])
        self.assertEqual(metrics['answer_relevancy'].strictness, 3)
        self.assertIn('reference', descriptions['context_recall']['required_fields'])
        self.assertTrue(all(d['parser_retries'] == 0 for d in descriptions.values()))

    @unittest.skipUnless(importlib.util.find_spec('ragas'), 'optional RAGAS environment')
    def test_malformed_official_judge_response_is_one_call_not_hidden_repair(self):
        import httpx
        from ragas.dataset_schema import SingleTurnSample
        calls = []
        def reply(request):
            import json
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                'model': 'fixture', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'NOT_JSON'},
                                                'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15}})
        values = dict(LLM_BASE_URL='https://fixture.invalid/v1', LLM_API_KEY='secret-sentinel', LLM_MODEL='fixture',
                      EMBEDDING_BASE_URL='https://fixture.invalid/v1', EMBEDDING_API_KEY='secret-sentinel', EMBEDDING_MODEL='embed')
        async def run():
            with httpx.Client(transport=httpx.MockTransport(reply)) as sync_http:
                async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as async_http:
                    judge, embeddings, settings = self.api.make_clients(values, sync_http, async_http)
                    metrics, _ = self.api.make_metrics(judge, embeddings)
                    metrics['answer_relevancy'].strictness = 1
                    with self.assertRaises(Exception):
                        await metrics['answer_relevancy'].single_turn_ascore(
                            SingleTurnSample(user_input='q', response='a', retrieved_contexts=['c']))
        asyncio.run(run())
        self.assertEqual(len(calls), 1, 'RAGAS parser must not make additional paid repair calls')

    @unittest.skipUnless(importlib.util.find_spec('ragas'), 'optional RAGAS environment')
    def test_concurrent_rows_do_not_share_mutable_judge_sampling_settings(self):
        from langchain_openai import ChatOpenAI
        from ragas.llms.base import LangchainLLMWrapper
        judge = LangchainLLMWrapper(ChatOpenAI(model='fixture', api_key='sentinel', n=1))
        left, _ = self.api.make_metrics(judge, None, initialize=False)
        right, _ = self.api.make_metrics(judge, None, initialize=False)
        left['answer_relevancy'].llm.langchain_llm.n = 3
        self.assertEqual(right['faithfulness'].llm.langchain_llm.n, 1)

    @unittest.skipUnless(importlib.util.find_spec('ragas'), 'optional RAGAS environment')
    def test_relevancy_keeps_three_samples_when_endpoint_only_supports_n_one(self):
        import json
        import httpx
        from ragas.dataset_schema import SingleTurnSample
        calls = []
        def transport(request):
            payload = json.loads(request.content)
            if request.url.path.endswith('/embeddings'):
                texts = payload['input']
                return httpx.Response(200, json={'data': [{'index': i, 'embedding': [1., 0.], 'object': 'embedding'}
                                                         for i, _ in enumerate(texts)], 'model': 'embed',
                                                'usage': {'prompt_tokens': 1, 'total_tokens': 1}})
            calls.append(payload)
            if payload.get('n', 1) != 1:
                return httpx.Response(400, json={'error': {'message': 'only n=1 supported', 'type': 'invalid_request_error'}})
            return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                'model': 'fixture', 'choices': [{'index': 0, 'message': {'role': 'assistant',
                    'content': '{"question":"What is the passphrase?","noncommittal":0}'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 2, 'total_tokens': 4}})
        values = dict(LLM_BASE_URL='https://fixture.invalid/v1', LLM_API_KEY='sentinel', LLM_MODEL='fixture',
                      EMBEDDING_BASE_URL='https://fixture.invalid/v1', EMBEDDING_API_KEY='sentinel', EMBEDDING_MODEL='embed')
        async def run():
            with httpx.Client(transport=httpx.MockTransport(transport)) as sync_http:
                async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as async_http:
                    judge, embeddings, _ = self.api.make_clients(values, sync_http, async_http)
                    metrics, _ = self.api.make_metrics(judge, embeddings)
                    return await metrics['answer_relevancy'].single_turn_ascore(
                        SingleTurnSample(user_input='What is the passphrase?', response='ORCHID-17'))
        self.assertAlmostEqual(asyncio.run(run()), 1.)
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
