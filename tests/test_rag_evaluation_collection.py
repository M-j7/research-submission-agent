import asyncio
import importlib
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


class CollectionTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module('evaluation.rag.collect')
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, 'retrieval collector is missing')

    def backend(self, warnings=None):
        class Backend:
            version = 'fixture'
            calls = 0
            async def recall(self, query):
                self.calls += 1
                return {'rankings': [['a', 'b'], ['b']],
                        'candidates': {'a': {'chunk_id': 'a'}, 'b': {'chunk_id': 'b'}},
                        'warnings': warnings or []}
            async def rerank(self, query, rows):
                return list(reversed(rows))
            def expand(self, rows, k):
                return [{'evidence_id': 'p' + r['chunk_id'], 'text': r['chunk_id']} for r in rows[:k]]
        return Backend()

    def test_baselines_use_real_loop_order_and_share_only_original_recall(self):
        backend = self.backend()
        rows = asyncio.run(self.api.collect_query(
            backend, {'_id': 'q.zh', 'text': '原问题', 'language': 'zh'}, top_k=5))
        self.assertEqual([r['baseline'] for r in rows], ['B1', 'B2', 'B3', 'B4'])
        self.assertEqual(rows[0]['parent_ids'], ['pa', 'pb'])
        self.assertEqual(rows[1]['parent_ids'], ['pb', 'pa'])
        self.assertEqual(rows[2]['parent_ids'], ['pa', 'pb'])
        self.assertEqual(backend.calls, 2)  # B1/B2/B3 share recall; B4 independently executes.
        self.assertTrue(all(r['status'] == 'ok' for r in rows))

    def test_failed_dense_path_is_not_silently_replaced_with_bm25(self):
        backend = self.backend(['向量召回不可用，保留BM25路径。'])
        rows = asyncio.run(self.api.collect_query(
            backend, {'_id': 'q.zh', 'text': '问题', 'language': 'zh'}))
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertEqual(rows[0]['parent_ids'], [])
        self.assertEqual(rows[1]['status'], 'degraded')

    def test_recall_errors_keep_every_baseline_and_do_not_leak_error_text(self):
        backend = self.backend()
        async def broken(query):
            raise RuntimeError('private-secret')
        backend.recall = broken
        rows = asyncio.run(self.api.collect_query(
            backend, {'_id': 'q.en', 'text': 'question', 'language': 'en'}))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['status'] == 'failed' and not r['parent_ids'] for r in rows))
        self.assertNotIn('private-secret', str(rows))

    def test_explicit_no_jev_experiment_recalls_all_rewrites_without_judge(self):
        self.assertIn('skip_jev', inspect.signature(self.api.collect_query).parameters,
                      'explicit unverified experiment mode is missing')
        backend = self.backend()
        recalled = []
        original_recall = backend.recall
        async def recall(query):
            recalled.append(query)
            return await original_recall(query)
        backend.recall = recall
        async def rewrite(query, rejected):
            return ['first rewrite', 'unrelated candidate', 'third rewrite']
        async def forbidden_judge(*args):
            raise AssertionError('JEV must not be called in the explicitly skipped experiment')
        rows = asyncio.run(self.api.collect_query(
            backend, {'_id': 'q.en', 'text': 'original', 'language': 'en'},
            rewriter=rewrite, judge=forbidden_judge, skip_jev=True, baselines=('B4',)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'ok')
        self.assertEqual(rows[0]['queries'], ['original', 'first rewrite', 'unrelated candidate', 'third rewrite'])
        self.assertEqual(set(recalled), set(rows[0]['queries']))
        self.assertEqual(len(recalled), 4)
        self.assertEqual(rows[0]['semantic_validation'], 'skipped_by_user')
        self.assertEqual(rows[0]['baseline_description'], 'experimental_multiquery_no_jev')

    def test_no_jev_mode_keeps_original_and_deduplicates_rewrites(self):
        self.assertIn('skip_jev', inspect.signature(self.api.collect_query).parameters)
        async def rewrite(*args):
            return ['original', ' rewrite ', 'rewrite']
        rows = asyncio.run(self.api.collect_query(
            self.backend(), {'_id': 'q.en', 'text': 'original', 'language': 'en'},
            rewriter=rewrite, skip_jev=True, baselines=('B4',)))
        self.assertEqual(rows[0]['queries'], ['original', 'rewrite'])

    def test_no_jev_mode_still_rejects_malformed_rewrites_and_marks_fallback(self):
        self.assertIn('skip_jev', inspect.signature(self.api.collect_query).parameters)
        for proposed in [[''], [None], ['x' * 2001], ['one', 'two', 'three', 'four']]:
            async def rewrite(*args):
                return proposed
            with self.subTest(proposed=proposed):
                rows = asyncio.run(self.api.collect_query(
                    self.backend(), {'_id': 'q.en', 'text': 'original', 'language': 'en'},
                    rewriter=rewrite, skip_jev=True, baselines=('B4',)))
                self.assertEqual(rows[0]['queries'], ['original'])
                self.assertEqual(rows[0]['status'], 'degraded')

    def test_default_mode_still_requires_a_successful_judge(self):
        async def rewrite(*args):
            return ['not verified']
        async def judge(*args):
            raise RuntimeError('unavailable')
        rows = asyncio.run(self.api.collect_query(
            self.backend(), {'_id': 'q.en', 'text': 'original', 'language': 'en'},
            rewriter=rewrite, judge=judge))
        self.assertEqual(rows[-1]['queries'], ['original'])
        self.assertEqual(rows[-1]['status'], 'degraded')


if __name__ == '__main__':
    unittest.main()
