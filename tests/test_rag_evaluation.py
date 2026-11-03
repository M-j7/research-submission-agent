"""Source-grounded retrieval data and collection contracts (no network)."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module('evaluation.rag.dataset')
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, 'retrieval dataset builder is missing')

    def fixture(self):
        parents = [
            {'chunk_id': 'p1', 'text': 'Abstracts require 150 to 250 words.',
             'metadata': {'title': 'Guide', 'source_url': 'https://example.org/guide'}},
            {'chunk_id': 'p2', 'text': 'Unrelated text.', 'metadata': {}},
        ]
        cases = [{'case_id': 'abstract', 'category': 'format',
                  'zh': '摘要多少词？', 'en': 'How many words should the abstract contain?',
                  'reference_zh': '150到250词。', 'reference_en': '150 to 250 words.',
                  'supports': [{'parent_id': 'p1', 'quote': '150 to 250 words'}]}]
        return parents, cases

    def test_full_corpus_and_bilingual_labels_are_exported_with_provenance(self):
        parents, cases = self.fixture()
        with tempfile.TemporaryDirectory() as d:
            manifest = self.api.build_dataset(parents, cases, Path(d))
            corpus = self.api.read_jsonl(Path(d) / 'corpus.jsonl')
            queries = self.api.read_jsonl(Path(d) / 'queries.jsonl')
            qrels = self.api.read_jsonl(Path(d) / 'qrels.jsonl')
            audit = self.api.read_jsonl(Path(d) / 'annotations.jsonl')
            self.assertEqual([r['_id'] for r in corpus], ['p1', 'p2'])
            self.assertEqual(queries[0]['text'], '摘要多少词？')
            self.assertEqual(qrels, [
                {'query-id': 'abstract.zh', 'corpus-id': 'p1', 'score': 1},
                {'query-id': 'abstract.en', 'corpus-id': 'p1', 'score': 1},
            ])
            self.assertEqual(audit[0]['supports'][0]['source_url'], 'https://example.org/guide')
            self.assertEqual(audit[0]['review_status'], 'agent_draft')
            self.assertEqual(manifest['corpus_count'], 2)
            self.assertFalse(manifest['exhaustive_qrels'])
            self.assertEqual(len(manifest['files']['qrels.jsonl']), 64)

    def test_missing_quote_or_parent_is_not_accepted_as_ground_truth(self):
        for field, value in [('quote', 'not in source'), ('parent_id', 'missing')]:
            parents, cases = self.fixture()
            cases[0]['supports'][0][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(ValueError, 'support'):
                    self.api.build_dataset(parents, cases, Path(d))
                self.assertFalse((Path(d) / 'qrels.jsonl').exists())

    def test_duplicate_ids_and_empty_support_are_rejected(self):
        parents, cases = self.fixture()
        for p, c in [(parents + parents, cases), (parents, cases + cases)]:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    self.api.build_dataset(p, c, Path(d))
        cases[0]['supports'] = []
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'support'):
                self.api.build_dataset(parents, cases, Path(d))

    def test_replacement_characters_and_non_chinese_zh_queries_are_rejected(self):
        for query in ['bad\ufffdquery', '???']:
            parents, cases = self.fixture()
            cases[0]['zh'] = query
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(ValueError, 'query'):
                    self.api.build_dataset(parents, cases, Path(d))

    def test_rank_scores_preserve_order_and_reject_duplicate_ids(self):
        self.assertEqual(self.api.ranking_to_run(['p2', 'p1']), {'p2': 2.0, 'p1': 1.0})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.api.ranking_to_run(['p1', 'p1'])

    def test_collection_retains_failed_queries_and_rejects_coverage_gaps(self):
        rows = [{'query_id': 'q1', 'status': 'failed', 'parent_ids': []}]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.api.validate_collection(rows, {'q1', 'q2'})
        self.api.validate_collection(rows, {'q1'})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.api.validate_collection(rows + rows, {'q1'})

    def test_existing_dataset_is_idempotent_but_never_overwritten_with_new_labels(self):
        parents, cases = self.fixture()
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            self.api.build_dataset(parents, cases, output)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            self.api.build_dataset(parents, cases, output)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})
            cases[0]['zh'] = '摘要的词数范围是什么？'
            with self.assertRaisesRegex(ValueError, 'immutable'):
                self.api.build_dataset(parents, cases, output)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
