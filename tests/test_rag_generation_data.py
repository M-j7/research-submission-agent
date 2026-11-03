import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation_fixture import make_source
from evaluation.rag.dataset import file_hash, read_jsonl, write_jsonl


class GenerationDataTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module('evaluation.rag.generation_data')
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, 'generation data adapter is missing')

    def test_fixed_actual_evidence_is_used_instead_of_gold_and_bilingual_refs_join(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = make_source(d, 'a'), make_source(d, 'b', 'B4')
            records = self.api.prepare_records([('A', a, 'B3'), ('B', b, 'B4')])
            self.assertEqual([r['sample_id'] for r in records], ['A:abstract.zh', 'A:abstract.en', 'B:abstract.zh', 'B:abstract.en'])
            self.assertEqual(records[0]['reference'], 'REFERENCE_ONLY_SENTINEL 中文答案')
            self.assertEqual(records[1]['reference'], 'REFERENCE_ONLY_SENTINEL English')
            self.assertEqual(records[0]['retrieved_contexts'], ['Submission files must not exceed 40 MB.'])
            self.assertEqual(records[-1]['semantic_validation'], 'skipped_by_user')
            messages = self.api.build_generation_messages(records[0])
            self.assertIn('Submission files', json.dumps(messages, ensure_ascii=False))
            self.assertNotIn('REFERENCE_ONLY_SENTINEL', json.dumps(messages))
            self.assertNotIn('150 to 250', json.dumps(messages))

    def test_duplicate_config_ids_and_modified_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = make_source(d)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                self.api.prepare_records([('A', source, 'B3'), ('A', source, 'B3')])
            with (source / 'rankings.jsonl').open('a', encoding='utf-8') as out:
                out.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.api.prepare_records([('A', source, 'B3')])

    def test_evidence_body_mismatch_rejected_even_if_run_hash_was_recomputed(self):
        with tempfile.TemporaryDirectory() as d:
            source = make_source(d)
            rows = read_jsonl(source / 'rankings.jsonl')
            rows[0]['evidence'][0]['text'] = 'wrong evidence text'
            write_jsonl(source / 'rankings.jsonl', rows)
            manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
            manifest['rankings_sha256'] = file_hash(source / 'rankings.jsonl')
            (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'evidence'):
                self.api.prepare_records([('A', source, 'B3')])

    def test_empty_evidence_is_preserved_and_reference_version_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            source = make_source(d)
            rows = read_jsonl(source / 'rankings.jsonl')
            rows[0].update(parent_ids=[], evidence=[], status='failed')
            write_jsonl(source / 'rankings.jsonl', rows)
            manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
            manifest['rankings_sha256'] = file_hash(source / 'rankings.jsonl')
            (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            records = self.api.prepare_records([('A', source, 'B3')])
            self.assertEqual(records[0]['retrieved_contexts'], [])
            dataset = Path(manifest['dataset_path'])
            (dataset / 'annotations.jsonl').write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.api.prepare_records([('A', source, 'B3')])


if __name__ == '__main__':
    unittest.main()
