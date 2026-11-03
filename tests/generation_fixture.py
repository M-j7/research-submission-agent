"""Small real on-disk retrieval artifacts for generation adapter tests."""
import json
from pathlib import Path
from evaluation.rag.dataset import build_dataset, file_hash, read_jsonl, write_jsonl


def make_source(root, name='source', baseline='B3'):
    root = Path(root)
    dataset = root / (name + '-data')
    parents = [
        {'chunk_id': 'gold', 'text': 'Abstracts contain 150 to 250 words.', 'metadata': {}},
        {'chunk_id': 'actual', 'text': 'Submission files must not exceed 40 MB.', 'metadata': {}},
    ]
    cases = [{'case_id': 'abstract', 'category': 'format', 'zh': '摘要多少词？', 'en': 'How long is the abstract?',
              'reference_zh': 'REFERENCE_ONLY_SENTINEL 中文答案', 'reference_en': 'REFERENCE_ONLY_SENTINEL English',
              'supports': [{'parent_id': 'gold', 'quote': '150 to 250 words'}]}]
    dm = build_dataset(parents, cases, dataset)
    run = root / name
    run.mkdir()
    queries = read_jsonl(dataset / 'queries.jsonl')
    rows = [{'query_id': q['_id'], 'language': q['language'], 'baseline': baseline,
             'parent_ids': ['actual'], 'evidence': [{'evidence_id': 'actual', 'text': parents[1]['text']}],
             'queries': [q['text']], 'status': 'ok', 'warnings': [],
             'semantic_validation': 'skipped_by_user' if baseline == 'B4' else 'not_applicable'} for q in queries]
    write_jsonl(run / 'rankings.jsonl', rows)
    manifest = {'complete': True, 'baselines': {baseline: 'fixture'},
                'dataset_path': str(dataset.resolve()), 'dataset': dm,
                'dataset_manifest_sha256': file_hash(dataset / 'manifest.json'),
                'rankings_sha256': file_hash(run / 'rankings.jsonl'),
                'expected_queries': [q['_id'] for q in queries]}
    (run / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return run
