"""Build a full-corpus, bilingual, source-auditable retrieval pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[2]
PARENTS = PROJECT / '解析试验/父子检索/结果/20261006-qwen-child-128/parents.jsonl'
DEFAULT_DATA = PROJECT / 'evaluation/rag/data/ieee-bilingual-v1'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(text.split())


def unique_map(rows, field):
    result = {}
    for row in rows:
        identifier = row.get(field)
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f'invalid {field}')
        if identifier in result:
            raise ValueError(f'duplicate {field}: {identifier}')
        result[identifier] = row
    return result


def build_dataset(parents, cases, output):
    parent_map = unique_map(parents, 'chunk_id')
    unique_map(cases, 'case_id')
    if not parent_map or not cases:
        raise ValueError('empty corpus/cases')
    corpus = []
    for identifier, parent in parent_map.items():
        if not isinstance(parent.get('text'), str) or not parent['text'].strip():
            raise ValueError('empty corpus text')
        metadata = parent.get('metadata', {})
        corpus.append({'_id': identifier, 'title': metadata.get('title', ''),
                       'text': parent['text'], 'metadata': metadata})
    queries, qrels, annotations = [], [], []
    for case in cases:
        supports = case.get('supports')
        if not supports:
            raise ValueError(f'missing support: {case["case_id"]}')
        audited, seen = [], set()
        for support in supports:
            parent_id, quote = support.get('parent_id'), support.get('quote')
            if parent_id not in parent_map or not isinstance(quote, str) or not quote.strip():
                raise ValueError(f'invalid support: {case["case_id"]}')
            if normalized(quote) not in normalized(parent_map[parent_id]['text']):
                raise ValueError(f'support quote missing: {case["case_id"]}/{parent_id}')
            if parent_id in seen:
                raise ValueError('duplicate support parent')
            seen.add(parent_id)
            metadata = parent_map[parent_id].get('metadata', {})
            audited.append({**support, 'score': 1, 'source_url': metadata.get('source_url'),
                            'source_file': metadata.get('source_file'),
                            'source_locator': metadata.get('header_path') or metadata.get('page_number'),
                            'source_sha256': metadata.get('source_sha256')})
        for language in ('zh', 'en'):
            text = case.get(language)
            if (not isinstance(text, str) or not text.strip() or '\ufffd' in text
                    or (language == 'zh' and not re.search(r'[\u3400-\u9fff]', text))):
                raise ValueError(f'invalid query: {case["case_id"]}/{language}')
            query_id = f'{case["case_id"]}.{language}'
            queries.append({'_id': query_id, 'text': text, 'language': language,
                            'case_id': case['case_id'], 'category': case.get('category', '')})
            qrels.extend({'query-id': query_id, 'corpus-id': s['parent_id'], 'score': 1} for s in audited)
        annotations.append({**case, 'supports': audited, 'review_status': 'agent_draft',
                            'exhaustive': False, 'relevance_scale': 'binary'})
    output = Path(output)
    artifacts = {'corpus.jsonl': corpus, 'queries.jsonl': queries,
                 'qrels.jsonl': qrels, 'annotations.jsonl': annotations}
    serialized = {name: ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
                  for name, rows in artifacts.items()}
    manifest = {'dataset_version': 'ieee-bilingual-v1', 'scoring_unit': 'parent_chunk',
                'corpus_count': len(corpus), 'case_count': len(cases), 'query_count': len(queries),
                'qrel_count': len(qrels), 'review_status': 'agent_draft', 'exhaustive_qrels': False,
                'relevance_scale': 'binary', 'languages': ['zh', 'en'],
                'limitations': ['Source-derived synthetic questions, not real user traffic.',
                               'Bilingual pairs are paired observations, not independent questions.',
                               'Unjudged documents are treated as non-relevant by standard metrics; incomplete labels can bias scores.',
                               'Not a human-reviewed gold set and not resume-ready evidence.'],
                'files': {name: hashlib.sha256(text.replace('\n', os.linesep).encode('utf-8')).hexdigest()
                          for name, text in serialized.items()}}
    # Exported versions are immutable: identical rebuilds are no-ops, changed
    # inputs require a new directory. Check EVERYTHING before writing ANY file.
    if output.exists() and any(output.iterdir()):
        unchanged = all((output / name).is_file() and file_hash(output / name) == digest
                        for name, digest in manifest['files'].items())
        existing_manifest = output / 'manifest.json'
        try:
            unchanged = unchanged and json.loads(existing_manifest.read_text(encoding='utf-8')) == manifest
        except (OSError, ValueError):
            unchanged = False
        if not unchanged:
            raise ValueError('immutable dataset exists; use a new output/version directory')
        return manifest
    for name, rows in artifacts.items():
        write_jsonl(output / name, rows)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def ranking_to_run(parent_ids):
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError('duplicate parent IDs in ranking')
    return {identifier: float(len(parent_ids) - index) for index, identifier in enumerate(parent_ids)}


def validate_collection(rows, expected_query_ids):
    mapped = unique_map(rows, 'query_id')
    if set(mapped) != set(expected_query_ids):
        raise ValueError('query coverage mismatch')
    for row in rows:
        if row.get('status') not in {'ok', 'degraded', 'failed'}:
            raise ValueError('invalid collection status')
        ranking_to_run(row['parent_ids'])
        if row['status'] == 'failed' and row['parent_ids']:
            raise ValueError('failed query has nonempty ranking')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents', type=Path, default=PARENTS)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('cases.json'))
    parser.add_argument('--output', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    manifest = build_dataset(read_jsonl(args.parents), json.loads(args.cases.read_text(encoding='utf-8')), args.output)
    print(json.dumps({k: manifest[k] for k in ('corpus_count', 'case_count', 'query_count', 'qrel_count', 'review_status')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
