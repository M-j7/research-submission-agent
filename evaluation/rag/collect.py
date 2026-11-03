"""Collect actual parent rankings, without reading/writing the Redis result cache."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from .dataset import DEFAULT_DATA, PROJECT, file_hash, read_jsonl, validate_collection

sys.path.insert(0, str(PROJECT / 'src'))
from submission_assistant.runtime.rag import (  # noqa: E402
    ExistingKnowledgeBackend, JevRewriteJudge, ModelQueryRewriter, MultiQueryRAG, RagUnavailable,
)

BASELINES = {'B1': 'dense', 'B2': 'dense_bm25_rrf',
             'B3': 'dense_bm25_rrf_rerank', 'B4': 'production_multiquery_jev'}


class _RecallView:
    """Reuse one original recall across ablations; the production loop still ranks."""
    def __init__(self, backend, recalled, baseline):
        self.backend, self.recalled, self.baseline = backend, recalled, baseline
        self.version = backend.version

    async def recall(self, query):
        if isinstance(self.recalled, Exception):
            raise self.recalled
        if self.baseline != 'B1':
            return self.recalled
        # ExistingKnowledgeBackend returns dense first only when it succeeded.
        # Never label the remaining sparse path as a dense-only baseline.
        if any('向量召回不可用' in warning for warning in self.recalled.get('warnings', [])):
            raise RagUnavailable('dense baseline unavailable')
        rankings = self.recalled['rankings']
        if not rankings:
            raise RagUnavailable('dense baseline unavailable')
        return {**self.recalled, 'rankings': rankings[:1], 'warnings': []}

    async def rerank(self, query, rows):
        if self.baseline in {'B1', 'B2'}:
            return rows
        return await self.backend.rerank(query, rows)

    def expand(self, rows, k):
        return self.backend.expand(rows, k)


async def _unverified_search(backend, original, rewriter, top_k):
    """Explicit experimental ablation: no judge and no pretend-positive verdicts.

    Expand recall only; keep production RRF, original-query reranking and parent
    expansion unchanged. The normal production MultiQueryRAG remains fail-closed.
    """
    queries, warnings = [original], []
    try:
        if rewriter is None:
            raise ValueError('missing rewriter')
        proposed = await rewriter(original, [])
        if (not isinstance(proposed, list) or len(proposed) > 3
                or any(not isinstance(q, str) or not q.strip() or len(q) > 2000 for q in proposed)):
            raise ValueError('invalid rewrites')
        queries.extend(q for q in dict.fromkeys(q.strip() for q in proposed) if q not in queries)
    except asyncio.CancelledError:
        raise
    except Exception:
        warnings.append('实验模式查询改写失败或格式不合法，退回原问题；未调用JEV。')

    class ExpandedRecall:
        version = backend.version
        async def recall(self, query):
            semaphore = asyncio.Semaphore(2)
            async def one(text):
                async with semaphore:
                    return await backend.recall(text)
            outcomes = await asyncio.gather(*(one(text) for text in queries), return_exceptions=True)
            rankings, candidates, notes, successes = [], {}, list(warnings), 0
            for result in outcomes:
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, BaseException):
                    notes.append('部分查询召回失败，保留其余成功路径。')
                    continue
                successes += 1
                rankings.extend(result['rankings'])
                candidates.update(result['candidates'])
                notes.extend(result.get('warnings', []))
            if not successes:
                raise RagUnavailable('所有召回路径不可用。')
            return {'rankings': rankings, 'candidates': candidates, 'warnings': notes}

        async def rerank(self, query, rows):
            return await backend.rerank(query, rows)

        def expand(self, rows, k):
            return backend.expand(rows, k)

    output = await MultiQueryRAG(ExpandedRecall())._search(original, top_k)
    output.update(queries=queries, rewrite_attempts=1 if rewriter is not None else 0,
                  semantic_validation='skipped_by_user')
    return output


async def collect_query(backend, query, *, top_k=5, rewriter=None, judge=None, timeout_s=90,
                        skip_jev=False, baselines=None):
    baselines = tuple(BASELINES) if baselines is None else tuple(baselines)
    if not baselines or len(set(baselines)) != len(baselines) or any(b not in BASELINES for b in baselines):
        raise ValueError('invalid baseline selection')
    started = time.perf_counter()
    recalled = None
    if any(b != 'B4' for b in baselines):
        try:
            recalled = await asyncio.wait_for(backend.recall(query['text']), timeout_s)
        except Exception as error:
            recalled = error
    recall_seconds = time.perf_counter() - started
    rows = []
    for baseline in baselines:
        begin = time.perf_counter()
        row = {'query_id': query['_id'], 'language': query['language'], 'baseline': baseline,
               'baseline_description': BASELINES[baseline], 'parent_ids': [], 'evidence': [],
               'status': 'failed', 'warnings': [], 'queries': [query['text']],
               'cache_bypassed': True, 'shared_original_recall': baseline != 'B4',
               'multi_query_enabled': baseline == 'B4' and rewriter is not None and (skip_jev or judge is not None),
               'semantic_validation': ('skipped_by_user' if skip_jev else 'jev') if baseline == 'B4' else 'not_applicable'}
        if baseline == 'B4' and skip_jev:
            row['baseline_description'] = 'experimental_multiquery_no_jev'
        try:
            if baseline == 'B4' and skip_jev:
                output = await asyncio.wait_for(_unverified_search(backend, query['text'], rewriter, top_k), timeout_s)
            else:
                view = backend if baseline == 'B4' else _RecallView(backend, recalled, baseline)
                rag = MultiQueryRAG(view, rewriter=rewriter if baseline == 'B4' else None,
                                    judge=judge if baseline == 'B4' else None)
                # SAME production implementation below its cache boundary.
                output = await asyncio.wait_for(rag._search(query['text'], top_k), timeout_s)
            row.update(parent_ids=[e['evidence_id'] for e in output['evidence']],
                       evidence=output['evidence'], warnings=output['warnings'],
                       queries=output['queries'], rewrite_attempts=output['rewrite_attempts'],
                       status='degraded' if output['warnings'] else 'ok')
        except Exception as error:
            row['error_type'] = type(error).__name__  # No raw provider payloads/secrets.
        row['processing_seconds'] = round(time.perf_counter() - begin, 6)
        row['shared_recall_seconds'] = round(recall_seconds, 6) if baseline != 'B4' else None
        rows.append(row)
    return rows


def verify_sources(backend, dataset, values):
    """Fail before paid calls if the corpus or embedding identity drifted."""
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if file_hash(dataset / name) != digest:
            raise ValueError(f'dataset file hash mismatch: {name}')
    corpus = {r['_id']: r['text'] for r in read_jsonl(dataset / 'corpus.jsonl')}
    if corpus != {identifier: row['text'] for identifier, row in backend.parents.items()}:
        raise ValueError('evaluation corpus does not match production parents')
    index = json.loads((backend.chroma_dir / 'index_manifest.json').read_text(encoding='utf-8'))
    if (index['embedding_model'] != values.get('EMBEDDING_MODEL')
            or index['collection_name'] != backend.collection_name
            or index['children_sha256'] != file_hash(backend.persist.CHILDREN_PATH)
            or index['child_count'] != len(backend.children)):
        raise ValueError('production index identity mismatch')
    return manifest, index


async def run(args):
    import httpx
    from dotenv import dotenv_values
    from submission_assistant.intent.jev import JevSettings
    from submission_assistant.runtime.provider import RoleModelRouter

    values = {**dotenv_values(PROJECT / '.env', encoding='utf-8-sig', interpolate=False), **os.environ}
    for key in ('EMBEDDING_MODEL', 'EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY',
                'RERANKER_MODEL', 'RERANKER_BASE_URL', 'RERANKER_API_KEY'):
        if not values.get(key):
            raise ValueError(f'missing setting: {key}')
    async with httpx.AsyncClient(trust_env=False) as client:
        backend = ExistingKnowledgeBackend(PROJECT, values, client=client)
        dataset_manifest, index_manifest = verify_sources(backend, args.dataset, values)
        model = RoleModelRouter(values, client=client).for_role('general')
        judge = None if args.skip_jev else JevRewriteJudge(JevSettings.from_env(PROJECT / '.env'), client=client)
        baselines = {key: ('experimental_multiquery_no_jev' if key == 'B4' and args.skip_jev else BASELINES[key])
                     for key in args.baselines}
        queries = read_jsonl(args.dataset / 'queries.jsonl')
        if args.limit:
            queries = queries[:args.limit]
        if not queries:
            raise ValueError('no queries')
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                    'source_mode': 'live_retrieval_not_generation', 'expected_queries': [q['_id'] for q in queries],
                    'baselines': baselines, 'dataset_path': str(args.dataset.resolve()),
                    'dataset_manifest_sha256': file_hash(args.dataset / 'manifest.json'),
                    'dataset': dataset_manifest, 'index': index_manifest, 'backend_version': backend.version,
                    'models': {key: values.get(key) for key in ('EMBEDDING_MODEL', 'RERANKER_MODEL', 'LLM_MODEL', 'JEV_MODEL')},
                    'resolved_rewriter_model': model.model,
                    'resolved_jev_model': None if judge is None else judge.settings.model,
                    'semantic_validation': 'skipped_by_user' if args.skip_jev else 'jev',
                    'parameters': {'recall_per_path': 30, 'rrf_constant': 60, 'fused_candidates': 60,
                                   'rerank_candidates': 30, 'parent_top_k': 5, 'query_concurrency': 2},
                    'cache_policy': 'bypass final Redis/local result cache; do not clear existing caches',
                    'latency_note': 'B1-B3 share original recall and are NOT independent end-to-end latency measurements.',
                    'complete': False}
        manifest_path = args.output / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        collected = []
        semaphore = asyncio.Semaphore(2)
        async def one(query):
            async with semaphore:
                rows = await collect_query(backend, query, rewriter=ModelQueryRewriter(model), judge=judge,
                                           skip_jev=args.skip_jev, baselines=args.baselines)
                with (args.output / 'rankings.jsonl').open('a', encoding='utf-8') as target:
                    for row in rows:
                        target.write(json.dumps(row, ensure_ascii=False) + '\n')
                collected.extend(rows)
                print(json.dumps({'query_id': query['_id'], 'completed': len(collected) // len(baselines),
                                  'total': len(queries), 'statuses': {r['baseline']: r['status'] for r in rows}}, ensure_ascii=False), flush=True)
        await asyncio.gather(*(one(query) for query in queries))
        expected = {q['_id'] for q in queries}
        for baseline in baselines:
            validate_collection([r for r in collected if r['baseline'] == baseline], expected)
        manifest.update(complete=True, rankings_sha256=file_hash(args.output / 'rankings.jsonl'))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; 0 means all queries.')
    parser.add_argument('--baselines', nargs='+', choices=tuple(BASELINES), default=list(BASELINES))
    parser.add_argument('--skip-jev', action='store_true', help='Explicit experimental ablation: include all well-formed rewrites without semantic validation; never calls JEV.')
    parser.add_argument('--live', action='store_true', help='Explicitly enable configured paid model services.')
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required; collection calls configured model services')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    if len(set(args.baselines)) != len(args.baselines):
        parser.error('--baselines must not contain duplicates')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
