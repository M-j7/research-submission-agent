"""Score exported real rankings with MTEB/pytrec_eval, optionally through EvalScope."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path

from .dataset import PROJECT, file_hash, ranking_to_run, read_jsonl, validate_collection

KS = (1, 3, 5)


def score_rankings(qrels, rankings, *, ks=KS):
    if not qrels or set(qrels) != set(rankings):
        raise ValueError('qrels/rankings query coverage mismatch')
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('invalid metric cutoffs')
    if any(not docs or not any(v > 0 for v in docs.values()) for docs in qrels.values()):
        raise ValueError('each query requires positive qrels')
    runs = {qid: ranking_to_run(ids) for qid, ids in rankings.items()}
    os.environ.setdefault('MTEB_CACHE', str(PROJECT / '.cache/evaluation/mteb'))
    import pytrec_eval
    from mteb._evaluators.retrieval_metrics import mrr, parse_metrics_from_scores

    cutoff = ','.join(map(str, ks))
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f'{name}.{cutoff}' for name in
                 ('ndcg_cut', 'map_cut', 'recall', 'P', 'success')})
    raw = evaluator.evaluate(runs)
    # pytrec_eval can omit empty runs. Restore zero metrics, never fake a document.
    for qid in qrels:
        raw.setdefault(qid, {f'{name}_{k}': 0.0 for name in
                       ('ndcg_cut', 'map_cut', 'recall', 'P', 'success') for k in ks})
    ndcg, _, recall, precision, *_, hit = parse_metrics_from_scores(raw, ks)
    reciprocal_ranks = mrr(qrels, runs, ks)
    aggregate = {**ndcg, **recall, **hit,
                 **{name.replace('P@', 'Precision@'): value for name, value in precision.items()},
                 **{name: sum(values) / len(qrels) for name, values in reciprocal_ranks.items()}}
    per_query = {}
    for qid in qrels:
        per_query[qid] = {}
        for k in ks:
            for label, key in [('Recall', 'recall'), ('NDCG', 'ndcg_cut'),
                               ('Precision', 'P'), ('HitRate', 'success')]:
                per_query[qid][f'{label}@{k}'] = raw[qid][f'{key}_{k}']
            per_query[qid][f'MRR@{k}'] = mrr({qid: qrels[qid]}, {qid: runs[qid]}, [k])[f'MRR@{k}'][0]
    return {'aggregate': aggregate, 'per_query': per_query}


def load_run(run_dir):
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('collection is incomplete; do not silently grade a partial run')
    if file_hash(run_dir / 'rankings.jsonl') != manifest['rankings_sha256']:
        raise ValueError('rankings hash mismatch')
    dataset = Path(manifest['dataset_path'])
    if file_hash(dataset / 'manifest.json') != manifest['dataset_manifest_sha256']:
        raise ValueError('dataset manifest hash mismatch')
    for name, digest in manifest['dataset']['files'].items():
        if file_hash(dataset / name) != digest:
            raise ValueError(f'dataset hash mismatch: {name}')
    query_map = {r['_id']: r for r in read_jsonl(dataset / 'queries.jsonl')}
    expected = set(manifest['expected_queries'])
    corpus_ids = {r['_id'] for r in read_jsonl(dataset / 'corpus.jsonl')}
    qrels = {qid: {} for qid in expected}
    for row in read_jsonl(dataset / 'qrels.jsonl'):
        if row['query-id'] in expected:
            qrels[row['query-id']][row['corpus-id']] = row['score']
    rows = read_jsonl(run_dir / 'rankings.jsonl')
    if {r['baseline'] for r in rows} != set(manifest['baselines']):
        raise ValueError('baseline coverage mismatch')
    for baseline in manifest['baselines']:
        subset = [r for r in rows if r['baseline'] == baseline]
        validate_collection(subset, expected)
        if any(pid not in corpus_ids for r in subset for pid in r['parent_ids']):
            raise ValueError('ranking contains unknown parent ID')
    return manifest, query_map, qrels, rows


def summarize(run_dir, *, evalscope=False):
    manifest, queries, qrels, rows = load_run(run_dir)
    report = {'review_status': manifest['dataset']['review_status'],
              'semantic_validation': manifest.get('semantic_validation', 'jev'),
              'baseline_descriptions': manifest['baselines'],
              'limitations': manifest['dataset']['limitations'],
              'source': 'actual production rankings; offline grading; no business answer generation',
              'versions': {p: importlib.metadata.version(p) for p in ('mteb', 'pytrec-eval-terrier')},
              'baselines': {}}
    scored_rows = []
    for baseline in manifest['baselines']:
        current = [r for r in rows if r['baseline'] == baseline]
        current_map = {r['query_id']: r for r in current}
        views = {}
        for language in ('all', 'zh', 'en'):
            selected = [qid for qid in manifest['expected_queries']
                        if language == 'all' or queries[qid]['language'] == language]
            if not selected:
                continue
            result = score_rankings({qid: qrels[qid] for qid in selected},
                                    {qid: current_map[qid]['parent_ids'] for qid in selected})
            views[language] = {'query_count': len(selected), 'status_counts': dict(Counter(current_map[q]['status'] for q in selected)),
                               'metrics': result['aggregate']}
            if language == 'all':
                for qid, metrics in result['per_query'].items():
                    scored_rows.append({'baseline': baseline, 'query_id': qid, 'language': queries[qid]['language'],
                                        'question': queries[qid]['text'], 'status': current_map[qid]['status'],
                                        'warnings': current_map[qid]['warnings'], 'metrics': metrics})
        report['baselines'][baseline] = views
    from .dataset import write_jsonl
    write_jsonl(run_dir / 'scores.jsonl', scored_rows)
    if evalscope:
        from .evalscope_adapter import run_evalscope
        run_evalscope(run_dir)
        report['versions']['evalscope'] = importlib.metadata.version('evalscope')
        report['evalscope_output'] = str((run_dir / 'evalscope').resolve())
    (run_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# RAG 检索首轮诊断报告', '', '**Agent 草稿标注，尚非人工金标准；不得直接用于简历效果声明。**', '',
             '完整语料中检索，以父块评分；中英文是 15 组配对题，不是 30 个独立业务问题。', '',
             '| 基线 | 语言 | 题数 | Recall@5 | NDCG@5 | Precision@5 | MRR@5 | HitRate@5 | 状态 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for baseline, views in report['baselines'].items():
        for language, view in views.items():
            m = view['metrics']
            lines.append(f'| {baseline} | {language} | {view["query_count"]} | ' +
                         ' | '.join(f'{m[name+"@5"]:.4f}' for name in ('Recall', 'NDCG', 'Precision', 'MRR', 'HitRate')) +
                         f' | {view["status_counts"]} |')
    descriptions = {'dense': '向量', 'dense_bm25_rrf': '向量＋BM25＋RRF',
                    'dense_bm25_rrf_rerank': '混合检索＋原问题重排',
                    'production_multiquery_jev': 'Multi-Query＋JEV（查看降级状态）',
                    'experimental_multiquery_no_jev': 'Multi-Query，用户明确跳过 JEV 语义校验'}
    lines += ['', '## 如何解读', '',
              '- 本轮方案：' + '；'.join(f'{b}：{descriptions.get(v, v)}' for b, v in manifest['baselines'].items()) + '。',
              '- 语义校验模式：' + ('未调用 JEV，所有格式合法的改写直接参与；这不是 JEV 校验通过。' if manifest.get('semantic_validation') == 'skipped_by_user' else 'JEV；失败时仅保留原问题及已经通过的候选。'),
              '- 降级结果保留并单独计数，不能据此认定完整 Multi-Query 功能有效。失败题保留零分，不从分母移除。',
              '- 未标注结果暂按不相关计分；等价证据可能尚未标全，分数会受标注覆盖影响。',
              '- 全部 qrels 为二元相关性，NDCG 衡量位置，不代表已经建立分级相关性标准。',
              '- B1–B3 复用原问题召回，不能比较它们的独立端到端耗时。',
              '- 语料、题目和参数绑定 manifest 哈希；这次不评生成，不修改生产索引或 Redis。',
              '', '## 下一步', '', '复核 scores.jsonl 中的低分题及返回证据，补齐等价标注，冻结审核版本后重新评分；诊断并单独验证降级路径。', '']
    (run_dir / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--evalscope', action='store_true')
    args = parser.parse_args()
    report = summarize(args.run, evalscope=args.evalscope)
    print(json.dumps({b: v['all'] for b, v in report['baselines'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
