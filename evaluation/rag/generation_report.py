"""Report saved official RAGAS scores without new model requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ragas_score import METRICS, load_scores, summarize


def replay_values(row):
    values = {}
    for name in METRICS:
        metric = row['metrics'][name]
        if metric['status'] == 'ok':
            values[name] = metric['value']
        values[name + '_unavailable'] = int(metric['status'] != 'ok')
    values['sample_accounted'] = 1
    return values


def render_summary(rows, manifest):
    lines = ['# RAG 生成评测报告', '',
             '固定真实检索上下文的离线评测，不是完整 Agent 端到端评测。评分由官方 RAGAS 计算，EvalScope 仅回放保存的分数。', '',
             '每格为「有效评分均值（有效数/总数；不可用数）」。真实零分参与均值；生成失败、裁判异常及非有限分数不参与质量均值，但保留在总数中。', '',
             'B3_original_rerank：原问题双路召回、融合、重排及父块回溯。',
             'B4_multiquery_no_jev：未经 JEV 校验的多查询实验；不能视为生产方案。', '',
             '| 配置 | 语言 | Faithfulness | Correctness | Relevancy | ContextPrecision | ContextRecall |',
             '|---|---|---|---|---|---|---|']
    for config, groups in summarize(rows).items():
        for language, metrics in groups.items():
            cells = []
            for name in METRICS:
                metric = metrics[name]
                mean = f"{metric['mean']:.4f}" if metric['mean'] is not None else '不可用'
                cells.append(f"{mean}（{metric['valid']}/{metric['total']}；{metric['unavailable']}）")
            lines.append('| ' + ' | '.join([config, language] + cells) + ' |')
    lines += ['', '## 解读边界', '',
              '主指标为 Faithfulness 与 AnswerCorrectness，其余为诊断。Correctness 默认权重为事实一致性 0.75、语义相似度 0.25；Relevancy 是未裁剪余弦相似度，不等于准确率。', '',
              '15 组双语业务问题仍是小样本；两组相同问题配对，不能宣称统计显著、上线准确率或整个 Agent 的完成率。', '']
    lines += ['- ' + str(item) for item in manifest.get('limitations', [])]
    unavailable = [(r['sample_id'], k, m['reason']) for r in rows for k, m in r['metrics'].items() if m['status'] != 'ok']
    lines += ['', '## 不可用评分', '']
    lines += [f'- {sample} / {metric}: {reason}' for sample, metric, reason in unavailable] or ['无。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    score_dir = args.scores or args.run / 'ragas'
    manifest, _, rows = load_scores(args.run, score_dir)
    summary = summarize(rows)
    (score_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    path = score_dir / 'report.zh.md'
    path.write_text(render_summary(rows, manifest), encoding='utf-8')
    print(path.resolve())
    if args.native:
        from .generation_evalscope import run_evalscope
        run_evalscope(args.run, score_dir)


if __name__ == '__main__':
    main()
