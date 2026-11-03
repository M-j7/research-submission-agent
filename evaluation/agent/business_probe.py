"""Append targeted business demonstrations; retain new clarity and all thresholds."""
import argparse
import asyncio
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import random

from . import clarity_probe as core
from .ablation_design import request_key
from .prompt_probe import write_json
from .routing_ablation import validate_cases

PROJECT = core.PROJECT
CONTRASTS = {
    'C03': [
        {'current_message': '期刊要求的利益冲突表是否需要与稿件分开提交？', 'expected': True},
        {'current_message': '利益冲突表在网站传不上去，怎么排查？', 'expected': False},
        {'current_message': '除了需要哪些合规证明，也告诉我证明文件传不进平台怎么排查。', 'expected': True},
    ],
    'C05': [
        {'current_message': '作者确认表上传时进度一直不动，怎么办？', 'expected': True},
        {'current_message': '作者确认表要求以PDF还是扫描件形式提交？', 'expected': False},
        {'current_message': '作者确认表应该包含什么内容？上传页面一直转圈又如何处理？', 'expected': True},
    ],
    'C07': [
        {'current_message': '我想比较本刊开放获取的付款渠道与优惠资格。', 'expected': True},
        {'current_message': '支付页面打不开，如何恢复页面访问？', 'expected': False},
        {'current_message': '支付页面进不去怎么修？另外正式付款有哪些渠道？', 'expected': True},
    ],
    'C08': [
        {'current_message': '录用后出版前签署许可证，应如何选择许可选项？', 'expected': True},
        {'current_message': '录用通知收到了，现在只想问发表费用的优惠资格。', 'expected': False},
        {'current_message': '既想确认发表费用优惠，也想了解录用后制作排期。', 'expected': True},
    ],
}


def bodies_for(message, history, variant):
    if variant not in core.VARIANTS:
        raise ValueError('unknown variant')
    # Both variants retain the newly verified clarity prompt and the G6 base.
    bodies = core.bodies_for(message, history, 'candidate')
    if variant == 'candidate':
        for body in bodies:
            for key, question in body['questions'].items():
                if key in CONTRASTS:
                    examples = [{'history': [], **e} for e in CONTRASTS[key]]
                    question['instructions'] += '\n[补充正反对照示例，expected仅指本类别]\n' + json.dumps(examples, ensure_ascii=False, separators=(',', ':'))
    return bodies


def load_cases():
    old = json.loads((PROJECT / 'evaluation/agent/clarity-probes/20261022-paired-01/cases.json').read_text(encoding='utf-8'))
    for c in old:
        c['previous_group'] = c['group']
        c['group'] = 'regression_45'
    new = json.loads((PROJECT / 'evaluation/agent/business-probes/data/background-12.json').read_text(encoding='utf-8'))
    return old + new


def prepare(home):
    cases = load_cases()
    if not cases or len(cases) > 57 or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('invalid case count/IDs')
    for c in cases:
        validate_cases([c])
        if 'expected_clarity' not in c or (c['expected_clarity'] is not None and type(c['expected_clarity']) is not bool):
            raise ValueError('clarity annotation required')
    queries = []
    for c in cases:
        for variant in core.VARIANTS:
            bodies = bodies_for(c['message'], c['history'], variant)
            queries.append({'case_id': c['id'], 'variant': variant, 'bodies': bodies, 'query_key': request_key(bodies)})
    random.Random(20261022).shuffle(queries)
    plan = {'model': core.MODEL, 'thresholds': core.RoutingSettings(.75, .8).to_dict(), 'queries': queries}
    manifest = {'trial_id': 'background-paired-v1', 'phase': 'prepared', 'complete': False, 'cases': len(cases),
        'logical_rows': len(queries), 'max_http_attempts': 2*len(queries),
        'plan_sha256': request_key(plan), 'cases_sha256': request_key(cases),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'only C03/C05/C07/C08 demonstration suffixes; new clarity fixed; no criteria/threshold/model/history/default changes; no retries/Qwen/RAG/tools',
        'caveat': '45条旧题为已知回归题；新增8条相邻类别题及4条多意图题为同一助手构造的合成探针，非独立盲测或人工金标，不代表完整多意图专项覆盖。'}
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    for name, value in (('cases.json', cases), ('plan.json', plan), ('manifest.json', manifest)):
        write_json(home / name, value)
    print(json.dumps({k: manifest[k] for k in ('cases', 'logical_rows', 'max_http_attempts')}, ensure_ascii=False))
    return manifest


def summarize(rows, cases):
    output = core.summarize(rows)
    by = {c['id']: c for c in cases}
    for variant, groups in output.items():
        for group, s in groups.items():
            selected = [r for r in rows if r['variant'] == variant and (group == 'all' or r['group'] == group)
                        and r['available'] and r['expected_clarity'] is True]
            tp = fp = fn = 0
            for r in selected:
                actual, expected = set(r['route']['matched_intents']), set(by[r['case_id']]['expected_intents'])
                tp += len(actual & expected)
                fp += len(actual - expected)
                fn += len(expected - actual)
            s.update(business_scored=len(selected), tp=tp, fp=fp, fn=fn,
                     precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                     f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return output


def report(home, manifest, cases, rows, summary, usage):
    table = []
    for variant, groups in summary.items():
        for group, s in groups.items():
            values = [variant, group, f"{s['exact_route']}/{s['available']}", f"{s['exact_intents']}/{s['available']}",
                s['business_scored'], s['tp'], s['fp'], s['fn'], '—' if s['f1'] is None else f"{s['f1']:.1%}",
                f"{s['false_clarify']}/{s['clear_samples']}", f"{s['false_pass']}/{s['unclear_samples']}", s['unavailable']]
            table.append('<tr>' + ''.join('<td>'+escape(str(v))+'</td>' for v in values) + '</tr>')
    by = {(r['case_id'], r['variant']): r for r in rows}
    detail = []
    for c in cases:
        cells = [c['id'], c['message'], ','.join(c['expected_intents']) + ' / ' + str(c.get('expected_fallback'))]
        for variant in core.VARIANTS:
            row = by.get((c['id'], variant))
            cells.append('未执行' if row is None else '不可用' if not row['available'] else
                f"{row['route']['matched_intents']}；{row['route']['fallback_reason']}；处理正确={row['exact_route']}；严格正确={row['exact_intents']}")
        detail.append('<tr>' + ''.join('<td>'+escape(str(v))+'</td>' for v in cells) + '</tr>')
    html = '''<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>业务背景误选配对验证</title><style>body{max-width:1400px;margin:32px auto;padding:0 22px;font:14px/1.8 system-ui,"Microsoft YaHei";background:#f6f8fb;color:#26374b}table{width:100%;border-collapse:collapse;background:white;font-size:12px}td,th{border:1px solid #dce4ed;padding:9px;text-align:left;vertical-align:top}th{background:#eaf0f7}a{color:#285fa6}</style><h1>业务背景误选：补充正反例配对验证</h1>'''
    html += '<p>'+('全部覆盖核验完成。' if manifest['complete'] else '尚未全部完成。')+escape(manifest['caveat'])+'</p>'
    html += '<p>两版均使用新清晰度prompt，阈值0.75/0.8不变。仅C03、C05、C07、C08追加各3个正反示例，不改类别边界、公共指令或其他类别。没有部署到线上。</p>'
    html += '<p>TP/FP/FN及micro-F1只在预期清晰的业务题计算；模糊题可保留可识别的大类，但应正确澄清，非业务应正确兜底。处理正确和严格类别集合正确分开统计。FP=多选标签，FN=漏选标签。</p>'
    html += '<table><tr><th>版本</th><th>题组</th><th>处理正确/可用</th><th>严格类别正确/可用</th><th>明确业务题数</th><th>TP</th><th>FP</th><th>FN</th><th>F1</th><th>误澄清/应清晰</th><th>误放行/应模糊</th><th>不可用</th></tr>'+''.join(table)+'</table>'
    html += '<h2>逐题结果</h2><table><tr><th>ID</th><th>当前问题</th><th>预期类别/兜底</th><th>原业务示例</th><th>补充对照示例</th></tr>'+''.join(detail)+'</table>'
    html += '<p>实际HTTP用量：'+escape(json.dumps(usage, ensure_ascii=False))+'</p><p><a href="cases.json">题目/标签</a> · <a href="plan.json">冻结请求</a> · <a href="results.json">结果</a> · <a href="summary.json">统计</a> · <a href="telemetry.json">HTTP用量</a> · <a href="manifest.json">清单</a></p></html>'
    (home / 'report.html').write_text(html, encoding='utf-8')


async def run(home):
    cases = json.loads((Path(home) / 'cases.json').read_text(encoding='utf-8'))
    await core.run(home, summary_fn=lambda rows: summarize(rows, cases), report_fn=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output) if args.prepare else asyncio.run(run(args.output))
