"""One fixed routing-only comparison; never runs tools, RAG or answer generation."""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
import httpx
from dotenv import dotenv_values
from langsmith import tracing_context
from submission_assistant.agents import AgentRequest
from submission_assistant.graph import build_agent_graph
from submission_assistant.intent.jev import build_requests, JevSettings, JevIntentRecognizer
from submission_assistant.intent.routing_prompts import ROUTING_PROMPT_VERSION
from .observe import ObservationSink, PaidCallGate
from .runtime import decisions_values

DATA = PROJECT / 'evaluation/agent/prompt-probes/fewshot-v2'
PRIOR = PROJECT / 'evaluation/agent/routing-probes/20261021-history-five-02/routing.json'
EXPECTED_ROLES = {
    'C01': 'submission_requirements', 'C02': 'submission_requirements', 'C03': 'submission_requirements',
    'C04': 'publication_policy', 'C05': 'editorial_workflow', 'C06': 'editorial_workflow',
    'C07': 'publication_fees', 'C08': 'publication_policy', 'C09': 'editorial_workflow'}


def build_cases():
    # Prior cases are already-used development diagnostics, not unseen test data.
    old = json.loads(PRIOR.read_text(encoding='utf-8'))
    expected = {'conversation-01': ['C03'], 'conversation-05': ['C03'],
                'multi-04': ['C04', 'C09'], 'multi-05': ['C02', 'C07'],
                'five-round-reference': ['C03'], 'long-five-round-reference': ['C03'],
                'vague-control': [], 'nonbusiness-control': []}
    cases = []
    for r in old:
        cases.append({k: deepcopy(r[k]) for k in ('id', 'message', 'history', 'expected_roles')})
        cases[-1].update(group='known_diagnostic', expected_intents=expected[r['id']], previous_route=r['route'])
        if 'expected_fallback' in r:
            cases[-1]['expected_fallback'] = r['expected_fallback']
    new = json.loads((DATA / 'new-cases.json').read_text(encoding='utf-8'))
    for c in new:
        c['group'] = 'new_probe'
        c['expected_roles'] = sorted({EXPECTED_ROLES[i] for i in c['expected_intents']}) or ['general']
    return cases + new


class FrozenQuestionsClient:
    """Swap only question definitions; actual HTTP, parser and routing stay real."""
    def __init__(self, client, questions):
        self.client, self.questions = client, questions

    async def post(self, url, **kwargs):
        body = deepcopy(kwargs['json'])
        body['questions'] = {k: deepcopy(self.questions[k]) for k in body['questions']}
        return await self.client.post(url, **{**kwargs, 'json': body})


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def score(case, recognition, route):
    available = bool((recognition.get('judgments') or {}).get('complete'))
    fallback_ok = route['fallback_reason'] == case.get('expected_fallback')
    return {'available': available,
        'exact_route': available and set(route['agent_types']) == set(case['expected_roles']) and fallback_ok,
        'exact_intents': available and set(route['matched_intents']) == set(case['expected_intents']) and fallback_ok}


def summarize(rows):
    result = {}
    for group in ('known_diagnostic', 'new_probe'):
        result[group] = {}
        for variant in ('baseline', 'candidate'):
            selected = [r for r in rows if r['group'] == group and r['variant'] == variant]
            result[group][variant] = {'samples': len(selected),
                'available': sum(r['available'] for r in selected),
                'exact_route': sum(r['exact_route'] for r in selected),
                'exact_intents': sum(r['exact_intents'] for r in selected)}
    return result


def render_report(home, manifest, rows, cases, usage):
    summary = summarize(rows)
    write_json(home / 'summary.json', summary)
    by = {(r['variant'], r['id']): r for r in rows}
    def label(route):
        return ', '.join(route['agent_types']) + (f" ({route['fallback_reason']})" if route['fallback_reason'] else '')
    lines = []
    for c in cases:
        old, new = by.get(('baseline', c['id'])), by.get(('candidate', c['id']))
        before = label(old['route']) if old else label(c['previous_route']) + '〔历史结果〕' if 'previous_route' in c else '未执行'
        after = label(new['route']) if new else '未执行'
        probabilities = ((new or {}).get('recognition', {}).get('judgments') or {}).get('probabilities', {})
        scores = '；'.join(f'{k}={v:.3f}' for k, v in probabilities.items() if v >= .2)
        status = '未执行' if new is None else '不可用' if not new['available'] else '通过' if new['exact_route'] else '未通过'
        cells = [c['id'], c['group'], c['message'], ', '.join(c['expected_roles']), before, after, status, scores]
        lines.append('<tr>' + ''.join('<td>' + escape(x) + '</td>' for x in cells) + '</tr>')
    summaries = []
    for group, label_name in [('known_diagnostic', '已知诊断样本'), ('new_probe', '新增问法探针')]:
        for variant, stats in summary[group].items():
            if stats['samples']:
                summaries.append(f"<li>{label_name}／{variant}：角色完全匹配 {stats['exact_route']}/{stats['samples']}；"
                    f"类别完全匹配 {stats['exact_intents']}/{stats['samples']}；判断可用 {stats['available']}/{stats['samples']}</li>")
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>路由 Few-shot V2 对照验证</title><style>body{font:15px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#26364a;background:#f5f7fb;margin:30px auto;padding:0 22px;max-width:1500px}h1{font-size:27px}.card{background:white;border:1px solid #dce3ed;border-radius:10px;padding:20px;margin:18px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;border-bottom:1px solid #e4e9ef;vertical-align:top}th{background:#edf2f9}a{color:#2462a7}</style>
<h1>路由 Few-shot V2 对照验证</h1>'''
    html += '<div class="card"><strong>' + ('调用与核验完成' if manifest['complete'] else '尚未全部完成') + '</strong><ul>' + ''.join(summaries) + '</ul></div>'
    old, new = summary['new_probe']['baseline'], summary['new_probe']['candidate']
    if manifest['complete'] and old['samples'] == new['samples'] and old['samples'] and new['exact_route'] < old['exact_route']:
        html += '<div class="card" style="border-left:4px solid #c43f3f"><strong>新版在配对样本上退步，不宜启用。</strong>已恢复 baseline 默认；V2 仅保留为显式选择的实验提示。问题主要是多选不相关类别，不能以正确类别分数提高代替整条路由正确。</div>'
    html += '<p>只修改类别边界与带标签示例；模型 flash-v1.2、阈值 0.75／0.80、最近五轮历史和 6／5 分组不变。没有调用千问、RAG、工具或付费裁判。</p>'
    html += '<p>已知 8 例仅复测新提示，旧提示沿用此前保存结果。新增 12 例按相同输入做新旧提示配对实测。新增样本在本次候选提示编写前保存，但同由本代理拟定、未独立人工复核，不是严格独立盲测，不能据此宣称总体准确率。</p>'
    html += '<div class="card table"><table><tr><th>样本</th><th>分组</th><th>当前问题</th><th>预期角色</th><th>旧提示</th><th>新提示</th><th>新提示角色结果</th><th>新提示较高原始分数</th></tr>' + ''.join(lines) + '</table></div>'
    html += '<p>角色正确不等于类别全部正确；原始概率、清晰度、失败批次和不可用项都保存在 results.json。模型返回概率不等于经过校准的真实准确率。</p>'
    html += '<div class="card">HTTP 用量：' + escape(json.dumps(usage, ensure_ascii=False)) + '</div>'
    html += '<p><a href="results.json">全部结果</a> · <a href="cases.json">样本与预期</a> · <a href="candidate-questions.json">新提示全文</a> · <a href="baseline-questions.json">旧提示全文</a> · <a href="telemetry.json">调用状态和用量</a> · <a href="manifest.json">运行清单</a></p>'
    html += '<p>本地 .env 未改；沿用此前评测的原生 Decisions 参数覆盖。生产记忆、Redis、知识索引和旧评测结果均未写入。无自动重试、无重复运行挑最好成绩。</p></html>'
    (home / 'report.html').write_text(html, encoding='utf-8')


async def run(home):
    cases = build_cases()
    questions = {k: q for b in build_requests('离线提示快照', provider='decisions', prompt_version=ROUTING_PROMPT_VERSION) for k, q in b['questions'].items()}
    baseline = json.loads((DATA / 'baseline-questions.json').read_text(encoding='utf-8'))
    if baseline['clarity'] != questions['clarity']:
        raise ValueError('comparison must not silently change clarity definition')
    values = dict(dotenv_values(PROJECT / '.env', encoding='utf-8-sig', interpolate=False))
    settings = JevSettings.from_values(decisions_values(values))
    home.mkdir(parents=True, exist_ok=False)
    write_json(home / 'cases.json', cases)
    for name, q in [('baseline', baseline), ('candidate', questions)]:
        write_json(home / f'{name}-questions.json', q)
    schedule = []
    for i, c in enumerate(cases):
        variants = ('candidate',) if c['group'] == 'known_diagnostic' else ('baseline', 'candidate') if i % 2 else ('candidate', 'baseline')
        schedule.extend((v, c) for v in variants)
    manifest = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'prompt_version': ROUTING_PROMPT_VERSION, 'model': settings.model, 'provider': settings.provider,
        'thresholds': {'clarity': .75, 'relevance': .80}, 'batch_sizes': [6, 5], 'expected_samples': len(schedule),
        'max_http_attempts': len(schedule) * 2, 'scope': 'routing-only; no retries, no generation, no RAG',
        'question_sha256': {v: hashlib.sha256(json.dumps(q, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                            for v, q in [('baseline', baseline), ('candidate', questions)]}}
    write_json(home / 'manifest.json', manifest)
    sink, gate, rows = ObservationSink('fewshot-v2'), PaidCallGate(), []
    def save():
        usage = sink.totals('http_result')
        write_json(home / 'results.json', rows)
        write_json(home / 'telemetry.json', {'events': sink.events, 'usage': usage})
        write_json(home / 'manifest.json', manifest)
        render_report(home, manifest, rows, cases, usage)
    try:
        with tracing_context(enabled=False):
            async with httpx.AsyncClient(trust_env=settings.trust_env, follow_redirects=False,
                                        event_hooks=sink.http_hooks(gate)) as client:
                for variant, c in schedule:
                    sink.turn_id = variant + ':' + c['id']
                    frozen = FrozenQuestionsClient(client, baseline if variant == 'baseline' else questions)
                    recognizer = JevIntentRecognizer(settings, client=frozen)
                    out = await build_agent_graph(recognizer).ainvoke({'request': AgentRequest(c['message'], c['history'], sink.turn_id)})
                    row = {'id': c['id'], 'group': c['group'], 'variant': variant,
                        'route': out['routing_decision'], 'recognition': out['recognition']}
                    row.update(score(c, row['recognition'], row['route']))
                    rows.append(row)
                    print(variant, c['id'], row['route']['agent_types'], row['exact_route'], flush=True)
                    errors = json.dumps(row['recognition'].get('source_errors', {}))
                    if gate.blocked or any(f'HTTP {status}' in errors for status in (401, 402, 403, 429)):
                        manifest['stopped_reason'] = 'credential_or_rate_limit'
                        save()
                        return
                    save()
        expected = {(v, c['id']) for v, c in schedule}
        actual = {(r['variant'], r['id']) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError('sample coverage mismatch')
        manifest.update(complete=all(r['available'] for r in rows), accounted=True,
                        finished_at=datetime.now(timezone.utc).isoformat())
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required for paid native Decisions requests')
    asyncio.run(run(args.output))
