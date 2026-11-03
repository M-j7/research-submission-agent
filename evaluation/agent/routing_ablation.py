"""Prepare or execute a fixed eight-arm, routing-only factorial experiment."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
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
from submission_assistant.intent.jev import JevIntentRecognizer, JevSettings
from submission_assistant.routing import RoutingSettings, INTENT_ROUTING
from .ablation_design import ARMS, ARM_BY_NAME, DESIGN_VERSION, load_cases, make_plan, request_key, score_arm
from .ablation_metrics import arm_summary, factor_effects
from .observe import ObservationSink, PaidCallGate
from .prompt_probe import FrozenQuestionsClient, write_json
from .runtime import decisions_values

MODEL = 'pplx-decider-flash-v1.2'


def fingerprint(value):
    return request_key(value)


def validate_cases(cases):
    if not isinstance(cases, list) or not cases or len(cases) > 40:
        raise ValueError('expected 1 to 40 cases')
    for c in cases:
        if not isinstance(c, dict) or not isinstance(c.get('id'), str) or not c['id']:
            raise ValueError('case ID required')
        if c.get('review_status') in ('pending', 'disputed'):
            raise ValueError('unreviewed/disputed proposals are not evaluation gold')
        labels, roles, fallback = c.get('expected_intents'), c.get('expected_roles'), c.get('expected_fallback')
        if not isinstance(labels, list) or any(not isinstance(k, str) or k not in INTENT_ROUTING for k in labels) or len(set(labels)) != len(labels):
            raise ValueError('reviewed expected_intents required')
        if fallback not in (None, 'unclear_request', 'non_business') or bool(labels) != (fallback is None):
            raise ValueError('expected fallback required and valid')
        expected_roles = ['general'] if fallback else sorted({INTENT_ROUTING[k].value for k in labels})
        if not isinstance(roles, list) or sorted(roles) != expected_roles:
            raise ValueError('roles inconsistent with expected intents/fallback')
        if 'message' not in c or 'history' not in c:
            raise ValueError('message and history required')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('duplicate case IDs')


def prepare(home, *, cases=None, dataset_note=None):
    cases = load_cases() if cases is None else cases
    validate_cases(cases)
    plan = make_plan(cases, model=MODEL)
    home.mkdir(parents=True, exist_ok=False)
    write_json(home / 'cases.json', cases)
    write_json(home / 'plan.json', plan)
    manifest = {'version': DESIGN_VERSION, 'phase': 'prepared', 'complete': False,
        'created_at': datetime.now(timezone.utc).isoformat(), 'model': MODEL, 'provider': 'decisions',
        'cases': len(cases), 'logical_rows': len(plan['logical']), 'unique_queries': len(plan['queries']),
        'max_http_attempts': plan['max_http_attempts'], 'batch_sizes': [6, 5],
        'thresholds': RoutingSettings(.75, .8).to_dict(),
        'plan_sha256': fingerprint(plan), 'cases_sha256': fingerprint(cases),
        'scope': 'routing-only; no generation/RAG/tools/retries; no production default changes',
        'dataset_note': dataset_note or '前20例已参与此前开发，新增4例为构造的历史依赖探针；标签未经独立人工审定，长历史填充为压力诊断。',
        'caveat': '开发消融，非独立盲测；同一请求复用不等于独立重复试验。'}
    write_json(home / 'manifest.json', manifest)
    return cases, plan, manifest


def render_report(home, cases, manifest, rows, summary, effects, usage):
    def pct(value):
        return '—' if value is None else f'{value*100:.1f}%'
    def pp(value):
        return '—' if value is None else f'{value:+.2f} 个百分点'
    arm_lines = []
    for a in ARMS:
        s = summary[a.name]
        values = [a.name, str(int(a.boundary)), str(int(a.examples)), str(int(a.history)),
            f"{s['intent_exact']}/{s['available']}", f"{s['task_exact']}/{s['available']}", f"{s['visible_exact']}/{s['available']}",
            str(s['fp']), str(s['fn']), pct(s['precision']), pct(s['recall']), pct(s['f1']),
            f"{s['available']}/{s['samples']}", str(s['logical_input_tokens'])]
        arm_lines.append('<tr>' + ''.join('<td>'+escape(v)+'</td>' for v in values) + '</tr>')
    effect_lines = []
    names = {'boundary': '边界', 'examples': '示例', 'history': '真实历史'}
    for f, value in effects['intent']['main'].items():
        effect_lines.append(f"<li>{names[f]}：{pp(value['delta_pp'])}；有效配对差值 {value['paired_contrasts']} 个</li>")
    interactions = []
    for f, value in effects['intent']['interactions'].items():
        interactions.append('<li>' + '×'.join(names[n] for n in f.split(':')) + '：' + pp(value['delta_pp']) + '</li>')
    by = {(r['case_id'], r['arm']): r for r in rows}
    case_lines = []
    for c in cases:
        cells = [escape(c['id']), escape(c['message']), escape(c.get('history_kind', 'none'))]
        for a in ARMS:
            row = by.get((c['id'], a.name))
            if row is None:
                text, color = '未执行', '#6c7785'
            elif not row['available']:
                text, color = '不可用', '#6c7785'
            else:
                route = row['route']
                text = ('✓ ' if row['intent_exact'] else '✗ ') + ', '.join(route['agent_types'])
                if route['fallback_reason']:
                    text += ' / ' + route['fallback_reason']
                text += f"；误选{len(row['fp'])}/漏选{len(row['fn'])}"
                if row['reused']:
                    text += '〔复用〕'
                color = '#187153' if row['intent_exact'] else '#a54829'
            cells.append(f'<span style="color:{color}">'+escape(text)+'</span>')
        case_lines.append('<tr>' + ''.join('<td>'+v+'</td>' for v in cells) + '</tr>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>路由 B/E/H 八组消融</title>
<style>body{font:14px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1800px;margin:30px auto;padding:0 20px;background:#f5f7fb;color:#28374b}h1{font-size:27px}.card{background:white;border:1px solid #dde5ef;border-radius:10px;padding:18px;margin:18px 0}.wide{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:9px;border-bottom:1px solid #e0e6ef;text-align:left;vertical-align:top;min-width:65px}th{background:#edf2f8}a{color:#2760a4}</style><h1>路由：边界／示例／历史，八组完整消融</h1>'''
    html += '<div class="card"><strong>' + ('调用及覆盖核验完成' if manifest['complete'] else '尚未全部完成') + '</strong><p>'
    html += f"{manifest['cases']} 个开发样本 × 8 组 = {manifest['logical_rows']} 条逻辑结果；最多 {manifest['unique_queries']} 个独立请求组合、{manifest['max_http_attempts']} 次 HTTP。"
    html += '</p>相同输入与提示去重，复用行标明来源；它们不是独立重复试验。本路由阶段无千问生成、RAG、工具或自动重试。线上默认仍为 baseline。</div>'
    html += '<p>B=显式扩展边界；E=同一组固定示例；H=真实最近五轮完整问答。B关闭仍有简短类别定义，H关闭不删除示例里的虚构历史。公共规则、类别名称、模型、阈值、清晰度定义与请求分组固定。</p>'
    html += '<div class="card wide"><table><tr><th>组</th><th>B</th><th>E</th><th>H</th><th>类别完全正确</th><th>Agent分配正确</th><th>可见信息下合理处理</th><th>误选标签</th><th>漏选标签</th><th>Precision</th><th>Recall</th><th>F1</th><th>可用/已记录</th><th>逻辑输入token</th></tr>' + ''.join(arm_lines) + '</table></div>'
    html += '<p>“类别完全正确”要求业务标签集合及兜底原因均正确；“Agent分配正确”只要求目标角色及兜底原因正确，同角色内选错类别也可能通过，因此两项分开统计。“可见信息下合理处理”允许标记为dependent的历史依赖样本在H关闭时澄清。因此缺历史时合理澄清不等于分类能力差。P/R/F1仅按C01–C09业务标签计算，非业务和澄清另按路由结果评估。不可用项不当作零分样本隐藏。</p>'
    html += '<div class="card"><h2>平均边际影响：类别完全正确</h2><ul>' + ''.join(effect_lines) + '</ul><h3>交互差值</h3><ul>' + ''.join(interactions) + '</ul><p>各主效应来自其余开关相同时的四种配对差值平均，交互为差值之差。这是本批样本的描述性结果，不是总体因果比例，不能相加成100%。Agent分配及可见信息处理的独立效应另存于effects.json。</p></div>'
    html += '<div class="card wide"><table><tr><th>样本</th><th>当前问题</th><th>历史类型</th>' + ''.join('<th>'+a.name+'</th>' for a in ARMS) + '</tr>' + ''.join(case_lines) + '</table></div>'
    html += '<p>' + escape(manifest.get('dataset_note', '开发样本，未独立人工审定。')) + '只有一次测量，不据此宣称统计显著、泛化准确率或自动更换线上配置。</p>'
    html += '<div class="card">实际去重后HTTP用量：' + escape(json.dumps(usage, ensure_ascii=False)) + '<p>表内逻辑token为每行若单独请求的输入用量之和，复用可能重复归属；不能与实际计费总量直接相加。单位是token，不是请求次数或金额。</p></div>'
    html += '<p><a href="summary.json">分组指标</a> · <a href="effects.json">主效应与交互</a> · <a href="results.json">全部逻辑结果</a> · <a href="query-results.json">真实查询结果</a> · <a href="plan.json">冻结提示及去重计划</a> · <a href="cases.json">样本及预期</a> · <a href="telemetry.json">实际HTTP记录</a> · <a href="manifest.json">运行清单</a></p></html>'
    (home / 'report.html').write_text(html, encoding='utf-8')


async def run(home, *, live=False, cases_path=None, dataset_note=None):
    home = Path(home)
    if home.exists():
        manifest = json.loads((home / 'manifest.json').read_text(encoding='utf-8'))
        if not live or manifest.get('phase') != 'prepared' or (home / 'query-results.json').exists():
            raise FileExistsError('已有执行结果或输出目录，拒绝覆盖与自动重跑。')
        cases = json.loads((home / 'cases.json').read_text(encoding='utf-8'))
        plan = json.loads((home / 'plan.json').read_text(encoding='utf-8'))
    else:
        supplied = json.loads(Path(cases_path).read_text(encoding='utf-8')) if cases_path else None
        cases, plan, manifest = prepare(home, cases=supplied, dataset_note=dataset_note)
    if fingerprint(plan) != manifest['plan_sha256'] or fingerprint(cases) != manifest['cases_sha256']:
        raise ValueError('frozen plan or cases changed')
    validate_cases(cases)
    if plan['arms'] != [asdict(a) for a in ARMS] or plan['model'] != MODEL:
        raise ValueError('frozen factors/model changed')
    if not live:
        print(json.dumps({k: manifest[k] for k in ('cases', 'logical_rows', 'unique_queries', 'max_http_attempts')}, ensure_ascii=False))
        return manifest
    values = dict(dotenv_values(PROJECT / '.env', encoding='utf-8-sig', interpolate=False))
    settings = JevSettings.from_values(decisions_values(values))
    if settings.model != plan['model']:
        raise ValueError('provider model does not match frozen plan')
    sink, gate, cache = ObservationSink(DESIGN_VERSION), PaidCallGate(), {}
    case_by_id = {c['id']: c for c in cases}
    manifest.update(phase='running', started_at=datetime.now(timezone.utc).isoformat())

    def save():
        rows = []
        for item in plan['logical']:
            q = cache.get(item['query_key'])
            if q is None:
                continue
            c, arm = case_by_id[item['case_id']], ARM_BY_NAME[item['arm']]
            reused = (item['case_id'], item['arm']) != (q['source_case_id'], q['source_arm'])
            row = {**item, 'group': c.get('group'), 'history_kind': c.get('history_kind'), 'reused': reused,
                'source_case_id': q['source_case_id'], 'source_arm': q['source_arm'],
                'route': q['route'], 'recognition': q['recognition'], 'latency_ms': q['latency_ms'],
                'input_tokens': (q['recognition'].get('judgments') or {}).get('usage', {}).get('input_tokens')}
            row.update(score_arm(c, arm, q['recognition'], q['route']))
            rows.append(row)
        usage = sink.totals('http_result')
        summary = arm_summary(rows)
        effects = {'intent': factor_effects(rows, metric='intent_exact'),
                   'task': factor_effects(rows), 'visible': factor_effects(rows, metric='visible_exact')}
        write_json(home / 'query-results.json', cache)
        write_json(home / 'results.json', rows)
        write_json(home / 'summary.json', summary)
        write_json(home / 'effects.json', effects)
        write_json(home / 'telemetry.json', {'events': sink.events, 'usage': usage})
        write_json(home / 'manifest.json', manifest)
        render_report(home, cases, manifest, rows, summary, effects, usage)
        return rows
    try:
        with tracing_context(enabled=False):
            async with httpx.AsyncClient(trust_env=settings.trust_env, follow_redirects=False,
                                        event_hooks=sink.http_hooks(gate)) as client:
                for index, (key, planned) in enumerate(plan['queries'].items(), 1):
                    bodies = planned['bodies']
                    source_case, source_arm = planned['first_case_id'], planned['first_arm']
                    sink.turn_id = source_arm + ':' + source_case
                    questions = {k: v for body in bodies for k, v in body['questions'].items()}
                    recognizer = JevIntentRecognizer(settings, client=FrozenQuestionsClient(client, questions))
                    state = bodies[0]['state']
                    out = await build_agent_graph(recognizer, settings=RoutingSettings(.75, .8)).ainvoke({
                        'request': AgentRequest(state['current_message'], state['history'], sink.turn_id)})
                    cache[key] = {'source_case_id': source_case, 'source_arm': source_arm,
                        'recognition': out['recognition'], 'route': out['routing_decision'],
                        'latency_ms': (out['recognition'].get('judgments') or {}).get('latency_ms', 0.)}
                    print(f"{index}/{len(plan['queries'])}", source_arm, source_case, out['routing_decision']['agent_types'], flush=True)
                    errors = json.dumps(out['recognition'].get('source_errors', {}))
                    if gate.blocked or any(f'HTTP {status}' in errors for status in (401, 402, 403, 429)):
                        manifest.update(phase='stopped', stopped_reason='credential_or_rate_limit')
                        return
                    save()
        rows = save()
        expected = {(x['case_id'], x['arm']) for x in plan['logical']}
        if len(rows) != len(expected) or {(r['case_id'], r['arm']) for r in rows} != expected:
            raise ValueError('logical coverage mismatch')
        if len(cache) != len(plan['queries']) or sum(e['type'] == 'http_call' for e in sink.events) > plan['max_http_attempts']:
            raise ValueError('unique-query accounting mismatch')
        manifest.update(phase='finished', accounted=True, complete=all(r['available'] for r in rows),
                        finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        manifest.update(phase='stopped', stopped_reason=type(exc).__name__)
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true', help='Freeze inputs and print the call cap; no config or network')
    modes.add_argument('--live', action='store_true', help='Execute only the frozen unique queries; paid API calls')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', type=Path, help='Reviewed JSON cases; only accepted during --prepare')
    parser.add_argument('--dataset-note', help='Dataset provenance and label limitations')
    args = parser.parse_args()
    if args.live and (args.cases or args.dataset_note):
        parser.error('--live consumes only the frozen prepared directory')
    asyncio.run(run(args.output, live=args.live, cases_path=args.cases, dataset_note=args.dataset_note))
