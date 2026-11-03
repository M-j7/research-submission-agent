"""Null-safe business, trajectory and performance summaries with explicit denominators."""
import math
from .rules import check_rules


def score_case(case, execution, semantic):
    rules = check_rules(case, execution)
    semantic = dict(semantic)
    checks = {}
    for item in case['semantic_checks']:
        p = semantic.get('checks', {}).get(item['id'], {}).get('probability')
        valid = type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
        checks[item['id']] = {'probability': p if valid else None,
                             'status': 'pass' if valid and p >= .8 else 'fail' if valid and p <= .2 else 'unavailable'}
    semantic['checks'] = checks
    states = [r['status'] for r in rules] + [v['status'] for v in checks.values()]
    outcome = 'fail' if 'fail' in states else 'unavailable' if 'unavailable' in states else 'pass'
    if execution.get('execution_status') == 'not_executed':
        outcome = 'unavailable'
    events = execution.get('events', [])
    return {'case_id': case['case_id'], 'trial_id': execution['trial_id'], 'group': case['group'],
            'business_outcome': outcome, 'rules': rules, 'semantic': semantic, 'review_status': 'uncalibrated',
            'performance': {'latency_ms': execution.get('latency_ms'),
                'model_calls': sum(e['type'] == 'model_call' for e in events),
                'tool_attempts': sum(e['type'] == 'tool_call' for e in events),
                'steps': sum(r.get('status', {}).get('current_step', 0) for r in execution.get('runs', [])),
                'usage': execution.get('usage', {}), 'stop_reason': execution.get('stop_reason')}}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    x = (len(values)-1)*p
    lo = int(x)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(x-lo)


def summarize(cases, executions, scores):
    by_e = {r['case_id']: r for r in executions}
    by_s = {r['case_id']: r for r in scores}
    summary = {}
    for group in ('business', 'fault'):
        selected = [c for c in cases if c['group'] == group]
        passed = failed = unavailable = unexecuted = 0
        durations, rules = [], []
        for c in selected:
            e, s = by_e.get(c['case_id']), by_s.get(c['case_id'])
            if not e or e['execution_status'] == 'not_executed':
                unexecuted += 1
            else:
                if type(e.get('latency_ms')) in (int, float):
                    durations.append(e['latency_ms'])
            status = s['business_outcome'] if s else 'unavailable'
            passed += status == 'pass'
            failed += status == 'fail'
            unavailable += status == 'unavailable'
            if s:
                rules += s.get('rules', [])
        valid = passed + failed
        summary[group] = {'expected': len(selected), 'executed': len(selected)-unexecuted,
            'pass': passed, 'fail': failed, 'unavailable': unavailable, 'not_executed': unexecuted,
            'verified_ratio': passed/len(selected) if selected else None,
            'valid_pass_rate': passed/valid if valid else None, 'valid': valid,
            'trajectory_checks_passed': sum(r['status']=='pass' for r in rules), 'trajectory_checks_total': len(rules),
            'latency_p50_ms': percentile(durations, .5), 'latency_p95_ms': percentile(durations, .95)}
    summary['limitations'] = ['中文草稿任务和参考说明，未人工校准；Decisions实验阈值0.8/0.2。',
        '业务与确定性故障检查分开；故障成绩不代表LLM抗提示注入能力。',
        '小样本P50/P95只作描述；未运行6×3重复稳定性；不声称生产能力。',
        '冷态隔离本地状态；不控制供应商远端Prompt Cache；后台画像维护关闭。']
    return summary
