"""Small static report helpers; no application frontend or dashboard."""
import json


def render_markdown(summary, executions, scores):
    lines = ['# Agent 三层诊断评测', '', '草稿任务、机器判分，未完成人工校准。业务与确定性故障分开。', '',
             '| 分组 | 预期 | 已执行 | 通过 | 失败 | 不可用 | 已验证/预期 | 通过/有效 | P50 ms | P95 ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('business','fault'):
        s = summary[group]
        lines.append('| '+' | '.join(str(x) for x in [group,s['expected'],s['executed'],s['pass'],s['fail'],s['unavailable'],
            f"{s['pass']}/{s['expected']}",f"{s['pass']}/{s['valid']}",s['latency_p50_ms'],s['latency_p95_ms']])+' |')
    lines += ['', '## 限制', *['- '+s for s in summary['limitations']], '', '## 逐题结果', '']
    es = {r['case_id']:r for r in executions}
    for r in scores:
        lines += [f"### {r['case_id']}：{r['business_outcome']}",
                  f"执行：{es[r['case_id']]['execution_status']}；退出：{es[r['case_id']]['stop_reason']}"]
        lines += ['- '+v['id']+': '+v['reason'] for v in r['rules'] if v['status']=='fail']
        lines += ['- 裁判：'+s for s in r['semantic'].get('errors',[])]
    return '\n'.join(lines)+'\n'
