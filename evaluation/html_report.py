"""Export saved evaluation metrics as a small standalone HTML, without model calls."""
import argparse
from collections import Counter
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def value_cell(value):
    return f'{value:.4f}' if type(value) in (float,int) and math.isfinite(value) else '—'


def usage_cell(usage, key):
    value=usage.get(key)
    if value is None:
        return '—（未知）'
    return str(value)+('（完整）' if usage.get('complete') else '（部分）')


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def load_results(agent_dirs=()):
    from evaluation.rag.score import load_run, score_rankings
    from evaluation.rag.ragas_score import load_scores, summarize as generation_summary
    data = {'retrieval':[], 'generation':[], 'agent':[], 'warnings':[]}
    for home in sorted((ROOT/'evaluation/rag/runs').glob('*')):
        if not (home/'manifest.json').exists():
            continue
        try:
            manifest, queries, qrels, rows = load_run(home)
            saved = json.loads((home/'summary.json').read_text(encoding='utf-8'))
            for baseline, views in saved['baselines'].items():
                for language, view in views.items():
                    selected = [r for r in rows if r['baseline']==baseline and (language=='all' or queries[r['query_id']]['language']==language)]
                    # Saved rounded metrics are checked against raw rank/qrel calculations.
                    ranks = {r['query_id']:r['parent_ids'] for r in selected}
                    fresh = score_rankings({k:qrels[k] for k in ranks},ranks)['aggregate']
                    if any(abs(fresh[k]-v)>.00011 for k,v in view['metrics'].items()):
                        raise ValueError('retrieval summary mismatch')
                    data['retrieval'].append({'run':home.name,'baseline':baseline,'language':language,
                        'count':view['query_count'],'status_counts':view['status_counts'],'metrics':view['metrics'],
                        'semantic_validation':manifest.get('semantic_validation','jev')})
        except Exception as exc:
            data['warnings'].append(f'检索来源 {home.name} 未通过校验：{type(exc).__name__}；未纳入指标。')
    for home in sorted((ROOT/'evaluation/rag/generation-runs').glob('*')):
        try:
            manifest, answers, rows = load_scores(home,home/'ragas')
            summary = generation_summary(rows)
            reasons = Counter(str(m.get('reason')) for r in rows for m in r['metrics'].values() if m['status']!='ok')
            data['generation'].append({'run':home.name,'summary':summary,'count':len(answers),
                'unavailable_reasons':dict(reasons),'limitations':manifest.get('limitations',[])})
        except Exception as exc:
            if home.is_dir():
                data['warnings'].append(f'生成来源 {home.name} 未完成或校验失败：{type(exc).__name__}；不与正式结果混算。')
    from evaluation.agent.run import load_agent_run
    for home in agent_dirs:
        home = Path(home)
        try:
            manifest,cases,executions,scores = load_agent_run(home)
            summary = json.loads((home/'summary.json').read_text(encoding='utf-8'))
            data['agent'].append({'run':home.name,'manifest':{'complete':manifest['complete'],'mode':manifest['mode'],
                                                           'runtime_config':manifest.get('runtime_config')},
                                  'cases':cases,'executions':executions,'scores':scores,'summary':summary,
                                  'environment_note':(home/'environment-note.zh.md').read_text(encoding='utf-8')
                                      if (home/'environment-note.zh.md').is_file() else None})
        except Exception as exc:
            data['warnings'].append(f'Agent来源 {home.name} 未通过校验：{type(exc).__name__}。')
    return data


def render_report(data):
    out = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>RAG 与 Agent 评测结果</title><style>body{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#213047;background:#f3f6fa;margin:0}main{max-width:1240px;margin:auto;padding:32px 24px}h1{margin-bottom:0;color:#10243b}h2{border-left:4px solid #258490;padding-left:12px;margin-top:36px}h3{margin-top:24px}.sub{color:#607187}section{background:white;padding:8px 24px 24px;border:1px solid #e1e7ef;border-radius:12px;margin-top:22px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}th,td{padding:10px 12px;border-bottom:1px solid #e5eaf1;text-align:left;vertical-align:top}th{background:#edf4f8;white-space:nowrap}tr:nth-child(even){background:#fafcfd}.note{background:#fff4df;padding:12px 16px;border-radius:8px}li{margin:4px 0}small{color:#607187}</style><main>',
        '<h1>RAG 与 Agent 评测结果</h1><p class="sub">静态结果汇总 · '+datetime.now().strftime('%Y-%m-%d %H:%M')+'</p>',
        '<p class="note">分数与覆盖率一起看：真实零分参与均值，缺失评分显示“—”，不补零。不同运行、语言与实验条件分开列示。</p>',
        '<section><h2>RAG 检索</h2><p>Recall 衡量相关证据找回比例，NDCG/MRR 衡量排序，HitRate 衡量是否至少命中一个相关证据。下面均为 @5。</p>']
    retrieval_rows = []
    for r in data['retrieval']:
        m = r['metrics']
        retrieval_rows.append([r['run'],r['baseline'],r['language'],r['count'],*[value_cell(m.get(k+'@5')) for k in ('Recall','NDCG','MRR','HitRate','Precision')],str(r['status_counts'])])
    out.append(table(['运行','方案','语言','样本','Recall@5','NDCG@5','MRR@5','HitRate@5','Precision@5','运行状态'],retrieval_rows))
    out.append('<p class="note">旧 pilot 的 B4 全部因校验服务失败而降级，不是真正 Multi-Query 成绩；multiquery-no-jev 的 B4 为用户批准的未校验实验。双语问题为15组配对问题，qrels 为非穷尽草稿标签。</p></section>')
    out.append('<section><h2>RAG 生成</h2><p>每格：有效评分均值（有效数/预期数）。Faithfulness＝证据忠实度；Correctness＝参考一致性；其余用于相关性与上下文诊断。</p>')
    for r in data['generation']:
        out.append('<h3>'+escape(r['run'])+' · '+str(r['count'])+' 个回答样本</h3>')
        rows = []
        for config, views in r['summary'].items():
            for lang, metrics in views.items():
                rows.append([config,lang,*[value_cell(m['mean'])+f"（{m['valid']}/{m['total']}）" for m in metrics.values()]])
        out.append(table(['方案','语言','Faithfulness','Correctness','Relevancy','ContextPrecision','ContextRecall'],rows))
        out.append('<p>不可用原因：'+escape(json.dumps(r['unavailable_reasons'],ensure_ascii=False))+'</p>')
    out.append('<p class="note">草稿参考答案，同一Qwen模型生成与裁判，存在偏差；有效均分不等于完整系统准确率。正式生成评测60个回答、300个预期评分，229个有效、71个不可用；历史烟测另列，不混入正式均值。</p></section>')
    out.append('<section><h2>Agent：端到端、轨迹与表现</h2>')
    if not data['agent']:
        out.append('<p>尚无通过覆盖校验的 Agent 运行结果。</p>')
    for a in data['agent']:
        out.append('<h3>'+escape(a['run'])+' · '+('已完成' if a['manifest']['complete'] else '部分结果／已暂停')+'</h3>')
        if a.get('environment_note'):
            out.append('<p class="note">'+escape(a['environment_note']).replace('\n','<br>')+'</p>')
        config=a['manifest'].get('runtime_config')
        if config:
            out.append('<p class="note">输入预算 '+escape(str(config['input_budget_tokens']))+
                       ' token；提前压缩水位 '+escape(str(config['compression_trigger_tokens']))+
                       ' token。本地估算（公开分词器＋余量），不是服务端精确用量；与旧字节预算条件不同。计数方式：'+
                       escape(config['counting_mode'])+'。</p>')
        blocked=Counter((e.get('component','unknown'),e.get('status_code')) for r in a['executions']
                        for e in r.get('events',[]) if e['type']=='http_result' and e.get('status_code') in (401,402,403))
        if blocked:
            text='；'.join(f'{component}: HTTP {code} × {count}' for (component,code),count in blocked.items())
            out.append('<p class="note">运行被服务权限／额度错误拦截：'+escape(text)+'。未启动正式30题评测；本次业务失败不能解读为业务能力得分。403本身不能区分密钥、权限、额度或网络策略问题，须核查供应商控制台。</p>')
        rows = []
        for group in ('business','fault'):
            s=a['summary'][group]
            rows.append([group,s['expected'],s['executed'],s['pass'],s['fail'],s['unavailable'],s['not_executed'],
                f"{s['pass']}/{s['expected']}",f"{s['pass']}/{s['valid']}",
                value_cell(s['latency_p50_ms']),value_cell(s['latency_p95_ms'])])
        out.append(table(['组','预期','已执行','通过','失败','不可用','未执行','已验证/预期','通过/有效','P50 ms','P95 ms'],rows))
        counts=Counter()
        for s in a['scores']:
            for r in s['rules']:
                counts[(r['id'],r['status'])]+=1
        out.append('<h3>轨迹规则</h3>')
        out.append(table(['检查项','通过','失败'],[[k,counts[(k,'pass')],counts[(k,'fail')]] for k in sorted({k for k,_ in counts})]))
        out.append('<h3>逐任务指标与退出原因</h3>')
        es={e['case_id']:e for e in a['executions']}
        out.append(table(['任务','组','执行状态','独立判分','耗时 ms','模型调用','工具尝试','实际输入 token','实际输出 token','退出／失败原因'],[
            [s['case_id'],s['group'],es[s['case_id']]['execution_status'],s['business_outcome'],value_cell(s['performance'].get('latency_ms')),
             s['performance'].get('model_calls'),s['performance'].get('tool_attempts'),
             usage_cell(s['performance'].get('usage',{}).get('agent',{}),'input_tokens'),
             usage_cell(s['performance'].get('usage',{}).get('agent',{}),'output_tokens'),
             '; '.join([es[s['case_id']]['stop_reason'],*[r['id']+': '+r['reason'] for r in s['rules'] if r['status']=='fail'],*s['semantic'].get('errors',[])])]
            for s in a['scores']]))
        out.append('<ul>'+''.join('<li>'+escape(s)+'</li>' for s in a['summary']['limitations'])+'</ul>')
    out.append('</section><section><h2>来源检查与说明</h2><ul>'+''.join('<li>'+escape(w)+'</li>' for w in data['warnings'])+'</ul>')
    out.append('<p>原始JSON/JSONL、逐题答案、证据及 EvalScope 报告保留在 evaluation/rag 与 evaluation/agent/runs；本报告只读汇总，不触发额外模型调用。无价格快照时不估算金额，HTTP与模型用量有重叠，不能相加。</p></section></main></html>')
    return '\n'.join(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agent',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,default=ROOT/'evaluation/reports/agent-rag-overview.html')
    args=p.parse_args()
    data=load_results(args.agent)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(render_report(data),encoding='utf-8')
    print(args.output.resolve())


if __name__=='__main__':
    main()
