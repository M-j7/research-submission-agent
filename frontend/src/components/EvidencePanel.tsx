import type { Turn } from '../types'
import { ExecutionProgress, roles } from './ExecutionProgress'
const reasons:Record<string,string>={unclear_request:'需求不明确，等待澄清',non_business:'非业务交流',recognition_failed:'识别服务异常',uncertain_intent:'分类未确定',conflicting_judgments:'分类存在冲突'}
export function EvidencePanel({turn}:{turn?:Turn}) {
  const evidence=[...new Map((turn?.agent_results.flatMap(r=>r.evidence)||[]).map(e=>[(e.source_url||'')+e.title+e.source_locator,e])).values()]
  return <><header className="insight-heading"><span>运行观察</span><small>LIVE CONTEXT</small></header>
    <section className="insight-section"><div className="section-label">意图与角色 <span>01</span></div>
      {turn?.routing?<><div className="role-chips">{turn.routing.agent_types.map(r=><span key={r}>{roles[r]||r}</span>)}</div>
        <p className="quiet">{turn.routing.matched_intents.join(' · ')||'无业务标签'}</p>{turn.routing.fallback_reason&&<p className="quiet">{reasons[turn.routing.fallback_reason]||'已进入反馈流程'}</p>}</>:
        <p className="empty-note">发送问题后，在这里查看<br/>实际路由与参与角色。</p>}</section>
    <section className="insight-section"><div className="section-label">执行状态 <span>02</span></div>
      {turn?<ExecutionProgress turn={turn}/>:
        <div className="idle-state"><span className="idle-orbit">◎</span><span>等待新的咨询</span><small>仅显示真实阶段，不展示内部思维链</small></div>}</section>
    <section className="insight-section evidence-section"><div className="section-label">参考证据 <span>{evidence.length.toString().padStart(2,'0')}</span></div>
      {!evidence.length?<div className="empty-evidence"><span>▤</span><strong>暂无证据</strong><p>检索获得的参考资料将在这里展示。<br/>没有证据时，不生成虚构引用。</p></div>:
      evidence.map((e,i)=><article className="evidence-card" key={i}><small>来源 {String(i+1).padStart(2,'0')}</small><h3>{e.source_url?<a href={e.source_url} target="_blank" rel="noopener noreferrer">{e.title} ↗</a>:e.title}</h3>
        {(e.page||e.source_locator)&&<p className="source-location">{e.page?`第 ${e.page} 页 · `:''}{e.source_locator}</p>}<p>{e.text}</p></article>)}</section>
    <footer className="insight-footer">基于资料作答，而非凭空推断。</footer></>
}
