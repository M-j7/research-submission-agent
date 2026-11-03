import { useEffect, useRef } from 'react'
import type { Turn } from '../types'
import { SafeMarkdown } from './SafeMarkdown'
import { ExecutionProgress } from './ExecutionProgress'

interface Props {turns:Turn[];draft:string;disabled:boolean;loading:boolean;uncertain:boolean;error:string;
  onDraft:(value:string)=>void;onSend:()=>void;onRetry:()=>void}
const states:Record<string,string>={pending:'准备中',running:'处理中',incomplete:'部分完成',failed:'执行失败',cancelled:'已取消',interrupted:'运行中断'}
export function ChatPanel({turns,draft,disabled,loading,uncertain,error,onDraft,onSend,onRetry}:Props) {
  const bottom=useRef<HTMLDivElement>(null)
  const following=useRef(true)
  useEffect(()=>{
    const trackPosition=()=>{
      following.current=document.documentElement.scrollHeight-window.innerHeight-window.scrollY<=96
    }
    window.addEventListener('scroll',trackPosition,{passive:true})
    return ()=>window.removeEventListener('scroll',trackPosition)
  },[])
  // Follow layout/action changes, not every heartbeat's counters. Respect manual history reading.
  const content=JSON.stringify(turns.map(t=>[t.turn_id,t.message,t.answer,t.state,t.error_code,
    t.agent_results.flatMap(r=>r.unresolved),
    t.state==='running'||t.state==='pending'?t.status.filter(s=>s.phase||s.agent_type)
      .map(s=>[s.agent_type,s.phase,s.execution_status,s.tools]):[]]))
  useEffect(()=>{
    if(following.current)bottom.current?.scrollIntoView?.({behavior:'auto',block:'end'})
  },[content,loading,error])
  const canSend=!disabled&&!uncertain&&Boolean(draft.trim())&&draft.length<=16000
  return <>
    <div className="chat-scroll" aria-live="polite">
      {loading?<div className="center-note">正在恢复会话…</div>:turns.length===0?<section className="welcome">
        <div className="welcome-mark">✧</div><span className="eyebrow">YOUR SUBMISSION COPILOT</span>
        <h1>让投稿问题，<br/><span>有据可循。</span></h1>
        <p>从稿件规范到出版流程，与专业 Agent 一起梳理。<br/>回答、参考证据与执行状态，在同一个工作空间。</p>
        <div className="suggestions">{[['01','稿件准备','IEEE Access 的摘要和关键词有哪些要求？'],['02','投稿流程','如何查询稿件的审稿状态？'],['03','费用政策','IEEE Access 的发表费用和减免政策是什么？']].map(([n,title,q])=>
          <button key={n} onClick={()=>onDraft(q)} disabled={disabled}><small>{n} <span>↗</span></small><strong>{title}</strong><span>{q}</span></button>)}</div>
        <div className="welcome-foot"><i/> 专业角色协作 <span>·</span> 来源可追溯 <span>·</span> 会话可恢复</div>
      </section>:turns.map(t=><section className="turn" key={t.turn_id}>
        <div className="user-message"><div className="message-label">你 <span>USER</span></div><div className="user-bubble">{t.message}</div></div>
        <div className="assistant-message"><div className="assistant-avatar">✧</div><div className="assistant-body"><div className="message-label">投稿咨询助手 <span>ASSISTANT</span></div>
          {(t.state==='pending'||t.state==='running'||(!t.answer&&t.state!=='completed'))&&<ExecutionProgress turn={t} compact/>}
          {t.answer?<SafeMarkdown text={t.answer}/>:t.state!=='pending'&&t.state!=='running'&&<p className="pending-text">本轮未生成最终答复。</p>}
          {t.answer&&t.state!=='completed'&&<div className={'turn-state '+t.state}>{states[t.state]||t.state}</div>}
          {t.agent_results.some(r=>r.unresolved.length>0)&&<div className="unresolved"><strong>未完成事项</strong><ul>{[...new Set(t.agent_results.flatMap(r=>r.unresolved))].map(x=><li key={x}>{x}</li>)}</ul></div>}
          {t.error_code&&<p className="quiet">{t.error_code==='service_interrupted'?'服务曾中断，未自动重新执行。':'本轮执行未完成，请检查模型服务或配置后发起新一轮。'}</p>}
        </div></div>
      </section>)}
    </div>
    <div className="composer-area">{error&&<div className="error-banner" role="alert"><span>{error}</span><button onClick={onRetry}>恢复状态</button></div>}
      <form className="composer" onSubmit={e=>{e.preventDefault();if(canSend)onSend()}}>
        <textarea aria-label="输入问题" placeholder="输入你的投稿问题，也可以继续追问…" value={draft} disabled={disabled} rows={3} maxLength={16000}
          onChange={e=>onDraft(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing&&e.keyCode!==229){e.preventDefault();if(canSend)onSend()}}}/>
        <div className="composer-bottom"><span><i/> {disabled?'正在处理 / 请稍候':uncertain?'等待确认 / 不会重复执行':'随时提问 / 支持连续对话'}</span><div><small>{draft.length.toLocaleString()} / 16,000</small><button aria-label="发送问题" title="发送问题" type="submit" disabled={!canSend}>↑</button></div></div>
      </form><p className="disclaimer">Enter 发送 · Shift + Enter 换行 <span>回答仅供参考，请以期刊官方最新政策为准。</span></p>
    </div>
    <div ref={bottom} aria-hidden="true"/>
  </>
}
