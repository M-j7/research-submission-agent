import type { ConversationSummary } from '../types'
interface Props { conversations:ConversationSummary[]; selectedId:string; onSelect:(id:string)=>void; onCreate:()=>void; disabled:boolean }
export function ConversationList({conversations,selectedId,onSelect,onCreate,disabled}:Props) {
  return <><div className="sidebar-brand"><span className="brand-icon">投</span><div><strong>投稿咨询助手</strong><small>RESEARCH WORKSPACE</small></div></div>
    <button className="new-chat" onClick={onCreate} disabled={disabled}><span>＋</span> 新建会话 <kbd>N</kbd></button>
    <div className="section-label">我的会话 <span>{conversations.length.toString().padStart(2,'0')}</span></div>
    <nav className="conversation-list" aria-label="历史会话">{conversations.length===0?<p className="quiet">开始一段新的投稿咨询</p>:
      conversations.map(c=><button key={c.id} className={'conversation '+(selectedId===c.id?'selected':'')} onClick={()=>onSelect(c.id)} aria-current={selectedId===c.id?'page':undefined}>
        <span className="chat-symbol">◷</span><span>{c.title}</span>{selectedId===c.id&&<i/>}</button>)}</nav>
    <div className="sidebar-bottom"><span className="local-dot"/><div><strong>本地工作空间</strong><small>会话保存在这台设备</small></div><span>↗</span></div>
  </>
}
