import { useState } from 'react'
import { useWorkbench } from './useWorkbench'
import { ConversationList } from './components/ConversationList'
import { ChatPanel } from './components/ChatPanel'
import { EvidencePanel } from './components/EvidencePanel'

export default function App() {
  const w=useWorkbench()
  const [showLeft,setShowLeft]=useState(false),[showRight,setShowRight]=useState(false)
  const current=w.conversations.find(c=>c.id===w.selectedId)
  return <div className={'workbench '+(showLeft?'show-left ':'')+(showRight?'show-right':'')}>
    {(showLeft||showRight)&&<button className="drawer-scrim" aria-label="关闭侧栏" onClick={()=>{setShowLeft(false);setShowRight(false)}}/>}
    <aside className="sidebar"><ConversationList conversations={w.conversations} selectedId={w.selectedId} disabled={!w.health||w.modeChanged}
      onCreate={()=>{void w.create();setShowLeft(false)}} onSelect={id=>{void w.select(id);setShowLeft(false)}}/></aside>
    <main className="main"><header className="topbar"><div className="breadcrumb"><button className="mobile-menu" aria-label="展开会话" onClick={()=>setShowLeft(true)}>☰</button><span>工作台</span><b>/</b><strong>{current?.title||'新会话'}</strong></div>
      <div className="topbar-right"><span className={'mode-badge '+w.health?.mode}>{w.health?.mode==='demo'?'离线演示':w.health?.mode==='live'?'真实模型':'连接中'}</span><button className="insight-toggle" aria-label="展开运行观察" onClick={()=>setShowRight(v=>!v)}>◫ <span>运行观察</span></button></div></header>
      {w.health?.mode==='demo'&&<div className="demo-banner">离线演示模式 · 不调用外部模型，演示回答不代表真实投稿政策</div>}
      <ChatPanel key={w.selectedId} turns={w.turns} draft={w.draft} loading={w.loading} disabled={!w.selectedId||Boolean(w.activeTurn)||w.loading||w.modeChanged} uncertain={w.uncertain} error={w.error}
        onDraft={w.setDraft} onSend={()=>void w.send()} onRetry={()=>void w.retry()}/>
      {!w.selectedId&&!w.loading&&w.health&&<div className="start-overlay"><button className="primary-button" onClick={()=>void w.create()}>＋ 新建会话，开始咨询</button></div>}
    </main>
    <aside className="insight"><button className="close-insight" aria-label="关闭运行观察" onClick={()=>setShowRight(false)}>×</button><EvidencePanel turn={w.turns.at(-1)}/></aside>
  </div>
}
