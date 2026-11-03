import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'
import { terminal } from './types'
import type { ConversationSummary, Health, SendBody, Turn } from './types'

interface Pending extends SendBody { conv: string }
function read<T>(key:string,fallback:T):T {
  try { return JSON.parse(localStorage.getItem(key) || 'null') ?? fallback } catch { return fallback }
}
function write(key:string,value:unknown) {
  try { localStorage.setItem(key,JSON.stringify(value)) } catch { /* storage may be disabled */ }
}
function message(error:unknown) {
  if((error as {code?:string})?.code==='service_mode_changed')return '服务运行模式已变化，请刷新页面确认模式后再发送。'
  const status=(error as {status?:number})?.status
  return status===429?'当前任务已满，请稍后再试。':status===409?'该会话正在运行，或轮次发生冲突，请恢复状态。':
    '请求未确认，请检查连接并恢复状态；不会自动重复调用模型。'
}

export function useWorkbench() {
  const [health,setHealth]=useState<Health|null>(null)
  const [conversations,setConversations]=useState<ConversationSummary[]>([])
  const [selectedId,setSelectedId]=useState('')
  const [turns,setTurns]=useState<Turn[]>([])
  const [draft,setDraft]=useState('')
  const [loading,setLoading]=useState(true)
  const [error,setError]=useState('')
  const [uncertain,setUncertain]=useState(false)
  const [modeChanged,setModeChanged]=useState(false)
  const selected=useRef(''), mode=useRef(''), busy=useRef(false)
  const selectionEpoch=useRef(0), modeBlocked=useRef(false)
  const pending=useRef<Pending[]>([])
  const mount=useRef(true)
  const key=()=>'submission-workbench:'+mode.current
  const remember=()=>write(key()+':pending',pending.current)
  const refreshList=useCallback(async()=>{ const list=await api.listConversations(); if(mount.current)setConversations(list) },[])
  const showError=useCallback((e:unknown)=>{
    if((e as {code?:string})?.code==='service_mode_changed'){modeBlocked.current=true;setModeChanged(true)}
    setError(message(e))
  },[])

  const select=useCallback(async(id:string)=>{
    const epoch=++selectionEpoch.current
    selected.current=id; setSelectedId(id); setTurns([]); setLoading(true); setError(''); setUncertain(false)
    write(key()+':selected',id)
    try {
      const detail=await api.getConversation(id)
      if(selected.current!==id || epoch!==selectionEpoch.current || !mount.current)return
      setTurns(detail.turns)
      const lost=pending.current.find(p=>p.conv===id && !detail.turns.some(t=>t.turn_id===p.turn_id))
      setUncertain(Boolean(lost))
      if(lost)setError('上次发送结果尚未确认，请点击恢复状态。')
      pending.current=pending.current.filter(p=>p.conv!==id || !detail.turns.some(t=>t.turn_id===p.turn_id && terminal(t)))
      remember()
    } catch(e) { if(selected.current===id && epoch===selectionEpoch.current)showError(e) }
    finally { if(selected.current===id && epoch===selectionEpoch.current && mount.current)setLoading(false) }
  },[showError])

  useEffect(()=>{
    mount.current=true
    let cancelled=false
    void (async()=>{
      try {
        const h=await api.health(); if(cancelled)return
        setHealth(h); mode.current=h.mode; pending.current=read<Pending[]>(key()+':pending',[])
        const list=await api.listConversations(); if(cancelled)return
        setConversations(list)
        const last=read<string>(key()+':selected','')
        if(list.length) await select(list.some(c=>c.id===last)?last:list[0].id)
        else setLoading(false)
      } catch { if(!cancelled){ setError('无法连接本地后端，请确认服务已启动。');setLoading(false) } }
    })()
    return ()=>{cancelled=true;mount.current=false}
  },[select])

  const apply=useCallback((id:string,turn:Turn)=>{
    if(!mount.current)return
    if(terminal(turn)){
      pending.current=pending.current.filter(p=>!(p.conv===id && p.turn_id===turn.turn_id)); remember()
      void refreshList().catch(()=>{})
    }
    if(selected.current!==id)return
    setTurns(old=>old.some(t=>t.turn_id===turn.turn_id)?old.map(t=>t.turn_id===turn.turn_id?turn:t):[...old,turn])
    setUncertain(false)
  },[refreshList])
  const activeTurn=turns.find(t=>!terminal(t))
  useEffect(()=>{
    if(!selectedId || !activeTurn)return
    let cancelled=false, checking=false
    const id=selectedId, turnId=activeTurn.turn_id
    const close=api.subscribeTurn(id,turnId,t=>{if(!cancelled)apply(id,t)},()=>{
      if(cancelled || checking)return
      checking=true
      void api.getTurn(id,turnId).then(t=>{if(!cancelled)apply(id,t)}).catch(e=>{
        if(!cancelled && selected.current===id)showError(e)
      }).finally(()=>{checking=false})
    })
    return ()=>{cancelled=true;close()}
  },[selectedId,activeTurn?.turn_id,apply,showError])

  const create=async()=>{
    if(!health || modeBlocked.current)return
    try { const c=await api.createConversation(); await refreshList(); await select(c.id); setDraft('') }
    catch(e){showError(e)}
  }
  const post=async(p:Pending)=>{
    try {
      const t=await api.submitTurn(p.conv,{message:p.message,turn_id:p.turn_id})
      apply(p.conv,t)
      if(selected.current===p.conv){setDraft(current=>current===p.message?'':current);if(!modeBlocked.current)setError('')}
      await refreshList()
    } catch(e){
      const status=(e as {status?:number})?.status
      if(status && status<500){pending.current=pending.current.filter(x=>x.turn_id!==p.turn_id);remember()}
      if(selected.current===p.conv){setUncertain(!status || status>=500);showError(e)}
    }
  }
  const send=async()=>{
    if(modeBlocked.current || busy.current || activeTurn || uncertain || !selected.current || !draft.trim() || draft.length>16000 || loading)return
    busy.current=true
    const p={conv:selected.current,message:draft,turn_id:crypto.randomUUID()}
    pending.current.push(p);remember();setUncertain(true);setError('')
    try {await post(p)} finally {busy.current=false}
  }
  const retry=async()=>{
    if(modeBlocked.current || busy.current)return
    const p=pending.current.find(p=>p.conv===selected.current)
    if(!p){await select(selected.current);return}
    busy.current=true
    try {
      try {apply(p.conv,await api.getTurn(p.conv,p.turn_id));setError('')}
      catch(e){ if((e as {status?:number})?.status===404)await post(p);else showError(e) }
    } finally {busy.current=false}
  }
  return {health,conversations,selectedId,turns,activeTurn,draft,loading,error,uncertain,modeChanged,
          create,select,send,retry,setDraft}
}
