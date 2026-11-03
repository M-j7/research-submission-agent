import type { Conversation, ConversationSummary, Health, SendBody, Turn } from './types'
let expectedMode: 'demo'|'live'|undefined

export class ApiError extends Error {
  constructor(public status: number, public code: string) { super(code) }
}
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response=await fetch('/api'+path, { ...init, headers: {'Content-Type':'application/json',
    ...(expectedMode?{'X-Workbench-Mode':expectedMode}:{}), ...init?.headers} })
  if (!response.ok) {
    let code='service_unavailable'
    try { code=(await response.json()).error || code } catch { /* never display raw responses */ }
    throw new ApiError(response.status, code)
  }
  return response.json()
}
const root=(id:string) => '/conversations/'+encodeURIComponent(id)
const turnPath=(id:string,turnId:string) => root(id)+'/turns/'+encodeURIComponent(turnId)
export const health=async() => {
  const result=await request<Health>('/health')
  expectedMode=result.mode
  return result
}
export const listConversations=() => request<ConversationSummary[]>('/conversations')
export const createConversation=() => request<ConversationSummary>('/conversations',{method:'POST'})
export const getConversation=(id:string) => request<Conversation>(root(id))
export const submitTurn=(id:string,body:SendBody) => request<Turn>(root(id)+'/turns',{method:'POST',body:JSON.stringify(body)})
export const getTurn=(id:string,turnId:string) => request<Turn>(turnPath(id,turnId))
export function subscribeTurn(id:string,turnId:string,onSnapshot:(t:Turn)=>void,onError:()=>void):()=>void {
  const source=new EventSource('/api'+turnPath(id,turnId)+'/events'+(expectedMode?'?mode='+expectedMode:''))
  const receive=(e:MessageEvent) => {
    try {
      const turn=JSON.parse(e.data).turn
      if(!turn || typeof turn.turn_id!=='string'){onError();return}
      onSnapshot(turn)
    } catch { onError() }
  }
  for (const name of ['stage','result','error']) source.addEventListener(name, receive as EventListener)
  source.onerror=() => onError()
  return () => source.close()
}
