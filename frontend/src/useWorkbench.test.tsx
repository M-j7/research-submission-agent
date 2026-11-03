import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, expect, test, vi } from 'vitest'
import { useWorkbench } from './useWorkbench'
import * as api from './api'

vi.mock('./api', () => ({ health: vi.fn(), listConversations: vi.fn(), createConversation: vi.fn(),
  getConversation: vi.fn(), submitTurn: vi.fn(), getTurn: vi.fn(), subscribeTurn: vi.fn() }))

const turn = { turn_id: 't1', message: '摘要字数', state: 'running', answer: null,
  routing: null, agent_results: [], status: [], error_code: null }
let onSnapshot: (t: any) => void
let onError: () => void
beforeEach(() => {
  vi.clearAllMocks(); localStorage.clear()
  vi.mocked(api.health).mockResolvedValue({ mode: 'demo', prompt_version: 'demo', dependencies: {} } as any)
  vi.mocked(api.listConversations).mockResolvedValue([{id:'c1',title:'会话一',updated_at:''},{id:'c2',title:'会话二',updated_at:''}])
  vi.mocked(api.getConversation).mockImplementation(async id => ({id,title:'会话',turns:[]}))
  vi.mocked(api.submitTurn).mockImplementation(async (_id, body) => ({...turn,turn_id:body.turn_id}) as any)
  vi.mocked(api.subscribeTurn).mockImplementation((_c,_t,s,e) => { onSnapshot=s; onError=e; return vi.fn() })
  vi.mocked(api.getTurn).mockResolvedValue(turn as any)
})
async function ready() {
  const hook=renderHook(() => useWorkbench())
  await waitFor(() => expect(hook.result.current.selectedId).toBe('c1'))
  await waitFor(() => expect(hook.result.current.loading).toBe(false))
  return hook
}
test('reconnect_does_not_post_again', async () => {
  const {result}=await ready()
  act(() => result.current.setDraft('摘要字数'))
  await act(() => result.current.send())
  await waitFor(() => expect(api.subscribeTurn).toHaveBeenCalled())
  await act(async () => { onError(); await Promise.resolve() })
  expect(api.submitTurn).toHaveBeenCalledTimes(1)
})
test('lost_post_response_keeps_turn_id', async () => {
  vi.mocked(api.submitTurn).mockRejectedValueOnce(new Error('network'))
  const {result}=await ready()
  act(() => result.current.setDraft('摘要字数'))
  await act(() => result.current.send())
  const initial=vi.mocked(api.submitTurn).mock.calls[0][1]
  vi.mocked(api.getTurn).mockRejectedValueOnce(Object.assign(new Error('not_found'), {status:404}))
  await act(() => result.current.retry())
  expect(vi.mocked(api.submitTurn).mock.calls[1][1]).toEqual(initial)
})
test('switch_ignores_stale_event', async () => {
  const {result}=await ready()
  act(() => result.current.setDraft('摘要字数'))
  await act(() => result.current.send())
  await waitFor(() => expect(api.subscribeTurn).toHaveBeenCalled())
  const old=onSnapshot
  await act(() => result.current.select('c2'))
  act(() => old({...turn,state:'completed',answer:'旧会话答案'}))
  expect(result.current.selectedId).toBe('c2')
  expect(result.current.turns.some(t=>t.answer==='旧会话答案')).toBe(false)
})
test('reload_restores_running_turn_without_post', async () => {
  vi.mocked(api.getConversation).mockResolvedValue({id:'c1',title:'恢复',turns:[turn]} as any)
  const {result}=await ready()
  await waitFor(() => expect(api.subscribeTurn).toHaveBeenCalled())
  expect(result.current.activeTurn?.turn_id).toBe('t1')
  expect(api.submitTurn).not.toHaveBeenCalled()
})
test('server_failure_preserves_draft', async () => {
  vi.mocked(api.submitTurn).mockRejectedValueOnce(new Error('network'))
  const {result}=await ready()
  act(() => result.current.setDraft('摘要字数'))
  await act(() => result.current.send())
  expect(result.current.draft).toBe('摘要字数')
  expect(result.current.error).toBeTruthy()
})
test('accepted_slow_post_does_not_clear_edited_draft', async()=>{
  let finish!:(value:any)=>void
  vi.mocked(api.submitTurn).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve}))
  const {result}=await ready()
  act(()=>result.current.setDraft('摘要字数'))
  let sending!:Promise<void>
  act(()=>{sending=result.current.send()})
  act(()=>result.current.setDraft('刚想到的新问题'))
  await act(async()=>{finish(turn);await sending})
  expect(result.current.draft).toBe('刚想到的新问题')
})
test('old_same_conversation_load_cannot_replace_new_turn',async()=>{
  const {result}=await ready()
  let release!:(value:any)=>void
  vi.mocked(api.getConversation).mockImplementationOnce(()=>new Promise(resolve=>{release=resolve}))
  let oldLoad!:Promise<void>
  act(()=>{oldLoad=result.current.select('c1')})
  await act(()=>result.current.select('c2'))
  await act(()=>result.current.select('c1'))
  act(()=>result.current.setDraft('摘要字数'))
  await act(()=>result.current.send())
  expect(result.current.activeTurn).toBeTruthy()
  await act(async()=>{release({id:'c1',title:'旧快照',turns:[]});await oldLoad})
  expect(result.current.activeTurn).toBeTruthy()
})
test('mode_changed_blocks_further_send_until_reload',async()=>{
  vi.mocked(api.submitTurn).mockRejectedValueOnce(Object.assign(new Error('mode'),{status:409,code:'service_mode_changed'}))
  const {result}=await ready()
  act(()=>result.current.setDraft('摘要字数'))
  await act(()=>result.current.send())
  expect(result.current.error).toContain('刷新')
  await act(()=>result.current.send())
  expect(api.submitTurn).toHaveBeenCalledTimes(1)
})
