import { afterEach, expect, test, vi } from 'vitest'
import * as api from './api'
afterEach(()=>vi.unstubAllGlobals())
test('writes_are_bound_to_mode_returned_by_health',async()=>{
  const calls:RequestInit[]=[]
  vi.stubGlobal('fetch',async(_url:string,init:RequestInit)=>{
    calls.push(init)
    return {ok:true,json:async()=>calls.length===1?{mode:'demo',prompt_version:'demo',dependencies:{}}:{id:'c1'}}
  })
  await api.health(); await api.createConversation()
  expect((calls[1].headers as Record<string,string>)['X-Workbench-Mode']).toBe('demo')
})
