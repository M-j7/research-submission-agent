import { fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { ChatPanel } from './ChatPanel'
import type { Turn } from '../types'

const props={draft:'',disabled:false,loading:false,uncertain:false,error:'',
  onDraft:vi.fn(),onSend:vi.fn(),onRetry:vi.fn()}
const turn:Turn={turn_id:'t1',message:'一个问题',answer:'初始回答',state:'running',
  routing:null,agent_results:[],status:[],error_code:null}
let scroll:ReturnType<typeof vi.fn>
const originalScroll=Object.getOwnPropertyDescriptor(HTMLElement.prototype,'scrollIntoView')
const originalHeight=Object.getOwnPropertyDescriptor(document.documentElement,'scrollHeight')
beforeEach(()=>{
  scroll=vi.fn()
  Object.defineProperty(HTMLElement.prototype,'scrollIntoView',{value:scroll,configurable:true})
  Object.defineProperty(document.documentElement,'scrollHeight',{value:1600,configurable:true})
  vi.stubGlobal('innerHeight',600)
  vi.stubGlobal('scrollY',1000)
})
afterEach(()=>{
  if(originalScroll)Object.defineProperty(HTMLElement.prototype,'scrollIntoView',originalScroll)
  else Reflect.deleteProperty(HTMLElement.prototype,'scrollIntoView')
  if(originalHeight)Object.defineProperty(document.documentElement,'scrollHeight',originalHeight)
  else Reflect.deleteProperty(document.documentElement,'scrollHeight')
  vi.unstubAllGlobals()
})
test('manual_history_reading_is_not_pulled_to_bottom_by_new_content',()=>{
  const {rerender}=render(<ChatPanel {...props} turns={[turn]}/>)
  scroll.mockClear()
  vi.stubGlobal('scrollY',200)
  fireEvent.scroll(window)
  rerender(<ChatPanel {...props} turns={[{...turn,answer:'更新的回答'}]}/>)
  expect(scroll).not.toHaveBeenCalled()
})
test('returning_near_bottom_resumes_content_following',()=>{
  const {rerender}=render(<ChatPanel {...props} turns={[turn]}/>)
  vi.stubGlobal('scrollY',200);fireEvent.scroll(window)
  rerender(<ChatPanel {...props} turns={[{...turn,answer:'第二段'}]}/>)
  scroll.mockClear()
  vi.stubGlobal('scrollY',980);fireEvent.scroll(window)
  rerender(<ChatPanel {...props} turns={[{...turn,answer:'第三段'}]}/>)
  expect(scroll).toHaveBeenCalledTimes(1)
})
test('status_only_refresh_does_not_force_scrolling',()=>{
  const {rerender}=render(<ChatPanel {...props} turns={[turn]}/>)
  scroll.mockClear()
  rerender(<ChatPanel {...props} turns={[{...turn,status:[{current_step:2,elapsed_ms:2000}]}]}/>)
  expect(scroll).not.toHaveBeenCalled()
})
test('new_progress_layout_follows_only_when_reader_is_near_bottom',()=>{
  const {rerender}=render(<ChatPanel {...props} turns={[turn]}/>);
  scroll.mockClear()
  rerender(<ChatPanel {...props} turns={[{...turn,status:[{phase:'executing'},
    {agent_type:'publication_fees',phase:'model_call'}]}]}/>);
  expect(scroll).toHaveBeenCalledTimes(1)
  vi.stubGlobal('scrollY',200);fireEvent.scroll(window)
  scroll.mockClear()
  rerender(<ChatPanel {...props} turns={[{...turn,status:[{phase:'executing'},
    {agent_type:'publication_fees',phase:'tool_call',tools:[{name:'read',state:'running'}]}]}]}/>);
  expect(scroll).not.toHaveBeenCalled()
})
