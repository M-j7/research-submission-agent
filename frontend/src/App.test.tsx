import { fireEvent, render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import { ChatPanel } from './components/ChatPanel'
import { EvidencePanel } from './components/EvidencePanel'
import { SafeMarkdown } from './components/SafeMarkdown'
import App from './App'

vi.mock('./useWorkbench',()=>({useWorkbench:()=>({health:{mode:'demo',prompt_version:'demo'},
  conversations:[],selectedId:'',turns:[],activeTurn:undefined,draft:'',loading:false,error:'',uncertain:false,
  create:vi.fn(),select:vi.fn(),send:vi.fn(),retry:vi.fn(),setDraft:vi.fn()})}))
const turn={turn_id:'t',message:'问题',answer:'部分答复',state:'incomplete' as const,routing:null,
  agent_results:[{agent_type:'general',intent_ids:[],execution_status:'incomplete',stop_reason:'max_steps',unresolved:['尚未核实费用'],evidence:[]}],status:[],error_code:null}
const props={turns:[],draft:'摘要',disabled:false,loading:false,uncertain:false,error:'',onDraft:vi.fn(),onSend:vi.fn(),onRetry:vi.fn()}
test('demo_badge_is_always_visible',()=>{
  render(<App/>);expect(screen.getByText('离线演示')).toBeVisible()
})
test('composition_enter_does_not_send',()=>{
  const send=vi.fn();render(<ChatPanel {...props} onSend={send}/>);
  const input=screen.getByRole('textbox',{name:'输入问题'})
  fireEvent.keyDown(input,{key:'Enter',isComposing:true});expect(send).not.toHaveBeenCalled()
  fireEvent.keyDown(input,{key:'Enter',shiftKey:true});expect(send).not.toHaveBeenCalled()
  fireEvent.keyDown(input,{key:'Enter'});expect(send).toHaveBeenCalledTimes(1)
})
test('empty_message_disabled',()=>{
  render(<ChatPanel {...props} draft=' '/>);expect(screen.getByRole('button',{name:'发送问题'})).toBeDisabled()
})
test('partial_result_and_reason_visible',()=>{
  render(<ChatPanel {...props} turns={[turn]}/>);
  expect(screen.getByText('部分完成')).toBeVisible();expect(screen.getByText('尚未核实费用')).toBeVisible()
})
test('unsafe_markdown_links_removed',()=>{
  const {container}=render(<SafeMarkdown text={'[坏链接](javascript:alert(1)) [本地](file:///C:/private) <script>alert(1)</script>'}/>);
  expect(container.querySelector('a[href^="javascript:"]')).toBeNull()
  expect(container.querySelector('a[href^="file:"]')).toBeNull()
  expect(container.querySelector('script')).toBeNull()
})
test('missing_evidence_is_explicit',()=>{
  render(<EvidencePanel turn={turn}/>);expect(screen.getByText('暂无证据')).toBeVisible()
})
