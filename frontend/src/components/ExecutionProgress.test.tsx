import { render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import { ChatPanel } from './ChatPanel'
import { EvidencePanel } from './EvidencePanel'
import type { Turn } from '../types'

const props={draft:'',disabled:true,loading:false,uncertain:false,error:'',
  onDraft:vi.fn(),onSend:vi.fn(),onRetry:vi.fn()}
const turn:Turn={turn_id:'progress',message:'摘要要求和费用',answer:null,state:'running',
  routing:null,agent_results:[],status:[{phase:'routing'}],error_code:null}

test('current_routing_stage_is_visible_in_the_answer_not_only_the_sidebar',()=>{
  render(<ChatPanel {...props} turns={[turn]}/>);
  expect(screen.getByRole('status',{name:'本轮执行进度'})).toHaveTextContent('正在识别意图')
})
test('parallel_agents_show_their_own_model_and_rag_actions',()=>{
  const running:Turn={...turn,status:[{phase:'executing'},
    {agent_type:'publication_fees',phase:'model_call',current_step:1},
    {agent_type:'submission_requirements',phase:'tool_call',current_step:2,
      tools:[{name:'search_knowledge',state:'running'},{name:'read',state:'completed'}]}]}
  render(<ChatPanel {...props} turns={[running]}/>);
  const progress=screen.getByRole('status',{name:'本轮执行进度'})
  expect(progress).toHaveTextContent('发表费用 Agent 正在思考 / 生成回复')
  expect(progress).toHaveTextContent('稿件规范 Agent 正在进行 RAG 知识检索')
  expect(progress).toHaveTextContent('read · 已完成')
})
test('sidebar_shows_tool_names_and_timeout_without_internal_arguments',()=>{
  render(<EvidencePanel turn={{...turn,status:[{phase:'executing'},
    {agent_type:'publication_fees',phase:'tool_result',tools:[{name:'search_knowledge',state:'timed_out'}]}]}}/>);
  const progress=screen.getByRole('status',{name:'本轮执行进度'})
  expect(progress).toHaveTextContent('search_knowledge · 超时')
  expect(progress).not.toHaveTextContent('正在进行 RAG')
})
test('terminal_turn_overrides_stale_running_snapshot',()=>{
  render(<ChatPanel {...props} turns={[{...turn,state:'failed',status:[{phase:'executing'},
    {agent_type:'publication_fees',phase:'tool_call',tools:[{name:'search_knowledge',state:'running'}]}]}]}/>);
  const progress=screen.getByRole('status',{name:'本轮执行进度'})
  expect(progress).toHaveTextContent('执行失败')
  expect(progress).not.toHaveTextContent('正在')
  expect(progress).not.toHaveTextContent('调用中')
})
test('completed_agent_does_not_look_active_while_another_agent_runs',()=>{
  render(<ChatPanel {...props} turns={[{...turn,status:[{phase:'executing'},
    {agent_type:'publication_fees',phase:'finished',execution_status:'completed'},
    {agent_type:'submission_requirements',phase:'model_call'}]}]}/>);
  const progress=screen.getByRole('status',{name:'本轮执行进度'})
  expect(progress).toHaveTextContent('发表费用 Agent · 已完成')
  expect(progress).toHaveTextContent('稿件规范 Agent 正在思考')
})
test('old_conversation_without_detailed_status_still_renders',()=>{
  render(<ChatPanel {...props} turns={[{...turn,state:'completed',answer:'历史答复',status:[]}]}/>);
  expect(screen.getByText('历史答复')).toBeVisible()
  expect(screen.queryByText('正在识别意图')).not.toBeInTheDocument()
})
