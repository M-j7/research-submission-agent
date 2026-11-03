import { terminal, type Status, type Turn, type ToolActivity } from '../types'

export const roles:Record<string,string>={general:'通用反馈',submission_requirements:'稿件规范',editorial_workflow:'投稿流程',publication_policy:'出版政策',publication_fees:'发表费用',composer:'汇总审核'}
const phases:Record<string,string>={pending:'等待处理',receive:'正在接收问题',routing:'正在识别意图',
  executing:'专业 Agent 正在处理',composing:'正在汇总各 Agent 结果',saving:'正在保存答复',
  completed:'已完成',incomplete:'部分完成',failed:'执行失败',cancelled:'已取消',interrupted:'运行中断'}
const toolNames:Record<string,string>={search_knowledge:'知识检索',read:'读取资料',glob:'查找文件',grep:'搜索资料',query_memory:'查询记忆'}
const toolStates:Record<ToolActivity['state'],string>={waiting:'等待调用',running:'调用中',completed:'已完成',failed:'失败',timed_out:'超时',cancelled:'已取消'}
const stopReasons:Record<string,string>={deadline:'执行超时',step_limit:'达到步骤上限',no_progress:'重复调用无进展',
  model_service_error:'模型服务异常',request_budget:'达到本轮预算上限',input_budget:'上下文预算不足',
  token_limit:'达到用量上限',invalid_tool_limit:'工具调用异常次数过多'}

function agentLabel(status:Status, ended:boolean) {
  const role=roles[status.agent_type||'']||'专业'
  const name=`${role} Agent`
  if(ended||status.phase==='finished')return `${name} · ${phases[status.execution_status||'']||'已结束'}`
  const running=status.tools?.filter(t=>t.state==='running')||[]
  if(running.some(t=>t.name==='search_knowledge'))return `${name} 正在进行 RAG 知识检索`
  if(running.length)return `${name} 正在调用 ${running.map(t=>t.name).join('、')} 工具`
  if(status.phase==='tool_call')return `${name} 正在等待工具执行`
  if(status.phase==='tool_result')return `${name} · 工具执行结束`
  if(status.phase==='preparing')return `${name} 正在准备上下文`
  if(status.phase==='model_call')return `${name} 正在思考 / 生成回复`
  return `${name} 等待状态更新`
}

export function ExecutionProgress({turn,compact=false}:{turn:Turn;compact?:boolean}) {
  const ended=terminal(turn)
  const stage=turn.status.find(s=>!s.agent_type)?.phase||turn.state
  const heading=phases[ended?turn.state:stage]||'正在处理'
  const agents=turn.status.filter(s=>s.agent_type)
  return <div className={`execution-progress ${compact?'compact':''} ${ended?'ended':'active'}`}
    role="status" aria-label="本轮执行进度" aria-live="polite" aria-atomic="true">
    <div className="execution-heading"><span className={ended?'execution-check':'pulse'} aria-hidden="true">{ended?'◇':null}</span>
      <strong>{heading}</strong><small>{ended?'执行状态':'实时执行'}</small></div>
    {agents.map((status,i)=><div className="execution-agent" key={`${status.agent_type}-${i}`}>
      <div className="execution-action">{agentLabel(status,ended)}</div>
      {!ended&&Boolean(status.tools?.length)&&<div className="execution-tools">{status.tools!.map((tool,j)=>
        <span className={`execution-tool ${tool.state}`} key={j} title={toolNames[tool.name]||'工具'}>{tool.name} · {toolStates[tool.state]||'已结束'}</span>)}</div>}
      {!compact&&<div className="execution-budget">
        {status.current_step!==undefined&&<span>步骤 {status.current_step}</span>}
        {status.tool_calls_used!==undefined&&<span>工具 {status.tool_calls_used} 次</span>}
        {status.elapsed_ms!==undefined&&<span>{(status.elapsed_ms/1000).toFixed(1)} 秒</span>}
        {status.stop_reason&&stopReasons[status.stop_reason]&&<span>{stopReasons[status.stop_reason]}</span>}
      </div>}
    </div>)}
    {agents.length===0&&!ended&&stage==='executing'&&<p className="execution-wait">正在准备已分配的角色…</p>}
  </div>
}
