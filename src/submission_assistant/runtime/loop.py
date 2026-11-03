"""Shared LangGraph Agent loop with bounded native calls and private memory."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import time
from typing import TypedDict

from langgraph.graph import START, END, StateGraph

from .contracts import RunResult, ToolResult
from .memory import dumps
from .prompt import PromptBudgetError


@dataclass(frozen=True)
class LoopSettings:
    max_steps: int = 8
    max_tool_calls: int = 20
    deadline_s: float = 180
    max_parallel_tools: int = 2
    max_total_tokens_estimate: int = 100000

    def __post_init__(self):
        if min(self.max_steps,self.max_tool_calls,self.deadline_s,self.max_parallel_tools,self.max_total_tokens_estimate)<=0:
            raise ValueError('loop budgets must be positive')


class RequestBudget:
    """Shared event-loop budget. Claims occur without awaiting, so siblings cannot race."""
    def __init__(self,max_steps=24,max_tools=48,deadline_s=240):
        self.steps=0; self.tools=0
        self.max_steps=max_steps; self.max_tools=max_tools
        self.deadline=time.monotonic()+deadline_s
        self.tool_semaphore=asyncio.Semaphore(4)

    def claim(self,kind):
        if time.monotonic()>=self.deadline:
            return False
        if kind=='step':
            if self.steps>=self.max_steps: return False
            self.steps+=1
        else:
            if self.tools>=self.max_tools: return False
            self.tools+=1
        return True


class LoopState(TypedDict,total=False):
    terminal: bool


class AgentLoop:
    def __init__(self,memory,prompt,model,tools,*,settings=None,skills=None,budget=None):
        self.memory,self.prompt,self.model,self.tools=memory,prompt,model,tools
        self.settings=settings or LoopSettings()
        self.skills=skills or []
        self.budget=budget

    async def run(self,req,run_id,prepared,public,*,collaboration_results=None):
        settings=self.settings
        budget=self.budget or RequestBudget()
        started=time.monotonic()
        deadline=min(started+settings.deadline_s,budget.deadline)
        step=0; tool_count=0; token_estimate=0; repair_errors=0
        history_fingerprints={}
        schemas=self.tools.schemas(prepared['registered_tools'])
        allowed=[s['function']['name'] for s in schemas]
        evidence=[]
        unresolved=[]
        evidence_keys=set()
        def merge_evidence(items):
            for item in items:
                identity=item.get('evidence_id') or item.get('parent_id') or item.get('resource_ref') or hashlib.sha256(dumps(item).encode()).hexdigest()
                if identity not in evidence_keys:
                    evidence_keys.add(identity)
                    evidence.append(item)
        for result in collaboration_results or []:
            merge_evidence(result.get('evidence',[]))
            unresolved.extend(result.get('unresolved',[]))
            if result.get('execution_status')!='completed' and not result.get('unresolved'):
                unresolved.append(f"{result.get('agent_type','角色')}未完成核验")
        terminal=None

        def outcome(status,reason,answer=None):
            if status=='completed' and unresolved:
                status,reason='incomplete','partial_agent_coverage'
            self.memory.set_status(req,run_id,{
                'phase':'finished','execution_status':status,'stop_reason':reason,
                'current_step':step,'max_steps':settings.max_steps,'steps_remaining':settings.max_steps-step,
                'tool_calls_used':tool_count,'elapsed_ms':int((time.monotonic()-started)*1000),
                'remaining_time_ms':max(0,int((deadline-time.monotonic())*1000)),
                'token_usage_estimate':token_estimate,'usage_type':'estimated_not_billable',
                'public_memory_revision':public['revision'],
            })
            return RunResult(run_id=run_id,agent_type=prepared['agent_type'],
                             intent_ids=prepared['input']['intent_ids'],execution_status=status,stop_reason=reason,
                             answer_text=answer or '本次咨询未能完成核验，请稍后重试或缩小问题范围。',
                             evidence=evidence,unresolved=list(dict.fromkeys(unresolved+([] if status=='completed' else [reason]))),
                             current_step=step,tool_calls_used=tool_count)

        async def iteration(state):
            nonlocal step,tool_count,token_estimate,terminal,repair_errors
            if time.monotonic()>=deadline:
                terminal=outcome('incomplete','deadline'); return {'terminal':True}
            if step>=settings.max_steps:
                terminal=outcome('incomplete','step_limit'); return {'terminal':True}
            if token_estimate>=settings.max_total_tokens_estimate:
                terminal=outcome('incomplete','token_limit'); return {'terminal':True}
            if not budget.claim('step'):
                terminal=outcome('incomplete','request_budget'); return {'terminal':True}
            step+=1
            status={'current_step':step,'max_steps':settings.max_steps,'steps_remaining':settings.max_steps-step,
                    'tool_calls_used':tool_count,'elapsed_ms':int((time.monotonic()-started)*1000),
                    'remaining_time_ms':max(0,int((deadline-time.monotonic())*1000)),
                    'token_usage_estimate':token_estimate,'usage_type':'estimated_not_billable',
                    'phase':'preparing','warnings':[],'public_memory_revision':public['revision']}
            self.memory.set_status(req,run_id,status)
            instant=self.memory.instant_snapshot(req,run_id)
            instant.update(guidance=prepared['input'].get('guidance',''),collaboration_results=collaboration_results or [])
            kwargs={'system_prompt':prepared['system_prompt'],'tools':schemas,'skills':self.skills}
            snapshot=self.prompt.assemble(public,instant,**kwargs)
            def trigger():
                return self.memory.trigger_compression(req,run_id,input_tokens=snapshot.input_tokens,
                    input_budget=self.prompt.input_budget,trigger_tokens=getattr(self.prompt,'compression_trigger_tokens',None),
                    measure=self.prompt.count)
            compression=trigger()
            for batch in range(4):
                if snapshot.input_tokens<=self.prompt.input_budget or compression is None:
                    break
                try:
                    committed=await asyncio.wait_for(asyncio.shield(compression),
                        min(self.memory.compression_timeout_s,max(.01,deadline-time.monotonic())))
                except TimeoutError:
                    break
                instant=self.memory.instant_snapshot(req,run_id)
                instant.update(guidance=prepared['input'].get('guidance',''),collaboration_results=collaboration_results or [])
                snapshot=self.prompt.assemble(public,instant,**kwargs)
                if not committed or snapshot.input_tokens<=self.prompt.input_budget or batch==3:
                    break
                compression=trigger()
            try:
                snapshot=self.prompt.build(public,instant,**kwargs)
            except PromptBudgetError:
                terminal=outcome('incomplete','input_budget'); return {'terminal':True}
            output_reserve=max(0,getattr(self.model,'max_output_tokens',0))*(1 if self.prompt.token_counter else 4)
            if token_estimate+snapshot.input_tokens+output_reserve>settings.max_total_tokens_estimate:
                terminal=outcome('incomplete','token_limit'); return {'terminal':True}
            token_estimate+=snapshot.input_tokens
            allow_tools=bool(schemas) and step<settings.max_steps and tool_count<settings.max_tool_calls
            self.memory.set_status(req,run_id,{**status,'phase':'model_call'})
            try:
                reply=await asyncio.wait_for(self.model.complete(snapshot.messages,schemas,allow_tools=allow_tools),max(.01,deadline-time.monotonic()))
            except TimeoutError:
                terminal=outcome('incomplete','deadline'); return {'terminal':True}
            except asyncio.CancelledError: raise
            except Exception:
                terminal=outcome('failed','model_service_error'); return {'terminal':True}
            message=reply.get('message') if isinstance(reply,dict) else None
            if not isinstance(message,dict) or message.get('role')!='assistant':
                terminal=outcome('failed','model_protocol_error'); return {'terminal':True}
            content=message.get('content') or ''
            calls=message.get('tool_calls') or []
            if not isinstance(content,str) or not isinstance(calls,list) or len(calls)>20:
                terminal=outcome('failed','model_protocol_error'); return {'terminal':True}
            seen=set()
            for call in calls:
                function=call.get('function') if isinstance(call,dict) else None
                identifier=call.get('id') if isinstance(call,dict) else None
                if not isinstance(identifier,str) or not identifier or identifier in seen or not isinstance(function,dict) or not isinstance(function.get('name'),str) or not isinstance(function.get('arguments'),str):
                    terminal=outcome('failed','model_protocol_error'); return {'terminal':True}
                seen.add(identifier)
            native={'content':content}
            if calls: native['tool_calls']=calls
            self.memory.append(req,run_id,'assistant',native,step)
            token_estimate+=self.prompt.count(native)
            if token_estimate>settings.max_total_tokens_estimate:
                for call in calls:
                    self.memory.append(req,run_id,'tool',{'tool_call_id':call['id'],'content':dumps(ToolResult.failure('BUDGET_EXHAUSTED','模型输出已超过估算预算，工具未执行。').model_dump())},step)
                terminal=outcome('incomplete','token_limit'); return {'terminal':True}
            reason=reply.get('finish_reason')
            if reason in ('length','content_filter') or message.get('refusal'):
                # Truncated native tool arguments are never executed. Pair them with safe errors.
                for call in calls:
                    self.memory.append(req,run_id,'tool',{'tool_call_id':call['id'],'content':dumps(ToolResult.failure('CANCELLED','模型响应未完整结束，工具未执行。').model_dump())},step)
                terminal=outcome('incomplete','truncated_response' if reason=='length' else 'refusal')
                return {'terminal':True}
            if not calls:
                if not content.strip(): terminal=outcome('incomplete','empty_response')
                elif reason!='stop': terminal=outcome('incomplete','invalid_finish_reason')
                else: terminal=outcome('completed','final_answer',content)
                return {'terminal':True}
            semaphore=asyncio.Semaphore(settings.max_parallel_tools)
            planned=[]
            no_progress=False
            for call in calls:
                fn=call['function']
                try: canonical=dumps(json.loads(fn['arguments']))
                except ValueError: canonical=fn['arguments']
                key=(fn['name'],canonical)
                if history_fingerprints.get(key,(None,0))[1]>=2:
                    result=ToolResult.failure('NO_PROGRESS','相同调用已连续返回相同结果，请停止重复查询。')
                    no_progress=True
                elif not allow_tools or tool_count>=settings.max_tool_calls or not budget.claim('tool'):
                    result=ToolResult.failure('BUDGET_EXHAUSTED','剩余预算不允许新的工具调用。')
                else:
                    tool_count+=1
                    result=None
                planned.append((call,key,result))
            # Keep each parallel call independent: one completed tool must not hide
            # another tool that is still running. Never put arguments in status.
            activities={call['id']:{'name':call['function']['name'],'state':'waiting'}
                        for call,_,result in planned if result is None and call['function']['name'] in allowed}
            def publish_tools():
                if not activities: return
                busy=any(t['state'] in ('waiting','running') for t in activities.values())
                self.memory.set_status(req,run_id,{**status,
                    'phase':'tool_call' if busy else 'tool_result',
                    'tools':list(activities.values()),'tool_calls_used':tool_count,
                    'elapsed_ms':int((time.monotonic()-started)*1000),
                    'remaining_time_ms':max(0,int((deadline-time.monotonic())*1000))})
            publish_tools()
            async def execute(item):
                call,key,result=item
                if result is not None: return call,key,result
                async with semaphore,budget.tool_semaphore:
                    if time.monotonic()>=deadline:
                        if call['id'] in activities:
                            activities[call['id']]['state']='timed_out'
                            publish_tools()
                        return call,key,ToolResult.failure('TIMEOUT','整轮执行时间已耗尽。')
                    remaining=min(90 if call['function']['name']=='search_knowledge' else 5,deadline-time.monotonic())
                    if call['id'] in activities:
                        activities[call['id']]['state']='running'
                        publish_tools()
                    tool_state='failed'
                    try:
                        try:
                            result=await asyncio.wait_for(self.tools.execute(call['function']['name'],call['function']['arguments'],allowed),remaining)
                        except TimeoutError:
                            result=ToolResult.failure('TIMEOUT','工具超过截止时间。',True)
                        tool_state=('timed_out' if result.error and result.error.code=='TIMEOUT' else
                                    'failed' if result.error else 'completed')
                    except asyncio.CancelledError:
                        tool_state='cancelled'
                        raise
                    finally:
                        if call['id'] in activities:
                            activities[call['id']]['state']=tool_state
                            publish_tools()
                    return call,key,result
            pending=[asyncio.create_task(execute(item)) for item in planned]
            try:
                results=await asyncio.gather(*pending)
            except asyncio.CancelledError:
                for task in pending: task.cancel()
                await asyncio.gather(*pending,return_exceptions=True)
                for call,_,_ in planned:
                    self.memory.append(req,run_id,'tool',{'tool_call_id':call['id'],'content':dumps(ToolResult.failure('CANCELLED','工具组已取消，结果未发布。').model_dump())},step)
                raise
            # Leave room for metadata, the next decision and status changes.
            remaining_presentation=max(0,self.prompt.input_budget-snapshot.input_tokens-self.prompt.count(native)-512)
            presentation_limit=min(12000,remaining_presentation//max(1,len(results)))
            presentation_exhausted=False
            for call,key,result in results:
                raw=result.model_dump()
                serialized=dumps(raw)
                fingerprint=hashlib.sha256(serialized.encode()).hexdigest()
                previous,count=history_fingerprints.get(key,(None,0))
                history_fingerprints[key]=(fingerprint,count+1 if fingerprint==previous else 1)
                if result.error and result.error.code in ('INVALID_ARGUMENT','TOOL_NOT_ALLOWED'):
                    repair_errors+=1
                if call['function']['name']=='search_knowledge':
                    merge_evidence(result.data.get('evidence',[]))
                view,ref=self.tools.present(call['function']['name'],call['function']['arguments'],result,
                                           limit=presentation_limit,count=self.prompt.count,call_id=call['id'])
                body={'tool_call_id':call['id'],'content':dumps(view.model_dump())}
                if ref: body['raw_resource_ref']=ref
                if view.error and view.error.code=='PRESENTATION_BUDGET':
                    presentation_exhausted=True
                self.memory.append(req,run_id,'tool',body,step)
            if presentation_exhausted:
                terminal=outcome('incomplete','input_budget'); return {'terminal':True}
            if no_progress:
                terminal=outcome('incomplete','no_progress'); return {'terminal':True}
            if repair_errors>=3:
                terminal=outcome('incomplete','invalid_tool_limit'); return {'terminal':True}
            return {'terminal':False}

        builder=StateGraph(LoopState)
        builder.add_node('agent_step',iteration)
        builder.add_edge(START,'agent_step')
        builder.add_conditional_edges('agent_step',lambda s: 'end' if s['terminal'] else 'continue',{'end':END,'continue':'agent_step'})
        graph=builder.compile()
        final_state='failed'
        try:
            await graph.ainvoke({},config={'recursion_limit':settings.max_steps+4})
            final_state=terminal.execution_status
            return terminal
        except asyncio.CancelledError:
            final_state='cancelled'; raise
        finally:
            await self.memory.seal_run(req,run_id,final_state)
