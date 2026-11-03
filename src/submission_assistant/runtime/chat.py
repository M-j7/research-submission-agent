"""Executable outer DAG; old graph.py remains the input-preview API."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END,START,StateGraph
from langsmith import tracing_context

from ..agents import AgentRequest
from ..graph import build_agent_graph
from .contracts import Request
from .loop import AgentLoop,RequestBudget,LoopSettings
from .prompt import PromptManager
from .skills import SkillCatalog
from .tools import ToolRegistry, DEFINITIONS
from .workspace import WorkspaceManager


class ChatState(TypedDict,total=False):
    request: Request
    cached: bool
    public: dict
    prepared_agents: list[dict]
    routing_decision: dict
    agent_results: list[dict]
    final_result: dict
    answer: str


class ChatService:
    def __init__(self,project:Path,memory,recognizer,model,*,rag=None,settings=None,prompt=None,maintenance=None,tool_registry_factory=ToolRegistry):
        self.project=Path(project)
        self.memory=memory; self.recognizer=recognizer; self.model=model; self.rag=rag
        self.settings=settings or LoopSettings()
        self.prompt=prompt or PromptManager()
        self.catalog=SkillCatalog(self.project/'resources'/'skills')
        self.maintenance=maintenance
        self.tool_registry_factory=tool_registry_factory
        self._background=set()

    def maintain(self,req):
        if self.maintenance is None: return
        task=asyncio.create_task(self.maintenance.update(req))
        self._background.add(task)

    async def close(self):
        if not self._background: return
        tasks=list(self._background)
        try:
            await asyncio.wait_for(asyncio.gather(*tasks,return_exceptions=True),60)
        except TimeoutError:
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
        finally:
            self._background.difference_update(tasks)

    async def execute(self,req,prepared,public,budget,collaboration_results=None):
        run_id=uuid4().hex
        self.memory.create_run(req,run_id,prepared['agent_type'],prepared['input']['intent_ids'])
        workspace=None
        state='failed'
        try:
            workspace=WorkspaceManager(self.project,req,run_id)
            tools=self.tool_registry_factory(workspace,self.memory,req,run_id,rag=self.rag)
            model=self.model.for_role(prepared['agent_type']) if hasattr(self.model,'for_role') else self.model
            skills=[] if prepared['agent_type']=='general' else self.catalog.entries()
            loop=AgentLoop(self.memory,self.prompt,model,tools,settings=self.settings,skills=skills,budget=budget)
            with tracing_context(enabled=False):
                result=await loop.run(req,run_id,prepared,public,collaboration_results=collaboration_results)
            self.memory.publish_result(req,run_id,result.model_dump())
            state=result.execution_status
            return result.model_dump()
        except asyncio.CancelledError:
            state='cancelled'; raise
        finally:
            await self.memory.seal_run(req,run_id,state)
            # Persistence failure retains scratch instead of pretending archival succeeded.
            if workspace is not None and state in ('completed','incomplete','cancelled'):
                workspace.cleanup()


def build_chat_graph(service:ChatService):
    # Budget is per invocation, not captured on the reusable service or graph.
    budgets={}
    def receive(state):
        req=state['request']
        if not isinstance(req,Request): req=Request.model_validate(req)
        previous=service.memory.begin_turn(req)
        if previous['state'] in ('completed','incomplete'):
            return {'request':req,'cached':True,'answer':previous['answer']}
        if not service.memory.claim_turn(req):
            raise RuntimeError('该轮正在执行或曾中断；未自动重放工具。请使用新的turn_id。')
        budgets[(req.user_id,req.conv_id,req.turn_id)]=RequestBudget()
        return {'request':req,'cached':False,'public':service.memory.public_snapshot(req)}

    async def recognize_and_prepare(state):
        req=state['request']; public=state['public']
        history=[{'role':r['role'],'content':r['content']} for r in public['recent']]
        req_old=AgentRequest(req.message,history,req.turn_id)
        registry={name:object() for name in DEFINITIONS if name!='search_knowledge' or service.rag is not None}
        try:
            with tracing_context(enabled=False):
                output=await build_agent_graph(service.recognizer,tool_registry=registry).ainvoke({'request':req_old})
        except BaseException as exc:
            service.memory.abort_turn(req,'cancelled' if isinstance(exc,asyncio.CancelledError) else 'failed')
            budgets.pop((req.user_id,req.conv_id,req.turn_id),None)
            raise
        return {'prepared_agents':output['prepared_agents'],'routing_decision':output['routing_decision']}

    async def run_agents(state):
        req=state['request']; budget=budgets[(req.user_id,req.conv_id,req.turn_id)]
        tasks=[asyncio.create_task(service.execute(req,p,state['public'],budget)) for p in state['prepared_agents']]
        try:
            outcomes=await asyncio.gather(*tasks,return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            service.memory.abort_turn(req,'cancelled')
            budgets.pop((req.user_id,req.conv_id,req.turn_id),None)
            raise
        results=[]
        for prepared,outcome in zip(state['prepared_agents'],outcomes):
            if isinstance(outcome,BaseException):
                if isinstance(outcome,asyncio.CancelledError): raise outcome
                # Persistence/authorization exceptions must not be silently made into a success.
                service.memory.abort_turn(req,'failed')
                budgets.pop((req.user_id,req.conv_id,req.turn_id),None)
                raise RuntimeError('角色执行或原文保存失败，未提交本轮最终回答。') from None
            results.append(outcome)
        return {'agent_results':results}

    async def compose(state):
        results=state['agent_results']; req=state['request']
        if len(results)==1: return {'final_result':results[0]}
        prepared={'agent_type':'composer','system_prompt':'你是IEEE Access咨询汇总审核Agent。核对原问题各意图覆盖、角色结论、来源和未解决事项。不得按多数票认定事实；冲突无法消除时明确说明，失败的角色不可包装成成功。你不读取其他Agent的私有轨迹，也不编造证据。',
                  'registered_tools':['query_memory','read'],'input':{'intent_ids':[i for r in results for i in r['intent_ids']],'guidance':'审核并融合正式角色结果，给出一个最终回答。'}}
        budget=budgets[(req.user_id,req.conv_id,req.turn_id)]
        try:
            result=await service.execute(req,prepared,state['public'],budget,collaboration_results=results)
        except BaseException as exc:
            service.memory.abort_turn(req,'cancelled' if isinstance(exc,asyncio.CancelledError) else 'failed')
            budgets.pop((req.user_id,req.conv_id,req.turn_id),None)
            raise
        return {'final_result':result}

    async def commit(state):
        req=state['request']; final=state['final_result']
        service.memory.finish_turn(req,final['answer_text'],state='completed' if final['execution_status']=='completed' else 'incomplete')
        budgets.pop((req.user_id,req.conv_id,req.turn_id),None)
        service.maintain(req)
        return {'answer':final['answer_text']}

    builder=StateGraph(ChatState)
    for name,fn in [('receive',receive),('recognize_and_prepare',recognize_and_prepare),('run_agents',run_agents),('compose',compose),('commit',commit)]: builder.add_node(name,fn)
    builder.add_edge(START,'receive')
    builder.add_conditional_edges('receive',lambda state:'cached' if state['cached'] else 'new',{'cached':END,'new':'recognize_and_prepare'})
    builder.add_edge('recognize_and_prepare','run_agents')
    builder.add_edge('run_agents','compose')
    builder.add_edge('compose','commit')
    builder.add_edge('commit',END)
    return builder.compile()
