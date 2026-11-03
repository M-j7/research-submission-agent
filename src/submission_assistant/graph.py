"""LangGraph编排到Agent接收为止；没有回答循环或RAG调用。"""
from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import AgentRequest, create_agent
from .intent.contracts import IntentJudgments
from .intent.ensemble import fuse_results
from .intent.jev import IntentInputError, JevProtocolError, JevServiceError
from .routing import RoutingSettings, route_intents


class AgentGraphState(TypedDict, total=False):
    request: AgentRequest
    recognition: dict
    routing_decision: dict
    prepared_agents: list[dict]


def build_agent_graph(recognizer, *, settings: RoutingSettings | None = None, tool_registry=None):
    settings = settings or RoutingSettings()

    async def recognize_intents(state: AgentGraphState):
        req = state['request']
        # 本地非法输入不消耗模型费用，也不伪装成正常通用回答。
        req = AgentRequest(req.message, req.history, req.request_id)
        try:
            result = await recognizer.recognize(req.message, history=req.history)
            if isinstance(result, IntentJudgments):
                result = fuse_results(jev=result)
            if not isinstance(result, dict):
                raise ValueError('识别接口未返回结果对象。')
            return {'recognition': result}
        except asyncio.CancelledError:
            raise
        except IntentInputError:
            raise
        except Exception as exc:
            # 未知异常可能包含密钥/原始服务正文，不能直接str(exc)。
            safe = str(exc) if isinstance(exc, (JevProtocolError, JevServiceError)) else '意图识别出现内部技术错误，未生成完整判断。'
            return {'recognition': {'judgments': None, 'source_errors': {'recognizer': safe}, 'degraded': True}}

    def route_agents(state: AgentGraphState):
        value = state['recognition'].get('judgments')
        try:
            judgments = IntentJudgments.from_dict(value) if value is not None else None
        except ValueError:
            judgments = None
        return {'routing_decision': route_intents(judgments, settings).to_dict()}

    def prepare_selected_agents(state: AgentGraphState):
        # 路由决定来自同一受检判断和设置，不另写第二套if/else。
        value = state['recognition'].get('judgments')
        try:
            judgments = IntentJudgments.from_dict(value) if value is not None else None
        except ValueError:
            judgments = None
        decision = route_intents(judgments, settings)
        prepared = [create_agent(role).prepare(state['request'], decision, tool_registry)
                    for role in decision.agent_types]
        return {'prepared_agents': prepared}

    builder = StateGraph(AgentGraphState)
    builder.add_node('recognize_intents', recognize_intents)
    builder.add_node('route_agents', route_agents)
    builder.add_node('prepare_selected_agents', prepare_selected_agents)
    builder.add_edge(START, 'recognize_intents')
    builder.add_edge('recognize_intents', 'route_agents')
    builder.add_edge('route_agents', 'prepare_selected_agents')
    builder.add_edge('prepare_selected_agents', END)
    # 未配置持久化记忆、后台调度或自动重试；不假装已完成完整聊天系统。
    return builder.compile()
