"""Native OpenAI-compatible chat transport. No parsing text into executable calls."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx


class ModelServiceError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleModel:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    client: httpx.AsyncClient | None = field(default=None,repr=False)
    max_output_tokens: int = 2048
    timeout_s: float = 60

    def __post_init__(self):
        parsed=urlsplit(self.base_url)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('LLM_BASE_URL must be HTTPS without embedded credentials')
        if not self.api_key or not self.model:
            raise ValueError('LLM settings incomplete')

    async def complete(self,messages,tools,*,allow_tools=True):
        payload={'model':self.model,'messages':messages,'max_tokens':self.max_output_tokens,'temperature':.2}
        if tools:
            payload.update(tools=tools,tool_choice='auto' if allow_tools else 'none')
        elif not allow_tools:
            # Compatible endpoints accept tool_choice=none even without tools.
            payload['tool_choice']='none'
        async def request(client):
            try:
                reply=await client.post(self.base_url.rstrip('/')+'/chat/completions',json=payload,
                                        headers={'Authorization':'Bearer '+self.api_key},timeout=self.timeout_s)
                if reply.status_code!=200:
                    raise ModelServiceError(f'模型服务请求失败（HTTP {reply.status_code}）。')
                if len(reply.content)>2*1024*1024:
                    raise ModelServiceError('模型响应超过接收上限。')
                body=reply.json()
                choices=body.get('choices')
                if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):
                    raise ModelServiceError('模型响应结构不符合协议。')
                choice=choices[0]
                return {'message':choice.get('message'),'finish_reason':choice.get('finish_reason'),'usage':body.get('usage')}
            except asyncio.CancelledError:
                raise
            except ModelServiceError:
                raise
            except Exception:
                raise ModelServiceError('模型服务网络或响应协议异常。') from None
        if self.client is not None:
            return await request(self.client)
        async with httpx.AsyncClient(trust_env=False) as client:
            return await request(client)


class RoleModelRouter:
    def __init__(self,values,*,client=None):
        self.values=values; self.client=client

    def for_role(self,role):
        model=self.values.get(f'AGENT_{role.upper()}_MODEL') or self.values.get('LLM_MODEL')
        return OpenAICompatibleModel(self.values.get('LLM_BASE_URL',''),self.values.get('LLM_API_KEY',''),model or '',client=self.client)
