"""One local runtime for CLI and Web. Construction never calls a paid model."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal

import httpx
from dotenv import dotenv_values

from .chat import ChatService
from .memory import MemoryManager
from ..intent.stable_routing_prompts import STABLE_ROUTING_PROMPT_VERSION


class RuntimeConfigurationError(ValueError):
    pass


@dataclass
class RuntimeBundle:
    service: ChatService
    mode: Literal['demo', 'live']
    prompt_version: str
    client: object | None = field(default=None, repr=False)
    redis: object | None = field(default=None, repr=False)
    dependencies: dict = field(default_factory=dict)
    routing_client: object | None = field(default=None, repr=False)
    _closed: bool = False

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.service.close()
        self.service.memory.close()
        try:
            if self.client is not None:
                await self.client.aclose()
        finally:
            try:
                if self.routing_client is not None and self.routing_client is not self.client:
                    await self.routing_client.aclose()
            finally:
                if self.redis is not None:
                    await self.redis.aclose()
        self._closed = True


async def create_runtime(project: Path, *, mode: Literal['demo', 'live'],
                         data_dir: Path | None = None, no_multi_query: bool = False) -> RuntimeBundle:
    if mode not in ('demo', 'live'):
        raise RuntimeConfigurationError('必须明确选择 demo 或 live。')
    project = Path(project)
    if mode == 'demo':
        from .demo import DemoModel, DemoRecognizer
        target = data_dir or project / 'data' / 'web-demo'
        memory = MemoryManager(target / 'memory.sqlite3')
        service = ChatService(target, memory, DemoRecognizer(), DemoModel())
        return RuntimeBundle(service, mode, 'controlled-offline-demo', dependencies={'models': 'offline', 'redis': 'disabled'})

    client = routing_client = memory = redis = service = None
    try:
        from .provider import RoleModelRouter
        from .context import build_runtime_config
        from .maintenance import MemoryMaintenance, ModelMemoryOps
        from .rag import ExistingKnowledgeBackend, MultiQueryRAG, ModelQueryRewriter, JevRewriteJudge
        from .cache import VersionedCache, optional_redis
        from ..intent import JevSettings, JevIntentRecognizer
        from ..intent.embedding import EmbeddingSettings, EmbeddingIntentRecognizer
        from ..intent.ensemble import EnsembleIntentRecognizer

        values = {**dotenv_values(project / '.env', encoding='utf-8-sig', interpolate=False), **os.environ}
        client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
        models = RoleModelRouter(values, client=client)
        model = models.for_role('general')
        config = build_runtime_config(values, project)
        jev_settings = JevSettings.from_values(values)
        # Decisions may need the configured system proxy while Qwen remains direct.
        # An explicitly supplied HTTP client must honor JEV_TRUST_ENV too.
        routing_client = httpx.AsyncClient(trust_env=jev_settings.trust_env, follow_redirects=False)
        jev = JevIntentRecognizer(jev_settings, client=routing_client,
                                 prompt_version=STABLE_ROUTING_PROMPT_VERSION)
        embedding = EmbeddingIntentRecognizer(EmbeddingSettings.from_env(environ=values), client=client,
                                               cache_dir=project / '.cache' / 'intent_embeddings')
        recognizer = EnsembleIntentRecognizer(jev, embedding)
        backend = ExistingKnowledgeBackend(project, values, client=client)
        redis = optional_redis(values.get('REDIS_URL'))
        rag = MultiQueryRAG(backend, cache=VersionedCache(redis),
                            rewriter=None if no_multi_query else ModelQueryRewriter(model),
                            judge=None if no_multi_query else JevRewriteJudge(jev.settings, client=routing_client))
        memory = MemoryManager((data_dir or project / 'data' / 'memory') / 'memory.sqlite3',
                               compressor=ModelMemoryOps(model))
        maintenance = MemoryMaintenance(memory, ModelMemoryOps(model),
                                        profile_every=int(values.get('MEMORY_PROFILE_EVERY') or 10))
        service = ChatService(project, memory, recognizer, models, rag=rag, maintenance=maintenance,
                              prompt=config.prompt, settings=config.settings)
        return RuntimeBundle(service, mode, STABLE_ROUTING_PROMPT_VERSION, client, redis,
                              {'models': 'configured_not_probed', 'knowledge': 'local_index_present',
                               'redis': 'configured_not_probed' if redis is not None else 'disabled'},
                              routing_client=routing_client)
    except BaseException as exc:
        if service is not None:
            await service.close()
        if memory is not None:
            memory.close()
        if client is not None:
            await client.aclose()
        if routing_client is not None:
            await routing_client.aclose()
        if redis is not None:
            await redis.aclose()
        if not isinstance(exc, Exception):
            raise
        raise RuntimeConfigurationError('运行配置或本地知识索引不可用；未发送模型请求。') from None
