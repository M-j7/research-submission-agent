"""Real Qwen/OpenAI-compatible embeddings with reusable template vectors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

from .templates import (TEMPLATE_PATH, NON_BUSINESS_TEMPLATE_PATH, TemplateIntentResult,
                        load_retrieval_templates, validate_message)

# 遵循当前向量服务的批量限制；75条示例分20、20、20、15请求，不是下载本地模型。
BATCH_SIZE = 20


class EmbeddingConfigurationError(ValueError):
    pass


class EmbeddingProtocolError(ValueError):
    pass


class EmbeddingServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingSettings:
    endpoint: str
    api_key: str = field(repr=False)
    model: str
    timeout_s: float = 90.0

    def __post_init__(self):
        try:
            url = urlsplit(self.endpoint.strip())
            if (url.scheme != "https" or not url.hostname or url.port == 0
                    or url.username or url.password or url.query or url.fragment):
                raise ValueError
            path = url.path.rstrip("/")
            if not path.endswith("/embeddings"):
                path += "/embeddings"
            endpoint = f"https://{url.netloc}{path}"
        except (AttributeError, ValueError):
            raise EmbeddingConfigurationError("EMBEDDING_BASE_URL 必须是有效 HTTPS API 地址，不允许内嵌凭证。") from None
        if not isinstance(self.api_key, str) or not self.api_key.strip() or any(c.isspace() for c in self.api_key):
            raise EmbeddingConfigurationError("请在本地 .env 中配置有效的 EMBEDDING_API_KEY。")
        if not isinstance(self.model, str) or not self.model.strip():
            raise EmbeddingConfigurationError("EMBEDDING_MODEL 不能为空。")
        if (isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise EmbeddingConfigurationError("EMBEDDING_TIMEOUT_S 必须是有限正数。")
        object.__setattr__(self, "endpoint", endpoint)

    @classmethod
    def from_env(cls, env_file: Path | None = None, *, environ: Mapping[str, str] | None = None):
        values = {}
        if env_file is not None:
            if not env_file.is_file():
                raise EmbeddingConfigurationError("找不到项目 .env 文件。")
            values.update(dotenv_values(env_file, encoding="utf-8-sig", interpolate=False))
        values.update(os.environ if environ is None else environ)
        def setting(name, default=""):
            return (values.get(name) or "").strip() or default
        try:
            timeout = float(setting("EMBEDDING_TIMEOUT_S", "90"))
        except ValueError:
            raise EmbeddingConfigurationError("EMBEDDING_TIMEOUT_S 必须是有限正数。") from None
        return cls(setting("EMBEDDING_BASE_URL"), setting("EMBEDDING_API_KEY"), setting("EMBEDDING_MODEL"), timeout)


def _vectors(data, count: int) -> list[list[float]]:
    # 向量长度、维度和数值必须有效；不能把坏响应当正常向量继续分类。
    # 将向量归一化为单位长度，后面两个向量的点积就是余弦相似度。
    if not isinstance(data, list) or len(data) != count:
        raise EmbeddingProtocolError("Embedding 返回向量数量不匹配。")
    output = []
    dimension = None
    for vector in data:
        if not isinstance(vector, list) or not vector:
            raise EmbeddingProtocolError("Embedding 向量必须是非空数值数组。")
        try:
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
                raise ValueError
            values = [float(x) for x in vector]
            norm = math.hypot(*values)
        except (ValueError, OverflowError):
            raise EmbeddingProtocolError("Embedding 向量含非法数值。") from None
        if norm == 0 or not math.isfinite(norm):
            raise EmbeddingProtocolError("Embedding 不能返回零向量或无效范数。")
        if dimension is not None and len(values) != dimension:
            raise EmbeddingProtocolError("Embedding 向量维度不一致。")
        dimension = len(values)
        output.append([x / norm for x in values])
    return output


def _add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: left.get(key, 0) + right.get(key, 0) for key in left.keys() | right.keys()}


class EmbeddingIntentRecognizer:
    def __init__(self, settings: EmbeddingSettings, *, client: httpx.AsyncClient | None = None,
                 cache_dir: Path | None = None, template_path: Path = TEMPLATE_PATH,
                 non_business_path: Path = NON_BUSINESS_TEMPLATE_PATH):
        self.settings = settings
        self._client = client
        self.templates = load_retrieval_templates(template_path, non_business_path=non_business_path)
        self.rows = [(key, text) for key, texts in self.templates.items() for text in texts]
        signature = json.dumps({"version": 1, "endpoint": settings.endpoint, "model": settings.model,
                                "templates": self.rows}, ensure_ascii=False, sort_keys=True)
        # 哈希用于标识“哪套模板、哪个模型与端点”的缓存，不是用哈希伪造语义向量。
        # 模型或模板变化会产生新指纹；同名模型服务端静默升级则不能靠此指纹察觉。
        self._fingerprint = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        self._cache_path = cache_dir / f"{self._fingerprint}.json" if cache_dir is not None else None
        self._template_vectors = None
        self._lock = asyncio.Lock()
        self._cache_warning = None

    async def _embed(self, client: httpx.AsyncClient, texts: list[str]):
        try:
            response = await client.post(self.settings.endpoint,
                json={"model": self.settings.model, "input": texts, "encoding_format": "float"},
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
                timeout=self.settings.timeout_s, follow_redirects=False)
        except httpx.TimeoutException:
            raise EmbeddingServiceError("Embedding 请求超时；未自动重试。") from None
        except httpx.RequestError:
            raise EmbeddingServiceError("无法连接 Embedding 服务；请检查网络、代理和服务地址。") from None
        if response.status_code != 200:
            raise EmbeddingServiceError(f"Embedding HTTP {response.status_code}；未自动重试，未使用字符哈希兜底。")
        try:
            body = response.json()
        except ValueError:
            raise EmbeddingProtocolError("Embedding 未返回有效 JSON。") from None
        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise EmbeddingProtocolError("Embedding 返回向量数量不匹配。")
        indexed = {}
        # API 返回顺序不一定等于输入顺序，必须按 index 还原，否则示例与向量会错配。
        for row in rows:
            index = row.get("index") if isinstance(row, dict) else None
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(texts) or index in indexed:
                raise EmbeddingProtocolError("Embedding 返回的 index 缺失、重复或越界。")
            indexed[index] = row.get("embedding")
        vectors = _vectors([indexed[i] for i in range(len(texts))], len(texts))
        raw_usage = body.get("usage", {})
        usage = {key: value for key, value in raw_usage.items()
                 if key in ("prompt_tokens", "total_tokens", "input_tokens")
                 and isinstance(value, int) and not isinstance(value, bool) and value >= 0} if isinstance(raw_usage, dict) else {}
        return vectors, usage

    def _read_cache(self):
        if self._cache_path is None or not self._cache_path.is_file():
            return None
        try:
            body = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if not isinstance(body, dict) or body.get("fingerprint") != self._fingerprint:
                return None
            return _vectors(body.get("vectors"), len(self.rows))
        except (OSError, ValueError):
            return None  # A derived cache is never evidence; rebuild from the actual service.

    def _write_cache(self, vectors):
        if self._cache_path is None:
            return
        temporary = None
        try:
            # 先写临时文件，再原子替换目标，避免中断时留下半个JSON缓存。
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._cache_path.parent,
                                             prefix=".intent-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"fingerprint": self._fingerprint, "vectors": vectors}, stream, allow_nan=False)
            os.replace(temporary, self._cache_path)
        except OSError:
            self._cache_warning = "模板磁盘缓存无法写入；本次仍使用真实向量，进程内缓存可用。"
        finally:
            if temporary is not None and temporary.is_file():
                temporary.unlink()

    async def _prepare(self, client):
        # 同一识别器首次并发调用时，仅一个任务初始化模板向量；不提供跨进程锁。
        # 优先用内存，再读磁盘，最后才调用真实Embedding服务重新计算。
        async with self._lock:
            if self._template_vectors is not None:
                return self._template_vectors, True, {}
            cached = self._read_cache()
            if cached is not None:
                self._template_vectors = cached
                return cached, True, {}
            vectors, usage = [], {}
            for offset in range(0, len(self.rows), BATCH_SIZE):
                batch, cost = await self._embed(client, [text for _, text in self.rows[offset:offset + BATCH_SIZE]])
                vectors.extend(batch)
                usage = _add_usage(usage, cost)
            # Validate dimensions across batches before accepting or persisting.
            vectors = _vectors(vectors, len(self.rows))
            self._write_cache(vectors)
            self._template_vectors = vectors
            return vectors, False, usage

    async def recognize(self, message: str) -> TemplateIntentResult:
        message = validate_message(message)
        started = time.perf_counter()
        if self._client is not None:
            return await self._recognize(self._client, message, started)
        async with httpx.AsyncClient(follow_redirects=False) as client:
            return await self._recognize(client, message, started)

    async def _recognize(self, client, message, started):
        # 模板向量可缓存；每次新请求仍需为本轮消息生成真实查询向量。
        # 当前不拼历史，所以这一路提供语义相近的示例，而不是会话指代推理。
        vectors, cache_hit, template_usage = await self._prepare(client)
        query, query_usage = await self._embed(client, [message])
        if len(query[0]) != len(vectors[0]):
            raise EmbeddingProtocolError("查询向量与模板向量维度不一致。")
        scores = {key: -1.0 for key in self.templates}
        matches = {key: None for key in self.templates}
        for (key, text), vector in zip(self.rows, vectors):
            # 两边已是单位向量，因此点积=余弦相似度；裁到[-1,1]消除浮点边界误差。
            score = min(1.0, max(-1.0, math.fsum(x * y for x, y in zip(query[0], vector))))
            if matches[key] is None or score > scores[key]:
                # 每类只保留最相近的一条例子，便于检查它是否真的支持该分类。
                scores[key], matches[key] = score, text
        # 最相近不等于相关：即使增加 C10，样本没覆盖的范围外问题也可能更像某类业务。
        # scores 不是正确概率，不能单靠最大值确认业务性质或需求是否清晰。
        return TemplateIntentResult("embedding", "cosine", scores, matches,
            round((time.perf_counter() - started) * 1000, 2),
            {"model": self.settings.model, "template_cache_hit": cache_hit,
             "cache_warning": self._cache_warning, "usage": _add_usage(template_usage, query_usage)})
