"""Backend-bound identities and validated execution envelopes."""
from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')


class Request(StrictModel):
    user_id: str = Field(min_length=1, max_length=128)
    conv_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=16000)

    @field_validator('message')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('message cannot be blank')
        return value


class ToolError(StrictModel):
    code: str
    message: str
    retryable: bool = False


class ToolResult(StrictModel):
    status: Literal['ok','empty','partial','error']
    data: dict = Field(default_factory=dict)
    partial: bool = False
    next_cursor: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = None

    @classmethod
    def failure(cls, code, message, retryable=False):
        return cls(status='error', error=ToolError(code=code,message=message,retryable=retryable))


class RunResult(StrictModel):
    run_id: str
    agent_type: str
    intent_ids: list[str] = Field(default_factory=list)
    execution_status: Literal['completed','incomplete','failed','cancelled']
    stop_reason: str
    answer_text: str
    evidence: list[dict] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    current_step: int = 0
    tool_calls_used: int = 0
