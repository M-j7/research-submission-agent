from typing import Literal
from pydantic import Field, field_validator
from ..runtime.contracts import StrictModel

TurnState = Literal['pending', 'running', 'completed', 'incomplete', 'failed', 'cancelled', 'interrupted']
TERMINAL = frozenset(('completed', 'incomplete', 'failed', 'cancelled', 'interrupted'))


class SendTurn(StrictModel):
    message: str = Field(min_length=1, max_length=16000)
    turn_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')

    @field_validator('message')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('消息不能为空')
        return value


class ConversationSummary(StrictModel):
    id: str
    title: str
    updated_at: str


class TurnView(StrictModel):
    turn_id: str
    message: str
    state: TurnState
    answer: str | None = None
    routing: dict | None = None
    agent_results: list[dict] = Field(default_factory=list)
    status: list[dict] = Field(default_factory=list)
    error_code: str | None = None


class TurnPatch(StrictModel):
    state: TurnState | None = None
    routing: dict | None = None
    agent_results: list[dict] | None = None
    status: list[dict] | None = None
    error_code: str | None = None


class ConversationView(StrictModel):
    id: str
    title: str
    turns: list[TurnView]
