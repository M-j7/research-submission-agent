"""Artifact dictionary contracts: wall-clock milliseconds; missing usage is null, not zero."""
from typing import TypedDict


class ExecutionRecord(TypedDict):
    case_id: str
    trial_id: str
    group: str
    execution_status: str
    turns: list[dict]
    route: list[dict]
    events: list[dict]
    usage: dict
    latency_ms: float
    stop_reason: str
    error: str | None


class ScoredRecord(TypedDict):
    case_id: str
    trial_id: str
    group: str
    business_outcome: str
    rules: list[dict]
    semantic: dict
    performance: dict
    review_status: str
