# Routing history and feedback-only fallback implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Execute in this session; user approved the bounded design.

**Goal:** Route with five complete public QA pairs and keep General feedback-only.

**Architecture:** Separate online state construction from the legacy nine-choice builder. Apply provider-specific limits before network calls; keep the two batches, model and thresholds unchanged. General has no tools or Skill catalog. Scripted tool faults move to a versioned specialist harness, without rewriting frozen runs.

**Tech Stack:** Python, unittest, httpx, LangGraph, SQLite.

**Spec:** User-approved discussion in this chat: five complete final QA pairs, current-message disambiguation, Decisions length correction, no professional QA fallback, offline regression then small live probes.

## Global constraints

- Do not edit .env, production memory/index/Redis or old evaluation runs.
- No full evaluation or RAG rerun; no threshold/model/grouping changes.
- No Git repository exists here: work in place; no Git-only workspace/commit steps.
- Online history must alternate user/assistant complete pairs; malformed/incomplete input is rejected rather than silently sliced.
- Decisions state cap: 256 KiB UTF-8 application guard, not an official token limit; SystemOne retains its old 8000-character gateway guard.

## Review focus

- Six rounds must become the latest five intact rounds, with no caller mutation.
- Long history must reach Decisions through Ensemble prevalidation, but oversized state must fail before paid calls.
- Private tool traces/current pending turn must never enter route history.
- General cannot execute tools even if a model invents tool calls; specialists keep existing access.
- Existing frozen fault runs must remain readable; new harness must still exercise actual tool protections.

### Task 1: Online routing input

**Files:** intent/jev.py, intent/ensemble.py, CLI previews; tests/test_independent_jev.py, tests/test_decisions_provider.py, tests/test_intent_ensemble.py.

- [ ] Write and observe failing full-pair, malformed-history, long-state HTTP and Ensemble-prevalidation regressions.
- [ ] Implement online state builder/provider guard; preserve legacy builder.
- [ ] Run targeted unittest tests.

### Task 2: General feedback boundary

**Files:** agents/roles.py, agents/base.py, runtime/chat.py, intent/catalog.py, evaluation/agent/{faults,dataset}.py and versioned harness metadata; new integration tests.

- [ ] Write and observe failures for General empty permissions/skills and five-public-pair integration.
- [ ] Remove General tools and skill catalog; strengthen current-intent/correction guidance.
- [ ] Move only new scripted fault cases to a specialist using explicit v2 overlay; keep old source/frozen cases unchanged.
- [ ] Run whole unittest suite.

### Task 3: Targeted live verification

**Files:** new evaluation/agent/routing-probes/20261021-* directory and focused probe script.

- [ ] Run at most a small fixed routing-only set and two General feedback checks, recording all judgments, usage and routes. No retries to chase passes.
- [ ] Review changes independently, repair important findings with RED/GREEN tests.
- [ ] Save a Chinese report distinguishing offline correctness, observed live behavior and remaining failures.

## Progress / rulings

- Setup: previous phase investigated root causes; no implementation yet. User has already approved execution.
- Ruling: explicitly reject incomplete online histories, instead of guessing pair boundaries; direct CLI callers must supply completed QA pairs. Legacy offline builder unchanged.
- Ruling: fault v2 overlay changes only new synthetic tool-loop scenarios to submission_requirements; old frozen runs are not migrated. Scores across harness versions must not be treated as role-routing improvements.
- Task 1/2: initial full offline suite 225 tests passed (71.444s); log .cache/evaluation/routing-history-offline-20261021.log. Expected prior RED failures observed for truncation, old limit, General permissions, and harness v2.
- Review fix: overlapping turns exposed interleaved recent public records. New test reproduced user A,user B,answer B,answer A. Project recent records by completed turn order into intact pairs, without rewriting stored raw records.
- Final review: independent read-only review by /root/review_routing_history. Two additional important probe findings (missing actual generation usage, missing General-role guard) fixed after observing both tests fail, then pass. Added coverage for invented General calls and oversized Ensemble input.
- Ruling: live probe uses the same Decisions overrides as the prior full benchmark; local .env currently has unsupported JEV_PROVIDER=jevmodel and model=jev-latest. File is preserved. First attempt stopped locally before HTTP; second output directory records the sole live run. Normal entrypoints still need configuration correction, not silently inferred aliases.
- Live: 8/8 route results complete, 16 native Decisions HTTP200; 2/2 feedback generations complete, 0 tools, 2 Qwen HTTP200. Six directed business probes still miss full expected roles; two control routes match. No repeated paid attempts or threshold tuning.
- Review rulings: live semantics judged from newly saved results, not reviewer speculation; historical-byte identity cannot be proven without original baseline, but no old result writes were made; fault v2 results are protection checks only, not routing improvement. Final whole-suite verification follows.
- Deferred minor: General technical feedback suggested a human adviser; no human handoff channel is actually configured, so this suggestion must not be described as a supported product capability.
- Final verification: 230 tests passed in 53.496s, .cache/evaluation/routing-history-delivery-20261021.log. The suite still emits existing Ragas deprecation and asyncio slow-task diagnostics; no test failures. All tasks completed within the bounded diagnostic scope; remaining semantic routing failures and local config mismatch reported, not claimed fixed.
