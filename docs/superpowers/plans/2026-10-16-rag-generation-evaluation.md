# RAG Generation Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline in the current session. Steps use checkbox syntax. Preserve the user's existing execution preference: implement directly in this chat, with one final review rather than per-task implementer agents. This plan still requires user review before implementation.

**Goal:** Evaluate fixed-context RAG answers for the existing B3 and no-JEV B4 retrieval experiments with official RAGAS metrics and EvalScope reports.

**Architecture:** Validate and join immutable retrieval artifacts to source-backed reference drafts; generate answers without exposing references; score the saved four-field records with RAGAS; use EvalScope to report actual saved scores. Keep raw collection separate from scoring so failures can be inspected and reports can be regenerated without paid calls.

**Tech Stack:** Existing application Conda Python; project-local `.venv-evaluation`; EvalScope 1.13.0; RAGAS 0.4.x compatible with the installed EvalScope integration; current Qwen generation/judge and embedding endpoints. Resolve and pin the tested RAGAS patch version during Task 3, without upgrading the application environment.

**Spec:** `../specs/2026-10-16-rag-generation-evaluation-design.md` (user approved).

## Global Constraints

- RAG evaluation remains two parts: retrieval and generation. This plan implements generation plus context diagnostics, not full Agent evaluation.
- No JEV requests, credentials checks, client creation, or balance usage.
- Sources are frozen existing B3 and no-JEV B4 rankings; do not rerun retrieval, rebuild indexes, or replace returned evidence with qrel positives.
- Two configurations × 30 queries = 60 answers. First smoke: apc_amount.zh/en in both configurations = 4 answers.
- Generator receives only the question, actual evidence and shared answer instructions. Reference answers and qrels are grader-only data.
- Same generator model, prompt, temperature 0.2 and max output 2048 tokens across both configurations; use the existing provider defaults explicitly. Record actual model identity and finish reason.
- Judge uses the same configured Qwen model initially; record self-judge limitation. Embedding uses the current configured embedding endpoint; never silently download a substitute model.
- Generation concurrency 2; per model request timeout 60 seconds; at most 2 generation attempts inside 125 seconds, with exponential backoff plus full jitter for retryable service/network failures only. Invalid/empty/truncated outputs are not silently retried as successes.
- Judge/SDK automatic retry budgets must be explicit and non-multiplicative: client max_retries=0, RAGAS RunConfig max_retries=1 initially. Verify actual behavior against the resolved version before full batch.
- No raw API credentials in any manifest, TaskConfig serialization, terminal output or generated report.
- Every expected sample remains accounted for. Unavailable judge scores are null with reason and effective sample counts, never fabricated zero/high scores or silently dropped samples.
- New output directory per generation/scoring run. Preserve retrieval artifacts and old runs; no Git repository exists, so retain a progress ledger instead of attempting commits/worktrees.

## Review Focus

1. Reference leakage: references or labels must not reach generator messages, including in metadata dumps.
2. Context drift: evidence text/IDs/order must match the frozen source run; a correct qrel document must not replace a missed retrieval result.
3. Language and coverage: both paired languages and all expected sample IDs must survive Windows encoding and partial execution.
4. Judge failure: null/NaN/errors and generation failures must retain their samples, status and denominator information in summaries and visual reports.
5. Provider safety: no JEV calls, no credential serialization, no layered hidden retries or unintended local model downloads.

## File Structure

- Create `evaluation/rag/generation_data.py`: immutable source joins and generator-only message construction.
- Create `evaluation/rag/generate.py`: bounded answer collection CLI and append-only raw records.
- Create `evaluation/rag/ragas_score.py`: official RAGAS adapter, per-metric status and summary generation.
- Create `evaluation/rag/generation_report.py`: EvalScope numeric-score replay, language/configuration reports and limitations.
- Create `evaluation/requirements-generation.txt`: tested exact additional dependency pins.
- Create `tests/test_rag_generation_data.py`, `tests/test_rag_generation.py`, `tests/test_rag_generation_scoring.py`.
- Extend `evaluation/rag/README.md` with commands, field definitions and interpretation limits; do not rewrite retrieval modules or runtime logic.

## Task 1: Join frozen evidence and references without leakage

**Consumes:** `evaluation.rag.score.load_run(run_dir)` returns `(manifest, queries, qrels, rankings)` after hash/coverage validation; the frozen dataset's annotations carry reference_zh/reference_en.

**Produces:** `prepare_records(sources: list[tuple[str, Path, str]]) -> list[dict]`, with tuples `(config_id, run_dir, baseline)`; `build_generation_messages(record: dict) -> list[dict]`.

Records contain `sample_id=config_id+':'+query_id`, query_id, config_id, language, user_input, retrieved_contexts, reference, evidence, source-run/dataset hashes and semantic_validation. References stay separate from generator messages. Actual empty evidence remains empty rather than being filled from labels.

- [ ] Write tests: assert exactly two configuration records for each source query, unchanged evidence order and text, stable IDs, correct bilingual reference selection, and preservation of `skipped_by_user` for B4.
- [ ] Write tests rejecting missing reference, duplicate sample IDs, mismatched evidence ID/body and changed source manifest; assert empty retrieved_contexts is not replaced by gold support.
- [ ] Write reference-leak test with `REFERENCE_ONLY_SENTINEL` solely in reference metadata; assert it is absent from serialized generator messages while actual evidence is present.
- [ ] Run with app Python: `-X utf8 -m unittest discover -s tests -p test_rag_generation_data.py -v`; expected RED for missing implementation, then implement and rerun GREEN.
- [ ] Run the full app suite; record the command/result in this plan's progress section. Do not modify exported retrieval data.

## Task 2: Collect bounded, traceable model answers

**Consumes:** Task 1 records/messages and `RoleModelRouter(...).for_role('general')` using the existing provider interface.

**Produces:** `collect_answer(record: dict, model, *, deadline_s=125) -> dict`; CLI `python -m evaluation.rag.generate --live --output <new-dir> [--smoke]`.

The answer record adds response, generation_status (`ok`/`failed`/`incomplete`), failure reason, finish_reason, attempts, usage, usage_complete, elapsed time and generator model. Save raw completed records incrementally. The manifest identifies both source runs, prompt hash, resolved models, parameters and expected IDs.

- [ ] Write tests exercising real collector logic with a deterministic transport fixture: normal text; empty response; finish_reason=length; unexpected tool calls; unavailable provider; and cancellation. Assert references never enter outbound payloads.
- [ ] Write tests for a retryable temporary failure followed by success: exactly two attempts; delay bounded by exponential full-jitter policy and total deadline. Permanent failures must not be blindly retried. Keep usage uncertainty explicit.
- [ ] Write tests that existing output directories are rejected before paid calls, partial records retain IDs, and no JEV setup is invoked even with invalid/absent JEV credentials.
- [ ] Run `-X utf8 -m unittest discover -s tests -p test_rag_generation.py -v`; expected RED, implement minimal collector/CLI, rerun GREEN and full app suite.
- [ ] Implement `--smoke` selecting exactly apc_amount.zh/en × two configurations, and full mode selecting all 60 records. No live calls until Task 3's isolated dependencies and evaluator configuration are ready.

## Task 3: Official RAGAS scoring and explicit availability

**Consumes:** Saved answer records and verified four-field data (`user_input`, `retrieved_contexts`, `response`, `reference`).

**Produces:** `score_records(records: list[dict], *, judge, embeddings) -> list[dict]`; per-sample canonical metrics `faithfulness`, `answer_correctness`, `answer_relevancy`, `context_precision`, `context_recall`, plus per-metric status/reason and known judge usage.

Use official RAGAS metric implementations, not custom approximate formulas. Resolve the actual available class names against EvalScope's packaged RAGAS 0.4.x adapter, and persist class names, defaults/weights, prompt language and versions. Do not substitute a semantically different metric under the same label.

- [ ] Inspect/download compatible `ragas>=0.4,<0.5` metadata into the isolated evaluation environment; perform dependency dry-run first. Choose a compatible patch release, pin exact tested additions in requirements-generation.txt, install only there, and run `pip check`.
- [ ] Add a dependency-gated integration test for official metric construction and required fields; app tests may skip these only when optional evaluation packages are absent, and eval-env tests must exercise them.
- [ ] Write adapter tests: preserve expected IDs; distinguish real 0 from unavailable/null/NaN; never evaluate reference-free correctness as valid; keep generation failures visible; reject mismatched source/reference hashes.
- [ ] Write a reporting fixture with three expected samples, two usable scores 1 and 0, one judge failure. Assert total=3, valid=2, unavailable=1, mean=0.5; the failed sample remains in per-sample results.
- [ ] Verify generator/judge API clients are separate roles and never serialized with credentials; capture only safe model/call/status/known-usage metadata. Explicitly disable hidden SDK retries and unnecessary tracing/telemetry.
- [ ] Run `-X utf8 -m unittest discover -s tests -p test_rag_generation_scoring.py -v`; expected RED, implement official adapter and rerun GREEN. If version/API compatibility fails, preserve answers and report the exact blocker instead of fabricating scores.

## Task 4: Smoke, full comparison and EvalScope visualization

**Consumes:** Tasks 1–3 outputs; all source/reference/model/prompt/version identities are frozen.

**Produces:** `python -m evaluation.rag.ragas_score --run <dir>` and `python -m evaluation.rag.generation_report --run <dir>`; raw answers, RAGAS data, per-metric scores, summary JSON, Chinese report and native EvalScope HTML.

- [ ] Write tests that replay uses saved actual scores rather than generating new answers; preserve the original score scale, metric identities, language/configuration partitions, unavailable counts and no-JEV labels. Never represent unavailable scores as valid 0 in native summaries.
- [ ] Create a new smoke run with exactly 4 answers. Inspect real generator outputs, source citations, finish reasons and known usage; do not expose references to generation.
- [ ] Score the smoke with official RAGAS. Separately calibrate supported versus unsupported statement canaries and inspect judge reasoning/status; do not assume every LLM score is a ground-truth decision. A broken judge/embedding interface prevents full-batch expansion.
- [ ] If smoke succeeds, create a new complete run for 60 answers and score it with the same frozen settings. No new JEV calls or retrieval calls. Keep every failed/incomplete/unscored sample with its reason.
- [ ] Generate per-configuration/per-language tables with mean, total, valid, failed/unavailable counts for all five metrics. Main conclusions use faithfulness and answer_correctness, with answer_relevancy/context metrics as diagnostics. Do not infer statistical significance from this small paired set.
- [ ] Verify saved native report metric counts and values against standalone summaries, including unavailable scores; if a native report cannot represent these safely, report that limitation and retain the correct JSON/Markdown rather than presenting misleading visual means.
- [ ] Run full app and eval suites, then one fresh read-only final code review. Fix Important/Critical findings with RED→GREEN regressions. Preserve minor findings in the ledger.
- [ ] Link actual reports and explicitly disclose: fixed-context RAG only, draft reference answers, same-model judge bias, unvalidated B4 rewrites, and any incomplete stage. No project/Agent accuracy or resume-ready claims.

## Progress

- Handoff checkpoint 2026-10-16 14:20+08:00: implementation and code review complete; final app suite 176 OK (11 optional skips), independent evaluation environment 176/176 OK. Full answers 60/60 OK; official scoring still running, 2/60 score records persisted at this checkpoint (each record includes five metric statuses). Do not claim full evaluation complete.
- Running sessions: 7002 = full RAGAS scoring; 55092 = dependent local report/native HTML generation and 60-ID/partition/mean-count parity verification after scoring manifest becomes complete (4h wait ceiling, no paid calls in report stage). Logs: .cache/evaluation/generation-full-scoring.log, generation-full-report.log, generation-full-verification.log. Keep these sessions/artifacts; do not rerun paid generation/scoring into the same directory. Task 4 live acceptance remains pending until the full output and verification log are checked.

- Full answer verification: 60/60 unique expected IDs, all generation_status=ok, 15 samples per configuration/language partition, 135193 known generator tokens with complete usage metadata. Full official RAGAS scoring running with corrected transport. No result/quality claim until final coverage and native parity checks.
- Finishing workflow: workspace reconfirmed not a Git repository; preserve code, ledger and artifacts in place rather than offer inapplicable merge/push/worktree deletion options. No repository initialized and no files removed.

- Final code review: fresh read-only generation_final_review (gpt-6-astra/high), no Critical or Important findings. Full-run results not certified by code review. Final app suite 176 OK/11 optional skips; full eval suite before the last report-only test 175 OK, latest targeted native/report test 4 OK. Native smoke parity verified for 8 available quality metrics, effective counts and per-language means (framework rounds to 4 decimals).
- Final: minor (deferred): README opening still describes retrieval-only/no RAGAS; the new generation section describes the actual added pipeline. No runtime impact; amend introduction in a later documentation pass per review policy.
- Final: Ruling: same-model judging, agent-draft references, English judge prompts and unvalidated B4 remain authorized diagnostic limitations — no official accuracy/production claim — cost: biased/noisy scores require human review.
- Final: Ruling: retain original failed smoke scores — immutable experiment history is more important than a clean-looking smoke — cost: readers must distinguish smoke from the full run and corrected relevancy probe.
- Final: Ruling: reject interrupted run scoring, retain raw records and expected IDs without resumption — fail-closed coverage prevents selected subsets masquerading as complete — cost: interrupted runs need a new directory and additional calls.
- Final: Ruling: provider-sanitized network errors are not blindly retried — identifiable transient failures only — cost: some potentially recoverable generation failures remain failed.
- Final: Ruling: native replay execution success does not mean judge success — replay is labeled and per-metric unavailable diagnostics are separate — cost: native execution-success alone must not be interpreted as quality.
- Final: Ruling: omit entirely unavailable quality aggregates rather than fabricate zero — availability remains in standalone summary and diagnostics — cost: some native quality panels may be absent.
- Final: Ruling: generation failures mark all metrics unavailable, including context diagnostics — one consistent expected-sample accounting policy — cost: context-only diagnostics are not recovered for failed generation rows.
- Final: Ruling: final real-world accuracy and performance are not certified by code review — verify completed artifacts separately and retain stated small-set limitations — cost: results cannot yet support production or resume accuracy claims.

- Task 3: complete. App targeted generation tests 20 OK (5 optional skips); full evaluation suite 175 OK. Score loading validates answer/reference hashes and preserves expected IDs. Official HTTP-fixture tests prove no hidden repair requests, concurrent settings isolation, and 3 samples via n=1.
- Task 4: original smoke scored 4/4 records, retaining endpoint n=3 failures and transient timeouts. Corrected live relevancy probe scored 0.7924620374909241 with exactly 3 judge calls; all five official metric and embedding paths are now live-verified. Supported/unsupported canary 1/0. Ruling: expand after interface correction without requiring every stochastic API request to succeed — availability counts capture transient failures rather than cherry-picking a perfect smoke — cost: full run may have incomplete metric coverage, explicitly reported.
- Full 60-answer generation started in new 20261016-full-01; original smoke and failed scores retained. No JEV. Full run uses corrected per-row clients and n=1 relevancy sampling.

- Task 3: 7 scoring tests now pass in evaluation environment (including actual HTTP-fixture official metric calls, no hidden repair, and concurrent-row sampling isolation). App suite 168 OK/6 skips before latest additions; eval suite 169 OK at prior checkpoint. RAGAS compatibility pin 0.4.1 verified with pip check.
- Task 4: 4/4 smoke answers generated successfully, references isolated. Official canary returned supported=1.0 and unsupported=0.0, all 4 HTTP calls successful. First smoke scoring exposed endpoint rejecting n=3 for relevancy; preserved failed scores, added regression RED→GREEN, use three separate n=1 requests with the same official strictness=3 (transport-only change, no metric substitution). Original scoring is retained as diagnostic, corrected score run will use a new directory.
- Task 4: 3 report tests RED→GREEN in evaluation environment; native replay omits unavailable quality keys and records coverage, with independently checked 1/0/null => mean .5, valid 2, expected 3.

- Task 1: complete. Frozen preparation produces 60 records from actual evidence, no gold replacement; 4 data tests RED→GREEN; app suite after data task 159 tests OK (5 optional skips).
- Task 2: complete. Five collector tests RED→GREEN, combined data/collector 9 passed; conservative retry only for identifiable transient errors, bounded 2 attempts. CLI writes each answer before batch completion; application/provider untouched.
- Task 3: in progress. RAGAS 0.4.3 installed only in evaluation venv; pip check passed. Three availability/accounting tests RED→GREEN in app; optional official integration exposed upstream langchain-community 0.4.2 deleting vertexai, which RAGAS imports unconditionally. Ruling: pin community 0.4.1 in isolated environment rather than patch framework imports — preserve official RAGAS implementation — cost if wrong: integration remains blocked, no paid full run.
- Task 3: Ruling: use pinned official legacy metric classes to preserve requested definitions (correctness 0.75 factuality + 0.25 embedding, relevancy strictness 3); explicitly disable SDK retries and prompt repair calls — bounded cost and auditable semantics — cost: malformed judge JSON is unavailable rather than silently repaired. Default English judge prompts apply to both languages and are hashed.

- 2026-10-16: User approved design/Qwen-as-judge, reviewed the plan, then explicitly said “写”. Execution approved inline. No additional implementation approval is needed.
- Pre-flight: Task 1 sample_id and evidence/reference hashes feed Tasks 2–4 unchanged; Task 2 statuses distinguish failed/incomplete/ok; Task 3 metric availability must survive Task 4 replay. Ruling: native visualization may use a separate coverage/status report when a quality score is unavailable; never coerce null into a quality zero.
- Workspace: previously verified not a Git repository and no applicable AGENTS.md; work additively in place with this retained ledger, no worktree/commit operations.
- Existing inspected interface: EvalScope 1.13.0 packaged RAGAS arguments/task_template supports 0.4.x four-field evaluation with OpenAI-compatible judge/embedding factories. Its ordinary backend writes score JSON; report replay will be added only if needed to expose the same actual scores without losing availability accounting.
- Existing retrieval sources are immutable and already include the required parent evidence and source-backed reference drafts; generation must not redo or contaminate those experiments.
