# RAG Retrieval Evaluation Implementation Plan

> **For agentic workers:** Execute inline in the current session. User has authorized proceeding; do not request the same execution approval again.

**Goal:** Run a source-grounded bilingual retrieval pilot on the existing IEEE corpus.

**Architecture:** Prepare immutable corpus/query/qrels artifacts, collect rankings through production retrieval helpers, then let EvalScope replay and MTEB score them. Keep online collection and offline grading separate.

**Tech Stack:** Existing Conda runtime; EvalScope 1.13.0, MTEB 2.26.1, CPU PyTorch in an isolated evaluation environment.

**Spec:** ../specs/2026-10-16-rag-evaluation-design.md

## Global Constraints

- No Git repository is present: make path-limited additive changes and retain this ledger instead of running worktree/commit helpers.
- No model/index rebuild, credentials in logs, invented results, or silent baseline substitution.
- @1/@3/@5 are scored over parent IDs with all queries retained, including execution failures.
- Bilingual pairs share qrels. Source labels are provisional, not human-confirmed golden labels.

## Review Focus

- UTF-8 Chinese questions must survive Windows process boundaries.
- Missing/failed baselines must not disappear from the denominator.
- Repeated child hits must not create duplicate parent ranks.
- Query and index model identities must agree before network calls.
- Metrics must score exported rankings rather than starting a different embedding pipeline.

## Tasks

- [x] 1. Prepare 15 source-grounded cases and validate/export full-corpus bilingual dataset; tests cover missing support and duplicate IDs.
- [x] 2. Collect four baseline rankings, persist incremental outputs and failures, validate coverage; tests cover ranking conversion.
- [x] 3. Integrate EvalScope replay benchmark and official MTEB metrics; verify a hand-checkable fixture.
- [x] 4. Run the live pilot, offline score, inspect failures, report provisional language-separated results; run relevant regression checks.

## Progress / Decisions

- 2026-10-16: User resumed implementation explicitly, retrieval only; EvalScope retained. Baseline suite: 136 tests passed in the application Conda environment.
- Ruling: Source labels use explicit parent IDs plus verbatim support excerpts, not keyword matching as automatic relevance judgments — avoids accidental labels from topic overlap. Binary relevance only; equivalent sources remain non-exhaustively annotated until review.
- Pre-flight: Task 1 exports parent IDs consumed by Task 2 rankings and Task 3 scoring; all three use `_id` in corpus/queries and `query-id`/`corpus-id` in qrels. Failures remain empty rankings, never omitted queries.
- Worktree check: no Git repository exists. Work in place with additive evaluation files; no worktree or commits can be created. This plan is the retained progress ledger.
- Task 1: complete — 6 dataset tests RED→GREEN; exported 966 parents, 15 pairs/30 queries, 66 positive binary qrels with source excerpts. Labels remain agent_draft and non-exhaustive.
- Task 2: complete — 3 collection tests RED→GREEN; smoke 2 queries and live pilot all 30 queries/120 rankings completed. B1/B2/B3 all normal; B4 all explicitly degraded. Direct diagnostic: rewriter returned 3 candidates; JEV raised JevServiceError. No fallback results described as successful Multi-Query.
- Task 3: integration tests exercise zero-result denominators, MRR/HitRate, hand-computable NDCG, and EvalScope primary metric resolution. Frameworks installed only in .venv-evaluation (EvalScope 1.13.0, MTEB 2.26.1, pytrec_eval-terrier 0.5.10, CPU Torch 2.8.0).
- Ruling: Set framework-specific cache directories inside project .cache/evaluation — MTEB import attempted a non-workspace default directory; keep application state and user directories untouched. Cost: local cache disk usage.
- Ruling: Register benchmark-local EvalScope 1.13 metric semantics in memory and expose cutoff K as an aggregate dimension — custom retrieval metrics were otherwise treated as unscored diagnostics. No installed library files modified; integration is pinned to 1.13.0. Cost: revisit adapter on framework upgrades.
- Task 3: complete — 5 scoring tests passed, including official zero-result denominator and structured EvalScope primary selection; actual EvalScope replay generated all four baseline reports and both language subsets. Attempts with old flat metric names failed during report construction and are retained only as diagnostic logs, not reported as successful runs.
- Task 4: complete — live pilot 120 records validated; REPORT.md, summary.json, scores.jsonl and four native EvalScope report sets generated. All 150 project tests passed in .venv-evaluation in 21.849s. Further isolated JEV diagnostic returned HTTP 502; do not claim full Multi-Query execution was verified. No business answers generated.
- Final review: fresh read-only reviewer found no Critical issues, one Important (dataset exports overwrite a version), and three Minor issues. Important is being fixed with a RED→GREEN regression covering idempotent rebuild and refusal to overwrite changed labels.
- Final: minor (deferred): Record resolved AGENT_GENERAL_MODEL/JEV model overrides in collection provenance, not just fallback LLM_MODEL/JEV_MODEL configuration. Current pilot uses the fallback configuration; an override would require additional provenance care.
- Final: minor (deferred): Include PDF page_start/page_end in annotation source_locator; parent IDs, source files, raw corpus metadata and support excerpts remain available.
- Final: minor (deferred): Derive REPORT.md introductory pair count for --limit smoke runs; current full pilot really contains 15 pairs, and all tables/manifests already show actual counts.
- Final: Ruling: Multi-Query effectiveness and external-service recovery are not established by this run — preserve explicit degraded B4 records; separately retry after JEV recovery. Cost: no valid full-Multi-Query comparison yet.
- Final: Ruling: Source labels are provisional and source-snapshot-specific, not exhaustive or current-policy guarantees — retain agent_draft warnings and require review before public metrics. Cost: current diagnostic scores may change after annotation review.
- Final: Ruling: Cross-version framework support was not tested — keep exact EvalScope/MTEB pins rather than claim broad compatibility. Cost: adapt and re-test before upgrades.
- Final: fixed immutable dataset export — new regression observed RED on unintended rewrites, then GREEN on idempotence/refusal; the current source dataset rebuild is an actual no-op. Historical pilot still validates 120 rows and 30 query IDs.
- Final verification: application environment 151 tests, OK (5 optional framework tests skipped), 15.395s; isolated evaluation environment 151/151 tests passed in 22.761s; pip check reports no broken requirements. Compared 60 native EvalScope metric values with standalone summary within 0.00011 rounding tolerance.
- Finish: no Git repository/branch exists, so retain all implementation, dataset and experiment artifacts in the current project; no merge, push, deletion or workspace cleanup performed.

## User-authorized follow-up: Multi-Query without JEV

- User explicitly reports no JEV balance and requests all rewrite candidates participate without validation. Resume existing retrieval plan with this bounded experimental ablation; no generation evaluation or production-default change.
- Ruling: Use an evaluation-only expanded-recall wrapper and production RRF/rerank/parent expansion, not a fake judge returning true. Keep original plus at most three schema-valid distinct rewrites; record skipped_by_user and a no-JEV model/report label. Cost: possible semantic drift is intentionally unfiltered and must be judged from retrieval results, not claimed guarded.
- Reuse frozen query/qrels and previous B1–B3 results; collect B4 alone into a new run directory to avoid redundant paid baseline calls and preserve the old degraded experiment.
- Tests written first: all rewrites (including an unrelated candidate) are recalled without invoking a forbidden judge; dedup/original retention; malformed rewrites cause explicit fallback; default path still requires JEV. Observed RED before implementation.
- Follow-up implementation: collection tests 7/7 passed; application suite 155 tests OK (5 framework skips), evaluation suite 155/155 passed. Live no-JEV run completed all 30 queries, all expanded, 118 total retrieval queries (30 originals + 88 distinct rewrites), 0 failed, 0 degraded; resolved_jev_model is null. Original pilot remains untouched.
- Provenance improvement in this follow-up: record resolved role rewriter model and resolved JEV model (null for skipped mode), addressing the earlier model-override recording limitation for new runs. Native replay model names now explicitly contain no-JEV and preserve mode in sample/review metadata.
- Follow-up scoring complete: EvalScope/MTEB generated separate no-JEV reports. Overall Recall@5 0.88889 (unchanged vs prior B3), NDCG@5 0.85610 (prior 0.83770). Chinese Recall@5 0.82222→0.88889; English 0.95556→0.88889. These are provisional source-draft results, not evidence of universal improvement or JEV-guarded behavior.
- Follow-up fresh review: no Critical or Important findings. Offline review confirms skipped mode never calls JevSettings.from_env, preserves unrelated distinct rewrites, and production default remains guarded.
- Follow-up minor (deferred): generic source prose in score.py/EvalScope benchmark description still says production rankings; actual no-JEV mode is explicitly experimental and named in row/manifest/native report labels. Treat it as experimental rankings from reused production ranking components, not a tested production no-JEV release.
- Follow-up review limits: no claims of rewrite semantic fidelity, exhaustive qrels, statistical effectiveness or byte-for-byte historical production snapshots; current default guarded behavior and preserved artifact hashes verified.

- Initial live probe reached both dense/BM25 services, but its Chinese stdin was corrupted by the Windows shell and console printing failed on Unicode. This probe is excluded from evaluation. Use UTF-8 source files and -X utf8 for actual collection.
- Official framework versions and custom dataset format were checked against downloaded official wheels and EvalScope documentation.
- Framework installation is isolated from the application environment.
