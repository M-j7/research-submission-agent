# SDD ledger — plan: docs/superpowers/plans/2026-10-22-local-chat-workbench.md

Execution: native, approved by user 2026-10-22. Spec and full plan read.

Ruling: No Git repository exists; use the approved project directory and this persistent ledger instead of Git worktrees/commit-based skill scripts — no repository initialization or data copying — cost: no automatic Git rollback; keep path-limited edits and verification logs.

Pre-flight interfaces: 1→2 profile argument is explicit; 2→4 bundle owns resource cleanup; 3→4 store owns public snapshots, graph owns core answers; 4→5 jobs map typed exceptions to HTTP; 3/5→6 DTOs shared by contract; 6→7 hook owns subscriptions; 1–7→8 no live paid QA. No conflict found.

Baseline: 259 tests PASS (51.955s), .cache/evaluation/workbench-baseline.log.
Task 1: complete. RED unknown stable profile; GREEN 3 new tests including all 45 frozen candidate payloads; full suite 262 PASS (51.883s), .cache/evaluation/workbench-task1.log.
Task 2: in progress. Factory tests written/run RED; demo real graph runs offline. Test fixture clearing all Windows environment caused SSL construction failure: preserve OS environment while overriding only model settings.
Task 2: complete. 4 factory tests GREEN; full suite 266 PASS (51.376s), workbench-task2.log; CLI demo succeeds. Only approved three nonsecret .env settings changed.
Task 3: complete. 5 store tests RED→GREEN; scoped legacy history, public view, restart interruption verified.
Task 4: complete. 7 jobs tests RED→GREEN; duplicate, capacity, disconnect, failed persistence, restart and slow subscription verified; full-suite log workbench-task4.log.
Task 5: in progress. 5 API tests written and run RED. FastAPI0.159.1 installed with approved network escalation; no model calls.
Ruling: available bundled pnpm will substitute for absent npm, preserving scripts and lockfile reproducibility — avoids global package-manager installation — cost: documented commands use pnpm.
Task 5: complete. 5 API tests RED→GREEN; full suite 283 PASS (61.654s), workbench-task5.log. Loopback/origin checks, sanitized validation errors, real demo graph SSE/history verified.
Task 6: complete. 5 hook tests RED→GREEN: ambiguous POST retry keeps ID, refresh/reconnect no POST, stale callbacks ignored, draft preserved. Typecheck passed during task7 build.
Task 7: complete. 6 UI tests RED→GREEN, total11 frontend tests PASS. React safe Markdown, composition Enter, incomplete and no-evidence states verified; production build succeeds. Removed unused invalid CSS import flagged by build.
Task 8: in progress. Browser QA, final review and documentation remain; no live model calls made.
Task 8 evidence: backend284 PASS (61.879s) before added partial-phase regression; live factory local config valid with network send forbidden (0 calls), stable profile confirmed. Browser checked real demo graph/new/switch/send/refresh/history/actual role status/no evidence. 1440×900 and390×844 layouts have no horizontal overflow; narrow drawers work. Independent test-only8001 fixture verified running refresh, partial result/unresolved/evidence card and failed turn; browser console errors/warnings empty. JS MIME regression6API tests and partial-phase8jobs tests RED→GREEN. Final reviewer /root/workbench_final_review running read-only.
Final review: independent reviewer supplied six concrete findings, then hit account usage limit before final verdict. No second reviewer dispatched; remaining pass is author self-review, not an independent clean signoff. Findings verified against current code before fixes.
Final: fixed unmanaged legacy running state — test_unmanaged_legacy_running_turn_is_not_resumable RED→GREEN, raw records unchanged.
Final: fixed cancellation-before-task-start cleanup — test_cancel_before_job_start_marks_interrupted RED→GREEN.
Final: fixed unsafe route exception propagation — test_unexpected_exception_does_not_escape_to_asgi_logs RED→GREEN.
Final: fixed stale demo/live client mode — backend test_stale_mode_rejected_before_create_or_execution and frontend writes_are_bound_to_mode_returned_by_health/mode_changed_blocks_further_send_until_reload RED→GREEN; no automatic paid replay.
Final: fixed edited-draft loss — accepted_slow_post_does_not_clear_edited_draft RED→GREEN.
Final: fixed stale A→B→A detail overwrite — old_same_conversation_load_cannot_replace_new_turn RED→GREEN.
Final: bounded shutdown and SSE midstream privacy — dedicated regressions observed RED, final GREEN pending below.
Final: minor (deferred): no final independent review verdict due account limit; disclose verification limit rather than claiming clean external approval.
Final verification: 291 backend tests PASS (53.470s), .cache/evaluation/workbench-final.log; all15 frontend tests PASS; TypeScript + Vite production build PASS, no build warnings. Targeted25 web tests PASS, including bounded cancellation and sanitized midstream errors. Each review fix observed RED before GREEN (workbench-review-red.log / workbench-review-stream-red.log and frontend tool output).
Task 8: complete with disclosed independent-review limitation. Latest backend restarted in demo mode; browser reloaded latest assets and successfully sent/received another offline turn. Health/config verification made0 external model requests. Port8001 QA process stopped; normal8000 local workbench remains available for user preview. Screenshot .cache/evaluation/workbench-preview.jpg. Source/DB/index/evaluation data retained; no Git actions or hosted publication.
Final integration: no Git repository, keep approved path-limited local implementation; no branch/PR menu applicable. Persistent ledger retained because there is no commit history to recover it from.

## 2026-10-22 live runtime hotfix

User authorized fixing two diagnosed issues, with no extra paid tests. Production logs showed maintenance coroutine not awaited and all3 saved feedback answers had post-commit web failure. Offline reproduction confirmed sync LangGraph commit ran in an executor without a running event loop. An unauthenticated root HEAD in diagnosis showed direct Perplexity connect timeout vs system-proxy HTTP response; no inference/billing endpoints called.

Fix: async commit node schedules maintenance in the running event loop. Separate owned routing HTTP client follows JEV_TRUST_ENV for recognizer and rewrite judge; Qwen/embedding/RAG data calls keep existing direct client policy. Both HTTP clients close on success/partial startup failure. No key, historical answer or frozen evaluation rewrite; no automatic replay.

Four offline regressions written first and observed failing for proxy/maintenance causes (live-runtime-fix-red.log). GREEN4/4 (live-runtime-fix-green.log); all295 backend tests PASS in57.130s (live-runtime-fix-full.log), no unawaited-coroutine or missing-event-loop warning in full log. Tests use MockTransport/temporary memory and zero live inference. Verification does not establish current provider credit balance or authenticated live availability.
Hotfix deployed: live8001 restarted after confirming0 active core turns and0 active web jobs. New service PID22604 startup completed; GET /api/health reports live/ok and frontend returns200. Startup/verification sent no model requests. Historical feedback answers were retained and not automatically replayed.
