# Memory-aware admission + scheduling design

Status: IMPLEMENTED on `feat/mem-admission` (off `main`). Full suite 5204 passed
/ 0 failed / 19 skipped (zero regression vs baseline). Owner deferred the two
product questions to engineering judgment (Q1 huge-prompt pre-reject -> keep the
existing mid-prefill backstop, do NOT build a fuzzy prompt estimator; Q2 pinning
-> memory-safety-first default). Not merged (feat branch, owner reviews).

## What shipped (files)

- `settings.py`: new `memory.memory_admission_headroom_gb` (0.0 = auto).
- `process_memory_enforcer.py`: `get_working_headroom()` (auto = max(transient
  margin, 0.20 x live ceiling)), `prefill_admission_ok()` (pre-stream gate
  mirroring the forward gate), `_cached_ceiling()` (poll-cached ceiling so the
  per-request gate never forks a sysctl subprocess), startup log now prints
  load_headroom + request_margin (fail-loud).
- `engine_pool.py`: `get_engine` reserves generation-model headroom above
  weights; evicts idle LRU to make room; scans `_entries` (not phys) to decide
  a lone model must still load (weights fit) vs a co-residency over-commit
  (transient reject).
- `server.py`: wires `_get_working_headroom`; `_memory_admission_gate()` called
  in `get_engine_for_model` (all LLM endpoints, even cache hits) -> 503 +
  Retry-After pre-stream; `InsufficientMemoryError` now maps 503 (was 507),
  `ModelTooLargeError` stays 507.
- Tests: `test_engine_pool.py::TestWorkingHeadroomAdmission` (7),
  `test_process_memory_enforcer.py::TestWorkingHeadroom` + `TestPrefillAdmissionOk`
  (7), `test_memory_admission_gate.py` (3).

## Problem (2026-07-17 production symptom)

The ugliest failure form is a **mid-stream rejection**: the server already
returned HTTP 200 and the SSE stream has started, then during prefill it emits
an error event and reneges. "Failure after 200" is the hardest form for every
client to handle.

Concrete trigger: two large models co-resident (68 GB qwen3.5-122b + 21 GB
gemma4-moe-26b) blow past the 107.5 GB Metal ceiling. The engine pool admits
both because it only checks model **weights** at load; the over-commit only
surfaces at prefill.

## Root cause chain (confirmed against code)

1. `EnginePool.get_engine` (engine_pool.py:384-421) pre-load admission projects
   `current(phys) + entry.estimated_size <= ceiling`. It reserves only model
   **weights**, no prefill/KV working-set headroom. Two big models fit by weight
   (68+21=89 < 107.5) but leave only ~18 GB for the prefill of BOTH models.
2. The request path acquires the engine (server.py:2323) and then constructs the
   `StreamingResponse` (server.py:2577) -> HTTP 200, SSE opens.
3. Inside the stream the scheduler runs prefill. The phys-based forward gate
   (`_prefill_forward_gate`, scheduler.py:1748-1851) and the chunk-end memcheck
   (`[memcheck:external]`, scheduler.py:2100-2138) raise `RuntimeError` on
   breach, which is caught and converted to a `finish_reason="error"`
   RequestOutput (scheduler.py:5237-5261) -> **mid-stream error event**.

The ceiling itself is live and phys-based: `enforcer.get_final_ceiling()` =
`min(static, dynamic, metal_cap) - video_lease` (process_memory_enforcer.py:517-550).
The dead `memory_monitor` (always None) only disables the estimate-based
preflight; it does NOT affect the ceiling or the phys-based gate.

## Three fixes (owner-mandated direction)

### Fix 2 -- load-time headroom admission (primary, prevents the symptom)

Change the `get_engine` admission projection so a NEW model is admitted only
when the resident set it would co-reside with, plus its weights, plus a
working-set **headroom**, fits under the ceiling. Precisely:

- Main projection stays phys-based: `current(max active,phys) + B.weights +
  headroom <= ceiling`. Evict idle LRU models to make room (pool already does
  this).
- When eviction can no longer proceed, decide "is B alone" by scanning
  `_entries` (`engine is not None and mid != model_id`), NOT by guessing from
  phys (avoids post-unload phys settle lag):
  - No other model loaded -> B is alone -> admit iff `B.weights <= ceiling`
    (reuse the ModelTooLargeError threshold). Single-model serving is never
    blocked by headroom -- honors the "model must be loadable" constraint.
  - Other model(s) still loaded (busy/pinned) -> raise a **transient**
    `InsufficientMemoryError` (there ARE un-evictable others AND adding B breaks
    the headroom budget) -> maps to 503 + Retry-After.
- **Headroom applies only to generation models** (llm / vlm / hrm_text).
  embedding / reranker / stt have small bounded working sets; give them headroom
  0 so they are never mis-evicted or mis-rejected.

**queue vs reject:** we reject (transient 503) + rely on client retry rather than
holding the pool lock to queue server-side (owner assigned the ~20-line client
retry to the flytoAgent side). Server-side queueing under the pool lock is
strictly worse.

Behavior table (headroom=25 GB, ceiling=107.5 GB):
- Load A alone: others=0, 68<=107.5 -> admit.
- Request B, A idle: evict A -> others=0, 21<=107.5 -> admit B alone. No co-residency.
- Request B, A busy: cannot evict A -> others=68, 68+21+25=114>107.5 -> reject B
  (transient). Client retries; when A goes idle the retry evicts A.
- Two small models (10+8): 10+8+25=43<=107.5 -> co-reside. Small models still multiplex.

### Fix 1 -- pre-stream admission gate (make the failure clean)

**Critical: Fix 2 only runs on the LOAD path.** A request hitting an
already-loaded model early-returns in `get_engine` (engine_pool.py:365-375) and
skips admission entirely. So the axis "many concurrent requests to the SAME big
model, KV accumulates, baseline climbs" is structurally outside Fix 2 -- that is
Fix 1's job. The two are orthogonal; Fix 1 is not redundant. This also fixes the
gate location: it MUST run AFTER `get_engine` returns (even on a cache hit) and
BEFORE the stream opens -- not only in the load branch.

**Design decision: Fix 1 is baseline-only (advisor-confirmed).** Between
`get_engine_for_model` and the `StreamingResponse`, check:
`max(active, phys, recent_peak) + working_headroom_reserve > ceiling` -> raise
`HTTPException(503, Retry-After)` before the stream opens. We do NOT build a
prompt-scaled KV estimator: the prompt token count is not cleanly available
pre-template (multimodal/tool prompts approximate badly), and a wrong estimate
false-rejects valid requests -- exactly the inert/wrong-guard failure mode. The
current symptom is baseline-driven (co-residency), which the baseline gate
covers. The single-model + huge-prompt spike keeps the phys-based forward gate
(scheduler.py:1748, already armed with `prefill_transient_margin_gb=12`) as its
mid-prefill backstop. Whether to add pre-stream big-prompt rejection (option B,
with its false-positive risk) is an OWNER question, not a default.

Also: map the transient `InsufficientMemoryError` from Fix 2 to **503 +
Retry-After** (currently 507, server.py:834). Confirmed the ONLY raise site is
engine_pool.py:410 (transient). Genuine `ModelTooLargeError` (engine_pool.py:407,
model alone > ceiling) stays 507 (permanent, not retryable). Retry-After reuses
the existing queue-full convention (`"1"`, server.py:638).

### Fix 3 -- dynamic working-headroom accounting

The ceiling is already dynamic (`min(static, dynamic, metal_cap)`, recomputed
per call). What is static is the headroom NUMBER. Derive it dynamically but keep
it simple to avoid recreating the inert-guard failure mode.

**Anchor to measured data, not a made-up percentage:** production already runs
`prefill_transient_margin_gb=12.0` (the forward gate's spike margin, ~ the
measured glm4.5 single-step transient of 10.6 GB; flyto-kv memory says margin>=14
is reliable). The working headroom should reuse this same signal as its floor,
not an independent guess. Proposed: `headroom = max(prefill_transient_margin_gb,
tier_floor)` resolved from live config, so it tracks the one number ops already
tunes. **Log the resolved headroom + ceiling at startup** so a
disabled/misconfigured guard is never silent (lesson: estimate-guards were inert
AND silent).

## Live smoke (m2max, isolated base-path, custom 8 GB ceiling)

Real `omlx serve` on port 8123 (production :8000 untouched), model gemma4-e2b-4bit
loaded (thin-admit: 4.11 GB < 8 GB), then a chat request:
- non-streaming -> `HTTP 503` + `retry-after: 1` + body "Server memory is at
  capacity (4.11GB of 8.00GB ceiling)".
- streaming (`stream=true`) -> `HTTP 503` + `retry-after: 1` +
  `content-type: application/json` (NOT `text/event-stream`) -> proves the 503
  lands BEFORE the SSE stream opens, i.e. the "mid-stream rejection after 200"
  symptom is gone for this path.

The smoke caught a real bug the mocks missed: the generic `http_exception_handler`
built its JSONResponse without `exc.headers`, silently dropping Retry-After (the
unit test only asserted on the exception OBJECT). Fixed by forwarding
`headers=exc.headers`; regression test added at the handler level.

## Known limitations / follow-ups (surfaced to owner)

- **Mid-stream rejection is not fully eliminated by design.** Fix 2 kills
  co-residency; Fix 1 kills baseline-pressure. A single huge prompt on a clean
  baseline still hits the phys-based forward-gate mid-stream backstop (owner's
  Q1 deferral -- fuzzy prompt pre-estimation was rejected as false-reject prone).
- **Retry-After is misleading when the blocker is pinned.** If model B is
  requested while a PINNED model C is resident and B+C+headroom > ceiling, the
  transient InsufficientMemoryError -> 503 + Retry-After, but a pinned model
  never evicts, so that condition is permanent and the client retries forever.
  No pins exist today and owner deferred pinning; latent, one-line follow-up
  (detect "all blockers pinned" -> 507 permanent instead of 503).

## Guardrails / lessons carried in

- Do NOT resurrect `memory_monitor`; compute estimates from live config.
- Fail loud: startup log of resolved guard state.
- Do NOT block single-model loads via headroom.
- feat branch, no self-merge; bilingual commit/docs; verify active==0 before any
  m5max restart (panic re-run risk -- needs owner go-ahead).
