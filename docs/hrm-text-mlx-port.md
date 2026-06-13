# HRM-Text -> MLX port (design + plan)

Status: design. Goal: serve `sapientinc/HRM-Text` (Apache-2.0, hierarchical
recurrent LM) on Apple Silicon through flyto, and run a tiny from-scratch
training smoke in MLX before any cloud-scale pretrain. Target use case:
internal narrow-domain, strong-reasoning workloads (the regime HRM-Text is
built for: reasoning / task-execution, knowledge supplied externally).

Spec verified against the real 1B checkpoint config.json (vocab 65536,
hidden 1536, 16 layers/module, H_cycles 2 / L_cycles 3, head_dim 128,
embedding_scale 39.19, prefix_lm true, tie_word_embeddings false) and the
reference source at github.com/sapientinc/HRM-Text@main.

## Feasibility verdict

Portable, and the only CUDA-bound piece is a non-issue. The architecture is
plain standard ops (parameterless RMSNorm, sigmoid-gated attention, NeoX
RoPE, SwiGLU, two-module H/L recurrence) -- all native in MLX. The one
genuinely CUDA-specific file (flash_attention_prefixlm_v2.py, FlashAttn3) is
just an attention-mask trick and is replaced by one MLX
scaled_dot_product_attention call plus an additive mask. Note: the paper's
"MagicNorm" is NOT in the released code -- it is parameterless RMSNorm.

The real cost is inference speed: each generated token re-runs the full
recurrence = 8 transformer passes (2 H + 6 L) with 8 separate KV-cache
stacks. Correct and deterministic, just ~8x a same-width plain transformer
per token. Inherent to HRM, not optimizable away (ACT is future work, not in
the release).

## Components to build (MLX)

All no-bias. Per-module layer count = config num_hidden_layers (already the
halved per-module count).

1. Layers
   - `rms_norm(x)` parameterless: `x * rsqrt(mean(x^2,-1) + 1e-6)`, no scale.
   - RoPE (NeoX rotate_half), computed in fp32, base 10000, dim = head_dim.
   - Sigmoid-gated attention: one fused `gqkv_proj`
     `[(2*nh + 2*nkv)*head_dim, hidden]` split into `(gate, q, k, v)` along
     the head axis in order `(nh, nh, nkv, nkv)` (nkv == nh in all configs).
     RoPE on q,k. `out = sigmoid(gate) * SDPA(q,k,v, scale=1/sqrt(head_dim))`,
     then `o_proj`. The gate is a parallel head-shaped sigmoid, NOT a function
     of the attention output.
   - SwiGLU MLP: fused `gate_up_proj` `[2*intermediate, hidden]` -> split ->
     `down_proj(silu(gate) * up)`. intermediate = find_multiple(round(4*hidden*2/3), 256).
   - Pre-norm TransformerBlock: `x += attn(rms_norm(x)); x += mlp(rms_norm(x))`,
     final `rms_norm` at the end of each module's stack.

2. HRM recurrence (LM head reads final z_H)
   ```
   z_H = embed(input_ids) * embedding_scale      # scale applied at runtime, NOT folded into weights
   z_L = z_L_init                                 # learned [hidden] vector, broadcast
   for i in range(H_cycles):       # 2
       for _ in range(L_cycles):   # 3
           z_L = L_module(z_L + z_H)   # additive input injection
       z_H = H_module(z_H + z_L)
   logits = lm_head(z_H)              # untied Linear, no bias
   ```

3. PrefixLM mask (prefill only; decode at seqlen 1 needs no mask)
   For a sequence of length T with prefix length P (instruction through
   `eoq`):
   - prefix rows `i < P`: attend `j < P` only (bidirectional over prefix, NOT
     causal, never sees response).
   - response rows `i >= P`: attend `j <= i` (causal over the whole sequence
     incl. prefix).
   Build the additive 0/-inf mask, pass to SDPA. Block-diagonal across packed
   sequences if packing is used (not needed for single-request serving).

4. KV cache: 8 stacks keyed by recurrence step -- H[0], H[1] and
   L[0..5] -- each `num_hidden_layers` deep. Decode re-runs all 8 passes on
   the single new token; prefill fills all 8 over the prompt.

5. Weight loader: map released safetensors. Names (HF export):
   `model.embed_tokens.weight`, `model.z_L_init`,
   `model.{H,L}_module.layers.{i}.attn.{gqkv_proj,o_proj}.weight`,
   `model.{H,L}_module.layers.{i}.mlp.{gate_up_proj,down_proj}.weight`,
   `lm_head.weight`. No biases, no norm weights (parameterless), no rotary
   buffers (regenerate). Split fused gqkv along output axis. Apply
   embedding_scale at runtime.

6. Tokenizer: HF fast tokenizer shipped with the checkpoint. Special tokens
   boq/eoq (prefix boundary) / eoa (stop). Prompt layout:
   `{boq}{condition_tokens}{prompt}{eoq}`, generate until `eoa`.

## Phases

- P0 -- standalone MLX forward + logit parity (the de-risk gate).
  Implement layers + recurrence + PrefixLM prefill in a standalone module.
  Load the released 1B weights. Compare MLX logits vs the HF reference (torch
  CPU, a few tokens) -- require near-exact match (bf16 tolerance). Nothing
  downstream is trustworthy until this passes.
- P1 -- generation: prefill + decode loop, 8-cache, greedy + temp/Gumbel
  sampling, stop on eoa. CLI smoke producing coherent text. **DONE** (see
  "P1 results" below).
- P2 -- flyto engine: a new engine type (standalone, NOT mlx-lm -- the
  PrefixLM + recurrence + per-token full-recurrence do not fit mlx-lm's
  causal-KV generate). model_discovery recognizes model_type "hrm_text";
  serve via OpenAI /v1/chat|completions like the other engines; lease/admission
  as usual. Inference is slower per token -- size the lease and document it.
  **DONE** (see "P2 results" below).
- P3 -- tiny from-scratch training smoke (correctness, not scale).
  Minimal MLX training loop: task-completion loss (NLL over response only) +
  PrefixLM mask, a B-shrunk config (e.g. hidden 128, 1 layer/module, vocab
  256, seq 128), ~1000 examples. Gate: loss decreases and the model overfits
  a tiny held-in set. This validates the MLX training recipe is correct
  before spending on a cloud H100 run. Full-unroll backprop through the 8
  passes is fine at tiny scale; the bp-warmup truncated-BPTT trick can be
  added later for fidelity at scale.

## P1 results (generation, done on m2max)

Scratch code in `~/Code/hrm-mlx/` (not in flyto until P2): `hrm_mlx.py` gained a
`KVCache` (128 slots = 8 recurrence steps x 16 layers) and `forward_cached`;
`generate.py` is the prompt build + sampling + decode loop + CLI;
`verify_gen.py` is the gate harness.

Why per-step KV caching is valid (the non-obvious part): every one of the 8
passes is causal over committed positions (the PrefixLM prefix block is
bidirectional only within itself and never attends the response), so a committed
token's K,V at each recurrence step are independent of future tokens. Decode
feeds one new token, re-runs all 8 passes attending each step's cached history,
appends one K,V per slot; all 128 slots advance in lockstep so a single `offset`
drives RoPE. This is exactly transformers' DynamicCache + cycle_offset scheme.

Verification (all pass, fp32 unless noted):
- Gate A (mechanism, pure MLX): cached-decode logits == a from-scratch no-cache
  recompute over the grown sequence, position for position. max|diff| 9e-5,
  top-1 all match. This is a numerical proof the cache equals full recompute.
- Gate B (integration): MLX greedy tokens == transformers `generate()` greedy
  tokens, token for token (exercises transformers' own cached recurrence).
- Qualitative (native bf16): "The capital of France is" -> "Paris" (stops at
  eos); "Write one sentence about the ocean." -> "A man is swimming in the
  ocean."; the sheep riddle in --think mode -> a full chain-of-thought ending in
  `\boxed{8}`, stops cleanly. (The riddle arithmetic is a model-quality miss,
  not a port bug.) temp>0 sampling produces varied coherent output, seed-stable.
- Zero regression: P0 logit parity still max|diff| 0.0000 after the refactor.

Throughput (m2max, bf16, single stream): ~13-22 tok/s. The inherent ~8x
recurrence cost; size the P2 lease accordingly.

Prompt format (authority = the released `HrmTextProcessor` / chat template, NOT
the earlier guess of `<|direct|>`/`<|cot|>`): `<|im_start|>`(6) + condition +
content + `<|im_end|>`(7), the whole prompt is the bidirectional PrefixLM block
(`token_type_ids` all 1). condition = `<|object_ref_start|>`(8) for a direct
answer, `<|quad_end|>`(13)`<|object_ref_end|>`(9) for thinking/CoT. Stop on
eos = `<|box_end|>`(11). The checkpoint ships no chat_template/generation_config.

Note for P2: the released checkpoint is FUSED (`gqkv_proj`, `gate_up_proj`).
`mlx_vlm.models.hrm_text` already implements this whole architecture as nn.Module
and its `sanitize()` splits the fused weights into q/k/v/gate + gate/up. So P2
can either reuse mlx_vlm's HRM-Text engine or keep the standalone functional
path; the standalone is parity-proven and gives full control over lease/admission.

## P2 results (flyto engine, done on m2max, branch feat/hrm-text-engine)

The port is now in the repo (no longer scratch):

- `omlx/models/hrm_text.py` -- the functional model (the parity+generation
  verified core: HrmTextMLX with __call__ + forward_cached, KVCache, masks,
  a loader that handles single or sharded safetensors).
- `omlx/engine/hrm_text.py` -- `HrmTextEngine(BaseEngine)`: a standalone text
  engine, NOT on the mlx-lm BatchGenerator path. All MLX work runs on the single
  global MLX executor thread (`get_mlx_executor`, max_workers=1), one
  `run_in_executor` per token -- true per-token streaming AND other engines
  interleave between our (slow) tokens. Each request owns its own 128-slot
  KVCache, so requests are independent and the executor serializes the compute;
  no continuous batching (inherent to HRM). Sampling reuses mlx-lm's
  `make_sampler` / `make_logits_processors` (temp/top_p/top_k/min_p + rep/
  presence penalty behave exactly like other engines). Prompt rendering mirrors
  the released HrmTextProcessor (the checkpoint ships no Jinja chat template):
  `<|im_start|>`+condition+content+`<|im_end|>`; the PrefixLM mask is built
  internally (whole prompt bidirectional, response causal); stop on eos.
- Wiring: model_discovery detects model_type "hrm_text" -> engine_type
  "hrm_text"; engine_pool maps + instantiates it; engine/__init__ exports it.

Verification (all pass):
- Engine async API: chat "The capital of France is" -> "Paris" (matches the P1
  baseline exactly), streaming concatenates identically, thinking-mode CoT,
  count_chat_tokens correct, penalties/top-k/top-p/min_p and stop-sequence paths
  exercised without error.
- Live HTTP server (isolated test server, /tmp base-path, port 8011):
  /v1/models lists it as hrm_text; /v1/chat/completions (non-stream + SSE
  stream), /v1/completions (raw prompt), and thinking-mode via
  chat_template_kwargs all return correct output with correct usage accounting.
- Tests: tests/test_hrm_text_discovery.py (12 tests, discovery + pool mapping +
  prompt-render + msg-text). Existing tests/test_model_discovery.py +
  tests/test_engine_pool.py still 165/165 (zero regression from the Literal /
  dispatch / detection edits).

Lease / admission: HRM-Text is a synchronous text engine on the standard pool
path (NOT the async media/worker + job-lease path of video/image), so it uses
the same pool memory admission as the VLM engine -- no special lease object.
The cost to document is throughput: ~13-22 tok/s single-stream on m2max (the
inherent ~8x recurrence). `has_active_requests()` is wired so the pool will not
evict the model mid-generation. Prefill over a long prompt runs 8 stack passes
and materializes an [P,P] attention mask per layer; the 1B is small (2.3GB) so
this is not a panic risk on either machine for typical prompt lengths.

Non-goals in P2 (documented, not bugs): no xgrammar structured output (the
recurrence does not fit xgrammar), no tool calling, no continuous batching, and
temp>0 sampling is not seed-reproducible (uses the global mx.random state, like
a standard server). These can be added later if needed.

## Honest costs / non-goals

- Real from-scratch pretrain (B/L/XL on a real corpus) stays a cloud job
  (CUDA + FlashAttn3, 8-16x H100, ~$800-1500, ~2 days). MLX training here is
  ONLY for the tiny correctness smoke, not the real run.
- Inference is ~8x a same-width transformer per token (full recurrence).
  Acceptable for narrow-domain reasoning where quality matters more than
  throughput; document it.
- HRM-Text is a reasoning/task engine, weak on broad factual recall by
  design (40B-token budget). For internal knowledge, pair with retrieval.
- Agent-critical axes (tools, long context, multi-turn robustness, OOD) are
  unevaluated in the paper -- validate on real internal tasks before
  committing.
