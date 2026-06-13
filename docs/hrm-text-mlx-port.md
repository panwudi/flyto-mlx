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

---

# HRM-Text -> MLX 移植 (设计 + 计划)

状态: 设计. 目标: 通过 flyto 在 Apple Silicon 上服务 `sapientinc/HRM-Text`
(Apache-2.0, 层级递归 LM), 并在任何云端规模预训练之前, 先在 MLX 里跑一个极小的
从头训练冒烟. 目标用例: 内部窄域, 强推理的负载 (正是 HRM-Text 的设计场景: 推理
/ 任务执行, 知识由外部提供).

规格已对真实 1B checkpoint 的 config.json 钉死 (vocab 65536, hidden 1536,
每模块 16 层, H_cycles 2 / L_cycles 3, head_dim 128, embedding_scale 39.19,
prefix_lm true, tie_word_embeddings false), 以及 github.com/sapientinc/HRM-Text@main
的参考源码.

## 可行性结论

可移植, 唯一绑定 CUDA 的部分也不是问题. 架构都是标准算子 (无参数 RMSNorm,
sigmoid 门控注意力, NeoX RoPE, SwiGLU, H/L 双模块递归) -- MLX 全部原生支持.
唯一真正 CUDA 专用的文件 (flash_attention_prefixlm_v2.py, FlashAttn3) 只是一个
注意力 mask 技巧, 用一次 MLX scaled_dot_product_attention 调用加一个加性 mask
即可替代. 注意: 论文里的 "MagicNorm" 不在放出的代码里 -- 它就是无参数 RMSNorm.

真正的代价是推理速度: 每生成一个 token 都要重跑整个递归 = 8 个 transformer pass
(2 个 H + 6 个 L), 配 8 个独立 KV cache stack. 正确且确定, 只是每 token 约为同宽
普通 transformer 的 8 倍. 这是 HRM 固有的, 优化不掉 (ACT 是未来工作, 不在本次
发布里).

## 要实现的组件 (MLX)

全部无 bias. 每模块层数 = config 的 num_hidden_layers (已经是折半后的每模块计数).

1. 各层
   - `rms_norm(x)` 无参数: `x * rsqrt(mean(x^2,-1) + 1e-6)`, 无缩放.
   - RoPE (NeoX rotate_half), fp32 计算, base 10000, dim = head_dim.
   - sigmoid 门控注意力: 一个 fused `gqkv_proj` `[(2*nh + 2*nkv)*head_dim, hidden]`
     沿 head 轴按 `(nh, nh, nkv, nkv)` 顺序拆成 `(gate, q, k, v)` (所有 config 里
     nkv == nh). q,k 上 RoPE. `out = sigmoid(gate) * SDPA(q,k,v, scale=1/sqrt(head_dim))`,
     再 `o_proj`. 这个 gate 是 head 形状的并行 sigmoid, 不是注意力输出的函数.
   - SwiGLU MLP: fused `gate_up_proj` `[2*intermediate, hidden]` -> 拆分 ->
     `down_proj(silu(gate) * up)`. intermediate = find_multiple(round(4*hidden*2/3), 256).
   - 前置 norm 的 TransformerBlock: `x += attn(rms_norm(x)); x += mlp(rms_norm(x))`,
     每个模块 stack 末尾再做一次 `rms_norm`.

2. HRM 递归 (LM head 读取最终 z_H)
   ```
   z_H = embed(input_ids) * embedding_scale      # scale applied at runtime, NOT folded into weights
   z_L = z_L_init                                 # learned [hidden] vector, broadcast
   for i in range(H_cycles):       # 2
       for _ in range(L_cycles):   # 3
           z_L = L_module(z_L + z_H)   # additive input injection
       z_H = H_module(z_H + z_L)
   logits = lm_head(z_H)              # untied Linear, no bias
   ```

3. PrefixLM mask (仅 prefill; decode 时 seqlen 为 1, 不需要 mask)
   对长度 T, 前缀长度 P (指令到 `eoq`) 的序列:
   - 前缀行 `i < P`: 只 attend `j < P` (在前缀内双向, 不是因果, 永不看 response).
   - 回复行 `i >= P`: attend `j <= i` (对整个序列因果, 含前缀).
   构造加性 0/-inf mask, 传给 SDPA. 若做 packing 则跨打包序列分块对角 (单请求
   服务不需要).

4. KV cache: 8 个 stack 按递归步 keying -- H[0], H[1] 和 L[0..5] -- 每个
   `num_hidden_layers` 深. decode 对单个新 token 重跑全部 8 个 pass; prefill 在
   整个 prompt 上填满全部 8 个.

5. 权重加载器: 映射放出的 safetensors. 名称 (HF 导出): `model.embed_tokens.weight`,
   `model.z_L_init`, `model.{H,L}_module.layers.{i}.attn.{gqkv_proj,o_proj}.weight`,
   `model.{H,L}_module.layers.{i}.mlp.{gate_up_proj,down_proj}.weight`,
   `lm_head.weight`. 无 bias, 无 norm 权重 (无参数), 无 rotary buffer (重新生成).
   沿输出轴拆 fused gqkv. embedding_scale 在运行时施加.

6. tokenizer: checkpoint 自带的 HF fast tokenizer. 特殊符 boq/eoq (前缀边界) /
   eoa (停止). prompt 布局: `{boq}{condition_tokens}{prompt}{eoq}`, 生成到 `eoa`.

## 阶段

- P0 -- 独立 MLX 前向 + logit parity (去风险的关口). 在独立模块里实现各层 + 递归
  + PrefixLM prefill. 加载放出的 1B 权重. 对比 MLX logits 与 HF 参考 (torch CPU,
  几个 token) -- 要求近乎精确一致 (bf16 容差). 这关不过, 下游都不可信.
- P1 -- 生成: prefill + decode 循环, 8-cache, greedy + temp/Gumbel 采样, 遇 eoa
  停. CLI 冒烟产出连贯文本. **已完成** (见下文 "P1 结果").
- P2 -- flyto 引擎: 一种新引擎类型 (独立, 不走 mlx-lm -- PrefixLM + 递归 + 每
  token 全递归不适配 mlx-lm 的因果 KV 生成). model_discovery 识别 model_type
  "hrm_text"; 像其他引擎一样走 OpenAI /v1/chat|completions 提供服务; 租约/准入
  照常. 每 token 更慢 -- 据此设定租约并写明. **已完成** (见下文 "P2 结果").
- P3 -- 极小从头训练冒烟 (验正确性, 非规模). 极简 MLX 训练循环: 任务完成 loss
  (只对 response 算 NLL) + PrefixLM mask, 一个 B 缩水配置 (如 hidden 128, 每模块
  1 层, vocab 256, seq 128), 约 1000 个样本. 门槛: loss 下降且模型过拟合一个极小的
  held-in 集. 这能在花钱跑云端 H100 之前, 验证 MLX 训练配方是对的. 极小规模下对
  8 个 pass 做全展开反传没问题; bp-warmup 截断 BPTT 技巧可以以后再加以贴合规模.

## P1 结果 (生成, 在 m2max 上完成)

scratch 代码在 `~/Code/hrm-mlx/` (P2 之前不进 flyto): `hrm_mlx.py` 加了一个
`KVCache` (128 槽 = 8 个递归步 x 16 层) 和 `forward_cached`; `generate.py` 是
prompt 构造 + 采样 + decode 循环 + CLI; `verify_gen.py` 是验收关口.

为什么逐步 KV cache 是对的 (非显然的那点): 8 个 pass 中的每一个对已提交位置都是
因果的 (PrefixLM 前缀块只在自身内部双向, 永不 attend response), 所以已提交 token
在每个递归步的 K,V 与未来 token 无关. decode 喂一个新 token, 重跑全部 8 个 pass
并各自 attend 该步的 cache 历史, 每槽 append 一份 K,V; 128 个槽锁步推进, 故单个
`offset` 驱动 RoPE. 这正是 transformers 的 DynamicCache + cycle_offset 方案.

验证 (除注明外全为 fp32, 全部通过):
- Gate A (机制, 纯 MLX): 带 cache 的 decode logits == 在增长后的序列上从头无 cache
  全量重算, 逐位置对齐. max|diff| 9e-5, top-1 全中. 这是 cache 等价于全量重算的
  数值证明.
- Gate B (集成): MLX greedy token == transformers `generate()` greedy token, 逐
  token 一致 (动用了 transformers 自己的带 cache 递归).
- 定性 (原生 bf16): "The capital of France is" -> "Paris" (遇 eos 停);
  "Write one sentence about the ocean." -> "A man is swimming in the ocean.";
  羊圈谜题用 --think 模式 -> 完整链式推理, 收尾 `\boxed{8}`, 干净停止. (谜题算术
  是模型质量的失误, 不是移植 bug.) temp>0 采样产出多样且连贯, seed 稳定.
- 零回归: 重构后 P0 logit parity 仍 max|diff| 0.0000.

吞吐 (m2max, bf16, 单流): 约 13-22 tok/s. 固有的约 8 倍递归代价; P2 租约据此设定.

prompt 格式 (权威 = 放出的 `HrmTextProcessor` / chat 模板, 不是早先猜的
`<|direct|>`/`<|cot|>`): `<|im_start|>`(6) + condition + content + `<|im_end|>`(7),
整个 prompt 是双向 PrefixLM 块 (`token_type_ids` 全 1). condition = 直接回答用
`<|object_ref_start|>`(8), 思考/CoT 用 `<|quad_end|>`(13)`<|object_ref_end|>`(9).
遇 eos = `<|box_end|>`(11) 停. checkpoint 不带 chat_template/generation_config.

P2 备注: 放出的 checkpoint 是 FUSED (`gqkv_proj`, `gate_up_proj`).
`mlx_vlm.models.hrm_text` 已把整套架构实现为 nn.Module, 其 `sanitize()` 把 fused
权重拆成 q/k/v/gate + gate/up. 所以 P2 既可复用 mlx_vlm 的 HRM-Text 引擎, 也可保留
独立 functional 路径; 独立路径已 parity 验证, 且对租约/准入有完整掌控.

## P2 结果 (flyto 引擎, 在 m2max 上完成, 分支 feat/hrm-text-engine)

移植现已进仓 (不再是 scratch):

- `omlx/models/hrm_text.py` -- functional 模型 (parity + 生成均已验证的核心:
  HrmTextMLX, 含 __call__ + forward_cached, KVCache, mask, 以及一个支持单文件或
  分片 safetensors 的加载器).
- `omlx/engine/hrm_text.py` -- `HrmTextEngine(BaseEngine)`: 独立文本引擎, 不走
  mlx-lm BatchGenerator 路径. 所有 MLX 运算都在全局单线程 MLX executor 上跑
  (`get_mlx_executor`, max_workers=1), 每 token 一次 `run_in_executor` -- 真正的
  逐 token 流式, 且其他引擎可在我们 (较慢) 的 token 之间穿插. 每个请求各持有自己的
  128 槽 KVCache, 故请求互相独立, 由 executor 串行化计算; 不做连续批处理 (HRM
  固有). 采样复用 mlx-lm 的 `make_sampler` / `make_logits_processors`
  (temp/top_p/top_k/min_p + 重复/出现惩罚的行为与其他引擎完全一致). prompt 渲染对齐
  放出的 HrmTextProcessor (checkpoint 不带 Jinja chat 模板):
  `<|im_start|>`+condition+content+`<|im_end|>`; PrefixLM mask 内部构建 (整个
  prompt 双向, response 因果); 遇 eos 停.
- 接线: model_discovery 识别 model_type "hrm_text" -> engine_type "hrm_text";
  engine_pool 做映射并实例化; engine/__init__ 导出.

验证 (全部通过):
- 引擎异步 API: chat "The capital of France is" -> "Paris" (与 P1 基线完全一致),
  流式拼接结果一致, 思考模式 CoT, count_chat_tokens 正确, 惩罚/top-k/top-p/min_p
  与 stop-sequence 路径均跑通无错.
- 真实 HTTP server (隔离测试 server, /tmp base-path, 端口 8011): /v1/models 列出
  其为 hrm_text; /v1/chat/completions (非流式 + SSE 流式), /v1/completions (原始
  prompt), 以及经 chat_template_kwargs 的思考模式, 输出均正确, usage 统计正确.
- 测试: tests/test_hrm_text_discovery.py (12 个, discovery + pool 映射 + prompt
  渲染 + msg-text). 既有 tests/test_model_discovery.py + tests/test_engine_pool.py
  仍 165/165 (Literal / dispatch / detection 改动零回归).

租约 / 准入: HRM-Text 是走标准 pool 路径的同步文本引擎 (不是 video/image 的异步
媒体/worker + job-lease 路径), 所以它用与 VLM 引擎相同的 pool 内存准入 -- 没有特殊
租约对象. 需要写明的代价是吞吐: m2max 上单流约 13-22 tok/s (固有的约 8 倍递归).
已接 `has_active_requests()`, pool 不会在生成中途逐出模型. 长 prompt 的 prefill 要
跑 8 个 stack pass 并每层物化一个 [P,P] 注意力 mask; 1B 很小 (2.3GB), 故对典型
prompt 长度而言在两台机器上都不是 panic 风险.

P2 的非目标 (已写明, 非 bug): 无 xgrammar 结构化输出 (递归不适配 xgrammar), 无
工具调用, 无连续批处理, temp>0 采样不可按 seed 复现 (用全局 mx.random 状态, 与
标准 server 一致). 需要时可以以后再加.

## 诚实的代价 / 非目标

- 真正的从头预训练 (B/L/XL 在真实语料上) 仍是云端任务 (CUDA + FlashAttn3,
  8-16x H100, 约 $800-1500, 约 2 天). 这里的 MLX 训练只用于极小的正确性冒烟, 不是
  真正的训练跑.
- 推理每 token 约为同宽 transformer 的 8 倍 (全递归). 对质量比吞吐更重要的窄域
  推理是可接受的; 写明即可.
- HRM-Text 是推理/任务引擎, 按设计在广义事实记忆上偏弱 (40B token 预算). 内部
  知识场景应配合检索.
- agent 关键维度 (工具, 长上下文, 多轮鲁棒性, OOD) 在论文里未评估 -- 投产前先在
  真实内部任务上验证.
