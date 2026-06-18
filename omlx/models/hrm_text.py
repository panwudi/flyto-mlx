# SPDX-License-Identifier: Apache-2.0
"""MLX implementation of HRM-Text (sapientinc/HRM-Text).

A hierarchical-recurrent LM: each token re-runs an H/L recurrence of
H_cycles*(L_cycles+1) transformer-stack passes (8 for the 1B), each
num_layers_per_stack deep, with one KV-cache slot per (pass, layer). The
released checkpoint ships FUSED projections (attn.gqkv_proj [gate|q|k|v],
mlp.gate_up_proj [gate|up]); this module loads them directly and splits in the
forward pass. parameterless RMSNorm, sigmoid-gated attention, NeoX rotate_half
RoPE, SwiGLU, pre-norm blocks + a final norm per stack, untied lm_head, no bias.

Verified against transformers' built-in hrm_text on the 1B checkpoint: forward
logits match to max|diff| 0.0000, and the cached decode here reproduces a
no-cache recompute (max|diff| ~9e-5) and matches transformers' greedy generate
token for token. See docs/hrm-text-mlx-port.md.

This is a functional model (weights held as a flat dict of mx arrays), not an
nn.Module: it has no lazy buffers, so it is safe to load and run on a single
shared MLX thread without cross-thread materialization concerns.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx


def rms_norm(x: mx.array, eps: float) -> mx.array:
    # parameterless; compute in float32 then cast back (matches HrmTextRMSNorm)
    xf = x.astype(mx.float32)
    out = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return out.astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return mx.concatenate([-x2, x1], axis=-1)


def rope_cos_sin(positions: mx.array, head_dim: int, theta: float) -> tuple[mx.array, mx.array]:
    # NeoX style: inv_freq over even dims, emb = cat(freqs, freqs). fp32.
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]  # [S, hd/2]
    emb = mx.concatenate([freqs, freqs], axis=-1)  # [S, hd]
    return mx.cos(emb), mx.sin(emb)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, H, S, hd]; cos/sin: [S, hd] -> broadcast over B,H
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return (x.astype(mx.float32) * c + rotate_half(x.astype(mx.float32)) * s).astype(x.dtype)


class HrmTextMLX:
    """Functional HRM-Text. Weights held as a flat dict of mx arrays."""

    def __init__(self, weights: dict[str, mx.array], cfg: dict):
        self.w = weights
        self.cfg = cfg
        self.hidden = cfg["hidden_size"]
        # config.json stores the per-stack count as num_hidden_layers (=16,
        # matching safetensors layers 0..15); the transformers config class
        # renames it to num_layers_per_stack and re-derives an inflated
        # num_hidden_layers. Accept either.
        self.n_layers = cfg.get("num_layers_per_stack") or cfg["num_hidden_layers"]
        self.H_cycles = cfg["H_cycles"]
        self.L_cycles = cfg["L_cycles"]
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.embedding_scale = cfg["embedding_scale"]
        self.scale = self.head_dim ** -0.5
        # One KV slot per unique attention invocation under the recurrent
        # forward: H_cycles * (L_cycles + 1) stacks, each n_layers deep.
        self.n_slots = self.H_cycles * (self.L_cycles + 1) * self.n_layers

    def _attn(self, x, prefix, cos, sin, mask, kv=None):
        # kv = (cache, slot) appends this step's post-rope K,V to that cache slot
        # and attends over the full cached history (decode); None = no cache.
        B, S, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        gqkv = x @ self.w[f"{prefix}.attn.gqkv_proj.weight"].T  # [B,S,(2nh+2nkv)*hd]
        g, q, k, v = mx.split(gqkv, [H * hd, 2 * H * hd, 3 * H * hd], axis=-1)
        q = q.reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, H, hd).transpose(0, 2, 1, 3)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv is not None:
            cache, slot = kv
            k, v = cache.update_and_fetch(slot, k, v)
        # SDPA needs an additive mask no wider than the compute dtype (a fp32
        # 0/-inf mask against bf16 weights would otherwise be rejected). -inf
        # is representable in bf16, so the cast preserves masking semantics.
        if mask is not None and mx.issubdtype(mask.dtype, mx.floating) and mask.dtype != q.dtype:
            mask = mask.astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H * hd)  # [B,S,hidden]
        gate = mx.sigmoid(g.astype(mx.float32)).astype(out.dtype)
        out = gate * out
        return out @ self.w[f"{prefix}.attn.o_proj.weight"].T

    def _mlp(self, x, prefix):
        gu = x @ self.w[f"{prefix}.mlp.gate_up_proj.weight"].T  # [B,S,2*inter]
        inter = gu.shape[-1] // 2
        gate, up = mx.split(gu, [inter], axis=-1)
        h = (gate * mx.sigmoid(gate)) * up  # silu(gate)*up
        return h @ self.w[f"{prefix}.mlp.down_proj.weight"].T

    def _layer(self, x, prefix, cos, sin, mask, kv=None):
        x = x + self._attn(rms_norm(x, self.eps), prefix, cos, sin, mask, kv)
        x = x + self._mlp(rms_norm(x, self.eps), prefix)
        return x

    def _stack(self, x, module, cos, sin, mask, cache=None, slot_base=0):
        # cache None -> no KV cache (parity path). Otherwise layer i uses
        # cache slot (slot_base + i), so each recurrence step owns its own
        # contiguous block of n_layers slots.
        for i in range(self.n_layers):
            kv = (cache, slot_base + i) if cache is not None else None
            x = self._layer(x, f"model.{module}_module.layers.{i}", cos, sin, mask, kv)
        return rms_norm(x, self.eps)  # final_norm (parameterless)

    def __call__(self, input_ids: mx.array, mask: mx.array | None = None) -> mx.array:
        B, S = input_ids.shape
        emb = self.w["model.embed_tokens.weight"][input_ids]  # [B,S,hidden]
        z_H = emb * self.embedding_scale
        z_L = mx.broadcast_to(self.w["model.z_L_init"], (B, S, self.hidden)).astype(z_H.dtype)
        pos = mx.arange(S)
        cos, sin = rope_cos_sin(pos, self.head_dim, self.theta)
        for _h in range(self.H_cycles):
            for _l in range(self.L_cycles):
                z_L = self._stack(z_L + z_H, "L", cos, sin, mask)
            z_H = self._stack(z_H + z_L, "H", cos, sin, mask)
        logits = z_H @ self.w["lm_head.weight"].T
        return logits

    def forward_cached(self, input_ids: mx.array, cache: "KVCache",
                       mask: mx.array | None = None) -> mx.array:
        """Cache-aware recurrent forward for generation.

        Prefill: pass the whole prompt + a PrefixLM mask; fills every slot.
        Decode: pass a single new token + mask=None; re-runs the full recurrence
        on it, attending each pass to that step's cached history.

        Validity: every pass is causal over committed positions (the prefix
        block is bidirectional only within itself and never attends the
        response), so a committed token's K,V at each recurrence step are
        independent of future tokens -- exactly what makes the cache correct.
        """
        B, S = input_ids.shape
        offset = cache.offset  # rope position base for the tokens fed this call
        emb = self.w["model.embed_tokens.weight"][input_ids]
        z_H = emb * self.embedding_scale
        z_L = mx.broadcast_to(self.w["model.z_L_init"], (B, S, self.hidden)).astype(z_H.dtype)
        pos = mx.arange(offset, offset + S)
        cos, sin = rope_cos_sin(pos, self.head_dim, self.theta)
        n = self.n_layers
        for h in range(self.H_cycles):
            for l in range(self.L_cycles):
                base = (h * (self.L_cycles + 1) + l) * n
                z_L = self._stack(z_L + z_H, "L", cos, sin, mask, cache, base)
            base = (h * (self.L_cycles + 1) + self.L_cycles) * n
            z_H = self._stack(z_H + z_L, "H", cos, sin, mask, cache, base)
        cache.advance(S)  # all slots grew by S in lockstep
        logits = z_H @ self.w["lm_head.weight"].T
        return logits


class KVCache:
    """Per-recurrence-step KV cache: n_slots independent (K, V) histories,
    each grown by concatenation. All slots advance in lockstep, so a single
    `offset` (committed length) drives RoPE for every slot.

    Assumes all batch elements share one sequence length/history (single-request
    serving, or a uniform batch). Ragged per-row lengths are not supported.
    """

    def __init__(self, n_slots: int):
        self.k: list[mx.array | None] = [None] * n_slots
        self.v: list[mx.array | None] = [None] * n_slots
        self.offset = 0

    def update_and_fetch(self, slot: int, k: mx.array, v: mx.array):
        if self.k[slot] is None:
            self.k[slot], self.v[slot] = k, v
        else:
            self.k[slot] = mx.concatenate([self.k[slot], k], axis=2)
            self.v[slot] = mx.concatenate([self.v[slot], v], axis=2)
        return self.k[slot], self.v[slot]

    def advance(self, n: int):
        self.offset += n


def causal_mask(S: int, dtype=mx.float32) -> mx.array:
    # additive 0/-inf, [S,S]; broadcasts over B,H in SDPA
    return mx.triu(mx.full((S, S), -mx.inf, dtype=dtype), k=1)


def prefixlm_mask(S: int, prefix_len: int, dtype=mx.float32) -> mx.array:
    """prefix rows (i<P): attend j<P only; response rows (i>=P): attend j<=i."""
    i = mx.arange(S)[:, None]
    j = mx.arange(S)[None, :]
    P = prefix_len
    prefix_row = i < P
    allow = mx.where(prefix_row, j < P, j <= i)
    return mx.where(allow, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))


def _load_weights(model_dir: Path) -> dict[str, mx.array]:
    """Load model.safetensors, or merge all *.safetensors shards if sharded."""
    single = model_dir / "model.safetensors"
    if single.exists():
        return mx.load(str(single))
    shards = sorted(glob.glob(str(model_dir / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors weights in {model_dir}")
    weights: dict[str, mx.array] = {}
    for shard in shards:
        weights.update(mx.load(shard))
    return weights


def load(model_dir: str) -> tuple[HrmTextMLX, dict]:
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    return HrmTextMLX(_load_weights(d), cfg), cfg
