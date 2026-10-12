# SPDX-License-Identifier: Apache-2.0
"""Real-forward check of the Qwen3_5Attention plain-RoPE patch.

A tiny random mlx-vlm Qwen3_5Attention runs a text-only prefill + decode
against the KV cache, patched and unpatched. On text-only positions plain
RoPE and mRoPE are the same rotation, so the outputs must match; the patch
counter proves the plain path actually ran. This would have caught the
mlx-vlm 0.7 upgrade breaking the patch (it imported helpers 0.7 removed).
"""

import importlib.util
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from omlx.patches import qwen3_5_attention as patch_mod

ARGS = SimpleNamespace(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    attention_bias=False,
    rms_norm_eps=1e-6,
    max_position_embeddings=4096,
    rope_parameters={
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000,
        "mrope_section": [2, 1, 1],
    },
)


def _fresh_language_module():
    """An unpatched copy of mlx_vlm.models.qwen3_5.language."""
    spec = importlib.util.find_spec("mlx_vlm.models.qwen3_5.language")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(attn, cache_cls):
    mx.random.seed(3)
    prompt = mx.random.normal((1, 7, ARGS.hidden_size))
    steps = [mx.random.normal((1, 1, ARGS.hidden_size)) for _ in range(3)]
    cache = cache_cls()
    outs = [attn(prompt, mask="causal", cache=cache)]
    for x in steps:
        outs.append(attn(x, mask=None, cache=cache))
    return mx.concatenate(outs, axis=1)


def test_patched_attention_matches_unpatched_on_text_only():
    from mlx_vlm.models.cache import KVCache

    stock_mod = _fresh_language_module()
    mx.random.seed(0)
    stock = stock_mod.Qwen3_5Attention(ARGS)
    mx.eval(stock.parameters())

    from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention

    assert patch_mod.apply_qwen3_5_attention_patch() is True
    patched = Qwen3_5Attention(ARGS)
    patched.load_weights(tree_flatten(stock.parameters()))

    before = patch_mod._call_counter["plain"] + patch_mod._call_counter["forced"]
    want = _run(stock, KVCache)
    got = _run(patched, KVCache)
    after = patch_mod._call_counter["plain"] + patch_mod._call_counter["forced"]

    assert after - before == 4  # prefill + 3 decode steps took the plain path
    assert mx.allclose(got, want, atol=1e-4).item(), mx.abs(got - want).max()
