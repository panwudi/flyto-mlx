# SPDX-License-Identifier: Apache-2.0
"""Single-sequence decode skips the all-True BatchKVCache mask.

The fast path must only fire when the dropped mask would have been all True,
and it must turn back off whenever padding can exist again.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import BatchKVCache, KVCache

from omlx.patches.batch_decode_mask import apply_batch_decode_mask_patch

H, D = 2, 8


@pytest.fixture(autouse=True, scope="module")
def _patched():
    assert apply_batch_decode_mask_patch()
    assert apply_batch_decode_mask_patch()  # idempotent


def _kv(n, seed=0):
    mx.random.seed(seed)
    return mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D))


def _filled_kvcache(n, seed=0):
    c = KVCache()
    c.update_and_fetch(*_kv(n, seed))
    return c


def _stock_mask(cache, N=1):
    """What unpatched mlx-lm would return for this cache state."""
    from mlx_lm.models.base import create_causal_mask

    return create_causal_mask(N, offset=cache._idx, left_padding=cache.left_padding)


def test_single_unpadded_sequence_gets_no_mask():
    cache = BatchKVCache.merge([_filled_kvcache(5)])
    h = mx.zeros((1, 1, H * D))

    assert create_attention_mask(h, cache) is None
    # The mask it replaces is all True, so None is exactly equivalent.
    assert _stock_mask(cache).all().item()


def test_none_mask_matches_stock_mask_in_attention():
    cache = BatchKVCache.merge([_filled_kvcache(37, seed=1)])
    stock = _stock_mask(cache)
    q = mx.random.normal((1, H, 1, D))
    k, v = cache.update_and_fetch(*_kv(1, seed=2))

    fast = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=None)
    slow = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=stock)
    assert mx.allclose(fast, slow, atol=1e-5).item()


def test_multi_token_step_and_return_array_keep_the_mask():
    cache = BatchKVCache.merge([_filled_kvcache(5)])

    assert cache.make_mask(2) is not None
    assert cache.make_mask(1, return_array=True) is not None
    assert cache.make_mask(1, window_size=4) is not None


def test_padded_batch_keeps_the_mask():
    cache = BatchKVCache.merge([_filled_kvcache(5), _filled_kvcache(3, seed=1)])
    assert cache.make_mask(1) is not None


def test_extend_to_two_sequences_restores_the_mask():
    cache = BatchKVCache.merge([_filled_kvcache(5)])
    assert cache.make_mask(1) is None

    cache.extend(BatchKVCache.merge([_filled_kvcache(3, seed=1)]))

    mask = cache.make_mask(1)
    assert mask is not None
    assert not mask.all().item()  # the shorter row is left padded


def test_filter_back_to_one_sequence_drops_padding_and_mask():
    cache = BatchKVCache.merge([_filled_kvcache(5), _filled_kvcache(3, seed=1)])

    cache.filter([1])  # keep the padded row

    assert cache.left_padding.tolist() == [0]
    assert _stock_mask(cache).all().item()
    assert cache.make_mask(1) is None


def test_padding_from_prepare_or_state_disables_fast_path():
    padded = BatchKVCache([0])
    padded.prepare(left_padding=[2])
    assert padded.make_mask(1) is not None

    right = BatchKVCache([0])
    right.prepare(right_padding=[1])
    assert right.make_mask(1) is not None

    restored = BatchKVCache([0])
    restored.state = BatchKVCache.merge([_filled_kvcache(4)]).state
    assert restored.make_mask(1) is not None


def test_array_left_padding_is_treated_as_unknown():
    cache = BatchKVCache(mx.array([0]))
    assert cache.make_mask(1) is not None


def _tiny_llama():
    from mlx_lm.models import llama

    mx.random.seed(7)
    model = llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=97,
    ))
    mx.eval(model.parameters())
    return model


def test_batched_decode_matches_stock_single_stream(monkeypatch):
    """Greedy tokens from the batched path equal stock mask-free decode."""
    from mlx_lm.generate import BatchGenerator, generate_step

    skipped = []
    patched = BatchKVCache.make_mask

    def counting(self, N, **kwargs):
        mask = patched(self, N, **kwargs)
        if N == 1:
            skipped.append(mask is None)
        return mask

    monkeypatch.setattr(BatchKVCache, "make_mask", counting)
    model = _tiny_llama()
    prompt = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46]
    n = 24

    stock = []
    for tok, _ in generate_step(mx.array(prompt), model, max_tokens=n):
        stock.append(int(tok))

    gen = BatchGenerator(model=model, max_tokens=n)
    (uid,) = gen.insert([prompt], max_tokens=[n])
    batched = []
    while len(batched) < n:
        for r in gen.next_generated():
            assert r.uid == uid
            batched.append(r.token)
    gen.close()

    assert batched == stock
    assert skipped and all(skipped)
