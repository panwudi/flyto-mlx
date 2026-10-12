# SPDX-License-Identifier: Apache-2.0
"""legacy_state / legacy_meta_state reproduce the mlx-lm 0.31 layout.

The prefix cache, boundary snapshots and the paged SSD cache store that
layout, so it must not change when mlx-lm changes what ``cache.state`` means.
"""

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)

from omlx.cache.state_compat import legacy_meta_state, legacy_state
from omlx.cache.type_handlers import SizedArraysCache

H, D = 2, 8


def _kv(n, seed=0):
    mx.random.seed(seed)
    return mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D))


def test_kvcache_is_keys_values_trimmed_to_offset():
    cache = KVCache()
    keys, values = _kv(300)
    cache.update_and_fetch(keys, values)  # buffer is padded to 512

    state = legacy_state(cache)
    assert len(state) == 2
    assert state[0].shape == (1, H, 300, D)
    assert mx.array_equal(state[0], keys).item()
    assert mx.array_equal(state[1], values).item()
    assert legacy_meta_state(cache) == ""


def test_rotating_meta_state_is_string_tuple():
    cache = RotatingKVCache(max_size=16, keep=4)
    cache.update_and_fetch(*_kv(10))

    keys, values = legacy_state(cache)
    assert keys.shape == (1, H, 10, D)
    assert legacy_meta_state(cache) == ("4", "16", "10", "10")


def test_batch_kvcache_is_four_tuple_trimmed_to_idx():
    a, b = KVCache(), KVCache()
    a.update_and_fetch(*_kv(5))
    b.update_and_fetch(*_kv(3, seed=1))
    cache = BatchKVCache.merge([a, b])

    keys, values, offset, left_padding = legacy_state(cache)
    assert keys.shape == (2, H, 5, D)
    assert offset.tolist() == [5, 3]
    assert left_padding.tolist() == [0, 2]
    assert legacy_meta_state(cache) == ""


def test_arrays_cache_is_the_slot_list():
    cache = ArraysCache(size=2)
    cache[0] = mx.zeros((1, 3, 4))
    cache[1] = mx.ones((1, 2, 2))

    state = legacy_state(cache)
    assert state is cache.cache
    assert legacy_state(SizedArraysCache(cache)) is cache.cache
    assert legacy_meta_state(cache) == ""


def test_cache_list_recurses():
    kv = KVCache()
    kv.update_and_fetch(*_kv(7))
    rot = RotatingKVCache(max_size=16)
    rot.update_and_fetch(*_kv(3, seed=1))
    cache = CacheList(kv, rot)

    state = legacy_state(cache)
    assert [s[0].shape[2] for s in state] == [7, 3]
    names, metas = legacy_meta_state(cache)
    assert names == ["KVCache", "RotatingKVCache"]
    assert metas == ["", ("0", "16", "3", "3")]


def test_foreign_cache_layout_passes_through():
    class OwnLayout:
        state = ("a", "b", "c")
        meta_state = ("m",)

    cache = OwnLayout()
    assert legacy_state(cache) == ("a", "b", "c")
    assert legacy_meta_state(cache) == ("m",)
