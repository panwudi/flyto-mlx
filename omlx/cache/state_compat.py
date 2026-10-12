# SPDX-License-Identifier: Apache-2.0
"""Read cache state in the mlx-lm 0.31 layout on any mlx-lm version.

mlx-lm 0.32 made ``cache.state`` the full object state: ``KVCache.state`` is
``(keys, values, offset)`` with keys padded to the buffer step,
``RotatingKVCache.state`` carries keep/max_size/_idx, ``BatchKVCache.state``
gained ``_idx``, ``ArraysCache.state`` gained the batch bookkeeping, and
``meta_state`` was removed. oMLX's prefix cache, boundary snapshots and the
paged SSD cache store the 0.31 layout (``(keys, values)`` trimmed to the
offset plus a string ``meta_state`` tuple), so blocks written before the
upgrade must stay readable and new blocks must look the same.

``legacy_state`` / ``legacy_meta_state`` return that layout. Only classes
whose ``state`` comes from ``mlx_lm.models.cache`` are converted; caches that
define their own ``state`` (mlx-vlm's cache module, oMLX's PoolingCache,
TurboQuant, ...) keep their own contract and are passed through.
"""

from __future__ import annotations

from typing import Any

_MLX_LM_CACHE_MODULE = "mlx_lm.models.cache"


def _mlx_lm_base(cache: Any, attr: str) -> str | None:
    """Name of the mlx-lm class that defines ``attr`` for this object.

    Returns None when a class outside ``mlx_lm.models.cache`` defines it
    first in the MRO (that class owns the layout).
    """
    for klass in type(cache).__mro__:
        if attr in klass.__dict__:
            if klass.__module__ == _MLX_LM_CACHE_MODULE:
                return klass.__name__
            return None
    return None


def _trimmed(cache: Any, length: int) -> tuple[Any, Any]:
    keys, values = cache.keys, cache.values
    if keys is None:
        return None, None
    if length < keys.shape[2]:
        return keys[..., :length, :], values[..., :length, :]
    return keys, values


def legacy_state(cache: Any) -> Any:
    """``cache.state`` as mlx-lm 0.31 returned it."""
    inner = getattr(cache, "_inner", None)
    if inner is not None and type(cache).__name__ == "SizedArraysCache":
        cache = inner

    base = _mlx_lm_base(cache, "state")
    if base is None:
        return cache.state
    if base in ("KVCache", "RotatingKVCache", "ChunkedKVCache", "QuantizedKVCache"):
        keys_and_values = getattr(cache, "keys_and_values", None)
        if cache.keys is None:
            return None, None
        if callable(keys_and_values):
            return keys_and_values()
        return _trimmed(cache, cache.offset)
    if base == "BatchKVCache":
        keys, values = _trimmed(cache, cache._idx)
        return keys, values, cache.offset, cache.left_padding
    if base == "BatchRotatingKVCache":
        keys, values = _trimmed(cache, cache._offset)
        return keys, values, cache.offset, cache.left_padding
    if base == "ArraysCache":
        return cache.cache
    if base == "CacheList":
        return [legacy_state(c) for c in cache.caches]
    return cache.state


def legacy_meta_state(cache: Any) -> Any:
    """``cache.meta_state`` as mlx-lm 0.31 returned it."""
    inner = getattr(cache, "_inner", None)
    if inner is not None and type(cache).__name__ == "SizedArraysCache":
        cache = inner

    owner = _mlx_lm_base(cache, "state")
    if owner is None or _mlx_lm_base(cache, "meta_state") is not None:
        # Own layout, or an mlx-lm version that still has meta_state.
        return getattr(cache, "meta_state", "")
    if owner == "RotatingKVCache":
        return tuple(
            map(str, (cache.keep, cache.max_size, cache.offset, cache._idx))
        )
    if owner == "QuantizedKVCache":
        return tuple(map(str, (cache.offset, cache.group_size, cache.bits)))
    if owner == "ChunkedKVCache":
        return tuple(map(str, (cache.chunk_size, cache.start_position)))
    if owner == "BatchRotatingKVCache":
        return tuple(
            map(str, (cache.max_size, cache._offset, cache._idx, cache.rotated))
        )
    if owner == "CacheList":
        return (
            [type(c).__name__ for c in cache.caches],
            [legacy_meta_state(c) for c in cache.caches],
        )
    return ""
