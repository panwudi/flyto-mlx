# SPDX-License-Identifier: Apache-2.0
"""SSD cache eviction order survives a rescan.

Every model load builds a new PagedSSDCacheManager, which rebuilds its index
by scanning the cache directory. The LRU used to come back in directory order
(by hash), so the first evictions after a reload removed effectively random
blocks, often the head of a live session's prefix chain, which loses every
block after it. A block hit on every turn also looked as old as its write
time, since hits were not written back to the file.
"""

import os
import time

import pytest

from omlx.cache import paged_ssd_cache
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


def _key(name):
    return name.encode().ljust(16, b"_")


def _manager(cache_dir):
    return PagedSSDCacheManager(
        cache_dir=cache_dir, max_size_bytes=100 * 1024**2, hot_cache_max_bytes=0
    )


def _save(mgr, name):
    assert mgr.save_block(
        block_hash=_key(name),
        cache_data=[(mx.ones((1, 2, 16, 16)), mx.ones((1, 2, 16, 16)))],
        token_count=16,
        model_name="test",
        layer_cache_types=["KVCache"],
    )


def _write_blocks(cache_dir, names):
    mgr = _manager(cache_dir)
    for name in names:
        _save(mgr, name)
    mgr.close()  # waits for the SSD writes
    return {name: next(cache_dir.rglob(f"{_key(name).hex()}*")) for name in names}


def test_rescan_evicts_least_recently_used_first(tmp_path):
    cache_dir = tmp_path / "ssd"
    # Pick an access order that is not the hash (directory) order.
    names = ["m", "c", "x", "a"]
    paths = _write_blocks(cache_dir, names)
    now = time.time()
    order = ["x", "a", "m", "c"]  # oldest access first
    for i, name in enumerate(order):
        t = now - 3600 + i * 60
        os.utime(paths[name], (t, t))

    mgr = _manager(cache_dir)
    try:
        one = mgr._index.get(_key("a")).file_size
        evicted = mgr._index.evict_until_size(mgr._index.total_size - 2 * one)
        assert [m.block_hash for m in evicted] == [_key("x"), _key("a")]
    finally:
        mgr.close()


def test_hit_writes_access_time_back_to_the_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "ssd"
    paths = _write_blocks(cache_dir, ["a", "b"])
    old = time.time() - 3600
    for p in paths.values():
        os.utime(p, (old, old))

    mgr = _manager(cache_dir)
    try:
        assert mgr.load_block(_key("a")) is not None
        assert paths["a"].stat().st_mtime > old + 3000
        assert paths["b"].stat().st_mtime == pytest.approx(old)

        # Within the interval a second hit does not touch the file again.
        monkeypatch.setattr(paged_ssd_cache, "_ACCESS_PERSIST_INTERVAL_S", 1e9)
        recent = time.time() - 5
        os.utime(paths["a"], (recent, recent))
        assert mgr.load_block(_key("a")) is not None
        assert paths["a"].stat().st_mtime == pytest.approx(recent)
    finally:
        mgr.close()

    # A rescan now ranks the block that was hit as the most recent.
    mgr = _manager(cache_dir)
    try:
        lru = list(mgr._index._lru)
        assert lru.index(_key("b")) < lru.index(_key("a"))
    finally:
        mgr.close()
