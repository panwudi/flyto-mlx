# SPDX-License-Identifier: Apache-2.0
"""A block evicted from the hot cache stays readable until its file is written.

Eviction adds the block to the SSD index and queues the file write. A read
that came in before the writer finished used to find no file, report it
missing and drop the index entry; the writer then deleted the file it had
just written because the block was no longer indexed. A long session's next
turn, seconds after the store, lost that block for good.
"""

import threading

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


def _save(mgr, name):
    assert mgr.save_block(
        block_hash=_key(name),
        cache_data=[(mx.ones((1, 2, 16, 16)), mx.ones((1, 2, 16, 16)))],
        token_count=16,
        model_name="test",
        layer_cache_types=["KVCache"],
    )


def test_block_is_readable_while_its_write_is_queued(tmp_path, monkeypatch):
    cache_dir = tmp_path / "ssd"
    release = threading.Event()
    real_write = paged_ssd_cache._write_safetensors_no_mx

    def slow_write(*args, **kwargs):
        release.wait(10)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(paged_ssd_cache, "_write_safetensors_no_mx", slow_write)

    # Room for one block: saving "b" evicts "a" to the write queue.
    mgr = PagedSSDCacheManager(
        cache_dir=cache_dir, max_size_bytes=100 * 1024**2, hot_cache_max_bytes=6000
    )
    try:
        _save(mgr, "a")
        _save(mgr, "b")
        path = mgr._index.get(_key("a")).file_path
        assert not path.exists()

        assert mgr.load_block(_key("a")) is not None
        data, meta = mgr.load_block_with_metadata(_key("a"))
        assert data is not None and meta["token_count"] == 16
        assert mgr._index.contains(_key("a"))
    finally:
        release.set()
        mgr.close()

    assert path.exists()
    mgr = PagedSSDCacheManager(
        cache_dir=cache_dir, max_size_bytes=100 * 1024**2, hot_cache_max_bytes=0
    )
    try:
        assert mgr.load_block(_key("a")) is not None
    finally:
        mgr.close()
