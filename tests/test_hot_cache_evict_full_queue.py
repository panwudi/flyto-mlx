# SPDX-License-Identifier: Apache-2.0
"""Hot cache eviction never loses a block because the SSD write queue is full.

Eviction queued every evicted block for an SSD write, including blocks that
were already on SSD (promoted back after a load). When the queue was full the
block was dropped and its index entry removed, so a block whose file was on
disk turned into a silent miss, and an unsaved one was gone. On 2026-10-24
10:00 a 220K-token session reloading its prefix filled the queue this way;
its next turn reused 44 of 108 blocks and re-prefilled 132K tokens.
"""

import threading
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


def _save(mgr, name):
    assert mgr.save_block(
        block_hash=_key(name),
        cache_data=[(mx.ones((1, 2, 16, 16)), mx.ones((1, 2, 16, 16)))],
        token_count=16,
        model_name="test",
        layer_cache_types=["KVCache"],
    )


def _wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


@pytest.fixture
def stalled(tmp_path, monkeypatch):
    """A manager with a one-slot write queue and a writer that can be held."""
    monkeypatch.setattr(paged_ssd_cache, "_MAX_PENDING_WRITES", 1)
    hold = threading.Event()
    hold.set()
    writes = []
    real_write = paged_ssd_cache._write_safetensors_no_mx

    def write(path, tensors_raw, metadata=None):
        hold.wait(10)
        writes.append(path)
        return real_write(path, tensors_raw, metadata)

    monkeypatch.setattr(paged_ssd_cache, "_write_safetensors_no_mx", write)
    # Room for one block in the hot cache.
    mgr = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=6000,
    )

    def on_ssd(name):
        meta = mgr._index.get(_key(name))
        return (
            meta is not None
            and meta.file_path.exists()
            and _key(name) not in mgr._pending_write_hashes
        )

    def fill_queue():
        """Park the writer on one write and fill the queue behind it,
        without touching the hot cache."""
        hold.clear()
        for i in range(2):
            mgr._write_queue.put(
                (_key(f"x{i}"), {}, {}, tmp_path / "ssd" / f"x{i}.safetensors")
            )
            if i == 0:
                _wait_for(lambda: mgr._write_queue.qsize() == 0)
        assert mgr._write_queue.full()

    mgr.on_ssd = on_ssd
    mgr.fill_queue = fill_queue
    mgr.hold = hold
    mgr.writes = writes
    try:
        yield mgr
    finally:
        hold.set()
        mgr.close()


def test_block_already_on_ssd_stays_indexed(stalled):
    mgr = stalled
    _save(mgr, "a")
    _save(mgr, "b")  # evicts a, which is written
    _wait_for(lambda: mgr.on_ssd("a"))
    assert mgr.load_block(_key("a")) is not None  # promotes a, evicts b
    _wait_for(lambda: mgr.on_ssd("b"))
    assert _key("a") in mgr._hot_cache

    mgr.fill_queue()
    _save(mgr, "c")  # evicts a from the hot cache
    assert _key("a") not in mgr._hot_cache

    assert mgr._index.contains(_key("a"))
    assert mgr.load_block(_key("a")) is not None
    # Its file is not written a second time.
    mgr.hold.set()
    _wait_for(lambda: mgr._write_queue.qsize() == 0)
    a_stem = mgr._index.get(_key("a")).file_path.stem
    assert sum(1 for p in mgr.writes if a_stem in p) == 1


def test_unsaved_block_is_kept_instead_of_dropped(stalled):
    mgr = stalled
    _save(mgr, "d")
    mgr.fill_queue()
    _save(mgr, "e")  # evicts d, which has no file yet and no queue slot

    assert mgr.load_block(_key("d")) is not None
    mgr.hold.set()
    mgr.close()
    assert mgr.on_ssd("d")
