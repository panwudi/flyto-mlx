# SPDX-License-Identifier: Apache-2.0
"""The hot cache gives its memory back before a request is refused.

The hot tier of the SSD cache holds raw KV bytes in process memory, so it
counts toward the resident figure every memory gate checks. Before this, a
full hot cache stayed full while the prefill gate and the HTTP admission gate
refused requests. These tests drive real PagedSSDCacheManager instances and
model resident memory as "everything else + what the hot caches hold".
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.process_memory_enforcer import ProcessMemoryEnforcer
from omlx.scheduler import Scheduler

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

# One block: 2 layers x (keys, values) of 1x2x16x16 float32.
BLOCK = 2 * 2 * 2 * 16 * 16 * 4


def _cache_data():
    return [
        (mx.ones((1, 2, 16, 16)), mx.ones((1, 2, 16, 16))) for _ in range(2)
    ]


def _save(mgr, name):
    assert mgr.save_block(
        block_hash=name.encode().ljust(16, b"_"),
        cache_data=_cache_data(),
        token_count=16,
        model_name="test",
        layer_cache_types=["KVCache"] * 2,
    )


def _key(name):
    return name.encode().ljust(16, b"_")


def _make_clean(mgr, name):
    """Mark a hot entry as already on SSD (as a promoted block would be)."""
    mgr._index.add(mgr._hot_cache[_key(name)]["block_metadata"])


@pytest.fixture
def mgr(tmp_path):
    m = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=BLOCK * 10,
    )
    yield m
    m.close()


class TestShedHotCache:
    def test_clean_entries_go_first(self, mgr):
        for name in ("a", "b", "c"):
            _save(mgr, name)
        _make_clean(mgr, "c")  # newest, but already on SSD

        freed, queued = mgr.shed_hot_cache(BLOCK, drop_dirty=True)

        assert (freed, queued) == (BLOCK, 0)
        assert mgr._hot_cache_get(_key("c")) is None
        assert mgr._hot_cache_get(_key("a")) is not None

    def test_dirty_entries_are_written_not_lost(self, mgr, tmp_path):
        for name in ("a", "b"):
            _save(mgr, name)

        freed, queued = mgr.shed_hot_cache(BLOCK, drop_dirty=False)

        assert (freed, queued) == (0, BLOCK)
        assert mgr._hot_cache_get(_key("a")) is None
        deadline = time.time() + 5
        while mgr.pending_eviction_bytes and time.time() < deadline:
            time.sleep(0.05)
        assert mgr.pending_eviction_bytes == 0
        assert mgr.has_block(_key("a"))
        assert list((tmp_path / "ssd").rglob("*.safetensors"))

    def test_drop_dirty_frees_at_once(self, mgr):
        for name in ("a", "b"):
            _save(mgr, name)

        freed, queued = mgr.shed_hot_cache(2 * BLOCK, drop_dirty=True)

        assert (freed, queued) == (2 * BLOCK, 0)
        assert mgr.hot_cache_size_bytes == 0
        assert mgr.pending_eviction_bytes == 0

    def test_limit_holds_until_relaxed(self, mgr):
        for name in ("a", "b", "c"):
            _save(mgr, name)
        mgr.shed_hot_cache(BLOCK, drop_dirty=True)
        assert mgr.hot_cache_limit_bytes == 2 * BLOCK

        # A new store must not grow the cache past the lowered limit.
        _save(mgr, "d")
        assert mgr.hot_cache_size_bytes == 2 * BLOCK

        mgr.relax_hot_cache_limit(BLOCK)
        _save(mgr, "e")
        assert mgr.hot_cache_size_bytes == 3 * BLOCK

        mgr.relax_hot_cache_limit(100 * BLOCK)
        assert mgr.hot_cache_limit_bytes == 10 * BLOCK


def _resident(base, managers):
    def read():
        r = base + sum(m.hot_cache_size_bytes for m in managers)
        return r, r, r, 0

    return read


def _enforcer(managers, ceiling, margin):
    entries = {
        f"m{i}": SimpleNamespace(engine=SimpleNamespace(
            scheduler=SimpleNamespace(paged_ssd_cache_manager=m)))
        for i, m in enumerate(managers)
    }
    enforcer = ProcessMemoryEnforcer(engine_pool=MagicMock(_entries=entries))
    enforcer._prefill_transient_margin_bytes = margin
    enforcer._cached_ceiling = lambda: ceiling
    enforcer._get_hard_limit_bytes = lambda: ceiling
    return enforcer


class TestAdmissionGate:
    def test_admits_after_shedding(self, mgr):
        for name in ("a", "b", "c", "d"):
            _save(mgr, name)
        # 1000 + 4 blocks resident, margin 100, ceiling fits 2 blocks.
        enforcer = _enforcer([mgr], ceiling=1100 + 2 * BLOCK, margin=100)
        with patch(
            "omlx.process_memory_enforcer.resident_usage_bytes",
            side_effect=_resident(1000, [mgr]),
        ):
            ok, current, _ = enforcer.prefill_admission_ok()
        assert ok
        assert current == 1000 + 2 * BLOCK

    def test_refuses_when_shedding_is_not_enough(self, mgr):
        _save(mgr, "a")
        enforcer = _enforcer([mgr], ceiling=1050, margin=100)
        with patch(
            "omlx.process_memory_enforcer.resident_usage_bytes",
            side_effect=_resident(1000, [mgr]),
        ):
            ok, _, _ = enforcer.prefill_admission_ok()
        assert not ok
        assert mgr.hot_cache_size_bytes == 0

    def test_keeping_unsaved_blocks_queues_them_instead(self, mgr):
        """drop_dirty=False: the gate's first try loses no cached prefix."""
        _save(mgr, "a")
        enforcer = _enforcer([mgr], ceiling=1050, margin=100)
        with patch(
            "omlx.process_memory_enforcer.resident_usage_bytes",
            side_effect=_resident(1000, [mgr]),
        ):
            ok, _, _ = enforcer.prefill_admission_ok(drop_dirty=False)
        assert not ok
        mgr.close()  # waits for the queued write
        assert mgr.load_block(_key("a")) is not None


class TestPressureBalance:
    def test_sheds_to_ssd_above_soft_and_relaxes_below(self, mgr):
        from omlx import process_memory_enforcer as pme

        for name in ("a", "b", "c", "d"):
            _save(mgr, name)
        enforcer = _enforcer([mgr], ceiling=0, margin=0)
        soft = pme._HOT_CACHE_SHED_PAD_BYTES + 2 * BLOCK
        with patch(
            "omlx.process_memory_enforcer.resident_usage_bytes",
            side_effect=_resident(0, [mgr]),
        ):
            enforcer._balance_hot_caches(soft)
        # Shed down to 2 blocks, and to SSD rather than dropped.
        assert mgr.hot_cache_size_bytes == 2 * BLOCK
        assert mgr.hot_cache_limit_bytes == 2 * BLOCK
        deadline = time.time() + 5
        while mgr.pending_eviction_bytes and time.time() < deadline:
            time.sleep(0.05)
        assert mgr.has_block(_key("a")) and mgr.has_block(_key("b"))

        # Plenty of room again: the limit grows back one step per tick.
        with patch(
            "omlx.process_memory_enforcer.resident_usage_bytes",
            return_value=(0, 0, 0, 0),
        ), patch.object(pme, "_HOT_CACHE_RELAX_STEP_BYTES", BLOCK):
            enforcer._balance_hot_caches(10**12)
        assert mgr.hot_cache_limit_bytes == 3 * BLOCK


def _gate_scheduler(mgr, cap, margin):
    s = Scheduler.__new__(Scheduler)
    s._prefill_memory_guard = True
    s._memory_hard_limit_bytes = cap
    s._memory_recent_peak_bytes = 0
    s._prefill_transient_margin_bytes = margin
    s._stream = None
    s.memory_monitor = None
    s.config = MagicMock(prefill_step_size=2048)
    s._prefill_gate_state_logged = True
    s.paged_ssd_cache_manager = mgr
    return s


class TestPrefillForwardGate:
    def _run(self, s, mgr):
        managers = [mgr] if mgr is not None else []
        read = _resident(1000, managers)
        with patch("omlx.scheduler.mx") as mock_mx, patch(
            "omlx.scheduler.get_phys_footprint", side_effect=lambda: read()[2]
        ), patch(
            "omlx.scheduler.resident_usage_bytes", side_effect=read
        ), patch("omlx.scheduler._sync_and_clear_cache"):
            mock_mx.get_active_memory.return_value = 0
            s._prefill_forward_gate(256, request_id="r", loop_label="test")

    def test_proceeds_after_shedding(self, mgr):
        for name in ("a", "b", "c", "d"):
            _save(mgr, name)
        s = _gate_scheduler(mgr, cap=1100 + 2 * BLOCK, margin=100)
        with patch("omlx.scheduler._HOT_CACHE_GATE_PAD_BYTES", 0):
            self._run(s, mgr)
        assert mgr.hot_cache_size_bytes == 2 * BLOCK

    def test_sheds_a_pad_beyond_the_shortfall(self, mgr):
        # Shedding exactly the shortfall leaves the re-measure on the
        # edge of the cap, where a little phys noise refuses the chunk.
        for name in ("a", "b", "c", "d"):
            _save(mgr, name)
        s = _gate_scheduler(mgr, cap=1100 + 3 * BLOCK, margin=100)
        with patch("omlx.scheduler._HOT_CACHE_GATE_PAD_BYTES", BLOCK):
            self._run(s, mgr)
        assert mgr.hot_cache_size_bytes == 2 * BLOCK

    def test_sheds_another_models_hot_cache(self, tmp_path):
        # This model's hot cache is empty; the room is held by another
        # resident model's. The enforcer's cross-model shed frees it.
        own = PagedSSDCacheManager(
            cache_dir=tmp_path / "own", max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=BLOCK * 10,
        )
        other = PagedSSDCacheManager(
            cache_dir=tmp_path / "other", max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=BLOCK * 10,
        )
        try:
            for name in ("a", "b", "c", "d"):
                _save(other, name)
            enforcer = _enforcer([own, other], ceiling=0, margin=0)
            s = _gate_scheduler(own, cap=1100 + 2 * BLOCK, margin=100)
            s._shed_all_hot_caches = enforcer.shed_hot_caches
            read = _resident(1000, [own, other])
            with patch("omlx.scheduler.mx") as mock_mx, patch(
                "omlx.scheduler.get_phys_footprint", side_effect=lambda: read()[2]
            ), patch(
                "omlx.scheduler.resident_usage_bytes", side_effect=read
            ), patch("omlx.scheduler._sync_and_clear_cache"), patch(
                "omlx.scheduler._HOT_CACHE_GATE_PAD_BYTES", 0
            ):
                mock_mx.get_active_memory.return_value = 0
                s._prefill_forward_gate(256, request_id="r", loop_label="test")
            assert other.hot_cache_size_bytes == 2 * BLOCK
        finally:
            own.close()
            other.close()

    def test_refuses_without_hot_cache(self):
        s = _gate_scheduler(None, cap=1050, margin=100)
        with pytest.raises(RuntimeError, match="Prefill refused"):
            self._run(s, None)

    def test_refuses_when_shedding_is_not_enough(self, mgr):
        _save(mgr, "a")
        s = _gate_scheduler(mgr, cap=1050, margin=100)
        with pytest.raises(RuntimeError, match="Prefill refused"):
            self._run(s, mgr)


class TestEnforcementPoll:
    async def test_poll_sheds_before_judging_pressure(self, mgr):
        from omlx import process_memory_enforcer as pme

        for name in ("a", "b", "c", "d"):
            _save(mgr, name)
        ceiling = 10_000 + 4 * BLOCK
        enforcer = _enforcer([mgr], ceiling=ceiling, margin=0)
        soft = int(ceiling * enforcer._soft_threshold)
        # Base usage leaves room below soft for about one block.
        base = soft - BLOCK - BLOCK // 2
        read = _resident(base, [mgr])
        with patch(
            "omlx.process_memory_enforcer.resident_usage_bytes", side_effect=read
        ), patch.object(
            enforcer, "_current_usage_bytes", side_effect=lambda: read()[0]
        ), patch.object(pme, "_HOT_CACHE_SHED_PAD_BYTES", 0):
            await enforcer._check_and_enforce()
        assert mgr.hot_cache_size_bytes == BLOCK
        assert enforcer.get_pressure_level() == "ok"
