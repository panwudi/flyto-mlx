# SPDX-License-Identifier: Apache-2.0
"""Tests for the entry-point memory-budget admission and its relationship to the
per-chunk forward gate.

The shared predicate _predicted_prefill_peak_bytes drives two outcomes:
  - _schedule_waiting DEFERS (re-queues, does NOT reject) a request that would
    breach the hard cap only because other requests are in-flight -- adaptive
    concurrency.
  - _preflight_memory_check REJECTS a lone request that cannot fit even alone.

_prefill_forward_gate stays as a concurrent-drift backstop. Because admission
estimates the FULL prompt while the gate estimates a single CHUNK (and both use
the same margin), a correctly-admitted request cannot trip the gate under static
memory; the gate only fires when memory drifts up after admission.
"""

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from omlx.memory_monitor import MemoryMonitor
from omlx.scheduler import Scheduler

GB = 1024**3


def _admission_scheduler(
    *,
    hard_limit,
    recent_peak,
    estimate,
    margin,
    running,
    prefilling=None,
    guard=True,
    monitor=True,
):
    """Bare Scheduler wired for _predicted_prefill_peak_bytes / admission."""
    s = Scheduler.__new__(Scheduler)
    s._prefill_memory_guard = guard
    s._memory_hard_limit_bytes = hard_limit
    s._memory_recent_peak_bytes = recent_peak
    s._prefill_transient_margin_bytes = margin
    s.running = running
    s.prefilling = prefilling if prefilling is not None else deque()
    s.config = MagicMock(prefill_step_size=2048)
    if monitor:
        s.memory_monitor = MagicMock()
        s.memory_monitor.estimate_prefill_peak_bytes = MagicMock(
            return_value=estimate
        )
    else:
        s.memory_monitor = None
    return s


def _request(*, num_prompt_tokens=8192, cached_tokens=0):
    r = MagicMock()
    r.request_id = "rid-1"
    r.num_prompt_tokens = num_prompt_tokens
    r.cached_tokens = cached_tokens
    return r


class TestPredictedPrefillPeakBytes:
    """Direct tests of the shared admission predicate."""

    def test_sums_current_estimate_and_margin(self):
        s = _admission_scheduler(
            hard_limit=100 * GB,
            recent_peak=0,
            estimate=20 * GB,
            margin=12 * GB,
            running={},
        )
        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=30 * GB
        ):
            mx.get_active_memory.return_value = 30 * GB
            predicted = s._predicted_prefill_peak_bytes(_request())
        assert predicted == 30 * GB + 20 * GB + 12 * GB

    def test_folds_recent_peak_when_running_or_prefilling(self):
        """In-flight (running OR prefilling) -> recent_peak high-water folded;
        fully idle -> instant reading only."""
        common = dict(
            hard_limit=100 * GB, recent_peak=90 * GB, estimate=10 * GB, margin=0
        )
        s_idle = _admission_scheduler(**common, running={}, prefilling=deque())
        s_running = _admission_scheduler(**common, running={"o": object()})
        s_prefilling = _admission_scheduler(
            **common, running={}, prefilling=deque([object()])
        )
        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=10 * GB
        ):
            mx.get_active_memory.return_value = 10 * GB
            idle = s_idle._predicted_prefill_peak_bytes(_request())
            running = s_running._predicted_prefill_peak_bytes(_request())
            prefilling = s_prefilling._predicted_prefill_peak_bytes(_request())
        assert idle == 10 * GB + 10 * GB  # instant reading only
        assert running == 90 * GB + 10 * GB  # folded: a decode is in flight
        assert prefilling == 90 * GB + 10 * GB  # folded: a chunked prefill too

    def test_cached_tokens_reduce_new_tokens(self):
        s = _admission_scheduler(
            hard_limit=100 * GB,
            recent_peak=0,
            estimate=5 * GB,
            margin=0,
            running={},
        )
        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=0
        ):
            mx.get_active_memory.return_value = 0
            s._predicted_prefill_peak_bytes(
                _request(num_prompt_tokens=8192, cached_tokens=8000)
            )
        # estimate is asked for the 192 UNCACHED tokens, not the full 8192.
        s.memory_monitor.estimate_prefill_peak_bytes.assert_called_once_with(
            192, 2048
        )

    def test_none_when_guard_off(self):
        s = _admission_scheduler(
            hard_limit=100 * GB,
            recent_peak=0,
            estimate=5 * GB,
            margin=0,
            running={},
            guard=False,
        )
        assert s._predicted_prefill_peak_bytes(_request()) is None

    def test_none_when_hard_limit_unset(self):
        s = _admission_scheduler(
            hard_limit=0,
            recent_peak=0,
            estimate=5 * GB,
            margin=0,
            running={},
        )
        assert s._predicted_prefill_peak_bytes(_request()) is None

    def test_none_when_monitor_missing(self):
        s = _admission_scheduler(
            hard_limit=100 * GB,
            recent_peak=0,
            estimate=5 * GB,
            margin=0,
            running={},
            monitor=False,
        )
        assert s._predicted_prefill_peak_bytes(_request()) is None

    def test_none_when_no_new_tokens(self):
        s = _admission_scheduler(
            hard_limit=100 * GB,
            recent_peak=0,
            estimate=5 * GB,
            margin=0,
            running={},
        )
        req = _request(num_prompt_tokens=4096, cached_tokens=4096)
        assert s._predicted_prefill_peak_bytes(req) is None

    def test_none_when_estimate_zero(self):
        s = _admission_scheduler(
            hard_limit=100 * GB,
            recent_peak=0,
            estimate=0,
            margin=0,
            running={},
        )
        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=0
        ):
            mx.get_active_memory.return_value = 0
            assert s._predicted_prefill_peak_bytes(_request()) is None


class TestScheduleWaitingBudgetDefer:
    """The defer branch in _schedule_waiting QUEUES, it does not reject."""

    def _defer_scheduler(
        self, *, hard_limit, recent_peak, estimate, margin, running=None, prefilling=None
    ):
        s = _admission_scheduler(
            hard_limit=hard_limit,
            recent_peak=recent_peak,
            estimate=estimate,
            margin=margin,
            running=running if running is not None else {"r-other": object()},
            prefilling=prefilling,
        )
        s.config = MagicMock(max_num_seqs=8, prefill_step_size=2048)
        s._admission_paused = False
        s._memory_limit_bytes = 0  # bypass the coarse generation soft-guard
        s.batch_generator = MagicMock()
        s._ensure_batch_generator = MagicMock()
        return s

    def _waiting_request(self):
        req = _request()
        req.prompt_cache = None
        req.remaining_tokens = None
        req.prompt_token_ids = [1, 2, 3]
        return req

    def test_defers_and_requeues_when_inflight_would_breach(self):
        # predicted = max(10, 95) + 20 + 12 = 127 > 100 -> defer.
        s = self._defer_scheduler(
            hard_limit=100 * GB,
            recent_peak=95 * GB,
            estimate=20 * GB,
            margin=12 * GB,
        )
        req = self._waiting_request()
        s.waiting = deque([req])

        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=10 * GB
        ):
            mx.get_active_memory.return_value = 10 * GB
            scheduled, rejected = s._schedule_waiting()

        # Nothing scheduled, nothing REJECTED -- the request is left queued.
        assert scheduled == []
        assert rejected == []
        assert list(s.waiting) == [req]

    def test_defers_when_only_a_chunked_prefill_is_in_flight(self):
        """The headline case: running is EMPTY but another request is
        mid-chunked-prefill (self.prefilling). The new request must still be
        deferred -- gating on self.running alone would stack a second prefill,
        which is exactly the documented crash.
        """
        # predicted = max(10, 95) + 20 + 12 = 127 > 100 -> defer.
        s = self._defer_scheduler(
            hard_limit=100 * GB,
            recent_peak=95 * GB,
            estimate=20 * GB,
            margin=12 * GB,
            running={},  # nothing decoding...
            prefilling=deque([object()]),  # ...but a chunked prefill is in flight
        )
        req = self._waiting_request()
        s.waiting = deque([req])

        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=10 * GB
        ):
            mx.get_active_memory.return_value = 10 * GB
            scheduled, rejected = s._schedule_waiting()

        assert scheduled == []
        assert rejected == []
        assert list(s.waiting) == [req]  # queued behind the in-flight prefill

    def test_does_not_defer_when_predicted_fits(self):
        # predicted = max(10, 30) + 20 + 12 = 62 <= 100 -> the budget branch
        # does not fire. Asserted at the predicate so we do not have to drive
        # the heavy insert path: a fitting request is never re-queued by the
        # budget defer.
        s = self._defer_scheduler(
            hard_limit=100 * GB,
            recent_peak=30 * GB,
            estimate=20 * GB,
            margin=12 * GB,
        )
        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=10 * GB
        ):
            mx.get_active_memory.return_value = 10 * GB
            predicted = s._predicted_prefill_peak_bytes(self._waiting_request())
        assert predicted is not None
        assert predicted <= s._memory_hard_limit_bytes


class TestAdmissionDominatesGate:
    """Property: a request admitted by the entry budget cannot trip the
    per-chunk forward gate under static memory; the gate is a drift backstop."""

    def _monitor(self):
        m = MemoryMonitor(max_kv_cache_memory=64 * GB)
        m.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
            num_attention_heads=32,
        )
        return m

    def _scheduler_with(self, monitor, *, cap, current, margin):
        s = Scheduler.__new__(Scheduler)
        s._prefill_memory_guard = True
        s._memory_hard_limit_bytes = cap
        s._memory_recent_peak_bytes = current
        s._prefill_transient_margin_bytes = margin
        s.running = {"r-other": object()}
        s.prefilling = deque()
        s.config = MagicMock(prefill_step_size=2048)
        s.memory_monitor = monitor
        return s

    def test_admitted_request_never_trips_gate_under_static_memory(self):
        monitor = self._monitor()
        full_prompt, chunk, step = 8192, 256, 2048
        admission_estimate = monitor.estimate_prefill_peak_bytes(full_prompt, step)
        gate_estimate = monitor.estimate_prefill_peak_bytes(chunk, step)
        # Real estimate is monotonic in prompt length -> full >= per-chunk.
        assert 0 < gate_estimate <= admission_estimate

        current = 80 * GB
        margin = 12 * GB
        # cap set exactly at the admission boundary: admission JUST passes.
        cap = current + admission_estimate + margin
        s = self._scheduler_with(monitor, cap=cap, current=current, margin=margin)

        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=current
        ):
            mx.get_active_memory.return_value = current
            predicted = s._predicted_prefill_peak_bytes(
                _request(num_prompt_tokens=full_prompt)
            )
            assert predicted is not None and predicted <= cap  # admitted

            # Same static memory: the gate, on any chunk of this request, must
            # NOT raise (gate predicted = current + gate_estimate + margin
            # <= current + admission_estimate + margin = cap).
            s._prefill_forward_gate(
                chunk, request_id="rid-1", loop_label="external"
            )  # no RuntimeError

    def test_gate_fires_when_memory_drifts_up_after_admission(self):
        monitor = self._monitor()
        full_prompt, chunk, step = 8192, 256, 2048
        admission_estimate = monitor.estimate_prefill_peak_bytes(full_prompt, step)

        current = 80 * GB
        margin = 12 * GB
        cap = current + admission_estimate + margin
        s = self._scheduler_with(monitor, cap=cap, current=current, margin=margin)

        # Other in-flight requests grow KV during this prefill: current drifts
        # up well past where admission snapshotted it. The gate re-reads and
        # catches what the admission snapshot could not.
        drifted = current + admission_estimate
        with patch("omlx.scheduler.mx") as mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=drifted
        ):
            mx.get_active_memory.return_value = drifted
            with pytest.raises(RuntimeError, match="refused before forward"):
                s._prefill_forward_gate(
                    chunk, request_id="rid-1", loop_label="external"
                )
