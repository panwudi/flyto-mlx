# SPDX-License-Identifier: Apache-2.0
"""The HTTP admission gate and the prefill forward gate must agree.

If admission lets a request through that the forward gate then refuses, the
client gets HTTP 200 followed by an error mid-stream -- the failure mode the
admission gate exists to prevent (PR #90). Both now decide on resident memory
(phys net of MLX's free-buffer pool) plus the transient margin; the forward
gate's high-water reading only triggers a reclaim. Enumerate readings and
check the two verdicts match.
"""

import itertools
from unittest.mock import MagicMock, patch

from omlx.process_memory_enforcer import ProcessMemoryEnforcer
from omlx.scheduler import Scheduler

GB = 1024**3
CEILING = 107 * GB
MARGIN = 12 * GB


def _admission(active, phys, cache, recent_peak):
    enforcer = ProcessMemoryEnforcer(
        engine_pool=MagicMock(), prefill_transient_margin_gb=MARGIN / GB
    )
    enforcer._get_hard_limit_bytes = lambda: CEILING
    enforcer._cached_ceiling = lambda: CEILING
    enforcer._recent_peak_bytes = recent_peak
    with patch(
        "omlx.process_memory_enforcer.resident_usage_bytes",
        return_value=(max(active, phys - cache), active, phys, cache),
    ):
        return enforcer.prefill_admission_ok()[0]


def _forward(active, phys, cache, recent_peak):
    s = Scheduler.__new__(Scheduler)
    s._prefill_memory_guard = True
    s._memory_hard_limit_bytes = CEILING
    s._memory_recent_peak_bytes = recent_peak
    s._prefill_transient_margin_bytes = MARGIN
    s._stream = None
    s.memory_monitor = None
    s.config = MagicMock(prefill_step_size=2048)
    s._prefill_gate_state_logged = True
    # After the reclaim the pool is returned: phys drops by the pool size.
    after = (max(active, phys - cache), active, phys - cache, 0)
    with patch("omlx.scheduler.mx") as mock_mx, patch(
        "omlx.scheduler.get_phys_footprint", return_value=phys
    ), patch("omlx.scheduler.resident_usage_bytes", return_value=after), patch(
        "omlx.scheduler._sync_and_clear_cache"
    ):
        mock_mx.get_active_memory.return_value = active
        try:
            s._prefill_forward_gate(256, request_id="r", loop_label="external")
        except RuntimeError:
            return False
    return True


def test_admission_and_forward_gate_agree():
    values = [g * GB for g in (40, 60, 80, 90, 94, 96, 100, 110)]
    pools = [g * GB for g in (0, 10, 30)]
    checked = 0
    for active, phys, cache, recent in itertools.product(values, values, pools, values):
        if cache > phys:
            continue
        a = _admission(active, phys, cache, recent)
        f = _forward(active, phys, cache, recent)
        assert a == f, (active / GB, phys / GB, cache / GB, recent / GB, a, f)
        checked += 1
    assert checked > 1000
