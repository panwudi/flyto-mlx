# SPDX-License-Identifier: Apache-2.0
"""The async store worker tells store_cache whether the live non-sliceable
state sits exactly at the end of the stored tokens.

Only the boundary-override path (intermediate snapshots present) merges a
snapshot taken at len(tokens) into the cache. Without it the live GDN /
rotating state is at the end of prompt + output, and storing it as the
last block's boundary makes a restore ingest the tail twice (upstream
8986da14).
"""

from unittest.mock import MagicMock, patch

import pytest

from omlx.scheduler import Scheduler


def _run_worker(intermediate_snapshots):
    sched = MagicMock()
    sched.block_aware_cache.store_cache.return_value = None
    with patch("omlx.scheduler._safe_sync_stream"):
        Scheduler._async_store_cache_worker(
            sched,
            "req-1",
            [1, 2, 3, 4],
            [{"state": ()}],
            None,
            intermediate_snapshots,
            None,
            None,
            None,
        )
    return sched.block_aware_cache.store_cache.call_args.kwargs


@pytest.mark.parametrize(
    "snapshots, expected", [(None, False), ({}, True), ({4: []}, True)]
)
def test_live_state_at_end_follows_boundary_override(snapshots, expected):
    kwargs = _run_worker(snapshots)
    assert kwargs["live_state_at_end"] is expected
