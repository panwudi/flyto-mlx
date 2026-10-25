# SPDX-License-Identifier: Apache-2.0
"""Tests for the pre-stream memory admission gate (server layer).

The gate rejects an over-budget generation request with HTTP 503 + Retry-After
BEFORE the SSE stream opens, instead of a mid-stream error event after HTTP 200
(see docs/memory-admission-gate-design.md).
"""

from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import HTTPException

from omlx import server


def _pool_with_enforcer(admission_result):
    """Build a mock engine pool whose enforcer returns admission_result."""
    enforcer = MagicMock()
    enforcer.prefill_admission_ok.return_value = admission_result
    pool = MagicMock()
    pool._process_memory_enforcer = enforcer
    return pool


class TestMemoryAdmissionGate:
    @pytest.mark.asyncio
    async def test_rejects_over_budget_with_503_and_retry_after(self):
        pool = _pool_with_enforcer((False, 100 * 1024**3, 107 * 1024**3))
        with patch("omlx.server.get_engine_pool", return_value=pool), patch(
            "omlx.server._ADMISSION_SETTLE_S", 0.05
        ), patch("omlx.server._ADMISSION_POLL_S", 0.01):
            with pytest.raises(HTTPException) as exc:
                await server._memory_admission_gate()
        assert exc.value.status_code == 503
        assert exc.value.headers.get("Retry-After") == "1"
        # Unsaved cache blocks are given up only as the last resort.
        calls = pool._process_memory_enforcer.prefill_admission_ok.call_args_list
        assert calls[0] == call(drop_dirty=False)
        assert calls[-1] == call(drop_dirty=True)

    @pytest.mark.asyncio
    async def test_passes_when_admission_ok(self):
        pool = _pool_with_enforcer((True, 0, 0))
        with patch("omlx.server.get_engine_pool", return_value=pool):
            await server._memory_admission_gate()  # must not raise
        pool._process_memory_enforcer.prefill_admission_ok.assert_called_once_with(
            drop_dirty=False
        )

    @pytest.mark.asyncio
    async def test_waits_for_memory_instead_of_dropping_unsaved_blocks(self):
        """The previous turn's KV is released a moment after it finishes.

        Production 2026-10-24 02:01: the gate dropped a 190K session's
        unsaved blocks at once; one second later 28GB came back by itself.
        """
        over = (False, 100 * 1024**3, 104 * 1024**3)
        pool = _pool_with_enforcer(None)
        enforcer = pool._process_memory_enforcer
        enforcer.prefill_admission_ok.return_value = None
        enforcer.prefill_admission_ok.side_effect = [over, over, (True, 0, 0)]
        with patch("omlx.server.get_engine_pool", return_value=pool), patch(
            "omlx.server._ADMISSION_POLL_S", 0.01
        ):
            await server._memory_admission_gate()  # must not raise
        assert call(drop_dirty=True) not in enforcer.prefill_admission_ok.call_args_list

    @pytest.mark.asyncio
    async def test_noop_when_enforcer_absent(self):
        pool = MagicMock()
        pool._process_memory_enforcer = None
        with patch("omlx.server.get_engine_pool", return_value=pool):
            await server._memory_admission_gate()  # must not raise


class TestHttpExceptionHeaderPropagation:
    """The generic HTTPException handler must forward exc.headers to the
    response. A live smoke found Retry-After was silently dropped: the handler
    built a JSONResponse without headers, so a unit test asserting only on the
    exception OBJECT passed while the real HTTP response lacked the header."""

    @pytest.mark.asyncio
    async def test_retry_after_reaches_the_response(self):
        req = MagicMock()
        req.url.path = "/v1/chat/completions"
        req.method = "POST"
        exc = HTTPException(
            status_code=503,
            detail="Server memory is at capacity. Retry shortly.",
            headers={"Retry-After": "1"},
        )
        resp = await server.http_exception_handler(req, exc)
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "1"

    @pytest.mark.asyncio
    async def test_no_headers_is_still_fine(self):
        req = MagicMock()
        req.url.path = "/v1/chat/completions"
        req.method = "POST"
        exc = HTTPException(status_code=404, detail="not found")
        resp = await server.http_exception_handler(req, exc)
        assert resp.status_code == 404
        assert resp.headers.get("Retry-After") is None
