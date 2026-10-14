# SPDX-License-Identifier: Apache-2.0
"""Tests for context window validation feature."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from omlx.model_settings import ModelSettings


class TestGetMaxContextWindow:
    """Tests for get_max_context_window() priority logic."""

    def _make_server_state(self, global_max_ctx=32768):
        """Create a mock server state with given global max_context_window."""
        from omlx.server import SamplingDefaults

        state = MagicMock()
        state.sampling = SamplingDefaults(max_context_window=global_max_ctx)
        state.settings_manager = None
        return state

    def test_returns_global_default(self):
        """Test returns global default when no model settings."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=32768)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window()
            assert result == 32768

    def test_model_setting_overrides_global(self):
        """Test model-specific setting takes priority over global."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=32768)
        mock_manager = MagicMock()
        mock_manager.get_settings.return_value = ModelSettings(
            max_context_window=4096
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            result = get_max_context_window("test-model")
            assert result == 4096

    def test_falls_back_to_global_when_model_not_set(self):
        """Test falls back to global when model has no max_context_window."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=65536)
        mock_manager = MagicMock()
        mock_manager.get_settings.return_value = ModelSettings(
            max_context_window=None
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            result = get_max_context_window("test-model")
            assert result == 65536

    def test_no_model_id_returns_global(self):
        """Test returns global when model_id is None."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=16384)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window(None)
            assert result == 16384


class TestValidateContextWindow:
    """Tests for validate_context_window()."""

    def _make_server_state(self, global_max_ctx=32768):
        from omlx.server import SamplingDefaults

        state = MagicMock()
        state.sampling = SamplingDefaults(max_context_window=global_max_ctx)
        state.settings_manager = None
        return state

    def test_passes_when_under_limit(self):
        """Test no exception when token count is under limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            # Should not raise
            validate_context_window(500)

    def test_passes_at_exact_limit(self):
        """Test no exception when token count equals limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            # Should not raise (equal is OK)
            validate_context_window(1000)

    def test_raises_when_over_limit(self):
        """Test HTTPException raised when token count exceeds limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(1001)
            # Overflow markers that clients match before compacting.
            detail = exc_info.value.detail.lower()
            assert exc_info.value.status_code == 400
            assert exc_info.value.code == "context_length_exceeded"
            assert "prompt is too long: 1001 tokens > 1000" in detail
            assert "maximum context length" in detail

    def test_raises_with_model_specific_limit(self):
        """Test uses model-specific limit when available."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=32768)
        mock_manager = MagicMock()
        mock_manager.get_settings.return_value = ModelSettings(
            max_context_window=100
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(200, "test-model")
            assert exc_info.value.status_code == 400
            assert "200 tokens > 100" in exc_info.value.detail


class TestCountChatTokens:
    """Tests for BatchedEngine.count_chat_tokens()."""

    def test_count_chat_tokens(self):
        """Test token counting with mocked tokenizer."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine._tokenizer = mock_tokenizer

        # Mock model (not gpt_oss)
        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        count = engine.count_chat_tokens(messages)

        assert count == 5
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_tokenizer.encode.assert_called_once_with("formatted prompt")

    def test_count_chat_tokens_with_tools(self):
        """Test token counting includes tools in template."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt with tools"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7]
        engine._tokenizer = mock_tokenizer
        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None

        messages = [{"role": "user", "content": "Call a tool"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        count = engine.count_chat_tokens(messages, tools)

        assert count == 7


class TestContextOverflowResponse:
    """The 400 body must be one Claude Code recognizes as an overflow.

    Claude Code 2.1.296 lowercases the error message and checks
    ``includes("prompt is too long")``, then reads the gap with
    ``/prompt is too long[^0-9]*(\\d+)\\s*tokens?\\s*>\\s*(\\d+)/i``. Any
    other wording is treated as a plain 400: no auto-compact, and the
    session stays stuck over the limit.
    """

    @pytest.mark.asyncio
    async def test_messages_route_body_matches_claude_code(self):
        import json
        import re
        from unittest.mock import MagicMock

        from omlx import server
        from omlx.server import SamplingDefaults, validate_context_window

        state = MagicMock()
        state.sampling = SamplingDefaults(max_context_window=1000)
        state.settings_manager = None
        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(1001)

        req = MagicMock()
        req.url.path = "/v1/messages"
        req.method = "POST"
        resp = await server.http_exception_handler(req, exc_info.value)
        body = json.loads(resp.body)

        assert resp.status_code == 400
        assert body["error"]["code"] == "context_length_exceeded"
        message = body["error"]["message"].lower()
        assert "prompt is too long" in message
        m = re.search(
            r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", message
        )
        assert m and m.groups() == ("1001", "1000")
