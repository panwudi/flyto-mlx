# SPDX-License-Identifier: Apache-2.0
"""Tests for HRM-Text (sapientinc/HRM-Text) discovery + engine wiring.

Covers detect_model_type "hrm_text", full discover_models registration
(model_type + engine_type), the engine-pool type->engine mapping, an LLM
regression guard, and the engine's pure prompt-rendering logic (direct vs
thinking condition tokens, boundary tokens, multi-message joining) -- the
prompt-render path needs no model weights, so it runs in CI.
"""

import json
from pathlib import Path

from omlx.engine.hrm_text import (
    _COND_DIRECT,
    _COND_THINK,
    _IM_END,
    _IM_START,
    HrmTextEngine,
    _msg_text,
)
from omlx.engine_pool import EnginePool
from omlx.model_discovery import detect_model_type, discover_models


def make_hrm_dir(parent: Path, name: str = "HRM-Text-1B", weight_bytes: int = 512) -> Path:
    model_dir = parent / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hrm_text",
                "architectures": ["HrmTextForCausalLM"],
                "hidden_size": 1536,
                "num_hidden_layers": 16,
                "num_attention_heads": 12,
                "head_dim": 128,
                "H_cycles": 2,
                "L_cycles": 3,
                "embedding_scale": 39.19,
                "prefix_lm": True,
                "eos_token_id": 11,
            }
        )
    )
    (model_dir / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    return model_dir


def make_llm_dir(parent: Path, name: str = "llama-3b", weight_bytes: int = 512) -> Path:
    model_dir = parent / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    (model_dir / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    return model_dir


class TestDetectModelType:
    def test_hrm_text_detected(self, tmp_path):
        d = make_hrm_dir(tmp_path)
        assert detect_model_type(d) == "hrm_text"

    def test_llm_still_llm(self, tmp_path):
        d = make_llm_dir(tmp_path)
        assert detect_model_type(d) == "llm"


class TestDiscovery:
    def test_registration(self, tmp_path):
        make_hrm_dir(tmp_path / "org")
        models = discover_models(tmp_path)
        assert set(models.keys()) == {"HRM-Text-1B"}
        entry = models["HRM-Text-1B"]
        assert entry.model_type == "hrm_text"
        assert entry.engine_type == "hrm_text"

    def test_type_to_engine_mapping(self):
        assert EnginePool._MODEL_TYPE_TO_ENGINE["hrm_text"] == "hrm_text"


class TestPromptRendering:
    """_render_prompt is pure (no weights); validate the HRM-Text format."""

    def _engine(self, **kw):
        return HrmTextEngine(model_name="/nonexistent", **kw)

    def test_direct_condition(self):
        eng = self._engine()
        out = eng._render_prompt([{"role": "user", "content": "hi"}])
        assert out == f"{_IM_START}{_COND_DIRECT}hi{_IM_END}"

    def test_thinking_via_chat_template_kwargs(self):
        eng = self._engine()
        out = eng._render_prompt(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        assert out == f"{_IM_START}{_COND_THINK}hi{_IM_END}"

    def test_thinking_via_engine_default(self):
        eng = self._engine(enable_thinking=True)
        out = eng._render_prompt([{"role": "user", "content": "hi"}])
        assert _COND_THINK in out and _COND_DIRECT not in out

    def test_multi_message_joined_by_newline(self):
        eng = self._engine()
        out = eng._render_prompt(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ]
        )
        assert out == f"{_IM_START}{_COND_DIRECT}be brief\nhello{_IM_END}"

    def test_chat_template_kwargs_override_engine_default(self):
        eng = self._engine(enable_thinking=True)
        out = eng._render_prompt(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        assert _COND_DIRECT in out and _COND_THINK not in out


class TestMsgText:
    def test_str_content(self):
        assert _msg_text("hello") == "hello"

    def test_list_of_parts(self):
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _msg_text(content) == "ab"

    def test_none(self):
        assert _msg_text(None) == ""
