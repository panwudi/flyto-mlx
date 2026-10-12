# SPDX-License-Identifier: Apache-2.0
"""Canvas diffusion detection across mlx-vlm's two router contracts."""

from types import SimpleNamespace

import pytest

from omlx.engine.vlm import VLMBatchedEngine


def _engine(canvas_length):
    config = SimpleNamespace(canvas_length=canvas_length)
    return SimpleNamespace(
        _vlm_model=SimpleNamespace(config=config), _model_name="m"
    )


@pytest.mark.parametrize(
    ("family", "canvas_length", "expected"),
    [
        ("diffusion", 256, "block"),  # mlx-vlm >= 0.7, canvas model
        ("diffusion", None, None),  # mlx-vlm >= 0.7, model-owned generator
        ("block", 256, "block"),  # mlx-vlm <= 0.6
        (None, None, None),  # not a diffusion model
    ],
)
def test_detect_diffusion_family(monkeypatch, family, canvas_length, expected):
    import importlib

    diffusion_mod = importlib.import_module("mlx_vlm.generate.diffusion")

    monkeypatch.setattr(
        diffusion_mod, "diffusion_generation_family", lambda model: family
    )
    engine = _engine(canvas_length)
    assert VLMBatchedEngine._detect_diffusion_family(engine) == expected
