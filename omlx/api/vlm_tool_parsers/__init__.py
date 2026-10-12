# SPDX-License-Identifier: Apache-2.0
"""Tool-parser lookup for VLM tokenizers.

mlx-vlm <= 0.6 shipped ``mlx_vlm.tool_parsers``: mlx-lm's template inference
plus two extra markers, with its own Gemma 4 and Cohere2 MoE parsers taking
precedence over mlx-lm's. mlx-vlm 0.7 removed the package, so the two parsers
are vendored here and the lookup keeps the same precedence.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

_EXTRA_PATTERNS = (
    ("<|tool_call>", "gemma4"),
    ("<|START_ACTION|>", "cohere2_moe"),
)
_VENDORED = frozenset(name for _, name in _EXTRA_PATTERNS)


def infer_tool_parser(tokenizer: Any) -> Optional[str]:
    """mlx-lm's inference first, then the markers mlx-vlm used to add."""
    from mlx_lm.tokenizer_utils import _infer_tool_parser

    result = _infer_tool_parser(tokenizer)
    if result is not None:
        return result
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        return None
    for marker, parser_type in _EXTRA_PATTERNS:
        if marker in chat_template:
            return parser_type
    return None


def load_tool_module(tool_parser_type: str) -> Any:
    """The vendored parser when there is one, else mlx-lm's."""
    if tool_parser_type in _VENDORED:
        return importlib.import_module(f"{__name__}.{tool_parser_type}")
    return importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
