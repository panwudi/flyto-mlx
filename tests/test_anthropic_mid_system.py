# SPDX-License-Identifier: Apache-2.0
"""Mid-conversation system messages on the Anthropic path.

Claude Code 2.1.296 (anthropic-beta mid-conversation-system-2026-04-07) puts
role="system" entries inside messages[]. They become ``[System note]`` text
on an adjacent user turn so the rendered prompt keeps its prefix from one
turn to the next. Hoisting them to the front would rewrite the prompt head
whenever a new one appears and force a full re-prefill.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from omlx.api.anthropic_models import AnthropicMessage, MessagesRequest
from omlx.api.anthropic_utils import convert_anthropic_to_internal

_QWEN38_TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen38_chat_template.jinja"
).read_text()

ENV_BLOCK = "# Environment\nPlatform: darwin\ncwd: /tmp"
REMINDER = "<system-reminder>Todo list is empty.</system-reminder>"


def _render(messages, tools=None):
    """Render like transformers' apply_chat_template (jinja2 sandbox)."""

    def raise_exception(message):
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = (
        lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False:
        json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                   separators=separators, sort_keys=sort_keys)
    )
    env.globals["raise_exception"] = raise_exception
    return env.from_string(_QWEN38_TEMPLATE).render(
        messages=messages, tools=tools, add_generation_prompt=True
    )


def _request(messages, system="You are Claude Code."):
    return MessagesRequest(
        model="qwen3.8", max_tokens=64, system=system, messages=messages
    )


def _turn1():
    return [
        AnthropicMessage(role="user", content="1+1=?"),
        AnthropicMessage(role="system", content=ENV_BLOCK),
    ]


def _turn2():
    return _turn1() + [
        AnthropicMessage(role="assistant", content="2"),
        AnthropicMessage(role="user", content="and 2+2?"),
        AnthropicMessage(role="system", content=REMINDER),
    ]


def test_first_request_shape_renders_on_qwen38():
    """The exact shape that returned 422: messages roles [user, system]."""
    messages = convert_anthropic_to_internal(_request(_turn1()))

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == (
        f"1+1=?\n\n[System note]\n{ENV_BLOCK}\n[/System note]"
    )
    _render(messages)  # must not raise


def test_prefix_is_stable_across_turns():
    """Turn 2 must render as an extension of turn 1 up to user1's end.

    This is what distinguishes notes from hoisting: hoisting puts the
    turn-2 reminder into the leading system block, so the prompt changes
    from its first message on.
    """
    first = _render(convert_anthropic_to_internal(_request(_turn1())))
    second = _render(convert_anthropic_to_internal(_request(_turn2())))

    end_of_user1 = first.index("<|im_end|>", first.index("1+1=?")) + len("<|im_end|>")
    assert second.startswith(first[:end_of_user1])
    assert REMINDER in second[end_of_user1:]


def test_markers_are_stripped_before_becoming_a_note():
    messages = convert_anthropic_to_internal(
        _request(
            [
                AnthropicMessage(role="user", content="hi"),
                AnthropicMessage(
                    role="system",
                    content="note\n\n<total_tokens>900 tokens left</total_tokens>",
                ),
            ]
        )
    )

    assert messages[-1]["content"] == "hi\n\n[System note]\nnote\n[/System note]"


_NATIVE_TOOLS = SimpleNamespace(has_tool_calling=True)
ENV_UPDATE = (
    "# Environment update\n - Primary working directory: /tmp/sub (was /tmp)"
)


def _cd_turn():
    """Claude Code after a Bash cd: env update lands right after the result."""
    tool_use = {
        "type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cd sub"},
    }
    tool_result = {"type": "tool_result", "tool_use_id": "t1", "content": "/tmp/sub"}
    return _turn1() + [
        AnthropicMessage(role="assistant", content=[tool_use]),
        AnthropicMessage(role="user", content=[tool_result]),
    ]


@pytest.mark.parametrize("tokenizer", [None, _NATIVE_TOOLS])
def test_system_after_tool_result_becomes_note_on_the_result(tokenizer):
    messages = convert_anthropic_to_internal(
        _request(_cd_turn() + [AnthropicMessage(role="system", content=ENV_UPDATE)]),
        tokenizer=tokenizer,
    )

    assert [m["role"] for m in messages].count("system") == 1
    assert messages[0]["content"] == "You are Claude Code."
    assert messages[-1]["role"] == ("tool" if tokenizer else "user")
    assert messages[-1]["content"].endswith(
        f"/tmp/sub\n\n[System note]\n{ENV_UPDATE}\n[/System note]"
    )
    _render(messages)  # must not raise


def test_prefix_is_stable_across_a_cd():
    """The env update after a cd must not rewrite the prompt head.

    Hoisting it moved every inline system block into the leading system
    message, so a 150K-token Claude Code session re-prefilled from the end
    of the system prompt on every cd.
    """
    tools = [{"name": "Bash", "input_schema": {"type": "object"}}]
    before = convert_anthropic_to_internal(
        _request(_cd_turn()), tokenizer=_NATIVE_TOOLS
    )
    after = convert_anthropic_to_internal(
        _request(_cd_turn() + [AnthropicMessage(role="system", content=ENV_UPDATE)]),
        tokenizer=_NATIVE_TOOLS,
    )
    first, second = _render(before, tools), _render(after, tools)

    result_at = first.index("/tmp/sub") + len("/tmp/sub")
    assert second.startswith(first[:result_at])
    assert ENV_UPDATE in second[result_at:]


def test_system_next_to_image_turn_falls_back_to_hoisting():
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }
    messages = convert_anthropic_to_internal(
        _request(
            [
                AnthropicMessage(
                    role="user", content=[{"type": "text", "text": "see"}, image]
                ),
                AnthropicMessage(role="system", content=REMINDER),
            ]
        ),
        preserve_images=True,
    )

    assert messages[0]["role"] == "system"
    assert REMINDER in messages[0]["content"]
    assert [m["role"] for m in messages].count("system") == 1


@pytest.mark.parametrize("system", [None, "You are Claude Code."])
def test_leading_inline_system_joins_canonical_system(system):
    messages = convert_anthropic_to_internal(
        _request(
            [
                AnthropicMessage(role="system", content="Inline lead."),
                AnthropicMessage(role="user", content="hi"),
            ],
            system=system,
        )
    )

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].endswith("Inline lead.")
    assert messages[1] == {"role": "user", "content": "hi"}
