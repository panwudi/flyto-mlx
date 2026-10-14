# SPDX-License-Identifier: Apache-2.0
"""Tool search (deferred tools) on the Anthropic path.

With ``ENABLE_TOOL_SEARCH`` Claude Code sends rarely used tools with
``defer_loading: true`` and loads one by answering its ToolSearch call with a
``tool_reference`` block. The request shapes below are trimmed from a real
Claude Code 2.1.296 capture. Deferred tools must stay out of the template's
tool list (the prompt head) and reach the model inline instead; otherwise
every newly loaded tool rewrites the head and forces a full re-prefill.
"""

import json
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from omlx.api.anthropic_models import MessagesRequest
from omlx.api.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_to_internal_harmony,
    convert_anthropic_tools_to_internal,
)

_QWEN38_TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen38_chat_template.jinja"
).read_text()

READ = {
    "name": "Read",
    "description": "Read a file.",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}
TOOL_SEARCH = {
    "name": "ToolSearch",
    "description": "Fetches full schema definitions for deferred tools.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
PLACEHOLDER = {
    "name": "DeferredToolPlaceholder",
    "description": "Reserved placeholder that keeps deferred tool loading active; never call this tool.",
    "input_schema": {"type": "object", "properties": {}},
    "defer_loading": True,
}
MACHO = {
    "name": "mcp__rea__inspect_macho",
    "description": "Inspect Mach-O load commands.",
    "input_schema": {
        "type": "object",
        "properties": {"binary": {"type": "string"}},
        "required": ["binary"],
    },
    "defer_loading": True,
}
STRINGS = {
    "name": "mcp__rea__list_strings",
    "description": "List strings in a binary.",
    "input_schema": {"type": "object", "properties": {}},
    "defer_loading": True,
}


def _render(messages, tools=None):
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


def _turn1():
    return [{"role": "user", "content": "inspect /bin/ls"}]


def _turn2():
    return _turn1() + [
        {"role": "assistant", "content": [{
            "type": "tool_use", "id": "toolu_1", "name": "ToolSearch",
            "input": {"query": "select:mcp__rea__inspect_macho"},
        }]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                {"type": "tool_reference", "tool_name": "mcp__rea__inspect_macho"},
            ]},
            {"type": "text", "text": "Tool loaded."},
        ]},
    ]


def _request(messages, tools):
    return MessagesRequest(
        model="qwen3.8", max_tokens=64, system="You are Claude Code.",
        messages=messages, tools=tools,
    )


def _names(tools):
    return [t["function"]["name"] for t in tools or []]


def test_deferred_tools_split_between_template_and_parser():
    tools = _request([], [READ, TOOL_SEARCH, PLACEHOLDER, MACHO]).tools

    assert _names(convert_anthropic_tools_to_internal(tools)) == [
        "Read", "ToolSearch", "DeferredToolPlaceholder", "mcp__rea__inspect_macho",
    ]
    assert _names(
        convert_anthropic_tools_to_internal(tools, include_deferred=False)
    ) == ["Read", "ToolSearch"]
    assert _names(
        convert_anthropic_tools_to_internal(tools, only_deferred=True)
    ) == ["DeferredToolPlaceholder", "mcp__rea__inspect_macho"]


def test_tool_reference_expands_to_function_definition():
    messages = convert_anthropic_to_internal(
        _request(_turn2(), [READ, TOOL_SEARCH, PLACEHOLDER, MACHO])
    )

    tool_result = messages[-1]["content"]
    expected = json.dumps({
        "description": MACHO["description"],
        "name": MACHO["name"],
        "parameters": MACHO["input_schema"],
    }, ensure_ascii=False)
    assert f"<functions>\n<function>{expected}</function>\n</functions>" in tool_result
    assert tool_result.endswith("Tool loaded.")


def test_consecutive_references_share_one_block_and_unknown_is_reported():
    messages = _turn1() + [
        {"role": "assistant", "content": [{
            "type": "tool_use", "id": "toolu_1", "name": "ToolSearch",
            "input": {"query": "rea"},
        }]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                {"type": "tool_reference", "tool_name": "mcp__rea__inspect_macho"},
                {"type": "tool_reference", "tool_name": "mcp__rea__list_strings"},
                {"type": "tool_reference", "tool_name": "mcp__rea__gone"},
            ]},
        ]},
    ]
    out = convert_anthropic_to_internal(
        _request(messages, [READ, TOOL_SEARCH, MACHO, STRINGS])
    )[-1]["content"]

    assert out.count("<functions>") == 1
    assert out.count("<function>") == 2
    assert (
        "Tool mcp__rea__gone was referenced but its definition is not in this request."
        in out
    )


def test_harmony_path_expands_references_too():
    messages = convert_anthropic_to_internal_harmony(
        _request(_turn2(), [READ, TOOL_SEARCH, PLACEHOLDER, MACHO])
    )
    tool_messages = [m for m in messages if m["role"] == "tool"]

    assert '"name": "mcp__rea__inspect_macho"' in tool_messages[-1]["content"]


def test_loading_a_tool_keeps_the_rendered_prefix_on_qwen38():
    """Turn 1 knows no rea tool, turn 2 has loaded one: the head must not move."""
    req1 = _request(_turn1(), [READ, TOOL_SEARCH, PLACEHOLDER])
    req2 = _request(_turn2(), [READ, TOOL_SEARCH, PLACEHOLDER, MACHO])

    prompt1 = _render(
        convert_anthropic_to_internal(req1),
        convert_anthropic_tools_to_internal(req1.tools, include_deferred=False),
    )
    prompt2 = _render(
        convert_anthropic_to_internal(req2),
        convert_anthropic_tools_to_internal(req2.tools, include_deferred=False),
    )

    head = prompt1.rsplit("<|im_start|>assistant", 1)[0]
    assert prompt2.startswith(head)
    # The loaded tool is described in the tool result, not in the tool list.
    tool_list = prompt2.split("</tools>", 1)[0]
    assert "mcp__rea__inspect_macho" not in tool_list
    assert "DeferredToolPlaceholder" not in tool_list
    assert '"name": "mcp__rea__inspect_macho"' in prompt2
