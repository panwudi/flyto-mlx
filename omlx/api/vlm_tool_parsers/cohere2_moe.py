# Vendored from mlx-vlm v0.6.3 (5a4222a) mlx_vlm/tool_parsers/cohere2_moe.py, MIT License,
# Copyright (c) Prince Canuma. mlx-vlm 0.7 dropped the tool_parsers package;
# keeping this copy leaves VLM tool-call parsing unchanged across the upgrade.

import json

tool_call_start = "<|START_ACTION|>"
tool_call_end = "<|END_ACTION|>"


def _loads_action(text: str):
    text = text.strip()
    # Command sometimes emits invalid JSON escapes such as "\|" in regex strings.
    text = text.replace("\\|", "|")
    return json.loads(text)


def _convert_call(call):
    name = call.get("tool_name") or call.get("name")
    if not name:
        raise ValueError("Cohere tool call is missing tool_name.")
    arguments = call.get("parameters", {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {"name": name, "arguments": arguments}


def parse_tool_call(text, tools=None):
    action = _loads_action(text)
    if isinstance(action, dict):
        return _convert_call(action)
    if isinstance(action, list):
        return [_convert_call(call) for call in action]
    raise ValueError("Cohere tool action must be a JSON object or array.")
