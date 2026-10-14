# SPDX-License-Identifier: Apache-2.0
"""Behavior the xgrammar pin (==0.2.8) is there for.

xgrammar 0.2.1 accepted a wrong-type value for an optional typed property
when the schema allows additional properties (xgrammar #595), so
response_format json_schema output could violate the declared type.
"""

import json

import pytest

xgr = pytest.importorskip("xgrammar")


def _accepts(schema, text):
    grammar = xgr.Grammar.from_json_schema(json.dumps(schema))
    return xgr.testing._is_grammar_accept_string(grammar, text)


SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "additionalProperties": True,
}


@pytest.mark.parametrize(
    "text, ok",
    [
        ('{"n": 1}', True),
        ('{"n": "x"}', False),
        ('{"m": "x"}', True),
        ('{"n": 1, "m": [1]}', True),
    ],
)
def test_optional_typed_property_with_additional_properties(text, ok):
    assert _accepts(SCHEMA, text) is ok
