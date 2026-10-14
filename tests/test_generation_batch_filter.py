# SPDX-License-Identifier: Apache-2.0
"""GenerationBatch.filter keeps samplers / logits_processors row-aligned.

Backport of mlx-lm e2f2fb2. At our pin, filter only shrinks these lists
when some row has an entry. Rows without processors ([] per row) are the
common case, so after one finishes the lists keep the removed row, and the
next request appended by extend reads a stale row: its grammar and
penalties are silently skipped.
"""

import mlx.core as mx

import omlx.scheduler  # noqa: F401  installs the patch
from mlx_lm.generate import GenerationBatch


def _batch(uids, samplers, processors):
    b = GenerationBatch.__new__(GenerationBatch)
    n = len(uids)
    b.uids = list(uids)
    b.prompt_cache = []
    b.tokens = [[] for _ in range(n)]
    b.samplers = list(samplers)
    b.logits_processors = list(processors)
    b.max_tokens = [16] * n
    b.state_machines = [None] * n
    b._next_tokens = mx.zeros((n,), dtype=mx.int32)
    b._next_logprobs = [None] * n
    b._token_context = [None] * n
    b._num_tokens = [0] * n
    b._matcher_states = [None] * n
    return b


def test_filter_shrinks_empty_processor_rows():
    b = _batch([0, 1], [None, None], [[], []])
    b.filter([1])
    assert b.uids == [1]
    assert b.samplers == [None]
    assert b.logits_processors == [[]]


def test_next_request_after_filter_gets_its_own_processors():
    grammar = object()
    b = _batch([0, 1], [None, None], [[], []])
    b.filter([1])
    # What extend does for a newly inserted request with a grammar.
    b.uids.append(2)
    b.samplers.append(None)
    b.logits_processors.append([grammar])
    row = b.uids.index(2)
    assert b.logits_processors[row] == [grammar]


def test_filter_with_processors_unchanged():
    p = object()
    b = _batch([0, 1, 2], ["s0", None, "s2"], [[p], [], []])
    b.filter([0, 2])
    assert b.samplers == ["s0", "s2"]
    assert b.logits_processors == [[p], []]
