# SPDX-License-Identifier: Apache-2.0
"""Skip the decode-step attention mask for a single unpadded sequence.

mlx-lm's ``BatchKVCache.make_mask`` always builds an array mask from
``left_padding``, even when the batch holds one sequence with no padding and
the step decodes one token. That mask is all True, so it is equivalent to
``mask=None``, but the masked SDPA kernel is slower and the gap grows with the
key length. Stock single-request decode (``KVCache``) passes ``None``.

Measured on M2 Max with the Qwen3.6-35B-A3B attention shape (10 full-attention
layers, 16 q / 2 kv heads, head_dim 256, one mask shared by all layers):
0 ms at 0.8K keys, +0.8 ms per token at 25K, +1.4 ms at 50K, +2.8 ms at 100K.

Whether the sequence is padded is tracked on the host. Reading
``left_padding`` (an ``mx.array``) in ``make_mask`` would force a GPU sync on
every decode step and cost more than the mask. The flag is set only on paths
whose padding is known to be zero and cleared on every other mutation, so an
unknown state keeps the stock mask.

The GatedDeltaNet layers need no change: the scheduler prefills each request
outside the batch, so the merged ``ArraysCache`` has neither ``left_padding``
nor ``lengths`` and its mask is already ``None``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_FLAG = "_omlx_unpadded"
_MARKER = "_omlx_decode_mask_patched"


def _all_zero(values: Any) -> bool:
    if values is None:
        return True
    if not isinstance(values, (list, tuple)):
        return False
    return all(int(v) == 0 for v in values)


def apply_batch_decode_mask_patch() -> bool:
    """Install the fast path on ``mlx_lm.models.cache.BatchKVCache``.

    Idempotent. Returns True when the patch is (already) installed.
    """
    try:
        from mlx_lm.models.cache import BatchKVCache
    except ImportError:  # pragma: no cover - mlx-lm always present
        return False

    if getattr(BatchKVCache, _MARKER, False):
        return True

    orig_init = BatchKVCache.__init__
    orig_prepare = BatchKVCache.prepare
    orig_filter = BatchKVCache.filter
    orig_extend = BatchKVCache.extend
    orig_make_mask = BatchKVCache.make_mask
    orig_state = BatchKVCache.state

    def __init__(self, left_padding, *args, **kwargs):
        orig_init(self, left_padding, *args, **kwargs)
        setattr(
            self,
            _FLAG,
            isinstance(left_padding, (list, tuple))
            and len(left_padding) == 1
            and _all_zero(left_padding),
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        orig_prepare(
            self,
            left_padding=left_padding,
            lengths=lengths,
            right_padding=right_padding,
        )
        if not _all_zero(left_padding) or not _all_zero(right_padding):
            setattr(self, _FLAG, False)

    def filter(self, batch_indices):
        orig_filter(self, batch_indices)
        # The stock filter shifts every row left by the smallest padding, so
        # a single surviving row always ends with zero padding.
        setattr(self, _FLAG, self.left_padding.shape[0] == 1)

    def extend(self, other):
        orig_extend(self, other)
        setattr(self, _FLAG, False)

    def _set_state(self, v):
        orig_state.fset(self, v)
        setattr(self, _FLAG, False)

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        if (
            N == 1
            and not return_array
            and kwargs.get("window_size") is None
            and getattr(self, _FLAG, False)
            and self.left_padding.shape[0] == 1
        ):
            return None
        return orig_make_mask(self, N, return_array=return_array, **kwargs)

    BatchKVCache.__init__ = __init__
    BatchKVCache.prepare = prepare
    BatchKVCache.filter = filter
    BatchKVCache.extend = extend
    BatchKVCache.state = property(orig_state.fget, _set_state)
    BatchKVCache.make_mask = make_mask
    setattr(BatchKVCache, _MARKER, True)
    logger.info("BatchKVCache single-sequence decode mask fast path installed")
    return True
