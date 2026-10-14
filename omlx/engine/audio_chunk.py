# SPDX-License-Identifier: Apache-2.0
"""Long-audio chunk planning for the STT transcription pipeline.

Qwen3-ASR (and similar decoder-only ASR models) transcribe a single real
call cleanly up to ~30 min, then degenerate past ~40 min: the decoder loses
the thread and emits a repeated-token loop (e.g. a Chinese call collapses to
"啊。啊。啊。"), dropping the back half of the real content. Raising
max_tokens or repetition_penalty only reshapes the garbage -- the model has
lost the thread, not run out of budget.

The fix is to split long audio at natural silence pauses into windows the
model handles cleanly, transcribe each independently (crucially WITHOUT any
previous-text context -- condition-on-previous-text is exactly what seeds the
Whisper-family repeat loop), and concatenate with per-window time offsets.

This module holds the pure, engine-free pieces of that pipeline:

* ``plan_chunks``      -- silence-aware split-point planning (numpy only).
* ``ngram_uniqueness`` -- the repeat-loop detector used as a per-chunk guard.
* ``bisect_at_silence``-- re-split point for a chunk the guard flags as runaway.

Silence detection is a small self-contained energy analyzer (frame RMS +
relative threshold). It is intentionally NOT pyannote (400 MB, gated) nor the
per-word RMS in engine/diarize.py (that labels speakers, it does not find
pauses). Zero deps beyond numpy, in the same style as diarize.py.
"""

from __future__ import annotations

import math

import numpy as np


def ngram_uniqueness(text: str, n: int = 12) -> float:
    """Fraction of distinct character ``n``-grams in ``text``.

    A healthy transcript has near-1.0 uniqueness; a decoder repeat loop
    ("啊。啊。啊。...") collapses to a handful of distinct windows and scores
    ~0.04-0.09. ``n=12`` is the window the healthy/degenerate numbers were
    measured at on real qwen3-asr output -- keep it 12 so the caller's
    threshold stays calibrated.

    Text shorter than ``n`` characters returns 1.0 (nothing to loop on).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    L = len(text)
    if L < n:
        return 1.0
    grams = {text[i:i + n] for i in range(L - n + 1)}
    total = L - n + 1
    return len(grams) / total


def _frame_rms(
    mono: np.ndarray,
    sr: int,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, int]:
    """Return (rms_per_frame, hop_samples) via an O(N) sliding RMS.

    Uses a cumulative-sum-of-squares so a 40-min 8 kHz buffer costs one
    linear pass, not a Python loop over ~250k frames. float64 accumulation
    keeps precision over the long sum.
    """
    mono = np.asarray(mono, dtype=np.float64).ravel()
    n_samples = mono.shape[0]
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if n_samples < frame:
        # Whole buffer is one frame's worth or less.
        rms = np.sqrt(np.mean(mono ** 2)) if n_samples else 0.0
        return np.array([rms], dtype=np.float64), hop
    n_frames = 1 + (n_samples - frame) // hop
    csum = np.empty(n_samples + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(mono ** 2, out=csum[1:])
    starts = np.arange(n_frames) * hop
    ends = starts + frame
    energy = (csum[ends] - csum[starts]) / frame
    return np.sqrt(np.maximum(energy, 0.0)), hop


def _silence_midpoints(
    mono: np.ndarray,
    sr: int,
    min_silence_s: float = 0.35,
    rel_threshold: float = 0.12,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
) -> list[float]:
    """Times (seconds) of the centre of each silent gap in ``mono``.

    A frame is "silent" when its RMS is below ``rel_threshold`` times a loud
    reference (the 90th percentile of frame RMS -- robust to the odd click).
    A gap qualifies only when it runs at least ``min_silence_s`` -- long
    enough to be a real pause, not an inter-word micro-gap. Returns the gap
    centres, ascending; empty when the audio has no qualifying pause.
    """
    rms, hop = _frame_rms(mono, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if rms.shape[0] == 0:
        return []
    ref = float(np.percentile(rms, 90))
    if ref <= 0.0:
        # Effectively silent buffer; no meaningful split points.
        return []
    thr = ref * rel_threshold
    silent = rms < thr
    min_frames = max(1, int(round(min_silence_s * 1000.0 / hop_ms)))
    hop_s = hop / sr

    mids: list[float] = []
    run_start = -1
    for i, s in enumerate(silent):
        if s and run_start < 0:
            run_start = i
        elif not s and run_start >= 0:
            if i - run_start >= min_frames:
                centre_frame = (run_start + i - 1) / 2.0
                mids.append(centre_frame * hop_s)
            run_start = -1
    if run_start >= 0 and len(silent) - run_start >= min_frames:
        centre_frame = (run_start + len(silent) - 1) / 2.0
        mids.append(centre_frame * hop_s)
    return mids


def plan_chunks(
    mono: np.ndarray,
    sr: int,
    *,
    target_s: float,
    max_s: float,
    min_silence_s: float = 0.35,
    rel_threshold: float = 0.12,
) -> list[tuple[float, float]]:
    """Plan silence-aware transcription windows over a mono buffer.

    Returns an ordered list of ``(start_s, end_s)`` covering ``[0, total]``
    with no gaps or overlaps. When the audio is at or under ``max_s`` a single
    window is returned (no chunking). Otherwise the buffer is divided into
    ``ceil(total / target_s)`` roughly equal windows whose ideal boundaries
    are each snapped to the nearest real silence pause, then clamped so every
    window stays within ``[min_chunk, max_s]``. Equal division (rather than
    greedy target-length cuts) avoids a tiny final sliver.

    Args:
        mono: 1-D float audio (down-mix stereo before calling).
        sr: sample rate (Hz).
        target_s: desired window length in seconds (e.g. 1200 = 20 min).
        max_s: hard upper bound on any single window (e.g. 1500 = 25 min).
        min_silence_s / rel_threshold: silence-detection tuning, forwarded
            to ``_silence_midpoints``.
    """
    mono = np.asarray(mono).ravel()
    n_samples = mono.shape[0]
    total = n_samples / sr if sr else 0.0
    if total <= 0.0:
        return [(0.0, 0.0)]
    if target_s <= 0 or max_s <= 0:
        raise ValueError("target_s and max_s must be positive")
    if target_s > max_s:
        target_s = max_s
    if total <= max_s:
        return [(0.0, total)]

    n_chunks = max(2, math.ceil(total / target_s))
    ideal_len = total / n_chunks
    # Search radius around each ideal boundary: half a window, but never so
    # wide that a snapped cut could push a neighbouring window past max_s.
    radius = min(ideal_len * 0.5, max(0.0, max_s - ideal_len))
    min_chunk = min(ideal_len * 0.5, max_s * 0.5)

    mids = _silence_midpoints(
        mono, sr, min_silence_s=min_silence_s, rel_threshold=rel_threshold,
    )
    mids_arr = np.asarray(mids, dtype=np.float64) if mids else None

    boundaries: list[float] = [0.0]
    for i in range(1, n_chunks):
        ideal = i * ideal_len
        cut = ideal
        if mids_arr is not None:
            lo, hi = ideal - radius, ideal + radius
            window = mids_arr[(mids_arr >= lo) & (mids_arr <= hi)]
            if window.size:
                # Nearest silence centre to the ideal boundary.
                cut = float(window[np.argmin(np.abs(window - ideal))])
        prev = boundaries[-1]
        # Keep each window within [min_chunk, max_s] and leave room for the
        # remaining windows so the last one cannot exceed max_s either.
        lo_bound = prev + min_chunk
        hi_bound = prev + max_s
        remaining_after = n_chunks - i  # windows still to place after this cut
        room_bound = total - remaining_after * min_chunk
        cut = max(lo_bound, min(cut, hi_bound, room_bound))
        # Monotonic guard against pathological clamps.
        cut = max(cut, prev + 1e-3)
        cut = min(cut, total - 1e-3)
        boundaries.append(cut)
    boundaries.append(total)

    return [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] - boundaries[i] > 1e-6
    ]


def offset_result_times(result: dict, offset_s: float) -> dict:
    """Shift every timestamp in a transcribe-result dict by ``offset_s``.

    Chunk windows are transcribed on slices whose clock starts at 0; adding
    the slice's global start realigns segment- and word-level start/end onto
    the full-audio timeline. ``duration`` is the slice length and is left
    untouched (the caller stamps the full-audio duration on the merged result).
    Mutates and returns ``result``.
    """
    for seg in result.get("segments") or []:
        if "start" in seg and seg["start"] is not None:
            seg["start"] = float(seg["start"]) + offset_s
        if "end" in seg and seg["end"] is not None:
            seg["end"] = float(seg["end"]) + offset_s
        for w in seg.get("words") or []:
            if "start" in w and w["start"] is not None:
                w["start"] = float(w["start"]) + offset_s
            if "end" in w and w["end"] is not None:
                w["end"] = float(w["end"]) + offset_s
    return result


def concat_chunk_results(
    results: list[dict],
    total_duration: float | None,
    language,
) -> dict:
    """Merge per-window transcribe results (already time-offset) into one.

    ``text`` is the direct concatenation of per-window text (windows are cut
    at silence, so no word is split across a boundary); ``segments`` are
    concatenated in order; ``duration`` is the full-audio length; ``language``
    is the first window's non-null language, falling back to the argument.
    """
    text_parts: list[str] = []
    segments: list[dict] = []
    lang = None
    for r in results:
        t = r.get("text") or ""
        if t:
            text_parts.append(t)
        segments.extend(r.get("segments") or [])
        if lang is None:
            lang = r.get("language")
    return {
        "text": "".join(text_parts),
        "language": lang if lang is not None else language,
        "duration": total_duration,
        "segments": segments,
    }


def bisect_at_silence(
    mono: np.ndarray,
    sr: int,
    min_silence_s: float = 0.35,
    rel_threshold: float = 0.12,
) -> float | None:
    """Best split point (seconds) near the middle of ``mono``, or None.

    Used by the repeat-loop guard: when a chunk's transcript degenerates it
    is re-split and re-transcribed. Returns the silence-gap centre closest to
    the midpoint; falls back to the exact midpoint when no qualifying pause
    exists; returns None when the buffer is too short to split usefully.
    """
    mono = np.asarray(mono).ravel()
    total = mono.shape[0] / sr if sr else 0.0
    if total < 2.0:
        return None
    mid = total / 2.0
    mids = _silence_midpoints(
        mono, sr, min_silence_s=min_silence_s, rel_threshold=rel_threshold,
    )
    if mids:
        arr = np.asarray(mids, dtype=np.float64)
        cut = float(arr[np.argmin(np.abs(arr - mid))])
        # Only trust an interior cut; a silence centre hugging either edge
        # would produce a degenerate sub-chunk.
        if 0.1 * total < cut < 0.9 * total:
            return cut
    return mid
