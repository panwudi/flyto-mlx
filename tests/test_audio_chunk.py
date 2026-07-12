# SPDX-License-Identifier: Apache-2.0
"""Unit tests for omlx.engine.audio_chunk (long-audio chunk planning).

Pure-numpy module: synthesize signals with known silence gaps and assert the
planner cuts at them, plus the repeat-loop detector calibration.
"""

import numpy as np
import pytest

from omlx.engine.audio_chunk import (
    _silence_midpoints,
    bisect_at_silence,
    concat_chunk_results,
    ngram_uniqueness,
    offset_result_times,
    plan_chunks,
)

SR = 8000


def _speech(seconds: float, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    return (rng.uniform(-amp, amp, size=n)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _build(segments: list[tuple[str, float]], seed: int = 0) -> np.ndarray:
    """segments: list of ('speech'|'silence', seconds) -> concatenated mono."""
    parts = []
    s = seed
    for kind, dur in segments:
        if kind == "speech":
            parts.append(_speech(dur, seed=s))
            s += 1
        else:
            parts.append(_silence(dur))
    return np.concatenate(parts)


# --------------------------------------------------------------------------
# ngram_uniqueness — the repeat-loop guard calibration
# --------------------------------------------------------------------------

class TestNgramUniqueness:
    def test_repeat_loop_scores_low(self):
        # The exact degeneration shape qwen3-asr collapses to.
        assert ngram_uniqueness("啊。" * 300) < 0.1

    def test_healthy_text_scores_high(self):
        rng = np.random.default_rng(1)
        text = "".join(chr(0x4e00 + int(rng.integers(0, 2000))) for _ in range(2000))
        assert ngram_uniqueness(text) > 0.9

    def test_short_text_returns_one(self):
        assert ngram_uniqueness("hi") == 1.0
        assert ngram_uniqueness("") == 1.0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            ngram_uniqueness("whatever", n=0)


# --------------------------------------------------------------------------
# _silence_midpoints
# --------------------------------------------------------------------------

class TestSilenceMidpoints:
    def test_detects_known_gaps(self):
        # speech 4.7s | silence 0.6s (centre 5.0) | speech 4.4s |
        #   silence 0.6s (centre 10.0) | speech 4.7s
        mono = _build([
            ("speech", 4.7), ("silence", 0.6),
            ("speech", 4.4), ("silence", 0.6),
            ("speech", 4.7),
        ])
        mids = _silence_midpoints(mono, SR)
        assert len(mids) == 2
        assert abs(mids[0] - 5.0) < 0.2
        assert abs(mids[1] - 10.0) < 0.2

    def test_ignores_micro_gaps(self):
        # 0.1s gap is below the 0.35s minimum -> not a split point.
        mono = _build([("speech", 3.0), ("silence", 0.1), ("speech", 3.0)])
        assert _silence_midpoints(mono, SR) == []

    def test_all_silence_returns_empty(self):
        assert _silence_midpoints(_silence(5.0), SR) == []


# --------------------------------------------------------------------------
# plan_chunks
# --------------------------------------------------------------------------

class TestPlanChunks:
    def test_short_audio_single_chunk(self):
        mono = _speech(10.0)
        chunks = plan_chunks(mono, SR, target_s=6.0, max_s=20.0)
        assert chunks == [(0.0, pytest.approx(10.0, abs=0.05))]

    def test_covers_full_range_no_gaps(self):
        mono = _build([
            ("speech", 4.7), ("silence", 0.6),
            ("speech", 4.4), ("silence", 0.6),
            ("speech", 4.4), ("silence", 0.6),
            ("speech", 4.7),
        ])
        total = mono.shape[0] / SR
        chunks = plan_chunks(mono, SR, target_s=6.0, max_s=8.0)
        assert len(chunks) >= 2
        # contiguous coverage
        assert chunks[0][0] == 0.0
        assert chunks[-1][1] == pytest.approx(total, abs=0.05)
        for a, b in zip(chunks, chunks[1:]):
            assert a[1] == pytest.approx(b[0], abs=1e-6)
        # every window within the hard cap
        for s, e in chunks:
            assert 0 < (e - s) <= 8.0 + 1e-6

    def test_cuts_snap_to_silence(self):
        # Ideal boundaries at 5, 10, 15 for a 20s / target-5 plan; put silence
        # gaps right there and assert the planner lands on them.
        mono = _build([
            ("speech", 4.7), ("silence", 0.6),   # centre 5.0
            ("speech", 4.4), ("silence", 0.6),   # centre 10.0
            ("speech", 4.4), ("silence", 0.6),   # centre 15.0
            ("speech", 4.7),
        ])
        chunks = plan_chunks(mono, SR, target_s=5.0, max_s=7.0)
        cuts = [c[0] for c in chunks[1:]]
        # 3 interior cuts, each near a silence centre
        assert len(cuts) == 3
        for cut, expected in zip(cuts, (5.0, 10.0, 15.0)):
            assert abs(cut - expected) < 0.5

    def test_no_silence_still_valid_plan(self):
        # Continuous speech, no pauses: planner falls back to even cuts and
        # must still return a valid contiguous cover within max_s.
        mono = _speech(20.0)
        total = mono.shape[0] / SR
        chunks = plan_chunks(mono, SR, target_s=6.0, max_s=8.0)
        assert chunks[0][0] == 0.0
        assert chunks[-1][1] == pytest.approx(total, abs=0.05)
        for s, e in chunks:
            assert 0 < (e - s) <= 8.0 + 1e-6


# --------------------------------------------------------------------------
# bisect_at_silence
# --------------------------------------------------------------------------

class TestOffsetAndConcat:
    def test_offset_shifts_segments_and_words(self):
        res = {
            "text": "hello",
            "segments": [{
                "start": 1.0, "end": 3.0,
                "words": [{"word": "hi", "start": 1.0, "end": 1.5}],
            }],
        }
        offset_result_times(res, 100.0)
        seg = res["segments"][0]
        assert seg["start"] == 101.0 and seg["end"] == 103.0
        assert seg["words"][0]["start"] == 101.0
        assert seg["words"][0]["end"] == 101.5

    def test_offset_tolerates_missing_times(self):
        res = {"segments": [{"text": "x"}]}  # no start/end/words
        offset_result_times(res, 5.0)  # must not raise
        assert res["segments"][0]["text"] == "x"

    def test_concat_merges_text_segments_duration(self):
        a = {"text": "前半", "language": "zh",
             "segments": [{"start": 0.0, "end": 10.0}]}
        b = {"text": "后半", "language": None,
             "segments": [{"start": 10.0, "end": 20.0}]}
        out = concat_chunk_results([a, b], total_duration=20.0, language="zh")
        assert out["text"] == "前半后半"
        assert out["duration"] == 20.0
        assert out["language"] == "zh"
        assert len(out["segments"]) == 2

    def test_concat_skips_empty_text(self):
        a = {"text": "", "segments": []}
        b = {"text": "有内容", "segments": [{"start": 0.0, "end": 1.0}]}
        out = concat_chunk_results([a, b], total_duration=5.0, language=None)
        assert out["text"] == "有内容"


class TestBisectAtSilence:
    def test_returns_interior_cut_near_middle(self):
        mono = _build([("speech", 4.7), ("silence", 0.6), ("speech", 4.7)])
        cut = bisect_at_silence(mono, SR)
        assert cut is not None
        assert abs(cut - 5.0) < 0.3

    def test_short_buffer_returns_none(self):
        assert bisect_at_silence(_speech(1.0), SR) is None

    def test_no_silence_falls_back_to_midpoint(self):
        mono = _speech(10.0)
        cut = bisect_at_silence(mono, SR)
        assert cut == pytest.approx(5.0, abs=0.05)
