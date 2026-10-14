# SPDX-License-Identifier: Apache-2.0
"""Cross-thread contract of the SSD cache store (upstream 74d71687).

The engine's generation stream is thread-local. An array still lazy on it
cannot be materialized from the omlx-store-cache worker thread; MLX raises
"There is no Stream(gpu, N) in current thread", which inside the buffer
protocol used to abort the process (production crash at
paged_ssd_cache.py _extract_tensor_bytes). Once the owner thread has
evaluated the array, the worker can slice and serialize it.

Runs in a subprocess so a regression that aborts cannot take pytest down.
"""

import subprocess
import sys
import textwrap

import pytest

mx = pytest.importorskip("mlx.core")

_SCRIPT = textwrap.dedent(
    """
    import sys, threading
    import mlx.core as mx
    from omlx.cache.paged_ssd_cache import _extract_tensor_bytes

    evaluate = sys.argv[1] == "eval"
    box = {}
    ready, done = threading.Event(), threading.Event()

    def owner():
        mx.new_thread_local_stream(mx.gpu)
        stream = mx.new_thread_local_stream(mx.gpu)
        with mx.stream(stream):
            a = (mx.ones((4, 64), dtype=mx.bfloat16) * 3)[:, :32]
            if evaluate:
                mx.eval(a)
        box["a"] = a
        ready.set()
        done.wait()

    t = threading.Thread(target=owner, daemon=True)
    t.start()
    ready.wait()
    try:
        raw, dtype, shape = _extract_tensor_bytes(box["a"][1:3])
        print("OK", dtype, shape, len(raw))
    except RuntimeError as e:
        print("ERR", e)
    finally:
        done.set()
    """
)


def _run(mode):
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT, mode],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_lazy_array_from_owner_stream_cannot_be_extracted_on_worker():
    proc = _run("lazy")
    assert "OK" not in proc.stdout, proc.stdout
    assert "no Stream" in proc.stdout + proc.stderr


def test_owner_evaluated_array_is_extracted_on_worker():
    proc = _run("eval")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[:3] == ["OK", "BF16", "[2,"]
