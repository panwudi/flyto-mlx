# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.compile_cache (thread-local MLX compile-cache clear)."""

import concurrent.futures as cf

import mlx.core as mx

import omlx.utils.compile_cache as cc


def test_clear_available_returns_bool():
    # Contract: never raises, always returns a bool. On a real Apple-Silicon
    # MLX install the symbol resolves, but the helper must degrade gracefully.
    assert isinstance(cc.compile_cache_clear_available(), bool)


def test_clear_thread_compile_cache_does_not_raise():
    # Safe to call on any thread, even with an empty cache.
    cc.clear_thread_compile_cache()


def test_clear_after_compile_on_worker_then_shutdown():
    """Core scenario behind the fix: run an @mx.compile fn on a worker thread,
    clear that thread's cache on the same thread, then shut the worker down.
    Must not crash (~CompilerCache runs on an empty cache)."""

    @mx.compile
    def f(x):
        return x * 2 + 1

    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        ex.submit(lambda: mx.eval(f(mx.arange(8)))).result()
        ex.submit(cc.clear_thread_compile_cache).result()
    finally:
        ex.shutdown(wait=True)


def test_noop_when_symbol_unavailable(monkeypatch):
    """When the symbol cannot be resolved, available() is False and clear() is
    a no-op (callers then fall back to keeping the worker thread alive)."""
    monkeypatch.setattr(cc, "_resolved", True)
    monkeypatch.setattr(cc, "_clear_fn", None)
    assert cc.compile_cache_clear_available() is False
    cc.clear_thread_compile_cache()  # must not raise


_THREAD_EXIT_SCRIPT = """
import concurrent.futures as cf
import sys

import mlx.core as mx

import omlx.utils.compile_cache as cc

# Multi-output (tuple / dict) compiled graphs are what crash at thread exit:
# their cache entries hold Python output-structure objects. A single-array
# output does not reproduce the crash.
pair = mx.compile(lambda a, b: (a + b, a * b))
named = mx.compile(lambda a: {"k": a + 1})


def work():
    stream = mx.new_thread_local_stream(mx.default_device())
    with mx.stream(stream):
        a = mx.ones((4,))
        mx.eval(pair(a, a))
        mx.eval(named(a)["k"])


for _ in range(3):
    ex = cf.ThreadPoolExecutor(max_workers=1)
    ex.submit(work).result()
    if sys.argv[1] == "clear":
        ex.submit(cc.clear_thread_compile_cache).result()
    ex.shutdown(wait=True)
print("survived")
"""


def _run_thread_exit_script(mode):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", _THREAD_EXIT_SCRIPT, mode],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_multi_output_compile_survives_thread_exit_after_clear():
    """Regression for the 2026-10-10 m5max unload SIGSEGV.

    Without the clear, the worker thread's ~CompilerCache frees the cached
    tuple/dict output structures without the GIL and the process dies with
    SIGSEGV. The clear must make the same sequence exit cleanly. Runs in a
    subprocess because the failure mode kills the interpreter.
    """
    if not cc.compile_cache_clear_available():
        import pytest

        pytest.skip("MLX compile-cache clear symbol unavailable")

    # Canary: on the MLX we ship (0.31.2) the unfixed run dies with SIGSEGV,
    # which proves this test exercises the real hazard. A future MLX may fix
    # the destructor, so a clean exit there is not a failure.
    unfixed = _run_thread_exit_script("noclear")
    assert unfixed.returncode in (0, -11, 139), unfixed.stderr[-2000:]

    fixed = _run_thread_exit_script("clear")
    assert fixed.returncode == 0, fixed.stderr[-2000:]
    assert "survived" in fixed.stdout
