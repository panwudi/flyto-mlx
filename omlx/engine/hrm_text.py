# SPDX-License-Identifier: Apache-2.0
"""HRM-Text engine (sapientinc/HRM-Text) for oMLX.

A standalone text-generation engine, NOT on the mlx-lm BatchGenerator path:
HRM-Text re-runs an H/L recurrence (8 transformer-stack passes for the 1B) per
generated token, over a 128-slot KV cache, with a PrefixLM mask (whole prompt
bidirectional, response causal). That does not fit mlx-lm's causal single-cache
generate, so this engine drives the proven functional model in
``omlx.models.hrm_text`` directly.

Design:
- All MLX work (load + every forward) runs on the single global MLX executor
  thread (``get_mlx_executor`` is max_workers=1), which serializes Metal ops
  across engines and avoids cross-thread stream races. One ``run_in_executor``
  per token gives true per-token streaming AND lets other engines interleave
  between our (relatively slow) tokens.
- Each request owns its own ``KVCache`` (the cache carries a single committed
  offset), so concurrent requests are independent; the executor serializes the
  actual compute. No continuous batching -- inference is ~8x a same-width plain
  transformer per token, inherent to HRM. Size leases accordingly.
- Sampling reuses mlx-lm's ``make_sampler`` / ``make_logits_processors`` so
  temperature/top_p/top_k/min_p/penalties behave identically to other engines.

Prompt format mirrors the released HrmTextProcessor (the checkpoint ships no
Jinja chat template): ``<|im_start|>`` + condition + content + ``<|im_end|>``,
condition = ``<|object_ref_start|>`` (direct) or ``<|quad_end|><|object_ref_end|>``
(thinking). The whole prompt is the bidirectional PrefixLM block. Stop on eos.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..models import hrm_text as hrm
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

# Prompt boundary tokens (literal strings; the tokenizer maps them to single
# ids). condition selects direct vs thinking/CoT generation.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_COND_DIRECT = "<|object_ref_start|>"
_COND_THINK = "<|quad_end|><|object_ref_end|>"
_DEFAULT_EOS = 11  # <|box_end|>


def _msg_text(content: Any) -> str:
    """Flatten a message's content to plain text (str or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or ""
                if txt:
                    parts.append(str(txt))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content)


class _GenState:
    """Per-request generation state, mutated on the MLX executor thread."""

    __slots__ = ("cache", "next_logits", "gen_ids", "sampler", "procs")

    def __init__(self, cache, next_logits, sampler, procs):
        self.cache = cache
        self.next_logits = next_logits  # [1, vocab] logits for the next token
        self.gen_ids: list[int] = []    # emitted (non-eos) tokens, for procs
        self.sampler = sampler
        self.procs = procs


class HrmTextEngine(BaseEngine):
    """oMLX engine for HRM-Text models (model_type == 'hrm_text')."""

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        scheduler_config: Any | None = None,
        enable_thinking: bool | None = None,
        model_settings: Any | None = None,
    ):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._enable_thinking = enable_thinking
        self._model_settings = model_settings

        self._model: hrm.HrmTextMLX | None = None
        self._cfg: dict | None = None
        self._tokenizer: Any = None
        self._eos_ids: set[int] = {_DEFAULT_EOS}
        self._loaded = False
        self._t_start = time.monotonic()

        self._active_lock = threading.Lock()
        self._active = 0

    # ----- properties -----
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        if self._cfg is not None:
            return self._cfg.get("model_type")
        return None

    @property
    def context_length(self) -> int | None:
        if self._cfg is not None:
            return self._cfg.get("max_position_embeddings")
        return None

    # ----- lifecycle -----
    async def start(self) -> None:
        if self._loaded:
            return
        from transformers import AutoTokenizer

        loop = asyncio.get_running_loop()

        def _load_sync():
            model, cfg = hrm.load(self._model_name)
            # Materialize weights on this (the MLX) thread so later forwards on
            # the same thread never trigger a cross-thread lazy eval.
            mx.eval(*[v for v in model.w.values()])
            return model, cfg

        self._model, self._cfg = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=self._trust_remote_code
        )

        eos = self._cfg.get("eos_token_id")
        ids = {_DEFAULT_EOS}
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(int(e) for e in eos)
        self._eos_ids = ids

        self._loaded = True
        logger.info(
            "HrmTextEngine loaded %s (n_slots=%d, eos=%s)",
            self._model_name, self._model.n_slots, sorted(self._eos_ids),
        )

    async def stop(self) -> None:
        self._model = None
        self._cfg = None
        self._tokenizer = None
        self._loaded = False
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info("HrmTextEngine stopped")

    # ----- prompt rendering -----
    def _render_prompt(
        self,
        messages: List[Dict[str, Any]],
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Render messages into the HRM-Text prompt string.

        Mirrors the released HrmTextProcessor: a single
        ``<|im_start|>{condition}{contents}<|im_end|>`` block, message contents
        joined by newline (the checkpoint was not trained with role markers).
        """
        thinking = self._enable_thinking or False
        if chat_template_kwargs and "enable_thinking" in chat_template_kwargs:
            thinking = bool(chat_template_kwargs["enable_thinking"])
        condition = _COND_THINK if thinking else _COND_DIRECT
        body = "\n".join(_msg_text(m.get("content")) for m in messages)
        return f"{_IM_START}{condition}{body}{_IM_END}"

    def _apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> str:
        return self._render_prompt(messages, chat_template_kwargs)

    def count_chat_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> int:
        prompt = self._render_prompt(messages, chat_template_kwargs)
        return len(self._tokenizer.encode(prompt, add_special_tokens=False))

    # ----- generation core (runs on the MLX executor thread) -----
    def _prefill(self, input_ids: list[int], sampler, procs) -> _GenState:
        cache = hrm.KVCache(self._model.n_slots)
        ids = mx.array([input_ids])
        P = ids.shape[1]
        mask = hrm.prefixlm_mask(P, P)  # whole prompt is the bidirectional prefix
        logits = self._model.forward_cached(ids, cache, mask=mask)
        next_logits = logits[:, -1, :]
        self._eval_state(next_logits, cache)
        return _GenState(cache, next_logits, sampler, procs)

    def _sample_and_advance(self, state: _GenState, is_last: bool) -> tuple[int, bool]:
        """Sample one token from the pending logits; unless stopping, feed it
        through the recurrence to produce the next token's logits. Returns
        (token_id, stop). eos tokens are not emitted/fed."""
        logits = state.next_logits
        if state.procs:
            toks = (
                mx.array(state.gen_ids, dtype=mx.int32)
                if state.gen_ids
                else mx.zeros((0,), dtype=mx.int32)
            )
            for proc in state.procs:
                logits = proc(toks, logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tok = state.sampler(logprobs)  # [1]
        tid = int(tok.item())
        if tid in self._eos_ids:
            return tid, True
        state.gen_ids.append(tid)
        if is_last:
            return tid, True
        nxt = self._model.forward_cached(tok.reshape(1, 1), state.cache, mask=None)
        state.next_logits = nxt[:, -1, :]
        self._eval_state(state.next_logits, state.cache)
        return tid, False

    @staticmethod
    def _eval_state(next_logits: mx.array, cache: "hrm.KVCache") -> None:
        # Force the lazy graph (logits + grown cache) to materialize on this
        # thread, so the cache does not accumulate an unbounded graph and is
        # never evaluated cross-thread.
        arrs = [a for a in (cache.k + cache.v) if a is not None]
        mx.eval(next_logits, *arrs)

    def _build_sampler(self, temperature, top_p, top_k, min_p):
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=float(temperature), top_p=float(top_p),
            top_k=int(top_k), min_p=float(min_p),
        )

    def _build_procs(self, repetition_penalty, presence_penalty):
        rp = repetition_penalty if repetition_penalty and repetition_penalty != 1.0 else None
        pp = presence_penalty if presence_penalty else None
        if rp is None and pp is None:
            return []
        from mlx_lm.sample_utils import make_logits_processors

        return make_logits_processors(repetition_penalty=rp, presence_penalty=pp)

    async def _stream_ids(
        self,
        input_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        stop: Optional[List[str]],
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        loop = asyncio.get_running_loop()
        ex = get_mlx_executor()
        sampler = self._build_sampler(temperature, top_p, top_k, min_p)
        procs = self._build_procs(repetition_penalty, presence_penalty)
        prompt_tokens = len(input_ids)
        stop_strs = [s for s in (stop or []) if s]

        with self._active_lock:
            self._active += 1
        t0 = time.monotonic()
        try:
            state = await loop.run_in_executor(ex, self._prefill, input_ids, sampler, procs)
            prev_text = ""
            for step in range(max_tokens):
                is_last = step == max_tokens - 1
                tid, stop_flag = await loop.run_in_executor(
                    ex, self._sample_and_advance, state, is_last
                )
                eos_hit = tid in self._eos_ids
                completion = len(state.gen_ids)
                tps = completion / max(time.monotonic() - t0, 1e-6)

                if eos_hit:
                    yield GenerationOutput(
                        text=prev_text, tokens=list(state.gen_ids),
                        prompt_tokens=prompt_tokens, completion_tokens=completion,
                        finish_reason="stop", new_text="", finished=True,
                        generation_tps=tps,
                    )
                    return

                full = self._tokenizer.decode(state.gen_ids)
                delta = full[len(prev_text):]
                prev_text = full

                finish_reason = None
                finished = False
                if stop_strs and any(s in full for s in stop_strs):
                    finished, finish_reason = True, "stop"
                elif is_last or stop_flag:
                    finished, finish_reason = True, "length"

                yield GenerationOutput(
                    text=full, tokens=list(state.gen_ids),
                    prompt_tokens=prompt_tokens, completion_tokens=completion,
                    finish_reason=finish_reason, new_text=delta, finished=finished,
                    generation_tps=tps,
                )
                if finished:
                    return
        finally:
            with self._active_lock:
                self._active -= 1

    def _to_ids(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, (list, tuple)):
            return list(prompt)
        return self._tokenizer.encode(prompt, add_special_tokens=False)

    # ----- BaseEngine API -----
    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        last: GenerationOutput | None = None
        async for out in self.stream_generate(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, stop=stop, **kwargs,
        ):
            last = out
        if last is None:
            return GenerationOutput(text="", prompt_tokens=len(self._to_ids(prompt)))
        return GenerationOutput(
            text=last.text, tokens=last.tokens, prompt_tokens=last.prompt_tokens,
            completion_tokens=last.completion_tokens, finish_reason=last.finish_reason,
            generation_tps=last.generation_tps,
        )

    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        input_ids = self._to_ids(prompt)
        async for out in self._stream_ids(
            input_ids, max_tokens, temperature, top_p, top_k, min_p,
            repetition_penalty, presence_penalty, stop,
        ):
            yield out

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        prompt = self._render_prompt(messages, kwargs.get("chat_template_kwargs"))
        return await self.generate(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, stop=kwargs.get("stop"),
        )

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        prompt = self._render_prompt(messages, kwargs.get("chat_template_kwargs"))
        async for out in self.stream_generate(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, min_p=min_p, repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty, stop=kwargs.get("stop"),
        ):
            yield out

    # ----- telemetry -----
    def has_active_requests(self) -> bool:
        with self._active_lock:
            return self._active > 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "engine_type": "hrm_text",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "model_type": self.model_type,
            "n_slots": self._model.n_slots if self._model else None,
            "context_length": self.context_length,
            "active_requests": self._active,
            "uptime_seconds": time.monotonic() - self._t_start,
        }

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        # No shared/persistent KV cache: each request holds its own transient
        # 128-slot cache for its lifetime only.
        return None
