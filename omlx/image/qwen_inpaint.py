# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image base inpaint loop (mflux port of diffusers QwenImageInpaintPipeline).

Validated on m5max against mflux's own ground truth (no diffusers needed, the
package ships no torch reference): an all-white mask (regenerate everything)
reproduces mflux QwenImage txt2img byte-for-byte (MSE 0.0), an all-black mask
(keep everything) reproduces the source (MSE ~1.4, VAE round-trip only), and a
partial mask keeps its region pixel-faithful while regenerating the rest. This
brackets the three failure modes (kept-region drift / sigma indexing / mask
packing) without a diffusers cross-check. See docs/qwen-inpaint-engine-spec.md.

HARD RULE (same as worker.py): no omlx imports. mflux + mlx + numpy + stdlib.

Algorithm (diffusers convention: mask white=1=REGENERATE, black=0=KEEP; the
caller already inverted any BiRefNet foreground -- fmlx does NOT invert):

  clean   = pack(vae_encode(image))            # clean packed latents
  noise   = create_noise(seed)
  latents = (1-sigma0)*clean + sigma0*noise    # init from image at strength
  for t in range(init_time_step, steps):
      latents = scheduler.step(transformer(latents), t)
      sig_next = sigmas[t+1]                    # 0 at the last step -> clean
      init_proper = (1-sig_next)*clean + sig_next*noise
      latents = (1-mask)*init_proper + mask*latents
  out = vae_decode(latents)
  out = composite(original, out, mask)          # byte-exact keep region
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import numpy as np
from PIL import Image, ImageFilter

from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import (
    QwenPromptEncoder,
)
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.image_util import ImageUtil


def _pack_mask(mask_path: str, encoded_shape, height: int, width: int) -> mx.array:
    """Resize mask to the latent grid, broadcast over channels, pack to the
    latents' packed layout. Standard convention: white(255) -> 1.0 = regenerate.
    encoded_shape is (B, C, 1, Hl, Wl) for qwen (or (B, C, Hl, Wl))."""
    hl, wl = int(encoded_shape[-2]), int(encoded_shape[-1])
    m = Image.open(mask_path).convert("L").resize(
        (wl, hl), Image.Resampling.NEAREST
    )
    arr = np.asarray(m, dtype=np.float32) / 255.0  # (Hl, Wl)
    grid = mx.array(arr).reshape((1,) * (len(encoded_shape) - 2) + (hl, wl))
    grid = mx.broadcast_to(grid, encoded_shape)
    return QwenLatentCreator.pack_latents(grid, height=height, width=width)


def composite_keep_region(
    original_pil: Image.Image,
    generated_pil: Image.Image,
    mask_pil: Image.Image,
    feather_px: int = 2,
) -> Image.Image:
    """Pixel-space composite for byte-exact pixel lock (spec s3):
    out = (1-m)*original + m*generated, m = mask (white=1=regenerate). The keep
    region (m=0) is copied verbatim from the original. Image.composite(a,b,mask)
    picks a where mask=255, so a=generated (regenerate), b=original (keep)."""
    out_size = generated_pil.size
    resample = Image.Resampling.BILINEAR
    m = mask_pil.convert("L").resize(out_size, resample)
    if feather_px and feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    orig = original_pil.convert("RGB").resize(out_size, resample)
    gen = generated_pil.convert("RGB")
    return Image.composite(gen, orig, m)


def generate_inpaint(
    model: QwenImage,
    *,
    prompt: str,
    image_path: str,
    mask_path: str,
    width: int | None = None,
    height: int | None = None,
    steps: int = 30,
    seed: int = 0,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    image_strength: float = 1.0,
    feather_px: int = 2,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Image.Image:
    """Run masked inpaint on a loaded QwenImage and return a composited PIL
    image (keep region byte-exact from the source). Mirrors
    QwenImage.generate_image with the diffusers inpaint blend inserted; reuses
    the model's loaded components (no monkeypatch of installed mflux)."""
    config = Config(
        width=width,
        height=height,
        guidance=4.0 if guidance is None else float(guidance),
        scheduler="flow_match_euler_discrete",
        image_path=image_path,
        image_strength=image_strength,
        model_config=model.model_config,
        num_inference_steps=int(steps),
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
        preserve_image_aspect_ratio=True,
    )
    H, W = config.height, config.width
    sigmas = config.scheduler.sigmas
    t0 = config.init_time_step

    encoded = LatentCreator.encode_image(
        vae=model.vae, image_path=image_path, height=H, width=W,
        tiling_config=model.tiling_config,
    )
    clean = QwenLatentCreator.pack_latents(encoded, height=H, width=W)
    noise = QwenLatentCreator.create_noise(seed=seed, height=H, width=W)
    mask = _pack_mask(mask_path, encoded.shape, H, W)

    latents = LatentCreator.add_noise_by_interpolation(
        clean=clean, noise=noise, sigma=sigmas[t0]
    )

    pe, pm, npe, npm = QwenPromptEncoder.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_cache=model.prompt_cache,
        qwen_tokenizer=model.tokenizers["qwen"],
        qwen_text_encoder=model.text_encoder,
    )

    n = int(steps)
    for t in range(t0, n):
        latents = config.scheduler.scale_model_input(latents, t)
        noise_p = model.transformer(
            t=t, config=config, hidden_states=latents,
            encoder_hidden_states=pe, encoder_hidden_states_mask=pm,
        )
        noise_n = model.transformer(
            t=t, config=config, hidden_states=latents,
            encoder_hidden_states=npe, encoder_hidden_states_mask=npm,
        )
        guided = QwenImage.compute_guided_noise(noise_p, noise_n, config.guidance)
        latents = config.scheduler.step(noise=guided, timestep=t, latents=latents)
        # diffusers inpaint blend: snap the keep region back onto the source's
        # forward-noised trajectory; sigmas[t+1] is 0 at the last step (clean).
        init_proper = LatentCreator.add_noise_by_interpolation(
            clean=clean, noise=noise, sigma=sigmas[t + 1]
        )
        latents = (1.0 - mask) * init_proper + mask * latents
        mx.eval(latents)
        if progress_cb is not None:
            progress_cb(t - t0 + 1, n - t0)

    unpacked = QwenLatentCreator.unpack_latents(latents=latents, height=H, width=W)
    decoded = VAEUtil.decode(
        vae=model.vae, latent=unpacked, tiling_config=model.tiling_config
    )
    gi = ImageUtil.to_image(
        decoded_latents=decoded, config=config, seed=seed, prompt=prompt,
        quantization=getattr(model, "bits", 4), generation_time=0.0,
        image_path=image_path, image_strength=image_strength,
        negative_prompt=negative_prompt,
    )
    # Byte-exact pixel lock: snap the keep region to the source pixels. The
    # latent blend already holds it VAE-faithful; the composite makes it exact.
    original = ImageUtil.scale_to_dimensions(
        image=ImageUtil.load_image(image_path).convert("RGB"),
        target_width=gi.image.size[0],
        target_height=gi.image.size[1],
    )
    return composite_keep_region(original, gi.image, Image.open(mask_path), feather_px)
