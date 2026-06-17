# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image base inpaint loop (mflux port of diffusers QwenImageInpaintPipeline).

=============================================================================
STATUS: UNVALIDATED SCAFFOLD (Phase 1, feat/image-inpaint).
The algorithm is grounded in mflux 0.18.14 QwenImage.generate_image +
diffusers QwenImageInpaintPipeline, but has NOT run on real weights. Several
mflux-internal calls (VAE encode, scheduler sigma/scale_noise access, Config
and transformer call surface) are marked TODO(m5max) and MUST be confirmed
against the live package before this is trusted. Exit criterion: a standalone
m5max run comparing this against the real diffusers pipeline on the same
image+mask+seed (docs/qwen-inpaint-engine-spec.md section 8). Do NOT register
an image_pipeline=inpaint model until that passes -- this module is dormant
(only reached when spec["pipeline"]=="inpaint") until then.
=============================================================================

HARD RULE (same as worker.py): no omlx imports. mflux + mlx + stdlib only.

Algorithm (diffusers convention: mask white=1=REGENERATE, black=0=KEEP; the
caller already inverted BiRefNet foreground -- fmlx does NOT invert, see spec
section 2):

  image_latents = vae_encode(image)            # clean latents of original
  noise         = create_noise(seed)
  latents       = scale_noise(image_latents, sigma_start, noise)
  for i, t in enumerate(timesteps):            # truncated by image_strength
      latents = denoise_one_step(latents, t)   # transformer + scheduler.step
      sig_next = sigmas[i+1] if i < len-1 else 0.0
      init_proper = (1 - sig_next) * image_latents + sig_next * noise
      latents = (1 - mask) * init_proper + mask * latents
  out = vae_decode(latents)
  out = pixel_composite(original, out, mask)   # byte-exact keep region

flow-match identity: scale_noise(x0, noise, sigma) = (1-sigma)*x0 + sigma*noise.
"""

from __future__ import annotations


def _flow_scale_noise(image_latents, noise, sigma):
    """Flow-match forward noising: (1-sigma)*x0 + sigma*noise. Pure, verified."""
    return (1.0 - sigma) * image_latents + sigma * noise


def _blend(init_proper, latents, mask):
    """Per-step mask blend. mask=1 -> regenerate (keep `latents`), mask=0 ->
    keep (snap to `init_proper`). Pure, verified against diffusers."""
    return (1.0 - mask) * init_proper + mask * latents


def _load_mask_latent(mask_path, height, width, num_channels_latents=16):
    """Load mask image, downsample to latent grid, pack to QwenLatentCreator
    sequence layout, broadcast over channels. mask in [0,1], white->1.

    TODO(m5max): confirm the latent grid size (height//8 // patch vs the exact
    QwenLatentCreator packing) and that pack_latents accepts a single-channel
    array broadcast to num_channels_latents. mflux ImageUtil.to_array(is_mask=
    True) gives the normalized mask array; QwenLatentCreator.pack_latents packs
    [B,C,H',W'] -> [B,seq,C]. The packed mask must align index-for-index with
    the packed latents the loop mutates.
    """
    import mlx.core as mx  # noqa: F401
    from mflux.models.qwen.latent_creator.qwen_latent_creator import (  # noqa: F401
        QwenLatentCreator,
    )
    from mflux.utils.image_util import ImageUtil  # noqa: F401

    raise NotImplementedError(
        "TODO(m5max): mask->latent packing. See docstring + spec section 1."
    )


def _encode_image_latents(model, image_path, height, width):
    """VAE-encode the source image to clean packed latents (image_latents).

    TODO(m5max): confirm the exact mflux call. QwenImage img2img uses an
    Img2Img helper that wraps VAE encode + QwenLatentCreator pack; we need the
    CLEAN packed image_latents held separately (the helper returns only the
    pre-noised init latents). Likely: VAEUtil.encode(model.vae, image_array,
    tiling_config) -> QwenLatentCreator.pack_latents(...). Must match the
    layout the transformer consumes.
    """
    raise NotImplementedError(
        "TODO(m5max): VAE encode -> packed image_latents. See spec section 1."
    )


def generate_inpaint(
    model,
    *,
    prompt: str,
    image_path: str,
    mask_path: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    image_strength: float = 1.0,
    feather_px: int = 2,
):
    """Run masked inpaint on a loaded mflux QwenImage model, return a PIL image
    with the keep-region composited byte-exact from the original.

    `model` is the QwenImage instance from worker._load_model (has .vae,
    .transformer, .text_encoder, .tokenizers, .prompt_cache, .callbacks,
    .tiling_config, .compute_guided_noise). Reuses those rather than
    monkeypatching the installed mflux package (maintainable seam, spec 6).

    TODO(m5max): the denoise loop below mirrors QwenImage.generate_image
    (verified source) with the blend inserted. Confirm: Config construction
    for img2img (init_time_step from image_strength), config.scheduler.sigmas
    indexing, transformer(t, config, hidden_states, encoder_hidden_states,
    encoder_hidden_states_mask) call surface, and QwenPromptEncoder import.
    """
    # --- verified-shape orchestration; internals are TODO(m5max) ---
    # 1. image_latents (clean) + noise + truncated timesteps from strength
    # 2. latents = scale_noise(image_latents, sigma_start, noise)
    # 3. mask_latent (packed, white=1=regenerate)
    # 4. loop: denoise one step, then
    #       latents = _blend(_flow_scale_noise(image_latents, noise, sig_next),
    #                         latents, mask_latent)
    # 5. decode -> PIL; pixel-composite keep region from original (feathered)
    raise NotImplementedError(
        "Phase 1 inpaint loop scaffold -- finalize mflux glue + validate on "
        "m5max before wiring a registered inpaint model. See "
        "docs/qwen-inpaint-engine-spec.md sections 1, 3, 8."
    )


def composite_keep_region(original_pil, generated_pil, mask_pil, feather_px=2):
    """Pixel-space composite for byte-exact pixel lock (spec section 3):
    out = (1-m)*original + m*generated, m = mask (white=1=regenerate). The keep
    region (m=0) is copied verbatim from the original. Pure PIL, verified.

    TODO(m5max): confirm mask is resized to the OUTPUT image size (edit/img2img
    canvas policy can resize), and feather direction (blur the mask edge so the
    composite seam is not a hard cutout).
    """
    from PIL import Image, ImageFilter

    out_size = generated_pil.size
    resample = Image.Resampling.BILINEAR
    m = mask_pil.convert("L").resize(out_size, resample)
    if feather_px and feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    orig = original_pil.convert("RGB").resize(out_size, resample)
    gen = generated_pil.convert("RGB")
    # Image.composite(image1, image2, mask): mask=255 -> image1. We want
    # m=255 (white=regenerate) -> generated, so image1=generated, image2=orig.
    return Image.composite(gen, orig, m)
