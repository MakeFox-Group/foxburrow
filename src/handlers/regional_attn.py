"""Regional attention processor for attention-mode regional prompting.

Hooks cross-attention (attn2) layers to composite per-region text conditioning
with spatial masks. Self-attention passes through unchanged.

NOTE: RegionalAttnState is not thread-safe. It is protected by the sdxl_unet
session concurrency limit of 1 — only one UNet denoise stage runs per GPU at
a time. Do not raise that limit without adding synchronization.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class RegionalAttnState:
    """Shared state referenced by all RegionalAttnProcessor instances."""

    def __init__(self):
        self.active: bool = False
        self.region_embeds: torch.Tensor | None = None   # [N, 77, embed_dim]
        self.region_masks: torch.Tensor | None = None     # [N, 1, h, w]
        # Cache keyed by (tile_bounds, seq_len) so interpolated masks persist
        # across denoising steps for the same tile geometry.
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        self._tile_bounds: tuple[int, int, int, int] | None = None  # (y0, x0, y1, x1)

    def set_regions(self, region_embeds: torch.Tensor, region_masks: torch.Tensor) -> None:
        """Store region data and clear mask cache."""
        self.region_embeds = region_embeds
        self.region_masks = region_masks
        self._tile_bounds = None
        self._mask_cache.clear()

    def set_tile_masks(
        self,
        tile_masks: torch.Tensor,
        tile_bounds: tuple[int, int, int, int],
    ) -> None:
        """Update masks for current MultiDiffusion tile.

        tile_bounds is (y0, x0, y1, x1) in latent coordinates, used as part
        of the mask cache key so interpolated masks persist across steps for
        the same tile position.
        """
        self.region_masks = tile_masks
        self._tile_bounds = tile_bounds

    def get_mask_for_seq_len(self, seq_len: int) -> torch.Tensor:
        """Resize region masks to match a given attention layer's spatial resolution.

        Derives layer_h, layer_w from seq_len using the known downscale factor
        from the mask dimensions, then interpolates and caches.
        """
        cache_key = (self._tile_bounds, seq_len)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        masks = self.region_masks
        full_h, full_w = masks.shape[2], masks.shape[3]

        if seq_len == full_h * full_w:
            self._mask_cache[cache_key] = masks
            return masks

        # SDXL UNet attention layers use power-of-2 downscale factors: 1, 2, 4
        # Try each to find exact match for seq_len
        layer_h, layer_w = full_h, full_w
        found = False
        for k in (1, 2, 4, 8):
            lh = full_h // k
            lw = full_w // k
            if lh >= 1 and lw >= 1 and lh * lw == seq_len:
                layer_h, layer_w = lh, lw
                found = True
                break

        if not found:
            raise RuntimeError(
                f"Regional attention: cannot determine layer dimensions for seq_len={seq_len} "
                f"from mask shape ({full_h}, {full_w}). "
                f"Tried downscale factors 1, 2, 4, 8."
            )

        result = F.interpolate(masks, size=(layer_h, layer_w), mode='nearest')
        # Renormalize after interpolation
        mask_sum = result.sum(dim=0, keepdim=True).clamp(min=1e-8)
        result = result / mask_sum

        self._mask_cache[cache_key] = result
        return result


class RegionalAttnProcessor:
    """Custom attention processor for regional prompting.

    Handles both self-attention (passthrough) and cross-attention (regional
    compositing) depending on whether encoder_hidden_states is provided and
    the shared state is active.
    """

    def __init__(self, state: RegionalAttnState):
        self.state = state

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None or not self.state.active:
            # Self-attention or inactive: standard SDPA
            return _standard_sdpa(attn, hidden_states, encoder_hidden_states,
                                  attention_mask, temb)

        # Cross-attention with regional prompting
        return self._regional_cross_attention(attn, hidden_states, attention_mask, temb)

    def _regional_cross_attention(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask,
        temb,
    ) -> torch.Tensor:
        N = self.state.region_embeds.shape[0]

        # Norm + residual
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        # Capture dims after all normalization/reshaping
        seq_len = hidden_states.shape[1]
        inner_dim = hidden_states.shape[2]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(self.state.region_embeds)
        else:
            encoder_hidden_states = self.state.region_embeds

        # Q from UNet features, expanded for N regions (contiguous for reshape)
        query = attn.to_q(hidden_states)                    # [1, seq_len, inner_dim]
        query = query.expand(N, -1, -1).contiguous()        # [N, seq_len, inner_dim]

        # K, V from regional embeddings
        key = attn.to_k(encoder_hidden_states)   # [N, 77, inner_dim]
        value = attn.to_v(encoder_hidden_states) # [N, 77, inner_dim]

        # Verify K/V projection dim matches Q — catches mismatches if this code
        # is reused for non-SDXL architectures (e.g. SD 1.5, Flux)
        kv_dim = key.shape[-1]
        assert kv_dim == inner_dim, (
            f"Regional attention: K/V inner_dim ({kv_dim}) != Q inner_dim ({inner_dim}). "
            f"The text encoder embedding dimension may not match this UNet's cross-attention."
        )

        # Multi-head reshape
        head_dim = inner_dim // attn.heads
        query = query.view(N, seq_len, attn.heads, head_dim).transpose(1, 2)
        key = key.view(N, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(N, -1, attn.heads, head_dim).transpose(1, 2)
        # Now [N, heads, seq_len, head_dim] and [N, heads, 77, head_dim]

        # Fused batched attention (flash attention compatible)
        attn_out = F.scaled_dot_product_attention(query, key, value)
        # [N, heads, seq_len, head_dim]

        attn_out = attn_out.transpose(1, 2).reshape(N, seq_len, inner_dim)
        # [N, seq_len, inner_dim]

        # Spatial masking and compositing
        mask = self.state.get_mask_for_seq_len(seq_len)  # [N, 1, layer_h, layer_w]
        layer_h, layer_w = mask.shape[2], mask.shape[3]

        attn_out = attn_out.reshape(N, layer_h, layer_w, inner_dim)
        mask_4d = mask.squeeze(1).unsqueeze(-1)  # [N, layer_h, layer_w, 1]

        composited = (attn_out * mask_4d).sum(dim=0, keepdim=True)  # [1, h, w, dim]
        composited = composited.reshape(1, seq_len, inner_dim)

        # Output projection
        composited = attn.to_out[0](composited)
        composited = attn.to_out[1](composited)

        if input_ndim == 4:
            composited = composited.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            composited = composited + residual

        composited = composited / attn.rescale_output_factor

        return composited


def _standard_sdpa(
    attn,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    attention_mask=None,
    temb=None,
) -> torch.Tensor:
    """Standard AttnProcessor2_0 behavior: Q/K/V → multi-head SDPA → output projection."""
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, _, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask,
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

    # Linear proj + dropout
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def install_regional_processors(unet, state: RegionalAttnState) -> dict:
    """Set RegionalAttnProcessor on cross-attention (attn2) layers only. Returns the original processors."""
    original_procs = dict(unet.attn_processors)
    procs = {}
    for name, proc in original_procs.items():
        if 'attn2' in name:
            procs[name] = RegionalAttnProcessor(state)
        else:
            procs[name] = proc
    unet.set_attn_processor(procs)
    return original_procs
