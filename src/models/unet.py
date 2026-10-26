"""Denoising U-Nets built on MONAI's `DiffusionModelUNetMaisi`.

* CT-Gen uses the stock `DiffusionModelUNetMaisi` (Report2CT configuration, 16-channel Wan latent).
* MR-Gen uses `DiffusionModelUNetMaisiTextPooledAdaGN`:
    - local text path : token sequence -> cross-attention (with a key-padding mask)
    - global text path: pooled text vector -> zero-init Linear, added to the timestep embedding
    - every ResBlock consumes the conditioning embedding through AdaGN (scale/shift after GroupNorm)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.blocks.crossattention import CrossAttentionBlock
from monai.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
from monai.utils.type_conversion import convert_to_tensor
from torch import nn

# Shared U-Net hyper-parameters (Report2CT configuration, adapted to the 16-channel Wan latent).
UNET_KWARGS = dict(
    spatial_dims=3,
    in_channels=16,
    out_channels=16,
    num_channels=[64, 128, 256, 512],
    num_res_blocks=2,
    attention_levels=[False, False, True, True],
    num_head_channels=[0, 0, 32, 32],
    num_class_embeds=128,
    include_fc=True,
    include_spacing_input=True,
    include_top_region_index_input=False,
    include_bottom_region_index_input=False,
    resblock_updown=True,
    with_conditioning=True,
)


# --------------------------------------------------------------------------------------------
# Cross-attention with a key-padding mask
# --------------------------------------------------------------------------------------------
class MaskedCrossAttentionBlock(CrossAttentionBlock):
    """`CrossAttentionBlock` that ignores padded context tokens.

    MONAI's cross-attention has no mask argument and the layers above it do not forward extra
    kwargs, so the mask is stored on the block (`context_mask`, ``(B, T)`` bool, True = real token)
    by the owning U-Net right before its forward. Existing blocks are converted in place
    (`enable_context_masking`), so parameters and state-dict keys are unchanged.
    """

    context_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        mask = self.context_mask
        if mask is None or context is None:
            return super().forward(x, context)

        q = self.input_rearrange(self.to_q(x))  # (B, heads, N, d)
        k = self.input_rearrange(self.to_k(context))  # (B, heads, T, d)
        v = self.input_rearrange(self.to_v(context))  # (B, heads, T, d)
        if self.attention_dtype is not None:
            q = q.to(self.attention_dtype)
            k = k.to(self.attention_dtype)
        attn_mask = mask.to(device=q.device)[:, None, None, :]  # (B, 1, 1, T)

        if self.use_flash_attention:
            out = F.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=attn_mask, scale=self.scale,
                dropout_p=self.dropout_rate, is_causal=False,
            )
        else:
            att = torch.einsum("blxd,blyd->blxy", q, k) * self.scale
            att = att.masked_fill(~attn_mask, torch.finfo(att.dtype).min)
            att = self.drop_weights(att.softmax(dim=-1))
            out = torch.einsum("bhxy,bhyd->bhxd", att, v)

        out = self.out_rearrange(out)
        out = self.out_proj(out)
        return self.drop_output(out)


def enable_context_masking(module: nn.Module) -> list[MaskedCrossAttentionBlock]:
    """Convert every `CrossAttentionBlock` under `module` into a `MaskedCrossAttentionBlock`."""
    blocks = []
    for m in module.modules():
        if isinstance(m, CrossAttentionBlock) and not isinstance(m, MaskedCrossAttentionBlock):
            m.__class__ = MaskedCrossAttentionBlock
            m.context_mask = None
            blocks.append(m)
    return blocks


# --------------------------------------------------------------------------------------------
# AdaGN ResBlock
# --------------------------------------------------------------------------------------------
class AdaGNResnetBlock(DiffusionUNetResnetBlock):
    """ResBlock that injects `emb` as scale/shift after `norm2` (AdaGN, Dhariwal & Nichol 2021).

    The stock MONAI block adds `emb` as a bias *before* `norm2`, where GroupNorm largely removes it.
    Same structure as diffusers' ``ResnetBlock2D(time_embedding_norm="scale_shift")``.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.nonlinearity(self.norm1(x))
        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)
        h = self.conv1(h)

        temb = self.time_emb_proj(self.nonlinearity(emb))  # (B, 2 * C_out)
        temb = temb[(...,) + (None,) * self.spatial_dims]
        scale, shift = temb.chunk(2, dim=1)

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.nonlinearity(h))  # conv2 is zero-initialised by MONAI
        return self.skip_connection(x) + h


def enable_adagn(module: nn.Module) -> list[AdaGNResnetBlock]:
    """Convert every `DiffusionUNetResnetBlock` under `module` into an `AdaGNResnetBlock`."""
    blocks = []
    for m in module.modules():
        if isinstance(m, DiffusionUNetResnetBlock) and not isinstance(m, AdaGNResnetBlock):
            m.time_emb_proj = nn.Linear(m.emb_channels, 2 * m.out_channels)
            m.__class__ = AdaGNResnetBlock
            blocks.append(m)
    return blocks


# --------------------------------------------------------------------------------------------
# U-Net with a pooled (global) text path
# --------------------------------------------------------------------------------------------
class DiffusionModelUNetMaisiTextPooled(DiffusionModelUNetMaisi):
    """`DiffusionModelUNetMaisi` + a zero-initialised pooled-text embedding path.

    The pooled findings/impression vectors are projected to `time_embed_dim` and added to the
    timestep+class segment of `emb` (the spacing segment is left untouched).

    Args:
        text_pooled_dim: input width of `text_pooled_proj`.
        text_pooled_concat: concatenate findings and impression before the projection (keeps the
            two sections distinguishable) instead of projecting each and summing.
    """

    def __init__(self, *args, text_pooled_dim: int = 2560, text_pooled_concat: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_embed_dim = self.block_out_channels[0] * 4
        self.text_pooled_concat = text_pooled_concat
        self.text_pooled_proj = nn.Linear(text_pooled_dim, self.time_embed_dim)
        nn.init.zeros_(self.text_pooled_proj.weight)
        nn.init.zeros_(self.text_pooled_proj.bias)
        self._xattn_blocks = enable_context_masking(self)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
        text_pooled_f: torch.Tensor | None = None,
        text_pooled_i: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Same as the base forward, plus pooled text in `emb` and a cross-attention mask.

        Args:
            text_pooled_f / text_pooled_i: pooled findings / impression vectors ``(B, D)``.
            context_mask: ``(B, T)`` bool key-padding mask over `context` (True = real token).
        """
        for block in self._xattn_blocks:
            block.context_mask = context_mask

        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor)

        if self.text_pooled_concat and text_pooled_f is not None and text_pooled_i is not None:
            terms = [self.text_pooled_proj(torch.cat([text_pooled_f.flatten(1), text_pooled_i.flatten(1)], dim=1))]
        else:
            terms = [self.text_pooled_proj(t.flatten(1)) for t in (text_pooled_f, text_pooled_i) if t is not None]
        if terms:
            text_emb = sum(terms)  # (B, time_embed_dim)
            emb = emb + F.pad(text_emb, (0, emb.shape[1] - text_emb.shape[1]))

        h = self.conv_in(x)
        h, res = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        h = self.middle_block(h, emb, context)
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual
        h = self._apply_up_blocks(h, emb, context, res)
        h = self.out(h)
        return convert_to_tensor(h)


class DiffusionModelUNetMaisiTextPooledAdaGN(DiffusionModelUNetMaisiTextPooled):
    """`DiffusionModelUNetMaisiTextPooled` whose ResBlocks use AdaGN (train from scratch only)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._adagn_blocks = enable_adagn(self)


def build_unet(arch: str, cross_attention_dim: int, text_pooled_dim: int | None = None,
               use_flash_attention: bool = False) -> nn.Module:
    """Build the U-Net used by a released configuration (for inference).

    Args:
        arch: ``"ctgen"`` (stock MAISI U-Net, pooled 3-encoder text) or
            ``"mrgen"`` (AdaGN U-Net, token cross-attention + pooled global path).
    """
    kwargs = dict(UNET_KWARGS, cross_attention_dim=cross_attention_dim, use_flash_attention=use_flash_attention)
    if arch == "ctgen":
        return DiffusionModelUNetMaisi(**kwargs)
    if arch == "mrgen":
        return DiffusionModelUNetMaisiTextPooledAdaGN(
            **kwargs, text_pooled_dim=text_pooled_dim, text_pooled_concat=True
        )
    raise ValueError(f"unknown arch {arch!r}")
