"""Denoising U-Net: MONAI MAISI U-Net + pooled-text AdaGN conditioning + padding-masked cross-attention.

``ReportUNet3D`` keeps every MONAI ``DiffusionModelUNetMaisi`` layer (so ``state_dict`` keys are the stock names) and
changes three things in place:

* a zero-initialised ``text_pooled_proj`` adds the pooled report vector into the leading ``time_embed_dim`` columns
  of ``emb`` (spacing columns untouched);
* every ``DiffusionUNetResnetBlock`` becomes an ``AdaGNResBlock`` (scale + shift after ``norm2``, as in the
  ``scale_shift`` ResNet block of diffusers / ADM);
* every ``CrossAttentionBlock`` becomes a ``MaskedCrossAttention`` that ignores right-padded context tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.blocks.crossattention import CrossAttentionBlock
from monai.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
from monai.utils.type_conversion import convert_to_tensor
from torch import Tensor, nn


class AdaGNResBlock(DiffusionUNetResnetBlock):
    """``DiffusionUNetResnetBlock`` with ``emb`` injected as AdaGN (scale + shift after ``norm2``).

    Converted in place by ``enable_adagn`` (``__class__`` swap); declares no ``__init__`` of its own.
    """

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Run the block.

        Args:
            x: ``(B, C_in, H, W, D)``.
            emb: ``(B, emb_channels)``.

        Returns:
            ``(B, C_out, H', W', D')``.
        """
        if emb is None:
            raise ValueError("AdaGNResBlock requires `emb` (scale_shift has no bias-only fallback).")
        # DUPLICATION INTENTIONAL: monai/networks/nets/diffusion_model_unet.py DiffusionUNetResnetBlock.forward,
        # verbatim through conv1; only the temb/norm2 part differs (diffusers ResnetBlock2D "scale_shift").
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        temb = self.time_emb_proj(self.nonlinearity(emb))  # (B, 2*C_out)
        if self.spatial_dims == 2:
            temb = temb[:, :, None, None]
        else:
            temb = temb[:, :, None, None, None]  # (B, 2*C_out, 1, 1, 1)
        scale, shift = temb.chunk(2, dim=1)  # (B, C_out, 1, 1, 1) each

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift  # AdaGN after norm2

        h = self.nonlinearity(h)
        h = self.conv2(h)
        output: Tensor = self.skip_connection(x) + h
        return output


def enable_adagn(module: nn.Module) -> list[AdaGNResBlock]:
    """Convert every ``DiffusionUNetResnetBlock`` under ``module`` into an ``AdaGNResBlock`` in place.

    Each block's ``time_emb_proj`` is replaced by a fresh ``nn.Linear(emb_channels, 2 * out_channels)`` (default
    PyTorch init, not zero: ``conv2`` is already zero-initialised by MONAI).

    Args:
        module: module containing MONAI diffusion ResBlocks.

    Returns:
        The converted blocks, in ``modules()`` order.
    """
    converted: list[AdaGNResBlock] = []
    for submodule in module.modules():
        if isinstance(submodule, DiffusionUNetResnetBlock) and not isinstance(submodule, AdaGNResBlock):
            submodule.time_emb_proj = nn.Linear(submodule.emb_channels, 2 * submodule.out_channels)
            submodule.__class__ = AdaGNResBlock
            converted.append(submodule)
    return converted


class MaskedCrossAttention(CrossAttentionBlock):
    """``CrossAttentionBlock`` that honours a key-padding mask stored on the instance.

    Adds no parameters; converted in place by ``enable_context_masking``. The mask is set by the owning U-Net
    before each forward because the MONAI layers in between only forward ``context``.
    """

    #: ``(B, T_context)`` bool, ``True`` = real token. ``None`` = stock behaviour.
    context_mask: Tensor | None = None

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        """Cross-attend ``x`` to ``context``, ignoring padded context positions.

        Args:
            x: query sequence ``(B, N, C)``.
            context: key/value sequence ``(B, T, context_dim)``.

        Returns:
            ``(B, N, C)``.
        """
        mask = self.context_mask
        if mask is None or context is None:
            return super().forward(x, context)

        # DUPLICATION INTENTIONAL: monai/networks/blocks/crossattention.py CrossAttentionBlock.forward,
        # plus the key-padding mask (rel_positional_embedding is unused in this U-Net).
        if self.causal:
            raise NotImplementedError("MaskedCrossAttention does not support causal=True with a context mask.")

        q = self.input_rearrange(self.to_q(x))  # (B, heads, N, head_dim)
        k = self.input_rearrange(self.to_k(context))  # (B, heads, T, head_dim)
        v = self.input_rearrange(self.to_v(context))  # (B, heads, T, head_dim)
        if self.attention_dtype is not None:
            q = q.to(self.attention_dtype)
            k = k.to(self.attention_dtype)

        attn_mask = mask.to(device=q.device)[:, None, None, :]  # (B, 1, 1, T) bool, True = attend

        if self.use_flash_attention:
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                scale=self.scale,
                dropout_p=self.dropout_rate,
                is_causal=False,
            )
        else:
            att_mat = torch.einsum("blxd,blyd->blxy", q, k) * self.scale  # (B, heads, N, T)
            # finfo.min rather than -inf: an all-padding row degrades to uniform instead of NaN.
            att_mat = att_mat.masked_fill(~attn_mask, torch.finfo(att_mat.dtype).min)
            att_mat = att_mat.softmax(dim=-1)
            att_mat = self.drop_weights(att_mat)
            out = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)  # (B, heads, N, head_dim)

        out = self.out_rearrange(out)  # (B, N, heads*head_dim)
        out = self.out_proj(out)
        out = self.drop_output(out)
        return out


def enable_context_masking(module: nn.Module) -> list[MaskedCrossAttention]:
    """Convert every ``CrossAttentionBlock`` under ``module`` into a ``MaskedCrossAttention`` in place.

    Args:
        module: module containing cross-attention blocks.

    Returns:
        The converted blocks, in ``modules()`` order.
    """
    converted: list[MaskedCrossAttention] = []
    for submodule in module.modules():
        if isinstance(submodule, CrossAttentionBlock) and not isinstance(submodule, MaskedCrossAttention):
            submodule.__class__ = MaskedCrossAttention
            submodule.context_mask = None
            converted.append(submodule)
    return converted


class ReportUNet3D(DiffusionModelUNetMaisi):
    """MAISI U-Net + zero-init pooled-text ``emb`` path + masked cross-attention + AdaGN ResBlocks.

    Args:
        *args, **kwargs: forwarded to ``DiffusionModelUNetMaisi.__init__``.
        text_pooled_dim: width of the concatenated pooled text vector (4096 = 2 sections x 2048).
    """

    def __init__(self, *args: object, text_pooled_dim: int, **kwargs: object) -> None:
        # Order is load-bearing: it fixes the parameter-init RNG order and the state_dict keys.
        super().__init__(*args, **kwargs)
        # DUPLICATION INTENTIONAL: time_embed_dim = num_channels[0] * 4 as in DiffusionModelUNetMaisi.__init__.
        self.time_embed_dim = self.block_out_channels[0] * 4
        self.text_pooled_proj = nn.Linear(text_pooled_dim, self.time_embed_dim)
        nn.init.zeros_(self.text_pooled_proj.weight)
        nn.init.zeros_(self.text_pooled_proj.bias)
        self._xattn_blocks = enable_context_masking(self)
        self._adagn_blocks = enable_adagn(self)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor | None = None,
        class_labels: Tensor | None = None,
        down_block_additional_residuals: tuple[Tensor] | None = None,
        mid_block_additional_residual: Tensor | None = None,
        top_region_index_tensor: Tensor | None = None,
        bottom_region_index_tensor: Tensor | None = None,
        spacing_tensor: Tensor | None = None,
        text_pooled_f: Tensor | None = None,
        text_pooled_i: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict the velocity.

        Args:
            x: noisy latent ``(B, 16, 64, 64, 64)``.
            timesteps: ``(B,)`` float in ``[0, 1000]``.
            context: text tokens ``(B, T, 1024)``.
            class_labels: ``(B,)`` long.
            down_block_additional_residuals, mid_block_additional_residual, top_region_index_tensor,
                bottom_region_index_tensor: as in ``DiffusionModelUNetMaisi.forward``.
            spacing_tensor: ``(B, 3)`` voxel spacing in mm x 100.
            text_pooled_f: pooled findings vector ``(B, 2048)``.
            text_pooled_i: pooled impression vector ``(B, 2048)``.
            context_mask: ``(B, T)`` bool, ``True`` = real token; ``None`` attends to every position.

        Returns:
            ``(B, 16, 64, 64, 64)``.
        """
        # Set unconditionally (also None) and never cleared, so activation-checkpoint recomputation sees it too.
        for block in self._xattn_blocks:
            block.context_mask = context_mask

        # DUPLICATION INTENTIONAL: monai/apps/generation/maisi/networks/diffusion_model_unet_maisi.py
        # DiffusionModelUNetMaisi.forward, extended by the pooled-text block.
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)  # (B, 256)
        emb = self._get_input_embeddings(
            emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor
        )  # (B, 512) = [time+class | spacing]
        if text_pooled_f is not None and text_pooled_i is not None:
            pooled = torch.cat([text_pooled_f.flatten(1), text_pooled_i.flatten(1)], dim=1)  # (B, 4096)
            text_terms = [self.text_pooled_proj(pooled)]  # (B, 256)
        else:
            text_terms = [self.text_pooled_proj(t.flatten(1)) for t in (text_pooled_f, text_pooled_i) if t is not None]
        if text_terms:
            text_emb = sum(text_terms)  # (B, 256)
            emb = emb + F.pad(text_emb, (0, emb.shape[1] - text_emb.shape[1]))  # text only in leading columns
        h = self.conv_in(x)
        h, _updated_down_block_res_samples = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        h = self.middle_block(h, emb, context)  # (B, 512, 8, 8, 8)
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual
        h = self._apply_up_blocks(h, emb, context, _updated_down_block_res_samples)
        h = self.out(h)
        h_tensor: Tensor = convert_to_tensor(h)
        return h_tensor


__all__ = ["AdaGNResBlock", "MaskedCrossAttention", "ReportUNet3D", "enable_adagn", "enable_context_masking"]
