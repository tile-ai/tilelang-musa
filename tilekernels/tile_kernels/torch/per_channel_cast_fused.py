from typing import Optional, Union

import torch

import tile_kernels
from tile_kernels.quant.types import QuantTensor


def per_channel_cast_fused(
    x: Union[torch.Tensor, QuantTensor],
    num_per_tokens: int,
    num_per_channels: Optional[int],
    round_sf: bool,
    pos_to_token: Optional[torch.Tensor],
) -> QuantTensor:
    """Cast a matrix to FP8 with per-channel scaling, optionally fusing resf and token expansion (PyTorch reference).

    Args:
        x: Input tensor of shape (num_tokens, hidden), either a plain tensor
            or a ``QuantTensor`` ``(data, sf_invs)`` for rescaling FP8 inputs.
        num_per_tokens: Number of tokens in each scaling block.
        num_per_channels: Number of channels in each input scaling block, or
            ``None`` if ``x`` is not a ``QuantTensor``.
        round_sf: Whether to round scaling factors to powers of two.
        pos_to_token: Optional int32 index tensor for token expansion/gather.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """
    is_fused_cast_back = isinstance(x, tuple)
    if pos_to_token is not None:
        gather_idx = pos_to_token.clamp(min=0).to(torch.int64)
        valid_mask = (pos_to_token >= 0).unsqueeze(1)
        if is_fused_cast_back:
            assert num_per_channels is not None
            # MUSA does not implement advanced indexing for float8 tensors.
            # Dequantize first, then gather and re-quantize below.
            x_data = tile_kernels.torch.cast_back(x, 'fp32', (1, num_per_channels))
            x = x_data[gather_idx]
        else:
            x = x[gather_idx]

        x = torch.where(valid_mask, x.to(torch.float32), torch.zeros_like(x, dtype=torch.float32))
        is_fused_cast_back = False
    x_block_size = (1, num_per_channels) if is_fused_cast_back else None
    return tile_kernels.torch.cast(x, 'e4m3', block_size=(num_per_tokens, 1), x_block_size=x_block_size, round_sf=round_sf)
