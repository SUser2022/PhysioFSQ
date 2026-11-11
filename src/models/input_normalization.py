"""Input normalization shared by frequency-separated models."""

from __future__ import annotations

import math

import torch
from torch import nn


class BandChannelRMSNorm(nn.Module):
    """Normalize each window, frequency band, and channel independently.

    The RMS is computed over the complete time axis of ``[B,K,C,L]``. A
    learned scalar gain is shared across examples and time for each ``[K,C]``
    pair. The reduction and gain application use at least float32 arithmetic;
    the returned tensor keeps the input dtype.
    """

    def __init__(
        self,
        num_bands: int,
        num_channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if isinstance(num_bands, bool) or not isinstance(num_bands, int) or num_bands < 1:
            raise ValueError(f"num_bands must be a positive integer, got {num_bands!r}")
        if (
            isinstance(num_channels, bool)
            or not isinstance(num_channels, int)
            or num_channels < 1
        ):
            raise ValueError(
                f"num_channels must be a positive integer, got {num_channels!r}"
            )
        try:
            eps = float(eps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"eps must be finite and positive, got {eps!r}") from exc
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"eps must be finite and positive, got {eps!r}")

        self.num_bands = num_bands
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_bands, num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"BandChannelRMSNorm input must be a tensor, got {type(x).__name__}")
        if x.ndim != 4:
            raise ValueError(f"input must have shape [B,K,C,L], got {tuple(x.shape)}")
        if x.shape[1] != self.num_bands or x.shape[2] != self.num_channels:
            raise ValueError(
                "input band/channel dimensions do not match the module: "
                f"expected [B,{self.num_bands},{self.num_channels},L], got {tuple(x.shape)}"
            )
        if x.shape[0] < 1 or x.shape[3] < 1:
            raise ValueError(f"batch and time dimensions must be positive, got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"input must be floating point, got {x.dtype}")

        # Half precision squaring can underflow for quiet bands. Keep all RMS
        # arithmetic in float32 for fp16/bf16/fp32 inputs, and preserve the
        # greater precision of float64 inputs.
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        work = x.to(dtype=compute_dtype)
        mean_square = work.square().mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        gain = self.weight.to(device=x.device, dtype=compute_dtype).view(
            1, self.num_bands, self.num_channels, 1
        )
        return (work * inverse_rms * gain).to(dtype=x.dtype)
