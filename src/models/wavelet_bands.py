"""Torch-native stationary-wavelet MRA filter bank for paired waveforms.

PyWavelets is used once at construction time to obtain the linear, periodic
SWT-MRA impulse responses.  Signal filtering itself stays in torch, including
on CUDA.  Each finite-window input is reflection-extended by the measured
support radius before circular FFT convolution and cropped back to its
original time grid.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn


DEFAULT_FIVE_LEVEL_GROUPS = (
    ("A5",),
    ("D5",),
    ("D4",),
    ("D3", "D2", "D1"),
)


def canonical_band_groups(level: int, groups: Sequence[Sequence[str]] | None = None):
    """Validate a complete, non-overlapping grouping of SWT-MRA components."""

    level = int(level)
    if level < 1:
        raise ValueError(f"wavelet level must be >= 1, got {level}")
    if groups is None:
        if level == 5:
            groups = DEFAULT_FIVE_LEVEL_GROUPS
        elif level >= 3:
            groups = ((f"A{level}",), (f"D{level}",), (f"D{level - 1}",), tuple(f"D{i}" for i in range(level - 2, 0, -1)))
        else:
            groups = ((f"A{level}",),) + tuple((f"D{i}",) for i in range(level, 0, -1))

    normalized = []
    seen = set()
    valid = {f"A{level}"} | {f"D{i}" for i in range(1, level + 1)}
    for band_index, group in enumerate(groups):
        if isinstance(group, str):
            group = (group,)
        components = tuple(str(item).strip().upper() for item in group)
        if not components:
            raise ValueError(f"band group {band_index} is empty")
        for component in components:
            if component not in valid:
                raise ValueError(
                    f"invalid SWT-MRA component {component!r}; expected one of "
                    f"{sorted(valid)}"
                )
            if component in seen:
                raise ValueError(f"SWT-MRA component {component} appears more than once")
            seen.add(component)
        normalized.append(components)
    missing = valid - seen
    if missing:
        raise ValueError(f"band groups omit SWT-MRA components: {sorted(missing)}")
    if not normalized:
        raise ValueError("at least one band group is required")
    return tuple(normalized)


class WaveletBandDecomposer(nn.Module):
    """Full-length additive stationary wavelet MRA components.

    Input is ``[B,C,L]`` and output is ``[B,K,C,L]``.  The component filters
    are shift-invariant SWT-MRA responses; the final axis remains on the
    original sample grid so paired ECG/PPG patches retain the same alignment.
    """

    IMPLEMENTATION_VERSION = "swt_mra_fft_reflect_v1"
    RESPONSE_FFT_CACHE_MAX_ENTRIES = 4

    def __init__(
        self,
        sample_rate: float = 125.0,
        wavelet: str = "sym4",
        level: int = 5,
        band_groups: Sequence[Sequence[str]] | None = None,
        support_tolerance: float = 1e-12,
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.wavelet = str(wavelet)
        self.level = int(level)
        self.band_groups = canonical_band_groups(self.level, band_groups)
        self.support_tolerance = float(support_tolerance)
        if not math.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive and finite, got {sample_rate}")
        if not math.isfinite(self.support_tolerance) or not 0 < self.support_tolerance < 1:
            raise ValueError(
                "support_tolerance must be between zero and one, got "
                f"{support_tolerance}"
            )

        try:
            import numpy as np
            import pywt
        except ImportError as exc:  # pragma: no cover - environment-specific message
            raise ImportError(
                "WaveletBandDecomposer requires PyWavelets for construction-time "
                "SWT-MRA filter generation; torch signal processing does not use it"
            ) from exc

        try:
            wavelet_object = pywt.Wavelet(self.wavelet)
        except ValueError as exc:
            raise ValueError(f"unknown PyWavelets wavelet {self.wavelet!r}") from exc

        # Compact orthogonal/biorthogonal wavelet MRA filters have finite
        # support.  A conservative bound selects an impulse length large
        # enough to keep the periodic copies disjoint, including analysis and
        # synthesis cascades at the configured dilation depth.
        estimated_radius = 2 * (int(wavelet_object.dec_len) - 1) * (2**self.level - 1)
        impulse_length = 1 << max(
            12,
            int(math.ceil(math.log2(max(2 * estimated_radius + 1, 2)))),
        )
        impulse_length = max(impulse_length, 2**self.level)
        center = impulse_length // 2
        impulse = np.zeros(impulse_length, dtype=np.float64)
        impulse[center] = 1.0
        try:
            mra_components = pywt.mra(
                impulse,
                wavelet_object,
                level=self.level,
                transform="swt",
                mode="periodization",
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to generate stationary SWT-MRA kernels; verify the "
                "PyWavelets version and wavelet decomposition depth"
            ) from exc

        component_names = [f"A{self.level}"] + [
            f"D{i}" for i in range(self.level, 0, -1)
        ]
        if len(mra_components) != len(component_names):
            raise RuntimeError(
                "PyWavelets SWT-MRA returned an unexpected component count: "
                f"{len(mra_components)} vs {len(component_names)}"
            )
        by_name = {
            name: np.asarray(component, dtype=np.float64)
            for name, component in zip(component_names, mra_components)
        }

        # Establish a meaningful support guard from all grouped responses.
        # The impulse response is indexed relative to its input impulse; values
        # beyond the tolerance are floating-point roundoff from compact FIR
        # filtering, not useful filter support.
        all_group_kernels = []
        for group in self.band_groups:
            response = np.sum([by_name[name] for name in group], axis=0)
            peak = float(np.max(np.abs(response)))
            if not math.isfinite(peak) or peak == 0:
                raise RuntimeError(f"zero or non-finite wavelet response for {group}")
            offsets = np.arange(impulse_length, dtype=np.int64) - center
            offsets = ((offsets + impulse_length // 2) % impulse_length) - impulse_length // 2
            active = np.abs(response) > peak * self.support_tolerance
            if not np.any(active):
                raise RuntimeError(f"could not determine response support for {group}")
            radius = int(np.max(np.abs(offsets[active])))
            kernel = np.zeros(2 * radius + 1, dtype=np.float64)
            kernel[offsets[active] + radius] = response[active]
            all_group_kernels.append((radius, kernel))

        guard = max(radius for radius, _ in all_group_kernels)
        kernels = np.zeros((len(all_group_kernels), 2 * guard + 1), dtype=np.float64)
        for index, (radius, kernel) in enumerate(all_group_kernels):
            kernels[index, guard - radius : guard + radius + 1] = kernel
        self.guard_samples = int(guard)
        self.pywavelets_version = str(pywt.__version__)
        self.wavelet_filter_length = int(wavelet_object.dec_len)

        self.register_buffer(
            "kernel_offsets", torch.as_tensor(kernels, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "offsets",
            torch.arange(-guard, guard + 1, dtype=torch.long),
            persistent=False,
        )
        self._band_ranges_hz = self._make_band_ranges(component_names)
        # This opt-in cache is deliberately ordinary Python runtime state:
        # it must not add checkpoint keys or move as a registered buffer.
        self._response_fft_cache_enabled = False
        self._response_fft_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._response_fft_cache_binding = None

    @property
    def num_bands(self) -> int:
        return len(self.band_groups)

    def _make_band_ranges(self, component_names):
        del component_names
        ranges = {}
        for group in self.band_groups:
            lows, highs = [], []
            for component in group:
                if component.startswith("A"):
                    low, high = 0.0, self.sample_rate / (2 ** (self.level + 1))
                else:
                    detail_level = int(component[1:])
                    low = self.sample_rate / (2 ** (detail_level + 1))
                    high = self.sample_rate / (2**detail_level)
                lows.append(low)
                highs.append(high)
            label = "+".join(group)
            ranges[label] = [float(min(lows)), float(max(highs))]
        return ranges

    @property
    def metadata(self) -> dict:
        return {
            "implementation": self.IMPLEMENTATION_VERSION,
            "sample_rate_hz": self.sample_rate,
            "wavelet": self.wavelet,
            "wavelet_filter_length": self.wavelet_filter_length,
            "level": self.level,
            "groups": [list(group) for group in self.band_groups],
            "nominal_group_ranges_hz": dict(self._band_ranges_hz),
            "ranges_are_nominal_overlapping_responses": True,
            "boundary_extension": "reflection_without_repeating_edge",
            "boundary_guard_samples": self.guard_samples,
            "support_tolerance": self.support_tolerance,
            "mra_kernel_source": "PyWavelets SWT-MRA impulse response, periodization",
            "pywavelets_version": self.pywavelets_version,
            "output_time_grid": "same_length_and_sample_alignment_as_input",
        }

    @staticmethod
    def _reflection_indices(length: int, guard: int, device: torch.device):
        if length < 2:
            raise ValueError("wavelet decomposition requires at least two samples")
        positions = torch.arange(-guard, length + guard, device=device)
        period = 2 * (length - 1)
        folded = torch.remainder(positions, period)
        return torch.where(folded < length, folded, period - folded).long()

    def enable_response_fft_cache(self, enabled: bool = True) -> None:
        """Explicitly enable or disable caching of the fixed filter FFTs.

        Only the filter response spectrum is cached. Input-dependent signal
        FFTs are always recomputed. The default is disabled to preserve the
        historical execution path unless a caller opts in.
        """

        enabled = bool(enabled)
        if enabled != self._response_fft_cache_enabled:
            self.clear_response_fft_cache()
        self._response_fft_cache_enabled = enabled

    def clear_response_fft_cache(self) -> None:
        """Drop all runtime filter spectra without changing model state."""

        self._response_fft_cache.clear()
        self._response_fft_cache_binding = None

    def response_fft_cache_info(self) -> dict[str, int | bool]:
        """Return small diagnostics useful for controlled cache validation."""

        return {
            "enabled": self._response_fft_cache_enabled,
            "entries": len(self._response_fft_cache),
        }

    @staticmethod
    def _buffer_binding(buffer: torch.Tensor) -> tuple:
        return (
            int(buffer.data_ptr()),
            int(buffer._version),
            tuple(buffer.shape),
            str(buffer.device),
            str(buffer.dtype),
            bool(buffer.requires_grad),
        )

    def _response_cache_binding(self) -> tuple:
        """Describe every fixed value that determines an SWT response."""

        return (
            self.sample_rate,
            self.wavelet,
            self.level,
            tuple(tuple(group) for group in self.band_groups),
            self.support_tolerance,
            self.pywavelets_version,
            self.guard_samples,
            self._buffer_binding(self.kernel_offsets),
            self._buffer_binding(self.offsets),
        )

    def _make_response_fft(
        self, *, signal: torch.Tensor, fft_length: int
    ) -> torch.Tensor:
        if torch.is_inference_mode_enabled():
            # Lightning may run validation under inference_mode before the first
            # training forward. The filter spectrum is constant and must stay
            # an ordinary tensor: autograd needs to save it to differentiate
            # the later signal FFT multiplication.
            with torch.inference_mode(False):
                return self._build_response_fft_tensor(
                    signal=signal,
                    fft_length=fft_length,
                )
        return self._build_response_fft_tensor(
            signal=signal,
            fft_length=fft_length,
        )

    def _build_response_fft_tensor(
        self, *, signal: torch.Tensor, fft_length: int
    ) -> torch.Tensor:
        offsets = torch.remainder(self.offsets.to(signal.device), fft_length)
        kernels = self.kernel_offsets.to(device=signal.device)
        response = torch.zeros(
            self.num_bands, fft_length, device=signal.device, dtype=torch.float32
        )
        response.index_copy_(1, offsets, kernels)
        return torch.fft.rfft(response, dim=-1)

    def _get_response_fft(
        self, *, signal: torch.Tensor, original_length: int, fft_length: int
    ) -> torch.Tensor:
        binding = self._response_cache_binding()
        if binding != self._response_fft_cache_binding:
            # A load_state_dict call, in-place kernel edit, buffer replacement,
            # dtype/device move, or Python filter-parameter edit invalidates
            # every previously cached response.
            self._response_fft_cache.clear()
            self._response_fft_cache_binding = binding

        key = (
            binding,
            str(signal.device),
            str(signal.dtype),
            int(original_length),
            int(fft_length),
        )
        cached = self._response_fft_cache.get(key)
        if cached is not None:
            self._response_fft_cache.move_to_end(key)
            return cached

        response_fft = self._make_response_fft(signal=signal, fft_length=fft_length)
        self._response_fft_cache[key] = response_fft
        self._response_fft_cache.move_to_end(key)
        while len(self._response_fft_cache) > self.RESPONSE_FFT_CACHE_MAX_ENTRIES:
            self._response_fft_cache.popitem(last=False)
        return response_fft

    def _apply(self, fn, recurse: bool = True):
        # A plain-attribute tensor cache is not moved by nn.Module._apply.
        # Clear it before and after the move while retaining the opt-in flag.
        self.clear_response_fft_cache()
        result = super()._apply(fn, recurse=recurse)
        self.clear_response_fft_cache()
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.clear_response_fft_cache()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.clear_response_fft_cache()

    def __getstate__(self):
        # Full-module pickle should also omit ephemeral spectra. Native
        # checkpoints use state_dict, but this prevents accidental cache
        # serialization by callers that pickle a complete model.
        state = super().__getstate__()
        state["_response_fft_cache"] = OrderedDict()
        state["_response_fft_cache_binding"] = None
        return state

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError(f"expected signal [B,C,L], got {tuple(signal.shape)}")
        if not signal.is_floating_point():
            raise TypeError(f"wavelet input must be floating point, got {signal.dtype}")
        batch, channels, length = signal.shape
        if length < 2:
            raise ValueError("wavelet decomposition requires at least two samples")

        guard = self.guard_samples
        reflected_indices = self._reflection_indices(length, guard, signal.device)
        padded = signal.to(dtype=torch.float32).index_select(-1, reflected_indices)
        fft_length = padded.shape[-1]
        if self._response_fft_cache_enabled:
            response_fft = self._get_response_fft(
                signal=signal,
                original_length=length,
                fft_length=fft_length,
            )
        else:
            # Keep the original non-cache response construction inline so
            # the default path retains its historical operations and order.
            offsets = torch.remainder(self.offsets.to(signal.device), fft_length)
            kernels = self.kernel_offsets.to(device=signal.device)
            response = torch.zeros(
                self.num_bands, fft_length, device=signal.device, dtype=torch.float32
            )
            response.index_copy_(1, offsets, kernels)
            response_fft = torch.fft.rfft(response, dim=-1)
        signal_fft = torch.fft.rfft(padded, dim=-1)
        filtered = torch.fft.irfft(
            signal_fft[:, None, :, :] * response_fft[None, :, None, :],
            n=fft_length,
            dim=-1,
        )
        components = filtered[..., guard : guard + length]
        return components.to(dtype=signal.dtype).reshape(batch, self.num_bands, channels, length)

