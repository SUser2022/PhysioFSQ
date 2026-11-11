"""Finite scalar quantization used by the MCFSQ-MR experiments.

The implementation deliberately keeps two physical contracts small and
explicit:

* C1-D128 quantizes 128 scalar channels independently to eight levels.  It
  never constructs the astronomically large ``8**128`` mixed-radix table;
  indices are represented directly as ``(B,N,1,128)`` level labels.
* C32-D4 quantizes the 32 four-dimensional groups supplied by the attentive
  adapter and packs each group into one ``0..4095`` mixed-radix id.  Its labels
  are ``(B,N,32,4)`` when decoded.

The projection-heavy C1-D4 layout from the earlier equal-capacity ablation is
intentionally not part of this project.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torch import atanh, clamp, int32, tanh, tensor
from torch.amp import autocast


def exists(value):
    return value is not None


def default(*values):
    for value in values:
        if exists(value):
            return value
    return None


def round_ste(value):
    rounded = value.round()
    return value + (rounded - value).detach()


def floor_ste(value):
    floored = value.floor()
    return value + (floored - value).detach()


class FSQ(nn.Module):
    """Finite scalar quantizer with safe C1/C32 index contracts."""

    def __init__(
        self,
        levels,
        dim=None,
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        scale=None,
        allowed_dtypes=(torch.float32, torch.float64),
        channel_first=False,
        projection_has_bias=True,
        return_indices=True,
        force_quantization_f32=True,
        preserve_symmetry=False,
        noise_dropout=0.0,
        bound_hard_clamp=False,
    ):
        super().__init__()
        if isinstance(levels, int):
            levels = [int(levels)]
        levels = tuple(int(level) for level in levels)
        if not levels or any(level < 2 for level in levels):
            raise ValueError(f"levels must contain integers >=2, got {levels}")
        num_codebooks = int(num_codebooks)
        if num_codebooks < 1:
            raise ValueError("num_codebooks must be >= 1")

        self.register_buffer("_levels", tensor(levels, dtype=int32), persistent=False)
        # A C1-D128 quantizer is factorized over 128 scalar axes.  There is no
        # packed mixed-radix index for this layout (``8**128`` is not a
        # representable tensor scalar), so do not even materialize its basis.
        # C32-D4, in contrast, only needs the four-element basis
        # ``[1, 8, 64, 512]``.  Keep the basis in int64 for safe arithmetic.
        if num_codebooks == 1 and len(levels) > 4:
            basis_values = []
        else:
            basis_values = [1]
            for level in levels[:-1]:
                basis_values.append(basis_values[-1] * int(level))
        self.register_buffer("_basis", tensor(basis_values, dtype=torch.long), persistent=False)
        self.scale = scale
        self.preserve_symmetry = bool(preserve_symmetry)
        self.noise_dropout = float(noise_dropout)
        self.codebook_dim = len(levels)
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = self.codebook_dim * num_codebooks
        self.keep_num_codebooks_dim = default(
            keep_num_codebooks_dim, num_codebooks > 1
        )
        if num_codebooks > 1 and not self.keep_num_codebooks_dim:
            raise ValueError("multi-codebook FSQ must retain the C axis")
        self.dim = int(default(dim, self.effective_codebook_dim))
        self.channel_first = bool(channel_first)
        has_projections = self.dim != self.effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, self.effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(self.effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections
        self.return_indices = bool(return_indices)
        # C1-D128 has no packed vocabulary.  Expose the scalar level count
        # instead of ever evaluating 8**128; C32-D4 has a 4096-way vocabulary.
        self.codebook_size = (
            max(levels)
            if num_codebooks == 1 and self.codebook_dim > 4
            else int(torch.tensor(levels, dtype=torch.long).prod().item())
        )
        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = bool(force_quantization_f32)
        self.bound_hard_clamp = bool(bound_hard_clamp)
        # ``bound`` historically used this value as a default argument.  Keep
        # it explicit for the boundary-aware target API so that the geometry
        # recorded in a Student checkpoint is the same geometry used by the
        # production quantizer.
        self.bound_eps = 1e-3

    @property
    def is_scalar_c1(self):
        return self.num_codebooks == 1 and self.codebook_dim > 4

    def bound(self, z, eps=None, hard_clamp=False):
        eps = self.bound_eps if eps is None else float(eps)
        maybe_tanh = tanh if not hard_clamp else partial(clamp, min=-1.0, max=1.0)
        maybe_atanh = atanh if not hard_clamp else (lambda value: value)
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = maybe_atanh(offset / half_l)
        bounded = maybe_tanh(z + shift) * half_l - offset
        half_width = self._levels // 2
        return round_ste(bounded) / half_width

    def _unrounded_bound(self, z, *, eps=None, hard_clamp=None):
        """Return FSQ's physical pre-round coordinate.

        This is deliberately separate from :meth:`bound`: ``bound`` uses a
        straight-through rounded value for the production quantized code,
        while boundary-aware targets must inspect the value *before* that
        round.  The method is valid only for the non-symmetry, tanh branch
        used by the current C1/C32 checkpoints.
        """

        if self.preserve_symmetry:
            raise ValueError(
                "boundary-aware FSQ targets require preserve_symmetry=False; "
                "the floor-based symmetry branch has different boundaries"
            )
        hard_clamp = self.bound_hard_clamp if hard_clamp is None else bool(hard_clamp)
        if hard_clamp:
            raise ValueError(
                "boundary-aware FSQ targets require bound_hard_clamp=False; "
                "clamped plateaus are not invertible with the audited geometry"
            )
        eps = self.bound_eps if eps is None else float(eps)
        levels = self._levels.to(device=z.device, dtype=z.dtype)
        half_l = (levels - 1) * (1.0 + eps) / 2.0
        offset = torch.where(levels.remainder(2) == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def level_targets_from_projected(
        self,
        projected,
        *,
        temperature=1.0,
        return_margin=False,
    ):
        """Return hard labels and local geometric soft FSQ targets.

        ``projected`` must be the output of ``AttentiveFSQ.project`` (already
        shaped as the flattened ``C*D`` adapter output), not post-FSQ codes.
        The returned tuple is ``(labels, probabilities, confidence)`` with
        trailing shapes ``(C,D)``, ``(C,D,L)``, and ``(C,D)`` respectively.
        Only the two adjacent scalar levels surrounding the unrounded value
        receive non-zero probability.  This keeps the target local to the
        ordered FSQ geometry instead of applying generic label smoothing.
        """

        temperature = float(temperature)
        if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
            raise ValueError(f"boundary target temperature must be positive, got {temperature}")
        if self.preserve_symmetry or self.bound_hard_clamp:
            # Fail closed rather than silently using the wrong decision
            # boundaries for an unsupported FSQ branch.
            self._unrounded_bound(projected.reshape(-1, self.num_codebooks, self.codebook_dim))

        original_shape = projected.shape[:-1]
        projected_flat = projected.reshape(-1, projected.shape[-1])
        if projected_flat.shape[-1] != self.effective_codebook_dim:
            raise ValueError(
                "projected FSQ coordinates must have flattened C*D width "
                f"{self.effective_codebook_dim}, got {projected_flat.shape[-1]}"
            )
        grouped = projected_flat.reshape(-1, self.num_codebooks, self.codebook_dim)
        # The production path quantizes in FP32 under an autocast-disabled
        # context.  Reuse that dtype contract for deterministic labels and
        # geometry, then expose probabilities in the same stable FP32 dtype.
        if self.force_quantization_f32 and grouped.dtype not in self.allowed_dtypes:
            grouped = grouped.float()
        bounded = self._unrounded_bound(grouped)
        levels = self._levels.to(device=bounded.device)
        half_width = (levels // 2).to(dtype=bounded.dtype)
        labels = (bounded.round() + half_width).long()
        labels = torch.minimum(torch.maximum(labels, torch.zeros_like(labels)), levels.long() - 1)

        # Scalar centers are q in {-h, ..., h}; for even L=8 this is
        # [-4,-3,...,3], exactly the values round() sees before scale/shift.
        # The current physical experiments use one level count per digit
        # (C1 [8]^128 or C32 [8]^4).  Keep a clear error for mixed level lists
        # instead of broadcasting an incorrect center map.
        if not torch.all(levels == levels[0]):
            raise ValueError(
                "boundary-aware local targets currently require identical "
                f"levels per digit, got {tuple(int(v) for v in levels.tolist())}"
            )
        level_count = int(levels[0].item())
        centers_1d = torch.arange(
            level_count, device=bounded.device, dtype=bounded.dtype
        ) - half_width[0]
        # Clamp the tiny tanh tails to the outer finite centers.  The outer
        # regions are tails, not extra levels, and therefore remain one-hot.
        bounded_clamped = bounded.clamp(
            min=float(centers_1d[0]), max=float(centers_1d[-1])
        )
        lower = torch.floor(bounded_clamped - centers_1d[0]).long()
        lower = lower.clamp(0, level_count - 1)
        upper = (lower + 1).clamp(max=level_count - 1)
        lower_center = centers_1d[lower]
        fraction = (bounded_clamped - lower_center).clamp(0.0, 1.0)
        # A power-temperature transform sharpens/flattens only the two local
        # neighbors while preserving exact one-hot targets at stable centers.
        lower_mass = 1.0 - fraction
        upper_mass = fraction
        if temperature != 1.0:
            inv_temperature = 1.0 / temperature
            lower_mass = lower_mass.pow(inv_temperature)
            upper_mass = upper_mass.pow(inv_temperature)
            normalizer = (lower_mass + upper_mass).clamp_min(torch.finfo(bounded.dtype).tiny)
            lower_mass = lower_mass / normalizer
            upper_mass = upper_mass / normalizer
        probabilities = torch.zeros(
            *bounded.shape, level_count, dtype=bounded.dtype, device=bounded.device
        )
        probabilities.scatter_add_(-1, lower.unsqueeze(-1), lower_mass.unsqueeze(-1))
        probabilities.scatter_add_(-1, upper.unsqueeze(-1), upper_mass.unsqueeze(-1))
        entropy = -(
            probabilities.clamp_min(torch.finfo(bounded.dtype).tiny)
            * probabilities.clamp_min(torch.finfo(bounded.dtype).tiny).log()
        ).sum(dim=-1)
        confidence = (1.0 - entropy / torch.log(torch.tensor(
            float(level_count), device=bounded.device, dtype=bounded.dtype
        ))).clamp(0.0, 1.0)
        # Physical FSQ margin: distance to the nearest half-integer decision
        # boundary, normalized so a quantization center is 1 and a boundary
        # is 0. ``fraction`` is the distance from the lower center, so the
        # nearest-boundary distance is ``abs(fraction - 0.5)``.  Normalize by
        # the half-step so centers have margin 1 and boundaries have margin
        # 0. Using ``min(fraction, 1-fraction)`` would invert these semantics
        # and make the margin-weighted hard-CE control emphasize unstable
        # labels.
        margin = (2.0 * torch.abs(fraction - 0.5)).clamp(0.0, 1.0)
        shape = (*original_shape, self.num_codebooks, self.codebook_dim)
        result = (
            labels.reshape(shape),
            probabilities.reshape(*shape, level_count),
            confidence.reshape(shape),
        )
        if return_margin:
            return (*result, margin.reshape(shape))
        return result

    def symmetry_preserving_bound(self, z, hard_clamp=False):
        maybe_tanh = tanh if not hard_clamp else partial(clamp, min=-1.0, max=1.0)
        levels_minus_1 = self._levels - 1
        bracket = (levels_minus_1 * (maybe_tanh(z) + 1) / 2) + 0.5
        bracket = floor_ste(bracket)
        return (2.0 / levels_minus_1) * bracket - 1.0

    def quantize(self, z):
        bound_fn = (
            self.symmetry_preserving_bound
            if self.preserve_symmetry
            else self.bound
        )
        bounded = bound_fn(z, hard_clamp=self.bound_hard_clamp)
        if not self.training or self.noise_dropout == 0.0:
            return bounded
        offset_mask = torch.bernoulli(torch.full_like(bounded, self.noise_dropout)).bool()
        offset = torch.rand_like(bounded) - 0.5
        return torch.where(offset_mask, bounded + offset, bounded)

    def _scale_and_shift(self, normalized):
        half_width = self._levels // 2
        return normalized * half_width + half_width

    def _scale_and_shift_inverse(self, values):
        half_width = self._levels // 2
        return (values - half_width) / half_width

    def codes_to_indices(self, codes):
        """Pack C32 ``(...,C,D)`` codes without allocating a global table."""

        expected = (self.num_codebooks, self.codebook_dim)
        if tuple(codes.shape[-2:]) != expected:
            raise ValueError(f"expected codes trailing {expected}, got {tuple(codes.shape[-2:])}")
        labels = self._scale_and_shift(codes).round().long()
        if self.is_scalar_c1:
            # C1 is deliberately factorized: each coordinate is an eight-way
            # label and there is no single packed integer representing all
            # 128 coordinates.
            return labels
        basis = self._basis.to(labels.device).view(1, 1, 1, -1)
        return (labels * basis).sum(-1)

    def indices_to_level_indices(self, indices):
        """Decode packed C32 ids or pass through C1 factorized labels."""

        if self.is_scalar_c1:
            labels = torch.as_tensor(indices).long()
            if labels.shape[-2:] == (1, self.codebook_dim):
                pass
            elif labels.shape[-1:] == (self.codebook_dim,):
                labels = labels.unsqueeze(-2)
            expected = (*labels.shape[:-2], 1, self.codebook_dim)
            if tuple(labels.shape) != expected:
                raise ValueError(
                    "C1-D128 labels must have shape (...,1,128), got "
                    f"{tuple(labels.shape)}"
                )
            return labels
        values = torch.as_tensor(indices)
        if self.num_codebooks == 1:
            values = values.unsqueeze(-1)
        values = values.long().unsqueeze(-1)
        return (
            (values // self._basis.to(values.device))
            % self._levels.to(values.device)
        ).long()

    def level_indices_to_indices(self, labels, *, validate=True):
        labels = torch.as_tensor(labels)
        expected = (*labels.shape[:-2], self.num_codebooks, self.codebook_dim)
        if tuple(labels.shape) != expected:
            raise ValueError(
                f"expected level labels trailing ({self.num_codebooks},{self.codebook_dim}), "
                f"got {tuple(labels.shape)}"
            )
        levels = self._levels.to(labels.device)
        labels = labels.long()
        # Validation is useful at public API boundaries, but the frozen Teacher
        # invokes this operation inside a CUDA Graph where device-wide any()
        # reductions are illegal.  Its labels are produced by this FSQ itself,
        # so the capture-safe path can explicitly skip the redundant check.
        if validate and (torch.any(labels < 0) or torch.any(labels >= levels)):
            raise ValueError("level labels are outside the configured FSQ levels")
        if self.is_scalar_c1:
            return labels
        basis = self._basis.to(labels.device).view(
            *([1] * (labels.ndim - 1)), -1
        )
        packed = (labels * basis).sum(-1)
        return packed.squeeze(-1) if self.num_codebooks == 1 else packed

    def forward(self, z):
        # All model callers use channel-last patch tensors.  Flattening here
        # keeps the C1 and C32 paths identical and avoids dynamic allocations.
        original_shape = z.shape[:-1]
        z_flat = z.reshape(-1, z.shape[-1])
        if z_flat.shape[-1] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, found {z_flat.shape[-1]}")
        z_flat = self.project_in(z_flat)
        z_grouped = z_flat.reshape(-1, self.num_codebooks, self.codebook_dim)
        context = (
            partial(autocast, "cuda", enabled=False)
            if self.force_quantization_f32
            else nullcontext
        )
        with context():
            orig_dtype = z_grouped.dtype
            if self.force_quantization_f32 and orig_dtype not in self.allowed_dtypes:
                z_grouped = z_grouped.float()
            codes = self.quantize(z_grouped)
            labels = self._scale_and_shift(codes).round().long()
            if self.return_indices:
                if self.is_scalar_c1:
                    indices = labels.reshape(*original_shape, 1, self.codebook_dim)
                else:
                    indices = (labels * self._basis.to(labels.device).view(1, 1, 1, -1)).sum(-1)
                    indices = indices.reshape(*original_shape, self.num_codebooks)
                    if self.num_codebooks == 1 and not self.keep_num_codebooks_dim:
                        indices = indices.squeeze(-1)
            else:
                indices = None
            codes = codes.reshape(-1, self.effective_codebook_dim).to(orig_dtype)
        out = self.project_out(codes).reshape(*original_shape, self.dim)
        return out, indices

    def level_indices(self, z):
        """Return physical scalar labels without a pack/decode round trip.

        This is the indices-only path used by a frozen Teacher.  It deliberately
        applies the same projection, FP32 quantization, scale, shift and rounding
        operations as :meth:`forward`; it only omits the unused quantized output
        and, for C32, the mixed-radix packing operation.
        """

        original_shape = z.shape[:-1]
        z_flat = z.reshape(-1, z.shape[-1])
        if z_flat.shape[-1] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, found {z_flat.shape[-1]}")
        z_grouped = self.project_in(z_flat).reshape(
            -1, self.num_codebooks, self.codebook_dim
        )
        context = (
            partial(autocast, "cuda", enabled=False)
            if self.force_quantization_f32
            else nullcontext
        )
        with context():
            if (
                self.force_quantization_f32
                and z_grouped.dtype not in self.allowed_dtypes
            ):
                z_grouped = z_grouped.float()
            codes = self.quantize(z_grouped)
            labels = self._scale_and_shift(codes).round().long()
        return labels.reshape(
            *original_shape, self.num_codebooks, self.codebook_dim
        )
