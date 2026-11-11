"""Final PhysioFSQ four-band GRU readout for downstream transfer."""

from __future__ import annotations

import math
import numbers
from typing import Any, Sequence

import torch
from torch import nn

from .frequency_student import FREQUENCY_SCALAR_DIM, FrequencyStudentBackbone


class FrequencyDownstreamModel(nn.Module):
    """Four-band concatenation, single-layer GRU, and task-specific MLP.

    Inputs have shape ``[W, 50, 25]``. The backbone returns four band
    representations per patch. Those representations are concatenated to
    ``[W, 50, 512]``; all real windows belonging to one target are joined in
    stored order and processed as one temporal sequence. Only the last shared
    temporal Transformer block and this readout are trainable.

    The module names ``head``, ``window_gru``, and ``gru_projection`` retain
    the state-dict layout used by the final seed-42 checkpoints.
    """

    def __init__(
        self,
        backbone: FrequencyStudentBackbone,
        output_dim: int,
        *,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, FrequencyStudentBackbone):
            raise TypeError("backbone must be a FrequencyStudentBackbone")
        if int(backbone.num_bands) != 4:
            raise ValueError(f"the final readout requires 4 bands, got {backbone.num_bands}")
        if FREQUENCY_SCALAR_DIM != 128:
            raise RuntimeError("the released FSQ scalar representation must be 128-dimensional")
        if isinstance(output_dim, bool) or not isinstance(output_dim, numbers.Integral):
            raise TypeError("output_dim must be a positive integer")
        if int(output_dim) < 1:
            raise ValueError("output_dim must be positive")
        if isinstance(head_dropout, bool) or not isinstance(head_dropout, numbers.Real):
            raise TypeError("head_dropout must be a finite number in [0, 1)")
        if not math.isfinite(float(head_dropout)) or not 0.0 <= float(head_dropout) < 1.0:
            raise ValueError("head_dropout must be a finite number in [0, 1)")
        if not backbone.transformer.layers:
            raise ValueError("the Student must contain at least one temporal Transformer block")

        self.backbone = backbone
        self.output_dim = int(output_dim)
        self.head_dropout = float(head_dropout)
        self.num_bands = 4
        self.feature_dim = FREQUENCY_SCALAR_DIM
        self.num_patches = int(backbone.num_patches)
        self.patch_size = int(backbone.patch_size)
        self.mode = "last_layer"
        self.readout = "concat_mlp"
        self.window_aggregation = "gru"
        # Legacy constructor metadata retained by the saved seed-42 format;
        # the final GRU path does not use the former band-fusion dimension.
        self.fusion_dim = 128
        self.unfreeze_last_n_layers = 1

        # Preserve the construction order used by the final experiment so a
        # fixed seed produces the same initial head, GRU, and projection.
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(256, self.output_dim),
        )
        self.window_gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.gru_projection = nn.Linear(256, 512)

        self.backbone.requires_grad_(False)
        self.backbone.transformer.layers[-1].requires_grad_(True)
        self._apply_training_policy(self.training)

    def _apply_training_policy(self, training: bool) -> None:
        self.backbone.eval()
        transformer = self.backbone.transformer
        transformer.training = bool(training)
        for index, layer in enumerate(transformer.layers):
            layer.train(bool(training) if index == len(transformer.layers) - 1 else False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_training_policy(bool(mode))
        return self

    @staticmethod
    def _validate_offsets(
        window_offsets: torch.Tensor | Sequence[int] | None,
        window_count: int,
    ) -> list[int]:
        if window_offsets is None:
            return list(range(window_count + 1))
        offsets = torch.as_tensor(window_offsets)
        if offsets.ndim != 1 or offsets.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        }:
            raise ValueError("window_offsets must be a one-dimensional integer sequence")
        values = [int(value) for value in offsets.detach().cpu().tolist()]
        if len(values) < 2 or values[0] != 0 or values[-1] != window_count:
            raise ValueError("window_offsets must start at 0 and end at the number of windows")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("each target must contain at least one real window")
        return values

    def forward(
        self,
        patches: torch.Tensor,
        window_offsets: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(patches) or patches.ndim != 3:
            raise ValueError("patches must have shape [W, 50, 25]")
        if tuple(patches.shape[1:]) != (self.num_patches, self.patch_size):
            raise ValueError(
                f"patches must have shape [W,{self.num_patches},{self.patch_size}], "
                f"got {tuple(patches.shape)}"
            )
        if patches.shape[0] < 1 or not patches.is_floating_point():
            raise ValueError("patches must contain at least one floating-point window")

        offsets = self._validate_offsets(window_offsets, int(patches.shape[0]))
        band_features = self.backbone(patches)
        expected = (patches.shape[0], 4, self.num_patches, self.feature_dim)
        if tuple(band_features.shape) != expected:
            raise RuntimeError(f"backbone must return [W,4,50,128], got {tuple(band_features.shape)}")

        # [W, 4, N, 128] -> [W, N, 512]
        patch_sequence = band_features.permute(0, 2, 1, 3).reshape(
            patches.shape[0], self.num_patches, 512
        )
        target_features = []
        for start, end in zip(offsets, offsets[1:]):
            sequence = patch_sequence[start:end].reshape(1, (end - start) * self.num_patches, 512)
            _, hidden = self.window_gru(sequence)
            target_features.append(self.gru_projection(hidden[-1, 0]))
        features = torch.stack(target_features, dim=0)
        return self.head(features)

    def parameter_groups(
        self,
        backbone_lr: float,
        readout_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        values = {
            "backbone_lr": backbone_lr,
            "readout_lr": readout_lr,
            "weight_decay": weight_decay,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < (0.0 if name == "weight_decay" else 1e-30):
                raise ValueError(f"invalid {name}: {value}")

        backbone_parameters = [
            parameter for parameter in self.backbone.parameters() if parameter.requires_grad
        ]
        readout_parameters = [
            parameter
            for module in (self.window_gru, self.gru_projection, self.head)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        groups = [
            {
                "name": "backbone",
                "params": backbone_parameters,
                "lr": float(backbone_lr),
                "weight_decay": float(weight_decay),
            },
            {
                "name": "downstream_head",
                "params": readout_parameters,
                "lr": float(readout_lr),
                "weight_decay": float(weight_decay),
            },
        ]
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError("optimizer parameter groups contain duplicate parameters")
        expected_ids = {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad
        }
        if set(ids) != expected_ids or not all(group["params"] for group in groups):
            raise RuntimeError("optimizer groups do not exactly cover trainable parameters")
        return groups

    def trainable_parameter_summary(self) -> dict[str, Any]:
        trainable = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        frozen = [
            name for name, parameter in self.named_parameters() if not parameter.requires_grad
        ]
        return {
            "mode": self.mode,
            "readout": self.readout,
            "window_aggregation": self.window_aggregation,
            "head_dropout": self.head_dropout,
            "output_dim": self.output_dim,
            "num_bands": self.num_bands,
            "feature_dim_per_band": self.feature_dim,
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "frozen_parameter_count": sum(
                parameter.numel() for parameter in self.parameters() if not parameter.requires_grad
            ),
            "trainable_parameter_names": [name for name, _ in trainable],
            "frozen_parameter_names": frozen,
        }
