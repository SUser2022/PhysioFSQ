"""Single-channel Student for band-specific dual-channel FSQ targets.

The Teacher sees a synchronized ECG+PPG pair and produces one quantized
representation per wavelet band.  This module keeps that full-pair target
contract while allowing each Student to observe one channel only.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import lightning as L
import torch
import torch.nn as nn

from .input_normalization import BandChannelRMSNorm
from .wavelet_bands import WaveletBandDecomposer
from ..utils.training_acceleration import build_training_adamw

FREQUENCY_STUDENT_V1_SCHEMA = "frequency-student-v1"
FREQUENCY_STUDENT_V2_SCHEMA = "frequency-student-v2"
# Keep the historical public name stable for callers that validate v1 runs.
FREQUENCY_STUDENT_SCHEMA = FREQUENCY_STUDENT_V1_SCHEMA
FREQUENCY_STUDENT_INPUT_V2_SCHEMA = "frequency-student-input-v2"
FREQUENCY_Q_TARGET_TYPE = "frequency_q"
FREQUENCY_QUANTIZER_FAMILY = "frequency_fsq"
FREQUENCY_SCALAR_DIM = 128
FREQUENCY_LEVELS = (8,) * FREQUENCY_SCALAR_DIM
RAW_PATCH_REPRESENTATION = "raw_patches"
WAVELET_BAND_REPRESENTATION = "wavelet_bands"

def _json_metadata(value: Any, *, name: str) -> dict[str, Any]:
    """Return a detached JSON-compatible metadata dictionary."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only JSON-compatible values") from error


def _band_count(metadata: Mapping[str, Any]) -> Optional[int]:
    for key in ("num_bands", "band_count", "num_groups"):
        value = metadata.get(key)
        if value is not None:
            return int(value)
    bands = metadata.get("bands")
    if isinstance(bands, (tuple, list)):
        return len(bands)
    groups = metadata.get("band_groups")
    if isinstance(groups, (tuple, list)):
        return len(groups)
    return None


def _torch_load(path: Path, *, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument.
        return torch.load(path, map_location=map_location)


def _checkpoint_tensor_matches(
    state_dict: Mapping[str, Any],
    key: str,
    expected: torch.Tensor,
    *,
    description: str,
) -> None:
    """Reject checkpoint buffers that disagree with reconstructed constants."""

    saved = state_dict.get(key)
    reference = expected.detach().cpu()
    if (
        not torch.is_tensor(saved)
        or saved.dtype != reference.dtype
        or tuple(saved.shape) != tuple(reference.shape)
        or not torch.equal(saved.detach().cpu(), reference)
    ):
        raise ValueError(
            f"Student checkpoint {description} buffer disagrees with metadata"
        )


def _checkpoint_learned_tensor_matches(
    state_dict: Mapping[str, Any],
    key: str,
    expected: torch.Tensor,
    *,
    description: str,
) -> None:
    """Validate a learned checkpoint tensor's shape/type without fixing its value."""

    saved = state_dict.get(key)
    if (
        not torch.is_tensor(saved)
        or saved.dtype != expected.dtype
        or tuple(saved.shape) != tuple(expected.shape)
        or not bool(torch.isfinite(saved.detach()).all())
    ):
        raise ValueError(
            f"Student checkpoint {description} has an invalid shape, dtype, or value"
        )


def _encoder_scaling_contract(
    value: Optional[Mapping[str, Any]], *, num_bands: int
) -> tuple[str, Optional[float], Optional[torch.Tensor]]:
    """Validate Teacher input scaling and resolve Student normalization mode."""

    if value is None:
        return "fixed", None, torch.ones(num_bands, 2, dtype=torch.float64)
    if not isinstance(value, Mapping):
        raise ValueError("Teacher encoder_input_scaling must be a mapping")
    scaling = _json_metadata(value, name="Teacher encoder_input_scaling")
    scaling_type = scaling.get("type")
    if scaling_type == "fixed_band_channel_divisors":
        if scaling.get("applied_before") != "patch_embedding":
            raise ValueError("Teacher encoder_input_scaling has an unsupported location")
        try:
            matrix = torch.as_tensor(scaling.get("divisors"), dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("Teacher encoder_input_scaling divisors are invalid") from error
        if tuple(matrix.shape) != (num_bands, 2):
            raise ValueError("Teacher encoder_input_scaling divisors have invalid shape")
        if not bool(torch.isfinite(matrix).all()) or not bool((matrix > 0).all()):
            raise ValueError(
                "Teacher encoder_input_scaling divisors must be finite and positive"
            )
        return "fixed", None, matrix
    if scaling_type == "identity":
        if set(scaling) != {"type", "applied_before"} or scaling.get(
            "applied_before"
        ) != "patch_embedding":
            raise ValueError("identity encoder_input_scaling metadata is malformed")
        return "fixed", None, torch.ones(num_bands, 2, dtype=torch.float64)
    if scaling_type == "per_band_channel_rmsnorm":
        required = {
            "type", "applied_before", "reduction", "eps", "gain_shape", "gain_init"
        }
        if set(scaling) != required:
            raise ValueError(
                "Teacher RMSNorm scaling metadata keys do not match the supported contract"
            )
        if scaling.get("applied_before") != "patch_embedding":
            raise ValueError("Teacher encoder_input_scaling has an unsupported location")
        if scaling.get("reduction") != "full_window_time":
            raise ValueError("Teacher RMSNorm must reduce over full_window_time")
        if scaling.get("gain_shape") != [num_bands, 2]:
            raise ValueError("Teacher RMSNorm gain_shape must be [num_bands,2]")
        gain_init = scaling.get("gain_init")
        if isinstance(gain_init, bool) or not isinstance(gain_init, (int, float)) or float(
            gain_init
        ) != 1.0:
            raise ValueError("Teacher RMSNorm gain_init must be 1.0")
        eps = scaling.get("eps")
        if isinstance(eps, bool):
            raise ValueError("Teacher RMSNorm eps must be finite and positive")
        try:
            eps = float(eps)
        except (TypeError, ValueError) as error:
            raise ValueError("Teacher RMSNorm eps must be finite and positive") from error
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Teacher RMSNorm eps must be finite and positive")
        return "band_rmsnorm", eps, None
    raise ValueError("unsupported Teacher encoder_input_scaling metadata")


def _validate_normalization_choice(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or value not in {"fixed", "band_rmsnorm"}:
        raise ValueError(
            "input_normalization must be exactly 'fixed' or 'band_rmsnorm', "
            f"got {value!r}"
        )
    return value


def _validate_rmsnorm_eps(value: Any, *, name: str = "input_rmsnorm_eps") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class FrequencyStudentBackbone(nn.Module):
    """Patch encoder with one continuous prediction head per Teacher band.

    Input is explicitly single-channel patches shaped ``[B, N, P]``.  The
    returned tensor is ``[B, K, N, 128]``; the frequency axis remains
    available to downstream tasks.
    """

    def __init__(
        self,
        *,
        num_bands: int,
        patch_size: int = 25,
        num_patches: int = 50,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        drop_rate: float = 0.1,
        band_metadata: Optional[Mapping[str, Any]] = None,
        input_representation: str = RAW_PATCH_REPRESENTATION,
        cross_band_layers: int = 0,
        input_metadata: Optional[Mapping[str, Any]] = None,
        input_normalization: str = "fixed",
        input_rmsnorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.drop_rate = float(drop_rate)
        self.input_representation = str(input_representation)
        self.cross_band_layers = int(cross_band_layers)
        self.input_normalization = _validate_normalization_choice(
            input_normalization
        )
        assert self.input_normalization is not None
        self.input_rmsnorm_eps = _validate_rmsnorm_eps(input_rmsnorm_eps)
        self.input_norm: Optional[BandChannelRMSNorm] = None
        if self.input_representation not in {
            RAW_PATCH_REPRESENTATION,
            WAVELET_BAND_REPRESENTATION,
        }:
            raise ValueError(
                "input_representation must be 'raw_patches' or 'wavelet_bands'"
            )
        if self.cross_band_layers < 0:
            raise ValueError("cross_band_layers must be non-negative")
        if self.input_representation == RAW_PATCH_REPRESENTATION and input_metadata is not None:
            raise ValueError("raw_patches Student cannot carry wavelet input_metadata")
        self.band_metadata = (
            _json_metadata(band_metadata, name="band_metadata")
            if band_metadata is not None
            else None
        )
        if self.num_bands < 1:
            raise ValueError("num_bands must be positive")
        if min(self.patch_size, self.num_patches, self.embed_dim, self.num_layers) < 1:
            raise ValueError("patch_size, num_patches, embed_dim and num_layers must be positive")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.input_metadata = None
        self.decomposer = None
        if self.input_representation == WAVELET_BAND_REPRESENTATION:
            self.input_metadata = _json_metadata(
                input_metadata, name="input_metadata"
            )
            self._initialize_wavelet_input(self.input_metadata)
        elif self.input_normalization == "band_rmsnorm":
            raise ValueError(
                "band_rmsnorm input normalization requires wavelet_bands input"
            )

        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_size, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        if self.input_representation == WAVELET_BAND_REPRESENTATION:
            self.band_embed = nn.Parameter(
                torch.zeros(1, self.num_bands, 1, self.embed_dim)
            )
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.embed_dim,
            dropout=self.drop_rate,
            activation=nn.GELU(),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, self.num_layers)
        self.norm = nn.LayerNorm(self.embed_dim)
        if self.input_representation == WAVELET_BAND_REPRESENTATION:
            cross_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                dim_feedforward=4 * self.embed_dim,
                dropout=self.drop_rate,
                activation=nn.GELU(),
                batch_first=True,
                norm_first=True,
            )
            if self.cross_band_layers:
                self.cross_band_encoder = nn.TransformerEncoder(
                    cross_layer, self.cross_band_layers
                )
            else:
                # Keep the established initialization order for the trained
                # single-channel backbone. The temporary encoder contributes
                # no parameters or forward operations to the final model.
                nn.TransformerEncoder(cross_layer, 1)
                self.cross_band_encoder = nn.Identity()
            self.cross_band_norm = nn.LayerNorm(self.embed_dim)
        self.band_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.embed_dim),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, FREQUENCY_SCALAR_DIM),
                )
                for _ in range(self.num_bands)
            ]
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.input_representation == WAVELET_BAND_REPRESENTATION:
            nn.init.trunc_normal_(self.band_embed, std=0.02)

    def _initialize_wavelet_input(self, input_metadata: Mapping[str, Any]) -> None:
        expected_keys = {
            "schema", "representation", "wavelet_mra", "encoder_input_scaling",
            "selected_channel_index", "selected_channel_divisors", "cross_band_layers",
        }
        if set(input_metadata) != expected_keys:
            raise ValueError(
                "wavelet input_metadata must contain exactly "
                f"{sorted(expected_keys)}, got {sorted(input_metadata)}"
            )
        if input_metadata.get("schema") != FREQUENCY_STUDENT_INPUT_V2_SCHEMA:
            raise ValueError("unsupported Student input_metadata schema")
        if input_metadata.get("representation") != WAVELET_BAND_REPRESENTATION:
            raise ValueError("wavelet input_metadata representation mismatch")
        if int(input_metadata.get("cross_band_layers", -1)) != self.cross_band_layers:
            raise ValueError("input_metadata cross_band_layers mismatch")
        wavelet_metadata = _json_metadata(
            input_metadata.get("wavelet_mra"), name="input_metadata.wavelet_mra"
        )
        try:
            decomposer = WaveletBandDecomposer(
                sample_rate=float(wavelet_metadata["sample_rate_hz"]),
                wavelet=str(wavelet_metadata["wavelet"]),
                level=int(wavelet_metadata["level"]),
                band_groups=wavelet_metadata["groups"],
                support_tolerance=float(wavelet_metadata["support_tolerance"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid wavelet_mra metadata for Student input") from error
        if decomposer.metadata != wavelet_metadata:
            differing = sorted(
                key
                for key in set(decomposer.metadata) | set(wavelet_metadata)
                if decomposer.metadata.get(key) != wavelet_metadata.get(key)
            )
            raise ValueError(
                "Student wavelet implementation metadata differs from checkpoint/Teacher: "
                f"{differing}"
            )
        if decomposer.num_bands != self.num_bands:
            raise ValueError(
                f"Student wavelet defines {decomposer.num_bands} bands, "
                f"expected {self.num_bands}"
            )

        selected_channel = int(input_metadata.get("selected_channel_index", -1))
        if selected_channel not in (0, 1):
            raise ValueError("input_metadata selected_channel_index must be 0 or 1")
        scaling = input_metadata.get("encoder_input_scaling")
        scaling_mode, scaling_eps, selected_matrix = _encoder_scaling_contract(
            scaling, num_bands=self.num_bands
        )
        if scaling_mode != self.input_normalization:
            raise ValueError(
                "Student input_normalization disagrees with Teacher "
                "encoder_input_scaling metadata"
            )
        if scaling_mode == "band_rmsnorm":
            assert scaling_eps is not None
            if not math.isclose(
                self.input_rmsnorm_eps, scaling_eps, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(
                    "Student input_rmsnorm_eps disagrees with Teacher metadata"
                )
            if input_metadata.get("selected_channel_divisors") is not None:
                raise ValueError(
                    "band_rmsnorm input_metadata must not contain stale fixed divisors"
                )
            self.input_norm = BandChannelRMSNorm(
                num_bands=self.num_bands,
                num_channels=1,
                eps=self.input_rmsnorm_eps,
            )
        else:
            assert selected_matrix is not None
            selected = selected_matrix[:, selected_channel]
            try:
                declared = torch.as_tensor(
                    input_metadata.get("selected_channel_divisors"),
                    dtype=torch.float64,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    "selected_channel_divisors must contain one finite value per band"
                ) from error
            if tuple(declared.shape) != (self.num_bands,) or not bool(
                torch.isfinite(declared).all()
            ):
                raise ValueError(
                    "selected_channel_divisors must contain one finite value per band"
                )
            if not torch.equal(declared, selected):
                raise ValueError(
                    "selected_channel_divisors disagree with the Teacher modality column"
                )
            self.register_buffer(
                "input_band_channel_divisors_tensor",
                selected.to(dtype=torch.float32),
                persistent=True,
            )
        self.decomposer = decomposer

    def decompose_patches(
        self, single_channel_patches: torch.Tensor, *, scaled: bool = True
    ) -> torch.Tensor:
        """Return aligned per-band patches ``[B,K,N,P]`` for v2 inputs."""

        if self.input_representation != WAVELET_BAND_REPRESENTATION:
            raise RuntimeError("decompose_patches requires input_representation='wavelet_bands'")
        if not torch.is_tensor(single_channel_patches) or single_channel_patches.ndim != 3:
            raise ValueError("single_channel_patches must have shape [B,N,P]")
        expected_shape = (self.num_patches, self.patch_size)
        if tuple(single_channel_patches.shape[1:]) != expected_shape:
            raise ValueError(
                "wavelet Student requires the full Teacher window with [N,P]="
                f"{expected_shape}, got {tuple(single_channel_patches.shape[1:])}"
            )
        if not single_channel_patches.is_floating_point():
            raise TypeError("single_channel_patches must be floating point")
        batch = single_channel_patches.shape[0]
        waveform = single_channel_patches.reshape(
            batch, 1, self.num_patches * self.patch_size
        )
        assert self.decomposer is not None
        bands = self.decomposer(waveform).squeeze(2)
        patches = bands.reshape(
            batch, self.num_bands, self.num_patches, self.patch_size
        )
        if scaled:
            if self.input_normalization == "band_rmsnorm":
                assert self.input_norm is not None
                full_window = patches.reshape(
                    batch, self.num_bands, 1, self.num_patches * self.patch_size
                )
                patches = self.input_norm(full_window).reshape_as(patches)
            else:
                divisors = self.input_band_channel_divisors_tensor.to(
                    device=patches.device, dtype=patches.dtype
                )
                patches = patches / divisors[None, :, None, None]
        return patches

    def encode_tokens(self, single_channel_patches: torch.Tensor) -> torch.Tensor:
        if single_channel_patches.ndim != 3:
            raise ValueError(
                "single-channel patches must have shape [B,N,P], got "
                f"{tuple(single_channel_patches.shape)}"
            )
        if single_channel_patches.shape[1] > self.num_patches:
            raise ValueError(
                f"received {single_channel_patches.shape[1]} patches; maximum is "
                f"{self.num_patches}"
            )
        if single_channel_patches.shape[2] != self.patch_size:
            raise ValueError(
                f"patch width must be {self.patch_size}, got {single_channel_patches.shape[2]}"
            )
        x = self.patch_embed(single_channel_patches)
        x = x + self.pos_embed[:, : x.shape[1]].to(dtype=x.dtype)
        return self.norm(self.transformer(x))

    def forward(self, single_channel_patches: torch.Tensor) -> torch.Tensor:
        if self.input_representation == WAVELET_BAND_REPRESENTATION:
            band_patches = self.decompose_patches(single_channel_patches, scaled=True)
            batch, bands, patches, width = band_patches.shape
            x = self.patch_embed(
                band_patches.reshape(batch * bands, patches, width)
            )
            x = x.reshape(batch, bands, patches, self.embed_dim)
            x = x + self.pos_embed[:, :patches].to(dtype=x.dtype).unsqueeze(1)
            x = x + self.band_embed.to(dtype=x.dtype)
            x = self.transformer(
                x.reshape(batch * bands, patches, self.embed_dim)
            )
            x = self.norm(x).reshape(batch, bands, patches, self.embed_dim)
            if self.cross_band_layers:
                cross_input = x.permute(0, 2, 1, 3).reshape(
                    batch * patches, bands, self.embed_dim
                )
                cross_output = self.cross_band_encoder(cross_input)
                x = cross_output.reshape(
                    batch, patches, bands, self.embed_dim
                ).permute(0, 2, 1, 3)
            x = self.cross_band_norm(x)
            return torch.stack(
                [self.band_heads[index](x[:, index]) for index in range(self.num_bands)],
                dim=1,
            )
        tokens = self.encode_tokens(single_channel_patches)
        return torch.stack([head(tokens) for head in self.band_heads], dim=1)

    @staticmethod
    def flatten_frequency_tokens(q_hat: torch.Tensor) -> torch.Tensor:
        """Flatten bands into each token for a strict linear feature probe."""

        if q_hat.ndim != 4:
            raise ValueError("q_hat must have shape [B,K,N,D]")
        return q_hat.transpose(1, 2).reshape(q_hat.shape[0], q_hat.shape[2], -1)


class FrequencyFSQStudent(L.LightningModule):
    """Train a single-channel Student against frozen full-pair Teacher Q."""

    def __init__(
        self,
        teacher_checkpoint: Optional[str] = None,
        *,
        encoder_input_channel: int = 1,
        patch_size: int = 25,
        num_patches: int = 50,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        drop_rate: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        mask_ratio: float = 0.0,
        num_bands: Optional[int] = None,
        band_metadata: Optional[Mapping[str, Any]] = None,
        input_representation: str = RAW_PATCH_REPRESENTATION,
        cross_band_layers: int = 0,
        input_band_channel_divisors: Optional[list[float]] = None,
        input_normalization: Optional[str] = None,
        input_rmsnorm_eps: Optional[float] = None,
        teacher_q_cache_dir: Optional[str] = None,
        teacher_q_cache_prepared_dir: Optional[str] = None,
        teacher_q_cache_allow_smoke: bool = False,
        wavelet_response_cache: bool = False,
    ) -> None:
        super().__init__()
        if int(encoder_input_channel) not in (0, 1):
            raise ValueError("encoder_input_channel must be 0 (ECG) or 1 (PPG)")
        if float(mask_ratio) != 0.0:
            raise ValueError("Frequency Student currently requires mask_ratio=0.0")
        if float(learning_rate) <= 0 or float(weight_decay) < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if input_representation not in {
            RAW_PATCH_REPRESENTATION,
            WAVELET_BAND_REPRESENTATION,
        }:
            raise ValueError(
                "input_representation must be 'raw_patches' or 'wavelet_bands'"
            )
        if int(cross_band_layers) < 0:
            raise ValueError("cross_band_layers must be non-negative")
        requested_normalization = _validate_normalization_choice(input_normalization)
        requested_rmsnorm_eps = (
            None
            if input_rmsnorm_eps is None
            else _validate_rmsnorm_eps(input_rmsnorm_eps)
        )

        # Do not save the Teacher path in Lightning hparams: inference-only
        # Student checkpoints must load without the Teacher artifact present.
        self.save_hyperparameters(
            ignore=[
                "teacher_checkpoint", "band_metadata", "teacher_q_cache_dir",
                "teacher_q_cache_prepared_dir", "teacher_q_cache_allow_smoke",
                "wavelet_response_cache",
            ],
        )
        self.encoder_input_channel = int(encoder_input_channel)
        self.input_representation = str(input_representation)
        self.cross_band_layers = int(cross_band_layers)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.mask_ratio = 0.0
        self.teacher_checkpoint = (
            str(Path(teacher_checkpoint).expanduser().resolve())
            if teacher_checkpoint is not None
            else None
        )
        self.teacher_q_cache_dir = (
            str(Path(teacher_q_cache_dir).expanduser().resolve())
            if teacher_q_cache_dir is not None
            else None
        )
        self.teacher_q_cache_prepared_dir = (
            str(Path(teacher_q_cache_prepared_dir).expanduser().resolve())
            if teacher_q_cache_prepared_dir is not None
            else None
        )
        self.teacher_q_cache_allow_smoke = bool(teacher_q_cache_allow_smoke)
        self._teacher_q_cache_readers: dict[str, Any] = {}
        self._teacher_q_cache_identity: dict[str, str] | None = None
        if self.teacher_q_cache_dir is not None:
            if self.teacher_checkpoint is None:
                raise ValueError("Teacher Q cache requires a validated trained Teacher checkpoint")
            if self.teacher_q_cache_prepared_dir is None:
                raise ValueError("Teacher Q cache requires teacher_q_cache_prepared_dir")
            from ..utils.teacher_q_cache import sha256_file

            data_manifest_path = Path(self.teacher_q_cache_prepared_dir) / "manifest.json"
            if not data_manifest_path.is_file():
                raise FileNotFoundError(
                    f"Teacher Q cache data manifest is missing: {data_manifest_path}"
                )
            self._teacher_q_cache_identity = {
                "teacher_sha256": sha256_file(self.teacher_checkpoint),
                "data_manifest_sha256": sha256_file(data_manifest_path),
            }

        teacher = None
        resolved_metadata = (
            _json_metadata(band_metadata, name="band_metadata")
            if band_metadata is not None
            else None
        )
        resolved_bands = int(num_bands) if num_bands is not None else None
        if self.teacher_checkpoint is not None:
            checkpoint_path = Path(self.teacher_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"frequency Teacher checkpoint does not exist: {checkpoint_path}"
                )
            teacher, checkpoint_metadata = self._load_trained_teacher(checkpoint_path)
            teacher_metadata = _json_metadata(
                getattr(teacher, "band_metadata", checkpoint_metadata),
                name="Teacher band_metadata",
            )
            checkpoint_metadata = _json_metadata(
                checkpoint_metadata, name="frequency_teacher checkpoint metadata"
            )
            if teacher_metadata != checkpoint_metadata.get("band_metadata"):
                raise ValueError(
                    "Teacher checkpoint band metadata disagrees with the loaded model"
                )
            teacher_bands = int(getattr(teacher, "num_bands"))
            if (int(teacher.num_patches), int(teacher.patch_size)) != (
                int(num_patches), int(patch_size)
            ):
                raise ValueError(
                    "Student patch geometry does not match the paired Teacher: "
                    f"Student N/P={num_patches}/{patch_size}, "
                    f"Teacher N/P={teacher.num_patches}/{teacher.patch_size}"
                )
            if resolved_bands is not None and resolved_bands != teacher_bands:
                raise ValueError(
                    f"configured num_bands={resolved_bands} does not match Teacher "
                    f"num_bands={teacher_bands}"
                )
            if resolved_metadata is not None and resolved_metadata != teacher_metadata:
                raise ValueError("configured band_metadata does not match Teacher checkpoint")
            resolved_bands = teacher_bands
            resolved_metadata = teacher_metadata
        if resolved_bands is None:
            resolved_bands = _band_count(resolved_metadata or {})
        if resolved_bands is None or resolved_bands < 1:
            raise ValueError(
                "num_bands is required when constructing a Student without a Teacher checkpoint"
            )
        if resolved_metadata is None:
            # This fallback permits architecture dry-runs only.  A training
            # loss still requires an explicitly loaded trained Teacher.
            resolved_metadata = {
                "schema": "frequency-band-placeholder-v1",
                "num_bands": int(resolved_bands),
                "bands": [
                    {"index": index, "name": f"band_{index}"}
                    for index in range(int(resolved_bands))
                ],
            }
        metadata_bands = _band_count(resolved_metadata)
        if metadata_bands is not None and metadata_bands != int(resolved_bands):
            raise ValueError(
                f"band_metadata describes {metadata_bands} bands, num_bands is {resolved_bands}"
            )

        self.num_bands = int(resolved_bands)
        self.band_metadata = copy.deepcopy(resolved_metadata)
        resolved_input_metadata = None
        selected_divisors = None
        resolved_input_normalization = requested_normalization or "fixed"
        resolved_input_rmsnorm_eps = None
        if self.input_representation == WAVELET_BAND_REPRESENTATION:
            wavelet_metadata = resolved_metadata.get("wavelet_mra")
            if wavelet_metadata is None:
                # An automatically generated placeholder supports dry-runs;
                # a real Teacher checkpoint must explicitly identify its bank.
                if teacher is not None or resolved_metadata.get("schema") != "frequency-band-placeholder-v1":
                    raise ValueError(
                        "wavelet Student requires Teacher wavelet_mra metadata"
                    )
                if self.num_bands != 4:
                    raise ValueError(
                        "wavelet_bands dry-runs require the canonical four-band grouping"
                    )
                decomposer = WaveletBandDecomposer()
                wavelet_metadata = decomposer.metadata
                self.band_metadata["wavelet_mra"] = wavelet_metadata
            else:
                wavelet_metadata = _json_metadata(
                    wavelet_metadata, name="band_metadata.wavelet_mra"
                )

            teacher_scaling = resolved_metadata.get("encoder_input_scaling")
            if teacher_scaling is None:
                scaling_metadata = {
                    "type": "identity",
                    "applied_before": "patch_embedding",
                }
            else:
                scaling_metadata = _json_metadata(
                    teacher_scaling, name="band_metadata.encoder_input_scaling"
                )
            teacher_normalization, teacher_eps, scaling_matrix = (
                _encoder_scaling_contract(
                    scaling_metadata, num_bands=self.num_bands
                )
            )
            if requested_normalization is not None and requested_normalization != teacher_normalization:
                raise ValueError(
                    "configured input_normalization disagrees with Teacher "
                    "encoder_input_scaling metadata"
                )
            resolved_input_normalization = teacher_normalization
            if teacher_normalization == "band_rmsnorm":
                assert teacher_eps is not None
                if requested_rmsnorm_eps is not None and requested_rmsnorm_eps != teacher_eps:
                    raise ValueError(
                        "configured input_rmsnorm_eps disagrees with Teacher metadata"
                    )
                if input_band_channel_divisors is not None:
                    raise ValueError(
                        "input_band_channel_divisors must be null when "
                        "input_normalization='band_rmsnorm'"
                    )
                selected_divisors = None
                resolved_input_rmsnorm_eps = teacher_eps
            else:
                assert scaling_matrix is not None
                expected_divisors = scaling_matrix[:, self.encoder_input_channel]
                selected_divisors = [float(value) for value in expected_divisors.tolist()]
                if input_band_channel_divisors is not None:
                    try:
                        supplied_divisors = torch.as_tensor(
                            input_band_channel_divisors, dtype=torch.float64
                        )
                    except (TypeError, ValueError, RuntimeError) as error:
                        raise ValueError(
                            "input_band_channel_divisors must have shape [K]"
                        ) from error
                    if tuple(supplied_divisors.shape) != (self.num_bands,):
                        raise ValueError(
                            "input_band_channel_divisors must have one divisor per band"
                        )
                    if not torch.equal(supplied_divisors, expected_divisors):
                        raise ValueError(
                            "input_band_channel_divisors do not match the selected "
                            "Teacher modality column"
                        )
                if requested_rmsnorm_eps is not None:
                    raise ValueError(
                        "input_rmsnorm_eps is only valid when "
                        "input_normalization='band_rmsnorm'"
                    )
            resolved_input_metadata = {
                "schema": FREQUENCY_STUDENT_INPUT_V2_SCHEMA,
                "representation": self.input_representation,
                "wavelet_mra": wavelet_metadata,
                "encoder_input_scaling": scaling_metadata,
                "selected_channel_index": self.encoder_input_channel,
                "selected_channel_divisors": selected_divisors,
                "cross_band_layers": self.cross_band_layers,
            }
            if teacher is None and self.band_metadata.get("schema") == "frequency-band-placeholder-v1":
                self.band_metadata["encoder_input_scaling"] = scaling_metadata
                self.band_metadata["num_bands"] = self.num_bands
        resolved_metadata = copy.deepcopy(self.band_metadata)
        self.teacher = teacher
        self.teacher_is_trained_checkpoint = teacher is not None
        if self.teacher is not None:
            self.teacher.requires_grad_(False)
            self.teacher.eval()
        self.student = FrequencyStudentBackbone(
            num_bands=self.num_bands,
            patch_size=int(patch_size),
            num_patches=int(num_patches),
            embed_dim=int(embed_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            drop_rate=float(drop_rate),
            band_metadata=self.band_metadata,
            input_representation=self.input_representation,
            cross_band_layers=self.cross_band_layers,
            input_metadata=resolved_input_metadata,
            input_normalization=resolved_input_normalization,
            input_rmsnorm_eps=resolved_input_rmsnorm_eps or 1e-6,
        )
        self.student.encoder_input_channel = self.encoder_input_channel
        self.student.modality = "ecg" if self.encoder_input_channel == 0 else "ppg"
        self.wavelet_response_cache_enabled = bool(wavelet_response_cache)
        if self.wavelet_response_cache_enabled:
            if self.student.decomposer is None:
                raise ValueError(
                    "wavelet_response_cache requires wavelet_bands input representation"
                )
            enable_cache = getattr(
                self.student.decomposer, "enable_response_fft_cache", None
            )
            if not callable(enable_cache):
                raise RuntimeError(
                    "wavelet response cache implementation is unavailable"
                )
            enable_cache(True)
        self._architecture_metadata = {
            "patch_size": int(patch_size),
            "num_patches": int(num_patches),
            "embed_dim": int(embed_dim),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "drop_rate": float(drop_rate),
        }
        if self.input_representation == WAVELET_BAND_REPRESENTATION:
            self._architecture_metadata.update(
                {
                    "input_representation": self.input_representation,
                    "cross_band_layers": self.cross_band_layers,
                }
            )
        # Lightning resume reconstructs the same selected Teacher scale. The
        # exact full input metadata is separately stored in frequency_student.
        self.hparams.input_representation = self.input_representation
        self.hparams.cross_band_layers = self.cross_band_layers
        self.hparams.input_band_channel_divisors = selected_divisors
        self.hparams.input_normalization = resolved_input_normalization
        self.hparams.input_rmsnorm_eps = resolved_input_rmsnorm_eps
        self.input_metadata = resolved_input_metadata
        self.input_normalization = resolved_input_normalization
        self.input_rmsnorm_eps = resolved_input_rmsnorm_eps

    @staticmethod
    def _load_trained_teacher(checkpoint_path: Path):
        raw_checkpoint = _torch_load(checkpoint_path, map_location="cpu")
        if not isinstance(raw_checkpoint, Mapping):
            raise ValueError("Teacher checkpoint must be a Lightning checkpoint mapping")
        if not isinstance(raw_checkpoint.get("state_dict"), Mapping):
            raise ValueError("Teacher checkpoint has no Lightning state_dict")
        if not isinstance(raw_checkpoint.get("hyper_parameters"), Mapping):
            raise ValueError("Teacher checkpoint has no hyper_parameters")
        if int(raw_checkpoint.get("global_step", 0)) < 1:
            raise ValueError(
                "Teacher checkpoint global_step must be positive; refusing random/untrained targets"
            )
        metadata = raw_checkpoint.get("frequency_teacher")
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("band_metadata"), Mapping
        ):
            raise ValueError(
                "Teacher checkpoint is missing versioned frequency_teacher band metadata"
            )
        from .frequency_fsq import FrequencyFSQTokenizer

        teacher = FrequencyFSQTokenizer.load_from_checkpoint(
            str(checkpoint_path), map_location="cpu"
        )
        if str(metadata.get("schema", "")) != "frequency-teacher-v1":
            raise ValueError(
                "unsupported frequency Teacher checkpoint schema: "
                f"{metadata.get('schema')!r}"
            )
        trained_steps = int(metadata.get("trained_steps", 0))
        if trained_steps < 1 or trained_steps != int(raw_checkpoint["global_step"]):
            raise ValueError(
                "Teacher checkpoint trained_steps must be positive and match "
                "Lightning global_step"
            )
        if not bool(getattr(teacher, "teacher_checkpoint_loaded", False)):
            raise ValueError(
                "loaded Teacher did not attest that its versioned checkpoint was validated"
            )
        return teacher, dict(metadata)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def _paired_patches(self, paired_batch: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(paired_batch) or paired_batch.ndim != 4:
            raise ValueError("paired_batch must be a tensor shaped [B,N,2,P]")
        if paired_batch.shape[2] != 2:
            raise ValueError(
                f"paired_batch must contain ECG and PPG channels, got {paired_batch.shape[2]}"
            )
        if paired_batch.shape[1] != self.student.num_patches:
            raise ValueError(
                f"paired_batch has {paired_batch.shape[1]} patches, expected "
                f"{self.student.num_patches}"
            )
        if paired_batch.shape[3] != self.student.patch_size:
            raise ValueError(
                f"paired_batch patch width is {paired_batch.shape[3]}, expected "
                f"{self.student.patch_size}"
            )
        return paired_batch

    def forward_single(self, single_channel_patches: torch.Tensor) -> torch.Tensor:
        """Public inference path for one channel: ``[B,N,P] -> [B,K,N,128]``."""

        return self.student(single_channel_patches)

    def forward(self, paired_batch: torch.Tensor) -> torch.Tensor:
        """Encode single-channel patches, or select a channel from a training pair."""

        if torch.is_tensor(paired_batch) and paired_batch.ndim == 3:
            return self.forward_single(paired_batch)
        paired_batch = self._paired_patches(paired_batch)
        return self.forward_single(
            paired_batch[:, :, self.encoder_input_channel, :]
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int):
        """Keep cache lookup IDs on CPU while transferring only paired signals.

        Stable IDs are used solely for CPU-side memmap indexing. Moving them to
        CUDA would force a device-to-host synchronization on every cached batch.
        The uncached path keeps Lightning's default recursive transfer intact.
        """

        if self.teacher_q_cache_dir is None:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        paired_batch, sample_ids = batch
        paired_batch = super().transfer_batch_to_device(
            paired_batch, device, dataloader_idx
        )
        if not torch.is_tensor(sample_ids):
            raise TypeError("cached Teacher Q stable sample IDs must be a tensor")
        if sample_ids.device.type != "cpu":
            raise ValueError("cached Teacher Q stable sample IDs must remain on CPU")
        return paired_batch, sample_ids

    def _cached_teacher_q(
        self, sample_ids: torch.Tensor, *, split: str, device: torch.device,
        expected_batch_size: int,
    ) -> torch.Tensor:
        if self.teacher_q_cache_dir is None or self._teacher_q_cache_identity is None:
            raise RuntimeError("Teacher Q cache was not explicitly enabled")
        if not torch.is_tensor(sample_ids):
            raise TypeError("cached Teacher Q requires collated stable sample ID tensors")
        if sample_ids.device.type != "cpu":
            raise ValueError("cached Teacher Q stable sample IDs must remain on CPU")
        if sample_ids.ndim != 1 or sample_ids.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        }:
            raise ValueError(
                "cached Teacher Q requires one integer stable sample ID per input sample"
            )
        if int(sample_ids.numel()) != int(expected_batch_size):
            raise ValueError(
                f"cached Teacher Q received {sample_ids.numel()} IDs for "
                f"a batch of {expected_batch_size} paired samples"
            )
        reader = self._teacher_q_cache_readers.get(split)
        if reader is None:
            from ..utils.teacher_q_cache import TeacherQCacheReader

            reader = TeacherQCacheReader(
                self.teacher_q_cache_dir,
                split=split,
                expected_teacher_sha256=self._teacher_q_cache_identity["teacher_sha256"],
                expected_data_manifest_sha256=self._teacher_q_cache_identity[
                    "data_manifest_sha256"
                ],
                expected_prepared_dir=self.teacher_q_cache_prepared_dir,
                allow_smoke=self.teacher_q_cache_allow_smoke,
            )
            self._teacher_q_cache_readers[split] = reader
        reader.validate_runtime(device=device, batch_size=expected_batch_size)
        # Transfer compact codes first, then decode exactly on-device. The
        # values are multiples of 1/4, so FP32 multiply/subtract is exact.
        codes = reader.get_codes(sample_ids).to(device=device, non_blocking=True)
        with torch.inference_mode(False):
            cached_target = codes.to(dtype=torch.float32).mul_(0.25).sub_(1.0)
        # Decoding under inference_mode(False) ensures this is a normal target
        # tensor even if lookup was called from an outer inference context.
        return cached_target

    def loss(
        self,
        paired_batch: torch.Tensor,
        *,
        cached_target_q: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Align Student predictions to frozen Teacher Q from the full pair."""

        if self.teacher is None or not self.teacher_is_trained_checkpoint:
            raise RuntimeError(
                "Frequency Q loss requires a loaded trained Teacher checkpoint; "
                "random Teacher targets are forbidden"
            )
        paired_batch = self._paired_patches(paired_batch)
        if self.teacher_q_cache_dir is not None:
            if cached_target_q is None:
                raise RuntimeError(
                    "Teacher Q cache is enabled but no ID-bound cached target was supplied; "
                    "refusing live/cache target fallback"
                )
            target_q = cached_target_q
        else:
            if cached_target_q is not None:
                raise RuntimeError(
                    "cached_target_q was supplied while Teacher Q cache is disabled"
                )
            self.teacher.eval()
            with torch.no_grad():
                target_q = self.teacher.encode_q(paired_batch)
        if target_q.ndim != 4:
            raise ValueError(
                "Teacher encode_q must return [B,K,N,128], got "
                f"{tuple(target_q.shape)}"
            )
        expected_shape = (
            paired_batch.shape[0], self.num_bands, paired_batch.shape[1], FREQUENCY_SCALAR_DIM
        )
        if tuple(target_q.shape) != expected_shape:
            raise ValueError(
                f"Teacher Q shape {tuple(target_q.shape)} does not match expected "
                f"{expected_shape}"
            )
        prediction = self.forward(paired_batch)
        per_band = (prediction.float() - target_q.float()).square().mean(dim=(0, 2, 3))
        return {
            "loss": per_band.mean(),
            "per_band_loss": per_band.detach(),
            "prediction": prediction,
            "target_q": target_q,
        }

    def _unpack_training_batch(self, batch: Any, *, split: str):
        if isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError("Student batch is empty")
            paired_batch = batch[0]
            sample_ids = batch[1] if len(batch) > 1 else None
            if self.teacher_q_cache_dir is not None and len(batch) != 2:
                raise ValueError(
                    "Teacher Q cache batches must contain exactly (paired_samples, stable_ids); "
                    "task labels or extra fields are not accepted"
                )
        else:
            paired_batch = batch
            sample_ids = None
        cached_target = None
        if self.teacher_q_cache_dir is not None:
            if sample_ids is None:
                raise RuntimeError(
                    f"Teacher Q cache is enabled for {split}, but the batch has no stable sample IDs"
                )
            cached_target = self._cached_teacher_q(
                sample_ids, split=split, device=paired_batch.device,
                expected_batch_size=int(paired_batch.shape[0]),
            )
        return paired_batch, cached_target

    def training_step(self, batch: Any, batch_idx: int):
        paired_batch, cached_target = self._unpack_training_batch(batch, split="train")
        output = self.loss(paired_batch, cached_target_q=cached_target)
        self.log("train/loss", output["loss"], prog_bar=True, on_step=True, on_epoch=True)
        interval = int(getattr(self, "train_diagnostics_interval", 1))
        if batch_idx % interval == 0:
            for band_index, band_loss in enumerate(output["per_band_loss"]):
                self.log(
                    f"train/band_{band_index}_loss", band_loss,
                    on_step=False, on_epoch=True,
                )
        return output["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        paired_batch, cached_target = self._unpack_training_batch(batch, split="val")
        output = self.loss(paired_batch, cached_target_q=cached_target)
        self.log(
            "val/loss", output["loss"], prog_bar=True, on_step=False, on_epoch=True,
            sync_dist=True,
        )
        for band_index, band_loss in enumerate(output["per_band_loss"]):
            self.log(
                f"val/band_{band_index}_loss", band_loss,
                on_step=False, on_epoch=True, sync_dist=True,
            )
        return output["loss"]

    def configure_optimizers(self):
        options = getattr(self, "_training_acceleration_options", {})
        optimizer = build_training_adamw(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=bool(options.get("fused_adamw", False)),
        )
        # Keep the live report linked to the optimizer's mutable backend
        # record. PyTorch restores saved param-group options during resume;
        # the optimizer's post-load hook then reapplies the requested runtime
        # backend and updates this same record.
        self._training_acceleration_optimizer_report = optimizer.training_acceleration
        report = getattr(self, "_training_acceleration_report", None)
        if isinstance(report, dict):
            report["effective"]["fused_adamw"] = optimizer.training_acceleration
        return optimizer

    def _checkpoint_metadata(self) -> dict[str, Any]:
        schema = (
            FREQUENCY_STUDENT_V2_SCHEMA
            if self.input_representation == WAVELET_BAND_REPRESENTATION
            else FREQUENCY_STUDENT_V1_SCHEMA
        )
        metadata = {
            "schema": schema,
            "target_type": FREQUENCY_Q_TARGET_TYPE,
            "quantizer_family": FREQUENCY_QUANTIZER_FAMILY,
            "full_target_view": "full_pair",
            "encoder_input_channel": self.encoder_input_channel,
            "modality": "ecg" if self.encoder_input_channel == 0 else "ppg",
            "num_bands": self.num_bands,
            "scalar_dim_per_band": FREQUENCY_SCALAR_DIM,
            "levels_per_scalar": 8,
            "levels": list(FREQUENCY_LEVELS),
            "band_metadata": copy.deepcopy(self.band_metadata),
            "architecture": dict(self._architecture_metadata),
        }
        if self.input_representation == WAVELET_BAND_REPRESENTATION:
            assert self.input_metadata is not None
            metadata.update(
                {
                    "input_representation": self.input_representation,
                    "cross_band_layers": self.cross_band_layers,
                    "selected_input_divisors": copy.deepcopy(
                        self.input_metadata["selected_channel_divisors"]
                    ),
                    "input_metadata": copy.deepcopy(self.input_metadata),
                }
            )
        return metadata

    def _teacher_q_cache_provenance(self) -> dict[str, Any]:
        if self.teacher_q_cache_dir is None or self._teacher_q_cache_identity is None:
            raise RuntimeError("Teacher Q cache provenance requested while cache is disabled")
        from ..utils.teacher_q_cache import CACHE_SCHEMA, _read_json, sha256_file

        root = Path(self.teacher_q_cache_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Teacher Q cache manifest is missing: {manifest_path}")
        cache_manifest = _read_json(manifest_path)
        if (
            cache_manifest.get("schema") != CACHE_SCHEMA
            or cache_manifest.get("status") != "complete"
        ):
            raise ValueError("Student checkpoint cannot attest to a partial/unsupported Teacher Q cache")
        if cache_manifest.get("mode") == "smoke" and not self.teacher_q_cache_allow_smoke:
            raise ValueError("Student checkpoint cannot use a smoke cache without explicit opt-in")
        if cache_manifest.get("teacher_checkpoint_sha256") != self._teacher_q_cache_identity[
            "teacher_sha256"
        ]:
            raise ValueError("Teacher Q cache Teacher SHA changed before checkpoint save")
        if cache_manifest.get("dataset_manifest_sha256") != self._teacher_q_cache_identity[
            "data_manifest_sha256"
        ]:
            raise ValueError("Teacher Q cache data manifest SHA changed before checkpoint save")
        return {
            "schema": "frequency-teacher-q-cache-training-provenance-v1",
            "cache_schema": CACHE_SCHEMA,
            "cache_manifest_sha256": sha256_file(manifest_path),
            "cache_mode": cache_manifest.get("mode"),
            "cache_path": str(root),
            "teacher_checkpoint_sha256": cache_manifest.get("teacher_checkpoint_sha256"),
            "dataset_manifest_sha256": cache_manifest.get("dataset_manifest_sha256"),
            "encoding": cache_manifest.get("encoding"),
            "split_sample_counts": {
                split: int(value["covered_samples"])
                for split, value in cache_manifest.get("splits", {}).items()
            },
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["frequency_student"] = self._checkpoint_metadata()
        checkpoint["training_acceleration"] = copy.deepcopy(
            getattr(
                self,
                "_training_acceleration_report",
                {
                    "schema": "training-acceleration-v1",
                    "requested": {
                        "wavelet_response_cache": False,
                        "fused_adamw": False,
                        "compile": False,
                        "compile_mode": "default",
                        "train_diagnostics_interval": 1,
                    },
                    "effective": {
                        "wavelet_response_cache": {"enabled": False, "decomposers": []},
                        "fused_adamw": {
                            "requested": False,
                            "enabled": False,
                            "backend": "torch.optim.AdamW",
                            "fallback_reason": None,
                        },
                        "compile": {
                            "enabled": False,
                            "mode": "default",
                            "compiled_modules": [],
                        },
                        "train_diagnostics_interval": 1,
                    },
                },
            )
        )
        if self.teacher_q_cache_dir is not None:
            checkpoint["teacher_q_cache_training"] = self._teacher_q_cache_provenance()

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        metadata = checkpoint.get("frequency_student")
        if not isinstance(metadata, Mapping):
            raise ValueError("Student checkpoint is missing frequency_student metadata")
        expected = self._checkpoint_metadata()
        if set(metadata) != set(expected):
            raise ValueError(
                "Student checkpoint metadata keys differ from the configured schema"
            )
        keys = [
            "schema", "target_type", "quantizer_family", "full_target_view",
            "encoder_input_channel", "modality", "num_bands",
            "scalar_dim_per_band", "levels_per_scalar", "levels",
            "band_metadata", "architecture",
        ]
        if expected["schema"] == FREQUENCY_STUDENT_V2_SCHEMA:
            keys.extend(
                [
                    "input_representation", "cross_band_layers",
                    "selected_input_divisors", "input_metadata",
                ]
            )
        for key in keys:
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    f"Student checkpoint {key} mismatch: "
                    f"{metadata.get(key)!r} != {expected[key]!r}"
                )
        if self.teacher_q_cache_dir is not None:
            provenance = checkpoint.get("teacher_q_cache_training")
            if not isinstance(provenance, Mapping):
                raise ValueError(
                    "cache-enabled Student resume requires a checkpoint that records "
                    "Teacher Q cache provenance"
                )
            expected_provenance = self._teacher_q_cache_provenance()
            for key in (
                "schema", "cache_schema", "cache_manifest_sha256", "cache_mode",
                "teacher_checkpoint_sha256", "dataset_manifest_sha256",
            ):
                if provenance.get(key) != expected_provenance.get(key):
                    raise ValueError(
                        "Student checkpoint Teacher Q cache provenance mismatch "
                        f"for {key}"
                    )
        if expected["schema"] == FREQUENCY_STUDENT_V2_SCHEMA:
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, Mapping):
                raise ValueError("v2 Student checkpoint has no state_dict")
            if self.input_normalization == "fixed":
                _checkpoint_tensor_matches(
                    state_dict,
                    "student.input_band_channel_divisors_tensor",
                    self.student.input_band_channel_divisors_tensor,
                    description="input divisor",
                )
                if "student.input_norm.weight" in state_dict:
                    raise ValueError(
                        "fixed-scaling Student checkpoint contains unexpected RMSNorm gains"
                    )
            else:
                if "student.input_band_channel_divisors_tensor" in state_dict:
                    raise ValueError(
                        "band_rmsnorm Student checkpoint contains stale fixed divisors"
                    )
                _checkpoint_learned_tensor_matches(
                    state_dict,
                    "student.input_norm.weight",
                    self.student.input_norm.weight,
                    description="RMSNorm gain",
                )
            assert self.student.decomposer is not None
            _checkpoint_tensor_matches(
                state_dict,
                "student.decomposer.kernel_offsets",
                self.student.decomposer.kernel_offsets,
                description="wavelet kernel",
            )




def load_frequency_backbone(
    checkpoint_path: str | Path,
    *,
    expected_modality: Optional[str] = None,
    encoder_input_channel: Optional[int] = None,
    expected_band_metadata: Optional[Mapping[str, Any]] = None,
    map_location: Any = "cpu",
) -> FrequencyStudentBackbone:
    """Load the complete Student backbone without loading/locating its Teacher."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Student checkpoint does not exist: {path}")
    checkpoint = _torch_load(path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Student checkpoint must be a Lightning checkpoint mapping")
    metadata = checkpoint.get("frequency_student")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint lacks frequency_student metadata or state_dict")
    schema = metadata.get("schema")
    if schema not in {FREQUENCY_STUDENT_V1_SCHEMA, FREQUENCY_STUDENT_V2_SCHEMA}:
        raise ValueError(f"unsupported Student checkpoint schema: {metadata.get('schema')!r}")
    if metadata.get("target_type") != FREQUENCY_Q_TARGET_TYPE:
        raise ValueError("Student checkpoint target_type is not frequency_q")
    if metadata.get("quantizer_family") != FREQUENCY_QUANTIZER_FAMILY:
        raise ValueError("Student checkpoint does not use frequency_fsq")
    if metadata.get("full_target_view") != "full_pair":
        raise ValueError("Student checkpoint does not target the full ECG+PPG Teacher view")
    if int(metadata.get("scalar_dim_per_band", -1)) != FREQUENCY_SCALAR_DIM:
        raise ValueError("Student checkpoint must preserve 128 scalar coordinates per band")
    if int(metadata.get("levels_per_scalar", -1)) != 8:
        raise ValueError("Student checkpoint must preserve eight levels per scalar")
    if tuple(metadata.get("levels", ())) != FREQUENCY_LEVELS:
        raise ValueError("Student checkpoint has incompatible per-coordinate FSQ levels")

    saved_channel = int(metadata.get("encoder_input_channel", -1))
    modality = str(metadata.get("modality", "")).lower()
    if saved_channel not in (0, 1) or modality != ("ecg" if saved_channel == 0 else "ppg"):
        raise ValueError("Student checkpoint modality and input channel metadata disagree")
    if encoder_input_channel is not None and int(encoder_input_channel) != saved_channel:
        raise ValueError(
            f"checkpoint input channel is {saved_channel}, requested {encoder_input_channel}"
        )
    if expected_modality is not None and str(expected_modality).lower() != modality:
        raise ValueError(
            f"checkpoint modality is {modality!r}, expected {expected_modality!r}"
        )

    band_metadata = metadata.get("band_metadata")
    if not isinstance(band_metadata, Mapping):
        raise ValueError("Student checkpoint has no band metadata")
    num_bands = int(metadata.get("num_bands", -1))
    if num_bands < 1 or _band_count(band_metadata) not in (None, num_bands):
        raise ValueError("Student band metadata and num_bands are inconsistent")
    normalized_band_metadata = _json_metadata(
        band_metadata, name="Student checkpoint band_metadata"
    )
    if expected_band_metadata is not None:
        expected_bands = _json_metadata(
            expected_band_metadata, name="expected_band_metadata"
        )
        if expected_bands != normalized_band_metadata:
            raise ValueError("Student checkpoint band definitions do not match expectation")
    architecture = metadata.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("Student checkpoint has no architecture metadata")
    input_representation = RAW_PATCH_REPRESENTATION
    cross_band_layers = 1
    input_metadata = None
    input_normalization = "fixed"
    input_rmsnorm_eps = 1e-6
    if schema == FREQUENCY_STUDENT_V1_SCHEMA:
        if architecture.get("input_representation", RAW_PATCH_REPRESENTATION) != RAW_PATCH_REPRESENTATION:
            raise ValueError("v1 Student checkpoint cannot use a wavelet input architecture")
        if any(
            key in metadata
            for key in (
                "input_representation", "cross_band_layers",
                "selected_input_divisors", "input_metadata",
            )
        ):
            raise ValueError("v1 Student checkpoint contains unexpected v2 input metadata")
    else:
        input_representation = str(metadata.get("input_representation", ""))
        if input_representation != WAVELET_BAND_REPRESENTATION:
            raise ValueError("v2 Student checkpoint must declare wavelet_bands input")
        cross_band_layers = int(metadata.get("cross_band_layers", -1))
        if cross_band_layers < 0:
            raise ValueError("v2 Student checkpoint has invalid cross_band_layers")
        if architecture.get("input_representation") != input_representation:
            raise ValueError("Student architecture and input representation metadata disagree")
        if int(architecture.get("cross_band_layers", -1)) != cross_band_layers:
            raise ValueError("Student architecture and cross_band_layers metadata disagree")
        input_metadata = _json_metadata(
            metadata.get("input_metadata"), name="Student checkpoint input_metadata"
        )
        if input_metadata.get("schema") != FREQUENCY_STUDENT_INPUT_V2_SCHEMA:
            raise ValueError("v2 Student checkpoint has an unsupported input_metadata schema")
        if input_metadata.get("representation") != input_representation:
            raise ValueError("Student input_metadata representation mismatch")
        if int(input_metadata.get("cross_band_layers", -1)) != cross_band_layers:
            raise ValueError("Student input_metadata cross_band_layers mismatch")
        if int(input_metadata.get("selected_channel_index", -1)) != saved_channel:
            raise ValueError("Student input_metadata selected channel mismatch")
        saved_selected_divisors = metadata.get("selected_input_divisors")
        input_selected_divisors = input_metadata.get("selected_channel_divisors")
        if saved_selected_divisors != input_selected_divisors:
            raise ValueError("Student selected divisor metadata fields disagree")
        expected_wavelet = normalized_band_metadata.get("wavelet_mra")
        if not isinstance(expected_wavelet, Mapping):
            raise ValueError("v2 Student band metadata is missing its self-contained wavelet bank")
        if input_metadata.get("wavelet_mra") != expected_wavelet:
            raise ValueError("Student input wavelet metadata disagrees with band metadata")
        expected_scaling = normalized_band_metadata.get("encoder_input_scaling")
        if expected_scaling is None:
            expected_scaling = {
                "type": "identity",
                "applied_before": "patch_embedding",
            }
        if input_metadata.get("encoder_input_scaling") != expected_scaling:
            raise ValueError("Student input scaling metadata disagrees with band metadata")
        input_normalization, scaling_eps, scaling_matrix = _encoder_scaling_contract(
            expected_scaling, num_bands=num_bands
        )
        if input_normalization == "band_rmsnorm":
            if input_selected_divisors is not None:
                raise ValueError(
                    "band_rmsnorm Student checkpoint has stale selected fixed divisors"
                )
            input_rmsnorm_eps = scaling_eps
            assert input_rmsnorm_eps is not None
        else:
            assert scaling_matrix is not None
            selected_expected = scaling_matrix[:, saved_channel].tolist()
            if input_selected_divisors != selected_expected:
                raise ValueError(
                    "Student selected divisors do not match the Teacher modality column"
                )
    backbone = FrequencyStudentBackbone(
        num_bands=num_bands,
        patch_size=int(architecture["patch_size"]),
        num_patches=int(architecture["num_patches"]),
        embed_dim=int(architecture["embed_dim"]),
        num_layers=int(architecture["num_layers"]),
        num_heads=int(architecture["num_heads"]),
        drop_rate=float(architecture["drop_rate"]),
        band_metadata=normalized_band_metadata,
        input_representation=input_representation,
        cross_band_layers=cross_band_layers,
        input_metadata=input_metadata,
        input_normalization=input_normalization,
        input_rmsnorm_eps=input_rmsnorm_eps or 1e-6,
    )
    backbone.encoder_input_channel = saved_channel
    backbone.modality = modality
    prefix = "student."
    student_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not student_state:
        raise ValueError("Student checkpoint has no complete `student.*` state")
    if schema == FREQUENCY_STUDENT_V2_SCHEMA:
        if input_normalization == "fixed":
            _checkpoint_tensor_matches(
                student_state,
                "input_band_channel_divisors_tensor",
                backbone.input_band_channel_divisors_tensor,
                description="input divisor",
            )
        else:
            if "input_band_channel_divisors_tensor" in student_state:
                raise ValueError(
                    "band_rmsnorm Student checkpoint contains stale fixed divisors"
                )
            _checkpoint_learned_tensor_matches(
                student_state,
                "input_norm.weight",
                backbone.input_norm.weight,
                description="RMSNorm gain",
            )
        assert backbone.decomposer is not None
        _checkpoint_tensor_matches(
            student_state,
            "decomposer.kernel_offsets",
            backbone.decomposer.kernel_offsets,
            description="wavelet kernel",
        )
    backbone.load_state_dict(student_state, strict=True)
    backbone.eval()
    return backbone


__all__ = [
    "FREQUENCY_STUDENT_SCHEMA",
    "FREQUENCY_STUDENT_V1_SCHEMA",
    "FREQUENCY_STUDENT_V2_SCHEMA",
    "FREQUENCY_STUDENT_INPUT_V2_SCHEMA",
    "FREQUENCY_SCALAR_DIM",
    "FREQUENCY_LEVELS",
    "FrequencyStudentBackbone",
    "FrequencyFSQStudent",
    "load_frequency_backbone",
]
