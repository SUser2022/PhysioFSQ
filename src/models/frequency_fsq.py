"""Frequency-separated, jointly encoded ECG/PPG FSQ Teacher."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Sequence

import lightning as L
import torch
import torch.nn as nn

from .input_normalization import BandChannelRMSNorm
from .quantization import FSQ
from .frequency_components import Decoder, JointEncoder, PositionalEncoding, TemporalEncoder
from .wavelet_bands import WaveletBandDecomposer
from ..utils.training_acceleration import build_training_adamw


class FrequencyFSQTokenizer(L.LightningModule):
    """Teacher that learns a separate 128-digit FSQ representation per band.

    Public waveform batches use the repository's patch layout
    ``[B,N,2,P]``. ECG and PPG are joint inside every patch. The complete
    paired window is reconstructed before wavelet decomposition, then a
    shared Transformer processes each band as an independent batch stream.
    """

    CHECKPOINT_SCHEMA = "frequency-teacher-v1"
    MODEL_VERSION = "joint_paired_frequency_fsq_teacher_v1"

    def __init__(
        self,
        num_patches: int = 50,
        patch_size: int = 25,
        max_seq_length: int = 1250,
        sample_rate: float = 125.0,
        wavelet: str = "sym4",
        wavelet_level: int = 5,
        band_groups: Sequence[Sequence[str]] | None = None,
        embed_dim: int = 128,
        num_encoder_heads: int = 8,
        num_encoder_layers: int = 10,
        num_decoder_heads: int = 8,
        num_decoder_layers: int = 4,
        drop_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        total_epochs: int = 400,
        min_lr: float = 1e-6,
        lr_scheduler_type: str = "cosine",
        channel_weights: Sequence[float] = (1.0, 1.0),
        band_weights: Sequence[float] | None = None,
        band_scales: Sequence[float] | None = None,
        band_channel_scales: Sequence[Sequence[float]] | None = None,
        input_band_channel_scales: Sequence[Sequence[float]] | None = None,
        input_normalization: str = "fixed",
        input_rmsnorm_eps: float = 1e-6,
        decoder_type: str = "transformer",
        band_metadata: Mapping | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_patches = int(num_patches)
        self.patch_size = int(patch_size)
        self.max_seq_length = int(max_seq_length)
        self.embed_dim = int(embed_dim)
        self.num_encoder_heads = int(num_encoder_heads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_heads = int(num_decoder_heads)
        self.num_decoder_layers = int(num_decoder_layers)
        self.drop_rate = float(drop_rate)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.total_epochs = int(total_epochs)
        self.min_lr = float(min_lr)
        self.lr_scheduler_type = str(lr_scheduler_type).strip().lower().replace("-", "_")
        self.decoder_type = str(decoder_type).strip().lower()
        if self.decoder_type not in {"transformer", "patch_mlp"}:
            raise ValueError(
                "decoder_type must be 'transformer' or 'patch_mlp', got "
                f"{decoder_type!r}"
            )

        if self.num_patches < 1 or self.patch_size < 2:
            raise ValueError("num_patches must be positive and patch_size must be >=2")
        if self.max_seq_length < self.num_patches:
            raise ValueError(
                f"max_seq_length={self.max_seq_length} must cover {self.num_patches} patches"
            )
        if self.embed_dim < 1 or self.embed_dim % self.num_encoder_heads:
            raise ValueError("embed_dim must be positive and divisible by num_encoder_heads")
        if self.embed_dim % self.num_decoder_heads:
            raise ValueError("embed_dim must be divisible by num_decoder_heads")
        if self.num_encoder_layers < 1 or self.num_decoder_layers < 1:
            raise ValueError("encoder and decoder must each have at least one layer")
        if not 0.0 <= self.drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0,1), got {drop_rate}")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.total_epochs < 1 or self.min_lr < 0:
            raise ValueError("total_epochs must be positive and min_lr non-negative")
        if self.lr_scheduler_type not in {"cosine", "reduce_on_plateau", "none"}:
            raise ValueError(
                "lr_scheduler_type must be cosine, reduce_on_plateau, or none; got "
                f"{lr_scheduler_type!r}"
            )

        self.decomposer = WaveletBandDecomposer(
            sample_rate=sample_rate,
            wavelet=wavelet,
            level=wavelet_level,
            band_groups=band_groups,
        )
        self.num_bands = self.decomposer.num_bands
        if not isinstance(input_normalization, str) or input_normalization not in {
            "fixed",
            "band_rmsnorm",
        }:
            raise ValueError(
                "input_normalization must be exactly 'fixed' or 'band_rmsnorm', "
                f"got {input_normalization!r}"
            )
        if isinstance(input_rmsnorm_eps, bool):
            raise ValueError("input_rmsnorm_eps must be finite and positive")
        try:
            input_rmsnorm_eps = float(input_rmsnorm_eps)
        except (TypeError, ValueError) as exc:
            raise ValueError("input_rmsnorm_eps must be finite and positive") from exc
        if not math.isfinite(input_rmsnorm_eps) or input_rmsnorm_eps <= 0:
            raise ValueError("input_rmsnorm_eps must be finite and positive")
        self.input_normalization = input_normalization
        self.input_rmsnorm_eps = input_rmsnorm_eps
        input_band_channel_scales_tuple = self._validate_band_channel_scales(
            input_band_channel_scales,
            expected_bands=self.num_bands,
            name="input_band_channel_scales",
        )
        if (
            self.input_normalization == "band_rmsnorm"
            and input_band_channel_scales_tuple is not None
        ):
            raise ValueError(
                "input_band_channel_scales cannot be set when "
                "input_normalization='band_rmsnorm'"
            )
        resolved_band_metadata = {
            "schema": self.CHECKPOINT_SCHEMA,
            "model_version": self.MODEL_VERSION,
            "input_shape": ["B", self.num_patches, 2, self.patch_size],
            "window_samples": self.num_patches * self.patch_size,
            "channels": ["ECG", "PPG"],
            "wavelet_mra": self.decomposer.metadata,
            "num_bands": self.num_bands,
            "num_patches": self.num_patches,
            "patch_size": self.patch_size,
            "encoder_dim": self.embed_dim,
            "position_length": self.max_seq_length,
            "encoder_heads": self.num_encoder_heads,
            "encoder_layers": self.num_encoder_layers,
            "decoder_heads": self.num_decoder_heads,
            "decoder_layers": self.num_decoder_layers,
            "fsq": {
                "coordinates_per_band": 128,
                "levels_per_coordinate": 8,
                "num_codebooks_per_band": 1,
                "packed_codebook_id": False,
                "q_shape": ["B", self.num_bands, self.num_patches, 128],
                "indices_shape": ["B", self.num_bands, self.num_patches, 128],
            },
            "decoder_target": "both_channels_of_the_corresponding_mra_band",
        }
        # Encoder input divisors are distinct from reconstruction-loss scales.
        # They are recorded only when explicitly enabled and are not assumed
        # to be RMS-derived for every band/channel.
        if input_band_channel_scales_tuple is not None:
            resolved_band_metadata["encoder_input_scaling"] = {
                "type": "fixed_band_channel_divisors",
                "applied_before": "patch_embedding",
                "divisors": [list(row) for row in input_band_channel_scales_tuple],
            }
        elif self.input_normalization == "band_rmsnorm":
            resolved_band_metadata["encoder_input_scaling"] = {
                "type": "per_band_channel_rmsnorm",
                "applied_before": "patch_embedding",
                "reduction": "full_window_time",
                "eps": self.input_rmsnorm_eps,
                "gain_shape": [self.num_bands, 2],
                "gain_init": 1.0,
            }
        # Preserve the exact legacy metadata contract for the default decoder.
        # Alternate decoders affect only reconstruction, but are recorded here
        # so downstream Student checkpoints can attest which Teacher produced Q.
        if self.decoder_type != "transformer":
            resolved_band_metadata["decoder"] = {
                "type": "patch_mlp",
                "hidden_dim": 256,
                "outputs_per_patch": 2 * self.patch_size,
                "position_conditioning": False,
                "band_embedding": False,
            }
        if band_metadata is not None:
            supplied_json = json.dumps(
                dict(band_metadata), sort_keys=True, separators=(",", ":")
            )
            resolved_json = json.dumps(
                resolved_band_metadata, sort_keys=True, separators=(",", ":")
            )
            if supplied_json != resolved_json:
                raise ValueError(
                    "frequency Teacher constructor band_metadata mismatch; "
                    "checkpoint architecture differs from this model"
                )
        self.band_metadata = resolved_band_metadata
        # Persist the resolved metadata as well as constructor arguments. This
        # catches a changed default grouping or a changed effective boundary
        # guard when a Lightning checkpoint is loaded.
        self.hparams.band_metadata = self.band_metadata
        self.hparams.band_groups = [list(group) for group in self.decomposer.band_groups]
        self.hparams.wavelet = self.decomposer.wavelet
        self.hparams.wavelet_level = self.decomposer.level
        self.hparams.sample_rate = self.decomposer.sample_rate

        self.temporal_encoder = TemporalEncoder(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_patches=self.num_patches,
            used_channel=[0, 1],
        )
        self.input_norm = (
            BandChannelRMSNorm(
                num_bands=self.num_bands,
                num_channels=2,
                eps=self.input_rmsnorm_eps,
            )
            if self.input_normalization == "band_rmsnorm"
            else nn.Identity()
        )
        self.position = PositionalEncoding(self.embed_dim, self.max_seq_length)
        self.band_embedding = nn.Embedding(self.num_bands, self.embed_dim)
        self.encoder = JointEncoder(
            self.embed_dim,
            nhead=self.num_encoder_heads,
            num_layers=self.num_encoder_layers,
            drop_rate=self.drop_rate,
        )
        # Each FSQ branch has its own learned 128->128 map and scalar grid.
        # FSQ indices remain factorized over coordinates; no 8**128 packed
        # vocabulary is ever constructed.
        self.band_projections = nn.ModuleList(
            [nn.Linear(self.embed_dim, 128) for _ in range(self.num_bands)]
        )
        self.band_quantizers = nn.ModuleList(
            [
                FSQ(
                    levels=[8] * 128,
                    dim=128,
                    num_codebooks=1,
                    keep_num_codebooks_dim=False,
                    return_indices=True,
                    force_quantization_f32=True,
                )
                for _ in range(self.num_bands)
            ]
        )

        if self.decoder_type == "transformer":
            self.decoder_position = PositionalEncoding(self.embed_dim, self.max_seq_length)
            self.decoder_band_embedding = nn.Embedding(self.num_bands, self.embed_dim)
            self.decoder_q_projection = (
                nn.Identity()
                if self.embed_dim == 128
                else nn.Linear(128, self.embed_dim)
            )
            self.decoder = Decoder(
                embed_dim=self.embed_dim,
                num_channels=2,
                patch_size=self.patch_size,
                nhead=self.num_decoder_heads,
                num_layers=self.num_decoder_layers,
                drop_rate=self.drop_rate,
            )
        else:
            # Each branch predicts only its corresponding two-channel patch
            # directly from Q. There is no position or band side channel.
            self.decoder_patch_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(128, 256),
                        nn.GELU(),
                        nn.Linear(256, 2 * self.patch_size),
                    )
                    for _ in range(self.num_bands)
                ]
            )

        channel_weights_tuple = self._validate_weights(
            channel_weights, expected=2, name="channel_weights", allow_zero=True
        )
        band_weights_tuple = self._validate_weights(
            [1.0] * self.num_bands if band_weights is None else band_weights,
            expected=self.num_bands,
            name="band_weights",
            allow_zero=True,
        )
        band_scales_tuple = self._validate_weights(
            [1.0] * self.num_bands if band_scales is None else band_scales,
            expected=self.num_bands,
            name="band_scales",
            allow_zero=False,
        )
        band_channel_scales_tuple = self._validate_band_channel_scales(
            band_channel_scales, expected_bands=self.num_bands
        )
        if sum(channel_weights_tuple) <= 0 or sum(band_weights_tuple) <= 0:
            raise ValueError("at least one channel weight and one band weight must be positive")
        self.register_buffer(
            "channel_weights_tensor", torch.tensor(channel_weights_tuple, dtype=torch.float32)
        )
        self.register_buffer(
            "band_weights_tensor", torch.tensor(band_weights_tuple, dtype=torch.float32)
        )
        self.register_buffer(
            "band_scales_tensor", torch.tensor(band_scales_tuple, dtype=torch.float32)
        )
        # A missing value deliberately adds no persistent state-dict key, so
        # historical checkpoints load strictly with the original state layout.
        self.register_buffer(
            "band_channel_scales_tensor",
            None
            if band_channel_scales_tuple is None
            else torch.tensor(band_channel_scales_tuple, dtype=torch.float32),
        )
        # A None buffer adds no state_dict key, so legacy checkpoints remain
        # strictly loadable with the default encoder-input behavior.
        self.register_buffer(
            "input_band_channel_scales_tensor",
            None
            if input_band_channel_scales_tuple is None
            else torch.tensor(input_band_channel_scales_tuple, dtype=torch.float32),
        )
        self.hparams.channel_weights = list(channel_weights_tuple)
        self.hparams.band_weights = list(band_weights_tuple)
        self.hparams.band_scales = list(band_scales_tuple)
        self.hparams.band_channel_scales = (
            None
            if band_channel_scales_tuple is None
            else [list(row) for row in band_channel_scales_tuple]
        )
        self.hparams.input_band_channel_scales = (
            None
            if input_band_channel_scales_tuple is None
            else [list(row) for row in input_band_channel_scales_tuple]
        )
        self.hparams.input_normalization = self.input_normalization
        self.hparams.input_rmsnorm_eps = self.input_rmsnorm_eps
        self.hparams.decoder_type = self.decoder_type
        self.teacher_checkpoint_loaded = False

    @staticmethod
    def _validate_weights(values, *, expected: int, name: str, allow_zero: bool):
        try:
            result = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain {expected} finite values") from exc
        if len(result) != expected:
            raise ValueError(f"{name} must contain {expected} values, got {len(result)}")
        if any(not math.isfinite(value) for value in result):
            raise ValueError(f"{name} must contain only finite values")
        if any(value < 0 if allow_zero else value <= 0 for value in result):
            condition = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} values must be {condition}")
        return result

    @staticmethod
    def _validate_band_channel_scales(
        values, *, expected_bands: int, name: str = "band_channel_scales"
    ):
        if values is None:
            return None
        try:
            scales = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"{name} must have shape [{expected_bands},2] with finite positive values"
            ) from exc
        if tuple(scales.shape) != (expected_bands, 2):
            raise ValueError(
                f"{name} must have shape "
                f"[{expected_bands},2], got {tuple(scales.shape)}"
            )
        if not bool(torch.isfinite(scales).all()):
            raise ValueError(f"{name} must contain only finite positive values")
        if not bool((scales > 0).all()):
            raise ValueError(f"{name} values must be positive")
        return tuple(tuple(float(value) for value in row) for row in scales.tolist())

    def _paired_full_window(self, batch) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError("paired Teacher batch cannot be empty")
            batch = batch[0]
        if isinstance(batch, Mapping):
            for key in ("signal", "x", "waveform", "input"):
                if key in batch:
                    batch = batch[key]
                    break
            else:
                raise ValueError(
                    "mapping batch must contain one of signal/x/waveform/input"
                )
        if not torch.is_tensor(batch):
            raise TypeError(f"Teacher input must be a tensor, got {type(batch).__name__}")
        if batch.ndim != 4:
            raise ValueError(
                "Teacher accepts paired patched input [B,N,2,P]; "
                f"got {tuple(batch.shape)}"
            )
        if batch.shape[1] != self.num_patches or batch.shape[2] != 2 or batch.shape[3] != self.patch_size:
            raise ValueError(
                "Teacher requires [B,N,2,P] with configured N/P; got "
                f"{tuple(batch.shape)}, expected [B,{self.num_patches},2,{self.patch_size}]"
            )
        if not batch.is_floating_point():
            raise TypeError(f"Teacher input must be floating point, got {batch.dtype}")
        return batch.permute(0, 2, 1, 3).reshape(
            batch.shape[0], 2, self.num_patches * self.patch_size
        )

    def _encode_decomposed(self, band_signals: torch.Tensor):
        batch, bands, channels, length = band_signals.shape
        expected_length = self.num_patches * self.patch_size
        if channels != 2 or length != expected_length or bands != self.num_bands:
            raise ValueError(
                "decomposed Teacher input must have shape "
                f"[B,{self.num_bands},2,{expected_length}], got {tuple(band_signals.shape)}"
            )
        encoder_signals = band_signals
        if self.input_normalization == "band_rmsnorm":
            encoder_signals = self.input_norm(encoder_signals)
        elif self.input_band_channel_scales_tensor is not None:
            divisors = self.input_band_channel_scales_tensor.to(
                device=band_signals.device, dtype=band_signals.dtype
            )
            encoder_signals = band_signals / divisors[None, :, :, None]
        patches = (
            encoder_signals.reshape(batch, bands, 2, self.num_patches, self.patch_size)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch * bands, self.num_patches, 2, self.patch_size)
        )
        tokens = self.temporal_encoder(patches)
        band_ids = torch.arange(bands, device=band_signals.device).repeat(batch)
        tokens = self.position(tokens) + self.band_embedding(band_ids).unsqueeze(1)
        hidden = self.encoder(tokens)
        hidden = hidden.reshape(batch, bands, self.num_patches, self.embed_dim)

        q_by_band = []
        indices_by_band = []
        for band_index, (projection, quantizer) in enumerate(
            zip(self.band_projections, self.band_quantizers)
        ):
            projected = projection(hidden[:, band_index])
            quantized, labels = quantizer(projected)
            q_by_band.append(quantized)
            # FSQ returns a factorized C1 label axis: [B,N,1,128].
            indices_by_band.append(labels.squeeze(-2))
        q = torch.stack(q_by_band, dim=1)
        labels = torch.stack(indices_by_band, dim=1)
        return q, labels

    @torch.no_grad()
    def encode_q(self, paired_batch: torch.Tensor) -> torch.Tensor:
        """Return frozen-Teacher targets ``[B,K,N,128]`` for a paired batch."""

        full_window = self._paired_full_window(paired_batch)
        bands = self.decomposer(full_window)
        q, _ = self._encode_decomposed(bands)
        return q

    def decode_q(self, q: torch.Tensor) -> torch.Tensor:
        """Decode Q alone into paired band waveforms ``[B,K,2,L]``.

        This public path intentionally remains differentiable and is also used
        by :meth:`forward`, so Q ablation and permutation checks exercise the
        exact decoder used during training.
        """

        if not torch.is_tensor(q) or q.ndim != 4:
            raise ValueError("q must be a tensor shaped [B,K,N,128]")
        expected = (self.num_bands, self.num_patches, 128)
        if tuple(q.shape[1:]) != expected:
            raise ValueError(
                f"q must have shape [B,{expected[0]},{expected[1]},128], "
                f"got {tuple(q.shape)}"
            )
        if not q.is_floating_point():
            raise TypeError(f"q must be floating point, got {q.dtype}")

        batch = q.shape[0]
        length = self.num_patches * self.patch_size
        if self.decoder_type == "patch_mlp":
            band_reconstructions = []
            for band_index, patch_mlp in enumerate(self.decoder_patch_mlps):
                patches = patch_mlp(q[:, band_index])
                patches = patches.reshape(
                    batch, self.num_patches, 2, self.patch_size
                )
                band_reconstructions.append(
                    patches.permute(0, 2, 1, 3).reshape(batch, 2, length)
                )
            return torch.stack(band_reconstructions, dim=1)

        decoder_tokens = self.decoder_q_projection(
            q.reshape(batch * self.num_bands, self.num_patches, 128)
        )
        band_ids = torch.arange(self.num_bands, device=q.device).repeat(batch)
        decoder_tokens = self.decoder_position(decoder_tokens)
        decoder_tokens = decoder_tokens + self.decoder_band_embedding(band_ids).unsqueeze(1)
        reconstructed_patches = self.decoder(decoder_tokens)
        return (
            reconstructed_patches.reshape(
                batch, self.num_bands, self.num_patches, 2, self.patch_size
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, self.num_bands, 2, length)
        )

    def forward(self, paired_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        full_window = self._paired_full_window(paired_batch)
        band_targets = self.decomposer(full_window)
        q, indices = self._encode_decomposed(band_targets)
        band_reconstruction = self.decode_q(q)
        return {
            "q": q,
            "indices": indices,
            "band_reconstruction": band_reconstruction,
            "band_targets": band_targets,
        }

    def loss(self, paired_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self(paired_batch)
        error = (output["band_reconstruction"] - output["band_targets"]).abs()
        # Average over examples and time, preserving each frequency band and
        # paired channel for transparent metrics.
        band_channel_l1 = error.mean(dim=(0, 3))  # [K,2]
        scales = self.band_scales_tensor.to(device=error.device, dtype=error.dtype)
        if self.band_channel_scales_tensor is not None:
            channel_scales = self.band_channel_scales_tensor.to(
                device=error.device, dtype=error.dtype
            )
            scales = scales[:, None] * channel_scales
        else:
            scales = scales[:, None]
        band_weights = self.band_weights_tensor.to(device=error.device, dtype=error.dtype)
        channel_weights = self.channel_weights_tensor.to(device=error.device, dtype=error.dtype)
        weighted = (
            band_channel_l1 / scales
            * band_weights[:, None]
            * channel_weights[None, :]
        )
        denominator = (band_weights[:, None] * channel_weights[None, :]).sum().clamp_min(
            torch.finfo(error.dtype).eps
        )
        total_loss = weighted.sum() / denominator
        return {
            **output,
            "total_loss": total_loss,
            "band_channel_l1": band_channel_l1,
            "band_l1": band_channel_l1.mean(dim=-1),
            "channel_l1": band_channel_l1.mean(dim=0),
        }

    def _log_losses(
        self,
        prefix: str,
        metrics: dict[str, torch.Tensor],
        *,
        on_step,
        on_epoch,
        diagnostics: bool = True,
    ):
        self.log(
            f"{prefix}/loss",
            metrics["total_loss"],
            prog_bar=prefix == "val",
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=prefix == "val",
        )
        if not diagnostics:
            return
        band_channel = metrics["band_channel_l1"]
        for band_index, group in enumerate(self.decomposer.band_groups):
            label = "+".join(group).lower()
            self.log(
                f"{prefix}/recon_{label}",
                band_channel[band_index].mean(),
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=prefix == "val",
            )
            for channel_index, channel_name in enumerate(("ecg", "ppg")):
                self.log(
                    f"{prefix}/recon_{label}_{channel_name}",
                    band_channel[band_index, channel_index],
                    on_step=on_step,
                    on_epoch=on_epoch,
                    sync_dist=prefix == "val",
                )
        # The scalar FSQ labels are factorized; report observed level coverage
        # per band without building a packed 8**128 vocabulary.
        labels = metrics["indices"].detach()
        for band_index, group in enumerate(self.decomposer.band_groups):
            used = torch.bincount(
                labels[:, band_index].reshape(-1).long(), minlength=8
            ).gt(0).float().mean()
            label = "+".join(group).lower()
            self.log(
                f"{prefix}/level_coverage_{label}",
                used,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=False,
            )

    def training_step(self, batch, batch_idx):
        metrics = self.loss(batch)
        interval = int(getattr(self, "train_diagnostics_interval", 1))
        emit_diagnostics = batch_idx % interval == 0
        self._log_losses(
            "train", metrics, on_step=True, on_epoch=True,
            diagnostics=emit_diagnostics,
        )
        return metrics["total_loss"]

    def validation_step(self, batch, batch_idx):
        del batch_idx
        metrics = self.loss(batch)
        self._log_losses("val", metrics, on_step=False, on_epoch=True)
        return metrics["total_loss"]

    def configure_optimizers(self):
        options = getattr(self, "_training_acceleration_options", {})
        optimizer = build_training_adamw(
            self.parameters(),
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
        if self.lr_scheduler_type == "none":
            return optimizer
        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.total_epochs, eta_min=self.min_lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=self.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["frequency_teacher"] = {
            "schema": self.CHECKPOINT_SCHEMA,
            "band_metadata": self.band_metadata,
            "trained_steps": int(self.global_step),
        }
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

    def on_load_checkpoint(self, checkpoint: Mapping) -> None:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("frequency Teacher checkpoint must be a mapping")
        contract = checkpoint.get("frequency_teacher")
        if not isinstance(contract, Mapping):
            raise ValueError(
                "checkpoint is missing frequency_teacher metadata; only native "
                "frequency Teacher checkpoints are supported"
            )
        if contract.get("schema") != self.CHECKPOINT_SCHEMA:
            raise ValueError(
                "frequency Teacher checkpoint schema mismatch: "
                f"{contract.get('schema')!r} vs {self.CHECKPOINT_SCHEMA!r}"
            )
        saved_metadata = contract.get("band_metadata")
        if not isinstance(saved_metadata, Mapping):
            raise ValueError("frequency Teacher checkpoint is missing band_metadata")
        current_json = json.dumps(self.band_metadata, sort_keys=True, separators=(",", ":"))
        saved_json = json.dumps(dict(saved_metadata), sort_keys=True, separators=(",", ":"))
        if current_json != saved_json:
            raise ValueError(
                "frequency Teacher band metadata mismatch; wavelet, grouping, "
                "boundary support, input shape or FSQ layout differs"
            )
        hparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams"))
        hparam_metadata = (
            hparams.get("band_metadata")
            if isinstance(hparams, Mapping)
            else getattr(hparams, "band_metadata", None)
        )
        if hparam_metadata is None:
            raise ValueError("frequency Teacher checkpoint hyper_parameters lack band_metadata")
        hparam_json = json.dumps(dict(hparam_metadata), sort_keys=True, separators=(",", ":"))
        if current_json != hparam_json:
            raise ValueError("frequency Teacher hyperparameter band metadata mismatch")
        self.teacher_checkpoint_loaded = True

