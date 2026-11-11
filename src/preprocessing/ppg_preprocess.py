#!/usr/bin/env python3
"""Build the final PPG downstream datasets from the arranged source datasets.

Supported outputs are PPG-DaLiA HR and BIDMC HR/RR.  Subject order is read
from this package's metadata-only split CSV files.  The independent
cycle-derived PPG-DaLiA RR dataset is built by
``build_dalia_rr10s_cycle.py``.

Set ``RAW_DATA_ROOT`` to the directory containing ``PPG_DaLiA/`` and
``bidmc/1.0.0/bidmc_csv/``.  Set ``DOWNSTREAM_OUTPUT_ROOT`` to choose where
new arrays are written.  Existing task output directories are never replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import common

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = PACKAGE_ROOT / "data_protocols"
PROTOCOL_PATH = PROTOCOL_ROOT / "downstream_manifest.json"
RAW_ROOT = common.RAW_ROOT
DATA_ROOT = common.DATA_ROOT
PPG_DALIA_ROOT = RAW_ROOT / "PPG_DaLiA"
BIDMC_CSV_ROOT = RAW_ROOT / "bidmc" / "1.0.0" / "bidmc_csv"

PPG_FS = 64
TARGET_FS = 125
WINDOW_SAMPLES = 1250
DALIA_HR_TARGET_SECONDS = 8.0
DALIA_HR_CONTEXT_SECONDS = 10.0
DALIA_HR_STEP_SECONDS = 2.0
DALIA_HR_NATIVE_CONTEXT = int(PPG_FS * DALIA_HR_CONTEXT_SECONDS)
DALIA_HR_NATIVE_TARGET = int(PPG_FS * DALIA_HR_TARGET_SECONDS)
DALIA_HR_STEP = int(PPG_FS * DALIA_HR_STEP_SECONDS)
BIDMC_FS = 125
BIDMC_WINDOW = 1250
BIDMC_STEP = 250
TASKS = ("ppg_dalia_hr", "bidmc_hr", "bidmc_rr")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_manifest() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"Package data protocol is missing: {PROTOCOL_PATH}")
    with PROTOCOL_PATH.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema") != "physiofsq-downstream-data-protocol-v1":
        raise ValueError(f"Unrecognized data protocol in {PROTOCOL_PATH}")
    if value.get("sampling_rate_hz") != TARGET_FS or value.get("window_samples") != WINDOW_SAMPLES:
        raise ValueError("The packaged PPG builder expects 125 Hz and 1,250-sample windows")
    return value


def _read_split_rows(task: str) -> dict[str, list[dict[str, str]]]:
    """Read and authenticate split anchors shipped with this package."""
    manifest = _protocol_manifest()
    task_info = manifest.get("tasks", {}).get(task)
    if not isinstance(task_info, dict):
        raise KeyError(f"No data protocol entry for task {task!r}")
    relpath = task_info.get("sample_split_file")
    expected_hash = task_info.get("sample_split_sha256")
    if not relpath or not expected_hash:
        raise ValueError(f"Missing packaged split metadata for {task}")
    path = PROTOCOL_ROOT / relpath
    if not path.is_file():
        raise FileNotFoundError(f"Packaged subject split file is missing: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Packaged split file hash mismatch for {task}: {actual_hash} != {expected_hash}")
    required_counts = task_info.get("stored_split_counts", {})
    result: dict[str, list[dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row.get("split") == split]
        selected.sort(key=lambda row: int(row["split_row"]))
        if not selected:
            raise ValueError(f"No {split} anchors in {path}")
        starts = [int(row["split_row"]) for row in selected]
        subjects = [row["subject_id"] for row in selected]
        if starts[0] != 0 or any(b <= a for a, b in zip(starts, starts[1:])):
            raise ValueError(f"Invalid {split} split-row anchors in {path}: {starts}")
        if len(set(subjects)) != len(subjects):
            raise ValueError(f"Duplicate subject anchors in {task}/{split}")
        total = int(required_counts[split]["samples"])
        if starts[-1] >= total:
            raise ValueError(f"Split anchor falls outside declared {task}/{split} sample count")
        for row in selected:
            source_id = row.get("source_id", "")
            subject_id = row.get("subject_id", "")
            if not source_id or source_id != subject_id:
                raise ValueError(f"Unexpected source/subject identity in {path}: {row}")
        result[split] = selected
    return result


def _expected_subject_counts(task: str, split: str) -> dict[str, int]:
    manifest = _protocol_manifest()
    rows = _read_split_rows(task)[split]
    total = int(manifest["tasks"][task]["stored_split_counts"][split]["samples"])
    return {
        row["subject_id"]: (int(rows[i + 1]["split_row"]) if i + 1 < len(rows) else total)
        - int(row["split_row"])
        for i, row in enumerate(rows)
    }


def _subject_order(task: str, split: str) -> list[str]:
    return [row["subject_id"] for row in _read_split_rows(task)[split]]


def _load_dalia(subject_id: str) -> dict[str, Any]:
    path = PPG_DALIA_ROOT / subject_id / f"{subject_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing PPG-DaLiA source file {path}; set RAW_DATA_ROOT to the arranged dataset root"
        )
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def _signal_1d(value: Any, name: str) -> np.ndarray:
    signal = np.asarray(value).reshape(-1)
    if signal.dtype.kind not in "fiu":
        signal = signal.astype(np.float64)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"{name} is empty or contains non-finite samples")
    return signal


def _sample_row(
    *, sample_id: str, source_id: str, subject_id: str, source_row: int,
    source_native_fs_hz: int, source_start_sample: int, source_end_sample_exclusive: int,
    input_start_time_s: float, input_end_time_s: float,
    target_start_time_s: float, target_end_time_s: float,
    source_target_index: int, label_semantics: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "subject_id": subject_id,
        # Retained for compatibility with the downstream reader's row-order key.
        "old_sample_index": int(source_row),
        "source_native_fs_hz": int(source_native_fs_hz),
        "source_start_sample": int(source_start_sample),
        "source_end_sample_exclusive": int(source_end_sample_exclusive),
        "input_start_time_s": float(input_start_time_s),
        "input_end_time_s": float(input_end_time_s),
        "target_start_time_s": float(target_start_time_s),
        "target_end_time_s": float(target_end_time_s),
        "source_target_index": int(source_target_index),
        "label_semantics": label_semantics,
    }


def _window_row(sample_index: int, source_id: str, fs: int, start: int, end: int) -> dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "window_index": 0,
        "source_id": source_id,
        "source_native_fs_hz": int(fs),
        "source_start_sample": int(start),
        "source_end_sample_exclusive": int(end),
        "window_start_time_s": float(start / fs),
        "window_end_time_s": float(end / fs),
    }


def _task_output_guard(tasks: list[str]) -> None:
    for task in tasks:
        destination = DATA_ROOT / task
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing task output: {destination}")


def _source_file_info(paths: list[Path]) -> list[dict[str, Any]]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing[:5]}")
    return [{"relative_path": str(path.relative_to(RAW_ROOT)), "size_bytes": path.stat().st_size} for path in paths]


def build_dalia_hr() -> dict[str, Any]:
    task = "ppg_dalia_hr"
    split_rows = _read_split_rows(task)
    task_protocol = _protocol_manifest()["tasks"][task]
    _task_output_guard([task])
    split_stats: dict[str, Any] = {}
    subject_counts: dict[str, dict[str, int]] = {}
    files = [PPG_DALIA_ROOT / sid / f"{sid}.pkl" for split in split_rows.values() for sid in (r["subject_id"] for r in split)]

    for split in ("train", "val", "test"):
        subject_ids = [row["subject_id"] for row in split_rows[split]]
        expected_counts = _expected_subject_counts(task, split)
        features: list[np.ndarray] = []
        labels: list[float] = []
        samples: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []
        source_label_offset = 0

        for subject_id in subject_ids:
            payload = _load_dalia(subject_id)
            bvp = _signal_1d(payload["signal"]["wrist"]["BVP"], f"{subject_id} wrist BVP")
            hr = _signal_1d(payload["label"], f"{subject_id} official HR labels")
            n_original = min(
                len(hr),
                max(0, (len(bvp) - DALIA_HR_NATIVE_TARGET) // DALIA_HR_STEP + 1),
            )
            if n_original < 2:
                raise ValueError(f"{subject_id} has fewer than two complete official HR target windows")

            # The label remains the official 8 s target.  Its input is the
            # preceding 10 s context, preserving the released task alignment.
            target_indices = np.arange(1, n_original, dtype=np.int64)
            contexts = np.stack([
                bvp[(int(index) - 1) * DALIA_HR_STEP:
                    (int(index) - 1) * DALIA_HR_STEP + DALIA_HR_NATIVE_CONTEXT]
                for index in target_indices
            ])
            if contexts.shape[1] != DALIA_HR_NATIVE_CONTEXT:
                raise ValueError(f"Incomplete 10 s BVP context in {subject_id}")
            contexts_125 = common.resample_signal(contexts, PPG_FS)
            if contexts_125.shape[1:] != (WINDOW_SAMPLES,):
                raise ValueError(f"Unexpected resampled DaLiA HR shape: {contexts_125.shape}")

            first_row = len(samples)
            features.append(contexts_125[:, None, :])
            labels.extend(hr[target_indices].astype(np.float64).tolist())
            for source_index in target_indices.tolist():
                sample_index = len(samples)
                native_start = (source_index - 1) * DALIA_HR_STEP
                native_end = native_start + DALIA_HR_NATIVE_CONTEXT
                target_start = source_index * DALIA_HR_STEP_SECONDS
                samples.append(_sample_row(
                    sample_id=f"{subject_id}_hr_label_{source_index:06d}",
                    source_id=subject_id,
                    subject_id=subject_id,
                    source_row=source_label_offset + source_index,
                    source_native_fs_hz=PPG_FS,
                    source_start_sample=native_start,
                    source_end_sample_exclusive=native_end,
                    input_start_time_s=native_start / PPG_FS,
                    input_end_time_s=native_end / PPG_FS,
                    target_start_time_s=target_start,
                    target_end_time_s=target_start + DALIA_HR_TARGET_SECONDS,
                    source_target_index=source_index,
                    label_semantics="Official HR label in beats/minute for the original 8-second interval; input includes 2 seconds of preceding BVP context",
                ))
                windows.append(_window_row(sample_index, subject_id, PPG_FS, native_start, native_end))
            emitted = len(samples) - first_row
            if emitted != expected_counts[subject_id]:
                raise ValueError(
                    f"{task}/{split}/{subject_id} emits {emitted} rows; packaged split anchors require {expected_counts[subject_id]}"
                )
            subject_counts.setdefault(split, {})[subject_id] = emitted
            source_label_offset += n_original

        feature_array = np.concatenate(features, axis=0).astype(np.float32, copy=False)
        label_array = np.asarray(labels, dtype=np.float64)
        split_stats[split] = common.save_split(
            task, split, feature_array, label_array, samples, windows,
            offsets=np.arange(len(samples) + 1, dtype=np.int64),
        )
        split_stats[split]["subjects"] = subject_ids
        split_stats[split]["samples_by_subject"] = subject_counts[split]

    details = {
        "dataset": task_protocol["dataset"],
        "task": "Heart-rate regression from wrist BVP",
        "source_root_layout": "RAW_DATA_ROOT/PPG_DaLiA/{subject}/{subject}.pkl",
        "source_files": _source_file_info(files),
        "source_sampling_rate_hz": PPG_FS,
        "source_signal": "signal/wrist/BVP",
        "source_label": "Official PPG-DaLiA HR label, beats/minute",
        "split_protocol": task_protocol["split_protocol"],
        "split_metadata_file": task_protocol["sample_split_file"],
        "split_metadata_sha256": task_protocol["sample_split_sha256"],
        "subject_order_by_split": {split: [r["subject_id"] for r in split_rows[split]] for split in split_rows},
        "window_protocol": {
            "input_context": "10 s BVP context [2*i-2, 2*i+8] for official HR label index i >= 1",
            "label_interval": "Unchanged official HR target interval [2*i, 2*i+8] seconds",
            "stride_seconds": DALIA_HR_STEP_SECONDS,
            "first_official_label_skipped": True,
            "resampling": "Each 640-sample context is independently polyphase-resampled from 64 Hz to 125 Hz",
        },
        "label_units": "beats/minute",
        "split_statistics": split_stats,
        "samples_by_subject": subject_counts,
        "total_samples": int(sum(item["samples"] for item in split_stats.values())),
    }
    return common.finish_task(task, details)


def _load_bidmc_subject(subject_id: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[Path]]:
    match = re.fullmatch(r"bidmc_(\d{2})", subject_id)
    if not match:
        raise ValueError(f"Unexpected BIDMC subject identifier in packaged split: {subject_id}")
    prefix = subject_id
    paths = [BIDMC_CSV_ROOT / f"{prefix}_{kind}.csv" for kind in ("Signals", "Numerics", "Breaths")]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing BIDMC source files: {missing}")
    signals, numerics, breaths = [pd.read_csv(path) for path in paths]
    for frame in (signals, numerics, breaths):
        frame.columns = frame.columns.str.strip()
    if "PLETH" not in signals or "Time [s]" not in numerics or "HR" not in numerics:
        raise ValueError(f"Required PLETH/Numerics columns are missing for {prefix}")
    if "breaths ann1 [signal sample no]" not in breaths:
        raise ValueError(f"Required manual breath event column is missing for {prefix}")
    ppg = signals["PLETH"].to_numpy()
    if not np.isfinite(ppg).all():
        raise ValueError(f"Nonfinite BIDMC PLETH samples for {prefix}")
    return ppg, numerics, breaths, paths


def build_bidmc(task: str) -> dict[str, Any]:
    if task not in ("bidmc_hr", "bidmc_rr"):
        raise ValueError(f"Unsupported BIDMC task: {task}")
    split_rows = _read_split_rows(task)
    companion = "bidmc_rr" if task == "bidmc_hr" else "bidmc_hr"
    companion_rows = _read_split_rows(companion)
    for split in ("train", "val", "test"):
        if [r["subject_id"] for r in split_rows[split]] != [r["subject_id"] for r in companion_rows[split]]:
            raise ValueError("BIDMC HR and RR split subject ordering differs in packaged split files")
    task_protocol = _protocol_manifest()["tasks"][task]
    _task_output_guard([task])
    split_stats: dict[str, Any] = {}
    samples_by_split: dict[str, dict[str, int]] = {}
    all_subjects = [row["subject_id"] for split in split_rows.values() for row in split]
    source_files = [
        BIDMC_CSV_ROOT / f"{subject_id}_{kind}.csv"
        for subject_id in all_subjects for kind in ("Signals", "Numerics", "Breaths")
    ]

    for split in ("train", "val", "test"):
        subject_ids = [row["subject_id"] for row in split_rows[split]]
        expected_counts = _expected_subject_counts(task, split)
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        samples: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []
        subject_counts: dict[str, int] = {}

        for subject_id in subject_ids:
            ppg, numerics, breaths, _ = _load_bidmc_subject(subject_id)
            starts = np.arange(
                0, max(0, len(ppg) - BIDMC_WINDOW + 1), BIDMC_STEP, dtype=np.int64,
            )
            if not len(starts):
                raise ValueError(f"No complete BIDMC windows in {subject_id}")
            raw_windows = np.stack([ppg[int(start):int(start) + BIDMC_WINDOW] for start in starts])
            first_row = len(samples)

            if task == "bidmc_hr":
                time_values = numerics["Time [s]"].to_numpy(dtype=np.float64)
                kept_positions: list[int] = []
                subject_labels: list[float] = []
                for position, start in enumerate(starts.tolist()):
                    end = start + BIDMC_WINDOW
                    start_sec, end_sec = start / BIDMC_FS, end / BIDMC_FS
                    mask = (time_values >= start_sec) & (time_values < end_sec)
                    values = numerics.loc[mask, "HR"]
                    if not values.empty:
                        average = float(values.mean())
                        if not pd.isna(average):
                            kept_positions.append(position)
                            subject_labels.append(average)
                if kept_positions:
                    features.append(raw_windows[kept_positions, None, :].astype(np.float32, copy=False))
                    labels.append(np.asarray(subject_labels, dtype=np.float64))
                    for position, label in zip(kept_positions, subject_labels):
                        start = int(starts[position])
                        sample_index = len(samples)
                        samples.append(_sample_row(
                            sample_id=f"{subject_id}_hr_start_{start:06d}",
                            source_id=subject_id, subject_id=subject_id, source_row=sample_index,
                            source_native_fs_hz=BIDMC_FS,
                            source_start_sample=start, source_end_sample_exclusive=start + BIDMC_WINDOW,
                            input_start_time_s=start / BIDMC_FS,
                            input_end_time_s=(start + BIDMC_WINDOW) / BIDMC_FS,
                            target_start_time_s=start / BIDMC_FS,
                            target_end_time_s=(start + BIDMC_WINDOW) / BIDMC_FS,
                            source_target_index=position,
                            label_semantics="Mean of non-NaN Numerics HR annotations with timestamps in [window_start, window_end)",
                        ))
                        windows.append(_window_row(sample_index, subject_id, BIDMC_FS, start, start + BIDMC_WINDOW))
            else:
                event_samples = breaths["breaths ann1 [signal sample no]"].to_numpy(dtype=np.float64)
                breath_counts = np.asarray([
                    np.sum((event_samples >= start) & (event_samples < start + BIDMC_WINDOW))
                    for start in starts
                ], dtype=np.float64)
                subject_labels = breath_counts / 10.0 * 60.0
                features.append(raw_windows[:, None, :].astype(np.float32, copy=False))
                labels.append(subject_labels)
                for position, start_value in enumerate(starts.tolist()):
                    start = int(start_value)
                    sample_index = len(samples)
                    samples.append(_sample_row(
                        sample_id=f"{subject_id}_rr_start_{start:06d}",
                        source_id=subject_id, subject_id=subject_id, source_row=sample_index,
                        source_native_fs_hz=BIDMC_FS,
                        source_start_sample=start, source_end_sample_exclusive=start + BIDMC_WINDOW,
                        input_start_time_s=start / BIDMC_FS,
                        input_end_time_s=(start + BIDMC_WINDOW) / BIDMC_FS,
                        target_start_time_s=start / BIDMC_FS,
                        target_end_time_s=(start + BIDMC_WINDOW) / BIDMC_FS,
                        source_target_index=position,
                        label_semantics="Manual breaths ann1 event count in [window_start_sample, window_end_sample), divided by 10 seconds and multiplied by 60",
                    ))
                    windows.append(_window_row(sample_index, subject_id, BIDMC_FS, start, start + BIDMC_WINDOW))

            emitted = len(samples) - first_row
            if emitted != expected_counts[subject_id]:
                raise ValueError(
                    f"{task}/{split}/{subject_id} emits {emitted} rows; packaged split anchors require {expected_counts[subject_id]}"
                )
            subject_counts[subject_id] = emitted

        feature_array = np.concatenate(features, axis=0).astype(np.float32, copy=False)
        label_array = np.concatenate(labels).astype(np.float64, copy=False)
        split_stats[split] = common.save_split(
            task, split, feature_array, label_array, samples, windows,
            offsets=np.arange(len(samples) + 1, dtype=np.int64),
        )
        split_stats[split]["subjects"] = subject_ids
        split_stats[split]["samples_by_subject"] = subject_counts
        samples_by_split[split] = subject_counts

    details = {
        "dataset": task_protocol["dataset"],
        "task": "Heart-rate regression from PLETH" if task == "bidmc_hr" else "Respiration-rate regression from PLETH",
        "source_root_layout": "RAW_DATA_ROOT/bidmc/1.0.0/bidmc_csv/bidmc_{subject}_{Signals,Numerics,Breaths}.csv",
        "source_files": _source_file_info(source_files),
        "source_sampling_rate_hz": BIDMC_FS,
        "source_signal": "Signals.csv column PLETH",
        "source_label": (
            "Mean of Numerics.csv HR annotations in each half-open 10-second interval, beats/minute"
            if task == "bidmc_hr" else
            "Count of manual breaths ann1 signal sample indices per half-open 10-second window, converted to breaths/minute"
        ),
        "label_units": "beats/minute" if task == "bidmc_hr" else "breaths/minute",
        "split_protocol": task_protocol["split_protocol"],
        "split_metadata_file": task_protocol["sample_split_file"],
        "split_metadata_sha256": task_protocol["sample_split_sha256"],
        "subject_order_by_split": {split: [r["subject_id"] for r in split_rows[split]] for split in split_rows},
        "window_protocol": {
            "window_samples": BIDMC_WINDOW,
            "stride_samples": BIDMC_STEP,
            "stride_seconds": BIDMC_STEP / BIDMC_FS,
            "interval_semantics": "Half-open [start_sample, end_sample); complete 10-second PLETH windows only",
            "resampling": "None; source sampling rate is already 125 Hz",
        },
        "label_filtering": (
            "Keep a window only when it contains at least one Numerics HR annotation and the mean is finite; pandas-style NaN values are ignored in the mean"
            if task == "bidmc_hr" else "Every complete PLETH window is retained"
        ),
        "split_statistics": split_stats,
        "samples_by_subject": samples_by_split,
        "total_samples": int(sum(item["samples"] for item in split_stats.values())),
    }
    return common.finish_task(task, details)


def verify_prepared_examples(task: str, prepared_root: Path) -> dict[str, Any]:
    """Compare three raw-derived examples with a prepared final array release."""
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    task_dir = prepared_root / task
    reports: list[dict[str, Any]] = []
    target_subject = "S4" if task == "ppg_dalia_hr" else "bidmc_20"

    for split in ("train", "val", "test"):
        sample_path = task_dir / f"{split}_samples.csv"
        feature_path = task_dir / f"{split}_features.npy"
        label_path = task_dir / f"{split}_labels.npy"
        if not all(path.is_file() for path in (sample_path, feature_path, label_path)):
            continue
        table = pd.read_csv(sample_path, dtype={"sample_id": str, "source_id": str, "subject_id": str})
        chosen = table.index[table["subject_id"].astype(str) == target_subject].tolist()
        if not chosen:
            continue
        indices = sorted(set([chosen[0], chosen[len(chosen) // 2], chosen[-1]]))
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if task == "ppg_dalia_hr":
            payload = _load_dalia(target_subject)
            bvp = _signal_1d(payload["signal"]["wrist"]["BVP"], f"{target_subject} wrist BVP")
            hr = _signal_1d(payload["label"], f"{target_subject} official HR labels")
            for row_index in indices:
                row = table.iloc[row_index]
                source_index = int(row["source_target_index"])
                begin = (source_index - 1) * DALIA_HR_STEP
                context = bvp[begin:begin + DALIA_HR_NATIVE_CONTEXT]
                reconstructed = common.resample_signal(context, PPG_FS)
                feature_error = float(np.max(np.abs(features[row_index, 0].astype(np.float64) - reconstructed.astype(np.float64))))
                label_error = float(abs(float(labels[row_index]) - float(hr[source_index])))
                reports.append({
                    "split": split, "subject_id": target_subject,
                    "sample_id": str(row["sample_id"]), "feature_max_abs_error": feature_error,
                    "label_abs_error": label_error,
                })
        else:
            ppg, numerics, breaths, _ = _load_bidmc_subject(target_subject)
            for row_index in indices:
                row = table.iloc[row_index]
                start = int(row["source_start_sample"])
                end = start + BIDMC_WINDOW
                reference = ppg[start:end].astype(np.float32)
                feature_error = float(np.max(np.abs(features[row_index, 0].astype(np.float64) - reference.astype(np.float64))))
                if task == "bidmc_hr":
                    times = numerics["Time [s]"].to_numpy(dtype=np.float64)
                    mask = (times >= start / BIDMC_FS) & (times < end / BIDMC_FS)
                    expected = float(numerics.loc[mask, "HR"].mean())
                else:
                    events = breaths["breaths ann1 [signal sample no]"].to_numpy(dtype=np.float64)
                    expected = float(np.sum((events >= start) & (events < end)) / 10.0 * 60.0)
                label_error = float(abs(float(labels[row_index]) - expected))
                reports.append({
                    "split": split, "subject_id": target_subject,
                    "sample_id": str(row["sample_id"]), "feature_max_abs_error": feature_error,
                    "label_abs_error": label_error,
                })
    if not reports:
        raise ValueError(f"No prepared examples found for {task}/{target_subject} under {prepared_root}")
    for item in reports:
        if item["feature_max_abs_error"] > 1e-6 or item["label_abs_error"] > 1e-8:
            raise AssertionError(f"Raw-to-prepared PPG check failed: {item}")
    return {
        "task": task,
        "prepared_reference_root": str(prepared_root),
        "source_subject": target_subject,
        "examples_compared": len(reports),
        "all_checks_passed": True,
        "examples": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True,
                        help="Build one final task, or select the task for --verify-prepared")
    parser.add_argument("--verify-prepared", action="store_true",
                        help="Compare several raw-derived windows with a prepared final array release; writes nothing")
    parser.add_argument("--prepared-data-root", type=Path,
                        help="Prepared arrays root used only by --verify-prepared")
    args = parser.parse_args()
    if args.verify_prepared:
        if args.prepared_data_root is None:
            parser.error("--verify-prepared requires --prepared-data-root")
        result = verify_prepared_examples(args.task, args.prepared_data_root.expanduser().resolve())
    elif args.task == "ppg_dalia_hr":
        result = build_dalia_hr()
    else:
        result = build_bidmc(args.task)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
