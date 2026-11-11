#!/usr/bin/env python3
"""Build the final PPG-DaLiA RR cycle-derived benchmark at 125 Hz.

Subject order comes from this package's metadata-only split file. Each sample
contains one 10 s wrist-BVP window and one rate integrated over accepted
respiration cycles from the synchronized chest Resp reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
RAW_ROOT = Path(os.environ.get("RAW_DATA_ROOT", str(PACKAGE_ROOT / "data" / "raw"))).expanduser()
SOURCE_ROOT = Path(os.environ.get("PPG_DALIA_RAW_ROOT", str(RAW_ROOT / "PPG_DaLiA"))).expanduser()
DATA_ROOT = Path(os.environ.get("DALIA_RR_OUTPUT_ROOT", str(PACKAGE_ROOT / "data" / "dalia_rr10s_cycle_v1"))).expanduser()
TASK = "ppg_dalia_rr"
PROTOCOL_ROOT = PACKAGE_ROOT / "data_protocols"
PROTOCOL_PATH = PROTOCOL_ROOT / "downstream_manifest.json"
SPLIT_FILE = PROTOCOL_ROOT / "splits" / "ppg_dalia_rr.csv"
OUTPUT_TASK = DATA_ROOT / TASK
QA_ROOT = HERE / "qa"
PILOT_ROOT = QA_ROOT / "pilot_S10"

# Set the shared writer's root before importing it. All writes go to the
# separately configured cycle-derived output directory.
os.environ["DOWNSTREAM_OUTPUT_ROOT"] = str(DATA_ROOT)
os.environ["DOWNSTREAM_CODE_ROOT"] = str(HERE)
sys.path.insert(0, str(HERE))
import common  # noqa: E402

def packaged_subject_splits() -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Load the published subject order from this package, without old arrays."""
    if not PROTOCOL_PATH.is_file() or not SPLIT_FILE.is_file():
        raise FileNotFoundError("Packaged PPG-DaLiA RR split metadata is missing")
    manifest = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    task_info = manifest.get("tasks", {}).get(TASK)
    if not isinstance(task_info, dict):
        raise ValueError(f"No packaged protocol entry for {TASK}")
    split_hash = hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest()
    if split_hash != task_info.get("sample_split_sha256"):
        raise ValueError(
            f"Packaged {TASK} split file hash mismatch: {split_hash} != "
            f"{task_info.get('sample_split_sha256')}"
        )
    result: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    with SPLIT_FILE.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for split in result:
        selected = sorted(
            (row for row in rows if row.get("split") == split),
            key=lambda row: int(row["split_row"]),
        )
        if not selected or int(selected[0]["split_row"]) != 0:
            raise ValueError(f"Packaged {TASK}/{split} subject anchors are missing or malformed")
        subjects = [row.get("subject_id", "") for row in selected]
        if any(not subject for subject in subjects) or len(set(subjects)) != len(subjects):
            raise ValueError(f"Packaged {TASK}/{split} contains invalid or duplicate subjects")
        starts = [int(row["split_row"]) for row in selected]
        total = int(task_info["stored_split_counts"][split]["samples"])
        if any(right <= left for left, right in zip(starts, starts[1:])) or starts[-1] >= total:
            raise ValueError(f"Packaged {TASK}/{split} split-row anchors are out of order")
        if any(row.get("source_id") != row.get("subject_id") for row in selected):
            raise ValueError(f"Packaged {TASK}/{split} source and subject IDs disagree")
        result[split] = subjects
    return result, {
        "manifest": manifest,
        "task_protocol": task_info,
        "split_file_sha256": split_hash,
    }


SPLITS, SPLIT_PROTOCOL = packaged_subject_splits()
PPG_FS = 64
RESP_FS = 700
RESP_WORK_FS = 25
TARGET_FS = 125
WINDOW_SECONDS = 10
STEP_SECONDS = 1
WINDOW_SAMPLES = WINDOW_SECONDS * TARGET_FS
RESP_FILTER_LOW_HZ = 0.08
RESP_FILTER_HIGH_HZ = 1.0
RESP_FILTER_ORDER = 4
PEAK_MIN_DISTANCE_SECONDS = 1.0
PEAK_PROMINENCE_ROBUST_SIGMA_MULTIPLIER = 0.35
LOW_PROMINENCE_QA_MULTIPLIER = 1.5
ACTIVITY_FS = 4
PERIOD_MIN_SECONDS = 1.0
PERIOD_MAX_SECONDS = 12.0
FILTER_EDGE_EXCLUSION_SECONDS = 12.5
COVERAGE_TOLERANCE_SECONDS = 1e-8
RESP_RESAMPLE_WINDOW = ("kaiser", 5.0)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(subject_id: str) -> dict[str, Any]:
    path = SOURCE_ROOT / subject_id / f"{subject_id}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def signal_1d(value: Any, name: str) -> np.ndarray:
    signal = np.asarray(value).reshape(-1)
    if signal.dtype.kind not in "fiu":
        signal = signal.astype(np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"{name} is empty or contains non-finite values")
    return signal


def robust_sigma(signal: np.ndarray) -> float:
    center = float(np.median(signal))
    return float(1.4826 * np.median(np.abs(signal - center)))


def prepare_resp(resp_native: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return (downsampled, filtered, prominence threshold, SOS coefficients)."""
    if RESP_FS % RESP_WORK_FS:
        raise ValueError("This builder expects an integer 700→25 Hz ratio")
    resp_work = resample_poly(
        np.asarray(resp_native, dtype=np.float64), 1, RESP_FS // RESP_WORK_FS,
        window=RESP_RESAMPLE_WINDOW, padtype="constant",
    )
    sos = butter(
        RESP_FILTER_ORDER,
        [RESP_FILTER_LOW_HZ, RESP_FILTER_HIGH_HZ],
        btype="bandpass", fs=RESP_WORK_FS, output="sos",
    )
    filtered = sosfiltfilt(sos, resp_work)
    scale = robust_sigma(filtered)
    prominence = PEAK_PROMINENCE_ROBUST_SIGMA_MULTIPLIER * scale
    if not np.isfinite(prominence) or prominence <= 0:
        raise ValueError("Resp filtered robust scale is zero or non-finite")
    return resp_work, filtered, float(prominence), sos


def detect_cycles(filtered_resp: np.ndarray, prominence: float) -> dict[str, np.ndarray]:
    """Detect fixed positive respiratory peaks and annotate each full period."""
    min_distance = int(round(PEAK_MIN_DISTANCE_SECONDS * RESP_WORK_FS))
    peaks, properties = find_peaks(
        filtered_resp,
        distance=min_distance,
        prominence=prominence,
    )
    peak_times = peaks.astype(np.float64) / RESP_WORK_FS
    peak_prominences = np.asarray(properties.get("prominences", []), dtype=np.float64)
    low_prominence_peak = peak_prominences < LOW_PROMINENCE_QA_MULTIPLIER * prominence
    if len(peaks) < 2:
        return {
            "peak_indices": peaks.astype(np.int64),
            "peak_times_s": peak_times,
            "peak_prominences": peak_prominences,
            "low_prominence_peak": low_prominence_peak,
            "period_s": np.empty(0, dtype=np.float64),
            "rate_bpm": np.empty(0, dtype=np.float64),
            "valid": np.empty(0, dtype=bool),
            "reason": np.empty(0, dtype=object),
        }
    starts, ends = peak_times[:-1], peak_times[1:]
    period = ends - starts
    reason = np.full(period.shape, "valid", dtype=object)
    short = period < PERIOD_MIN_SECONDS
    long = period > PERIOD_MAX_SECONDS
    edge = (starts < FILTER_EDGE_EXCLUSION_SECONDS) | (
        ends > len(filtered_resp) / RESP_WORK_FS - FILTER_EDGE_EXCLUSION_SECONDS
    )
    reason[short] = "period_shorter_than_1s"
    reason[long] = "period_longer_than_12s"
    reason[edge] = np.where(
        starts[edge] < FILTER_EDGE_EXCLUSION_SECONDS,
        "filter_edge_left",
        "filter_edge_right",
    )
    # The physiological interval rule takes precedence over the edge label so
    # reports can count long/short intervals independently of end buffering.
    reason[short] = "period_shorter_than_1s"
    reason[long] = "period_longer_than_12s"
    valid = (period >= PERIOD_MIN_SECONDS) & (period <= PERIOD_MAX_SECONDS) & ~edge
    return {
        "peak_indices": peaks.astype(np.int64),
        "peak_times_s": peak_times,
        "peak_prominences": peak_prominences,
        "low_prominence_peak": low_prominence_peak,
        "period_s": period.astype(np.float64),
        "rate_bpm": (60.0 / period).astype(np.float64),
        "valid": valid,
        "reason": reason,
    }


def _prefix_at_times(
    times: np.ndarray,
    cycle_starts: np.ndarray,
    cycle_ends: np.ndarray,
    cycle_valid: np.ndarray,
    cycle_rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative valid coverage and rate integral at arbitrary times."""
    duration = cycle_ends - cycle_starts
    valid_duration = np.where(cycle_valid, duration, 0.0)
    valid_integral = np.where(cycle_valid, duration * cycle_rates, 0.0)
    prefix_duration = np.concatenate(([0.0], np.cumsum(valid_duration, dtype=np.float64)))
    prefix_integral = np.concatenate(([0.0], np.cumsum(valid_integral, dtype=np.float64)))
    if len(cycle_ends) == 0:
        return np.zeros_like(times, dtype=np.float64), np.zeros_like(times, dtype=np.float64)
    positions = np.searchsorted(cycle_ends, times, side="right")
    coverage = prefix_duration[positions].copy()
    integral = prefix_integral[positions].copy()
    active = positions < len(cycle_ends)
    p = positions[active]
    partial = np.clip(times[active] - cycle_starts[p], 0.0, cycle_ends[p] - cycle_starts[p])
    partial = np.where(cycle_valid[p], partial, 0.0)
    coverage[active] += partial
    integral[active] += partial * cycle_rates[p]
    return coverage, integral


def integrate_windows(
    cycles: dict[str, np.ndarray], duration_s: float
) -> dict[str, Any]:
    """Create integer-second 10 s labels; invalid/uncovered windows are dropped."""
    if duration_s < WINDOW_SECONDS:
        raise ValueError(f"Recording too short for a {WINDOW_SECONDS}s window")
    candidate_starts = np.arange(
        0, int(np.floor(duration_s - WINDOW_SECONDS)) + 1, STEP_SECONDS,
        dtype=np.float64,
    )
    candidate_ends = candidate_starts + WINDOW_SECONDS
    cycle_starts = cycles["peak_times_s"][:-1]
    cycle_ends = cycles["peak_times_s"][1:]
    valid = cycles["valid"]
    rates = cycles["rate_bpm"]
    cov_start, int_start = _prefix_at_times(candidate_starts, cycle_starts, cycle_ends, valid, rates)
    cov_end, int_end = _prefix_at_times(candidate_ends, cycle_starts, cycle_ends, valid, rates)
    coverage = cov_end - cov_start
    rate_integral = int_end - int_start
    edge = (candidate_starts < FILTER_EDGE_EXCLUSION_SECONDS) | (
        candidate_ends > duration_s - FILTER_EDGE_EXCLUSION_SECONDS
    )
    covered = coverage >= WINDOW_SECONDS - COVERAGE_TOLERANCE_SECONDS
    retained = ~edge & covered

    # For every rejected non-edge candidate, identify whether an explicit
    # invalid interval intersects it or whether no detected period brackets it.
    bad = ~valid
    bad_duration = np.where(bad, cycle_ends - cycle_starts, 0.0)
    prefix_bad = np.concatenate(([0.0], np.cumsum(bad_duration, dtype=np.float64)))
    pos_s = np.searchsorted(cycle_ends, candidate_starts, side="right")
    pos_e = np.searchsorted(cycle_ends, candidate_ends, side="right")
    invalid_overlap = np.zeros(candidate_starts.shape, dtype=np.float64)
    full_segments = pos_e > pos_s
    if np.any(full_segments):
        left = pos_s[full_segments]
        right = pos_e[full_segments]
        invalid_overlap[full_segments] = prefix_bad[right] - prefix_bad[left]
        # Correct the two partial edge cycles. Since periods are sorted and
        # non-overlapping this handles starts/ends lying inside a bad cycle.
        for i in np.flatnonzero(full_segments):
            if pos_s[i] < len(bad) and bad[pos_s[i]]:
                invalid_overlap[i] -= max(0.0, candidate_starts[i] - cycle_starts[pos_s[i]])
            if pos_e[i] > 0 and pos_e[i] - 1 < len(bad) and bad[pos_e[i] - 1]:
                invalid_overlap[i] -= max(0.0, cycle_ends[pos_e[i] - 1] - candidate_ends[i])
    # The direct overlap calculation for only failed windows is cheap and avoids
    # relying on prefix-edge bookkeeping for exact reason counts.
    invalid_intersects = np.zeros(candidate_starts.shape, dtype=bool)
    for i in np.flatnonzero(~edge & ~covered):
        if len(bad):
            overlap = np.maximum(
                0.0,
                np.minimum(cycle_ends[bad], candidate_ends[i])
                - np.maximum(cycle_starts[bad], candidate_starts[i]),
            )
            invalid_intersects[i] = bool(np.any(overlap > COVERAGE_TOLERANCE_SECONDS))

    kept_indices = np.flatnonzero(retained)
    labels = rate_integral[retained] / WINDOW_SECONDS
    if not np.isfinite(labels).all() or np.any(labels <= 0):
        raise AssertionError("Cycle-integrated RR labels must be finite and positive")
    return {
        "candidate_starts_s": candidate_starts,
        "candidate_ends_s": candidate_ends,
        "candidate_coverage_s": coverage,
        "candidate_retained": retained,
        "candidate_edge": edge,
        "candidate_invalid_cycle": invalid_intersects,
        "candidate_indices": kept_indices,
        "starts_s": candidate_starts[retained],
        "ends_s": candidate_ends[retained],
        "labels_bpm": labels,
        "coverage_s": coverage[retained],
        "integral_bpm_seconds": rate_integral[retained],
        "counts": {
            "candidate_windows": int(len(candidate_starts)),
            "retained_windows": int(retained.sum()),
            "excluded_filter_edge_windows": int(edge.sum()),
            "excluded_invalid_cycle_windows": int(np.sum(~edge & ~covered & invalid_intersects)),
            "excluded_uncovered_or_undetected_windows": int(np.sum(~edge & ~covered & ~invalid_intersects)),
        },
    }


def _cycle_rows(
    subject_id: str, cycles: dict[str, np.ndarray], prominence_threshold: float
) -> list[dict[str, Any]]:
    rows = []
    peaks = cycles["peak_indices"]
    peak_times = cycles["peak_times_s"]
    prominences = cycles["peak_prominences"]
    for i, (period, rate, valid, reason) in enumerate(zip(
        cycles["period_s"], cycles["rate_bpm"], cycles["valid"], cycles["reason"]
    )):
        rows.append({
            "subject_id": subject_id,
            "cycle_index": i,
            "start_peak_index_resp25hz": int(peaks[i]),
            "end_peak_index_resp25hz": int(peaks[i + 1]),
            "start_time_s": float(peak_times[i]),
            "end_time_s": float(peak_times[i + 1]),
            "period_s": float(period),
            "cycle_rate_bpm": float(rate),
            "start_peak_prominence": float(prominences[i]),
            "end_peak_prominence": float(prominences[i + 1]),
            "start_prominence_threshold_ratio": float(prominences[i] / prominence_threshold),
            "end_prominence_threshold_ratio": float(prominences[i + 1] / prominence_threshold),
            "low_prominence_quality_flag": bool(
                cycles["low_prominence_peak"][i] or cycles["low_prominence_peak"][i + 1]
            ),
            "used_for_label": bool(valid),
            "exclusion_reason": "" if valid else str(reason),
        })
    return rows


def _window_metadata(
    subject_id: str,
    sample_index: int,
    source_row_index: int,
    start_s: float,
    label_bpm: float,
    coverage_s: float,
    cycles: dict[str, np.ndarray],
    candidate_index: int,
    activity_codes: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_sample_64 = int(round(start_s * PPG_FS))
    start_sample_125 = int(round(start_s * TARGET_FS))
    resp_start_25 = int(round(start_s * RESP_WORK_FS))
    resp_start_700 = int(round(start_s * RESP_FS))
    overlapping = np.flatnonzero(
        (cycles["peak_times_s"][:-1] < start_s + WINDOW_SECONDS)
        & (cycles["peak_times_s"][1:] > start_s)
        & cycles["valid"]
    )
    if len(overlapping) == 0:
        raise AssertionError("A retained sample must overlap at least one valid respiration cycle")
    activity_codes = np.asarray(activity_codes, dtype=np.float64).reshape(-1)
    unique_activity, activity_counts = np.unique(activity_codes, return_counts=True)
    low_prominence_cycles = (
        cycles["low_prominence_peak"][overlapping]
        | cycles["low_prominence_peak"][overlapping + 1]
    )
    sample_id = f"{subject_id}_rr10s_{start_sample_64:07d}"
    sample = {
        "sample_id": sample_id,
        "source_id": subject_id,
        "subject_id": subject_id,
        "sample_index": int(sample_index),
        "source_row_in_split": int(source_row_index),
        "candidate_window_index": int(candidate_index),
        "source_native_fs_hz": PPG_FS,
        "source_start_sample": start_sample_64,
        "source_end_sample_exclusive": start_sample_64 + PPG_FS * WINDOW_SECONDS,
        "input_start_time_s": float(start_s),
        "input_end_time_s": float(start_s + WINDOW_SECONDS),
        "target_start_time_s": float(start_s),
        "target_end_time_s": float(start_s + WINDOW_SECONDS),
        "resp_source_fs_hz": RESP_FS,
        "resp_source_start_sample": resp_start_700,
        "resp_source_end_sample_exclusive": resp_start_700 + RESP_FS * WINDOW_SECONDS,
        "resp_work_fs_hz": RESP_WORK_FS,
        "resp_work_start_sample": resp_start_25,
        "resp_work_end_sample_exclusive": resp_start_25 + RESP_WORK_FS * WINDOW_SECONDS,
        "label_breaths_per_minute": float(label_bpm),
        "label_semantics": "time-weighted mean of 60/period over complete valid positive-peak Resp cycle intervals covering the entire 10-second target window",
        "cycle_count_overlapping_window": int(len(overlapping)),
        "low_prominence_cycle_count": int(np.sum(low_prominence_cycles)),
        "low_prominence_cycle_fraction": float(np.mean(low_prominence_cycles)),
        "low_prominence_quality_flag": bool(np.any(low_prominence_cycles)),
        "first_cycle_index": int(overlapping[0]),
        "last_cycle_index": int(overlapping[-1]),
        "valid_coverage_seconds": float(coverage_s),
        "coverage_fraction": float(coverage_s / WINDOW_SECONDS),
        "activity_sampling_rate_hz": ACTIVITY_FS,
        "activity_codes_observed": json.dumps(unique_activity.astype(int).tolist()),
        "activity_code_counts": json.dumps(activity_counts.astype(int).tolist()),
        "activity_nonzero_fraction": float(np.mean(activity_codes != 0.0)),
        "activity_nonzero_quality_flag": bool(np.any(activity_codes != 0.0)),
    }
    window = {
        "sample_index": int(sample_index),
        "window_index": 0,
        "source_id": subject_id,
        "source_native_fs_hz": PPG_FS,
        "source_start_sample": start_sample_64,
        "source_end_sample_exclusive": start_sample_64 + PPG_FS * WINDOW_SECONDS,
        "window_start_time_s": float(start_s),
        "window_end_time_s": float(start_s + WINDOW_SECONDS),
        "valid_length": WINDOW_SAMPLES,
        "resampled_start_sample_125hz": start_sample_125,
    }
    return sample, window


def _subject_stats(
    subject_id: str,
    resp_filtered: np.ndarray,
    prominence: float,
    cycles: dict[str, np.ndarray],
    windows: dict[str, Any],
    source_duration_s: float,
) -> dict[str, Any]:
    labels = windows["labels_bpm"]
    reason_counts = {
        str(reason): int(np.sum(cycles["reason"] == reason))
        for reason in np.unique(cycles["reason"])
    }
    period = cycles["period_s"]
    low_prominence_cycle = (
        cycles["low_prominence_peak"][:-1] | cycles["low_prominence_peak"][1:]
    ) if len(period) else np.empty(0, dtype=bool)
    return {
        "subject_id": subject_id,
        "duration_seconds": float(source_duration_s),
        "filtered_resp_robust_sigma": robust_sigma(resp_filtered),
        "peak_prominence_threshold": float(prominence),
        "detected_positive_peaks": int(len(cycles["peak_indices"])),
        "candidate_cycle_intervals": int(len(period)),
        "valid_cycle_intervals": int(np.sum(cycles["valid"])),
        "excluded_cycle_intervals": int(np.sum(~cycles["valid"])),
        "excluded_cycles_by_reason": reason_counts,
        "low_prominence_cycle_intervals": int(np.sum(low_prominence_cycle)),
        "low_prominence_valid_cycle_intervals": int(np.sum(low_prominence_cycle & cycles["valid"])),
        "candidate_windows": windows["counts"]["candidate_windows"],
        "retained_windows": windows["counts"]["retained_windows"],
        "excluded_filter_edge_windows": windows["counts"]["excluded_filter_edge_windows"],
        "excluded_invalid_cycle_windows": windows["counts"]["excluded_invalid_cycle_windows"],
        "excluded_uncovered_or_undetected_windows": windows["counts"]["excluded_uncovered_or_undetected_windows"],
        "retained_fraction": float(len(labels) / max(1, windows["counts"]["candidate_windows"])),
        "cycle_period_seconds_percentiles": (
            np.percentile(period[cycles["valid"]], [1, 5, 25, 50, 75, 95, 99]).tolist()
            if np.any(cycles["valid"]) else []
        ),
        "label_count": int(len(labels)),
        "label_distribution_bpm": (
            {
                "mean": float(np.mean(labels)), "std": float(np.std(labels)),
                "min": float(np.min(labels)), "max": float(np.max(labels)),
                "count_above_30_bpm": int(np.sum(labels > 30.0)),
                "fraction_above_30_bpm": float(np.mean(labels > 30.0)),
                "percentiles_1_5_25_50_75_95_99": np.percentile(
                    labels, [1, 5, 25, 50, 75, 95, 99]
                ).tolist(),
                "unique_values_rounded_6dp": int(len(np.unique(np.round(labels, 6)))),
            } if len(labels) else None
        ),
    }


def source_paths() -> dict[str, Path]:
    return {
        subject: SOURCE_ROOT / subject / f"{subject}.pkl"
        for ids in SPLITS.values() for subject in ids
    }


def verify_split_identity() -> dict[str, Any]:
    intersections = {
        a: {b: sorted(set(SPLITS[a]) & set(SPLITS[b])) for b in SPLITS if a != b}
        for a in SPLITS
    }
    if any(values for per_split in intersections.values() for values in per_split.values()):
        raise AssertionError(f"Subject leakage across splits: {intersections}")
    return {
        "package_protocol_manifest": str(PROTOCOL_PATH.relative_to(PACKAGE_ROOT)),
        "package_split_file": str(SPLIT_FILE.relative_to(PACKAGE_ROOT)),
        "package_split_sha256": SPLIT_PROTOCOL["split_file_sha256"],
        "exact_package_split_identity": True,
        "split_membership": SPLITS,
        "split_subject_intersections": intersections,
    }


def process_subject(subject_id: str) -> tuple[np.ndarray, list[float], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = SOURCE_ROOT / subject_id / f"{subject_id}.pkl"
    print(f"LOAD {subject_id} {path}", flush=True)
    data = load_pickle(subject_id)
    bvp = signal_1d(data["signal"]["wrist"]["BVP"], f"{subject_id} BVP")
    resp = signal_1d(data["signal"]["chest"]["Resp"], f"{subject_id} Resp")
    activity = signal_1d(data["activity"], f"{subject_id} activity")
    bvp_duration = len(bvp) / PPG_FS
    resp_duration = len(resp) / RESP_FS
    if not np.isclose(bvp_duration, resp_duration, rtol=0.0, atol=1e-10):
        raise AssertionError(
            f"{subject_id} BVP/Resp time axes differ: {bvp_duration} vs {resp_duration}s"
        )
    if not float(bvp_duration).is_integer():
        raise AssertionError(f"Expected integer-second synchronized record, got {bvp_duration}s")
    activity_duration = len(activity) / ACTIVITY_FS
    if not np.isclose(activity_duration, bvp_duration, rtol=0.0, atol=1.0 / ACTIVITY_FS):
        raise AssertionError(
            f"{subject_id} activity/BVP durations differ: {activity_duration} vs {bvp_duration}s"
        )

    resp_work, resp_filtered, prominence, _sos = prepare_resp(resp)
    cycles = detect_cycles(resp_filtered, prominence)
    win = integrate_windows(cycles, bvp_duration)
    if len(win["labels_bpm"]) == 0:
        raise ValueError(f"{subject_id} has no complete valid 10-second RR labels")

    bvp_125 = common.resample_signal(bvp, PPG_FS)
    expected_samples = int(round(bvp_duration * TARGET_FS))
    if len(bvp_125) != expected_samples:
        raise AssertionError(f"{subject_id}: resampled BVP length {len(bvp_125)} != {expected_samples}")
    start_indices = np.rint(win["starts_s"] * TARGET_FS).astype(np.int64)
    windows_x = np.stack([
        bvp_125[int(start):int(start) + WINDOW_SAMPLES] for start in start_indices
    ]).astype(np.float32, copy=False)
    if windows_x.shape != (len(start_indices), 1, WINDOW_SAMPLES):
        windows_x = windows_x.reshape(-1, 1, WINDOW_SAMPLES)
    if not np.isfinite(windows_x).all():
        raise AssertionError(f"{subject_id}: BVP windows contain non-finite values")

    sample_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for i, (start, label, coverage, candidate_index) in enumerate(zip(
        win["starts_s"], win["labels_bpm"], win["coverage_s"], win["candidate_indices"]
    )):
        activity_start = int(round(float(start) * ACTIVITY_FS))
        activity_end = activity_start + ACTIVITY_FS * WINDOW_SECONDS
        sample, window = _window_metadata(
            subject_id, i, i, float(start), float(label), float(coverage), cycles,
            int(candidate_index), activity[activity_start:activity_end],
        )
        sample_rows.append(sample)
        window_rows.append(window)
    cycle_rows = _cycle_rows(subject_id, cycles, prominence)
    stats = _subject_stats(subject_id, resp_filtered, prominence, cycles, win, bvp_duration)
    activity_codes, activity_counts = np.unique(activity.astype(np.int64), return_counts=True)
    stats.update({
        "activity_sampling_rate_hz": ACTIVITY_FS,
        "activity_codes_count": {
            str(int(k)): int(v) for k, v in zip(activity_codes, activity_counts)
        },
        "retained_windows_with_nonzero_activity": int(sum(
            bool(row["activity_nonzero_quality_flag"]) for row in sample_rows
        )),
        "retained_windows_with_low_prominence_cycle": int(sum(
            bool(row["low_prominence_quality_flag"]) for row in sample_rows
        )),
        "retained_windows_with_both_quality_flags": int(sum(
            bool(row["activity_nonzero_quality_flag"] and row["low_prominence_quality_flag"])
            for row in sample_rows
        )),
        "quality_flags_are_descriptive_only": True,
        "source_path": str(path),
        "source_size_bytes": int(path.stat().st_size),
        "bvp_source_samples": int(len(bvp)),
        "resp_source_samples": int(len(resp)),
        "resp_downsampled_samples": int(len(resp_work)),
        "bvp_resampled_samples_125hz": int(len(bvp_125)),
        "native_time_axes_synchronized": True,
    })
    # Keep the full Resp arrays only in pilot mode; full processing needs only
    # the tabular cycle records and resulting labels/features.
    del data, bvp, resp, activity, resp_work, resp_filtered, bvp_125
    return windows_x, win["labels_bpm"].tolist(), sample_rows, window_rows, cycle_rows, stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats_distribution(labels: np.ndarray) -> dict[str, Any]:
    values = np.asarray(labels, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count_above_30_bpm": int(np.sum(values > 30.0)),
        "fraction_above_30_bpm": float(np.mean(values > 30.0)),
        "percentiles_1_5_25_50_75_95_99": np.percentile(
            values, [1, 5, 25, 50, 75, 95, 99]
        ).tolist(),
        "unique_values_rounded_6dp": int(len(np.unique(np.round(values, 6)))),
    }


def protocol_manifest(split_identity: dict[str, Any]) -> dict[str, Any]:
    paths = source_paths()
    source_info = []
    for subject_id in [sid for members in SPLITS.values() for sid in members]:
        path = paths[subject_id]
        if not path.is_file():
            raise FileNotFoundError(path)
        source_info.append({
            "subject_id": subject_id,
            "split": next(split for split, ids in SPLITS.items() if subject_id in ids),
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        })
        print(f"HASH {subject_id} {source_info[-1]['sha256']}", flush=True)
    return {
        "dataset": "PPG-DaLiA",
        "task": "RR regression from one 10-second wrist BVP window",
        "created_utc": now_utc(),
        "complete": False,
        "sampling_rate_hz": TARGET_FS,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "use_mask": False,
        "downstream_mask": False,
        "normalization": "Stored waveform values preserve source amplitude; the downstream loader applies independent per-window min-max scaling.",
        "input_signal": {
            "path": "signal/wrist/BVP",
            "native_sampling_rate_hz": PPG_FS,
            "output_sampling_rate_hz": TARGET_FS,
            "channel_count": 1,
            "resampling": "scipy.signal.resample_poly; up=125, down=64; Kaiser beta=5; constant boundary padding",
            "normalization": "No offline normalization; consumer loader performs independent per-window min-max to [0,1]",
            "band_filtering": "No additional BVP filtering",
        },
        "reference_signal": {
            "path": "signal/chest/Resp",
            "native_sampling_rate_hz": RESP_FS,
            "working_sampling_rate_hz": RESP_WORK_FS,
            "time_alignment": "Raw BVP and Resp are required to have exactly equal durations; official RR v1 has both windows aligned at integer-second offsets.",
            "resampling": "scipy.signal.resample_poly; up=1, down=28; Kaiser beta=5; constant boundary padding",
            "filter": {
                "design": "scipy.signal.butter SOS bandpass, order 4",
                "passband_hz": [RESP_FILTER_LOW_HZ, RESP_FILTER_HIGH_HZ],
                "application": "one continuous full-record scipy.signal.sosfiltfilt per subject, zero-phase",
                "edge_exclusion_seconds": FILTER_EDGE_EXCLUSION_SECONDS,
                "edge_rule": "A cycle is invalid if either endpoint peak lies within 12.5s of a record endpoint; a target window is excluded when it starts before 12.5s or ends later than duration−12.5s.",
            },
            "cycle_detector": {
                "event": "positive local maxima of the filtered Resp waveform (same polarity for all subjects)",
                "algorithm": "scipy.signal.find_peaks",
                "prominence": "0.35 × 1.4826 × median(abs(filtered_resp − median(filtered_resp))); scale is computed once over the subject's complete filtered Resp record",
                "minimum_peak_distance_seconds": PEAK_MIN_DISTANCE_SECONDS,
                "qa_only_flags": {
                    "low_prominence_multiplier": LOW_PROMINENCE_QA_MULTIPLIER,
                    "activity_sampling_rate_hz": ACTIVITY_FS,
                    "meaning": "A cycle is flagged low-prominence if either endpoint peak is under 1.5× the detection threshold. A sample is flagged active if any aligned 4 Hz source activity code is nonzero. Both flags are descriptive only and never exclude samples.",
                },
                "cycle_definition": "time interval between consecutive accepted positive peaks",
                "valid_cycle_period_seconds_inclusive": [PERIOD_MIN_SECONDS, PERIOD_MAX_SECONDS],
                "invalid_cycles": "short/long periods, or any period touching a 12.5s filter-edge exclusion zone; they contribute zero coverage and labels crossing them are rejected",
            },
            "target_label": {
                "unit": "breaths/minute",
                "formula": "For each accepted cycle k, r_k=60/(t[k+1]-t[k]); label for [a,a+10s) is sum_k(overlap_duration_k × r_k)/10s.",
                "coverage_requirement": "Valid cycle intervals must cover the full [start,end) interval within 1e-8 seconds. The visible-period weights are never renormalized over partial coverage.",
                "windowing": "candidate windows start at integer seconds; each stored label is one sample, one BVP window, no padding",
                "quality_note": "This is a cycle-derived reference from chest Resp, not a claim of error-free physiological ground truth; peak-detection omissions and Resp artifacts can bias it.",
            },
        },
        "split_protocol": SPLIT_PROTOCOL["task_protocol"]["split_protocol"],
        "split_metadata_file": str(SPLIT_FILE.relative_to(PACKAGE_ROOT)),
        "split_metadata_sha256": SPLIT_PROTOCOL["split_file_sha256"],
        "split_identity_audit": split_identity,
        "split_membership": SPLITS,
        "source_files": source_info,
        "label_units": "breaths/minute",
        "features_layout": "[N,1,1250]; each sample contains exactly one 10-second window",
        "label_layout": "[N]; one continuous cycle-integrated target per window",
        "window_offsets": "arange(N+1), so each label maps to exactly one feature window",
        "valid_lengths": "all entries are 1250; retained for loader compatibility; no masking",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "builder_sha256": sha256(Path(__file__).resolve()),
        },
    }


def _scalar_integral_reference(
    start_s: float, end_s: float, starts: np.ndarray, ends: np.ndarray,
    rates: np.ndarray, valid: np.ndarray,
) -> tuple[float, float]:
    coverage = 0.0
    integral = 0.0
    for left, right, rate, is_valid in zip(starts, ends, rates, valid):
        if not is_valid:
            continue
        overlap = max(0.0, min(float(right), end_s) - max(float(left), start_s))
        coverage += overlap
        integral += overlap * float(rate)
    return coverage, integral


def synthetic_checks() -> dict[str, Any]:
    """Known-rate and changing-rate signals exercise detection and integration."""
    duration = 180.0
    t = np.arange(int(duration * RESP_FS), dtype=np.float64) / RESP_FS
    known_hz = 0.25  # 15 breaths/min, four-second cycles.
    known = np.sin(2.0 * np.pi * known_hz * t)
    _, known_filtered, known_prom, _ = prepare_resp(known)
    known_cycles = detect_cycles(known_filtered, known_prom)
    known_windows = integrate_windows(known_cycles, duration)
    known_labels = known_windows["labels_bpm"]
    if len(known_labels) < 100 or np.max(np.abs(known_labels - 60.0 * known_hz)) > 0.08:
        raise AssertionError(
            f"Known 15 bpm synthetic Resp failed: count={len(known_labels)}, "
            f"max_abs_error={np.max(np.abs(known_labels - 15.0)) if len(known_labels) else None}"
        )

    freq = np.where(t < 60.0, 0.20, np.where(t < 120.0, 0.30, 0.15))
    phase = 2.0 * np.pi * np.cumsum(freq) / RESP_FS
    varying = np.sin(phase)
    _, varying_filtered, varying_prom, _ = prepare_resp(varying)
    varying_cycles = detect_cycles(varying_filtered, varying_prom)
    varying_windows = integrate_windows(varying_cycles, duration)
    vs, ve = varying_cycles["peak_times_s"][:-1], varying_cycles["peak_times_s"][1:]
    vr, vv = varying_cycles["rate_bpm"], varying_cycles["valid"]
    max_scalar_error = 0.0
    scalar_checks = 0
    for start, end, value in zip(
        varying_windows["starts_s"], varying_windows["ends_s"], varying_windows["labels_bpm"]
    ):
        cov, integ = _scalar_integral_reference(start, end, vs, ve, vr, vv)
        if abs(cov - 10.0) > COVERAGE_TOLERANCE_SECONDS:
            raise AssertionError("Synthetic variable-rate reference window lacks full cycle coverage")
        error = abs(float(value) - integ / 10.0)
        max_scalar_error = max(max_scalar_error, error)
        scalar_checks += 1
    if scalar_checks < 100 or max_scalar_error > 1e-10:
        raise AssertionError(f"Vector integration differs from independent cycle loop: {max_scalar_error}")
    early = varying_windows["labels_bpm"][varying_windows["starts_s"] < 45]
    middle = varying_windows["labels_bpm"][(varying_windows["starts_s"] >= 80) & (varying_windows["starts_s"] < 105)]
    late = varying_windows["labels_bpm"][varying_windows["starts_s"] >= 135]
    if not (np.mean(early) < np.mean(middle) and np.mean(middle) > np.mean(late)):
        raise AssertionError("Synthetic label did not follow the lower/higher/lower RR pattern")
    return {
        "known_constant_rr": {
            "input_breaths_per_minute": 15.0,
            "detected_peaks": int(len(known_cycles["peak_indices"])),
            "retained_windows": int(len(known_labels)),
            "label_mean_bpm": float(np.mean(known_labels)),
            "label_max_abs_error_bpm": float(np.max(np.abs(known_labels - 15.0))),
        },
        "piecewise_varying_rr": {
            "frequency_segments_hz": [[0, 60, 0.20], [60, 120, 0.30], [120, 180, 0.15]],
            "retained_windows": int(len(varying_windows["labels_bpm"])),
            "early_mean_label_bpm": float(np.mean(early)),
            "middle_mean_label_bpm": float(np.mean(middle)),
            "late_mean_label_bpm": float(np.mean(late)),
            "independent_scalar_integral_checks": scalar_checks,
            "maximum_vector_vs_scalar_error_bpm": float(max_scalar_error),
        },
    }


def plot_s10_pilot(output_path: Path) -> dict[str, Any]:
    """Render fixed, training-only S10 zooms with aligned Resp and BVP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_pickle("S10")
    bvp = signal_1d(data["signal"]["wrist"]["BVP"], "S10 BVP")
    resp = signal_1d(data["signal"]["chest"]["Resp"], "S10 Resp")
    activity = signal_1d(data["activity"], "S10 activity")
    resp_work, filtered, prominence, _sos = prepare_resp(resp)
    cycles = detect_cycles(filtered, prominence)
    duration = len(bvp) / PPG_FS
    windows = integrate_windows(cycles, duration)
    # Use predetermined training-subject times. The middle examples span one
    # ordinary segment and one higher-amplitude segment; none come from val/test.
    fixed_starts = [15.0, 120.0, 3600.0]
    accepted_starts = windows["starts_s"]
    starts = []
    for fixed in fixed_starts:
        match = accepted_starts[(accepted_starts >= fixed) & (accepted_starts <= fixed + 10.0)]
        starts.append(float(match[0]) if len(match) else float(fixed))
    selected_labels = []
    selected = []
    for start in starts:
        ids = np.flatnonzero(np.isclose(windows["starts_s"], start, rtol=0.0, atol=1e-9))
        label = float(windows["labels_bpm"][ids[0]]) if len(ids) else float("nan")
        selected_labels.append(label)
        selected.append(ids[0] if len(ids) else -1)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=False, constrained_layout=True)
    for ax, start, label in zip(axes, starts, selected_labels):
        end = start + 30.0
        # Native Resp samples are shown at 700Hz; plotting all points keeps the
        # original reference visible while the filtered series is at 25Hz.
        r0 = max(0, int(np.floor(start * RESP_FS)))
        r1 = min(len(resp), int(np.ceil(end * RESP_FS)))
        rt = np.arange(r0, r1, dtype=np.float64) / RESP_FS
        f0 = max(0, int(np.floor(start * RESP_WORK_FS)))
        f1 = min(len(filtered), int(np.ceil(end * RESP_WORK_FS)))
        ft = np.arange(f0, f1, dtype=np.float64) / RESP_WORK_FS
        b0 = max(0, int(np.floor(start * PPG_FS)))
        b1 = min(len(bvp), int(np.ceil(end * PPG_FS)))
        bt = np.arange(b0, b1, dtype=np.float64) / PPG_FS
        ax.plot(rt, resp[r0:r1], color="#adb5bd", linewidth=0.45, alpha=0.65, label="raw chest Resp (700 Hz)")
        filtered_line, = ax.plot(ft, filtered[f0:f1], color="#264653", linewidth=1.25, label="filtered Resp (25 Hz)")
        peaks = cycles["peak_indices"]
        peak_mask = (peaks / RESP_WORK_FS >= start) & (peaks / RESP_WORK_FS <= end)
        psel = peaks[peak_mask]
        ax.scatter(
            psel / RESP_WORK_FS, filtered[psel], s=24, color="#e76f51",
            marker="o", zorder=5, label="positive Resp peaks",
        )
        low_psel = psel[cycles["low_prominence_peak"][peak_mask]]
        if len(low_psel):
            ax.scatter(
                low_psel / RESP_WORK_FS, filtered[low_psel], s=62, facecolors="none",
                edgecolors="#9d4edd", linewidths=1.2, marker="D", zorder=6,
                label="low-prominence QA flag (kept)",
            )
        # Color each complete cycle interval and label its 60/period rate.
        for ci in range(len(cycles["period_s"])):
            left = cycles["peak_times_s"][ci]
            right = cycles["peak_times_s"][ci + 1]
            if right <= start or left >= end:
                continue
            valid = bool(cycles["valid"][ci])
            color = "#2a9d8f" if valid else "#e63946"
            ax.axvspan(max(left, start), min(right, end), color=color, alpha=0.10)
            mid = (max(left, start) + min(right, end)) / 2.0
            if left >= start and right <= end:
                ax.text(mid, 0.97, f"{cycles['rate_bpm'][ci]:.1f}", color=color,
                        fontsize=7, ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.axvspan(start, start + WINDOW_SECONDS, color="#457b9d", alpha=0.18,
                   label="10 s labeled BVP window")
        ax.set_xlim(start, end)
        ax.set_ylabel("Resp (native amplitude)")
        ax.grid(alpha=0.18)
        ax.set_title(
            f"S10 training subject: {start:.0f}–{end:.0f}s; first 10s label={label:.2f} breaths/min"
        )
        twin = ax.twinx()
        twin.plot(bt, bvp[b0:b1], color="#6c757d", linewidth=0.65, alpha=0.72, label="aligned wrist BVP (64 Hz)")
        twin.set_ylabel("BVP (native amplitude)")
        a0 = max(0, int(np.floor(start * ACTIVITY_FS)))
        a1 = min(len(activity), int(np.ceil(end * ACTIVITY_FS)))
        at = np.arange(a0, a1, dtype=np.float64) / ACTIVITY_FS
        active = activity[a0:a1] != 0.0
        ax.scatter(
            at[active], np.full(int(np.sum(active)), 0.035), transform=ax.get_xaxis_transform(),
            marker="|", s=80, color="#d00000", linewidths=1.0,
            label="nonzero source activity code (kept)", zorder=7,
        )
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        if ax is axes[0]:
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Time from recording start (seconds)")
    fig.suptitle(
        f"S10 pilot cycle detector: 0.08–1.0 Hz, positive peak prominence {prominence:.3f} "
        f"(0.35 × robust MAD scale), accepted periods 1–12 s; green=valid, red=invalid",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "figure": str(output_path),
        "selected_window_starts_seconds": starts,
        "selected_10s_labels_breaths_per_minute": selected_labels,
        "figure_x_limits_seconds": [[s, s + 30.0] for s in starts],
        "figure_is_training_subject_only": True,
        "filtered_resp_sampling_rate_hz": RESP_WORK_FS,
        "raw_resp_sampling_rate_hz": RESP_FS,
        "bvp_sampling_rate_hz": PPG_FS,
        "peak_prominence_threshold": prominence,
        "candidate_window_counts": windows["counts"],
        "cycle_interval_count": int(len(cycles["period_s"])),
    }


def run_pilot() -> dict[str, Any]:
    """S10-only protocol pilot; writes no benchmark arrays."""
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_pickle("S10")
    bvp = signal_1d(data["signal"]["wrist"]["BVP"], "S10 BVP")
    resp = signal_1d(data["signal"]["chest"]["Resp"], "S10 Resp")
    activity = signal_1d(data["activity"], "S10 activity")
    del data
    if not np.isclose(len(bvp) / PPG_FS, len(resp) / RESP_FS, rtol=0.0, atol=1e-10):
        raise AssertionError("S10 raw BVP and chest Resp do not share a time axis")
    resp_work, filtered, prominence, sos = prepare_resp(resp)
    cycles = detect_cycles(filtered, prominence)
    windows = integrate_windows(cycles, len(bvp) / PPG_FS)
    stats = _subject_stats("S10", filtered, prominence, cycles, windows, len(bvp) / PPG_FS)
    activity_codes, activity_counts = np.unique(activity.astype(np.int64), return_counts=True)
    stats.update({
        "raw_bvp_samples": int(len(bvp)),
        "raw_resp_samples": int(len(resp)),
        "resp_work_samples": int(len(resp_work)),
        "raw_bvp_duration_s": len(bvp) / PPG_FS,
        "raw_resp_duration_s": len(resp) / RESP_FS,
        "bvp_resp_exact_duration_match": True,
        "sos_coefficients": sos.tolist(),
        "coverage_definition": "sum of overlaps with valid adjacent-positive-peak cycle intervals, with no renormalization; retained only if coverage >= 10s-1e-8s",
        "activity_sampling_rate_hz": ACTIVITY_FS,
        "activity_codes_count": {str(int(k)): int(v) for k, v in zip(activity_codes, activity_counts)},
        "low_prominence_valid_cycle_intervals": int(np.sum(
            (cycles["low_prominence_peak"][:-1] | cycles["low_prominence_peak"][1:]) & cycles["valid"]
        )),
        "low_prominence_flags_are_descriptive_only": True,
    })
    plot_info = plot_s10_pilot(PILOT_ROOT / "S10_training_cycle_QA.png")
    pilot = {
        "created_utc": now_utc(),
        "subject_id": "S10",
        "split": "train",
        "protocol_constants": {
            "resp_filter_band_hz": [RESP_FILTER_LOW_HZ, RESP_FILTER_HIGH_HZ],
            "resp_filter_order": RESP_FILTER_ORDER,
            "peak_type": "positive maxima",
            "prominence_formula": "0.35 × 1.4826 × median absolute deviation of the complete filtered Resp record",
            "period_bounds_seconds": [PERIOD_MIN_SECONDS, PERIOD_MAX_SECONDS],
            "filter_edge_exclusion_seconds": FILTER_EDGE_EXCLUSION_SECONDS,
            "candidate_window_step_seconds": STEP_SECONDS,
            "coverage_tolerance_seconds": COVERAGE_TOLERANCE_SECONDS,
        },
        "subject_pilot_summary": stats,
        "plot_details": plot_info,
        "synthetic_checks": synthetic_checks(),
        "interpretation": "This pilot validates synchronized timing, cycle coverage and label continuity on one training subject. It does not establish label accuracy or compare held-out task performance.",
    }
    common.write_json(PILOT_ROOT / "S10_pilot_report.json", pilot)
    del bvp, resp, activity, resp_work, filtered
    return pilot


def _import_downstream_loader(project_root: Path):
    source = project_root / "src" / "datamodule" / "frequency_downstream.py"
    spec = importlib.util.spec_from_file_location("lowfreq7_frequency_downstream", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_saved_dataset(
    split_stats: dict[str, Any],
    subject_stats: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    from collections import Counter

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The CMSSL environment must include torch for loader audit") from exc
    project_root = Path(os.environ.get("PHYSIOFSQ_PROJECT_ROOT", str(PACKAGE_ROOT))).expanduser()
    loader_module = _import_downstream_loader(project_root)
    hashes: dict[str, str] = {}
    split_subjects: dict[str, list[str]] = {}
    split_label_distributions: dict[str, Any] = {}
    loader_reload: dict[str, Any] = {}
    all_sample_sources: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        directory = OUTPUT_TASK
        x_path = directory / f"{split}_features.npy"
        y_path = directory / f"{split}_labels.npy"
        offsets_path = directory / f"{split}_window_offsets.npy"
        lengths_path = directory / f"{split}_valid_lengths.npy"
        x = np.load(x_path, mmap_mode="r", allow_pickle=False)
        y = np.load(y_path, mmap_mode="r", allow_pickle=False)
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        lengths = np.load(lengths_path, mmap_mode="r", allow_pickle=False)
        if x.shape != (len(y), 1, WINDOW_SAMPLES):
            raise AssertionError(f"{split} input shape invalid: {x.shape} / {y.shape}")
        if offsets.shape != (len(y) + 1,) or not np.array_equal(offsets, np.arange(len(y) + 1)):
            raise AssertionError(f"{split} must have exactly one 10s window per label")
        if lengths.shape != (len(y),) or not np.all(lengths == WINDOW_SAMPLES):
            raise AssertionError(f"{split} valid_lengths must all equal 1250")
        if not np.isfinite(y).all():
            raise AssertionError(f"{split} labels have non-finite values")
        for begin in range(0, len(x), 4096):
            if not np.isfinite(x[begin:begin + 4096]).all():
                raise AssertionError(f"{split} features have non-finite values")
        if len(np.unique(np.round(np.asarray(y), 6))) < 100:
            raise AssertionError(f"{split} RR labels collapsed to too few values")
        sample_table = __import__("pandas").read_csv(directory / f"{split}_samples.csv", dtype=str, keep_default_na=False)
        window_table = __import__("pandas").read_csv(directory / f"{split}_windows.csv")
        if len(sample_table) != len(y) or len(window_table) != len(y):
            raise AssertionError(f"{split} metadata counts disagree with arrays")
        if not np.array_equal(sample_table["sample_index"].astype(np.int64).to_numpy(), np.arange(len(y))):
            raise AssertionError(f"{split} sample_index does not follow saved row order")
        if not np.array_equal(window_table["sample_index"].to_numpy(dtype=np.int64), np.arange(len(y))):
            raise AssertionError(f"{split} window/sample mapping is not one-to-one")
        source_sequence = sample_table["source_id"].astype(str).tolist()
        sources = set(source_sequence)
        source_order = list(dict.fromkeys(source_sequence))
        all_sample_sources[split] = sources
        split_subjects[split] = source_order
        expected = set(SPLITS[split])
        if sources != expected:
            raise AssertionError(f"{split} sources {sources} != exact old sources {expected}")
        if source_order != SPLITS[split]:
            raise AssertionError(f"{split} source order {source_order} != old split order {SPLITS[split]}")
        labels_np = np.asarray(y, dtype=np.float64)
        split_label_distributions[split] = stats_distribution(labels_np)
        for file_path in [
            x_path, y_path, offsets_path, lengths_path,
            directory / f"{split}_samples.csv", directory / f"{split}_windows.csv",
        ]:
            hashes[str(file_path)] = sha256(file_path)
        dataset = loader_module.GroupedDownstreamDataset(
            DATA_ROOT, TASK, split, normalization="minmax"
        )
        if len(dataset) != len(y):
            raise AssertionError(f"{split} loader sample count mismatch")
        first = dataset[0]
        normalized = first["patches"].numpy()
        if normalized.shape != (1, 50, 25):
            raise AssertionError(f"{split} loaded patch shape {normalized.shape}")
        if not np.isfinite(normalized).all() or float(normalized.min()) < -1e-7 or float(normalized.max()) > 1.000001:
            raise AssertionError(f"{split} loader output min-max normalization invalid")
        if first["source_id"] not in expected:
            raise AssertionError(f"{split} sample loader source ID mismatch")
        loader_reload[split] = {
            "loaded_samples": int(len(dataset)),
            "first_sample_id": first["sample_id"],
            "first_source_id": first["source_id"],
            "first_patch_shape": list(normalized.shape),
            "first_patch_min": float(normalized.min()),
            "first_patch_max": float(normalized.max()),
            "finite": bool(np.isfinite(normalized).all()),
        }
        if int(split_stats[split]["samples"]) != len(y):
            raise AssertionError("Build and reload sample counts disagree")
        del dataset, first, x, y, offsets, lengths, labels_np
    for extra in [
        OUTPUT_TASK / "resp_cycles.csv",
        OUTPUT_TASK / "manifest.json",
        DATA_ROOT / "manifest.json",
        Path(__file__).resolve(),
        HERE / "dalia_rr10s_cycle_preprocessing.ipynb",
        HERE / "source_input_data_preprocessing.ipynb",
        HERE / "README.md",
    ]:
        if extra.is_file():
            hashes[str(extra)] = sha256(extra)
    intersections = {
        first: {second: sorted(all_sample_sources[first] & all_sample_sources[second])
                for second in all_sample_sources if first != second}
        for first in all_sample_sources
    }
    if any(v for d in intersections.values() for v in d.values()):
        raise AssertionError(f"Actual sample source leakage across splits: {intersections}")
    if split_subjects != SPLITS:
        # Preserve source-order identity as well as membership in the manifest;
        # sample metadata is naturally emitted in that order by construction.
        raise AssertionError(f"Actual subjects differ: {split_subjects}")
    return {
        "created_utc": now_utc(),
        "passed": True,
        "complete": True,
        "split_statistics": split_stats,
        "subjects_per_split": split_subjects,
        "actual_source_intersections": intersections,
        "label_distributions": split_label_distributions,
        "loader_end_to_end_reload": loader_reload,
        "file_sha256": hashes,
        "per_subject_quality": subject_stats,
        "protocol_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "checks": {
            "finite_features_and_labels": True,
            "one_window_per_label": True,
            "exact_train_val_test_subject_membership": True,
            "cross_split_source_intersection_empty": True,
            "continuous_label_cardinality": True,
            "consumer_loader_reloaded_all_splits": True,
            "consumer_patch_shape_and_normalization_valid": True,
            "no_padding_or_mask": True,
        },
    }


def reconstruct_subject_stats_from_saved(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Recover subject summaries from the completed arrays and CSV provenance."""
    import pandas as pd

    all_stats: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        split_info = manifest["split_statistics"][split]
        samples = pd.read_csv(OUTPUT_TASK / f"{split}_samples.csv", keep_default_na=False)
        labels = np.load(OUTPUT_TASK / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
        cycle_table = pd.read_csv(OUTPUT_TASK / "resp_cycles.csv", keep_default_na=False)
        for subject_id in SPLITS[split]:
            sample_rows = samples[samples["source_id"].astype(str) == subject_id]
            cycle_rows = cycle_table[cycle_table["subject_id"].astype(str) == subject_id]
            row_indices = sample_rows["sample_index"].to_numpy(dtype=np.int64)
            subject_labels = np.asarray(labels[row_indices], dtype=np.float64)
            valid_cycle = cycle_rows["used_for_label"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
            low_cycle = cycle_rows["low_prominence_quality_flag"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
            periods = cycle_rows["period_s"].to_numpy(dtype=np.float64)
            reasons = cycle_rows["exclusion_reason"].astype(str)
            reason_counts = {
                str(reason): int(count)
                for reason, count in reasons[reasons != ""].value_counts().items()
            }
            candidate_n = int(split_info["candidate_windows_by_subject"][subject_id])
            excluded_reasons = split_info["excluded_windows_by_reason_by_subject"][subject_id]
            source_info = next(
                entry for entry in manifest["source_files"] if entry["subject_id"] == subject_id
            )
            threshold_ratio = cycle_rows["start_prominence_threshold_ratio"].to_numpy(dtype=np.float64)
            start_prominence = cycle_rows["start_peak_prominence"].to_numpy(dtype=np.float64)
            threshold_est = float(np.median(start_prominence / threshold_ratio)) if len(cycle_rows) else None
            nonzero_activity = sample_rows["activity_nonzero_quality_flag"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
            low_prom_samples = sample_rows["low_prominence_quality_flag"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
            all_stats[subject_id] = {
                "subject_id": subject_id,
                "split": split,
                "source_path": source_info["path"],
                "source_size_bytes": int(source_info["size_bytes"]),
                "source_sha256": source_info["sha256"],
                "peak_prominence_threshold_reconstructed": threshold_est,
                "candidate_cycle_intervals": int(len(cycle_rows)),
                "detected_positive_peaks": int(len(cycle_rows) + 1),
                "valid_cycle_intervals": int(valid_cycle.sum()),
                "excluded_cycle_intervals": int((~valid_cycle).sum()),
                "excluded_cycles_by_reason": reason_counts,
                "low_prominence_cycle_intervals": int(low_cycle.sum()),
                "low_prominence_valid_cycle_intervals": int(np.sum(low_cycle & valid_cycle)),
                "cycle_period_seconds_percentiles": (
                    np.percentile(periods[valid_cycle], [1, 5, 25, 50, 75, 95, 99]).tolist()
                    if valid_cycle.any() else []
                ),
                "candidate_windows": candidate_n,
                "retained_windows": int(len(sample_rows)),
                "excluded_windows_by_reason": excluded_reasons,
                "retained_fraction": float(len(sample_rows) / max(1, candidate_n)),
                "label_distribution_bpm": stats_distribution(subject_labels),
                "retained_windows_with_nonzero_activity": int(nonzero_activity.sum()),
                "retained_windows_with_low_prominence_cycle": int(low_prom_samples.sum()),
                "retained_windows_with_both_quality_flags": int(np.sum(nonzero_activity & low_prom_samples)),
                "quality_flags_are_descriptive_only": True,
                "sample_metadata_rows": int(len(sample_rows)),
                "cycle_metadata_rows": int(len(cycle_rows)),
            }
        del samples, labels, cycle_table
    return all_stats


def write_root_manifest(complete: bool) -> dict[str, Any]:
    root_manifest = {
        "dataset": "PPG-DaLiA cycle-derived RR 10-second benchmark",
        "version": "dalia_rr10s_cycle_v1_20261026",
        "complete": bool(complete),
        "sampling_rate_hz": TARGET_FS,
        "window_samples": WINDOW_SAMPLES,
        "use_mask": False,
        "tasks": {TASK: str(OUTPUT_TASK)},
        "task_manifests": {TASK: str(OUTPUT_TASK / "manifest.json")},
        "dataset_audit": str(OUTPUT_TASK / "DATASET_AUDIT.json"),
    }
    common.write_json(DATA_ROOT / "manifest.json", root_manifest)
    return root_manifest


def audit_existing_release() -> dict[str, Any]:
    """Finish manifest and consumer-loader audit after an interrupted build tail."""
    manifest_path = OUTPUT_TASK / "manifest.json"
    if not manifest_path.is_file() or not (OUTPUT_TASK / "resp_cycles.csv").is_file():
        raise FileNotFoundError("No completed arrays/cycle table are available to audit")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True or set(manifest.get("split_statistics", {})) != set(SPLITS):
        raise ValueError("Saved build manifest is incomplete")
    manifest.setdefault("software", {})["builder_sha256"] = sha256(Path(__file__).resolve())
    manifest["dataset_audit_path"] = str(OUTPUT_TASK / "DATASET_AUDIT.json")
    common.write_json(manifest_path, manifest)
    write_root_manifest(complete=True)
    subjects = reconstruct_subject_stats_from_saved(manifest)
    audit = audit_saved_dataset(manifest["split_statistics"], subjects, manifest)
    common.write_json(OUTPUT_TASK / "DATASET_AUDIT.json", audit)
    common.write_json(HERE / "DATASET_AUDIT.json", audit)
    common.write_json(HERE / "build_report.json", {
        "created_utc": now_utc(),
        "complete": True,
        "output_root": str(DATA_ROOT),
        "task_output": str(OUTPUT_TASK),
        "audit_path": str(OUTPUT_TASK / "DATASET_AUDIT.json"),
        "split_statistics": manifest["split_statistics"],
        "dataset_audit_passed": bool(audit["passed"]),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "task_manifest_sha256": sha256(manifest_path),
        "root_manifest_sha256": sha256(DATA_ROOT / "manifest.json"),
    })
    return audit


def run_build() -> dict[str, Any]:
    if OUTPUT_TASK.exists():
        raise FileExistsError(f"Refusing to overwrite new task output: {OUTPUT_TASK}")
    split_identity = verify_split_identity()
    synthetic = synthetic_checks()
    print("SYNTHETIC_CHECKS_PASSED", json.dumps(synthetic, ensure_ascii=False), flush=True)
    manifest = protocol_manifest(split_identity)
    OUTPUT_TASK.mkdir(parents=True, exist_ok=False)
    common.write_json(OUTPUT_TASK / "manifest.json", manifest)
    code_manifest = {
        "created_utc": now_utc(),
        "protocol_manifest_path": str(OUTPUT_TASK / "manifest.json"),
        "split_identity": split_identity,
        "synthetic_checks": synthetic,
        "split_statistics": {},
        "per_subject": {},
        "cycle_table_path": str(OUTPUT_TASK / "resp_cycles.csv"),
    }
    overall_cycles: list[dict[str, Any]] = []
    split_stats: dict[str, Any] = {}
    all_subject_stats: dict[str, dict[str, Any]] = {}
    for split, subject_ids in SPLITS.items():
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        samples: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []
        old_cycle_start = len(overall_cycles)
        for subject_id in subject_ids:
            x, y, subject_samples, subject_windows, cycles, stats = process_subject(subject_id)
            row_start = len(samples)
            for local_index, sample in enumerate(subject_samples):
                sample["sample_index"] = row_start + local_index
                sample["source_row_in_split"] = row_start + local_index
            for local_index, window in enumerate(subject_windows):
                window["sample_index"] = row_start + local_index
            x_parts.append(x)
            y_parts.append(np.asarray(y, dtype=np.float64))
            samples.extend(subject_samples)
            windows.extend(subject_windows)
            stats["split"] = split
            all_subject_stats[subject_id] = stats
            overall_cycles.extend(cycles)
            print(
                f"DONE {split}/{subject_id}: samples={stats['retained_windows']}/{stats['candidate_windows']} "
                f"cycles={stats['valid_cycle_intervals']}/{stats['candidate_cycle_intervals']} "
                f"mean_rr={stats['label_distribution_bpm']['mean']:.3f}", flush=True,
            )
            del x, y, subject_samples, subject_windows, cycles
        features = np.concatenate(x_parts, axis=0).astype(np.float32, copy=False)
        labels = np.concatenate(y_parts, axis=0).astype(np.float64, copy=False)
        offsets = np.arange(len(labels) + 1, dtype=np.int64)
        lengths = np.full(len(features), WINDOW_SAMPLES, dtype=np.int16)
        saved = common.save_split(
            TASK, split, features, labels, samples, windows,
            offsets=offsets, valid_lengths=lengths,
        )
        saved["subjects"] = subject_ids.copy()
        saved["samples_by_subject"] = {
            sid: int(all_subject_stats[sid]["retained_windows"]) for sid in subject_ids
        }
        saved["candidate_windows_by_subject"] = {
            sid: int(all_subject_stats[sid]["candidate_windows"]) for sid in subject_ids
        }
        saved["excluded_windows_by_reason_by_subject"] = {
            sid: {
                "filter_edge": int(all_subject_stats[sid]["excluded_filter_edge_windows"]),
                "invalid_cycle": int(all_subject_stats[sid]["excluded_invalid_cycle_windows"]),
                "uncovered_or_undetected": int(all_subject_stats[sid]["excluded_uncovered_or_undetected_windows"]),
            } for sid in subject_ids
        }
        saved["label_distribution_bpm"] = stats_distribution(labels)
        saved["labels_unique_rounded_6dp"] = int(len(np.unique(np.round(labels, 6))))
        split_stats[split] = saved
        code_manifest["split_statistics"][split] = saved
        del x_parts, y_parts, features, labels, offsets, lengths, samples, windows
        print(f"SAVED_SPLIT {split}: {saved}", flush=True)

    # One cycle table keeps every accepted and excluded interval traceable.
    write_csv(OUTPUT_TASK / "resp_cycles.csv", overall_cycles)
    manifest.update({
        "complete": True,
        "created_utc": now_utc(),
        "protocol_frozen_after_training_subject_S10_pilot": True,
        "split_statistics": split_stats,
        "total_samples": int(sum(v["samples"] for v in split_stats.values())),
        "total_windows": int(sum(v["windows"] for v in split_stats.values())),
        "total_cycle_intervals": int(len(overall_cycles)),
        "cycle_table": "resp_cycles.csv",
        "synthetic_checks": synthetic,
    })
    manifest["dataset_audit_path"] = str(OUTPUT_TASK / "DATASET_AUDIT.json")
    common.write_json(OUTPUT_TASK / "manifest.json", manifest)
    code_manifest["per_subject"] = all_subject_stats
    # The consumer loader requires a completed root manifest. Publish this only
    # after all arrays and the task manifest are present, then execute its full
    # end-to-end reload audit.
    write_root_manifest(complete=True)
    audit = audit_saved_dataset(split_stats, all_subject_stats, manifest)
    common.write_json(OUTPUT_TASK / "DATASET_AUDIT.json", audit)
    common.write_json(HERE / "DATASET_AUDIT.json", audit)
    common.write_json(HERE / "build_report.json", {
        "created_utc": now_utc(),
        "complete": True,
        "output_root": str(DATA_ROOT),
        "task_output": str(OUTPUT_TASK),
        "audit_path": str(OUTPUT_TASK / "DATASET_AUDIT.json"),
        "split_statistics": split_stats,
        "dataset_audit_passed": bool(audit["passed"]),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "task_manifest_sha256": sha256(OUTPUT_TASK / "manifest.json"),
        "root_manifest_sha256": sha256(DATA_ROOT / "manifest.json"),
    })
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run the S10-only pilot and QA plot")
    parser.add_argument("--build", action="store_true", help="Build the complete three-split dataset")
    parser.add_argument("--synthetic-check", action="store_true", help="Run synthetic cycle/integration checks")
    parser.add_argument("--audit-existing", action="store_true", help="Audit an already-written complete dataset without rebuilding arrays")
    args = parser.parse_args()
    if args.pilot:
        result = run_pilot()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.synthetic_check:
        print(json.dumps(synthetic_checks(), ensure_ascii=False, indent=2))
        return 0
    if args.build:
        result = run_build()
        print(json.dumps({"passed": result["passed"], "split_statistics": result["split_statistics"]}, ensure_ascii=False, indent=2))
        return 0
    if args.audit_existing:
        result = audit_existing_release()
        print(json.dumps({"passed": result["passed"], "split_statistics": result["split_statistics"]}, ensure_ascii=False, indent=2))
        return 0
    parser.error("choose --pilot, --synthetic-check, --build, or --audit-existing")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
