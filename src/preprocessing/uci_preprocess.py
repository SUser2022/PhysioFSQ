#!/usr/bin/env python3
"""Materialize a previously verified UCI 10 s PPG product in the common contract.

This adapter reuses a complete verified product. It checks the original Part
MAT files by identity and SHA-256, but does not rebuild the product from MAT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import common

REPORT_ROOT = common.CODE_ROOT
DATA_ROOT = common.DATA_ROOT
RAW_SOURCE = Path(os.environ.get('UCI_RAW_SOURCE_ROOT', str(common.RAW_ROOT / 'cuff_less_blood_pressure_estimation'))).expanduser()
VERIFIED_10S = Path(os.environ.get('UCI_VERIFIED_10S_ROOT', str(common.RAW_ROOT / 'cmssl_downstream_uci10s'))).expanduser()
PART_TO_SPLIT = {'1': 'train', '2': 'train', '3': 'val', '4': 'test'}
EXPECTED_COUNTS = {'train': 100803, 'val': 44587, 'test': 53694}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _part_hashes_by_identity(input_hashes: dict) -> dict[int, tuple[str, str]]:
    """Index historical absolute-path input hashes by the official Part file."""
    found: dict[int, tuple[str, str]] = {}
    for key, digest in input_hashes.items():
        name = str(key).replace('\\', '/').rsplit('/', 1)[-1]
        if not name.startswith('Part_') or not name.endswith('.mat'):
            continue
        number = name[len('Part_'):-len('.mat')]
        if number not in {'1', '2', '3', '4'}:
            continue
        part = int(number)
        value = (str(key), str(digest))
        previous = found.get(part)
        if previous is not None and previous[1] != value[1]:
            raise ValueError(f'Conflicting historical hashes for {name}: {previous[0]} and {key}')
        found[part] = value
    return found


def ensure_verified_uci_10s_product(
    raw_part_source_dir: Path = RAW_SOURCE,
    verified_product_dir: Path = VERIFIED_10S,
) -> tuple[Path, dict]:
    """Return a validated complete 10 s product and its external provenance."""
    raw_part_source_dir = Path(raw_part_source_dir).resolve()
    verified_product_dir = Path(verified_product_dir).resolve()
    manifest_path = verified_product_dir / 'manifest.json'
    verify_path = verified_product_dir / 'verification.json'
    if not manifest_path.is_file() or not verify_path.is_file():
        raise FileNotFoundError(f'UCI product needs manifest and full verification: {verified_product_dir}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    verification = json.loads(verify_path.read_text(encoding='utf-8'))
    if manifest.get('complete') is not True or manifest.get('is_smoke') is not False:
        raise ValueError('UCI source product is incomplete or marked as a smoke run')
    if verification.get('status') != 'ok' or verification.get('is_smoke') is not False:
        raise ValueError('UCI source product does not have a successful full verification')
    parameters = manifest.get('parameters', {})
    expected_parameters = {
        'sampling_rate_hz': 125,
        'window_seconds': 10,
        'window_samples': 1250,
        'window_stride_samples': 1250,
        'split_by_part': {'1': 'train', '2': 'train', '3': 'val', '4': 'test'},
    }
    for name, expected in expected_parameters.items():
        if parameters.get(name) != expected:
            raise ValueError(f'Unexpected UCI 10-second protocol parameter {name}: {parameters.get(name)!r}')
    if manifest.get('split_windows') != EXPECTED_COUNTS:
        raise ValueError(f'UCI manifest split counts are not the complete expected product: {manifest.get("split_windows")}')
    verified_counts = verification.get('count_accounting', {}).get('split_counts')
    if verified_counts != EXPECTED_COUNTS:
        raise ValueError(f'UCI verification split counts are not the complete expected product: {verified_counts}')
    records_seen = verification.get('count_accounting', {}).get('records_seen_by_part', {})
    if records_seen != {str(part): 3000 for part in range(1, 5)}:
        raise ValueError(f'UCI verification does not cover all 3,000 records in Parts 1-4: {records_seen}')
    # MAT locations are provenance, while Part number identifies each official
    # input. This lets the full verified product move without weakening its
    # raw-byte identity check.
    input_hashes = manifest.get('input_files_sha256', {})
    expected_by_part = _part_hashes_by_identity(input_hashes)
    raw_hash_audit = {}
    for part in range(1, 5):
        path = raw_part_source_dir / 'data' / f'Part_{part}.mat'
        if not path.is_file():
            raise FileNotFoundError(path)
        saved = expected_by_part.get(part)
        if saved is None:
            raise ValueError(f'UCI manifest has no historical SHA-256 for Part_{part}.mat')
        historical_key, expected = saved
        actual = sha256(path)
        raw_hash_audit[f'Part_{part}.mat'] = {
            'historical_manifest_key': historical_key,
            'current_source_path': str(path),
            'expected_sha256': expected,
            'current_sha256': actual,
            'matches': actual == expected,
        }
        if actual != expected:
            raise ValueError(f'Raw UCI source hash mismatch: {path}')

    output_path_provenance = {
        'manifest_recorded_output_dir': manifest.get('output_dir'),
        'verification_recorded_output_dir': verification.get('output_dir'),
        'current_output_dir': str(verified_product_dir),
        'historical_paths_are_provenance_only': True,
    }
    # The removed builder hashes describe how the external product was made.
    # This package does not contain that builder, so it records provenance
    # without claiming a local source-code comparison.
    expected_program_hashes = manifest.get('program_and_upstream_sha256', {})
    if not isinstance(expected_program_hashes, dict):
        raise ValueError('UCI build manifest program hashes must be a mapping')
    build_provenance = {
        'basis': 'historical manifest build snapshot',
        'upstream_repository': manifest.get('upstream_repository'),
        'upstream_commit': manifest.get('upstream_commit'),
        'historical_program_and_upstream_sha256': {},
        'local_source_comparison': 'not_performed; the raw MAT builder and upstream sources are not included',
        'historical_manifest_paths_are_not_current_paths': True,
    }
    for relative, expected in expected_program_hashes.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError(f'Unsafe path in UCI build manifest program hashes: {relative!r}')
        build_provenance['historical_program_and_upstream_sha256'][relative] = expected
    common.write_json(REPORT_ROOT / 'uci_build_provenance.json', build_provenance)
    return verified_product_dir, {
        'manifest': manifest, 'verification': verification,
        'build_provenance': build_provenance,
        'raw_input_hash_audit': raw_hash_audit,
        'output_path_provenance': output_path_provenance,
    }


def _read_metadata_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _verify_source_product(source_dir: Path, reports: dict) -> dict:
    manifest = reports['manifest']
    verification = reports['verification']
    observed_counts = {}
    part_counts = defaultdict(Counter)
    record_sets: dict[str, set[str]] = {split: set() for split in common.SPLITS}
    part_to_split = {}
    for split in common.SPLITS:
        source_meta = source_dir / 'metadata' / f'{split}.csv'
        rows = _read_metadata_rows(source_meta)
        observed_counts[split] = len(rows)
        if len(rows) != EXPECTED_COUNTS[split]:
            raise ValueError(f'{split}: expected {EXPECTED_COUNTS[split]} rows, found {len(rows)}')
        sample_ids = set()
        for row in rows:
            part = str(int(row['Part']))
            expected_split = PART_TO_SPLIT.get(part)
            if expected_split != split:
                raise ValueError(f'Part {part} row found in {split}, expected {expected_split}')
            if row['trial'] in sample_ids:
                raise ValueError(f'Duplicate UCI source window identifier in {split}: {row["trial"]}')
            sample_ids.add(row['trial'])
            raw_len = int(row['raw_record_length_samples'])
            if (int(row['ppg_source_start_sample']) < 0 or int(row['abp_source_start_sample']) < 0
                    or int(row['ppg_source_start_sample']) + 1250 > raw_len
                    or int(row['abp_source_start_sample']) + 1250 > raw_len):
                raise ValueError(f'Invalid source interval for UCI window {row["trial"]}')
            source_record = f"Part_{part}_{int(row['record_index'])}"
            record_sets[split].add(source_record)
            part_to_split[part] = split
            part_counts[split][part] += 1
        sbp = np.load(source_dir / 'uci_sbp' / f'{split}_labels.npy', mmap_mode='r', allow_pickle=False)
        dbp = np.load(source_dir / 'uci_dbp' / f'{split}_labels.npy', mmap_mode='r', allow_pickle=False)
        expected_sbp = np.asarray([float(row['SP']) for row in rows], dtype=np.float32)
        expected_dbp = np.asarray([float(row['DP']) for row in rows], dtype=np.float32)
        if (sbp.dtype != np.float32 or dbp.dtype != np.float32
                or sbp.shape != (observed_counts[split],) or dbp.shape != (observed_counts[split],)):
            raise ValueError(f'{split}: source BP labels have invalid shape or dtype')
        if (not np.isfinite(sbp).all() or not np.isfinite(dbp).all()
                or not np.array_equal(np.asarray(sbp), expected_sbp)
                or not np.array_equal(np.asarray(dbp), expected_dbp)):
            raise ValueError(f'{split}: source BP labels differ from the source metadata or are nonfinite')
        if np.any((sbp < 30) | (sbp > 220) | (dbp < 30) | (dbp > 220) | (sbp < dbp)):
            raise ValueError(f'{split}: source BP labels do not satisfy the documented mmHg limits')
    intersections = {
        f'{a}_{b}': sorted(record_sets[a] & record_sets[b])
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))
    }
    if any(intersections.values()):
        raise ValueError(f'Source records cross splits: {intersections}')
    if manifest.get('split_windows') != observed_counts:
        raise ValueError('Source manifest split counts disagree with the metadata rows')
    if verification.get('count_accounting', {}).get('split_counts') != observed_counts:
        raise ValueError('Source verification split counts disagree with metadata')
    expected_part_counts = {
        part: sum(counts.get(part, 0) for counts in part_counts.values())
        for part in ('1', '2', '3', '4')
    }
    verified_part_counts = verification.get('count_accounting', {}).get('part_counts', {})
    if {str(k): int(v) for k, v in verified_part_counts.items()} != expected_part_counts:
        raise ValueError('Source verification Part counts disagree with the metadata rows')
    expected_parts = {'train': {'1', '2'}, 'val': {'3'}, 'test': {'4'}}
    for split, parts in expected_parts.items():
        actual_parts = set(part_counts[split])
        if actual_parts != parts:
            raise ValueError(f'{split}: expected source Parts {sorted(parts)}, found {sorted(actual_parts)}')
    if manifest.get('parameters', {}).get('split_by_part') != {
        '1': 'train', '2': 'train', '3': 'val', '4': 'test'
    }:
        raise ValueError('UCI source manifest does not use Parts 1+2 / 3 / 4 for train/val/test')
    return {
        'split_counts': observed_counts,
        'part_counts': {split: dict(counts) for split, counts in part_counts.items()},
        'source_records_by_split': {split: len(ids) for split, ids in record_sets.items()},
        'source_record_intersections_between_splits': intersections,
        'part_to_split': part_to_split,
        'source_verification_status': verification['status'],
        'source_verification_path': str(source_dir / 'verification.json'),
        'sampling_rate_hz': 125,
        'window_seconds': 10,
        'window_samples': 1250,
        'label_units': 'mmHg',
    }


def _metadata_for_common(rows: list[dict[str, str]], split: str) -> tuple[list[dict], list[dict]]:
    samples, windows = [], []
    for i, row in enumerate(rows):
        part = int(row['Part'])
        record_index = int(row['record_index'])
        window_index = int(row['window_index'])
        sample_id = str(row['trial'])
        source_id = f'Part_{part}_{record_index}'
        ppg_start = int(row['ppg_source_start_sample'])
        abp_start = int(row['abp_source_start_sample'])
        ecg_start = int(row['ecg_source_start_sample'])
        sample = {
            'sample_id': sample_id, 'source_id': source_id, 'split': split,
            'source_part': part, 'source_record_index': record_index,
            'source_window_index': window_index, 'legacy_10s_row': i,
            'raw_record_length_samples': int(row['raw_record_length_samples']),
            'alignment_offset_samples': int(row['alignment_offset_samples']),
            'ppg_source_start_sample': ppg_start,
            'abp_source_start_sample': abp_start,
            'ecg_source_start_sample': ecg_start,
            'sampling_rate_hz': 125, 'window_seconds': 10,
            'sbp_mmhg': float(row['SP']), 'dbp_mmhg': float(row['DP']),
        }
        samples.append(sample)
        windows.append({
            'sample_index': i, 'window_index': 0,
            'source_part': part, 'source_record_index': record_index,
            'source_window_index': window_index,
            'ppg_source_start_sample': ppg_start,
            'ppg_source_end_sample': ppg_start + 1250,
            'abp_source_start_sample': abp_start,
            'ecg_source_start_sample': ecg_start,
            'start_time_s': window_index * 10.0,
            'end_time_s': (window_index + 1) * 10.0,
            'valid_length': 1250,
        })
    return samples, windows


def _compare_arrays(actual_path: Path, expected_path: Path, chunk_rows: int = 2048) -> dict:
    actual = np.load(actual_path, mmap_mode='r', allow_pickle=False)
    expected = np.load(expected_path, mmap_mode='r', allow_pickle=False)
    if actual.shape != expected.shape or actual.dtype != np.float32 or expected.dtype != np.float32:
        raise ValueError(f'Array shape/dtype mismatch: {actual_path} {actual.shape}/{actual.dtype} vs {expected_path} {expected.shape}/{expected.dtype}')
    max_abs = 0.0
    for start in range(0, len(actual), chunk_rows):
        a = np.asarray(actual[start:start + chunk_rows])
        b = np.asarray(expected[start:start + chunk_rows])
        if not np.isfinite(a).all() or not np.array_equal(a, b):
            raise ValueError(f'New UCI data differs or is nonfinite at rows {start}:{start + len(a)} in {actual_path}')
        if a.size:
            max_abs = max(max_abs, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    return {'shape': list(actual.shape), 'dtype': str(actual.dtype), 'exact_equal': True, 'max_abs_difference': max_abs}


def _audit_saved_metadata(task: str, split: str, source_rows: list[dict[str, str]]) -> dict:
    task_dir = DATA_ROOT / task
    manifest = json.loads((task_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('complete') is not True or manifest.get('sampling_rate_hz') != 125:
        raise ValueError(f'{task}: incomplete or wrong-rate output manifest')
    offsets = np.load(task_dir / f'{split}_window_offsets.npy', mmap_mode='r', allow_pickle=False)
    valid = np.load(task_dir / f'{split}_valid_lengths.npy', mmap_mode='r', allow_pickle=False)
    if not np.array_equal(offsets, np.arange(len(source_rows) + 1, dtype=np.int64)):
        raise ValueError(f'{task}/{split}: output window offsets are not one window per target')
    if valid.shape != (len(source_rows),) or not np.all(valid == 1250):
        raise ValueError(f'{task}/{split}: output valid-length values are invalid')
    sample_path = task_dir / f'{split}_samples.csv'
    window_path = task_dir / f'{split}_windows.csv'
    sample_count = window_count = 0
    with sample_path.open('r', newline='', encoding='utf-8') as stream:
        for i, saved in enumerate(csv.DictReader(stream)):
            if i >= len(source_rows):
                raise ValueError(f'{task}/{split}: extra output sample rows')
            source = source_rows[i]
            expected = {
                'sample_id': source['trial'],
                'source_id': f"Part_{int(source['Part'])}_{int(source['record_index'])}",
                'sample_index': str(i), 'source_part': str(int(source['Part'])),
                'source_record_index': str(int(source['record_index'])),
                'source_window_index': str(int(source['window_index'])),
                'ppg_source_start_sample': str(int(source['ppg_source_start_sample'])),
                'abp_source_start_sample': str(int(source['abp_source_start_sample'])),
                'ecg_source_start_sample': str(int(source['ecg_source_start_sample'])),
            }
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f'{task}/{split}: saved sample lineage mismatch at row {i}')
            sample_count += 1
    with window_path.open('r', newline='', encoding='utf-8') as stream:
        for i, saved in enumerate(csv.DictReader(stream)):
            if i >= len(source_rows):
                raise ValueError(f'{task}/{split}: extra output window rows')
            source = source_rows[i]
            expected = {
                'sample_index': str(i), 'window_index': '0',
                'source_part': str(int(source['Part'])),
                'source_record_index': str(int(source['record_index'])),
                'source_window_index': str(int(source['window_index'])),
                'ppg_source_start_sample': str(int(source['ppg_source_start_sample'])),
                'ppg_source_end_sample': str(int(source['ppg_source_start_sample']) + 1250),
                'abp_source_start_sample': str(int(source['abp_source_start_sample'])),
                'ecg_source_start_sample': str(int(source['ecg_source_start_sample'])),
                'start_time_s': str(int(source['window_index']) * 10.0),
                'end_time_s': str((int(source['window_index']) + 1) * 10.0),
                'valid_length': '1250',
            }
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f'{task}/{split}: saved window lineage mismatch at row {i}')
            window_count += 1
    if sample_count != len(source_rows) or window_count != len(source_rows):
        raise ValueError(f'{task}/{split}: metadata row count mismatch')
    return {'sample_rows_source_matched': sample_count, 'window_rows_source_matched': window_count,
            'one_window_per_target_offsets': True, 'all_valid_lengths_1250': True}


def audit_existing(
    raw_part_source_dir: Path = RAW_SOURCE,
    verified_product_dir: Path = VERIFIED_10S,
) -> dict:
    """Re-read the finished output metadata and source provenance."""
    product_dir, source_reports = ensure_verified_uci_10s_product(raw_part_source_dir, verified_product_dir)
    source_summary = _verify_source_product(product_dir, source_reports)
    source_summary['source_build_provenance'] = source_reports['build_provenance']
    source_summary['raw_input_hash_audit'] = source_reports['raw_input_hash_audit']
    source_summary['output_path_provenance'] = source_reports['output_path_provenance']
    result_path = REPORT_ROOT / 'uci_verification.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {
        'tasks': {}, 'source_summary': source_summary, 'split_validation': {}
    }
    result['source_summary'] = source_summary
    result['saved_metadata_validation'] = {}
    for split in common.SPLITS:
        source_rows = _read_metadata_rows(product_dir / 'metadata' / f'{split}.csv')
        for task in ('uci_sbp', 'uci_dbp'):
            result['saved_metadata_validation'][f'{task}/{split}'] = _audit_saved_metadata(task, split, source_rows)

    result.pop('raw_pipeline_smoke_validation', None)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def run_uci(
    raw_part_source_dir: Path = RAW_SOURCE,
    verified_product_dir: Path = VERIFIED_10S,
) -> dict:
    """Validate a verified product and create independent SBP/DBP outputs."""
    dest_dirs = [DATA_ROOT / 'uci_sbp', DATA_ROOT / 'uci_dbp']
    if any(path.exists() for path in dest_dirs):
        raise FileExistsError(f'Refusing to overwrite UCI task output(s): {[str(p) for p in dest_dirs if p.exists()]}')
    raw_part_source_dir = Path(raw_part_source_dir).resolve()
    source_dir, source_reports = ensure_verified_uci_10s_product(
        raw_part_source_dir=raw_part_source_dir, verified_product_dir=verified_product_dir,
    )
    source_summary = _verify_source_product(source_dir, source_reports)
    source_manifest = source_reports['manifest']
    source_summary['source_build_provenance'] = source_reports['build_provenance']
    source_summary['raw_input_hash_audit'] = source_reports['raw_input_hash_audit']
    source_summary['output_path_provenance'] = source_reports['output_path_provenance']
    result = {'tasks': {}, 'source_summary': source_summary, 'split_validation': {}}

    for split in common.SPLITS:
        metadata_rows = _read_metadata_rows(source_dir / 'metadata' / f'{split}.csv')
        sbp = np.load(source_dir / 'uci_sbp' / f'{split}_labels.npy', mmap_mode='r', allow_pickle=False)
        dbp = np.load(source_dir / 'uci_dbp' / f'{split}_labels.npy', mmap_mode='r', allow_pickle=False)
        source_features = np.load(source_dir / 'uci_sbp' / f'{split}_features.npy', mmap_mode='r', allow_pickle=False)
        source_features_dbp = np.load(source_dir / 'uci_dbp' / f'{split}_features.npy', mmap_mode='r', allow_pickle=False)
        expected_shape = (len(metadata_rows), 1, 1250)
        for name, array in [('features', source_features), ('dbp_features', source_features_dbp)]:
            if array.shape != expected_shape or array.dtype != np.float32:
                raise ValueError(f'{split} {name}: unexpected {array.shape}/{array.dtype}, expected {expected_shape}/float32')
            for start in range(0, len(array), 2048):
                if not np.isfinite(array[start:start + 2048]).all():
                    raise ValueError(f'{split} {name}: nonfinite feature values at row {start}')
        if not (np.array_equal(source_features, source_features_dbp) and np.isfinite(sbp).all() and np.isfinite(dbp).all()):
            raise ValueError(f'{split}: UCI input feature arrays or BP labels invalid')
        labels_from_meta = np.asarray([[float(row['SP']), float(row['DP'])] for row in metadata_rows], dtype=np.float32)
        if not np.array_equal(np.asarray(sbp), labels_from_meta[:, 0]) or not np.array_equal(np.asarray(dbp), labels_from_meta[:, 1]):
            raise ValueError(f'{split}: SBP/DBP labels do not exactly match source metadata after float32 storage')
        if np.any((sbp < 30) | (sbp > 220) | (dbp < 30) | (dbp > 220) | (sbp < dbp)):
            raise ValueError(f'{split}: BP labels violate accepted UCI value ranges/order')
        samples, windows = _metadata_for_common(metadata_rows, split)
        offsets = np.arange(len(sbp) + 1, dtype=np.int64)
        for task, labels, source_features_task in (
            ('uci_sbp', sbp, source_features), ('uci_dbp', dbp, source_features_dbp),
        ):
            info = common.save_split(task, split, source_features_task, labels, samples.copy(),
                                     [row.copy() for row in windows], offsets=offsets,
                                     valid_lengths=np.full(len(sbp), 1250, dtype=np.int16))
            result['tasks'].setdefault(task, {})[split] = info
            out_features = DATA_ROOT / task / f'{split}_features.npy'
            out_labels = DATA_ROOT / task / f'{split}_labels.npy'
            source_feature_path = source_dir / task / f'{split}_features.npy'
            source_label_path = source_dir / task / f'{split}_labels.npy'
            feature_check = _compare_arrays(out_features, source_feature_path)
            label_check = _compare_arrays(out_labels, source_label_path)
            if label_check['shape'] != [len(metadata_rows)]:
                raise ValueError(f'{task}/{split}: label shape mismatch')
            result['split_validation'][f'{task}/{split}'] = {
                'feature_values_vs_verified_10s': feature_check,
                'labels_vs_verified_10s': label_check,
                'source_rows': len(metadata_rows),
                'source_part_counts': dict(Counter(str(int(r['Part'])) for r in metadata_rows)),
            }

    # Both tasks must carry independently saved copies of the same PPG samples.
    for split in common.SPLITS:
        a = DATA_ROOT / 'uci_sbp' / f'{split}_features.npy'
        b = DATA_ROOT / 'uci_dbp' / f'{split}_features.npy'
        if os.path.samefile(a, b):
            raise AssertionError('SBP and DBP feature files must be independent files, not hard links')
        if sha256(a) != sha256(b):
            raise AssertionError(f'SBP and DBP PPG feature copies differ in {split}')

    details = {
        'source_dataset': 'UCI Cuff-Less Blood Pressure Estimation Dataset 340, Parts 1-4',
        'raw_part_source_dir': str(raw_part_source_dir),
        'materialized_from': str(source_dir),
        'materialization_mode': 'reused_full_verified_10s_product_then_independently_saved',
        'upstream_repository': source_manifest['upstream_repository'],
        'upstream_commit': source_manifest['upstream_commit'],
        'source_manifest_sha256': sha256(source_dir / 'manifest.json'),
        'source_verification_sha256': sha256(source_dir / 'verification.json'),
        'input_files_sha256': source_manifest['input_files_sha256'],
        'sampling_rate_hz': 125, 'window_seconds': 10, 'window_samples': 1250,
        'channels': ['PPG'], 'signal_unit': 'source processed PPG amplitude; no offline normalization',
        'label': 'median accepted ABP systolic/diastolic peak/valley amplitude',
        'label_units': 'mmHg', 'split_by_part': {'Part_1': 'train', 'Part_2': 'train', 'Part_3': 'val', 'Part_4': 'test'},
        'source_signal_processing': source_manifest['parameters']['ppg_saved_signal'],
        'source_cleaning_and_alignment': source_manifest['parameters']['cleaning'],
        'part_counts': source_summary['part_counts'],
        'source_records_by_split': source_summary['source_records_by_split'],
        'source_record_intersections_between_splits': source_summary['source_record_intersections_between_splits'],
        'source_existing_verification_status': source_summary['source_verification_status'],
        'output_array_exact_comparison': 'Every output feature and label float32 value compared row-by-row against the previously fully verified 10 s product.',
        'sample_membership_caveat': 'Part/record IDs are source record identifiers, not confirmed patient IDs. The public Part split is preserved; cross-Part patient identity is not established.',
        'protocol_scope': 'This adapter reuses the full verified Part-based 10-second product and makes no claim about old 5-second NPY membership.',
    }
    for task in ('uci_sbp', 'uci_dbp'):
        details['task'] = task
        common.finish_task(task, details)
    report_path = REPORT_ROOT / 'uci_verification.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    _write_readme(result, details)
    return result


def _write_readme(result: dict, details: dict) -> None:
    text = f"""# UCI 10-second PPG tasks in the common downstream format

This directory contains UCI SBP and DBP tasks materialized from a previously verified product built under the official Part_1..Part_4 dataset protocol. It uses 125 Hz, nonoverlapping 10-second windows (1,250 samples), keeps one PPG channel, and saves one mmHg target per window. Parts 1+2 are train, Part 3 is validation, and Part 4 is test. The split is the public Part split; Part record identifiers do not establish person-level identity across Parts.

The features were independently saved into the new output directory and compared exactly, row by row, with the complete, previously verified UCI 10-second artifact at `{details['materialized_from']}`. The source artifact reports verification status `{details['source_existing_verification_status']}`. Full output comparisons and lineage checks are in `uci_verification.json`.

The saved input is the upstream cubic-spline baseline-corrected PPG. The 0.5–8 Hz Butterworth branch is used for quality/features during cleaning and is not applied to the stored PPG waveform. Labels are median accepted ABP systolic peaks (SBP) or diastolic valleys (DBP), in mmHg.

This package only reuses a previously complete and verified 10-second product; it does not rebuild the product from raw MAT files. The original `Part_1.mat` through `Part_4.mat` files are still required at `{details['raw_part_source_dir']}/data/` so the adapter can verify each file's identity and raw-byte SHA-256 against the product manifest. The adapter also validates the product metadata, arrays, labels, full split counts, 125 Hz sampling rate, 10-second windows, and fixed Part split. It refuses to overwrite an existing task directory.

Historical program and upstream source hashes remain in the product provenance. Since those source files are not included in this package, no local source-code comparison is performed. The provenance report is written under `outputs/preprocessing/`.

This is a new 10-second protocol and does not claim to preserve the old 5-second NPY membership.
"""
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / 'uci_processing.md').write_text(text, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verified-product-dir', type=Path, default=VERIFIED_10S)
    parser.add_argument('--raw-part-source-dir', type=Path, default=RAW_SOURCE)
    parser.add_argument('--audit-existing', action='store_true')
    args = parser.parse_args()
    if args.audit_existing:
        result = audit_existing(raw_part_source_dir=args.raw_part_source_dir,
                                verified_product_dir=args.verified_product_dir)
        print(json.dumps({'saved_metadata_validation': result['saved_metadata_validation'],
                          'report': str(REPORT_ROOT / 'uci_verification.json')}, indent=2))
        return
    result = run_uci(raw_part_source_dir=args.raw_part_source_dir,
                     verified_product_dir=args.verified_product_dir)
    print(json.dumps({'tasks': result['tasks'], 'report': str(REPORT_ROOT / 'uci_verification.json')}, indent=2))


if __name__ == '__main__':
    main()
