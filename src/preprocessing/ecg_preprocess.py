#!/usr/bin/env python3
"""Build one final PhysioFSQ ECG downstream dataset from raw recordings.

The bundled ``data_protocols/downstream_manifest.json`` and its split CSVs
define the final record membership, labels, and row order. No old experiment
arrays or split-generation code are required.

Example::

    python src/preprocessing/ecg_preprocess.py --task csn \
        --raw-root /path/to/datasets --output-root /path/to/downstream

Choose exactly one task per invocation. Existing task outputs are never
overwritten.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import wfdb

import common


TASKS = ('csn', 'ptb-xl', 'physionet2017')


def _task_root(raw_root: Path, task: str) -> Path:
    roots = {
        'csn': raw_root / 'csn' / '1.0.0' / 'data' / 'WFDBRecords',
        'ptb-xl': raw_root / 'ptb-xl' / '1.0.3',
        'physionet2017': raw_root / 'physionet2017' / '1.0.0',
    }
    return roots[task]


def _ensure_fresh_output(task: str) -> Path:
    task_dir = common.DATA_ROOT / task
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(
            f'Refusing to overwrite existing output directory: {task_dir}. '
            'Choose a fresh --output-root.'
        )
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def _onehot(label_sets: list[Iterable[str]], classes: list[str]) -> np.ndarray:
    class_to_index = {value: index for index, value in enumerate(classes)}
    result = np.zeros((len(label_sets), len(classes)), dtype=np.int64)
    for row, values in enumerate(label_sets):
        unknown = set(values).difference(class_to_index)
        if unknown:
            raise ValueError(f'Labels outside the frozen class list: {sorted(unknown)[:10]}')
        for value in values:
            result[row, class_to_index[value]] = 1
    return result


def _split_id_list(split_rows: dict[str, list[dict[str, str]]]) -> list[str]:
    return [row['source_id'] for split in common.SPLITS for row in split_rows[split]]


def _iter_ordered_map(
    pool: ThreadPoolExecutor,
    function,
    items: list[Any],
    *,
    batch_size: int,
) -> Iterator[Any]:
    """Map in small batches to bound buffered WFDB records in memory."""
    for begin in range(0, len(items), batch_size):
        yield from pool.map(function, items[begin:begin + batch_size])


def _stage_features(shape: tuple[int, ...], task: str, split: str):
    """Create a disk-backed array and its temporary directory for one split."""
    common.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(
        prefix=f'.{task}-{split}-', dir=common.DATA_ROOT
    )
    path = Path(tmp.name) / 'features.npy'
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    return tmp, array


def _finish_staged_split(
    task: str,
    split: str,
    staged: np.memmap,
    labels: np.ndarray,
    samples: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    *,
    offsets: np.ndarray | None = None,
    valid_lengths: np.ndarray | None = None,
) -> dict[str, Any]:
    staged.flush()
    saved = common.save_split(
        task, split, staged, labels, samples, windows,
        offsets=offsets, valid_lengths=valid_lengths,
    )
    return saved


def _parse_csn_dx(header: Path) -> list[str]:
    with header.open('r', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if line.strip().startswith('#Dx:'):
                values = [value.strip() for value in line.split(':', 1)[1].split(',')]
                return [value for value in values if value]
    raise ValueError(f'CSN header has no #Dx diagnosis line: {header}')


def _read_csn(record_base: str):
    signal, fields = wfdb.rdsamp(record_base)
    return signal, fields


def process_csn(raw_root: Path, workers: int) -> dict[str, Any]:
    """Build the frozen 94-label CSN Lead-I task in split-CSV row order."""
    task = 'csn'
    protocol, split_rows = common.load_protocol_split(task)
    spec = protocol['tasks'][task]
    classes = [str(value) for value in spec['label_classes']]
    root = _task_root(raw_root, task)
    headers = sorted(root.rglob('*.hea'), key=lambda path: path.as_posix())
    header_by_id = {path.stem: path.with_suffix('') for path in headers}
    if len(headers) != 45152 or len(header_by_id) != len(headers):
        raise ValueError(f'Expected 45,152 uniquely named CSN headers; found {len(headers)}')

    protocol_ids = set(_split_id_list(split_rows))
    missing = sorted(protocol_ids.difference(header_by_id))
    excluded = sorted(set(header_by_id).difference(protocol_ids))
    if missing:
        raise FileNotFoundError(f'{len(missing)} frozen CSN IDs are missing, e.g. {missing[:10]}')
    if len(protocol_ids) != 45150 or len(excluded) != 2:
        raise ValueError(
            f'Frozen CSN split must contain 45,150 records and exclude two unreadable records; '
            f'got {len(protocol_ids)} included and {len(excluded)} excluded.'
        )
    # Confirm that the two records absent from the frozen membership list are
    # still the known unreadable sources, without carrying their IDs or any
    # old-array restoration logic into the builder.
    excluded_errors = {}
    for record_id in excluded:
        try:
            wfdb.rdsamp(str(header_by_id[record_id]))
        except Exception as exc:
            excluded_errors[record_id] = f'{type(exc).__name__}: {exc}'
        else:
            raise ValueError(f'Excluded CSN record became readable: {record_id}')
    if len(excluded_errors) != 2:
        raise ValueError(f'Expected two unreadable CSN records, confirmed {len(excluded_errors)}')

    task_dir = _ensure_fresh_output(task)
    output_summary = {}
    source_nonfinite_other_lead_records = 0
    observed_classes: set[str] = set()
    for split in common.SPLITS:
        rows = split_rows[split]
        count = len(rows)
        tmp, staged = _stage_features((count, 1, common.WINDOW), task, split)
        labels_list: list[list[str]] = []
        samples: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []

        def read_item(row: dict[str, str]):
            record_id = row['source_id']
            header = header_by_id[record_id]
            diagnosis = _parse_csn_dx(header.with_suffix('.hea'))
            signal, fields = _read_csn(str(header))
            return row, header, diagnosis, signal, fields

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = _iter_ordered_map(
                    pool, read_item, rows, batch_size=max(8, workers * 4)
                )
                for row_index, (row, header, diagnosis, signal, fields) in enumerate(results):
                    record_id = row['source_id']
                    if record_id != row['sample_id']:
                        raise ValueError(f'CSN sample/source ID mismatch at {split}[{row_index}]')
                    if signal.shape != (5000, 12) or int(fields.get('fs', -1)) != 500:
                        raise ValueError(
                            f'CSN expected [5000,12] at 500 Hz for {record_id}, '
                            f'got {signal.shape} at {fields.get("fs")} Hz'
                        )
                    lead_names = [str(value) for value in fields.get('sig_name', [])]
                    if len(lead_names) != 12 or lead_names[0] != 'I':
                        raise ValueError(f'CSN channel 0 is not Lead I for {record_id}: {lead_names}')
                    selected = np.asarray(signal[:, 0], dtype=np.float64)
                    if not np.isfinite(selected).all():
                        raise ValueError(f'Nonfinite CSN Lead-I waveform: {record_id}')
                    diagnosis_set = set(diagnosis)
                    unknown = diagnosis_set.difference(classes)
                    if unknown:
                        raise ValueError(f'CSN diagnosis outside frozen 94-class list: {sorted(unknown)}')
                    observed_classes.update(diagnosis_set)
                    labels_list.append(sorted(diagnosis_set))
                    staged[row_index] = common.resample_signal(selected[None, :], 500)
                    relpath = header.with_suffix('.hea').relative_to(root).as_posix()
                    units = [str(value) for value in fields.get('units', [])]
                    samples.append({
                        'sample_id': row['sample_id'],
                        'source_id': record_id,
                        'source_path': relpath,
                        'source_fs_hz': 500,
                        'source_length_samples': 5000,
                        'source_duration_s': 10.0,
                        'lead_names': json.dumps(['I']),
                        'source_lead_names': json.dumps(lead_names),
                        'source_units': json.dumps(units),
                        'source_channel_count': 12,
                        'selected_lead_index': 0,
                        'selected_lead_name': 'I',
                        'selected_channel_index': 0,
                        'selected_channel_name': 'I',
                        'selected_channel_nonfinite_count': 0,
                        'source_other_lead_nonfinite_count': int(
                            np.size(signal[:, 1:]) - np.isfinite(signal[:, 1:]).sum()
                        ),
                        'source_gap_repairs': '[]',
                        'diagnosis_codes': json.dumps(sorted(diagnosis_set)),
                        'original_interval_start_s': 0.0,
                        'original_interval_end_s': 10.0,
                        'split_row': row_index,
                    })
                    windows.append({
                        'sample_index': row_index,
                        'window_index': 0,
                        'start_sample_125hz': 0,
                        'end_sample_125hz_exclusive': common.WINDOW,
                        'start_time_s': 0.0,
                        'end_time_s': 10.0,
                        'valid_length': common.WINDOW,
                        'window_note': 'complete 10-second native Lead-I interval',
                    })
                    if np.any(~np.isfinite(signal[:, 1:])):
                        source_nonfinite_other_lead_records += 1
                    if (row_index + 1) % 5000 == 0:
                        print(f'CSN {split}: {row_index + 1}/{count}', flush=True)

            labels = _onehot(labels_list, classes)
            expected_windows = int(spec['stored_split_counts'][split]['windows'])
            if len(staged) != expected_windows or labels.shape != (count, 94):
                raise ValueError(f'Unexpected CSN {split} geometry: {staged.shape}/{labels.shape}')
            output_summary[split] = _finish_staged_split(
                task, split, staged, labels, samples, windows,
                offsets=np.arange(count + 1, dtype=np.int64),
                valid_lengths=np.full(count, common.WINDOW, dtype=np.int16),
            )
        finally:
            del staged
            tmp.cleanup()

    if observed_classes != set(classes):
        missing_classes = sorted(set(classes).difference(observed_classes))
        raise ValueError(f'Frozen CSN labels did not cover all 94 classes: {missing_classes}')
    details = {
        'dataset': spec['dataset'],
        'source_root': 'csn/1.0.0/data/WFDBRecords',
        'source_sampling_rate_hz': 500,
        'source_window': 'one 10-second WFDB record; channel 0 must be named I',
        'source_channel_count': 12,
        'selected_lead_index': 0,
        'selected_lead_name': 'I',
        'lead_names': ['I'],
        'label_classes': classes,
        'label_format': '94-class multi-hot SNOMED CT diagnosis codes from each WFDB #Dx line',
        'split_protocol': spec['split_protocol'],
        'sample_split_file': spec['sample_split_file'],
        'sample_split_sha256': spec['sample_split_sha256'],
        'included_record_count': len(protocol_ids),
        'excluded_unreadable_record_count': len(excluded_errors),
        'excluded_unreadable_source_ids': excluded,
        'excluded_unreadable_errors': excluded_errors,
        'resampling_from_hz': 500,
        'resampling_to_hz': 125,
        'split_counts': {split: len(split_rows[split]) for split in common.SPLITS},
        'saved_arrays': output_summary,
        'validation': {
            'protocol_ids_present': len(protocol_ids),
            'source_header_count': len(headers),
            'unreadable_exclusions_confirmed': len(excluded_errors),
            'all_frozen_classes_observed': len(observed_classes),
            'nonfinite_values_on_nonselected_leads_are_ignored': source_nonfinite_other_lead_records,
            'split_row_order': 'Copied directly from the checksummed protocol split CSV.',
        },
    }
    common.finish_task(task, details)
    return {'task': task, 'split_counts': details['split_counts'],
            'excluded_unreadable_source_ids': excluded, 'output': str(task_dir)}


def _ptb_labels(metadata: pd.DataFrame, statement_df: pd.DataFrame,
                classes: list[str]) -> dict[int, np.ndarray]:
    diagnostic = statement_df[statement_df['diagnostic'] == 1]
    code_to_class = diagnostic['diagnostic_class'].dropna().astype(str).to_dict()
    labels = {}
    for ecg_id, row in metadata.iterrows():
        raw_codes = row['scp_codes']
        codes = ast.literal_eval(raw_codes) if isinstance(raw_codes, str) else raw_codes
        superclass = {code_to_class[code] for code in codes if code in code_to_class}
        labels[int(ecg_id)] = _onehot([superclass], classes)[0]
    return labels


def process_ptbxl(raw_root: Path, workers: int) -> dict[str, Any]:
    """Build final PTB-XL 12-lead 125 Hz arrays in frozen fold order."""
    del workers  # WFDB files are read in metadata order; I/O is already cached by the OS.
    task = 'ptb-xl'
    protocol, split_rows = common.load_protocol_split(task)
    spec = protocol['tasks'][task]
    classes = [str(value) for value in spec['label_classes']]
    root = _task_root(raw_root, task)
    database_path = root / 'ptbxl_database.csv'
    statement_path = root / 'scp_statements.csv'
    metadata = pd.read_csv(database_path, index_col='ecg_id')
    statements = pd.read_csv(statement_path, index_col=0)
    if metadata.index.has_duplicates or len(metadata) != 21799:
        raise ValueError(f'Unexpected PTB-XL metadata shape: {metadata.shape}')
    metadata['patient_id'] = metadata['patient_id'].astype('string')
    metadata['strat_fold'] = pd.to_numeric(metadata['strat_fold'], errors='raise').astype(int)
    labels_by_id = _ptb_labels(metadata, statements, classes)

    task_dir = _ensure_fresh_output(task)
    output_summary = {}
    patient_sets = {split: set() for split in common.SPLITS}
    for split in common.SPLITS:
        rows = split_rows[split]
        count = len(rows)
        tmp, staged = _stage_features((count, 12, common.WINDOW), task, split)
        labels = np.empty((count, len(classes)), dtype=np.int64)
        samples: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []
        try:
            for row_index, row in enumerate(rows):
                try:
                    ecg_id = int(row['source_id'])
                    source = metadata.loc[ecg_id]
                except (ValueError, KeyError) as exc:
                    raise ValueError(f'PTB-XL protocol ID cannot be found: {row["source_id"]}') from exc
                if str(ecg_id) != row['sample_id']:
                    raise ValueError(f'PTB-XL sample/source ID mismatch at {split}[{row_index}]')
                fold = int(source['strat_fold'])
                if (split == 'train' and fold not in range(1, 9)) or \
                        (split == 'val' and fold != 9) or (split == 'test' and fold != 10):
                    raise ValueError(f'PTB-XL {ecg_id} fold {fold} conflicts with protocol split {split}')
                patient_id = str(source['patient_id'])
                patient_sets[split].add(patient_id)
                relative_path = str(source['filename_lr'])
                record_base = root / relative_path
                signal, fields = wfdb.rdsamp(str(record_base))
                if int(fields.get('fs', -1)) != 100 or signal.shape != (1000, 12):
                    raise ValueError(
                        f'PTB-XL expected [1000,12] at 100 Hz for {ecg_id}, '
                        f'got {signal.shape} at {fields.get("fs")} Hz'
                    )
                if not np.isfinite(signal).all():
                    raise ValueError(f'Nonfinite PTB-XL waveform: {ecg_id}')
                lead_names = [str(value) for value in fields.get('sig_name', [])]
                units = [str(value) for value in fields.get('units', [])]
                if len(lead_names) != 12:
                    raise ValueError(f'PTB-XL expected 12 stored leads for {ecg_id}: {lead_names}')
                staged[row_index] = common.resample_signal(signal.T, 100)
                labels[row_index] = labels_by_id[ecg_id]
                samples.append({
                    'sample_id': row['sample_id'],
                    'source_id': row['source_id'],
                    'source_path': relative_path,
                    'patient_id': patient_id,
                    'strat_fold': fold,
                    'source_fs_hz': 100,
                    'source_length_samples': 1000,
                    'source_duration_s': 10.0,
                    'lead_names': json.dumps(lead_names),
                    'source_units': json.dumps(units),
                    'diagnostic_superclasses': json.dumps([
                        classes[index] for index in np.flatnonzero(labels[row_index])
                    ]),
                    'original_interval_start_s': 0.0,
                    'original_interval_end_s': 10.0,
                    'split_row': row_index,
                })
                windows.append({
                    'sample_index': row_index,
                    'window_index': 0,
                    'start_sample_125hz': 0,
                    'end_sample_125hz_exclusive': common.WINDOW,
                    'start_time_s': 0.0,
                    'end_time_s': 10.0,
                    'valid_length': common.WINDOW,
                    'window_note': 'full 10-second PTB-XL filename_lr record',
                })
                if (row_index + 1) % 2000 == 0:
                    print(f'PTB-XL {split}: {row_index + 1}/{count}', flush=True)
            expected_windows = int(spec['stored_split_counts'][split]['windows'])
            if len(staged) != expected_windows:
                raise ValueError(f'Unexpected PTB-XL {split} record count: {len(staged)}')
            output_summary[split] = _finish_staged_split(
                task, split, staged, labels, samples, windows,
                offsets=np.arange(count + 1, dtype=np.int64),
                valid_lengths=np.full(count, common.WINDOW, dtype=np.int16),
            )
        finally:
            del staged
            tmp.cleanup()

    overlap = {
        f'{left}_{right}': sorted(patient_sets[left] & patient_sets[right])
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test'))
    }
    if any(overlap.values()):
        raise ValueError(f'PTB-XL patient IDs cross official folds: {overlap}')
    details = {
        'dataset': spec['dataset'],
        'source_root': 'ptb-xl/1.0.3',
        'source_sampling_rate_hz': 100,
        'source_window': 'one full 10-second filename_lr record, 1,000 samples',
        'source_channel_count': 12,
        'stored_leads': 'All 12 native leads are retained; the downstream model loader selects Lead I.',
        'label_classes': classes,
        'label_format': 'Five-class multi-hot diagnostic superclass labels; zero-label records are retained.',
        'split_protocol': spec['split_protocol'],
        'sample_split_file': spec['sample_split_file'],
        'sample_split_sha256': spec['sample_split_sha256'],
        'source_record_count': int(sum(len(split_rows[s]) for s in common.SPLITS)),
        'included_record_count': int(sum(len(split_rows[s]) for s in common.SPLITS)),
        'resampling_from_hz': 100,
        'resampling_to_hz': 125,
        'split_counts': {split: len(split_rows[split]) for split in common.SPLITS},
        'patient_id_split_overlap': overlap,
        'metadata_sha256': common.sha256(database_path),
        'scp_statements_sha256': common.sha256(statement_path),
        'saved_arrays': output_summary,
        'validation': {
            'patient_id_split_overlap': overlap,
            'all_rows_follow_checksummed_protocol_order': True,
        },
    }
    common.finish_task(task, details)
    return {'task': task, 'split_counts': details['split_counts'], 'output': str(task_dir)}


def _physio_window_starts(length: int) -> list[int]:
    if length <= common.WINDOW:
        return [0]
    starts = list(range(0, length - common.WINDOW + 1, common.WINDOW))
    end_aligned = length - common.WINDOW
    if starts[-1] + common.WINDOW < length and end_aligned not in starts:
        starts.append(end_aligned)
    return sorted(starts)


def _read_physio(record_base: str):
    return wfdb.rdrecord(record_base)


def process_physionet2017(raw_root: Path, workers: int) -> dict[str, Any]:
    """Build grouped 10-second PhysioNet2017 windows and record labels."""
    task = 'physionet2017'
    protocol, split_rows = common.load_protocol_split(task)
    spec = protocol['tasks'][task]
    label_map = {str(key): int(value) for key, value in spec['label_classes'].items()}
    root = _task_root(raw_root, task)
    training_root = root / 'training'
    reference_path = training_root / 'REFERENCE-v3.csv'
    reference = pd.read_csv(reference_path, header=None, names=['record', 'label'], dtype=str)
    if reference['record'].duplicated().any():
        raise ValueError('PhysioNet REFERENCE-v3 contains duplicate record IDs')
    reference_by_id = dict(zip(reference['record'].astype(str), reference['label'].astype(str)))
    if set(reference_by_id.values()).difference(label_map):
        raise ValueError('PhysioNet REFERENCE-v3 contains labels outside the frozen mapping')
    all_ids = set(_split_id_list(split_rows))
    if len(all_ids) != 8528 or all_ids != set(reference_by_id):
        raise ValueError('PhysioNet split IDs do not match the complete REFERENCE-v3 record set')

    header_by_id = {
        header.relative_to(training_root).with_suffix('').as_posix(): header.with_suffix('')
        for header in training_root.rglob('*.hea')
    }
    if set(header_by_id) != all_ids:
        missing = sorted(all_ids.difference(header_by_id))
        extra = sorted(set(header_by_id).difference(all_ids))
        raise FileNotFoundError(
            f'PhysioNet header membership mismatch: missing={missing[:5]}, extra={extra[:5]}'
        )

    task_dir = _ensure_fresh_output(task)
    output_summary = {}
    window_counts = {}
    partial_windows = {}
    label_symbols = set()
    for split in common.SPLITS:
        rows = split_rows[split]
        count = len(rows)
        expected_windows = int(spec['stored_split_counts'][split]['windows'])
        tmp, staged = _stage_features((expected_windows, 1, common.WINDOW), task, split)
        sample_rows: list[dict[str, Any]] = []
        labels = np.empty(count, dtype=np.int64)
        offsets = [0]
        window_metadata: list[dict[str, Any]] = []
        valid_lengths: list[int] = []
        feature_count = 0

        def read_item(row: dict[str, str]):
            record_id = row['source_id']
            return row, _read_physio(str(header_by_id[record_id]))

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = _iter_ordered_map(
                    pool, read_item, rows, batch_size=max(8, workers * 4)
                )
                for sample_index, (row, record) in enumerate(results):
                    record_id = row['source_id']
                    if row['sample_id'] != record_id:
                        raise ValueError(f'PhysioNet sample/source ID mismatch at {split}[{sample_index}]')
                    symbol = reference_by_id[record_id]
                    label_symbols.add(symbol)
                    labels[sample_index] = label_map[symbol]
                    if int(record.fs) != 300 or record.p_signal is None:
                        raise ValueError(f'PhysioNet expected a physical waveform at 300 Hz: {record_id}')
                    signal = np.asarray(record.p_signal.T, dtype=np.float64)
                    if signal.ndim != 2 or signal.shape[0] != 1 or not np.isfinite(signal).all():
                        raise ValueError(f'Invalid PhysioNet single-lead waveform {record_id}: {signal.shape}')
                    lead_names = [str(value) for value in record.sig_name]
                    units = [str(value) for value in (record.units or [])]
                    native_length = int(signal.shape[-1])
                    resampled = common.resample_signal(signal, 300)
                    output_length = int(resampled.shape[-1])
                    starts = _physio_window_starts(output_length)
                    sample_rows.append({
                        'sample_id': row['sample_id'],
                        'source_id': record_id,
                        'source_path': header_by_id[record_id].relative_to(root).as_posix(),
                        'source_fs_hz': 300,
                        'source_length_samples': native_length,
                        'source_duration_s': native_length / 300.0,
                        'resampled_length_samples': output_length,
                        'resampled_duration_s': output_length / 125.0,
                        'lead_names': json.dumps(lead_names),
                        'source_units': json.dumps(units),
                        'label_symbol': symbol,
                        'original_interval_start_s': 0.0,
                        'original_interval_end_s': native_length / 300.0,
                        'windows_count': len(starts),
                        'split_row': sample_index,
                    })
                    for window_index, start in enumerate(starts):
                        end = min(start + common.WINDOW, output_length)
                        valid = end - start
                        chunk = np.zeros((1, common.WINDOW), dtype=np.float32)
                        chunk[:, :valid] = resampled[:, start:end]
                        staged[feature_count] = chunk
                        window_metadata.append({
                            'sample_index': sample_index,
                            'window_index': window_index,
                            'start_sample_125hz': start,
                            'end_sample_125hz_exclusive': end,
                            'start_time_s': start / 125.0,
                            'end_time_s': end / 125.0,
                            'source_start_sample_approx': start * 300.0 / 125.0,
                            'source_end_sample_approx_exclusive': end * 300.0 / 125.0,
                            'source_start_time_s': start / 125.0,
                            'source_end_time_s': min(end / 125.0, native_length / 300.0),
                            'valid_length': valid,
                            'window_has_overlap': bool(
                                window_index > 0 and start < starts[window_index - 1] + common.WINDOW
                            ),
                            'window_note': (
                                'short record; right-zero-padded; valid_length is metadata only, no mask'
                                if output_length < common.WINDOW else
                                ('end-aligned tail window; may overlap preceding window'
                                 if window_index > 0 and start < starts[window_index - 1] + common.WINDOW
                                 else 'full 10-second window')
                            ),
                        })
                        valid_lengths.append(valid)
                        feature_count += 1
                    offsets.append(feature_count)
                    if (sample_index + 1) % 1000 == 0:
                        print(f'PhysioNet2017 {split}: {sample_index + 1}/{count}', flush=True)
            if feature_count != expected_windows:
                raise ValueError(
                    f'PhysioNet2017 {split} window count changed: {feature_count} != {expected_windows}'
                )
            output_summary[split] = _finish_staged_split(
                task, split, staged, labels, sample_rows, window_metadata,
                offsets=np.asarray(offsets, dtype=np.int64),
                valid_lengths=np.asarray(valid_lengths, dtype=np.int16),
            )
        finally:
            del staged
            tmp.cleanup()
        window_counts[split] = feature_count
        partial_windows[split] = int(np.sum(np.asarray(valid_lengths) != common.WINDOW))

    if label_symbols != set(label_map):
        raise ValueError(f'PhysioNet labels seen differ from frozen label map: {label_symbols}')
    details = {
        'dataset': spec['dataset'],
        'source_root': 'physionet2017/1.0.0/training',
        'source_reference': 'training/REFERENCE-v3.csv',
        'source_reference_sha256': common.sha256(reference_path),
        'source_sampling_rate_hz': 300,
        'source_duration_policy': (
            'Use the actual WFDB record length. Create nonoverlapping full 10-second '
            'windows and append an end-aligned tail window when a remainder exists. '
            'Right-zero-pad records shorter than 10 seconds; valid_lengths is metadata only.'
        ),
        'source_channel_count': 1,
        'label_classes': label_map,
        'label_format': 'One four-class record target; windows from each record share a sample index and offsets.',
        'split_protocol': spec['split_protocol'],
        'sample_split_file': spec['sample_split_file'],
        'sample_split_sha256': spec['sample_split_sha256'],
        'included_record_count': len(all_ids),
        'resampling_from_hz': 300,
        'resampling_to_hz': 125,
        'split_counts': {split: len(split_rows[split]) for split in common.SPLITS},
        'window_counts': window_counts,
        'partial_windows': partial_windows,
        'saved_arrays': output_summary,
        'validation': {
            'reference_rows': len(reference),
            'labels_seen': sorted(label_symbols),
            'window_counts': window_counts,
            'partial_windows': partial_windows,
            'split_row_order': 'Copied directly from the checksummed protocol split CSV.',
            'window_grouping': 'window_offsets preserve one record label per sample and group all its windows.',
        },
    }
    common.finish_task(task, details)
    return {'task': task, 'split_counts': details['split_counts'],
            'window_counts': window_counts, 'partial_windows': partial_windows,
            'output': str(task_dir)}


def run_task(task: str, raw_root: Path, output_root: Path, workers: int) -> dict[str, Any]:
    common.DATA_ROOT = output_root.expanduser().resolve()
    common.RAW_ROOT = raw_root.expanduser().resolve()
    if task == 'csn':
        return process_csn(common.RAW_ROOT, workers)
    if task == 'ptb-xl':
        return process_ptbxl(common.RAW_ROOT, workers)
    if task == 'physionet2017':
        return process_physionet2017(common.RAW_ROOT, workers)
    raise ValueError(f'Unknown task {task!r}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASKS,
                        help='Build exactly one task; the option is required to prevent accidental full rebuilds.')
    parser.add_argument('--raw-root', default=str(common.RAW_ROOT),
                        help='Parent directory containing csn/, ptb-xl/, and physionet2017/.')
    parser.add_argument('--output-root', default=str(common.DATA_ROOT),
                        help='Fresh common downstream output root; defaults to package data/downstream_125hz_v1/.')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='Parallel WFDB reads for CSN and PhysioNet2017 (default: 4).')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    result = run_task(args.task, Path(args.raw_root), Path(args.output_root), args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
