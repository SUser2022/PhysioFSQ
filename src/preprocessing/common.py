"""Shared, explicit storage contract for the downstream 125 Hz rebuild."""
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy.signal import resample_poly

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(os.environ.get('PREPROCESSING_WORK_ROOT', str(PACKAGE_ROOT / 'outputs' / 'preprocessing'))).expanduser()
DATA_ROOT = Path(os.environ.get('DOWNSTREAM_OUTPUT_ROOT', os.environ.get('DOWNSTREAM_DATA_ROOT', str(PACKAGE_ROOT / 'data' / 'downstream_125hz_v1')))).expanduser()
RAW_ROOT = Path(os.environ.get('RAW_DATA_ROOT', str(PACKAGE_ROOT / 'data' / 'raw'))).expanduser()
PROTOCOL_ROOT = Path(os.environ.get('DOWNSTREAM_PROTOCOL_ROOT', str(PACKAGE_ROOT / 'data_protocols'))).expanduser()
SPLITS = ('train', 'val', 'test')
FS = 125
WINDOW = 1250

# These eight tasks share DOWNSTREAM_DATA_ROOT. Cycle-derived DaLiA RR is
# intentionally published under its own root by build_dalia_rr10s_cycle.py.
COMMON_ROOT_TASKS = (
    'ptb-xl', 'csn', 'physionet2017', 'ppg_dalia_hr', 'bidmc_hr',
    'bidmc_rr', 'uci_sbp', 'uci_dbp',
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_protocol_manifest():
    """Load the bundled metadata-only protocol, never legacy waveform arrays."""
    path = PROTOCOL_ROOT / 'downstream_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing bundled downstream protocol: {path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != 'physiofsq-downstream-data-protocol-v1':
        raise ValueError(f'Unsupported downstream protocol schema in {path}')
    return value


def load_protocol_split(task):
    """Return protocol split rows in their frozen output order.

    The packaged split CSV is authoritative for membership and row order. Its
    SHA256 and per-split counts are checked before it is used to build arrays.
    """
    import csv

    protocol = load_protocol_manifest()
    try:
        task_spec = protocol['tasks'][task]
        relative = task_spec['sample_split_file']
        expected_sha = task_spec['sample_split_sha256']
        expected_counts = task_spec['stored_split_counts']
    except KeyError as exc:
        raise ValueError(f'{task!r} is not declared in the bundled protocol') from exc

    path = PROTOCOL_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f'Missing split membership file: {path}')
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f'{task} split file SHA256 mismatch: {actual_sha} != {expected_sha}')

    split_rows = {split: [] for split in SPLITS}
    with path.open('r', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'split', 'split_row', 'sample_id', 'source_id'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'{path} must contain columns {sorted(required)}')
        seen_ids = set()
        for row in reader:
            split = row['split']
            if split not in split_rows:
                raise ValueError(f'Unknown split {split!r} in {path}')
            expected_row = len(split_rows[split])
            if int(row['split_row']) != expected_row:
                raise ValueError(f'{task}/{split}: split_row is not contiguous at {expected_row}')
            if not row['sample_id'] or not row['source_id']:
                raise ValueError(f'{task}/{split}[{expected_row}]: empty sample/source ID')
            identity = (split, row['sample_id'])
            if identity in seen_ids:
                raise ValueError(f'{task}: duplicate sample ID {row["sample_id"]!r} in {split}')
            seen_ids.add(identity)
            split_rows[split].append(row)

    for split in SPLITS:
        expected = int(expected_counts[split]['samples'])
        if len(split_rows[split]) != expected:
            raise ValueError(f'{task}/{split}: protocol has {len(split_rows[split])} rows, expected {expected}')
    return protocol, split_rows


def resample_signal(signal, native_fs):
    """Float64 polyphase anti-aliasing, then float32 storage; time is last axis."""
    from math import gcd
    g = gcd(int(native_fs), FS)
    x = np.asarray(signal, dtype=np.float64)
    y = x if native_fs == FS else resample_poly(x, FS // g, int(native_fs) // g, axis=-1, window=('kaiser', 5.0), padtype='constant')
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite source/resampled waveform; no silent imputation is allowed')
    return np.asarray(y, dtype=np.float32)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'Empty metadata: {path}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_split(task, split, features, labels, samples, windows, offsets=None, valid_lengths=None):
    """features[W,C,1250], labels[N,...]; offsets[N+1] groups windows per target.

    samples has N rows; windows has W rows with sample_index referring to samples.
    Labels are never expanded to per-window pseudo-targets.
    """
    dest = DATA_ROOT / task
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / f'{split}_features.npy').exists():
        raise FileExistsError(f'Refusing to overwrite {dest}/{split}_features.npy')
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    n, w = len(y), len(x)
    if x.ndim != 3 or x.shape[-1] != WINDOW:
        raise ValueError(f'{task}/{split}: expected [windows,channels,1250], got {x.shape}')
    offsets = np.arange(n + 1, dtype=np.int64) if offsets is None else np.asarray(offsets, dtype=np.int64)
    valid = np.full(w, WINDOW, dtype=np.int16) if valid_lengths is None else np.asarray(valid_lengths, dtype=np.int16)
    assert offsets.shape == (n + 1,) and offsets[0] == 0 and offsets[-1] == w
    assert np.all(np.diff(offsets) > 0)
    assert valid.shape == (w,) and np.all((valid > 0) & (valid <= WINDOW))
    assert len(samples) == n and len(windows) == w
    assert np.isfinite(y).all()
    for begin in range(0, w, 1024):
        assert np.isfinite(x[begin:begin + 1024]).all(), (task, split, begin)
    for i, row in enumerate(samples):
        assert 'sample_id' in row and 'source_id' in row
        row['sample_index'] = i
    for i in range(n):
        for j in range(int(offsets[i]), int(offsets[i + 1])):
            assert int(windows[j]['sample_index']) == i
    for name, array in [('features', x), ('labels', y), ('window_offsets', offsets), ('valid_lengths', valid)]:
        np.save(dest / f'{split}_{name}.npy', array, allow_pickle=False)
    write_csv(dest / f'{split}_samples.csv', samples)
    write_csv(dest / f'{split}_windows.csv', windows)
    info = {'samples': n, 'windows': w, 'features_shape': list(x.shape), 'labels_shape': list(y.shape),
            'features_dtype': str(x.dtype), 'labels_dtype': str(y.dtype),
            'partial_windows': int(np.sum(valid != WINDOW))}
    print(f'SAVED {task}/{split}: {info}', flush=True)
    return info


def finish_task(task, details):
    details = dict(details)
    details.update({'complete': True, 'sampling_rate_hz': FS, 'window_samples': WINDOW,
                    'features_layout': '[W,C,T]; W windows, C channels, T=1250',
                    'label_layout': '[N,...]; one target per sample/record; W may differ from N',
                    'grouping': 'sample i uses features[window_offsets[i]:window_offsets[i+1]]',
                    'valid_lengths': 'Real-sample counts at 125 Hz; metadata only, not used for a downstream mask. Right-zero padding remains in the fixed-length input.',
                    'downstream_mask': False,
                    'normalization': 'No additional offline normalization or band filtering in this adapter. Preserve source amplitude units (physical units where defined); UCI retains its documented upstream baseline correction. Apply pretrained-model normalization online.',
                    'resampling': 'scipy.signal.resample_poly; Kaiser beta=5; constant boundary padding; native rate specified by source, never inferred from array length',
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}})
    write_json(DATA_ROOT / task / 'manifest.json', details)
    _finish_common_root_manifest()
    return details


def _finish_common_root_manifest():
    """Keep the root manifest usable by the grouped downstream loader.

    A root is complete once it contains at least one complete task product;
    ``tasks`` lists the products currently available at that root, so a single
    task can be loaded immediately. ``all_common_tasks_complete`` separately
    records whether the entire common-root suite has been materialized. The
    separately built DaLiA RR root has its own manifest writer.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = {}
    task_manifests = {}
    for task in COMMON_ROOT_TASKS:
        manifest_path = DATA_ROOT / task / 'manifest.json'
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get('complete') is True:
            tasks[task] = task
            task_manifests[task] = f'{task}/manifest.json'

    existing_path = DATA_ROOT / 'manifest.json'
    existing = {}
    if existing_path.is_file():
        try:
            existing = json.loads(existing_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing.update({
        'dataset': 'PhysioFSQ common downstream datasets',
        'schema': 'physiofsq-downstream-root-v1',
        'complete': bool(tasks),
        'all_common_tasks_complete': set(tasks) == set(COMMON_ROOT_TASKS),
        'sampling_rate_hz': FS,
        'window_samples': WINDOW,
        'use_mask': False,
        'tasks': tasks,
        'task_manifests': task_manifests,
        'known_common_tasks': list(COMMON_ROOT_TASKS),
        'updated_utc': datetime.now(timezone.utc).isoformat(),
    })
    write_json(existing_path, existing)
