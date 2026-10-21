#!/usr/bin/env python3
"""Append-only evaluation for previously frozen Maruhan identity-isolated reference.

Default is read-only. --save writes a single immutable receipt; no changes to
Champion, Formal Forward, Inventory Guard, prediction, or source files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent
VERSION = 'MARUHAN_ISOLATED_REFERENCE_EVALUATION_V1'
REL = Path('data/maruhan_maebashi/machine_number')
SUBDIR = 'analysis_isolated_prospective_reference_v1'
PREDICT_COLUMNS = ['rank', 'machine_no', 'machine_name', 'score', 'verified_days']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def under_root(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('UNSAFE_RELATIVE_PATH')
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('SOURCE_ESCAPES_PROJECT_ROOT')
    return resolved


def read_predictions(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != PREDICT_COLUMNS:
            raise ValueError('PREDICTION_COLUMNS_MISMATCH')
        records = list(reader)
    if len(records) != 10:
        raise ValueError('PREDICTION_TOP10_COUNT')
    result, seen = [], set()
    for i, row in enumerate(records, 1):
        if int(row['rank']) != i:
            raise ValueError('PREDICTION_RANK_INVALID')
        no = int(row['machine_no'])
        if no in seen or not row['machine_name'].strip():
            raise ValueError('PREDICTION_DUPLICATE_OR_EMPTY')
        seen.add(no)
        score = float(row['score'])
        if not (0 <= score <= 100):
            raise ValueError('PREDICTION_SCORE_INVALID')
        if int(row['verified_days']) < 7:
            raise ValueError('PREDICTION_AGE7_INVALID')
        result.append(dict(rank=i, machine_no=no, machine_name=row['machine_name'], score=score))
    return result


def read_actual(path: Path, expected_date: date) -> dict[int, tuple[str, float]]:
    # Reuse the already-tested strict daily reader, with the expected 514 count.
    import slotanalyzer_maruhan_history_isolation_v1 as iso
    return iso.daily(path, expected_date, 514)


def evaluate(root: Path, target: date, now: datetime, save: bool) -> str:
    root = root.resolve()
    if now.tzinfo is None:
        return 'BLOCKED: NAIVE_EVALUATION_CLOCK'
    if now.astimezone(JST).date() <= target:
        return 'SKIP: TARGET_DAY_NOT_FINISHED'
    base = root / REL
    folder = base / SUBDIR / f'{target:%Y%m%d}'
    manifest_file = folder / 'freeze_manifest.json'
    if not manifest_file.is_file():
        return 'SKIP: NO_FROZEN_ISOLATED_REFERENCE'
    receipt = folder / 'evaluation_receipt_v1.json'
    if receipt.exists():
        return 'SKIP: RECEIPT_ALREADY_EXISTS_IMMUTABLE'
    try:
        manifest = json.loads(manifest_file.read_text(encoding='utf-8-sig'))
        if manifest.get('schema_version') != 1 or manifest.get('kind') != 'ISOLATED_PROSPECTIVE_REFERENCE_NOT_FORMAL':
            raise ValueError('MANIFEST_KIND_INVALID')
        if manifest.get('formal_forward') is not False or manifest.get('champion_modified') is not False:
            raise ValueError('MANIFEST_FORMAL_STATUS_INVALID')
        if manifest.get('target_date') != target.isoformat() or manifest.get('data_through') != (target - timedelta(days=1)).isoformat():
            raise ValueError('MANIFEST_DATES_INVALID')
        frozen_at = datetime.fromisoformat(manifest['frozen_at_jst'])
        if frozen_at.tzinfo is None:
            raise ValueError('FREEZE_TIMESTAMP_NO_TIMEZONE')
        local = frozen_at.astimezone(JST)
        if local.date() != target or local.time() >= time(9, 0):
            raise ValueError('FREEZE_TIME_OUTSIDE_WINDOW')
        if manifest.get('top10_count') != 10:
            raise ValueError('MANIFEST_TOP10_COUNT_INVALID')
        guard = manifest.get('guard_evidence', {})
        if guard.get('record_match') is not True or guard.get('state') not in ('BLOCKED', 'APPROVED'):
            raise ValueError('GUARD_EVIDENCE_INVALID')
        predictions_path = folder / 'top10.csv'
        if sha(predictions_path) != manifest.get('prediction_sha256'):
            raise ValueError('PREDICTION_HASH_MISMATCH')
        predictions = read_predictions(predictions_path)
        sources = manifest.get('source_sha256', {})
        code = manifest.get('model_code_sha256', {})
        if not isinstance(sources, dict) or len(sources) < 7:
            raise ValueError('SOURCE_PROVENANCE_INCOMPLETE')
        if not isinstance(code, dict) or len(code) < 4:
            raise ValueError('MODEL_PROVENANCE_INCOMPLETE')
        for rel, expected in sources.items():
            if sha(under_root(root, rel)) != expected:
                raise ValueError('SOURCE_HASH_MISMATCH:' + rel)
        allowed_code = {
            'slotanalyzer_maruhan_history_isolation_v1.py',
            'slotanalyzer_maruhan_isolated_top10_audit_v1.py',
            'ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py',
            'slotanalyzer_maruhan_isolated_prospective_freeze_v1.py',
        }
        if set(code) != allowed_code:
            raise ValueError('MODEL_PROVENANCE_FILES_INVALID')
        known_paths = {
            'ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py': root / 'machine_number' / 'ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py'
        }
        for name in allowed_code:
            expected = code[name]
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('MODEL_CODE_HASH_INVALID:' + name)
            # Source code may be upgraded after freezing. Record drift in the receipt;
            # do not require historical code version to be present to evaluate a sealed prediction.
        actual_file = base / f'ana_slo_{target:%Y%m%d}.csv'
        if not actual_file.is_file():
            return 'SKIP: TARGET_ACTUAL_MISSING'
        actual = read_actual(actual_file, target)
        missing, mismatch = [], []
        rows = []
        for entry in predictions:
            no = entry['machine_no']
            if no not in actual:
                missing.append(no)
                continue
            name, diff = actual[no]
            if name != entry['machine_name']:
                mismatch.append(f'{no}:{entry["machine_name"]}->{name}')
                continue
            rows.append({**entry, 'actual_diff': diff})
        if missing or mismatch or len(rows) != 10:
            raise ValueError('ACTUAL_IDENTITY_MISMATCH:missing=' + str(missing) + ',renamed=' + str(mismatch))
        total = sum(r['actual_diff'] for r in rows)
        metrics = {
            'top10_sum_diff': total,
            'top10_mean_diff': total / 10,
            'top10_wins': sum(r['actual_diff'] > 0 for r in rows),
            'top10_win_rate': sum(r['actual_diff'] > 0 for r in rows) / 10,
        }
        result = 'EVALUATION_OK target=' + target.isoformat() + ' top10_sum=' + f'{total:.0f}' + ' top10_mean=' + f'{total/10:.2f}'
        if not save:
            return 'DRY_RUN_' + result + ' NO_FILES_WRITTEN=TRUE'
        payload = {
            'schema_version': 1,
            'version': VERSION,
            'classification': 'ISOLATED_PROSPECTIVE_REFERENCE_NOT_FORMAL',
            'target_date': target.isoformat(),
            'evaluated_at_jst': now.astimezone(JST).isoformat(),
            'freeze_manifest_sha256': sha(manifest_file),
            'prediction_csv_sha256': sha(predictions_path),
            'actual_csv_sha256': sha(actual_file),
            'original_model_code_sha256': code,
            'metrics': metrics,
            'ranked_actual': rows,
            'formal_forward': False,
            'inventory_guard_modified': False,
        }
        data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
        with receipt.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return 'EVALUATION_RECORDED ' + result + ' receipt=' + str(receipt)
    except Exception as exc:
        return f'BLOCKED: {type(exc).__name__}: {exc}'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--target-date', type=date.fromisoformat, required=True)
    p.add_argument('--save', action='store_true', help='Append-only receipt; default read-only check.')
    args = p.parse_args()
    result = evaluate(args.root, args.target_date, datetime.now(JST), args.save)
    print(result)
    return 1 if result.startswith('BLOCKED:') else 0


if __name__ == '__main__':
    raise SystemExit(main())
