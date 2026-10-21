"""Read-only eligibility audit for a separate Maruhan reference model.

Does not authorize any freeze or bypass the production Inventory Guard.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

EXPECTED = 514
MIN_DAYS = 7


def load(path: Path, day: date):
    if not path.is_file():
        raise RuntimeError(f'MISSING_DAY={day}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        keys = reader.fieldnames or []
        no = 'machine_no' if 'machine_no' in keys else '台番号' if '台番号' in keys else None
        name = 'machine_name' if 'machine_name' in keys else '機種名' if '機種名' in keys else None
        rowdate = 'date' if 'date' in keys else '日付' if '日付' in keys else None
        if not no or not name or not rowdate:
            raise RuntimeError(f'INVALID_COLUMNS={path.name}: requires number, name, date')
        result = {}
        for row in reader:
            try:
                n = int(str(row[no]).strip())
                nm = str(row[name]).strip()
                actual_day = date.fromisoformat(str(row[rowdate]).strip())
            except (ValueError, TypeError, KeyError) as exc:
                raise RuntimeError(f'INVALID_ROW={path.name}') from exc
            if not nm or actual_day != day or n in result:
                raise RuntimeError(f'INVALID_IDENTITY_OR_DUPLICATE={path.name}:{n}')
            result[n] = nm
    if len(result) != EXPECTED:
        raise RuntimeError(f'INVALID_COUNT={path.name}:{len(result)} expected {EXPECTED}')
    return result, digest


def run(root: Path, target: date):
    directory = root / 'data' / 'maruhan_maebashi' / 'machine_number'
    latest = target - timedelta(days=1)
    days = [latest - timedelta(days=i) for i in range(MIN_DAYS)]
    inventories = {}
    hashes = {}
    for d in days:
        inventories[d], hashes[d.isoformat()] = load(directory / f'ana_slo_{d:%Y%m%d}.csv', d)
    guard_path = directory / 'inventory_guard_state.json'
    if not guard_path.is_file():
        raise RuntimeError('GUARD_STATE_MISSING; no bypass permitted')
    guard = json.loads(guard_path.read_text(encoding='utf-8'))
    if guard.get('store') != 'MARUHAN_MAEBASHI':
        raise RuntimeError('GUARD_STORE_MISMATCH')
    # Explicitly distinguish a legitimate unchanged day from a blocked remodel.
    previous = inventories[latest-timedelta(days=1)]
    current = inventories[latest]
    renames = sorted(n for n in current if n in previous and current[n] != previous[n])
    added = sorted(set(current) - set(previous))
    removed = sorted(set(previous) - set(current))
    blocked = guard.get('status') == 'BLOCKED'
    incident_match = (guard.get('change_date') == latest.isoformat()
                      and guard.get('previous_date') == (latest-timedelta(days=1)).isoformat()
                      and guard.get('previous_machine_count') == EXPECTED
                      and guard.get('current_machine_count') == EXPECTED
                      and guard.get('added_machine_numbers') == added
                      and guard.get('removed_machine_numbers') == removed
                      and guard.get('renamed_machine_numbers') == [
                          {'machine_no': n, 'previous_machine_name': previous[n], 'current_machine_name':current[n]}
                          for n in renames])
    eligible = []
    excluded = []
    for n, name in sorted(current.items()):
        age = 0
        for d in days:
            if inventories[d].get(n) != name:
                break
            age += 1
        if age >= MIN_DAYS:
            eligible.append(n)
        else:
            excluded.append((n, age, name))
    print('MODE=READ_ONLY_REFERENCE_ELIGIBILITY_V1')
    print('FORMAL_FORWARD=UNCHANGED INVENTORY_GUARD=UNCHANGED')
    print(f'TARGET={target} DATA_THROUGH={latest} DAYS_REQUIRED={MIN_DAYS}')
    print(f'GUARD_STATUS={guard.get("status", "MISSING")}')
    print(f'LATEST_TRANSITION=ADDED:{len(added)} REMOVED:{len(removed)} RENAMED:{len(renames)}')
    print(f'GUARD_LATEST_INCIDENT_MATCH={str(incident_match).upper()}')
    print(f'ELIGIBLE_AGE7={len(eligible)} EXCLUDED_AGE7={len(excluded)}')
    for n, age, name in excluded:
        print(f'EXCLUDED_MACHINE={n} CONTIGUOUS_IDENTITY_DAYS={age} NAME={name}')
    for d, digest in sorted(hashes.items()):
        print(f'SOURCE_SHA256={d}:{digest}')
    if added or removed or (blocked and not incident_match and guard.get('change_date') == latest.isoformat()):
        print('AUDIT_RESULT=REVIEW_REQUIRED_INVENTORY_IDENTITY')
    elif blocked:
        print('AUDIT_RESULT=ELIGIBILITY_EVIDENCE_ONLY_GUARD_STILL_BLOCKED')
    else:
        print('AUDIT_RESULT=ELIGIBILITY_EVIDENCE_ONLY')
    print('MODEL_UNRESET_HISTORY=NOT_VERIFIED_SAFE')
    print('REFERENCE_FREEZE_AUTHORIZED=FALSE')


def main():
    parser = argparse.ArgumentParser(description='Read-only age-7 Maruhan reference eligibility audit; never lifts inventory guard')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--target-date', required=True, type=date.fromisoformat)
    args = parser.parse_args()
    try:
        run(args.root.resolve(), args.target_date)
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'AUDIT_RESULT=PRECHECK_FAIL {type(exc).__name__}: {exc}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
