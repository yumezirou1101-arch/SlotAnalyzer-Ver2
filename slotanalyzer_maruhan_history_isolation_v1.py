"""Read-only Maruhan identity-isolated feature-source audit.

Creates NO files. Does NOT authorize any prediction or bypass Inventory Guard.
For each machine use only the maximal uninterrupted same-name run ending cutoff.
Global averages are computed only from rows that passed the same rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

DATE_RE = re.compile(r'^ana_slo_(\d{8})\.csv$')
EXPECTED = 514


def daily(path: Path, expected: date, expected_count: int) -> dict[int, tuple[str, float]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        keys = set(reader.fieldnames or ())
        def col(*names):
            return next((x for x in names if x in keys), None)
        ncol, mcol, dcol, diffcol = (col('machine_no', '台番号'), col('machine_name', '機種名'),
                                     col('date', '日付'), col('diff', '差枚', '差枚数', '差枚(枚)'))
        if not all((ncol, mcol, dcol, diffcol)):
            raise ValueError(f'MISSING_COLUMNS:{path.name}:diff_column={diffcol}')
        data = {}
        for row in reader:
            raw_no = str(row[ncol]).strip()
            if not raw_no.isdigit():
                raise ValueError(f'INVALID_MACHINE_NO:{path.name}:{raw_no}')
            no = int(raw_no)
            name = str(row[mcol]).strip()
            try:
                dt = date.fromisoformat(str(row[dcol]).strip())
                raw = str(row[diffcol]).strip().replace(',', '').replace('＋', '+').replace('－', '-')
                value = float(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'INVALID_DATE_OR_DIFF:{path.name}:{no}') from exc
            if not name or dt != expected or no in data or not (-1e8 < value < 1e8):
                raise ValueError(f'INVALID_ROW:{path.name}:{no}')
            data[no] = (name, value)
        if len(data) != expected_count:
            raise ValueError(f'UNEXPECTED_MACHINE_COUNT:{path.name}:{len(data)}!= {expected_count}')
        return data


def analyze(root: Path, target: date, expected: int = EXPECTED) -> dict:
    folder = root / 'data' / 'maruhan_maebashi' / 'machine_number'
    cutoff = target - timedelta(days=1)
    paths = {}
    for f in folder.glob('ana_slo_*.csv'):
        m = DATE_RE.fullmatch(f.name)
        if m:
            d = datetime.strptime(m.group(1), '%Y%m%d').date()
            if d <= cutoff:
                paths[d] = f
    if not paths or cutoff not in paths:
        raise ValueError('CUTOFF_MISSING')
    source = {d: daily(paths[d], d, expected) for d in sorted(paths)}
    last7 = [cutoff - timedelta(days=i) for i in range(6, -1, -1)]
    if any(d not in source for d in last7):
        raise ValueError('LAST7_MISSING')
    guard_path = folder / 'inventory_guard_state.json'
    if not guard_path.exists():
        raise ValueError('GUARD_MISSING')
    guard = json.loads(guard_path.read_text(encoding='utf-8'))
    if guard.get('store') != 'MARUHAN_MAEBASHI':
        raise ValueError('GUARD_STORE_MISMATCH')
    current = source[cutoff]
    stats = Counter()
    sanitized = []
    starts = {}
    for no, (name, _) in sorted(current.items()):
        day = cutoff
        length = 0
        while day in source and no in source[day] and source[day][no][0] == name:
            sanitized.append((day, no, name, source[day][no][1]))
            length += 1
            day -= timedelta(days=1)
        starts[no] = day + timedelta(days=1)
        if length >= 7:
            stats['eligible_age7'] += 1
        else:
            stats['excluded_age7'] += 1
        if day not in source and day >= min(source):
            stats['ended_at_missing_day'] += 1
        elif day in source and no in source[day] and source[day][no][0] != name:
            stats['ended_at_change'] += 1
        else:
            stats['ended_at_source_start_or_absence'] += 1
    # The invariant checked below is the critical distinction from simply filtering candidate machines:
    # every row supplied to ANY downstream statistic must belong to its current verified run.
    for d, no, name, _ in sanitized:
        if d < starts[no] or current[no][0] != name:
            raise AssertionError('HISTORY_LEAK_IN_SANITIZED_ROWS')
    type_groups = {}
    band10_groups = {}
    band50_groups = {}
    for _, no, name, value in sanitized:
        type_groups.setdefault(name, []).append(value)
        band10_groups.setdefault(no // 10 * 10, []).append(value)
        band50_groups.setdefault(no // 50 * 50, []).append(value)
    # Reproducible fingerprint for read-only isolated input; no predictive model is run.
    fingerprint = hashlib.sha256()
    for d, no, name, value in sorted(sanitized):
        fingerprint.update(f'{d.isoformat()}\t{no}\t{name}\t{value:.9f}\n'.encode('utf-8'))
    all_days = {min(source) + timedelta(days=i) for i in range((cutoff-min(source)).days+1)}
    missing = sorted(all_days - set(source))
    return dict(target=target.isoformat(), cutoff=cutoff.isoformat(), source_days=len(source),
                missing_dates=[d.isoformat() for d in missing], inventory=len(current),
                guard_status=guard.get('status', 'UNKNOWN'), stats=dict(stats),
                isolated_rows=len(sanitized), type_groups=len(type_groups),
                band10_groups=len(band10_groups), band50_groups=len(band50_groups),
                fingerprint=fingerprint.hexdigest(), age7_count=sum(
                    all(source[d][no][0] == name for d in last7) for no, (name, _) in current.items()
                ))


def main():
    p = argparse.ArgumentParser(description='Read-only current-identity isolation audit; no prediction or guard approval.')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--target-date', type=date.fromisoformat, required=True)
    p.add_argument('--expected-machines', type=int, default=EXPECTED, help='Only for synthetic fixture tests; production=514')
    args = p.parse_args()
    try:
        if args.expected_machines <= 0:
            raise ValueError('INVALID_EXPECTED_MACHINE_COUNT')
        r = analyze(args.root.resolve(), args.target_date, args.expected_machines)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f'ISOLATION_AUDIT=PRECHECK_FAIL {type(e).__name__}: {e}')
        return 2
    print('MODE=READ_ONLY_IDENTITY_ISOLATION_AUDIT_V1')
    print(f'TARGET={r["target"]} SOURCE_THROUGH={r["cutoff"]} SOURCE_DAYS={r["source_days"]}')
    print(f'INVENTORY={r["inventory"]} AGE7_ELIGIBLE={r["stats"].get("eligible_age7",0)} AGE7_EXCLUDED={r["stats"].get("excluded_age7",0)}')
    print(f'MISSING_SOURCE_DAYS={len(r["missing_dates"])} DATES={",".join(r["missing_dates"])}')
    print(f'ISOLATED_INPUT_ROWS={r["isolated_rows"]} TYPE_GROUPS={r["type_groups"]} BAND10_GROUPS={r["band10_groups"]} BAND50_GROUPS={r["band50_groups"]}')
    print(f'STOP_AT_IDENTITY_CHANGE={r["stats"].get("ended_at_change",0)} STOP_AT_MISSING_DAY={r["stats"].get("ended_at_missing_day",0)}')
    print(f'ISOLATED_INPUT_SHA256={r["fingerprint"]}')
    print(f'GUARD_STATUS={r["guard_status"]}')
    print('GLOBAL_INPUT_ISOLATION=PASS_FOR_THIS_AUDIT_ONLY')
    print('REAL_PREDICTION_MODEL_INTEGRATION=NOT_TESTED')
    print('REFERENCE_FREEZE_AUTHORIZED=FALSE FORMAL_FORWARD=UNCHANGED INVENTORY_GUARD=UNCHANGED')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
