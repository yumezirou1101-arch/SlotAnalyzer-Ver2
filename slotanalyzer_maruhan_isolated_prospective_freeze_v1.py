"""Independent Maruhan identity-isolated reference freeze, never Formal Forward.

Default = read-only audit. --freeze only on the target date before 09:00 JST.
BLOCKED inventory state is NOT approved or modified; reference uses its own exact
source/identity checks. No effect on existing paired reference or Champion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import slotanalyzer_maruhan_history_isolation_v1 as iso
import slotanalyzer_maruhan_isolated_top10_audit_v1 as audit

JST = ZoneInfo('Asia/Tokyo')
EXPECTED_COUNT = 514
REL_OUT = Path('data/maruhan_maebashi/machine_number/analysis_isolated_prospective_reference_v1')
MODEL_REL = Path('machine_number/ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py')
SOURCE_RE = re.compile(r'^ana_slo_(\d{8})\.csv$')
TOP_RE = re.compile(r'^TOP10 rank=(\d+) machine=(\d+) name=(.*?) score=([\d.]+) verified_days=(\d+)$')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_guard(folder: Path, cutoff: date) -> dict:
    """Independent allow-list policy, NOT the formal Inventory Guard release."""
    p = folder / 'inventory_guard_state.json'
    state = json.loads(p.read_text(encoding='utf-8'))
    if state.get('store') != 'MARUHAN_MAEBASHI':
        raise ValueError('GUARD_STORE_MISMATCH')
    if state.get('status') not in ('BLOCKED', 'APPROVED'):
        raise ValueError('GUARD_STATE_UNRECOGNIZED')
    if state.get('incident_kind') not in ('ACTUAL_CHANGE', ''):
        raise ValueError('GUARD_INCIDENT_UNSUPPORTED')
    change_date = date.fromisoformat(state['change_date'])
    previous_date = date.fromisoformat(state['previous_date'])
    if change_date - previous_date != timedelta(days=1):
        raise ValueError('GUARD_INCIDENT_DATE_GAP')
    if change_date > cutoff:
        raise ValueError('GUARD_INCIDENT_AFTER_SOURCE_CUTOFF')
    old = iso.daily(folder / f'ana_slo_{previous_date:%Y%m%d}.csv', previous_date, EXPECTED_COUNT)
    new = iso.daily(folder / f'ana_slo_{change_date:%Y%m%d}.csv', change_date, EXPECTED_COUNT)
    old_set, new_set = set(old), set(new)
    renamed = [{'machine_no': n, 'previous_machine_name': old[n][0], 'current_machine_name': new[n][0]}
               for n in sorted(old_set & new_set) if old[n][0] != new[n][0]]
    if (sorted(old_set - new_set) != state.get('removed_machine_numbers') or
        sorted(new_set - old_set) != state.get('added_machine_numbers') or
        renamed != state.get('renamed_machine_numbers') or
        len(old) != state.get('previous_machine_count') or
        len(new) != state.get('current_machine_count')):
        raise ValueError('GUARD_INCIDENT_DIFF_MISMATCH')
    # Detect unrecorded identity changes after the persisted incident.
    for d in (cutoff - timedelta(days=1), cutoff):
        if d <= change_date:
            continue
        first = iso.daily(folder / f'ana_slo_{d:%Y%m%d}.csv', d, EXPECTED_COUNT)
        second_date = d - timedelta(days=1)
        second = iso.daily(folder / f'ana_slo_{second_date:%Y%m%d}.csv', second_date, EXPECTED_COUNT)
        if {n: first[n][0] for n in first} != {n: second[n][0] for n in second}:
            raise ValueError(f'NEWER_UNRECORDED_IDENTITY_CHANGE:{d}')
    return {'state': state['status'], 'incident_date': change_date.isoformat(), 'record_match': True, 'state_sha256': sha256(p)}


def source_files(folder: Path, cutoff: date) -> list[Path]:
    result = []
    for p in folder.iterdir():
        m = SOURCE_RE.fullmatch(p.name)
        if m and date(int(m[1][:4]), int(m[1][4:6]), int(m[1][6:])) <= cutoff:
            result.append(p)
    result.sort()
    if not result or result[-1].stem != f'ana_slo_{cutoff:%Y%m%d}':
        raise ValueError('LATEST_DAILY_CSV_MISSING')
    return result


def run(root: Path, target: date, freeze: bool):
    now = datetime.now(JST)
    if freeze and (now.date() != target or now.time() >= time(9, 0)):
        raise ValueError(f'FREEZE_WINDOW_EXPIRED_OR_WRONG_DAY:JST_NOW={now.isoformat()}')
    folder = root / 'data/maruhan_maebashi/machine_number'
    cutoff = target - timedelta(days=1)
    paths = source_files(folder, cutoff)
    guard = validate_guard(folder, cutoff)
    scripts = [root / 'slotanalyzer_maruhan_history_isolation_v1.py',
               root / 'slotanalyzer_maruhan_isolated_top10_audit_v1.py',
               root / MODEL_REL,
               Path(__file__).resolve()]
    if any(not p.is_file() for p in scripts):
        raise ValueError('DEPENDENCY_MISSING')
    before = {str(p.relative_to(root)): sha256(p) for p in paths}
    code_hashes = {p.name: sha256(p) for p in scripts}
    # Reuse the exact scorer and its sentinel comparison validated by the user.
    out = io.StringIO()
    with redirect_stdout(out):
        audit.perform(root, target)
    content = out.getvalue()
    if 'SYNTHETIC_TARGET_SENTINEL_AND_SCORE=PASS' not in content or 'NO_FILES_WRITTEN=TRUE' not in content:
        raise ValueError('TOP10_AUDIT_NOT_CLEAN')
    parsed = [TOP_RE.fullmatch(line) for line in content.splitlines() if line.startswith('TOP10 ')]
    if len(parsed) != 10 or any(m is None for m in parsed):
        raise ValueError('TOP10_PARSE_FAILED')
    picks = [{'rank': int(m[1]), 'machine_no': int(m[2]), 'machine_name': m[3],
              'score': float(m[4]), 'verified_days': int(m[5])} for m in parsed]
    if [x['rank'] for x in picks] != list(range(1, 11)) or len({x['machine_no'] for x in picks}) != 10:
        raise ValueError('TOP10_DUPLICATE_OR_BAD_RANK')
    if before != {str(p.relative_to(root)): sha256(p) for p in paths}:
        raise ValueError('SOURCE_CHANGED_DURING_AUDIT')
    if guard['state_sha256'] != sha256(folder / 'inventory_guard_state.json'):
        raise ValueError('GUARD_CHANGED_DURING_AUDIT')
    if code_hashes != {p.name: sha256(p) for p in scripts}:
        raise ValueError('MODEL_CODE_CHANGED_DURING_AUDIT')
    print('MODE=' + ('PROSPECTIVE_REFERENCE_FREEZE' if freeze else 'READ_ONLY_PRECHECK'))
    print(f'TARGET={target} SOURCE_THROUGH={cutoff} GUARD={guard["state"]} RECORD_MATCH=TRUE')
    print('MODEL=V4.2_C_WEIGHTS_IDENTITY_ISOLATED_NOT_CHAMPION')
    for pick in picks:
        print(f'TOP10 rank={pick["rank"]} machine={pick["machine_no"]} name={pick["machine_name"]} score={pick["score"]:.6f}')
    if not freeze:
        print('PRECHECK=PASS NO_FILES_WRITTEN=TRUE REFERENCE_FREEZE_AUTHORIZED=FALSE')
        return
    destination = root / REL_OUT / f'{target:%Y%m%d}'
    if destination.exists():
        raise ValueError('FROZEN_DATE_ALREADY_EXISTS')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f'.freeze_{target:%Y%m%d}_', dir=destination.parent))
    try:
        top_file = temp_dir / 'top10.csv'
        with top_file.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['rank', 'machine_no', 'machine_name', 'score', 'verified_days'])
            w.writeheader()
            w.writerows(picks)
        manifest = dict(schema_version=1, kind='ISOLATED_PROSPECTIVE_REFERENCE_NOT_FORMAL',
                        target_date=target.isoformat(), data_through=cutoff.isoformat(), frozen_at_jst=now.isoformat(),
                        inventory_guard_not_approved_by_this_script=True, guard_evidence=guard,
                        independent_reference_policy='EXACT_514_AND_RECORDED_INCIDENT_AND_IDENTITY_ISOLATION_AGE7',
                        source_sha256=before, model_code_sha256=code_hashes,
                        prediction_sha256=sha256(top_file), top10_count=len(picks),
                        formal_forward=False, champion_modified=False)
        (temp_dir / 'freeze_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if datetime.now(JST).date() != target or datetime.now(JST).time() >= time(9, 0):
            raise ValueError('FREEZE_TIME_EXPIRED_BEFORE_COMMIT')
        if destination.exists():
            raise ValueError('CONCURRENT_FREEZE_EXISTS')
        os.rename(temp_dir, destination)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    print(f'FROZEN={destination}')
    print('RESULT=PROSPECTIVE_REFERENCE_NOT_FORMAL; FORMAL_FORWARD_UNCHANGED; INVENTORY_GUARD_UNCHANGED')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--target-date', type=date.fromisoformat, required=True)
    p.add_argument('--freeze', action='store_true', help='Write once, only before 09:00 JST on target day.')
    args = p.parse_args()
    try:
        run(args.root.resolve(), args.target_date, args.freeze)
        return 0
    except Exception as exc:
        print(f'ISOLATED_REFERENCE=PRECHECK_FAIL {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
