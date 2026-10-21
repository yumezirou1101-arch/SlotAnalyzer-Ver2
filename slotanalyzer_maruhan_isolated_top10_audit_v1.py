"""Read-only historical isolated Top10 score audit. NOT a prospective prediction.

All history must predate target. Target day rows are synthetic join stubs only.
No files are written, no Guard changes, no Formal Forward changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import slotanalyzer_maruhan_history_isolation_v1 as iso

EXPECTED_COUNT = 514

def perform(root: Path, target: date):
    folder = root / 'data' / 'maruhan_maebashi' / 'machine_number'
    cutoff = target - timedelta(days=1)
    state_path = folder / 'inventory_guard_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('store') != 'MARUHAN_MAEBASHI':
        raise ValueError('GUARD_STORE_MISMATCH')
    audit = iso.analyze(root, target, EXPECTED_COUNT)
    if audit['inventory'] != EXPECTED_COUNT or audit['age7_count'] < 10:
        raise ValueError('INVALID_ELIGIBLE_SET')
    days = sorted(date.fromisoformat(f'{p.name[8:12]}-{p.name[12:14]}-{p.name[14:16]}') for p in folder.glob('ana_slo_????????.csv') if p.name[8:16].isdigit() and p.name[8:16] <= f'{cutoff:%Y%m%d}')
    if not days or days[-1] != cutoff or len(days) != len(set(days)):
        raise ValueError('SOURCE_DATES_INVALID')
    sources = {d: iso.daily(folder / f'ana_slo_{d:%Y%m%d}.csv', d, EXPECTED_COUNT) for d in days}
    current = sources[cutoff]
    records = []
    ages = {}
    for no, (name, _) in sorted(current.items()):
        day = cutoff
        while day in sources and no in sources[day] and sources[day][no][0] == name:
            val = sources[day][no][1]
            records.append(dict(date=pd.Timestamp(day), machine_no=no, machine_name=name, diff=val,
                                win=float(val > 0), plus1000=float(val >= 1000), plus2000=float(val >= 2000)))
            day -= timedelta(days=1)
        ages[no] = (cutoff-day).days
    if len(records) != audit['isolated_rows']:
        raise ValueError('ISOLATED_ROWS_MISMATCH')
    digest = hashlib.sha256()
    for r in sorted(records, key=lambda r: (r['date'], r['machine_no'])):
        digest.update(f"{r['date'].date().isoformat()}\t{r['machine_no']}\t{r['machine_name']}\t{r['diff']:.9f}\n".encode())
    if digest.hexdigest() != audit['fingerprint']:
        raise ValueError('ISOLATED_SOURCE_HASH_MISMATCH')
    eligible = {n for n, age in ages.items() if age >= 7}
    if len(eligible) != audit['age7_count']:
        raise ValueError('ELIGIBLE_MISMATCH')
    sys.path.insert(0, str(root / 'machine_number'))
    import ana_slo_prediction_v4_2_machine_number_position_ablation_oos as model
    hist = pd.DataFrame.from_records(records)
    if (hist['date'] >= pd.Timestamp(target)).any():
        raise ValueError('FUTURE_HISTORY')
    edge = model.build_number_edge_distance(sorted(current))
    def features(sentinel):
        target_rows = [dict(date=pd.Timestamp(target), machine_no=n, machine_name=current[n][0],
                            diff=sentinel, win=0., plus1000=0., plus2000=0.) for n in sorted(current)]
        full = pd.concat((hist, pd.DataFrame(target_rows)), ignore_index=True)
        panel = model.build_features(full, pd.Timestamp(target), edge)
        if len(panel) != EXPECTED_COUNT or panel.machine_no.duplicated().any():
            raise ValueError('FEATURE_COUNT_OR_DUPLICATE')
        result = panel[panel.machine_no.isin(eligible)].copy()
        if len(result) != len(eligible):
            raise ValueError('ELIGIBLE_PANEL_SIZE')
        # Synthetic actual diff is NEVER consumed by the scorer.
        return result.drop(columns=['diff']).sort_values('machine_no').reset_index(drop=True)
    f0, f1 = features(0.), features(9876543.)
    pd.testing.assert_frame_equal(f0, f1, check_exact=True)
    weights = dict(model.V42_C_WEIGHTS)
    expected_factors = {'avg31','recent7_avg','last_diff','prev_change','weekday_avg',
                        'type_avg','plus1000_rate','plus2000_rate','neighbor_avg'}
    if set(weights) != expected_factors or abs(sum(weights.values())-1.0)>1e-10:
        raise ValueError('MODEL_WEIGHTS_UNEXPECTED')
    if not np.isfinite(f0[list(weights)].to_numpy(dtype=float)).all():
        raise ValueError('NONFINITE_FEATURES')
    p0 = model.rank_score(f0, weights)
    p1 = model.rank_score(f1, weights)
    pd.testing.assert_frame_equal(p0.sort_values('machine_no').reset_index(drop=True),
                                  p1.sort_values('machine_no').reset_index(drop=True), check_exact=True)
    if not np.isfinite(p0['score'].to_numpy(dtype=float)).all():
        raise ValueError('NONFINITE_SCORE')
    ranking = p0.sort_values(['score','machine_no'], ascending=[False,True], kind='mergesort').head(10)
    print('MODE=READ_ONLY_ISOLATED_TOP10_AUDIT_V1')
    print(f'TARGET={target} HISTORICAL_RECONSTRUCTION=TRUE SOURCE_THROUGH={cutoff}')
    print(f'GUARD_STATUS={state.get("status")} REFERENCE_FREEZE_AUTHORIZED=FALSE')
    print(f'ISOLATED_INPUT_ROWS={len(records)} SOURCE_SHA256={digest.hexdigest()}')
    print(f'FEATURE_ROWS={len(f0)} AGE7_ELIGIBLE={len(eligible)} EXCLUDED={EXPECTED_COUNT-len(eligible)}')
    print('MODEL=V4.2_C_WEIGHTS_ISOLATED_HISTORY; SCORE_NORMALIZATION=AGE7_CANDIDATES_ONLY')
    print('SYNTHETIC_TARGET_SENTINEL_AND_SCORE=PASS')
    for rank, r in enumerate(ranking.itertuples(index=False), 1):
        print(f'TOP10 rank={rank} machine={int(r.machine_no)} name={r.machine_name} score={float(r.score):.6f} verified_days={ages[int(r.machine_no)]}')
    print('RESULT=OFFLINE_SCORE_AUDIT_ONLY; NOT_FORMAL_FORWARD; NOT_PROSPECTIVE_FREEZE')
    print('NO_FILES_WRITTEN=TRUE INVENTORY_GUARD=UNCHANGED CHAMPION=UNCHANGED')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--target-date',type=date.fromisoformat,required=True)
    args=parser.parse_args()
    try:
        perform(args.root.resolve(),args.target_date)
        return 0
    except Exception as e:
        print(f'SCORE_AUDIT=PRECHECK_FAIL {type(e).__name__}: {e}')
        return 2

if __name__=='__main__':
    raise SystemExit(main())
