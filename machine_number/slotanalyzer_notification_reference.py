"""Read-only D-1 comparisons for mail. Never writes evaluation or prediction files."""
from __future__ import annotations

import csv
import math
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from slotanalyzer_derived_prediction_evaluation import validate_actual_quality
from slotanalyzer_evaluation_quarantine import assess_evaluation_quarantine, load_registry, sha256_file


SPECS = {
    "NORMAL": ("64_Ver4_2_future_top10", "64_prediction", "prediction_rank"),
    "A_TYPE": ("74_Ver4_2_A_type_prediction", "74_A_type_prediction", "a_type_rank"),
    "JUGGLER": ("75_Ver4_2_Juggler_prediction", "75_Juggler_prediction", "juggler_rank"),
}


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def has_quarantine(root: Path, target, category: str) -> bool:
    return any(entry["store"] == "MARUHAN_MAEBASHI"
               and entry["target_date"] == target.isoformat() and entry["category"] == category
               for entry in load_registry(root))


def load_reference_result(root: Path, target, category: str, reason: str):
    """Return display fields only; missing/unsafe evidence is an explicit explanation."""
    base = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep"
    directory, prefix, rank_key = SPECS[category]
    prediction = base / directory / f"{prefix}_{target:%Y%m%d}_top10.csv"
    metadata = base / directory / f"{prefix}_{target:%Y%m%d}_metadata.csv"
    actual = base.parent / f"ana_slo_{target:%Y%m%d}.csv"
    if not prediction.is_file():
        return "REFERENCE_UNAVAILABLE", f"{reason} / 前日予想なし（{target.isoformat()}、古い日付へ代替しません）", [], []
    if not actual.is_file():
        return "REFERENCE_UNAVAILABLE", f"{reason} / 前日実績未取得", [], []
    try:
        source = base / "64_Ver4_2_future_top10"
        decision = assess_evaluation_quarantine(
            root, "MARUHAN_MAEBASHI", target, category, prediction, metadata,
            source / f"64_prediction_{target:%Y%m%d}_all514.csv",
            source / f"64_prediction_{target:%Y%m%d}_metadata.csv",
        )
        if decision.status == "MANUAL_REVIEW_QUARANTINE_SHA_MISMATCH":
            raise ValueError(f"quarantine証跡不整合: {decision.reason}")
        meta_rows = _rows(metadata)
        if len(meta_rows) != 1 or meta_rows[0].get("target_date") != target.isoformat():
            raise ValueError("予想metadata日付不一致")
        meta = meta_rows[0]
        generated = datetime.fromisoformat(meta["generated_at_jst"])
        if generated.utcoffset() is None:
            raise ValueError("朝凍結時刻のtimezone証跡なし")
        generated = generated.astimezone(ZoneInfo("Asia/Tokyo"))
        # Morning display evidence is separate from the formal 09:00 cutoff.
        # A 09:15 frozen prediction can be viewed, but never promoted here.
        if generated.date() != target or generated.time() >= time(12):
            raise ValueError("対象日午前の凍結証跡なし")
        digest = sha256_file(prediction)
        if meta.get("top10_sha256") and meta["top10_sha256"].lower() != digest:
            raise ValueError("予想SHA-256不一致")
        predictions = _rows(prediction)
        if len(predictions) != 10 or {int(r[rank_key]) for r in predictions} != set(range(1, 11)):
            raise ValueError("予想Top10順位不整合")
        if any(r.get("target_date") != target.isoformat() for r in predictions):
            raise ValueError("予想内部日付不一致")
        if len({int(r["machine_no"]) for r in predictions}) != 10:
            raise ValueError("予想台番号重複")
        actuals = validate_actual_quality(actual, target).set_index("machine_no")
        details = []
        unmatched = []
        for row in sorted(predictions, key=lambda r: int(r[rank_key])):
            number = int(row["machine_no"])
            if number not in actuals.index or str(actuals.loc[number, "machine_name"]) != row["machine_name"]:
                unmatched.append(str(number))
                continue
            diff = float(actuals.loc[number, "diff"])
            score = row.get("score", "")
            if not math.isfinite(diff) or (score and not math.isfinite(float(score))):
                raise ValueError("差枚/score非有限値")
            details.append(dict(row, prediction_rank=str(int(row[rank_key])),
                                actual_diff=str(diff), actual_win=str(int(diff > 0))))
        if not details:
            raise ValueError("台番号・機種名一致の実績なし")
        summaries = []
        for n in (3, 5, 10):
            selected = [float(r["actual_diff"]) for r in details if int(r["prediction_rank"]) <= n]
            if not selected:
                continue
            count = len(selected)
            summaries.append({"band": f"TOP{n}", "selected_n": str(count),
                "avg_diff": str(sum(selected) / count), "sum_diff": str(sum(selected)),
                "win_rate": str(100 * sum(x > 0 for x in selected) / count),
                "plus1000_rate": str(100 * sum(x >= 1000 for x in selected) / count),
                "plus2000_rate": str(100 * sum(x >= 2000 for x in selected) / count)})
        status = "REFERENCE / QUARANTINED" if decision.quarantined else "REFERENCE"
        message = f"正式Forward成績には加算しない / {reason}"
        if decision.quarantined:
            message += f" / evaluation quarantine / {decision.reason_code}: {decision.reason}"
        message += f" / 照合 {len(details)}/10台（台番号・機種名一致のみ）"
        if unmatched:
            message += f" / 未照合台: {', '.join(unmatched)}。集計は照合台のみ"
        return status, message, details, summaries
    except Exception as exc:
        return "REFERENCE_UNAVAILABLE", f"{reason} / 参考結果表示不可: {type(exc).__name__}: {exc}", [], []
