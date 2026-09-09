from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import pandas as pd


ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = ROOT / "data/maruhan_maebashi/machine_number"
OUTPUT_DIR = DATA_DIR / "analysis_31days_deep/research_inventory_epoch_stage_e"
SHADOW_PATH = ROOT / "machine_number/ana_slo_prediction_v4_2_inventory_epoch_shadow.py"
TOP_NS = (1, 3, 5, 10)
THRESHOLDS = (7, 10, 14, 21)
POLICY_BASELINE = "BASELINE"
POLICY1 = "EPOCH_LT14_RANK_EXCLUDE_FULL_ZPOP"
P2_NAMES = {value: f"P2_T{value}" for value in THRESHOLDS}


def _load_shadow():
    spec = importlib.util.spec_from_file_location("inventory_epoch_shadow_stage_e", SHADOW_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(SHADOW_PATH)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shadow = _load_shadow()


@dataclass(frozen=True)
class StoreEvent:
    event_id: str
    previous_date: date
    change_date: date
    added: tuple[int, ...]
    removed: tuple[int, ...]
    renamed: tuple[int, ...]
    affected_ratio: float
    size: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame[columns]
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def discover_daily_paths(data_dir: Path = DATA_DIR) -> dict[date, Path]:
    paths = {}
    for path in sorted(Path(data_dir).glob("ana_slo_????????.csv")):
        token = path.stem.removeprefix("ana_slo_")
        if len(token) != 8 or not token.isdigit():
            continue
        day = datetime.strptime(token, "%Y%m%d").date()
        if day in paths:
            raise ValueError(f"Duplicate daily date: {day}")
        paths[day] = path
    if not paths:
        raise ValueError("No daily files found.")
    return paths


def load_daily(path: Path, day: date, maximum: date):
    return shadow.read_daily_csv(path, day, maximum)


def load_history(paths: dict[date, Path], as_of: date) -> pd.DataFrame:
    frames = [load_daily(paths[day], day, as_of) for day in sorted(paths) if day <= as_of]
    if not frames:
        raise ValueError(f"No history through {as_of}")
    return pd.concat(frames, ignore_index=True)


def detect_store_events(paths: dict[date, Path]) -> tuple[list[StoreEvent], dict[date, object]]:
    dates = sorted(paths)
    snapshots = {}
    events = []
    for day in dates:
        snapshots[day] = load_daily(paths[day], day, day)
    for previous, current in zip(dates, dates[1:]):
        if current - previous != timedelta(days=1):
            continue
        diff = shadow.compare_inventories(snapshots[previous], snapshots[current])
        added = tuple(sorted(diff.machine_numbers("added")))
        removed = tuple(sorted(diff.machine_numbers("removed")))
        renamed = tuple(sorted(diff.machine_numbers("renamed")))
        if not (added or removed or renamed):
            continue
        affected = len(added) + len(removed) + len(renamed)
        denominator = max(diff.previous_machine_count, diff.current_machine_count, 1)
        ratio = affected / denominator
        size = "SMALL" if ratio < .01 else "MEDIUM" if ratio < .10 else "LARGE"
        events.append(StoreEvent(
            f"MARUHAN_MAEBASHI_{current:%Y%m%d}", previous, current,
            added, removed, renamed, ratio, size,
        ))
    return events, snapshots


def primary_targets(events: list[StoreEvent], paths: dict[date, Path]):
    rows = []
    for index, event in enumerate(events):
        next_date = events[index + 1].change_date if index + 1 < len(events) else None
        primary_end = event.change_date + timedelta(days=14)
        if next_date is not None:
            primary_end = min(primary_end, next_date - timedelta(days=1))
        # Beyond DAY14 we retain only secondary re-entry evidence, stopping
        # before the next store event so event regimes never overlap.
        tracking_end = next_date - timedelta(days=1) if next_date else max(paths)
        for target in (event.change_date + timedelta(days=n) for n in range(1, (tracking_end - event.change_date).days + 1)):
            as_of = target - timedelta(days=1)
            available = target in paths and as_of in paths
            rows.append({
                "event_id": event.event_id,
                "change_date": event.change_date,
                "target_date": target,
                "as_of_date": as_of,
                "event_calendar_day": (target - event.change_date).days,
                "day_bucket": day_bucket((target - event.change_date).days),
                "evaluation_available": available,
                "skip_reason": "" if available else "MISSING_T_OR_T_MINUS_1_DAILY",
                "independent_primary": target <= primary_end,
            })
    return pd.DataFrame(rows)


def day_bucket(day: int) -> str:
    if day == 1:
        return "DAY1"
    if day == 2:
        return "DAY2"
    if day == 3:
        return "DAY3"
    if 4 <= day <= 7:
        return "DAY4-7"
    if 8 <= day <= 14:
        return "DAY8-14"
    return "SECONDARY"


def ranking_digest(stage_by_policy: dict[str, tuple[object, str]]) -> str:
    rows = []
    for policy, (stage, source_policy) in sorted(stage_by_policy.items()):
        normal = stage.normal.policies[source_policy]["final"]
        for row in normal.itertuples():
            rows.append((policy, "NORMAL", int(row.machine_no), float(row.score), int(row.final_candidate_rank)))
        for category in ("A-TYPE", "JUGGLER"):
            frame = stage.categories[category][source_policy]["candidates"]
            for row in frame.itertuples():
                if float(row.category_score) != float(row.normal_score):
                    raise AssertionError("category_score differs from NORMAL score")
                rows.append((policy, category, int(row.machine_no), float(row.score), int(row.category_rank)))
    return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()


def _event_for_changed_machine(events, machine_no: int, epoch_start_date, current_event_id: str):
    """Return the owning event only while that event is the active study regime."""
    epoch_start = shadow.normalize_day(epoch_start_date)
    event = next(
        (
            candidate
            for candidate in reversed(events)
            if candidate.change_date == epoch_start and int(machine_no) in candidate.renamed
        ),
        None,
    )
    if event is None or event.event_id != current_event_id:
        return None
    return event

def _policy_stages(history, target: date, as_of: date, active_diff):
    stage14 = shadow.build_stage_d_shadow(history, target, as_of, active_diff, 14)
    stages = {
        POLICY_BASELINE: (stage14, shadow.POLICY_BASELINE),
        POLICY1: (stage14, shadow.POLICY1),
        "P2_T14": (stage14, shadow.POLICY2),
    }
    for threshold in (7, 10, 21):
        stage = shadow.build_stage_d_shadow(history, target, as_of, active_diff, threshold)
        stages[P2_NAMES[threshold]] = (stage, shadow.POLICY2)
    return stages


def _rank_frame(stage, source_policy: str, category: str) -> pd.DataFrame:
    if category == "NORMAL":
        frame = stage.normal.policies[source_policy]["final"].copy()
        frame["rank"] = frame["final_candidate_rank"].astype(int)
    else:
        frame = stage.categories[category][source_policy]["candidates"].copy()
        frame["rank"] = frame["category_rank"].astype(int)
    return frame.sort_values("rank").reset_index(drop=True)


def _metric_rows(joined, context):
    rows = []
    for top_n in TOP_NS:
        selected = joined.head(top_n)
        values = selected.actual_diff.astype(float)
        negative = values[values < 0]
        rows.append({**context, "top_n": top_n, "selected_count": len(values),
            "actual_total_diff": float(values.sum()),
            "avg_diff_per_selected_machine": float(values.mean()),
            "median_diff": float(values.median()), "win_count": int((values > 0).sum()),
            "win_rate": float((values > 0).mean()), "plus1000_count": int((values >= 1000).sum()),
            "plus1000_rate": float((values >= 1000).mean()), "plus2000_count": int((values >= 2000).sum()),
            "plus2000_rate": float((values >= 2000).mean()), "loss_count": int((values <= 0).sum()),
            "negative_total_diff": float(negative.sum()),
            "downside_average": float(negative.mean()) if len(negative) else 0.0,
            "worst_machine_diff": float(values.min()), "daily_total_diff": float(values.sum()),
            "positive_day": bool(values.sum() > 0)})
    return rows


def _replacement_rows(ranks, actual, context):
    rows = []
    base = ranks[(POLICY_BASELINE, context["category"])]
    actual_map = actual.set_index("machine_no").actual_diff.astype(float).to_dict()
    for policy in sorted({key[0] for key in ranks if key[0] != POLICY_BASELINE}):
        other = ranks[(policy, context["category"])]
        for top_n in TOP_NS:
            left = base.head(top_n).machine_no.astype(int).tolist()
            right = other.head(top_n).machine_no.astype(int).tolist()
            common = set(left) & set(right)
            left_only, right_only = set(left) - set(right), set(right) - set(left)
            rows.append({**context, "policy": policy, "top_n": top_n,
                "common_count": len(common), "baseline_only": "|".join(map(str, sorted(left_only))),
                "policy_only": "|".join(map(str, sorted(right_only))),
                "replacement_count": len(left_only),
                "order_changes": sum(left.index(no) != right.index(no) for no in common),
                "replacement_actual_total": sum(actual_map[no] for no in right_only),
                "removed_selection_actual_total": sum(actual_map[no] for no in left_only),
                "replacement_gain_loss": sum(actual_map[no] for no in right_only) - sum(actual_map[no] for no in left_only)})
    return rows


def _ripple_rows(stages, category, context, changed):
    rows = []
    base = _rank_frame(*stages[POLICY_BASELINE], category).set_index("machine_no")
    for policy, pair in stages.items():
        if policy == POLICY_BASELINE:
            continue
        other = _rank_frame(*pair, category).set_index("machine_no")
        common = sorted((set(base.index) & set(other.index)) - changed)
        score = (other.loc[common, "score"] - base.loc[common, "score"]).abs()
        rank = (other.loc[common, "rank"] - base.loc[common, "rank"]).abs()
        row = {**context, "category": category, "policy": policy, "unchanged_common_n": len(common),
            "mean_abs_score_delta": float(score.mean()), "median_abs_score_delta": float(score.median()),
            "p95_abs_score_delta": float(score.quantile(.95)), "max_abs_score_delta": float(score.max()),
            "mean_abs_rank_delta": float(rank.mean()), "p95_abs_rank_delta": float(rank.quantile(.95)),
            "max_abs_rank_delta": float(rank.max()), "rank_changed_count": int((rank > 0).sum())}
        for n in TOP_NS:
            left, right = base.head(n).index.tolist(), other.head(n).index.tolist()
            common_top = set(left) & set(right)
            row[f"top{n}_replacement_count"] = len(set(left) - set(right))
            row[f"top{n}_common"] = "|".join(map(str, sorted(common_top)))
            row[f"top{n}_baseline_only"] = "|".join(map(str, sorted(set(left) - set(right))))
            row[f"top{n}_policy_only"] = "|".join(map(str, sorted(set(right) - set(left))))
            row[f"top{n}_order_changes"] = sum(left.index(no) != right.index(no) for no in common_top)
        rows.append(row)
    return rows


def _normalization_rows(stages, context):
    rows = []
    for policy, (stage, source_policy) in stages.items():
        z = stage.normal.z_audit[stage.normal.z_audit.policy == source_policy]
        scored = stage.normal.policies[source_policy]["scored"]
        final = stage.normal.policies[source_policy]["final"]
        components = stage.normal.component_audit[stage.normal.component_audit.policy == source_policy]
        for item in z.itertuples():
            values = scored.loc[scored.z_population, item.feature].astype(float)
            component = components[components.feature == item.feature].component
            scores = scored.loc[scored.z_population, "score"].astype(float)
            rows.append({**context, "policy": policy, "feature": item.feature,
                "z_population_size": int(item.population_n), "excluded_count": len(scored) - int(item.population_n),
                "excluded_ratio": (len(scored) - int(item.population_n)) / len(scored), "candidate_size": len(final),
                "feature_mean": float(item.mean), "feature_std": float(item.std),
                "feature_min": float(values.min()), "feature_max": float(values.max()),
                "feature_median": float(values.median()), "zero_std": bool(item.std == 0),
                "score_mean": float(scores.mean()), "score_std": float(scores.std(ddof=0)),
                "score_min": float(scores.min()), "score_max": float(scores.max()),
                "score_q05": float(scores.quantile(.05)), "score_q50": float(scores.quantile(.5)),
                "score_q95": float(scores.quantile(.95)),
                "component_clip_low_count": int((component <= 0).sum()),
                "component_clip_high_count": int((component >= 100).sum())})
    return rows


def run(data_dir: Path = DATA_DIR, output_dir: Path | None = OUTPUT_DIR, write_outputs: bool = True):
    started = time.perf_counter()
    paths = discover_daily_paths(data_dir)
    events, snapshots = detect_store_events(paths)
    if len(events) != 5 or sum(len(event.renamed) for event in events) != 223:
        raise AssertionError("Expected five store events and 223 renamed machines.")
    target_plan = primary_targets(events, paths)
    event_map = {event.event_id: event for event in events}
    results, replacements, changed_rows, ripple_rows, normalization_rows, leakage_rows = [], [], [], [], [], []
    rank_cache = {}
    for plan in target_plan[target_plan.evaluation_available].itertuples():
        history = load_history(paths, plan.as_of_date)
        active_diff = shadow.build_persistent_inventory_diff(history, plan.as_of_date)
        stages = _policy_stages(history, plan.target_date, plan.as_of_date, active_diff)
        pre_digest = ranking_digest(stages)
        ranks = {(policy, category): _rank_frame(stage, source, category)
                 for policy, (stage, source) in stages.items() for category in ("NORMAL", "A-TYPE", "JUGGLER")}
        rank_cache[(plan.target_date, plan.event_id)] = (stages, ranks, active_diff)
        actual_frame = load_daily(paths[plan.target_date], plan.target_date, plan.target_date)
        actual = actual_frame[["machine_no", "machine_name", "diff"]].rename(
            columns={"machine_name": "actual_machine_name", "diff": "actual_diff"})
        if actual.machine_no.duplicated().any():
            raise AssertionError("Actual duplicate machine_no")
        if ranking_digest(stages) != pre_digest:
            raise AssertionError("Ranking mutated after actual load")
        context = {"event_id": plan.event_id, "target_date": plan.target_date,
            "event_calendar_day": plan.event_calendar_day, "day_bucket": plan.day_bucket,
            "independent_primary": bool(plan.independent_primary)}
        for category in ("NORMAL", "A-TYPE", "JUGGLER"):
            for policy in stages:
                joined = ranks[(policy, category)].merge(actual, on="machine_no", how="left", validate="one_to_one")
                if joined.actual_diff.isna().any():
                    raise AssertionError("Actual join is not 1:1")
                results.extend(_metric_rows(joined, {**context, "category": category, "policy": policy}))
            replacements.extend(_replacement_rows(ranks, actual, {**context, "category": category}))
            ripple_rows.extend(_ripple_rows(stages, category, context, active_diff.machine_numbers("renamed")))
        normalization_rows.extend(_normalization_rows(stages, context))
        epochs = shadow.build_continuous_epochs(history, plan.as_of_date)
        current = epochs.current.set_index("machine_no")
        active_rows = active_diff.rows[active_diff.rows.classification == "renamed"].set_index("machine_no")
        for no in sorted(active_rows.index):
            summary = current.loc[no]
            event = _event_for_changed_machine(
                events, no, summary.epoch_start_date, plan.event_id
            )
            if event is None:
                continue
            actual_row = actual[actual.machine_no == no]
            if actual_row.empty:
                continue
            lineage = stages[POLICY_BASELINE][0].normal.raw.lineage
            observed = lineage[(lineage.machine_no == no) & (lineage.policy == shadow.OBSERVED_G_GT_0)].iloc[0]
            for policy, (stage, source) in stages.items():
                scored = stage.normal.policies[source]["scored"]
                row = scored[scored.machine_no == no]
                eligible = not row.empty and bool(row.iloc[0].final_candidate)
                final = stage.normal.policies[source]["final"]
                ranked = final[final.machine_no == no]
                rank = int(ranked.iloc[0].final_candidate_rank) if len(ranked) else 0
                hypothetical = int(row.iloc[0].full_population_score_order) if len(row) else 0
                changed_rows.append({"event_id": event.event_id, "target_date": plan.target_date,
                    "machine_no": no, "old_machine_name": active_rows.loc[no].old_machine_name,
                    "new_machine_name": active_rows.loc[no].new_machine_name, "epoch_id": summary.epoch_id,
                    "event_calendar_day": (plan.target_date - event.change_date).days,
                    "day_bucket": day_bucket((plan.target_date - event.change_date).days),
                    "epoch_calendar_history_n": int(summary.epoch_calendar_days),
                    "epoch_observed_history_n": int(summary.epoch_observed_days), "policy": policy,
                    "eligible": eligible, "excluded_reason": "" if eligible else "INSUFFICIENT_OBSERVED_HISTORY",
                    "score_calculated": not row.empty, "score": float(row.iloc[0].score) if len(row) else None,
                    "rank": rank, "hypothetical_rank": hypothetical,
                    "diagnostic_only": not eligible, "top1": 0 < rank <= 1, "top3": 0 < rank <= 3,
                    "top5": 0 < rank <= 5, "top10": 0 < rank <= 10,
                    "actual_diff": float(actual_row.iloc[0].actual_diff),
                    "win": float(actual_row.iloc[0].actual_diff) > 0,
                    "plus1000": float(actual_row.iloc[0].actual_diff) >= 1000,
                    "plus2000": float(actual_row.iloc[0].actual_diff) >= 2000,
                    "fallback": observed.fallback_kind, "history_min_date": observed.history_min_date,
                    "history_max_date": observed.history_max_date})
        leakage_rows.append({**context, "target_actual_loaded_before_ranking": False,
            "feature_max_date": plan.as_of_date, "epoch_max_history_date": plan.as_of_date,
            "type_prior_max_date": plan.as_of_date, "store_prior_max_date": plan.as_of_date,
            "neighbor_source_date": plan.as_of_date, "inventory_snapshot_date": plan.as_of_date,
            "inventory_transition_max_date": plan.as_of_date, "category_membership_source_date": plan.as_of_date,
            "actual_date": plan.target_date, "ranking_frozen_before_actual_join": True,
            "actual_join_one_to_one": True, "pre_actual_ranking_digest": pre_digest,
            "post_actual_ranking_digest": ranking_digest(stages), "leakage_violation": False})

    result_frame = pd.DataFrame(results)
    primary_results = result_frame[result_frame.independent_primary].copy()
    topn = primary_results.groupby(["event_id", "category", "policy", "top_n"], as_index=False).agg(
        evaluation_days=("target_date", "nunique"), selected_count=("selected_count", "sum"),
        actual_total_diff=("actual_total_diff", "sum"), avg_diff_per_selected_machine=("avg_diff_per_selected_machine", "mean"),
        median_diff=("median_diff", "median"), win_rate=("win_rate", "mean"),
        plus1000_rate=("plus1000_rate", "mean"), plus2000_rate=("plus2000_rate", "mean"),
        positive_day_rate=("positive_day", "mean"), worst_day=("daily_total_diff", "min"),
        best_day=("daily_total_diff", "max"))
    replacement_frame = pd.DataFrame(replacements)
    comparison = primary_results.merge(
        primary_results[primary_results.policy == POLICY_BASELINE][["event_id", "target_date", "category", "top_n", "actual_total_diff"]]
        .rename(columns={"actual_total_diff": "baseline_actual_total_diff"}),
        on=["event_id", "target_date", "category", "top_n"], how="left")
    comparison["paired_delta_vs_baseline"] = comparison.actual_total_diff - comparison.baseline_actual_total_diff
    comparison = comparison.merge(replacement_frame, on=["event_id", "target_date", "event_calendar_day", "day_bucket", "independent_primary", "category", "policy", "top_n"], how="left")

    event_rows = []
    classify_a, is_juggler = shadow._load_classifiers()
    for index, event in enumerate(events):
        current = snapshots[event.change_date].set_index("machine_no")
        names = [str(current.loc[no].machine_name) for no in event.renamed]
        next_date = events[index + 1].change_date if index + 1 < len(events) else None
        event_rows.append({**asdict(event), "added": "|".join(map(str, event.added)),
            "removed": "|".join(map(str, event.removed)), "renamed": "|".join(map(str, event.renamed)),
            "added_count": len(event.added), "removed_count": len(event.removed), "renamed_count": len(event.renamed),
            "affected_count": len(event.added) + len(event.removed) + len(event.renamed),
            "a_type_changed_count": sum(bool(classify_a(name)["is_a_type"]) for name in names),
            "juggler_changed_count": sum(bool(is_juggler(name)) for name in names),
            "next_store_event_date": next_date,
            "primary_window_end": min(event.change_date + timedelta(days=14), next_date - timedelta(days=1)) if next_date else event.change_date + timedelta(days=14)})

    changed = pd.DataFrame(changed_rows)
    reentry_rows = []
    if not changed.empty:
        for (event_id, no, policy), group in changed.sort_values("target_date").groupby(["event_id", "machine_no", "policy"]):
            if not policy.startswith("P2_T"):
                continue
            threshold = int(policy.removeprefix("P2_T"))
            rows = group.reset_index(drop=True)
            for index in range(1, len(rows)):
                before, after = rows.iloc[index - 1], rows.iloc[index]
                if before.epoch_observed_history_n < threshold <= after.epoch_observed_history_n and before.epoch_id == after.epoch_id:
                    reentry_rows.append({"event_id": event_id, "machine_no": no, "epoch_id": after.epoch_id,
                        "threshold": threshold, "previous_target": before.target_date, "reentry_target": after.target_date,
                        "history_n_before": before.epoch_observed_history_n, "history_n_after": after.epoch_observed_history_n,
                        "reentry_score": after.score, "reentry_rank": after["rank"], "zpop_size_before": None,
                        "zpop_size_after": None, "feature_mean_std_shift": "SEE_NORMALIZATION_DIAGNOSTICS",
                        "unchanged_score_ripple": "SEE_UNCHANGED_RIPPLE", "unchanged_rank_ripple": "SEE_UNCHANGED_RIPPLE",
                        "topn_replacement": "SEE_UNCHANGED_RIPPLE", "category_ripple": "SEE_UNCHANGED_RIPPLE",
                        "replacement_actual_gain_loss": "SEE_THRESHOLD_COMPARISON"})

    missing = []
    for day in pd.date_range(min(paths), max(paths), freq="D"):
        if day.date() not in paths:
            missing.append(day.date().isoformat())
    frames = {
        "event_summary.csv": pd.DataFrame(event_rows), "target_day_results.csv": result_frame,
        "topn_outcomes.csv": topn, "changed_machine_outcomes.csv": changed,
        "unchanged_ripple.csv": pd.DataFrame(ripple_rows),
        "normalization_diagnostics.csv": pd.DataFrame(normalization_rows),
        "reentry_diagnostics.csv": pd.DataFrame(reentry_rows), "threshold_comparison.csv": comparison,
    }
    output_digests = {}
    if write_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, frame in frames.items():
            frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
            output_digests[filename] = sha256_path(output_dir / filename)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    lineage = {"research_only": True, "formal": False, "forward_valid": False,
        "production_outputs_written": False, "model": "CHAMPION_V4.2_C",
        "fingerprint": shadow.V42_C_FINGERPRINT, "weights": shadow.V42_C_WEIGHTS,
        "git_head": head, "stage_a_d_module_sha256": sha256_path(SHADOW_PATH),
        "stage_e_script_sha256": sha256_path(Path(__file__)), "daily_count": len(paths),
        "daily_start": min(paths).isoformat(), "daily_end": max(paths).isoformat(), "missing_dates": missing,
        "event_ids": [event.event_id for event in events], "thresholds": list(THRESHOLDS),
        "policy_definitions": {"BASELINE": "production raw/scoring reproduction",
            "P1": "OBSERVED_G_GT_0; full z-pop; LT14 rank exclusion",
            "P2": "OBSERVED_G_GT_0; threshold eligible-only z-pop and ranking"},
        "input_sha256": {day.isoformat(): sha256_path(path) for day, path in paths.items()},
        "actual_sha256": {row.target_date.isoformat(): sha256_path(paths[row.target_date])
            for row in target_plan[target_plan.evaluation_available].itertuples()},
        "leakage_assertions": leakage_rows, "output_digest": output_digests,
        "formal_quarantined_supported": True, "formal_scorecard_included": False}
    if write_outputs:
        lineage_path = Path(output_dir) / "lineage.json"
        lineage_path.write_text(json.dumps(lineage, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        output_digests["lineage.json"] = sha256_path(lineage_path)
    return {"events": events, "target_plan": target_plan, "frames": frames, "lineage": lineage,
        "runtime_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description="Research-only Inventory Epoch Stage E walk-forward")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir, True)
    print(json.dumps({"events": len(result["events"]),
        "renamed": sum(len(event.renamed) for event in result["events"]),
        "primary_targets": int((result["target_plan"].evaluation_available & result["target_plan"].independent_primary).sum()),
        "runtime_seconds": result["runtime_seconds"], "production_outputs_written": False}, indent=2))


if __name__ == "__main__":
    main()
