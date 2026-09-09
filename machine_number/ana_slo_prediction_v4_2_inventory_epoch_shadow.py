from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd


ALIASES = {
    "date": ("date", "日付"),
    "machine_no": ("machine_no", "台番号"),
    "machine_name": ("machine_name", "機種名"),
    "games": ("games", "G数"),
    "diff": ("diff", "差枚"),
}


@dataclass(frozen=True)
class InventoryDiff:
    rows: pd.DataFrame
    previous_machine_count: int
    current_machine_count: int
    unchanged_count: int
    renamed_count: int
    added_count: int
    removed_count: int

    def machine_numbers(self, classification: str) -> frozenset[int]:
        rows = self.rows[self.rows["classification"] == classification]
        return frozenset(int(value) for value in rows["machine_no"])


@dataclass(frozen=True)
class EpochBuildResult:
    rows: pd.DataFrame
    current: pd.DataFrame
    as_of_date: date
    history_n_definition: str = "G_GT_0"


RAW_FEATURES = ("avg31", "recent7_avg", "last_diff", "prev_change", "weekday_avg", "type_avg", "plus1000_rate", "plus2000_rate", "neighbor_avg")
EPOCH_FEATURES = ("avg31", "recent7_avg", "last_diff", "prev_change", "weekday_avg", "plus1000_rate", "plus2000_rate")
ROW_BASED = "ROW_BASED"
OBSERVED_G_GT_0 = "OBSERVED_G_GT_0"


@dataclass(frozen=True)
class FeatureShadowResult:
    baseline: pd.DataFrame
    row_based: pd.DataFrame
    observed: pd.DataFrame
    lineage: pd.DataFrame
    type_diagnostics: pd.DataFrame
    neighbor_diagnostics: pd.DataFrame
    target_date: date
    as_of_date: date
    target_actual_loaded: bool = False


V42_C_WEIGHTS = {
    "avg31": 0.0670952025611345, "recent7_avg": 0.05164896703284082,
    "last_diff": 0.12382294629381808, "prev_change": 0.10484738021281044,
    "weekday_avg": 0.05672674990073483, "type_avg": 0.05843723530102936,
    "plus1000_rate": 0.17725354845070532, "plus2000_rate": 0.13298938481323394,
    "neighbor_avg": 0.06161296683628432,
}
_weight_total = sum(V42_C_WEIGHTS.values())
V42_C_WEIGHTS = {key: value / _weight_total for key, value in V42_C_WEIGHTS.items()}
V42_C_FINGERPRINT = "a1eaf45d71ded209"
POLICY_BASELINE = "BASELINE"
POLICY1 = "EPOCH_LT14_RANK_EXCLUDE_FULL_ZPOP"
POLICY2 = "EPOCH_LT14_ZPOP_AND_RANK_EXCLUDE"
POLICY3 = "BASELINE_ZPOP_REFERENCE_RANK_EXCLUDE"
ROW_POLICY1 = "ROW_POLICY1_FULL_ZPOP"


@dataclass(frozen=True)
class StageCResult:
    raw: FeatureShadowResult
    policies: dict[str, dict[str, pd.DataFrame]]
    z_audit: pd.DataFrame
    component_audit: pd.DataFrame
    eligibility: pd.DataFrame
    score_ripple: pd.DataFrame
    top10_comparisons: pd.DataFrame
    metadata: dict


@dataclass(frozen=True)
class StageDResult:
    normal: StageCResult
    categories: dict[str, dict[str, dict[str, pd.DataFrame]]]
    membership: pd.DataFrame
    lineage: pd.DataFrame
    summaries: pd.DataFrame
    comparisons: pd.DataFrame
    ripple: pd.DataFrame
    metadata: dict


def normalize_day(value) -> date:
    parsed = pd.to_datetime(value, errors="raise")
    if isinstance(parsed, pd.DatetimeIndex):
        raise ValueError("A single date is required.")
    return pd.Timestamp(parsed).date()


def read_daily_csv(path: Path, file_date, max_allowed_date) -> pd.DataFrame:
    """Read one daily while refusing to open a target/future dated file."""
    path = Path(path)
    expected = normalize_day(file_date)
    maximum = normalize_day(max_allowed_date)
    if expected > maximum:
        raise ValueError(
            f"Future daily is forbidden: file_date={expected}, max_allowed_date={maximum}"
        )
    raw = None
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            raw = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if raw is None:
        raise RuntimeError(f"Could not decode daily CSV: {path}") from last_error
    columns = {
        key: next((name for name in names if name in raw.columns), "")
        for key, names in ALIASES.items()
    }
    missing = [key for key, name in columns.items() if not name]
    if missing:
        raise ValueError(f"Required daily columns are missing: {missing}")
    frame = raw[[columns[key] for key in ALIASES]].rename(
        columns={source: canonical for canonical, source in columns.items()}
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    internal_dates = {value.date() for value in frame["date"]}
    if internal_dates != {expected}:
        raise ValueError(
            f"Daily filename/internal date mismatch: expected={expected}, actual={sorted(internal_dates)}"
        )
    frame["machine_no"] = pd.to_numeric(frame["machine_no"], errors="raise")
    if (frame["machine_no"] % 1 != 0).any():
        raise ValueError("machine_no must be an integer.")
    frame["machine_no"] = frame["machine_no"].astype(int)
    if frame["machine_no"].duplicated().any():
        raise ValueError("Duplicate machine_no in daily CSV.")
    frame["machine_name"] = frame["machine_name"].astype("string").str.strip()
    if frame["machine_name"].isna().any() or frame["machine_name"].eq("").any():
        raise ValueError("machine_name must not be empty.")
    frame["games"] = pd.to_numeric(
        frame["games"].astype(str).str.replace(",", "", regex=False), errors="raise"
    )
    frame["diff"] = pd.to_numeric(
        frame["diff"].astype(str).str.replace(",", "", regex=False), errors="raise"
    )
    if frame[["games", "diff"]].isna().any().any():
        raise ValueError("games and diff must be numeric and non-null.")
    if (frame["games"] < 0).any():
        raise ValueError("games must not be negative.")
    frame["observed"] = frame["games"] > 0
    return frame.sort_values("machine_no").reset_index(drop=True)


def compare_inventories(previous: pd.DataFrame, current: pd.DataFrame) -> InventoryDiff:
    """Classify the union of two snapshots by machine_no."""
    for label, frame in (("previous", previous), ("current", current)):
        if not {"machine_no", "machine_name"}.issubset(frame.columns):
            raise ValueError(f"{label} inventory columns are missing.")
        if frame["machine_no"].duplicated().any():
            raise ValueError(f"{label} inventory contains duplicate machine_no.")
    old = previous.set_index("machine_no")["machine_name"].astype(str).to_dict()
    new = current.set_index("machine_no")["machine_name"].astype(str).to_dict()
    rows = []
    for machine_no in sorted(set(old) | set(new)):
        if machine_no not in old:
            classification = "added"
        elif machine_no not in new:
            classification = "removed"
        elif old[machine_no] != new[machine_no]:
            classification = "renamed"
        else:
            classification = "unchanged"
        rows.append({
            "machine_no": int(machine_no),
            "old_machine_name": old.get(machine_no, ""),
            "new_machine_name": new.get(machine_no, ""),
            "classification": classification,
        })
    detail = pd.DataFrame(rows, columns=(
        "machine_no", "old_machine_name", "new_machine_name", "classification"
    ))
    counts = detail["classification"].value_counts()
    return InventoryDiff(
        detail,
        int(previous["machine_no"].nunique()),
        int(current["machine_no"].nunique()),
        int(counts.get("unchanged", 0)),
        int(counts.get("renamed", 0)),
        int(counts.get("added", 0)),
        int(counts.get("removed", 0)),
    )


def build_persistent_inventory_diff(observations: pd.DataFrame, as_of_date) -> InventoryDiff:
    """Return the active rename epoch at *as_of_date* as an InventoryDiff.

    Unlike a one-day inventory comparison, a renamed machine remains classified
    as ``renamed`` until its current continuous epoch ends.  First appearances
    (added machines) are deliberately not promoted to renamed machines; Stage E
    records those as event evidence but does not apply the rename policy to them.
    """
    as_of = normalize_day(as_of_date)
    epochs = build_continuous_epochs(observations, as_of)
    if epochs.rows.empty:
        return InventoryDiff(
            pd.DataFrame(columns=(
                "machine_no", "old_machine_name", "new_machine_name", "classification"
            )), 0, 0, 0, 0, 0, 0,
        )
    current = epochs.current.set_index("machine_no")
    rows = []
    for machine_no in sorted(current.index):
        summary = current.loc[machine_no]
        sequence = int(summary.epoch_sequence)
        old_name = ""
        classification = "unchanged"
        if sequence > 1:
            prior = epochs.rows[
                (epochs.rows.machine_no == int(machine_no))
                & (epochs.rows.epoch_sequence == sequence - 1)
            ].sort_values("date")
            if prior.empty:
                raise AssertionError(f"Missing predecessor epoch for machine_no={machine_no}")
            old_name = str(prior.iloc[-1].machine_name)
            classification = "renamed"
        rows.append({
            "machine_no": int(machine_no),
            "old_machine_name": old_name,
            "new_machine_name": str(summary.machine_name),
            "classification": classification,
        })
    detail = pd.DataFrame(rows)
    renamed = int(detail.classification.eq("renamed").sum())
    unchanged = int(detail.classification.eq("unchanged").sum())
    count = int(len(detail))
    return InventoryDiff(detail, count, count, unchanged, renamed, 0, 0)


def build_continuous_epochs(observations: pd.DataFrame, as_of_date) -> EpochBuildResult:
    """Build machine-level epochs from observations no later than as-of."""
    required = {"date", "machine_no", "machine_name", "games", "diff"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Epoch input columns are missing: {sorted(missing)}")
    as_of = normalize_day(as_of_date)
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame[frame["date"].dt.date <= as_of].copy()
    if frame.empty:
        return EpochBuildResult(pd.DataFrame(), pd.DataFrame(), as_of)
    if frame.duplicated(["date", "machine_no"]).any():
        raise ValueError("Duplicate date/machine_no observation.")
    frame["machine_no"] = pd.to_numeric(frame["machine_no"], errors="raise").astype(int)
    frame["machine_name"] = frame["machine_name"].astype("string").str.strip()
    if frame["machine_name"].isna().any() or frame["machine_name"].eq("").any():
        raise ValueError("machine_name must not be empty.")
    frame["games"] = pd.to_numeric(frame["games"], errors="raise")
    if (frame["games"] < 0).any():
        raise ValueError("games must not be negative.")
    frame["observed"] = frame["games"] > 0
    frame = frame.sort_values(["machine_no", "date"]).reset_index(drop=True)
    prior_name = frame.groupby("machine_no")["machine_name"].shift()
    first = prior_name.isna()
    changed = (~first) & frame["machine_name"].ne(prior_name)
    frame["epoch_sequence"] = (first | changed).groupby(frame["machine_no"]).cumsum().astype(int)
    starts = frame.groupby(["machine_no", "epoch_sequence"])["date"].transform("min")
    frame["epoch_start_date"] = starts.dt.date
    frame["epoch_id"] = frame.apply(
        lambda row: f"{int(row.machine_no)}:{int(row.epoch_sequence):04d}:{row.epoch_start_date:%Y%m%d}",
        axis=1,
    )
    previous_date = frame.groupby("machine_no")["date"].shift()
    frame["days_since_previous_observation"] = (frame["date"] - previous_date).dt.days
    frame["observation_continues_name"] = (~first) & ~changed
    frame["calendar_contiguous_from_previous"] = (
        frame["observation_continues_name"] & frame["days_since_previous_observation"].eq(1)
    )
    latest = frame.groupby("machine_no")["epoch_sequence"].transform("max")
    current_rows = frame[frame["epoch_sequence"] == latest]
    summaries = []
    for (machine_no, sequence), group in current_rows.groupby(
        ["machine_no", "epoch_sequence"], sort=True
    ):
        group = group.sort_values("date")
        observed = group[group["games"] > 0]
        start, end = group["date"].min().date(), group["date"].max().date()
        calendar_days = int(group["date"].nunique())
        span_days = (end - start).days + 1
        summaries.append({
            "machine_no": int(machine_no),
            "machine_name": str(group.iloc[-1]["machine_name"]),
            "epoch_sequence": int(sequence),
            "epoch_id": str(group.iloc[-1]["epoch_id"]),
            "epoch_start_date": start,
            "epoch_last_date": end,
            "epoch_calendar_days": calendar_days,
            "epoch_observed_days": int(observed["date"].nunique()),
            "epoch_zero_g_days": int(group.loc[group["games"] == 0, "date"].nunique()),
            "epoch_total_games": float(group["games"].sum()),
            "last_observed_date": observed["date"].max().date() if not observed.empty else None,
            "epoch_calendar_span_days": span_days,
            "epoch_missing_calendar_days": span_days - calendar_days,
            "calendar_continuous": span_days == calendar_days,
            "eligibility_history_n": int(observed["date"].nunique()),
            "history_n_definition": "G_GT_0",
        })
    current = pd.DataFrame(summaries).sort_values("machine_no").reset_index(drop=True)
    return EpochBuildResult(frame, current, as_of)


def _prepare_history(observations, target_date, as_of_date):
    target, as_of = normalize_day(target_date), normalize_day(as_of_date)
    if as_of >= target:
        raise ValueError("as_of_date must be before target_date.")
    required = {"date", "machine_no", "machine_name", "games", "diff"}
    if missing := required - set(observations.columns):
        raise ValueError(f"Feature input columns are missing: {sorted(missing)}")
    hist = observations.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="raise").dt.normalize()
    hist = hist[hist["date"].dt.date <= as_of].copy()
    if hist.empty or not (hist["date"].dt.date == as_of).any():
        raise ValueError("No inventory snapshot exists on as_of_date.")
    hist["machine_no"] = pd.to_numeric(hist["machine_no"], errors="raise").astype(int)
    hist["machine_name"] = hist["machine_name"].astype("string").str.strip()
    hist["games"] = pd.to_numeric(hist["games"], errors="raise")
    hist["diff"] = pd.to_numeric(hist["diff"], errors="raise").astype(float)
    if hist.duplicated(["date", "machine_no"]).any():
        raise ValueError("Duplicate date/machine_no observation.")
    if hist[["games", "diff"]].isna().any().any() or (hist["games"] < 0).any():
        raise ValueError("Invalid games/diff values.")
    hist["plus1000"] = (hist["diff"] >= 1000).astype(float)
    hist["plus2000"] = (hist["diff"] >= 2000).astype(float)
    return hist.sort_values(["machine_no", "date"]).reset_index(drop=True), target, as_of


def _machine_values(rows, target):
    rows = rows.sort_values("date")
    avg, recent, last = float(rows["diff"].mean()), rows.tail(7), float(rows.iloc[-1]["diff"])
    wd = rows[rows["date"].dt.dayofweek == pd.Timestamp(target).dayofweek]
    raw = float(wd["diff"].mean()) if len(wd) else avg
    weight = len(wd) / (len(wd) + 15.0)
    return {"avg31": avg, "recent7_avg": float(recent["diff"].mean()), "last_diff": last,
            "prev_change": last - float(rows.iloc[-2]["diff"]) if len(rows) >= 2 else 0.0,
            "weekday_avg": raw * weight + avg * (1.0 - weight),
            "plus1000_rate": float(rows["plus1000"].mean()), "plus2000_rate": float(rows["plus2000"].mean())}


def _prior_values(hist, target, name):
    typed = hist[hist["machine_name"] == name].sort_values("date")
    source, kind = (typed, "TYPE_PRIOR") if len(typed) else (hist, "STORE_PRIOR")
    latest = source["date"].max()
    recent = source[source["date"] >= latest - pd.Timedelta(days=6)]
    wd = source[source["date"].dt.dayofweek == pd.Timestamp(target).dayofweek]
    avg, tail = float(source["diff"].mean()), typed.tail(7) if kind == "TYPE_PRIOR" else recent
    return {"avg31": avg, "recent7_avg": float(tail["diff"].mean()), "last_diff": float(tail["diff"].mean()),
            "prev_change": 0.0, "weekday_avg": float(wd["diff"].mean()) if len(wd) else avg,
            "plus1000_rate": float(source["plus1000"].mean()), "plus2000_rate": float(source["plus2000"].mean())}, kind, len(source)


def _baseline_features(hist, target, as_of):
    current = hist[hist["date"].dt.date == as_of].set_index("machine_no")
    type_means = hist.groupby("machine_name")["diff"].mean().to_dict()
    rows = []
    for no in sorted(current.index):
        machine = hist[hist["machine_no"] == no]
        values, name = _machine_values(machine, target), str(machine.iloc[-1]["machine_name"])
        neighbors = [float(current.loc[n, "diff"]) for n in (int(no) - 1, int(no) + 1) if n in current.index]
        values.update(machine_no=int(no), machine_name=name, type_avg=float(type_means.get(name, 0.0)),
                      neighbor_avg=float(sum(neighbors) / len(neighbors)) if neighbors else 0.0)
        rows.append(values)
    return pd.DataFrame(rows).sort_values("machine_no").reset_index(drop=True)


def build_raw_feature_shadow(observations, target_date, as_of_date, inventory_diff):
    """Build pure raw-feature comparisons; never scores, ranks, or writes files."""
    hist, target, as_of = _prepare_history(observations, target_date, as_of_date)
    baseline = _baseline_features(hist, target, as_of)
    row_based, observed = baseline.copy(), baseline.copy()
    epochs = build_continuous_epochs(hist, as_of)
    current = epochs.current.set_index("machine_no")
    renamed = inventory_diff.machine_numbers("renamed")
    changes = inventory_diff.rows.set_index("machine_no")
    latest = hist[hist["date"].dt.date == as_of].set_index("machine_no")
    lineage, type_diag, neighbor_diag = [], [], []
    for no in sorted(renamed):
        if no not in current.index or no not in set(baseline.machine_no):
            continue
        summary = current.loc[no]
        epoch = epochs.rows[epochs.rows.epoch_id == summary.epoch_id].copy()
        name, old_name = str(summary.machine_name), str(changes.loc[no, "old_machine_name"])
        for policy, selected, output in ((ROW_BASED, epoch, row_based), (OBSERVED_G_GT_0, epoch[epoch.games > 0], observed)):
            values, fallback, source_n = (_machine_values(selected, target), "NONE", 0) if len(selected) else _prior_values(hist, target, name)
            index = output.index[output.machine_no == no][0]
            for feature in EPOCH_FEATURES:
                output.at[index, feature] = values[feature]
            lineage.append({"machine_no": no, "current_machine_name": name, "epoch_start_date": summary.epoch_start_date,
                "policy": policy, "history_n": len(selected),
                "history_min_date": selected.date.min().date() if len(selected) else None,
                "history_max_date": selected.date.max().date() if len(selected) else None,
                "history_machine_names": tuple(sorted(set(selected.machine_name.astype(str)))),
                "pre_epoch_row_count": int((selected.date.dt.date < summary.epoch_start_date).sum()),
                "old_machine_name_row_count": int((selected.machine_name.astype(str) == old_name).sum()),
                "epoch_observed_history_n": int((epoch.games > 0).sum()),
                "fallback_kind": fallback, "fallback_source_row_count": int(source_n)})
        typed = hist[hist.machine_name == name]
        type_diag.append({"machine_no": no, "current_machine_name": name,
            "baseline_type_avg": float(baseline.loc[baseline.machine_no == no, "type_avg"].iloc[0]),
            "current_name_row_count": len(typed), "current_name_observed_row_count": int((typed.games > 0).sum()),
            "current_name_machine_no_count": int(typed.machine_no.nunique()), "current_name_history_none": not len(typed),
            "current_name_zero_g_ratio": float((typed.games == 0).mean()) if len(typed) else 0.0,
            "type_avg_observed_g_gt_0": float(typed.loc[typed.games > 0, "diff"].mean()) if (typed.games > 0).any() else None})
        diagnostic, neighbors = {}, []
        for side, neighbor_no in (("left", no - 1), ("right", no + 1)):
            diagnostic.update({f"{side}_machine_no": neighbor_no, f"{side}_machine_name": None,
                               f"{side}_games": None, f"{side}_diff": None})
            if neighbor_no in latest.index:
                neighbor = latest.loc[neighbor_no]
                diagnostic.update({f"{side}_machine_name": str(neighbor.machine_name), f"{side}_games": float(neighbor.games),
                                   f"{side}_diff": float(neighbor["diff"])})
                neighbors.append(neighbor)
        observed_neighbors = [float(x["diff"]) for x in neighbors if float(x.games) > 0]
        diagnostic.update({"machine_no": no, "neighbor_count": len(neighbors),
            "zero_g_neighbor_count": sum(float(x.games) == 0 for x in neighbors),
            "both_neighbors_zero_g": len(neighbors) == 2 and all(float(x.games) == 0 for x in neighbors),
            "neighbor_inventory_changed": any(int(x.name) in renamed for x in neighbors),
            "production_neighbor_avg": float(baseline.loc[baseline.machine_no == no, "neighbor_avg"].iloc[0]),
            "observed_neighbor_reference_avg": float(sum(observed_neighbors) / len(observed_neighbors)) if observed_neighbors else None})
        neighbor_diag.append(diagnostic)
    for output in (baseline, row_based, observed):
        numeric = output[list(RAW_FEATURES)].astype(float)
        if numeric.isna().any().any() or (numeric.abs() == float("inf")).any().any():
            raise ValueError("Raw features must be finite.")
    return FeatureShadowResult(baseline, row_based, observed, pd.DataFrame(lineage), pd.DataFrame(type_diag),
                               pd.DataFrame(neighbor_diag), target, as_of, False)


def weight_fingerprint(weights=V42_C_WEIGHTS):
    text = "|".join(f"{key}:{weights[key]:.15f}" for key in sorted(weights))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _population_stats(panel, population, policy):
    selected = panel[panel.machine_no.isin(population)]
    stats, rows = {}, []
    for feature, weight in V42_C_WEIGHTS.items():
        values = selected[feature].astype(float)
        mean, std = float(values.mean()), float(values.std(ddof=0))
        stats[feature] = (mean, std)
        rows.append({"policy": policy, "feature": feature, "population_n": len(values),
                     "mean": mean, "std": std, "ddof": 0, "weight": weight})
    return stats, rows


def _score_panel(panel, stats, policy, z_population, candidates):
    scored, audit = panel.copy(), []
    score = pd.Series(0.0, index=scored.index)
    for feature, weight in V42_C_WEIGHTS.items():
        raw = scored[feature].astype(float)
        mean, std = stats[feature]
        z = pd.Series(0.0, index=scored.index) if std == 0 or pd.isna(std) else (raw - mean) / std
        component = (50.0 + z * 12.5).clip(0, 100)
        weighted = component * weight
        score += weighted
        for index in scored.index:
            audit.append({"policy": policy, "machine_no": int(scored.at[index, "machine_no"]), "feature": feature,
                          "raw": float(raw.at[index]), "z": float(z.at[index]), "component": float(component.at[index]),
                          "weighted_component": float(weighted.at[index])})
    scored["score"] = score
    scored["z_population"] = scored.machine_no.isin(z_population)
    scored["final_candidate"] = scored.machine_no.isin(candidates)
    full = scored.sort_values("score", ascending=False).copy()
    full["full_population_score_order"] = range(1, len(full) + 1)
    final = scored[scored.final_candidate].sort_values("score", ascending=False).copy()
    final["final_candidate_rank"] = range(1, len(final) + 1)
    merged = full.merge(final[["machine_no", "final_candidate_rank"]], on="machine_no", how="left")
    return merged, final, pd.DataFrame(audit)


def _top10_comparison(left_name, left, right_name, right, changed, excluded):
    a, b = left.head(10).machine_no.astype(int).tolist(), right.head(10).machine_no.astype(int).tolist()
    sa, sb = set(a), set(b)
    common = sa & sb
    return {"left_policy": left_name, "right_policy": right_name, "common_count": len(common),
            "left_only": tuple(sorted(sa - sb)), "right_only": tuple(sorted(sb - sa)),
            "replacement_count": len(sa - sb),
            "order_changes": sum(a.index(no) != b.index(no) for no in common),
            "changed_in_left_top10": len(sa & changed), "changed_in_right_top10": len(sb & changed),
            "excluded_changed_count": len(excluded), "candidate_count": len(right)}


def build_stage_c_shadow(observations, target_date, as_of_date, inventory_diff, insufficient_history_threshold=14):
    """Score and rank shadow panels in memory only; this is not a formal prediction."""
    if weight_fingerprint() != V42_C_FINGERPRINT:
        raise AssertionError("V4.2_C weight fingerprint mismatch.")
    raw = build_raw_feature_shadow(observations, target_date, as_of_date, inventory_diff)
    all_machines = frozenset(int(x) for x in raw.baseline.machine_no)
    changed = inventory_diff.machine_numbers("renamed") & all_machines
    observed_lineage = raw.lineage[raw.lineage.policy == OBSERVED_G_GT_0].set_index("machine_no")
    history_n = {no: int(observed_lineage.at[no, "history_n"]) for no in changed}
    excluded = frozenset(no for no in changed if history_n[no] < insufficient_history_threshold)
    candidates = all_machines - excluded
    specifications = (
        (POLICY_BASELINE, raw.baseline, all_machines, all_machines, None, "BASELINE"),
        (POLICY1, raw.observed, all_machines, candidates, None, OBSERVED_G_GT_0),
        (POLICY2, raw.observed, candidates, candidates, None, OBSERVED_G_GT_0),
        (POLICY3, raw.baseline, all_machines, candidates, POLICY_BASELINE, "BASELINE_REFERENCE"),
        (ROW_POLICY1, raw.row_based, all_machines, candidates, None, ROW_BASED),
    )
    policies, z_rows, component_frames = {}, [], []
    reference_stats = None
    for name, panel, population, ranking, reference, raw_policy in specifications:
        if reference == POLICY_BASELINE:
            stats = reference_stats
            rows = [{"policy": name, "feature": feature, "population_n": len(all_machines), "mean": mean,
                     "std": std, "ddof": 0, "weight": V42_C_WEIGHTS[feature]}
                    for feature, (mean, std) in stats.items()]
        else:
            stats, rows = _population_stats(panel, population, name)
            if name == POLICY_BASELINE:
                reference_stats = stats
        scored, final, audit = _score_panel(panel, stats, name, population, ranking)
        policies[name] = {"scored": scored, "final": final, "top10": final.head(10).copy()}
        z_rows.extend(rows)
        component_frames.append(audit)
    eligibility_rows = []
    for name, _, population, ranking, _, raw_policy in specifications:
        for no in sorted(changed):
            eligible = True if name == POLICY_BASELINE else no not in excluded
            eligibility_rows.append({"policy": name, "machine_no": no, "history_n": history_n[no], "eligible": eligible,
                "excluded_reason": "INSUFFICIENT_OBSERVED_HISTORY_LT14" if no not in ranking else "",
                "raw_feature_policy": raw_policy, "score_calculated": no in population,
                "z_population": no in population, "final_ranking": no in ranking})
    base_scores = policies[POLICY_BASELINE]["scored"].set_index("machine_no")
    unchanged = sorted(all_machines - changed)
    ripple = []
    for name in (POLICY1, POLICY2, POLICY3, ROW_POLICY1):
        other = policies[name]["scored"].set_index("machine_no")
        delta = other.loc[unchanged, "score"] - base_scores.loc[unchanged, "score"]
        maximum = int(delta.abs().idxmax())
        ripple.append({"policy": name, "count": len(delta), "mean_delta": float(delta.mean()),
            "mean_abs_delta": float(delta.abs().mean()), "median_abs_delta": float(delta.abs().median()),
            "p95_abs_delta": float(delta.abs().quantile(.95)), "max_abs_delta": float(delta.abs().max()),
            "max_affected_machine_no": maximum})
    comparisons = []
    pairs = ((POLICY_BASELINE, POLICY1), (POLICY_BASELINE, POLICY2), (POLICY_BASELINE, POLICY3),
             (POLICY1, POLICY2), (POLICY1, ROW_POLICY1))
    for left, right in pairs:
        comparisons.append(_top10_comparison(left, policies[left]["final"], right, policies[right]["final"], changed, excluded))
    metadata = {"target_date": raw.target_date, "as_of_date": raw.as_of_date,
        "insufficient_history_threshold": insufficient_history_threshold, "history_n_definition": "G_GT_0",
        "weight_fingerprint": V42_C_FINGERPRINT, "formal": False, "forward_valid": False,
        "shadow_only": True, "target_actual_loaded": False, "changed_count": len(changed),
        "unchanged_count": len(unchanged), "excluded_changed_count": len(excluded)}
    return StageCResult(raw, policies, pd.DataFrame(z_rows), pd.concat(component_frames, ignore_index=True),
                        pd.DataFrame(eligibility_rows), pd.DataFrame(ripple), pd.DataFrame(comparisons), metadata)


def _load_classifiers():
    directory = Path(__file__).resolve().parent
    modules = []
    inserted = str(directory) not in sys.path
    if inserted:
        sys.path.insert(0, str(directory))
    try:
        for name, filename in (("shadow_a_type_classifier", "ana_slo_prediction_v4_2_A_type_separated.py"),
                               ("shadow_juggler_classifier", "ana_slo_prediction_v4_2_Juggler_separated.py")):
            spec = importlib.util.spec_from_file_location(name, directory / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
    finally:
        if inserted:
            sys.path.remove(str(directory))
    return modules[0].classify_a_type, modules[1].is_juggler


def _category_comparison(category, left_name, left, right_name, right):
    a, b = left.head(10).machine_no.astype(int).tolist(), right.head(10).machine_no.astype(int).tolist()
    sa, sb, common = set(a), set(b), set(a) & set(b)
    return {"category": category, "left_policy": left_name, "right_policy": right_name,
            "common_top10_count": len(common), "replacement_count": len(sa - sb),
            "left_only": tuple(sorted(sa - sb)), "right_only": tuple(sorted(sb - sa)),
            "order_changes": sum(a.index(no) != b.index(no) for no in common)}


def build_stage_d_shadow(observations, target_date, as_of_date, inventory_diff, insufficient_history_threshold=14):
    """Rerank NORMAL shadow scores with production category classifiers, in memory only."""
    normal = build_stage_c_shadow(observations, target_date, as_of_date, inventory_diff, insufficient_history_threshold)
    classify_a_type, is_juggler = _load_classifiers()
    classifiers = {"A-TYPE": lambda name: bool(classify_a_type(name)["is_a_type"]),
                   "JUGGLER": lambda name: bool(is_juggler(name))}
    policy_names = (POLICY_BASELINE, POLICY1, POLICY2)
    changed = inventory_diff.machine_numbers("renamed")
    eligibility = normal.eligibility.set_index(["policy", "machine_no"])
    categories, membership_rows, lineage_rows, summaries = {}, [], [], []
    for category, classifier in classifiers.items():
        categories[category] = {}
        for policy in policy_names:
            scored = normal.policies[policy]["scored"].copy()
            scored["classified_membership"] = scored.machine_name.map(classifier)
            scored["ranking_eligible"] = scored.final_candidate.astype(bool)
            scored["final_category_candidate"] = scored.classified_membership & scored.ranking_eligible
            candidates = scored[scored.final_category_candidate].sort_values("score", ascending=False).copy()
            candidates["category_rank"] = range(1, len(candidates) + 1)
            candidates["category"] = category
            candidates["policy"] = policy
            candidates["normal_score"] = candidates.score
            candidates["category_score"] = candidates.score
            categories[category][policy] = {"all": scored, "candidates": candidates, "top10": candidates.head(10).copy()}
            for _, row in scored.iterrows():
                no = int(row.machine_no)
                membership_rows.append({"machine_no": no, "machine_name": str(row.machine_name), "category": category,
                    "policy": policy, "classified_membership": bool(row.classified_membership),
                    "ranking_eligible": bool(row.ranking_eligible),
                    "final_category_candidate": bool(row.final_category_candidate), "changed": no in changed})
            for _, row in candidates.iterrows():
                no = int(row.machine_no)
                info = eligibility.loc[(policy, no)] if no in changed else None
                lineage_rows.append({"machine_no": no, "machine_name": str(row.machine_name), "category": category,
                    "policy": policy, "normal_score": float(row.score), "category_score": float(row.score),
                    "normal_full_rank": int(row.full_population_score_order),
                    "normal_final_candidate_rank": int(row.final_candidate_rank), "category_rank": int(row.category_rank),
                    "changed": no in changed, "history_n": int(info.history_n) if info is not None else None,
                    "eligible": True, "exclusion_reason": ""})
            classified = scored[scored.classified_membership]
            summaries.append({"category": category, "policy": policy, "classified_count": len(classified),
                "candidate_count": len(candidates), "changed_classified_count": int(classified.machine_no.isin(changed).sum()),
                "changed_excluded_count": int((classified.machine_no.isin(changed) & ~classified.final_candidate).sum()),
                "top10_changed_count": int(candidates.head(10).machine_no.isin(changed).sum())})
    comparisons = []
    for category in classifiers:
        for left, right in ((POLICY_BASELINE, POLICY1), (POLICY_BASELINE, POLICY2), (POLICY1, POLICY2)):
            comparisons.append(_category_comparison(category, left, categories[category][left]["candidates"],
                                                    right, categories[category][right]["candidates"]))
    ripple = []
    for category in classifiers:
        base = categories[category][POLICY_BASELINE]["candidates"].set_index("machine_no")
        for policy in (POLICY1, POLICY2):
            other = categories[category][policy]["candidates"].set_index("machine_no")
            common = sorted((set(base.index) & set(other.index)) - changed)
            score_delta = other.loc[common, "score"] - base.loc[common, "score"]
            rank_delta = other.loc[common, "category_rank"] - base.loc[common, "category_rank"]
            ripple.append({"category": category, "policy": policy, "count": len(common),
                "mean_abs_score_delta": float(score_delta.abs().mean()),
                "p95_abs_score_delta": float(score_delta.abs().quantile(.95)),
                "max_abs_score_delta": float(score_delta.abs().max()),
                "mean_abs_category_rank_delta": float(rank_delta.abs().mean()),
                "max_abs_category_rank_delta": int(rank_delta.abs().max())})
    metadata = {**normal.metadata, "stage": "D", "categories": tuple(classifiers),
                "classification_source": "production pure functions", "formal": False,
                "forward_valid": False, "shadow_only": True, "target_actual_loaded": False}
    return StageDResult(normal, categories, pd.DataFrame(membership_rows), pd.DataFrame(lineage_rows),
                        pd.DataFrame(summaries), pd.DataFrame(comparisons), pd.DataFrame(ripple), metadata)
