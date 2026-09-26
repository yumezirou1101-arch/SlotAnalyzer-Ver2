from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bic_tsubame_takasaki"
    / "machine_number"
)

OUTPUT_DIR = (
    INPUT_DIR
    / "analysis_external_v42c_inventory_reset_ab_v1"
)

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py"
)

MODEL_NAME = "CHAMPION_V4.2_C"
EXPECTED_FINGERPRINT = "a1eaf45d71ded209"
EXPECTED_MACHINES = 517

PRIMARY_BURNIN_DAYS = 14
SENSITIVITY_BURNIN_DAYS = 21

TOP_NS = (1, 3, 5, 10)

ACTIVE_FEATURES = (
    "avg31",
    "recent7_avg",
    "last_diff",
    "prev_change",
    "weekday_avg",
    "type_avg",
    "plus1000_rate",
    "plus2000_rate",
    "neighbor_avg",
)

BOOTSTRAP_SEED = 20260924
BOOTSTRAP_REPS = 20000


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v42c_source",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load source module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(weights: dict[str, float]) -> str:
    text = "|".join(
        f"{k}:{weights[k]:.15f}"
        for k in sorted(weights)
    )
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_daily_files():
    files = sorted(
        INPUT_DIR.glob(
            "ana_slo_bic_tsubame_takasaki_*.csv"
        )
    )

    if not files:
        raise RuntimeError(
            f"No input files found: {INPUT_DIR}"
        )

    all_frames = []
    qa_rows = []

    for path in files:
        raw = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        required = {
            "date",
            "machine_name",
            "machine_no",
            "G",
            "diff",
        }

        missing_columns = sorted(
            required - set(raw.columns)
        )

        if missing_columns:
            raise RuntimeError(
                f"{path.name}: missing columns "
                f"{missing_columns}"
            )

        df = raw.copy()

        original_name_missing = (
            df["machine_name"].isna()
            | df["machine_name"]
            .astype(str)
            .str.strip()
            .eq("")
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df["machine_no"] = pd.to_numeric(
            df["machine_no"],
            errors="coerce",
        )

        df["G"] = pd.to_numeric(
            df["G"],
            errors="coerce",
        )

        df["diff"] = pd.to_numeric(
            df["diff"],
            errors="coerce",
        )

        invalid_date = int(
            df["date"].isna().sum()
        )

        invalid_machine = int(
            df["machine_no"].isna().sum()
        )

        invalid_g = int(
            df["G"].isna().sum()
        )

        invalid_diff = int(
            df["diff"].isna().sum()
        )

        duplicate_machine = int(
            df["machine_no"].duplicated().sum()
        )

        negative_g = int(
            (df["G"] < 0).sum()
        )

        unique_machine = int(
            df["machine_no"].nunique(
                dropna=True
            )
        )

        file_date_text = (
            path.stem.rsplit("_", 1)[-1]
        )

        file_date = pd.to_datetime(
            file_date_text,
            format="%Y%m%d",
        )

        embedded_dates = (
            df["date"]
            .dropna()
            .dt.normalize()
            .unique()
        )

        embedded_date_ok = bool(
            len(embedded_dates) == 1
            and pd.Timestamp(
                embedded_dates[0]
            ).normalize()
            == file_date.normalize()
        )

        qa_rows.append(
            {
                "date":
                    file_date.date().isoformat(),
                "file":
                    path.name,
                "rows":
                    len(df),
                "unique_machine":
                    unique_machine,
                "duplicate_machine":
                    duplicate_machine,
                "missing_machine":
                    invalid_machine,
                "missing_name":
                    int(
                        original_name_missing.sum()
                    ),
                "invalid_date":
                    invalid_date,
                "invalid_G":
                    invalid_g,
                "invalid_diff":
                    invalid_diff,
                "negative_G":
                    negative_g,
                "embedded_date_ok":
                    embedded_date_ok,
            }
        )

        if (
            invalid_date
            or invalid_machine
            or invalid_g
            or invalid_diff
        ):
            continue

        df["machine_no"] = (
            df["machine_no"].astype(int)
        )

        df["machine_name"] = (
            df["machine_name"]
            .astype(str)
            .str.strip()
        )

        df["win"] = (
            df["diff"] > 0
        ).astype(int)

        df["plus1000"] = (
            df["diff"] >= 1000
        ).astype(int)

        df["plus2000"] = (
            df["diff"] >= 2000
        ).astype(int)

        all_frames.append(
            df[
                [
                    "date",
                    "machine_name",
                    "machine_no",
                    "G",
                    "diff",
                    "win",
                    "plus1000",
                    "plus2000",
                ]
            ].copy()
        )

    qa = pd.DataFrame(qa_rows)

    if not all_frames:
        raise RuntimeError(
            "No valid input frames."
        )

    data = pd.concat(
        all_frames,
        ignore_index=True,
    )

    data["date"] = pd.to_datetime(
        data["date"]
    ).dt.normalize()

    return data, qa, files


def validate_input_qa(
    data: pd.DataFrame,
    qa: pd.DataFrame,
):
    failures = []

    if not bool(
        (qa["rows"] == EXPECTED_MACHINES).all()
    ):
        failures.append(
            "517 rows/day failed"
        )

    if not bool(
        (
            qa["unique_machine"]
            == EXPECTED_MACHINES
        ).all()
    ):
        failures.append(
            "517 unique machine_no/day failed"
        )

    for col in (
        "duplicate_machine",
        "missing_machine",
        "missing_name",
        "invalid_date",
        "invalid_G",
        "invalid_diff",
        "negative_G",
    ):
        if int(qa[col].sum()) != 0:
            failures.append(
                f"{col} nonzero"
            )

    if not bool(
        qa["embedded_date_ok"].all()
    ):
        failures.append(
            "embedded date mismatch"
        )

    dates = sorted(
        data["date"].drop_duplicates()
    )

    expected = list(
        pd.date_range(
            min(dates),
            max(dates),
            freq="D",
        )
    )

    if dates != expected:
        failures.append(
            "calendar continuity failed"
        )

    if failures:
        raise RuntimeError(
            "TECHNICAL STOP: "
            + "; ".join(failures)
        )


def build_inventory_change_qa(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    dates = sorted(
        data["date"].drop_duplicates()
    )

    for prev_date, cur_date in zip(
        dates[:-1],
        dates[1:],
    ):
        prev = (
            data[
                data["date"] == prev_date
            ][
                [
                    "machine_no",
                    "machine_name",
                ]
            ]
            .set_index("machine_no")
            ["machine_name"]
        )

        cur = (
            data[
                data["date"] == cur_date
            ][
                [
                    "machine_no",
                    "machine_name",
                ]
            ]
            .set_index("machine_no")
            ["machine_name"]
        )

        common = (
            prev.index.intersection(
                cur.index
            )
        )

        machine_name_change_count = int(
            (
                prev.loc[common]
                != cur.loc[common]
            ).sum()
        )

        added = int(
            len(
                cur.index.difference(
                    prev.index
                )
            )
        )

        removed = int(
            len(
                prev.index.difference(
                    cur.index
                )
            )
        )

        inventory_change_count = (
            machine_name_change_count
            + added
            + removed
        )

        rows.append(
            {
                "previous_date":
                    pd.Timestamp(
                        prev_date
                    ).date().isoformat(),
                "date":
                    pd.Timestamp(
                        cur_date
                    ).date().isoformat(),
                "inventory_change_count":
                    inventory_change_count,
                "machine_name_change_count":
                    machine_name_change_count,
                "added_machine_count":
                    added,
                "removed_machine_count":
                    removed,
            }
        )

    return pd.DataFrame(rows)


def make_synthetic_target(
    prior_inventory: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    synthetic = (
        prior_inventory[
            [
                "machine_no",
                "machine_name",
            ]
        ]
        .copy()
    )

    synthetic["date"] = target_date
    synthetic["G"] = 0.0
    synthetic["diff"] = 0.0
    synthetic["win"] = 0
    synthetic["plus1000"] = 0
    synthetic["plus2000"] = 0

    return synthetic



def apply_inventory_reset_history(
    hist: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only the current machine-name segment for each machine_no.

    This resets machine-specific history when a machine number changes
    machine_name.  It uses history strictly before the target date.

    Returns
    -------
    reset_hist:
        History with obsolete machine-name segments removed per machine_no.
    reset_qa:
        One row per machine_no describing whether/how much history was reset.
    """
    kept = []
    qa_rows = []

    for machine_no, group in hist.groupby(
        "machine_no",
        sort=False,
    ):
        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )

        names = group["machine_name"].astype(str)
        current_name = str(names.iloc[-1])

        change_mask = names.ne(
            names.shift(1)
        )
        segment_id = change_mask.cumsum()
        current_segment_id = int(
            segment_id.iloc[-1]
        )

        current_segment = group[
            segment_id == current_segment_id
        ].copy()

        kept.append(current_segment)

        qa_rows.append(
            {
                "machine_no":
                    int(machine_no),
                "current_machine_name":
                    current_name,
                "history_rows_before_reset":
                    int(len(group)),
                "history_rows_after_reset":
                    int(len(current_segment)),
                "history_rows_removed":
                    int(
                        len(group)
                        - len(current_segment)
                    ),
                "reset_applied":
                    bool(
                        len(current_segment)
                        < len(group)
                    ),
                "current_segment_start":
                    pd.Timestamp(
                        current_segment[
                            "date"
                        ].min()
                    ).date().isoformat(),
            }
        )

    reset_hist = pd.concat(
        kept,
        ignore_index=True,
    )

    reset_qa = pd.DataFrame(
        qa_rows
    )

    return reset_hist, reset_qa


def build_ranked_target(
    source_module,
    data: pd.DataFrame,
    target_date: pd.Timestamp,
    weights: dict[str, float],
    inventory_reset: bool = False,
):
    hist = data[
        data["date"] < target_date
    ].copy()

    actual = data[
        data["date"] == target_date
    ].copy()

    if hist.empty or actual.empty:
        raise RuntimeError(
            f"{target_date.date()}: "
            "missing history or actual"
        )

    latest_history_date = (
        hist["date"].max()
    )

    if not (
        latest_history_date
        < target_date
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "latest history date is not "
            "before target"
        )

    prior_inventory = (
        hist[
            hist["date"]
            == latest_history_date
        ][
            [
                "machine_no",
                "machine_name",
            ]
        ]
        .copy()
        .sort_values("machine_no")
        .reset_index(drop=True)
    )

    if len(prior_inventory) != EXPECTED_MACHINES:
        raise RuntimeError(
            f"TECHNICAL STOP: "
            f"{target_date.date()} "
            f"prior inventory count="
            f"{len(prior_inventory)}"
        )

    if (
        prior_inventory["machine_no"]
        .nunique()
        != EXPECTED_MACHINES
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "prior inventory unique "
            "machine count failed"
        )

    synthetic = make_synthetic_target(
        prior_inventory,
        target_date,
    )

    full_model_input = pd.concat(
        [
            hist,
            synthetic,
        ],
        ignore_index=True,
    )

    edge_distance_map = (
        source_module
        .build_number_edge_distance(
            prior_inventory[
                "machine_no"
            ]
            .astype(int)
            .tolist()
        )
    )

    reference_panel = (
        source_module.build_features(
            full_model_input,
            target_date,
            edge_distance_map,
        )
    )

    reset_qa = pd.DataFrame()

    if inventory_reset:
        reset_hist, reset_qa = (
            apply_inventory_reset_history(
                hist
            )
        )

        reset_model_input = pd.concat(
            [
                reset_hist,
                synthetic,
            ],
            ignore_index=True,
        )

        panel = source_module.build_features(
            reset_model_input,
            target_date,
            edge_distance_map,
        )

        # type_avg is a store/model-type statistic, not a machine-number
        # identity statistic.  Preserve the CONTROL value so the A/B
        # difference isolates machine-specific history reset.
        reference_type_avg = (
            reference_panel[
                [
                    "machine_no",
                    "type_avg",
                ]
            ]
            .rename(
                columns={
                    "type_avg":
                        "_control_type_avg",
                }
            )
        )

        panel = panel.merge(
            reference_type_avg,
            on="machine_no",
            how="left",
            validate="one_to_one",
        )

        panel["type_avg"] = (
            panel["_control_type_avg"]
        )
        panel = panel.drop(
            columns=[
                "_control_type_avg",
            ]
        )
    else:
        panel = reference_panel.copy()

    if panel.empty:
        raise RuntimeError(
            "TECHNICAL STOP: "
            f"{target_date.date()} "
            "feature panel empty"
        )

    if len(panel) != EXPECTED_MACHINES:
        raise RuntimeError(
            "TECHNICAL STOP: "
            f"{target_date.date()} "
            f"feature rows={len(panel)}"
        )

    if (
        panel["machine_no"].nunique()
        != EXPECTED_MACHINES
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "feature machine uniqueness "
            "failed"
        )

    missing_features = [
        x
        for x in ACTIVE_FEATURES
        if x not in panel.columns
    ]

    if missing_features:
        raise RuntimeError(
            "TECHNICAL STOP: "
            f"missing features "
            f"{missing_features}"
        )

    ranked = (
        source_module
        .rank_score(
            panel,
            weights,
        )
        .reset_index(drop=True)
    )

    if len(ranked) != EXPECTED_MACHINES:
        raise RuntimeError(
            "TECHNICAL STOP: "
            "ranked row count failed"
        )

    ranked[
        "prediction_rank"
    ] = (
        np.arange(
            1,
            len(ranked) + 1,
        )
    )

    actual_eval = (
        actual[
            [
                "machine_no",
                "machine_name",
                "diff",
                "G",
            ]
        ]
        .copy()
        .rename(
            columns={
                "machine_name":
                    "actual_machine_name",
                "diff":
                    "actual_diff",
                "G":
                    "actual_G",
            }
        )
    )

    evaluated = ranked.merge(
        actual_eval,
        on="machine_no",
        how="left",
        validate="one_to_one",
    )

    if (
        evaluated["actual_diff"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "target actual missing after "
            "machine_no evaluation join"
        )

    evaluated.insert(
        0,
        "target_date",
        target_date.date().isoformat(),
    )

    evaluated.insert(
        1,
        "inventory_source_date",
        latest_history_date
        .date()
        .isoformat(),
    )

    evaluated[
        "machine_name_changed_on_target"
    ] = (
        evaluated["machine_name"]
        != evaluated[
            "actual_machine_name"
        ]
    )

    changed_on_target = int(
        evaluated[
            "machine_name_changed_on_target"
        ].sum()
    )

    qa_row = {
        "target_date":
            target_date.date().isoformat(),
        "latest_history_date":
            latest_history_date
            .date()
            .isoformat(),
        "latest_history_before_target":
            bool(
                latest_history_date
                < target_date
            ),
        "inventory_source_date":
            latest_history_date
            .date()
            .isoformat(),
        "prior_inventory_rows":
            len(prior_inventory),
        "feature_rows":
            len(panel),
        "ranking_rows":
            len(ranked),
        "top10_rows":
            min(10, len(ranked)),
        "target_day_actual_used_for_features":
            False,
        "target_day_machine_name_used_for_ranking":
            False,
        "target_day_machine_name_change_count":
            changed_on_target,
        "ranking_inventory_is_prior_known":
            True,
        "inventory_reset":
            bool(inventory_reset),
        "reset_machine_count":
            (
                int(
                    reset_qa[
                        "reset_applied"
                    ].sum()
                )
                if inventory_reset
                and not reset_qa.empty
                else 0
            ),
        "reset_history_rows_removed":
            (
                int(
                    reset_qa[
                        "history_rows_removed"
                    ].sum()
                )
                if inventory_reset
                and not reset_qa.empty
                else 0
            ),
    }

    evaluated[
        "ab_variant"
    ] = (
        "B_INVENTORY_RESET"
        if inventory_reset
        else "A_CONTROL"
    )

    return evaluated, qa_row


def daily_band_metrics(
    evaluated: pd.DataFrame,
    target_date: pd.Timestamp,
    band: int,
):
    selected = (
        evaluated
        .sort_values(
            "prediction_rank"
        )
        .head(band)
        .copy()
    )

    actual_diff = (
        pd.to_numeric(
            selected["actual_diff"],
            errors="raise",
        )
    )

    store_diff = (
        pd.to_numeric(
            evaluated["actual_diff"],
            errors="raise",
        )
    )

    selected_avg = float(
        actual_diff.mean()
    )

    store_avg = float(
        store_diff.mean()
    )

    lift = (
        selected_avg
        - store_avg
    )

    return {
        "target_date":
            target_date.date().isoformat(),
        "band":
            f"TOP{band}",
        "selected_rows":
            len(selected),
        "avg_diff":
            selected_avg,
        "median_diff":
            float(
                actual_diff.median()
            ),
        "win_rate":
            float(
                (
                    actual_diff > 0
                ).mean()
            ),
        "plus1000_rate":
            float(
                (
                    actual_diff >= 1000
                ).mean()
            ),
        "plus2000_rate":
            float(
                (
                    actual_diff >= 2000
                ).mean()
            ),
        "total_diff":
            float(
                actual_diff.sum()
            ),
        "positive_day":
            int(
                selected_avg > 0
            ),
        "store_avg_diff":
            store_avg,
        "store_win_rate":
            float(
                (
                    store_diff > 0
                ).mean()
            ),
        "store_plus1000_rate":
            float(
                (
                    store_diff >= 1000
                ).mean()
            ),
        "store_plus2000_rate":
            float(
                (
                    store_diff >= 2000
                ).mean()
            ),
        "store_relative_lift":
            lift,
        "excess_vs_store":
            float(
                lift
                * len(selected)
            ),
    }


def bootstrap_mean_ci(
    values: pd.Series,
    seed: int,
):
    x = (
        pd.to_numeric(
            values,
            errors="raise",
        )
        .to_numpy(
            dtype=float
        )
    )

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        seed
    )

    index = rng.integers(
        0,
        len(x),
        size=(
            BOOTSTRAP_REPS,
            len(x),
        ),
    )

    means = x[index].mean(
        axis=1
    )

    low, high = np.quantile(
        means,
        [0.025, 0.975],
    )

    return float(low), float(high)


def build_leave_one_day_out(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for band, group in daily.groupby(
        "band",
        sort=False,
    ):
        group = (
            group
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        full_mean = float(
            group[
                "store_relative_lift"
            ].mean()
        )

        full_sign = int(
            np.sign(full_mean)
        )

        for _, row in group.iterrows():
            rest = group[
                group["target_date"]
                != row["target_date"]
            ]

            if rest.empty:
                loo_mean = np.nan
                reversal = False
            else:
                loo_mean = float(
                    rest[
                        "store_relative_lift"
                    ].mean()
                )

                loo_sign = int(
                    np.sign(
                        loo_mean
                    )
                )

                reversal = bool(
                    loo_sign != full_sign
                )

            rows.append(
                {
                    "band":
                        band,
                    "omitted_date":
                        row["target_date"],
                    "full_mean_store_relative_lift":
                        full_mean,
                    "loo_mean_store_relative_lift":
                        loo_mean,
                    "direction_reversal":
                        reversal,
                }
            )

    return pd.DataFrame(rows)


def overall_metrics(
    daily: pd.DataFrame,
    loo: pd.DataFrame,
    seed_base: int,
) -> pd.DataFrame:
    rows = []

    for band, group in daily.groupby(
        "band",
        sort=False,
    ):
        group = (
            group
            .sort_values("target_date")
            .reset_index(drop=True)
        )

        band_number = int(
            band.replace(
                "TOP",
                "",
            )
        )

        abs_low, abs_high = (
            bootstrap_mean_ci(
                group["avg_diff"],
                seed_base
                + band_number,
            )
        )

        lift_low, lift_high = (
            bootstrap_mean_ci(
                group[
                    "store_relative_lift"
                ],
                seed_base
                + 1000
                + band_number,
            )
        )

        band_loo = loo[
            loo["band"] == band
        ]

        best_idx = (
            group[
                "store_relative_lift"
            ].idxmax()
        )

        worst_idx = (
            group[
                "store_relative_lift"
            ].idxmin()
        )

        mean_lift = float(
            group[
                "store_relative_lift"
            ].mean()
        )

        reversal_any = bool(
            band_loo[
                "direction_reversal"
            ].any()
        )

        if band == "TOP10":
            if (
                mean_lift > 0
                and lift_low > 0
                and not reversal_any
            ):
                verdict = (
                    "EXTERNAL_SIGNAL_SUPPORTED"
                )
            elif (
                mean_lift > 0
                and lift_low <= 0
                <= lift_high
                and not reversal_any
            ):
                verdict = (
                    "DIRECTIONAL_ONLY"
                )
            else:
                verdict = (
                    "NOT_REPLICATED"
                )
        else:
            verdict = "SUPPORTIVE_ONLY"

        rows.append(
            {
                "band":
                    band,
                "evaluated_days":
                    int(
                        group[
                            "target_date"
                        ].nunique()
                    ),
                "selected_rows":
                    int(
                        group[
                            "selected_rows"
                        ].sum()
                    ),
                "mean_daily_avg_diff":
                    float(
                        group[
                            "avg_diff"
                        ].mean()
                    ),
                "median_daily_avg_diff":
                    float(
                        group[
                            "avg_diff"
                        ].median()
                    ),
                "mean_win_rate":
                    float(
                        group[
                            "win_rate"
                        ].mean()
                    ),
                "positive_day_rate":
                    float(
                        group[
                            "positive_day"
                        ].mean()
                    ),
                "total_diff":
                    float(
                        group[
                            "total_diff"
                        ].sum()
                    ),
                "mean_store_avg_diff":
                    float(
                        group[
                            "store_avg_diff"
                        ].mean()
                    ),
                "mean_store_relative_lift":
                    mean_lift,
                "total_excess_vs_store":
                    float(
                        group[
                            "excess_vs_store"
                        ].sum()
                    ),
                "absolute_bootstrap_ci95_lower":
                    abs_low,
                "absolute_bootstrap_ci95_upper":
                    abs_high,
                "lift_bootstrap_ci95_lower":
                    lift_low,
                "lift_bootstrap_ci95_upper":
                    lift_high,
                "best_day":
                    str(
                        group.loc[
                            best_idx,
                            "target_date",
                        ]
                    ),
                "best_day_lift":
                    float(
                        group.loc[
                            best_idx,
                            "store_relative_lift",
                        ]
                    ),
                "worst_day":
                    str(
                        group.loc[
                            worst_idx,
                            "target_date",
                        ]
                    ),
                "worst_day_lift":
                    float(
                        group.loc[
                            worst_idx,
                            "store_relative_lift",
                        ]
                    ),
                "loo_direction_reversal":
                    reversal_any,
                "loo_min_mean_lift":
                    float(
                        band_loo[
                            "loo_mean_store_relative_lift"
                        ].min()
                    ),
                "loo_max_mean_lift":
                    float(
                        band_loo[
                            "loo_mean_store_relative_lift"
                        ].max()
                    ),
                "phase1_verdict":
                    verdict,
            }
        )

    return pd.DataFrame(rows)


def build_baseline(
    data: pd.DataFrame,
    targets: list[pd.Timestamp],
) -> pd.DataFrame:
    rows = []

    for target in targets:
        day = data[
            data["date"] == target
        ]

        diff = pd.to_numeric(
            day["diff"],
            errors="raise",
        )

        rows.append(
            {
                "target_date":
                    target.date().isoformat(),
                "store_rows":
                    len(day),
                "store_avg_diff":
                    float(
                        diff.mean()
                    ),
                "store_win_rate":
                    float(
                        (
                            diff > 0
                        ).mean()
                    ),
                "store_plus1000_rate":
                    float(
                        (
                            diff >= 1000
                        ).mean()
                    ),
                "store_plus2000_rate":
                    float(
                        (
                            diff >= 2000
                        ).mean()
                    ),
                "store_total_diff":
                    float(
                        diff.sum()
                    ),
            }
        )

    return pd.DataFrame(rows)



def build_ab_daily_comparison(
    daily_a: pd.DataFrame,
    daily_b: pd.DataFrame,
) -> pd.DataFrame:
    key = [
        "target_date",
        "band",
    ]

    cols = key + [
        "avg_diff",
        "win_rate",
        "store_relative_lift",
        "total_diff",
    ]

    paired = daily_a[cols].merge(
        daily_b[cols],
        on=key,
        how="inner",
        suffixes=(
            "_A",
            "_B",
        ),
        validate="one_to_one",
    )

    for metric in (
        "avg_diff",
        "win_rate",
        "store_relative_lift",
        "total_diff",
    ):
        paired[
            f"{metric}_B_minus_A"
        ] = (
            paired[f"{metric}_B"]
            - paired[f"{metric}_A"]
        )

    return paired


def build_ab_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for band, group in paired.groupby(
        "band",
        sort=False,
    ):
        band_number = int(
            band.replace(
                "TOP",
                "",
            )
        )

        delta = pd.to_numeric(
            group[
                "store_relative_lift_B_minus_A"
            ],
            errors="raise",
        )

        ci_low, ci_high = (
            bootstrap_mean_ci(
                delta,
                BOOTSTRAP_SEED
                + 50000
                + band_number,
            )
        )

        rows.append(
            {
                "band":
                    band,
                "paired_days":
                    int(len(group)),
                "mean_A_store_relative_lift":
                    float(
                        group[
                            "store_relative_lift_A"
                        ].mean()
                    ),
                "mean_B_store_relative_lift":
                    float(
                        group[
                            "store_relative_lift_B"
                        ].mean()
                    ),
                "mean_B_minus_A_lift":
                    float(delta.mean()),
                "median_B_minus_A_lift":
                    float(delta.median()),
                "B_better_days":
                    int((delta > 0).sum()),
                "A_better_days":
                    int((delta < 0).sum()),
                "tie_days":
                    int((delta == 0).sum()),
                "delta_bootstrap_ci95_lower":
                    ci_low,
                "delta_bootstrap_ci95_upper":
                    ci_high,
            }
        )

    return pd.DataFrame(rows)


def run():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = load_module(
        SOURCE_MODEL_PATH
    )

    weights = (
        source
        .V42_C_WEIGHTS
        .copy()
    )

    fp = fingerprint(
        weights
    )

    if fp != EXPECTED_FINGERPRINT:
        raise RuntimeError(
            "TECHNICAL STOP: "
            f"fingerprint mismatch "
            f"{fp}"
        )

    if tuple(
        weights.keys()
    ) != ACTIVE_FEATURES:
        raise RuntimeError(
            "TECHNICAL STOP: "
            "active feature order/definition "
            "mismatch"
        )

    data, input_qa, files = (
        load_daily_files()
    )

    input_qa.to_csv(
        OUTPUT_DIR
        / "01_input_qa.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validate_input_qa(
        data,
        input_qa,
    )

    inventory_qa = (
        build_inventory_change_qa(
            data
        )
    )

    inventory_qa.to_csv(
        OUTPUT_DIR
        / "02_inventory_change_qa.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dates = sorted(
        data[
            "date"
        ].drop_duplicates()
    )

    if len(dates) <= SENSITIVITY_BURNIN_DAYS:
        raise RuntimeError(
            "TECHNICAL STOP: "
            "insufficient data days"
        )

    primary_targets = [
        pd.Timestamp(x)
        for x in dates[
            PRIMARY_BURNIN_DAYS:
        ]
    ]

    sensitivity_targets = [
        pd.Timestamp(x)
        for x in dates[
            SENSITIVITY_BURNIN_DAYS:
        ]
    ]

    expected_first_primary = (
        pd.Timestamp(
            dates[
                PRIMARY_BURNIN_DAYS
            ]
        )
    )

    if (
        primary_targets[0]
        != expected_first_primary
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "unexpected first primary target"
        )

    expected_primary_days = (
        len(dates)
        - PRIMARY_BURNIN_DAYS
    )
    expected_sensitivity_days = (
        len(dates)
        - SENSITIVITY_BURNIN_DAYS
    )

    if (
        len(primary_targets)
        != expected_primary_days
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "primary target count mismatch"
        )

    if (
        len(sensitivity_targets)
        != expected_sensitivity_days
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "sensitivity target count mismatch"
        )

    eligibility_rows = []
    feature_qa_rows = []
    all_score_a_frames = []
    all_score_b_frames = []
    top10_a_frames = []
    top10_b_frames = []
    daily_a_rows = []
    daily_b_rows = []

    ranked_a_cache = {}
    ranked_b_cache = {}

    for target in primary_targets:
        evaluated_a, qa_a = (
            build_ranked_target(
                source,
                data,
                target,
                weights,
                inventory_reset=False,
            )
        )

        evaluated_b, qa_b = (
            build_ranked_target(
                source,
                data,
                target,
                weights,
                inventory_reset=True,
            )
        )

        ranked_a_cache[target] = (
            evaluated_a
        )
        ranked_b_cache[target] = (
            evaluated_b
        )

        qa_a["ab_variant"] = (
            "A_CONTROL"
        )
        qa_b["ab_variant"] = (
            "B_INVENTORY_RESET"
        )

        feature_qa_rows.extend(
            [
                qa_a,
                qa_b,
            ]
        )

        prior_days = int(
            data[
                data["date"]
                < target
            ]["date"].nunique()
        )

        eligibility_rows.append(
            {
                "target_date":
                    target.date().isoformat(),
                "prior_days":
                    prior_days,
                "primary_burnin_required":
                    PRIMARY_BURNIN_DAYS,
                "primary_eligible":
                    prior_days
                    >= PRIMARY_BURNIN_DAYS,
                "sensitivity_burnin_required":
                    SENSITIVITY_BURNIN_DAYS,
                "sensitivity_eligible":
                    prior_days
                    >= SENSITIVITY_BURNIN_DAYS,
                "latest_history_date":
                    qa_a[
                        "latest_history_date"
                    ],
                "inventory_source_date":
                    qa_a[
                        "inventory_source_date"
                    ],
            }
        )

        all_score_a_frames.append(
            evaluated_a
        )
        all_score_b_frames.append(
            evaluated_b
        )

        top10_a_frames.append(
            evaluated_a
            .sort_values(
                "prediction_rank"
            )
            .head(10)
            .copy()
        )
        top10_b_frames.append(
            evaluated_b
            .sort_values(
                "prediction_rank"
            )
            .head(10)
            .copy()
        )

        for n in TOP_NS:
            row_a = daily_band_metrics(
                evaluated_a,
                target,
                n,
            )
            row_a["ab_variant"] = (
                "A_CONTROL"
            )
            daily_a_rows.append(
                row_a
            )

            row_b = daily_band_metrics(
                evaluated_b,
                target,
                n,
            )
            row_b["ab_variant"] = (
                "B_INVENTORY_RESET"
            )
            daily_b_rows.append(
                row_b
            )

    feature_qa = pd.DataFrame(
        feature_qa_rows
    )
    eligibility = pd.DataFrame(
        eligibility_rows
    )

    all_scores_a = pd.concat(
        all_score_a_frames,
        ignore_index=True,
    )
    all_scores_b = pd.concat(
        all_score_b_frames,
        ignore_index=True,
    )

    top10_a = pd.concat(
        top10_a_frames,
        ignore_index=True,
    )
    top10_b = pd.concat(
        top10_b_frames,
        ignore_index=True,
    )

    daily_a = pd.DataFrame(
        daily_a_rows
    )
    daily_b = pd.DataFrame(
        daily_b_rows
    )

    if not bool(
        feature_qa[
            "latest_history_before_target"
        ].all()
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "future leakage history guard"
        )

    if bool(
        feature_qa[
            "target_day_actual_used_for_features"
        ].any()
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "target actual leakage"
        )

    if bool(
        feature_qa[
            "target_day_machine_name_used_for_ranking"
        ].any()
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "target machine_name leakage"
        )

    if not bool(
        (
            feature_qa[
                "ranking_rows"
            ]
            == EXPECTED_MACHINES
        ).all()
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "ranking row count"
        )

    if not bool(
        (
            feature_qa[
                "top10_rows"
            ]
            == 10
        ).all()
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "Top10 unavailable"
        )

    loo_a = build_leave_one_day_out(
        daily_a
    )
    loo_b = build_leave_one_day_out(
        daily_b
    )

    overall_a = overall_metrics(
        daily_a,
        loo_a,
        BOOTSTRAP_SEED,
    )
    overall_a.insert(
        0,
        "ab_variant",
        "A_CONTROL",
    )

    overall_b = overall_metrics(
        daily_b,
        loo_b,
        BOOTSTRAP_SEED + 10000,
    )
    overall_b.insert(
        0,
        "ab_variant",
        "B_INVENTORY_RESET",
    )

    paired = build_ab_daily_comparison(
        daily_a,
        daily_b,
    )
    ab_summary = build_ab_summary(
        paired
    )

    baseline = build_baseline(
        data,
        primary_targets,
    )

    sensitivity_a_rows = []
    sensitivity_b_rows = []

    for target in sensitivity_targets:
        evaluated_a = (
            ranked_a_cache[target]
        )
        evaluated_b = (
            ranked_b_cache[target]
        )

        for n in TOP_NS:
            row_a = daily_band_metrics(
                evaluated_a,
                target,
                n,
            )
            row_a["ab_variant"] = (
                "A_CONTROL"
            )
            sensitivity_a_rows.append(
                row_a
            )

            row_b = daily_band_metrics(
                evaluated_b,
                target,
                n,
            )
            row_b["ab_variant"] = (
                "B_INVENTORY_RESET"
            )
            sensitivity_b_rows.append(
                row_b
            )

    sensitivity_a = pd.DataFrame(
        sensitivity_a_rows
    )
    sensitivity_b = pd.DataFrame(
        sensitivity_b_rows
    )

    sensitivity_paired = (
        build_ab_daily_comparison(
            sensitivity_a,
            sensitivity_b,
        )
    )
    sensitivity_summary = (
        build_ab_summary(
            sensitivity_paired
        )
    )

    # A and B must be exactly identical before any machine-number
    # inventory reset has become applicable.
    no_reset_dates = set(
        feature_qa[
            (
                feature_qa["ab_variant"]
                == "B_INVENTORY_RESET"
            )
            & (
                feature_qa[
                    "reset_machine_count"
                ]
                == 0
            )
        ]["target_date"]
    )

    for date_text in no_reset_dates:
        a_day = (
            all_scores_a[
                all_scores_a[
                    "target_date"
                ]
                == date_text
            ]
            .sort_values("machine_no")
            .reset_index(drop=True)
        )
        b_day = (
            all_scores_b[
                all_scores_b[
                    "target_date"
                ]
                == date_text
            ]
            .sort_values("machine_no")
            .reset_index(drop=True)
        )

        if not np.allclose(
            a_day["score"].to_numpy(),
            b_day["score"].to_numpy(),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(
                "TECHNICAL STOP: "
                "A/B differ on no-reset date "
                f"{date_text}"
            )

    feature_reproduction_qa = (
        pd.DataFrame(
            [
                {
                    "model":
                        MODEL_NAME,
                    "weight_fingerprint":
                        fp,
                    "fingerprint_ok":
                        fp
                        == EXPECTED_FINGERPRINT,
                    "weight_sum":
                        float(
                            sum(
                                weights.values()
                            )
                        ),
                    "active_features":
                        "|".join(
                            ACTIVE_FEATURES
                        ),
                    "feature_source":
                        str(
                            SOURCE_MODEL_PATH
                        ),
                    "feature_function":
                        "build_features",
                    "ranking_function":
                        "rank_score",
                    "A_policy":
                        "CONTROL: full machine_no history",
                    "B_policy":
                        "reset machine-specific history at latest machine_name segment",
                    "B_type_avg_policy":
                        "preserve CONTROL type_avg to isolate machine-specific reset",
                    "target_actual_used_for_ranking":
                        False,
                    "target_machine_name_used_for_ranking":
                        False,
                    "inventory_policy":
                        "latest known inventory strictly before target",
                    "production_outputs_written":
                        False,
                }
            ]
        )
    )

    outputs = {
        "03_feature_reproduction_qa.csv":
            feature_reproduction_qa,
        "04_target_eligibility.csv":
            eligibility,
        "05_feature_qa_ab.csv":
            feature_qa,
        "06_A_daily_all517_score.csv":
            all_scores_a,
        "07_B_daily_all517_score.csv":
            all_scores_b,
        "08_A_daily_top10.csv":
            top10_a,
        "09_B_daily_top10.csv":
            top10_b,
        "10_A_daily_evaluation.csv":
            daily_a,
        "11_B_daily_evaluation.csv":
            daily_b,
        "12_A_overall_evaluation.csv":
            overall_a,
        "13_B_overall_evaluation.csv":
            overall_b,
        "14_AB_daily_paired.csv":
            paired,
        "15_AB_summary.csv":
            ab_summary,
        "16_store_baseline.csv":
            baseline,
        "17_A_leave_one_day_out.csv":
            loo_a,
        "18_B_leave_one_day_out.csv":
            loo_b,
        "19_AB_burnin21_daily_paired.csv":
            sensitivity_paired,
        "20_AB_burnin21_summary.csv":
            sensitivity_summary,
    }

    for name, frame in outputs.items():
        frame.to_csv(
            OUTPUT_DIR / name,
            index=False,
            encoding="utf-8-sig",
        )

    metadata_rows = [
        {
            "key": "generated_at",
            "value":
                datetime.now()
                .astimezone()
                .isoformat(),
        },
        {
            "key": "store",
            "value":
                "bic_tsubame_takasaki",
        },
        {
            "key": "mode",
            "value":
                "OFFLINE_RESEARCH_ONLY_AB",
        },
        {
            "key": "model",
            "value":
                MODEL_NAME,
        },
        {
            "key": "weight_fingerprint",
            "value":
                fp,
        },
        {
            "key": "expected_machines",
            "value":
                str(EXPECTED_MACHINES),
        },
        {
            "key": "data_start",
            "value":
                min(dates)
                .date()
                .isoformat(),
        },
        {
            "key": "data_end",
            "value":
                max(dates)
                .date()
                .isoformat(),
        },
        {
            "key": "data_days",
            "value":
                str(len(dates)),
        },
        {
            "key": "primary_minimum_prior_days",
            "value":
                str(PRIMARY_BURNIN_DAYS),
        },
        {
            "key": "primary_target_days",
            "value":
                str(len(primary_targets)),
        },
        {
            "key": "primary_first_target",
            "value":
                primary_targets[0]
                .date()
                .isoformat(),
        },
        {
            "key": "primary_last_target",
            "value":
                primary_targets[-1]
                .date()
                .isoformat(),
        },
        {
            "key": "sensitivity_minimum_prior_days",
            "value":
                str(
                    SENSITIVITY_BURNIN_DAYS
                ),
        },
        {
            "key": "sensitivity_target_days",
            "value":
                str(
                    len(
                        sensitivity_targets
                    )
                ),
        },
        {
            "key": "A_policy",
            "value":
                "CONTROL_FULL_MACHINE_NO_HISTORY",
        },
        {
            "key": "B_policy",
            "value":
                "RESET_MACHINE_HISTORY_ON_MACHINE_NAME_CHANGE",
        },
        {
            "key": "B_type_avg_policy",
            "value":
                "PRESERVE_CONTROL_TYPE_AVG",
        },
        {
            "key": "actual_join_policy",
            "value":
                "evaluation-only by machine_no after ranking",
        },
        {
            "key": "production_outputs_written",
            "value":
                "False",
        },
    ]

    metadata = pd.DataFrame(
        metadata_rows
    )
    metadata.to_csv(
        OUTPUT_DIR
        / "21_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_files = sorted(
        p
        for p in OUTPUT_DIR.iterdir()
        if p.is_file()
        and p.name
        != "22_output_manifest.json"
    )

    manifest = {
        "generated_at":
            datetime.now()
            .astimezone()
            .isoformat(),
        "mode":
            "OFFLINE_RESEARCH_ONLY_AB",
        "model":
            MODEL_NAME,
        "weight_fingerprint":
            fp,
        "production_outputs_written":
            False,
        "source_model_path":
            str(
                SOURCE_MODEL_PATH
            ),
        "source_model_sha256":
            sha256_file(
                SOURCE_MODEL_PATH
            ),
        "input_file_count":
            len(files),
        "outputs": [
            {
                "name":
                    p.name,
                "size_bytes":
                    p.stat().st_size,
                "sha256":
                    sha256_file(p),
            }
            for p in output_files
        ],
    }

    (
        OUTPUT_DIR
        / "22_output_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 96)
    print(
        "Bic Tsubame Takasaki "
        "V4.2_C Inventory Reset A/B V1"
    )
    print("=" * 96)
    print(
        f"model                : "
        f"{MODEL_NAME}"
    )
    print(
        f"fingerprint          : "
        f"{fp}"
    )
    print(
        f"input days           : "
        f"{len(dates)}"
    )
    print(
        f"primary target days  : "
        f"{len(primary_targets)}"
    )
    print(
        f"sensitivity days     : "
        f"{len(sensitivity_targets)}"
    )
    print(
        f"data end             : "
        f"{max(dates).date()}"
    )
    print(
        f"output dir           : "
        f"{OUTPUT_DIR}"
    )
    print()
    print("=== PRIMARY A/B SUMMARY ===")
    print(
        ab_summary.to_string(
            index=False
        )
    )
    print()
    print(
        "NOTE: Positive mean_B_minus_A_lift "
        "favors inventory reset."
    )


def main():
    run()


if __name__ == "__main__":
    main()
