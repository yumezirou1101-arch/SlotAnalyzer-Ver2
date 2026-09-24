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
    / "analysis_external_v42c_phase1"
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


def build_ranked_target(
    source_module,
    data: pd.DataFrame,
    target_date: pd.Timestamp,
    weights: dict[str, float],
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

    model_input = pd.concat(
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

    panel = source_module.build_features(
        model_input,
        target_date,
        edge_distance_map,
    )

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
    }

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
        / "11_inventory_change_qa.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dates = sorted(
        data[
            "date"
        ].drop_duplicates()
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
        pd.Timestamp("2026-08-22")
    )

    expected_latest = (
        pd.Timestamp("2026-09-23")
    )

    if (
        primary_targets[0]
        != expected_first_primary
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "unexpected first primary target"
        )

    if (
        primary_targets[-1]
        != expected_latest
    ):
        raise RuntimeError(
            "TECHNICAL STOP: "
            "unexpected latest target"
        )

    if len(primary_targets) != 33:
        raise RuntimeError(
            "TECHNICAL STOP: "
            f"primary target days="
            f"{len(primary_targets)}"
        )

    if len(sensitivity_targets) != 26:
        raise RuntimeError(
            "TECHNICAL STOP: "
            f"sensitivity target days="
            f"{len(sensitivity_targets)}"
        )

    eligibility_rows = []
    feature_qa_rows = []
    all_score_frames = []
    top10_frames = []
    daily_rows = []

    ranked_cache = {}

    for target in primary_targets:
        evaluated, feature_qa = (
            build_ranked_target(
                source,
                data,
                target,
                weights,
            )
        )

        ranked_cache[
            target
        ] = evaluated

        feature_qa_rows.append(
            feature_qa
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
                    feature_qa[
                        "latest_history_date"
                    ],
                "inventory_source_date":
                    feature_qa[
                        "inventory_source_date"
                    ],
            }
        )

        all_score_frames.append(
            evaluated
        )

        top10_frames.append(
            evaluated
            .sort_values(
                "prediction_rank"
            )
            .head(10)
            .copy()
        )

        for n in TOP_NS:
            daily_rows.append(
                daily_band_metrics(
                    evaluated,
                    target,
                    n,
                )
            )

    feature_qa = pd.DataFrame(
        feature_qa_rows
    )

    eligibility = pd.DataFrame(
        eligibility_rows
    )

    all_scores = pd.concat(
        all_score_frames,
        ignore_index=True,
    )

    top10 = pd.concat(
        top10_frames,
        ignore_index=True,
    )

    daily = pd.DataFrame(
        daily_rows
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

    loo = (
        build_leave_one_day_out(
            daily
        )
    )

    overall = overall_metrics(
        daily,
        loo,
        BOOTSTRAP_SEED,
    )

    baseline = build_baseline(
        data,
        primary_targets,
    )

    sensitivity_daily_rows = []

    for target in sensitivity_targets:
        evaluated = ranked_cache.get(
            target
        )

        if evaluated is None:
            evaluated, _ = (
                build_ranked_target(
                    source,
                    data,
                    target,
                    weights,
                )
            )

        for n in TOP_NS:
            sensitivity_daily_rows.append(
                daily_band_metrics(
                    evaluated,
                    target,
                    n,
                )
            )

    sensitivity_daily = pd.DataFrame(
        sensitivity_daily_rows
    )

    sensitivity_loo = (
        build_leave_one_day_out(
            sensitivity_daily
        )
    )

    sensitivity_overall = (
        overall_metrics(
            sensitivity_daily,
            sensitivity_loo,
            BOOTSTRAP_SEED
            + 21000,
        )
    )

    feature_reproduction_qa = pd.DataFrame(
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
                "neighbor_definition":
                    "latest_history_day machine_no-1/+1 diff mean; missing neighbors ignored; none=0",
                "zscore_ddof":
                    0,
                "component_formula":
                    "clip(50 + z*12.5, 0, 100)",
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

    feature_reproduction_qa.to_csv(
        OUTPUT_DIR
        / "02_feature_reproduction_qa.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eligibility.to_csv(
        OUTPUT_DIR
        / "03_target_eligibility.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_scores.to_csv(
        OUTPUT_DIR
        / "04_daily_all517_score.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top10.to_csv(
        OUTPUT_DIR
        / "05_daily_top10.csv",
        index=False,
        encoding="utf-8-sig",
    )

    daily.to_csv(
        OUTPUT_DIR
        / "06_daily_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        OUTPUT_DIR
        / "07_overall_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline.to_csv(
        OUTPUT_DIR
        / "08_store_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )

    loo.to_csv(
        OUTPUT_DIR
        / "09_leave_one_day_out.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensitivity_overall.to_csv(
        OUTPUT_DIR
        / "10_burnin21_sensitivity.csv",
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
                "OFFLINE_RESEARCH_ONLY",
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
            "key": "primary_history_policy",
            "value":
                "EXPANDING_HISTORY",
        },
        {
            "key": "primary_minimum_prior_days",
            "value":
                str(
                    PRIMARY_BURNIN_DAYS
                ),
        },
        {
            "key": "primary_target_days",
            "value":
                str(
                    len(
                        primary_targets
                    )
                ),
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
            "key": "bootstrap_seed",
            "value":
                str(
                    BOOTSTRAP_SEED
                ),
        },
        {
            "key": "bootstrap_reps",
            "value":
                str(
                    BOOTSTRAP_REPS
                ),
        },
        {
            "key": "bootstrap_unit",
            "value":
                "DAY",
        },
        {
            "key": "primary_endpoint",
            "value":
                "TOP10 mean_store_relative_lift",
        },
        {
            "key": "best_worst_day_definition",
            "value":
                "store_relative_lift",
        },
        {
            "key": "candidate_inventory_policy",
            "value":
                "latest inventory strictly before target",
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
        / "12_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_files = sorted(
        p
        for p in OUTPUT_DIR.iterdir()
        if p.is_file()
        and p.name
        != "13_output_manifest.json"
    )

    manifest = {
        "generated_at":
            datetime.now()
            .astimezone()
            .isoformat(),
        "mode":
            "OFFLINE_RESEARCH_ONLY",
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
        / "13_output_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print(
        "Bic Tsubame Takasaki "
        "V4.2_C External Validation Phase1"
    )
    print("=" * 88)
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
        f"output dir           : "
        f"{OUTPUT_DIR}"
    )
    print()
    print(
        overall.to_string(
            index=False
        )
    )


def main():
    run()


if __name__ == "__main__":
    main()
