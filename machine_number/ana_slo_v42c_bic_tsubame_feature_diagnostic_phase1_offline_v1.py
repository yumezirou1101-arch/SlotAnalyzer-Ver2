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
    / "analysis_external_v42c_inventory_reset_ab_v1"
)

INPUT_CSV = INPUT_DIR / "06_A_daily_all517_score.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bic_tsubame_takasaki"
    / "machine_number"
    / "analysis_v42c_feature_diagnostic_phase1"
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

BOOTSTRAP_SEED = 20260926
BOOTSTRAP_REPS = 20000


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "v42c_source_feature_diag",
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
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_weights(
    weights: dict[str, float],
) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise RuntimeError(
            "TECHNICAL STOP: non-positive weight sum"
        )
    return {
        key: float(value) / total
        for key, value in weights.items()
    }


def load_input() -> pd.DataFrame:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "target_date",
        "machine_no",
        "machine_name",
        "score",
        "prediction_rank",
        "actual_diff",
        "actual_G",
        *ACTIVE_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "TECHNICAL STOP: missing columns "
            f"{missing}"
        )

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    )

    numeric_cols = [
        "machine_no",
        "score",
        "prediction_rank",
        "actual_diff",
        "actual_G",
        *ACTIVE_FEATURES,
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df.duplicated(
        ["target_date", "machine_no"]
    ).any():
        raise RuntimeError(
            "TECHNICAL STOP: duplicate target_date/machine_no"
        )

    per_day = df.groupby(
        "target_date"
    )["machine_no"].nunique()

    if not bool(
        (per_day == EXPECTED_MACHINES).all()
    ):
        bad = per_day[
            per_day != EXPECTED_MACHINES
        ]
        raise RuntimeError(
            "TECHNICAL STOP: expected 517 machines/day; "
            f"bad={bad.to_dict()}"
        )

    if df[list(ACTIVE_FEATURES)].isna().any().any():
        raise RuntimeError(
            "TECHNICAL STOP: missing active feature value"
        )

    if df["actual_diff"].isna().any():
        raise RuntimeError(
            "TECHNICAL STOP: missing actual_diff"
        )

    return df.sort_values(
        ["target_date", "machine_no"]
    ).reset_index(drop=True)


def rerank(
    source,
    day: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    # rank_score only needs the feature panel. Keep the complete
    # evaluation columns so the returned order can be evaluated.
    ranked = source.rank_score(
        day.copy(),
        weights,
    ).reset_index(drop=True)

    if len(ranked) != EXPECTED_MACHINES:
        raise RuntimeError(
            "TECHNICAL STOP: reranked row count failed"
        )

    if ranked["machine_no"].nunique() != EXPECTED_MACHINES:
        raise RuntimeError(
            "TECHNICAL STOP: reranked machine uniqueness failed"
        )

    ranked["diagnostic_rank"] = np.arange(
        1,
        len(ranked) + 1,
    )
    return ranked


def validate_control_reproduction(
    source,
    data: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    rows = []

    for target, day in data.groupby(
        "target_date",
        sort=True,
    ):
        ranked = rerank(
            source,
            day,
            weights,
        )

        check = (
            day[
                [
                    "machine_no",
                    "score",
                    "prediction_rank",
                ]
            ]
            .rename(
                columns={
                    "score": "stored_score",
                    "prediction_rank": "stored_rank",
                }
            )
            .merge(
                ranked[
                    [
                        "machine_no",
                        "score",
                        "diagnostic_rank",
                    ]
                ].rename(
                    columns={
                        "score": "reproduced_score",
                        "diagnostic_rank": "reproduced_rank",
                    }
                ),
                on="machine_no",
                how="inner",
                validate="one_to_one",
            )
        )

        score_max_abs_error = float(
            np.max(
                np.abs(
                    check["stored_score"]
                    - check["reproduced_score"]
                )
            )
        )
        rank_mismatch_count = int(
            (
                check["stored_rank"]
                != check["reproduced_rank"]
            ).sum()
        )

        rows.append(
            {
                "target_date":
                    target.date().isoformat(),
                "rows":
                    len(check),
                "score_max_abs_error":
                    score_max_abs_error,
                "rank_mismatch_count":
                    rank_mismatch_count,
                "score_reproduced":
                    bool(
                        np.allclose(
                            check["stored_score"],
                            check["reproduced_score"],
                            rtol=0.0,
                            atol=1e-10,
                            equal_nan=True,
                        )
                    ),
                "rank_reproduced":
                    rank_mismatch_count == 0,
            }
        )

    qa = pd.DataFrame(rows)

    if not bool(qa["score_reproduced"].all()):
        raise RuntimeError(
            "TECHNICAL STOP: CONTROL score reproduction failed"
        )

    if not bool(qa["rank_reproduced"].all()):
        raise RuntimeError(
            "TECHNICAL STOP: CONTROL rank reproduction failed"
        )

    return qa


def make_variants(
    control_weights: dict[str, float],
) -> dict[str, dict[str, float]]:
    variants = {
        "CONTROL": control_weights.copy(),
    }

    for feature in ACTIVE_FEATURES:
        reduced = {
            key: value
            for key, value in control_weights.items()
            if key != feature
        }
        variants[
            f"DROP_{feature}"
        ] = normalize_weights(reduced)

    return variants


def daily_band_metrics(
    ranked: pd.DataFrame,
    target: pd.Timestamp,
    variant: str,
    n: int,
) -> dict:
    ranked = ranked.sort_values(
        "diagnostic_rank"
    )

    selected = ranked.head(n)

    store_mean = float(
        ranked["actual_diff"].mean()
    )
    selected_mean = float(
        selected["actual_diff"].mean()
    )

    return {
        "target_date":
            target.date().isoformat(),
        "variant":
            variant,
        "band":
            f"TOP{n}",
        "selected_n":
            int(len(selected)),
        "store_mean_actual_diff":
            store_mean,
        "selected_mean_actual_diff":
            selected_mean,
        "store_relative_lift":
            selected_mean - store_mean,
        "selected_total_actual_diff":
            float(selected["actual_diff"].sum()),
        "selected_win_rate":
            float(
                (selected["actual_diff"] > 0).mean()
            ),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(values)

    # Chunked to keep memory small.
    means = np.empty(
        BOOTSTRAP_REPS,
        dtype=float,
    )
    chunk = 2000
    pos = 0

    while pos < BOOTSTRAP_REPS:
        size = min(
            chunk,
            BOOTSTRAP_REPS - pos,
        )
        sample_idx = rng.integers(
            0,
            n,
            size=(size, n),
        )
        means[
            pos:pos + size
        ] = values[
            sample_idx
        ].mean(axis=1)
        pos += size

    lower, upper = np.quantile(
        means,
        [0.025, 0.975],
    )
    return float(lower), float(upper)


def build_paired(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    control = (
        daily[
            daily["variant"] == "CONTROL"
        ][
            [
                "target_date",
                "band",
                "store_relative_lift",
            ]
        ]
        .rename(
            columns={
                "store_relative_lift":
                    "control_store_relative_lift"
            }
        )
    )

    challenger = daily[
        daily["variant"] != "CONTROL"
    ][
        [
            "target_date",
            "variant",
            "band",
            "store_relative_lift",
        ]
    ].copy()

    paired = challenger.merge(
        control,
        on=["target_date", "band"],
        how="inner",
        validate="many_to_one",
    )

    paired[
        "delta_vs_control"
    ] = (
        paired["store_relative_lift"]
        - paired[
            "control_store_relative_lift"
        ]
    )

    return paired.sort_values(
        ["variant", "band", "target_date"]
    ).reset_index(drop=True)


def summarize_paired(
    paired: pd.DataFrame,
    seed_offset: int = 0,
) -> pd.DataFrame:
    rows = []

    for index, (
        variant,
        band,
    ) in enumerate(
        paired.groupby(
            ["variant", "band"],
            sort=True,
        ).groups.keys()
    ):
        group = paired[
            (paired["variant"] == variant)
            & (paired["band"] == band)
        ].copy()

        delta = group[
            "delta_vs_control"
        ].to_numpy(dtype=float)

        lower, upper = bootstrap_mean_ci(
            delta,
            BOOTSTRAP_SEED
            + seed_offset
            + index,
        )

        rows.append(
            {
                "variant":
                    variant,
                "dropped_feature":
                    variant.removeprefix("DROP_"),
                "band":
                    band,
                "paired_days":
                    len(group),
                "mean_control_store_relative_lift":
                    float(
                        group[
                            "control_store_relative_lift"
                        ].mean()
                    ),
                "mean_variant_store_relative_lift":
                    float(
                        group[
                            "store_relative_lift"
                        ].mean()
                    ),
                "mean_delta_vs_control":
                    float(np.mean(delta)),
                "median_delta_vs_control":
                    float(np.median(delta)),
                "variant_better_days":
                    int((delta > 0).sum()),
                "control_better_days":
                    int((delta < 0).sum()),
                "tie_days":
                    int((delta == 0).sum()),
                "delta_bootstrap_ci95_lower":
                    lower,
                "delta_bootstrap_ci95_upper":
                    upper,
            }
        )

    return pd.DataFrame(rows)


def build_univariate_diagnostic(
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for feature in ACTIVE_FEATURES:
        daily_corrs = []

        for _, day in data.groupby(
            "target_date",
            sort=True,
        ):
            x = day[feature].rank(
                method="average"
            )
            y = day["actual_diff"].rank(
                method="average"
            )
            corr = x.corr(y)
            if pd.notna(corr):
                daily_corrs.append(
                    float(corr)
                )

        pooled_x = data[feature].rank(
            method="average"
        )
        pooled_y = data["actual_diff"].rank(
            method="average"
        )
        pooled_corr = pooled_x.corr(
            pooled_y
        )

        rows.append(
            {
                "feature":
                    feature,
                "days":
                    len(daily_corrs),
                "mean_daily_spearman":
                    float(
                        np.mean(daily_corrs)
                    ),
                "median_daily_spearman":
                    float(
                        np.median(daily_corrs)
                    ),
                "positive_days":
                    int(
                        (
                            np.asarray(
                                daily_corrs
                            )
                            > 0
                        ).sum()
                    ),
                "negative_days":
                    int(
                        (
                            np.asarray(
                                daily_corrs
                            )
                            < 0
                        ).sum()
                    ),
                "pooled_spearman":
                    float(pooled_corr),
            }
        )

    return pd.DataFrame(rows)


def build_assessment(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    top10 = (
        summary[
            summary["band"] == "TOP10"
        ]
        .copy()
        .sort_values(
            "mean_delta_vs_control",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    def label(row) -> str:
        mean_delta = float(
            row["mean_delta_vs_control"]
        )
        lower = float(
            row[
                "delta_bootstrap_ci95_lower"
            ]
        )
        upper = float(
            row[
                "delta_bootstrap_ci95_upper"
            ]
        )

        if mean_delta > 0 and lower > 0:
            return (
                "DROP_IMPROVES_WITH_CI_EXCLUDING_ZERO"
            )
        if mean_delta > 0:
            return (
                "DROP_IMPROVES_MEAN_BUT_UNCERTAIN"
            )
        if mean_delta < 0 and upper < 0:
            return (
                "FEATURE_HELPFUL_WITH_CI_EXCLUDING_ZERO"
            )
        if mean_delta < 0:
            return (
                "FEATURE_HELPFUL_MEAN_BUT_UNCERTAIN"
            )
        return "NO_MEAN_DIFFERENCE"

    top10[
        "phase1_assessment"
    ] = top10.apply(
        label,
        axis=1,
    )

    top10[
        "promotion_decision"
    ] = "RESEARCH_ONLY_NO_AUTO_PROMOTION"

    return top10


def run() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = load_module(
        SOURCE_MODEL_PATH
    )

    control_weights = (
        source.V42_C_WEIGHTS.copy()
    )

    fp = fingerprint(
        control_weights
    )

    if fp != EXPECTED_FINGERPRINT:
        raise RuntimeError(
            "TECHNICAL STOP: "
            f"fingerprint mismatch {fp}"
        )

    if tuple(
        control_weights.keys()
    ) != ACTIVE_FEATURES:
        raise RuntimeError(
            "TECHNICAL STOP: "
            "active feature order/definition mismatch"
        )

    data = load_input()

    targets = sorted(
        data["target_date"].drop_duplicates()
    )

    expected_primary_days = (
        49 - PRIMARY_BURNIN_DAYS
    )
    expected_sensitivity_days = (
        49 - SENSITIVITY_BURNIN_DAYS
    )

    # This Phase 1 intentionally uses the already-produced A_CONTROL
    # target set: 2026-08-22 through 2026-09-25, 35 days.
    if len(targets) != expected_primary_days:
        raise RuntimeError(
            "TECHNICAL STOP: expected 35 primary target days; "
            f"found {len(targets)}"
        )

    if (
        targets[0]
        != pd.Timestamp("2026-08-22")
        or targets[-1]
        != pd.Timestamp("2026-09-25")
    ):
        raise RuntimeError(
            "TECHNICAL STOP: unexpected target date range"
        )

    control_qa = validate_control_reproduction(
        source,
        data,
        control_weights,
    )

    variants = make_variants(
        control_weights
    )

    weight_rows = []
    for variant, weights in variants.items():
        for feature in ACTIVE_FEATURES:
            weight_rows.append(
                {
                    "variant":
                        variant,
                    "feature":
                        feature,
                    "weight":
                        float(
                            weights.get(
                                feature,
                                0.0,
                            )
                        ),
                    "dropped":
                        feature not in weights,
                    "weight_sum":
                        float(
                            sum(
                                weights.values()
                            )
                        ),
                }
            )

    weight_table = pd.DataFrame(
        weight_rows
    )

    daily_rows = []

    for target in targets:
        day = data[
            data["target_date"] == target
        ].copy()

        for variant, weights in variants.items():
            ranked = rerank(
                source,
                day,
                weights,
            )

            for n in TOP_NS:
                daily_rows.append(
                    daily_band_metrics(
                        ranked,
                        target,
                        variant,
                        n,
                    )
                )

    daily = pd.DataFrame(
        daily_rows
    )

    paired = build_paired(
        daily
    )

    summary = summarize_paired(
        paired
    )

    sensitivity_start = (
        targets[
            SENSITIVITY_BURNIN_DAYS
            - PRIMARY_BURNIN_DAYS
        ]
    )

    sensitivity_daily = daily[
        pd.to_datetime(
            daily["target_date"]
        )
        >= sensitivity_start
    ].copy()

    sensitivity_targets = sorted(
        sensitivity_daily[
            "target_date"
        ].unique()
    )

    if (
        len(sensitivity_targets)
        != expected_sensitivity_days
    ):
        raise RuntimeError(
            "TECHNICAL STOP: expected 28 sensitivity days; "
            f"found {len(sensitivity_targets)}"
        )

    sensitivity_paired = build_paired(
        sensitivity_daily
    )
    sensitivity_summary = summarize_paired(
        sensitivity_paired,
        seed_offset=10000,
    )

    univariate = build_univariate_diagnostic(
        data
    )

    assessment = build_assessment(
        summary
    )

    metadata = pd.DataFrame(
        [
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
                "key": "input",
                "value":
                    str(INPUT_CSV),
            },
            {
                "key": "primary_target_days",
                "value":
                    str(len(targets)),
            },
            {
                "key": "primary_first_target",
                "value":
                    targets[0]
                    .date()
                    .isoformat(),
            },
            {
                "key": "primary_last_target",
                "value":
                    targets[-1]
                    .date()
                    .isoformat(),
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
                "key": "primary_endpoint",
                "value":
                    "TOP10 mean_delta_vs_control",
            },
            {
                "key": "ablation_policy",
                "value":
                    "drop exactly one active feature and renormalize remaining weights to 1.0",
            },
            {
                "key": "control_policy",
                "value":
                    "A_CONTROL existing full machine_no history",
            },
            {
                "key": "production_outputs_written",
                "value":
                    "False",
            },
            {
                "key": "auto_promotion",
                "value":
                    "False",
            },
        ]
    )

    outputs = {
        "01_control_reproduction_qa.csv":
            control_qa,
        "02_variant_weights.csv":
            weight_table,
        "03_univariate_diagnostic.csv":
            univariate,
        "04_ablation_daily.csv":
            daily,
        "05_ablation_paired_daily.csv":
            paired,
        "06_ablation_summary.csv":
            summary,
        "07_burnin21_paired_daily.csv":
            sensitivity_paired,
        "08_burnin21_summary.csv":
            sensitivity_summary,
        "09_phase1_top10_assessment.csv":
            assessment,
        "10_metadata.csv":
            metadata,
    }

    for name, frame in outputs.items():
        frame.to_csv(
            OUTPUT_DIR / name,
            index=False,
            encoding="utf-8-sig",
        )

    output_files = sorted(
        p
        for p in OUTPUT_DIR.iterdir()
        if p.is_file()
        and p.name
        != "11_output_manifest.json"
    )

    manifest = {
        "generated_at":
            datetime.now()
            .astimezone()
            .isoformat(),
        "mode":
            "OFFLINE_RESEARCH_ONLY",
        "store":
            "bic_tsubame_takasaki",
        "model":
            MODEL_NAME,
        "weight_fingerprint":
            fp,
        "production_outputs_written":
            False,
        "auto_promotion":
            False,
        "source_model_path":
            str(SOURCE_MODEL_PATH),
        "source_model_sha256":
            sha256_file(
                SOURCE_MODEL_PATH
            ),
        "input_path":
            str(INPUT_CSV),
        "input_sha256":
            sha256_file(
                INPUT_CSV
            ),
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
        / "11_output_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 104)
    print(
        "Bic Tsubame Takasaki "
        "V4.2_C Feature Diagnostic Phase 1"
    )
    print("=" * 104)
    print(
        f"model                : {MODEL_NAME}"
    )
    print(
        f"fingerprint          : {fp}"
    )
    print(
        f"input rows           : {len(data):,}"
    )
    print(
        f"primary target days  : {len(targets)}"
    )
    print(
        f"sensitivity days     : {len(sensitivity_targets)}"
    )
    print(
        f"target range         : "
        f"{targets[0].date()} -> "
        f"{targets[-1].date()}"
    )
    print(
        f"control reproduction : PASS"
    )
    print(
        f"output dir           : {OUTPUT_DIR}"
    )
    print()
    print("=== TOP10 FEATURE ABLATION ===")
    print(
        assessment[
            [
                "dropped_feature",
                "paired_days",
                "mean_control_store_relative_lift",
                "mean_variant_store_relative_lift",
                "mean_delta_vs_control",
                "median_delta_vs_control",
                "variant_better_days",
                "control_better_days",
                "tie_days",
                "delta_bootstrap_ci95_lower",
                "delta_bootstrap_ci95_upper",
                "phase1_assessment",
            ]
        ].to_string(
            index=False
        )
    )
    print()
    print(
        "NOTE: Positive mean_delta_vs_control means "
        "the DROP variant outperformed CONTROL."
    )
    print(
        "NOTE: Research only. No production ranking, "
        "Champion, Forward Guard, or Morning Automation "
        "is changed."
    )


def main():
    run()


if __name__ == "__main__":
    main()
