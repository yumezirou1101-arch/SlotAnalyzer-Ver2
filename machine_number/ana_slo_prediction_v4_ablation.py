from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# Ver.4 Ablation Test
# 各因子を1つずつ除外し、残りのウェイトを再正規化して
# OOS期間でVer.4と比較する。
# ============================================================

BASE = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_DIR = (
    BASE
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)

TEST_START = pd.Timestamp("2026-07-26")
TEST_END = pd.Timestamp("2026-08-10")

# ------------------------------------------------------------
# 既存Ver.4の特徴量計算を利用
# ------------------------------------------------------------

from ana_slo_prediction_v4_oos import (
    load_data,
    build_features,
)

# ------------------------------------------------------------
# Ver.4固定ウェイト
# ------------------------------------------------------------

V4_WEIGHTS = {
    "avg31": 0.0670952025611345,
    "recent7_avg": 0.05164896703284082,
    "recent7_win": 0.06602967770818714,
    "last_diff": 0.12382294629381808,
    "prev_change": 0.10484738021281044,
    "weekday_avg": 0.05672674990073483,
    "type_avg": 0.05843723530102936,
    "plus1000_rate": 0.17725354845070532,
    "plus2000_rate": 0.13298938481323394,
    "neighbor_avg": 0.06161296683628432,
    "bounce_signal": 0.09953594088922124,
}

FACTORS = list(V4_WEIGHTS.keys())


# ============================================================
# ウェイトを除外して再正規化
# ============================================================

def make_ablation_weights(exclude_factor=None):

    weights = V4_WEIGHTS.copy()

    if exclude_factor is not None:
        weights[exclude_factor] = 0.0

    total = sum(weights.values())

    if total <= 0:
        raise ValueError(
            "Weight sum became zero."
        )

    for factor in weights:
        weights[factor] /= total

    return weights


# ============================================================
# Z-score
# ============================================================

def zscore(series):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)

    std = float(
        s.std(ddof=0)
    )

    if std == 0 or np.isnan(std):

        return pd.Series(
            0.0,
            index=s.index
        )

    return (
        s - s.mean()
    ) / std


# ============================================================
# スコアリング
# ============================================================

def rank_score(df, weights):

    x = df.copy()

    score = pd.Series(
        0.0,
        index=x.index
    )

    for factor in FACTORS:

        if factor not in x.columns:
            continue

        z = zscore(
            x[factor]
        )

        transformed = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100
        )

        score += (
            transformed
            * weights.get(
                factor,
                0.0
            )
        )

    x["score"] = score

    return x.sort_values(
        "score",
        ascending=False
    )


# ============================================================
# 1日分評価
# ============================================================

def evaluate_day(
    panel,
    weights,
    top_n
):

    if panel.empty:
        return None

    ranked = rank_score(
        panel,
        weights
    )

    top = ranked.head(
        min(
            top_n,
            len(ranked)
        )
    )

    d = pd.to_numeric(
        top["diff"],
        errors="coerce"
    ).dropna()

    if d.empty:
        return None

    return {
        "avg_diff": float(
            d.mean()
        ),
        "median_diff": float(
            d.median()
        ),
        "win_rate": float(
            (d > 0).mean()
            * 100
        ),
        "plus1000_rate": float(
            (d >= 1000).mean()
            * 100
        ),
        "plus2000_rate": float(
            (d >= 2000).mean()
            * 100
        ),
        "positive": int(
            d.sum() > 0
        ),
        "total_diff": float(
            d.sum()
        ),
    }


# ============================================================
# メイン
# ============================================================

def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.4 Ablation Test"
    )
    print("=" * 70)

    print()
    print(
        "BASE MODEL = Ver.4 TOP20_MEAN"
    )

    print()
    print("BASE WEIGHTS")
    print("-" * 70)

    for factor in FACTORS:

        print(
            f"{factor:<18}: "
            f"{V4_WEIGHTS[factor] * 100:7.2f}%"
        )

    print(
        f"weight sum       : "
        f"{sum(V4_WEIGHTS.values()) * 100:7.2f}%"
    )

    print()

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    print(
        f"OOS period = "
        f"{TEST_START.date()} to "
        f"{TEST_END.date()}"
    )

    print()

    # --------------------------------------------------------
    # 日次パネル作成
    # --------------------------------------------------------

    panels = {}

    print(
        "Building daily feature panels..."
    )

    for target_date in pd.date_range(
        TEST_START,
        TEST_END
    ):

        panel = build_features(
            df,
            target_date
        )

        if panel.empty:

            print(
                f"{target_date.date()} "
                f"EMPTY"
            )

            continue

        panels[target_date] = panel

        print(
            f"{target_date.date()} "
            f"machines={len(panel)}"
        )

    print()

    # --------------------------------------------------------
    # モデル作成
    # --------------------------------------------------------

    models = []

    # BASE
    models.append(
        (
            "V4_BASE",
            V4_WEIGHTS.copy(),
            None
        )
    )

    # 各因子を1つずつ除外
    for factor in FACTORS:

        weights = make_ablation_weights(
            factor
        )

        models.append(
            (
                f"ABLATE_{factor}",
                weights,
                factor
            )
        )

    print(
        f"models = {len(models)}"
    )

    print(
        "  V4_BASE + "
        f"{len(FACTORS)} factor ablations"
    )

    print()

    # --------------------------------------------------------
    # 評価
    # --------------------------------------------------------

    daily_rows = []
    summary_rows = []

    for model_name, weights, excluded in models:

        print(
            f"Evaluating {model_name}..."
        )

        for target_date, panel in panels.items():

            for top_n in (
                5,
                10,
                20
            ):

                result = evaluate_day(
                    panel,
                    weights,
                    top_n
                )

                if result is None:
                    continue

                daily_rows.append({

                    "model":
                        model_name,

                    "excluded_factor":
                        excluded
                        if excluded is not None
                        else "",

                    "date":
                        target_date.date(),

                    "top_n":
                        top_n,

                    **result,
                })

        # ----------------------------------------------------
        # summary
        # ----------------------------------------------------

        model_daily = [
            r
            for r in daily_rows
            if r["model"] == model_name
        ]

        for top_n in (
            5,
            10,
            20
        ):

            rows = [
                r
                for r in model_daily
                if r["top_n"] == top_n
            ]

            if not rows:
                continue

            avg_diff = float(
                np.mean(
                    [
                        r["avg_diff"]
                        for r in rows
                    ]
                )
            )

            median_daily_avg = float(
                np.median(
                    [
                        r["avg_diff"]
                        for r in rows
                    ]
                )
            )

            win_rate = float(
                np.mean(
                    [
                        r["win_rate"]
                        for r in rows
                    ]
                )
            )

            plus1000_rate = float(
                np.mean(
                    [
                        r["plus1000_rate"]
                        for r in rows
                    ]
                )
            )

            plus2000_rate = float(
                np.mean(
                    [
                        r["plus2000_rate"]
                        for r in rows
                    ]
                )
            )

            positive_days = float(
                np.mean(
                    [
                        r["positive"]
                        for r in rows
                    ]
                )
                * 100
            )

            total_diff = float(
                np.sum(
                    [
                        r["total_diff"]
                        for r in rows
                    ]
                )
            )

            summary_rows.append({

                "model":
                    model_name,

                "excluded_factor":
                    excluded
                    if excluded is not None
                    else "",

                "top_n":
                    top_n,

                "days":
                    len(rows),

                "avg_diff":
                    avg_diff,

                "median_daily_avg":
                    median_daily_avg,

                "win_rate":
                    win_rate,

                "plus1000_rate":
                    plus1000_rate,

                "plus2000_rate":
                    plus2000_rate,

                "positive_days":
                    positive_days,

                "total_diff":
                    total_diff,
            })

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    daily_df = pd.DataFrame(
        daily_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # BASEとの差分
    # --------------------------------------------------------

    base = summary_df[
        summary_df["model"]
        == "V4_BASE"
    ][
        [
            "top_n",
            "avg_diff",
            "total_diff",
            "win_rate",
            "positive_days"
        ]
    ].copy()

    base = base.rename(
        columns={
            "avg_diff":
                "base_avg_diff",

            "total_diff":
                "base_total_diff",

            "win_rate":
                "base_win_rate",

            "positive_days":
                "base_positive_days",
        }
    )

    comparison_df = summary_df.merge(
        base,
        on="top_n",
        how="left"
    )

    comparison_df[
        "avg_diff_change_vs_base"
    ] = (
        comparison_df["avg_diff"]
        - comparison_df["base_avg_diff"]
    )

    comparison_df[
        "total_diff_change_vs_base"
    ] = (
        comparison_df["total_diff"]
        - comparison_df["base_total_diff"]
    )

    comparison_df[
        "win_rate_change_vs_base"
    ] = (
        comparison_df["win_rate"]
        - comparison_df["base_win_rate"]
    )

    comparison_df[
        "positive_days_change_vs_base"
    ] = (
        comparison_df["positive_days"]
        - comparison_df["base_positive_days"]
    )

    # --------------------------------------------------------
    # TOP10ランキング
    # --------------------------------------------------------

    top10_compare = comparison_df[
        comparison_df["top_n"] == 10
    ].copy()

    top10_compare = top10_compare.sort_values(
        "total_diff",
        ascending=False
    )

    # --------------------------------------------------------
    # ウェイト保存
    # --------------------------------------------------------

    weight_rows = []

    for model_name, weights, excluded in models:

        for factor in FACTORS:

            weight_rows.append({

                "model":
                    model_name,

                "excluded_factor":
                    excluded
                    if excluded is not None
                    else "",

                "factor":
                    factor,

                "weight":
                    weights[factor],
            })

    weights_df = pd.DataFrame(
        weight_rows
    )

    # --------------------------------------------------------
    # 出力
    # --------------------------------------------------------

    out_daily = (
        OUT_DIR
        / "22_Ver4_ablation_daily.csv"
    )

    out_summary = (
        OUT_DIR
        / "22_Ver4_ablation_summary.csv"
    )

    out_compare = (
        OUT_DIR
        / "22_Ver4_ablation_compare.csv"
    )

    out_weights = (
        OUT_DIR
        / "22_Ver4_ablation_weights.csv"
    )

    daily_df.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig"
    )

    comparison_df.to_csv(
        out_compare,
        index=False,
        encoding="utf-8-sig"
    )

    weights_df.to_csv(
        out_weights,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 表示
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ABLATION RESULT"
    )
    print("=" * 70)

    display_cols = [
        "model",
        "top_n",
        "avg_diff",
        "median_daily_avg",
        "win_rate",
        "plus1000_rate",
        "plus2000_rate",
        "positive_days",
        "total_diff",
    ]

    print(
        summary_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=============================================================="
    )
    print(
        "TOP10 ABLATION RANKING"
    )
    print(
        "=============================================================="
    )

    print(
        top10_compare[
            [
                "model",
                "excluded_factor",
                "avg_diff",
                "total_diff",
                "win_rate",
                "positive_days",
                "avg_diff_change_vs_base",
                "total_diff_change_vs_base",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 重要な診断
    # --------------------------------------------------------

    print()
    print(
        "=============================================================="
    )
    print(
        "DIAGNOSTIC"
    )
    print(
        "=============================================================="
    )

    base_top10 = comparison_df[
        (
            comparison_df["model"]
            == "V4_BASE"
        )
        &
        (
            comparison_df["top_n"]
            == 10
        )
    ].iloc[0]

    print(
        f"V4 BASE TOP10 avg diff   : "
        f"{base_top10['avg_diff']:+.2f}"
    )

    print(
        f"V4 BASE TOP10 total diff : "
        f"{base_top10['total_diff']:+.0f}"
    )

    best = top10_compare.iloc[0]

    print()
    print(
        f"BEST TOP10 MODEL          : "
        f"{best['model']}"
    )

    print(
        f"Excluded factor           : "
        f"{best['excluded_factor']}"
    )

    print(
        f"Average diff              : "
        f"{best['avg_diff']:+.2f}"
    )

    print(
        f"Total diff                : "
        f"{best['total_diff']:+.0f}"
    )

    print(
        f"Change vs V4              : "
        f"{best['avg_diff_change_vs_base']:+.2f}"
    )

    print()

    # --------------------------------------------------------
    # 保存先
    # --------------------------------------------------------

    print(
        "Saved:"
    )

    print(
        out_daily
    )

    print(
        out_summary
    )

    print(
        out_compare
    )

    print(
        out_weights
    )

    print()
    print(
        "Ver.4 ablation test complete."
    )


if __name__ == "__main__":
    main()