from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 TOP N Optimization
#
# 目的:
#   直前のRolling Walk-Forward OOS結果から
#   V4.2_Cの「何台選ぶのが最適か」を評価する。
#
#   モデルそのものは変更しない。
#   TOP Nだけを比較する。
#
# 入力:
#   29_Ver4_2_walk_forward_daily.csv
#
# 出力:
#   30_Ver4_2_topn_optimization_daily.csv
#   30_Ver4_2_topn_optimization_summary.csv
#   30_Ver4_2_topn_optimization_compare.csv
#   30_Ver4_2_topn_optimization_stability.csv
# ============================================================


from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PATH
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

ANALYSIS_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)


INPUT_FILE = (
    ANALYSIS_DIR
    / "29_Ver4_2_walk_forward_daily.csv"
)


OUTPUT_DAILY = (
    ANALYSIS_DIR
    / "30_Ver4_2_topn_optimization_daily.csv"
)

OUTPUT_SUMMARY = (
    ANALYSIS_DIR
    / "30_Ver4_2_topn_optimization_summary.csv"
)

OUTPUT_COMPARE = (
    ANALYSIS_DIR
    / "30_Ver4_2_topn_optimization_compare.csv"
)

OUTPUT_STABILITY = (
    ANALYSIS_DIR
    / "30_Ver4_2_topn_optimization_stability.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_MODEL = "V4.2_C"

# Walk-Forwardで実際に検証済みのTOP Nだけを使用する。
TOP_NS = [
    5,
    10,
    20,
    30,
]


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print()
    print("INPUT")
    print("-" * 70)
    print(INPUT_FILE)
    print()
    print(
        f"records = {len(df):,}"
    )

    print()
    print(
        "columns ="
    )

    print(
        list(df.columns)
    )

    return df


# ============================================================
# COLUMN CHECK
# ============================================================

def normalize_columns(df):

    required = [
        "model",
        "top_n",
        "date",
        "avg_diff",
        "win_rate",
        "plus1000_rate",
        "plus2000_rate",
        "total_diff",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(missing)
        )

    x = df.copy()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce"
    )

    x["top_n"] = pd.to_numeric(
        x["top_n"],
        errors="coerce"
    )

    for col in [
        "avg_diff",
        "win_rate",
        "plus1000_rate",
        "plus2000_rate",
        "total_diff",
    ]:

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce"
        )

    x = x.dropna(
        subset=[
            "date",
            "top_n",
            "avg_diff",
            "total_diff",
        ]
    ).copy()

    x["top_n"] = (
        x["top_n"]
        .astype(int)
    )

    return x


# ============================================================
# MAX LOSING STREAK
# ============================================================

def max_losing_streak(values):

    streak = 0
    maximum = 0

    for value in values:

        if value < 0:

            streak += 1

            maximum = max(
                maximum,
                streak
            )

        else:

            streak = 0

    return maximum


# ============================================================
# MAX WINNING STREAK
# ============================================================

def max_winning_streak(values):

    streak = 0
    maximum = 0

    for value in values:

        if value > 0:

            streak += 1

            maximum = max(
                maximum,
                streak
            )

        else:

            streak = 0

    return maximum


# ============================================================
# SUMMARY
# ============================================================

def build_summary(df):

    rows = []

    for top_n in TOP_NS:

        x = (
            df[
                df["top_n"]
                == top_n
            ]
            .sort_values("date")
            .copy()
        )

        if x.empty:

            continue

        avg = (
            x["avg_diff"]
            .mean()
        )

        median = (
            x["avg_diff"]
            .median()
        )

        total = (
            x["total_diff"]
            .sum()
        )

        positive_days = int(
            (
                x["avg_diff"]
                > 0
            ).sum()
        )

        negative_days = int(
            (
                x["avg_diff"]
                < 0
            ).sum()
        )

        tie_days = int(
            (
                x["avg_diff"]
                == 0
            ).sum()
        )

        days = len(x)

        positive_rate = (
            positive_days
            / days
            * 100.0
            if days
            else 0.0
        )

        worst_day = float(
            x["avg_diff"].min()
        )

        best_day = float(
            x["avg_diff"].max()
        )

        std = float(
            x["avg_diff"].std(
                ddof=0
            )
        )

        q25 = float(
            x["avg_diff"].quantile(
                0.25
            )
        )

        q75 = float(
            x["avg_diff"].quantile(
                0.75
            )
        )

        max_loss_streak = (
            max_losing_streak(
                x["avg_diff"]
                .tolist()
            )
        )

        max_win_streak = (
            max_winning_streak(
                x["avg_diff"]
                .tolist()
            )
        )

        rows.append({

            "top_n":
                top_n,

            "days":
                days,

            "avg_diff":
                avg,

            "median_diff":
                median,

            "std_diff":
                std,

            "q25_diff":
                q25,

            "q75_diff":
                q75,

            "best_day":
                best_day,

            "worst_day":
                worst_day,

            "win_rate":
                x["win_rate"].mean(),

            "plus1000_rate":
                x["plus1000_rate"].mean(),

            "plus2000_rate":
                x["plus2000_rate"].mean(),

            "positive_days":
                positive_days,

            "negative_days":
                negative_days,

            "tie_days":
                tie_days,

            "positive_day_rate":
                positive_rate,

            "max_losing_streak":
                max_loss_streak,

            "max_winning_streak":
                max_win_streak,

            "total_diff":
                total,

            # 1台あたりの累積差枚。
            # total_diff / (days * top_n)
            "per_machine_avg_diff":
                total
                / (
                    days
                    * top_n
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMPARE
# ============================================================

def build_compare(summary):

    base = summary[
        summary["top_n"] == 10
    ]

    if base.empty:

        raise ValueError(
            "TOP10 result not found."
        )

    base_avg = float(
        base.iloc[0]["avg_diff"]
    )

    base_total = float(
        base.iloc[0]["total_diff"]
    )

    rows = []

    for _, r in summary.iterrows():

        rows.append({

            "top_n":
                int(r["top_n"]),

            "avg_diff":
                float(r["avg_diff"]),

            "avg_diff_vs_top10":
                float(
                    r["avg_diff"]
                    - base_avg
                ),

            "total_diff":
                float(r["total_diff"]),

            "total_diff_vs_top10":
                float(
                    r["total_diff"]
                    - base_total
                ),

            "per_machine_avg_diff":
                float(
                    r[
                        "per_machine_avg_diff"
                    ]
                ),

            "positive_day_rate":
                float(
                    r[
                        "positive_day_rate"
                    ]
                ),

            "max_losing_streak":
                int(
                    r[
                        "max_losing_streak"
                    ]
                ),

            "worst_day":
                float(
                    r["worst_day"]
                ),

            "best_day":
                float(
                    r["best_day"]
                ),
        })

    out = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # 実戦向け総合順位
    #
    # 重要:
    # 単純に平均差枚だけで決めない。
    #
    # 1. 平均差枚
    # 2. プラス日率
    # 3. 1台あたり差枚
    # 4. 最大連敗
    # を確認できるようにする。
    # --------------------------------------------------------

    out = out.sort_values(
        [
            "avg_diff",
            "per_machine_avg_diff",
            "positive_day_rate",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    ).reset_index(
        drop=True
    )

    out["ranking"] = (
        np.arange(
            1,
            len(out) + 1
        )
    )

    return out


# ============================================================
# STABILITY
# ============================================================

def build_stability(df):

    rows = []

    for top_n in TOP_NS:

        x = (
            df[
                df["top_n"]
                == top_n
            ]
            .sort_values("date")
            .copy()
        )

        if x.empty:

            continue

        # ----------------------------------------------------
        # 前半 / 後半
        # ----------------------------------------------------

        n = len(x)

        split_point = (
            n // 2
        )

        first = (
            x.iloc[
                :split_point
            ]
        )

        second = (
            x.iloc[
                split_point:
            ]
        )

        first_avg = (
            first["avg_diff"]
            .mean()
            if not first.empty
            else np.nan
        )

        second_avg = (
            second["avg_diff"]
            .mean()
            if not second.empty
            else np.nan
        )

        # ----------------------------------------------------
        # ブロック別
        # ----------------------------------------------------

        block_values = []

        for _, g in x.groupby(
            x["date"]
            .dt.to_period("W")
        ):

            block_values.append(
                float(
                    g["avg_diff"].mean()
                )
            )

        positive_blocks = sum(
            v > 0
            for v in block_values
        )

        negative_blocks = sum(
            v < 0
            for v in block_values
        )

        # ----------------------------------------------------
        # 下振れリスク
        # ----------------------------------------------------

        worst = float(
            x["avg_diff"].min()
        )

        p10 = float(
            x["avg_diff"].quantile(
                0.10
            )
        )

        # ----------------------------------------------------
        # 累積推移
        # ----------------------------------------------------

        cumulative = (
            x["total_diff"]
            .cumsum()
        )

        max_drawdown = 0.0

        if not cumulative.empty:

            running_max = (
                cumulative
                .cummax()
            )

            drawdown = (
                cumulative
                - running_max
            )

            max_drawdown = float(
                drawdown.min()
            )

        rows.append({

            "top_n":
                top_n,

            "days":
                len(x),

            "first_half_avg":
                first_avg,

            "second_half_avg":
                second_avg,

            "first_second_change":
                second_avg
                - first_avg,

            "positive_blocks":
                positive_blocks,

            "negative_blocks":
                negative_blocks,

            "block_count":
                len(block_values),

            "positive_block_rate":
                (
                    positive_blocks
                    / len(block_values)
                    * 100.0
                    if block_values
                    else 0.0
                ),

            "worst_day":
                worst,

            "p10_day":
                p10,

            "max_drawdown":
                max_drawdown,

            "avg_diff":
                float(
                    x["avg_diff"]
                    .mean()
                ),

            "total_diff":
                float(
                    x["total_diff"]
                    .sum()
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "Ana-Slo Ver.4.2 TOP N Optimization"
    )

    print("=" * 70)

    print()

    print(
        f"TARGET MODEL = {TARGET_MODEL}"
    )

    print(
        f"TOP N = {TOP_NS}"
    )

    print()

    df = load_data()

    df = normalize_columns(
        df
    )

    # --------------------------------------------------------
    # V4.2_Cのみ
    # --------------------------------------------------------

    df = df[
        df["model"]
        == TARGET_MODEL
    ].copy()

    if df.empty:

        raise ValueError(
            f"Model not found: {TARGET_MODEL}"
        )

    df = df[
        df["top_n"]
        .isin(TOP_NS)
    ].copy()

    if df.empty:

        raise ValueError(
            "Requested TOP N results not found."
        )

    df = df.sort_values(
        [
            "date",
            "top_n",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        "FILTERED INPUT"
    )

    print(
        f"model = {TARGET_MODEL}"
    )

    print(
        f"records = {len(df):,}"
    )

    print(
        f"dates = {df['date'].nunique()}"
    )

    print()

    # --------------------------------------------------------
    # DAILY
    # --------------------------------------------------------

    daily = df.copy()

    daily.to_csv(
        OUTPUT_DAILY,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = build_summary(
        df
    )

    print()
    print("=" * 70)
    print(
        "TOP N SUMMARY"
    )
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    compare = build_compare(
        summary
    )

    print()
    print("=" * 70)
    print(
        "TOP N COMPARISON"
    )
    print("=" * 70)

    print(
        compare.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    compare.to_csv(
        OUTPUT_COMPARE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    stability = build_stability(
        df
    )

    print()
    print("=" * 70)
    print(
        "TOP N STABILITY"
    )
    print("=" * 70)

    print(
        stability.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )

    stability.to_csv(
        OUTPUT_STABILITY,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    best = (
        summary
        .sort_values(
            [
                "avg_diff",
                "per_machine_avg_diff",
                "positive_day_rate",
            ],
            ascending=[
                False,
                False,
                False,
            ]
        )
        .iloc[0]
    )

    best_per_machine = (
        summary
        .sort_values(
            "per_machine_avg_diff",
            ascending=False
        )
        .iloc[0]
    )

    most_stable = (
        stability
        .sort_values(
            [
                "positive_block_rate",
                "max_drawdown",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .iloc[0]
    )

    print()
    print("=" * 70)
    print(
        "JUDGMENT"
    )
    print("=" * 70)

    print(
        f"Best by average diff : TOP{int(best['top_n'])}"
    )

    print(
        f"Average diff         : "
        f"{best['avg_diff']:+.2f}"
    )

    print(
        f"Total diff           : "
        f"{best['total_diff']:+.0f}"
    )

    print()

    print(
        f"Best per-machine     : "
        f"TOP{int(best_per_machine['top_n'])}"
    )

    print(
        f"Per-machine avg diff : "
        f"{best_per_machine['per_machine_avg_diff']:+.2f}"
    )

    print()

    print(
        f"Most stable block rate: "
        f"TOP{int(most_stable['top_n'])}"
    )

    print(
        f"Positive block rate  : "
        f"{most_stable['positive_block_rate']:.2f}%"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is a TOP N optimization analysis "
        "using existing OOS Walk-Forward results."
    )

    print(
        "It does not establish statistical significance."
    )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "FILES SAVED"
    )

    print("=" * 70)

    print(
        OUTPUT_DAILY
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_COMPARE
    )

    print(
        OUTPUT_STABILITY
    )

    print()

    print(
        "Ver.4.2 TOP N optimization complete."
    )


if __name__ == "__main__":

    main()