from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.2 Rolling Walk-Forward Validation
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

CSV1 = DATA_DIR / "ana_slo_20260711.csv"
CSV2 = DATA_DIR / "ana_slo_20260712_20260810.csv"

START = pd.Timestamp("2026-07-11")
TEST_START = pd.Timestamp("2026-07-21")
TEST_END = pd.Timestamp("2026-08-10")

# 4日単位のWalk-Forward評価
BLOCK_SIZE = 4

FACTORS = [
    "avg31",
    "recent7_avg",
    "recent7_win",
    "last_diff",
    "prev_change",
    "weekday_avg",
    "type_avg",
    "plus1000_rate",
    "plus2000_rate",
    "neighbor_avg",
    "bounce_signal",
]

BASE_WEIGHTS = {
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


MODELS = {
    "V4_BASE": [],
    "V4.2_A": ["recent7_win"],
    "V4.2_B": ["bounce_signal"],
    "V4.2_C": ["recent7_win", "bounce_signal"],
}


TOP_NS = [5, 10, 20, 30]


# ============================================================
# CSV
# ============================================================

def read_csv(path):

    for enc in (
        "utf-8-sig",
        "utf-8",
        "cp932",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=enc
            )
        except Exception:
            pass

    raise RuntimeError(
        "CSV read failed: " + str(path)
    )


def find_column(df, candidates):

    for col in candidates:
        if col in df.columns:
            return col

    return None


def load_data():

    frames = []

    for path in (
        CSV1,
        CSV2,
    ):

        if path.exists():

            print(
                "Loading:",
                path
            )

            frames.append(
                read_csv(path)
            )

    if not frames:

        raise FileNotFoundError(
            "Input CSV not found."
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    date_col = find_column(
        df,
        [
            "date",
            "日付",
            "譌･莉・",
        ]
    )

    no_col = find_column(
        df,
        [
            "machine_no",
            "台番号",
            "蜿ｰ逡ｪ蜿ｷ",
        ]
    )

    name_col = find_column(
        df,
        [
            "machine_name",
            "機種名",
            "讖溽ｨｮ蜷・",
        ]
    )

    diff_col = find_column(
        df,
        [
            "diff",
            "差枚",
            "蟾ｮ譫・",
        ]
    )

    if not all([
        date_col,
        no_col,
        name_col,
        diff_col,
    ]):

        raise ValueError(
            "Required columns not found.\n"
            f"date={date_col}\n"
            f"machine_no={no_col}\n"
            f"machine_name={name_col}\n"
            f"diff={diff_col}"
        )

    df = df.rename(
        columns={
            date_col: "date",
            no_col: "machine_no",
            name_col: "machine_name",
            diff_col: "diff",
        }
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["machine_no"] = pd.to_numeric(
        df["machine_no"],
        errors="coerce"
    )

    df["diff"] = (
        df["diff"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.replace(
            "+",
            "",
            regex=False
        )
        .str.strip()
    )

    df["diff"] = pd.to_numeric(
        df["diff"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "machine_no",
            "diff",
        ]
    ).copy()

    df["machine_no"] = (
        df["machine_no"]
        .astype(int)
    )

    df["machine_name"] = (
        df["machine_name"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["date"] >= START)
        & (df["date"] <= TEST_END)
    ].copy()

    df = df.sort_values(
        [
            "date",
            "machine_no",
        ]
    )

    df = df.drop_duplicates(
        [
            "date",
            "machine_no",
        ],
        keep="last"
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

    return df


# ============================================================
# Feature construction
# ============================================================

def build_features(
    df,
    target_date
):

    hist = df[
        df["date"] < target_date
    ].copy()

    actual = df[
        df["date"] == target_date
    ][
        [
            "machine_no",
            "machine_name",
            "diff",
        ]
    ].copy()

    if hist.empty or actual.empty:
        return pd.DataFrame(), 0

    latest_date = hist["date"].max()

    latest_day = (
        hist[
            hist["date"] == latest_date
        ]
        .set_index("machine_no")
    )

    type_stats = (
        hist.groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    rows = []

    no_history_count = 0

    for no, actual_row in actual.iterrows():

        machine_no = int(
            actual_row["machine_no"]
        )

        machine_name_actual = str(
            actual_row["machine_name"]
        ).strip()

        m = hist[
            hist["machine_no"] == machine_no
        ].sort_values("date").copy()

        # 同一台番の履歴がなければ除外
        if m.empty:
            no_history_count += 1
            continue

        # 現在の機種名と過去履歴の最新機種名を比較
        latest_machine_name = str(
            m.iloc[-1]["machine_name"]
        ).strip()

        # 機種変更があった台は除外
        if latest_machine_name != machine_name_actual:
            no_history_count += 1
            continue

        avg31 = float(
            m["diff"].mean()
        )

        recent7 = m.tail(7)

        recent7_avg = float(
            recent7["diff"].mean()
        )

        recent7_win = float(
            recent7["win"].mean()
        )

        last_diff = float(
            m.iloc[-1]["diff"]
        )

        if len(m) >= 2:

            prev_diff = float(
                m.iloc[-2]["diff"]
            )

        else:

            prev_diff = last_diff

        prev_change = (
            last_diff - prev_diff
        )

        target_weekday = (
            target_date.dayofweek
        )

        wd = m[
            m["date"].dt.dayofweek
            == target_weekday
        ]

        weekday_n = len(wd)

        if weekday_n:

            weekday_avg_raw = float(
                wd["diff"].mean()
            )

        else:

            weekday_avg_raw = avg31

        prior_n = 15.0

        wd_weight = (
            weekday_n
            / (
                weekday_n + prior_n
            )
        )

        weekday_avg = (
            weekday_avg_raw * wd_weight
            + avg31 * (1.0 - wd_weight)
        )

        plus1000_rate = float(
            m["plus1000"].mean()
        )

        plus2000_rate = float(
            m["plus2000"].mean()
        )

        type_avg = float(
            type_stats.get(
                latest_machine_name,
                avg31
            )
        )

        neighbor_values = []

        for n2 in (
            machine_no - 1,
            machine_no + 1,
        ):

            if n2 in latest_day.index:

                neighbor_values.append(
                    float(
                        latest_day.loc[
                            n2,
                            "diff"
                        ]
                    )
                )

        if neighbor_values:

            neighbor_avg = float(
                np.mean(
                    neighbor_values
                )
            )

        else:

            neighbor_avg = 0.0

        if last_diff <= -1000:

            bounce_signal = 1.0

        elif last_diff <= -500:

            bounce_signal = 0.5

        elif last_diff >= 1000:

            bounce_signal = -0.25

        else:

            bounce_signal = 0.0

        rows.append({

            "machine_no": machine_no,

            "machine_name":
                latest_machine_name,

            "avg31":
                avg31,

            "recent7_avg":
                recent7_avg,

            "recent7_win":
                recent7_win,

            "last_diff":
                last_diff,

            "prev_change":
                prev_change,

            "weekday_avg":
                weekday_avg,

            "type_avg":
                type_avg,

            "plus1000_rate":
                plus1000_rate,

            "plus2000_rate":
                plus2000_rate,

            "neighbor_avg":
                neighbor_avg,

            "bounce_signal":
                bounce_signal,

            "actual_diff":
                float(
                    actual_row["diff"]
                ),
        })

    if not rows:

        return pd.DataFrame(), no_history_count

    return (
        pd.DataFrame(rows),
        no_history_count
    )


# ============================================================
# Score
# ============================================================

def zscore(series):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)

    mean = float(
        s.mean()
    )

    std = float(
        s.std(ddof=0)
    )

    if std == 0 or np.isnan(std):

        return pd.Series(
            0.0,
            index=s.index
        )

    return (
        (s - mean) / std
    )


def score_features(
    feat,
    excluded
):

    score = pd.Series(
        0.0,
        index=feat.index
    )

    active = [
        f
        for f in FACTORS
        if f not in excluded
    ]

    active_weight_sum = sum(
        BASE_WEIGHTS[f]
        for f in active
    )

    if active_weight_sum <= 0:
        raise ValueError(
            "Active weight sum is zero."
        )

    for factor in active:

        w = (
            BASE_WEIGHTS[factor]
            / active_weight_sum
        )

        score += (
            w
            * zscore(
                feat[factor]
            )
        )

    return score


# ============================================================
# Daily evaluation
# ============================================================

def evaluate_day(
    df,
    target_date,
    model_name,
    excluded,
    top_n
):

    feat, excluded_count = build_features(
        df,
        target_date
    )

    if feat.empty:

        return None

    feat = feat.copy()

    feat["score"] = score_features(
        feat,
        excluded
    )

    feat = feat.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)

    top = feat.head(top_n)

    if top.empty:
        return None

    diffs = top["actual_diff"]

    return {
        "date":
            target_date.strftime(
                "%Y-%m-%d"
            ),

        "model":
            model_name,

        "top_n":
            top_n,

        "machines":
            len(feat),

        "excluded_machines":
            excluded_count,

        "avg_diff":
            float(diffs.mean()),

        "median_diff":
            float(diffs.median()),

        "win_rate":
            float(
                (diffs > 0).mean()
                * 100
            ),

        "plus1000_rate":
            float(
                (diffs >= 1000).mean()
                * 100
            ),

        "plus2000_rate":
            float(
                (diffs >= 2000).mean()
                * 100
            ),

        "positive":
            int(
                diffs.mean() > 0
            ),

        "total_diff":
            float(
                diffs.sum()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.4.2 Rolling Walk-Forward Validation"
    )
    print("=" * 70)

    print()
    print("MODELS")
    print("-" * 70)

    for name, excluded in MODELS.items():

        print(
            f"{name:12s} "
            f"exclude="
            f"{','.join(excluded) if excluded else 'NONE'}"
        )

    print()
    print(
        f"TEST PERIOD = "
        f"{TEST_START.date()} "
        f"to "
        f"{TEST_END.date()}"
    )

    print(
        f"BLOCK SIZE = "
        f"{BLOCK_SIZE} days"
    )

    print()

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    all_dates = sorted(
        df[
            (df["date"] >= TEST_START)
            & (df["date"] <= TEST_END)
        ]["date"].unique()
    )

    if not all_dates:

        raise RuntimeError(
            "No test dates found."
        )

    # --------------------------------------------------------
    # Build blocks
    # --------------------------------------------------------

    blocks = []

    for i in range(
        0,
        len(all_dates),
        BLOCK_SIZE
    ):

        block_dates = all_dates[
            i:i + BLOCK_SIZE
        ]

        blocks.append(
            block_dates
        )

    print()
    print(
        f"walk-forward blocks = "
        f"{len(blocks)}"
    )

    for idx, block in enumerate(
        blocks,
        start=1
    ):

        print(
            f"BLOCK {idx}: "
            f"{pd.Timestamp(block[0]).date()} "
            f"to "
            f"{pd.Timestamp(block[-1]).date()}"
        )

    # --------------------------------------------------------
    # Daily evaluation
    # --------------------------------------------------------

    daily_rows = []

    print()
    print("=" * 70)
    print("DAILY WALK-FORWARD EVALUATION")
    print("=" * 70)

    for block_no, block in enumerate(
        blocks,
        start=1
    ):

        block_start = pd.Timestamp(
            block[0]
        )

        block_end = pd.Timestamp(
            block[-1]
        )

        print()
        print(
            f"BLOCK {block_no}: "
            f"{block_start.date()} "
            f"to "
            f"{block_end.date()}"
        )

        for target in block:

            target_date = pd.Timestamp(
                target
            )

            print(
                f"  {target_date.date()}",
                end=" "
            )

            for model_name, excluded in MODELS.items():

                for top_n in TOP_NS:

                    result = evaluate_day(
                        df,
                        target_date,
                        model_name,
                        excluded,
                        top_n
                    )

                    if result is None:
                        continue

                    result["block"] = (
                        block_no
                    )

                    result["block_start"] = (
                        block_start.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    result["block_end"] = (
                        block_end.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    daily_rows.append(
                        result
                    )

            print("done")

    daily = pd.DataFrame(
        daily_rows
    )

    if daily.empty:

        raise RuntimeError(
            "No daily results generated."
        )

    # --------------------------------------------------------
    # Block summary
    # --------------------------------------------------------

    block_summary = (
        daily
        .groupby(
            [
                "block",
                "block_start",
                "block_end",
                "model",
                "top_n",
            ],
            as_index=False
        )
        .agg(
            days=("date", "count"),
            avg_diff=("avg_diff", "mean"),
            median_daily_avg=(
                "avg_diff",
                "median"
            ),
            win_rate=("win_rate", "mean"),
            plus1000_rate=(
                "plus1000_rate",
                "mean"
            ),
            plus2000_rate=(
                "plus2000_rate",
                "mean"
            ),
            positive_days=(
                "positive",
                "sum"
            ),
            total_diff=(
                "total_diff",
                "sum"
            ),
        )
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    overall = (
        daily
        .groupby(
            [
                "model",
                "top_n",
            ],
            as_index=False
        )
        .agg(
            days=("date", "count"),
            avg_diff=("avg_diff", "mean"),
            median_daily_avg=(
                "avg_diff",
                "median"
            ),
            win_rate=("win_rate", "mean"),
            plus1000_rate=(
                "plus1000_rate",
                "mean"
            ),
            plus2000_rate=(
                "plus2000_rate",
                "mean"
            ),
            positive_days=(
                "positive",
                "sum"
            ),
            total_diff=(
                "total_diff",
                "sum"
            ),
        )
    )

    # --------------------------------------------------------
    # V4.2 comparison
    # --------------------------------------------------------

    comparison_rows = []

    for top_n in TOP_NS:

        base = overall[
            (overall["model"] == "V4_BASE")
            & (overall["top_n"] == top_n)
        ]

        candidate = overall[
            (overall["model"] == "V4.2_C")
            & (overall["top_n"] == top_n)
        ]

        if base.empty or candidate.empty:
            continue

        base_row = base.iloc[0]
        cand_row = candidate.iloc[0]

        comparison_rows.append({

            "top_n":
                top_n,

            "v4_mean":
                float(
                    base_row["avg_diff"]
                ),

            "v42c_mean":
                float(
                    cand_row["avg_diff"]
                ),

            "improvement":
                float(
                    cand_row["avg_diff"]
                    - base_row["avg_diff"]
                ),

            "v4_total_diff":
                float(
                    base_row["total_diff"]
                ),

            "v42c_total_diff":
                float(
                    cand_row["total_diff"]
                ),

            "total_improvement":
                float(
                    cand_row["total_diff"]
                    - base_row["total_diff"]
                ),
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    # --------------------------------------------------------
    # Block-level V4.2_C vs BASE
    # --------------------------------------------------------

    block_compare_rows = []

    for block_no in sorted(
        block_summary["block"].unique()
    ):

        for top_n in TOP_NS:

            b = block_summary[
                (block_summary["block"] == block_no)
                & (block_summary["model"] == "V4_BASE")
                & (block_summary["top_n"] == top_n)
            ]

            c = block_summary[
                (block_summary["block"] == block_no)
                & (block_summary["model"] == "V4.2_C")
                & (block_summary["top_n"] == top_n)
            ]

            if b.empty or c.empty:
                continue

            br = b.iloc[0]
            cr = c.iloc[0]

            block_compare_rows.append({

                "block":
                    block_no,

                "block_start":
                    br["block_start"],

                "block_end":
                    br["block_end"],

                "top_n":
                    top_n,

                "v4_avg_diff":
                    float(
                        br["avg_diff"]
                    ),

                "v42c_avg_diff":
                    float(
                        cr["avg_diff"]
                    ),

                "improvement":
                    float(
                        cr["avg_diff"]
                        - br["avg_diff"]
                    ),

                "v4_total_diff":
                    float(
                        br["total_diff"]
                    ),

                "v42c_total_diff":
                    float(
                        cr["total_diff"]
                    ),

                "total_improvement":
                    float(
                        cr["total_diff"]
                        - br["total_diff"]
                    ),
            })

    block_compare = pd.DataFrame(
        block_compare_rows
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    stability_rows = []

    for model_name in MODELS.keys():

        for top_n in TOP_NS:

            x = overall[
                (overall["model"] == model_name)
                & (overall["top_n"] == top_n)
            ]

            if x.empty:
                continue

            blocks_for_model = block_summary[
                (block_summary["model"] == model_name)
                & (block_summary["top_n"] == top_n)
            ]

            stability_rows.append({

                "model":
                    model_name,

                "top_n":
                    top_n,

                "mean_block_avg":
                    float(
                        blocks_for_model[
                            "avg_diff"
                        ].mean()
                    ),

                "median_block_avg":
                    float(
                        blocks_for_model[
                            "avg_diff"
                        ].median()
                    ),

                "min_block_avg":
                    float(
                        blocks_for_model[
                            "avg_diff"
                        ].min()
                    ),

                "max_block_avg":
                    float(
                        blocks_for_model[
                            "avg_diff"
                        ].max()
                    ),

                "positive_blocks":
                    int(
                        (
                            blocks_for_model[
                                "avg_diff"
                            ] > 0
                        ).sum()
                    ),

                "negative_blocks":
                    int(
                        (
                            blocks_for_model[
                                "avg_diff"
                            ] < 0
                        ).sum()
                    ),

                "block_count":
                    int(
                        len(blocks_for_model)
                    ),

                "overall_avg":
                    float(
                        x.iloc[0]["avg_diff"]
                    ),

                "overall_total_diff":
                    float(
                        x.iloc[0]["total_diff"]
                    ),
            })

    stability = pd.DataFrame(
        stability_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    daily_path = (
        OUT_DIR
        / "29_Ver4_2_walk_forward_daily.csv"
    )

    block_path = (
        OUT_DIR
        / "29_Ver4_2_walk_forward_blocks.csv"
    )

    overall_path = (
        OUT_DIR
        / "29_Ver4_2_walk_forward_summary.csv"
    )

    compare_path = (
        OUT_DIR
        / "29_Ver4_2_walk_forward_compare.csv"
    )

    stability_path = (
        OUT_DIR
        / "29_Ver4_2_walk_forward_stability.csv"
    )

    daily.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig"
    )

    block_summary.to_csv(
        block_path,
        index=False,
        encoding="utf-8-sig"
    )

    overall.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig"
    )

    comparison.to_csv(
        compare_path,
        index=False,
        encoding="utf-8-sig"
    )

    stability.to_csv(
        stability_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VER.4.2 WALK-FORWARD RESULT"
    )
    print("=" * 70)

    print()

    print(
        overall.sort_values(
            [
                "top_n",
                "avg_diff",
            ],
            ascending=[
                True,
                False,
            ]
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "V4 BASE vs V4.2_C"
    )
    print("=" * 70)

    print()

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "BLOCK-LEVEL COMPARISON"
    )
    print("=" * 70)

    print()

    print(
        block_compare.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "STABILITY"
    )
    print("=" * 70)

    print()

    print(
        stability.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(daily_path)
    print(block_path)
    print(overall_path)
    print(compare_path)
    print(stability_path)

    print()
    print(
        "Ver.4.2 Rolling Walk-Forward "
        "validation complete."
    )


if __name__ == "__main__":
    main()