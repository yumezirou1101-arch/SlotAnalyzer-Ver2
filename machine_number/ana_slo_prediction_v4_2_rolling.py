from pathlib import Path
import pandas as pd
import numpy as np


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
END = pd.Timestamp("2026-08-10")


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


# ============================================================
# Ver.4 / Ver.4.2 weights
# ============================================================

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


# V4.2_A:
# recent7_win を除外
V42_A = V4_WEIGHTS.copy()
V42_A.pop("recent7_win")


# V4.2_B:
# bounce_signal を除外
V42_B = V4_WEIGHTS.copy()
V42_B.pop("bounce_signal")


# V4.2_C:
# recent7_win + bounce_signal を除外
V42_C = V4_WEIGHTS.copy()
V42_C.pop("recent7_win")
V42_C.pop("bounce_signal")


def normalize_weights(weights):

    total = sum(weights.values())

    if total <= 0:
        raise ValueError("Weight sum must be positive.")

    return {
        k: v / total
        for k, v in weights.items()
    }


MODELS = {
    "V4_BASE": normalize_weights(V4_WEIGHTS),
    "V4.2_A": normalize_weights(V42_A),
    "V4.2_B": normalize_weights(V42_B),
    "V4.2_C": normalize_weights(V42_C),
}


# ============================================================
# Rolling OOS periods
#
# 各TEST期間は、TEST開始日の前日までのデータだけを
# 学習・特徴量生成に使用する。
#
# 今回は固定ウェイトモデルなので、ここでは
# 「未来期間を完全に分離した再現性テスト」として扱う。
# ============================================================

ROLLING_SPLITS = [
    (
        "ROLL1",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-07-20"),
        pd.Timestamp("2026-07-21"),
        pd.Timestamp("2026-07-24"),
    ),
    (
        "ROLL2",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-07-24"),
        pd.Timestamp("2026-07-25"),
        pd.Timestamp("2026-07-28"),
    ),
    (
        "ROLL3",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-07-28"),
        pd.Timestamp("2026-07-29"),
        pd.Timestamp("2026-08-01"),
    ),
    (
        "ROLL4",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-08-01"),
        pd.Timestamp("2026-08-02"),
        pd.Timestamp("2026-08-05"),
    ),
    (
        "ROLL5",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-08-05"),
        pd.Timestamp("2026-08-06"),
        pd.Timestamp("2026-08-10"),
    ),
]


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
        "CSV read failed: "
        + str(path)
    )


def load_data():

    frames = []

    for path in (
        CSV1,
        CSV2
    ):

        if path.exists():

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

    def find(cols):

        for col in cols:

            if col in df.columns:

                return col

        return None

    date_col = find([
        "date",
        "日付",
        "譌･莉・",
    ])

    no_col = find([
        "machine_no",
        "台番号",
        "蜿ｰ逡ｪ蜿ｷ",
    ])

    name_col = find([
        "machine_name",
        "機種名",
        "讖溽ｨｮ蜷・",
    ])

    diff_col = find([
        "diff",
        "差枚",
        "蟾ｮ譫・",
    ])

    if not all([
        date_col,
        no_col,
        name_col,
        diff_col,
    ]):

        raise ValueError(
            "Required columns not found: "
            f"date={date_col}, "
            f"no={no_col}, "
            f"name={name_col}, "
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
        & (df["date"] <= END)
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
    target_date,
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

        return pd.DataFrame()

    target_weekday = (
        target_date.dayofweek
    )

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

    for no, m in hist.groupby(
        "machine_no"
    ):

        m = m.sort_values(
            "date"
        )

        if m.empty:
            continue

        name = str(
            m.iloc[-1]["machine_name"]
        )

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
            last_diff
            - prev_diff
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
                weekday_n
                + prior_n
            )
        )

        weekday_avg = (
            weekday_avg_raw
            * wd_weight
            + avg31
            * (1.0 - wd_weight)
        )

        plus1000_rate = float(
            m["plus1000"].mean()
        )

        plus2000_rate = float(
            m["plus2000"].mean()
        )

        type_avg = float(
            type_stats.get(
                name,
                0.0
            )
        )

        neighbor_values = []

        for n2 in (
            no - 1,
            no + 1
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

            "machine_no":
                int(no),

            "machine_name":
                name,

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
        })

    feat = pd.DataFrame(
        rows
    )

    if feat.empty:

        return feat

    return feat.merge(
        actual,
        on=[
            "machine_no",
            "machine_name",
        ],
        how="inner"
    )


# ============================================================
# Scoring
# ============================================================

def zscore(series):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)

    std = float(
        s.std(ddof=0)
    )

    if (
        std == 0
        or np.isnan(std)
    ):

        return pd.Series(
            0.0,
            index=s.index
        )

    return (
        s - s.mean()
    ) / std


def rank_score(
    df,
    weights,
):

    x = df.copy()

    score = pd.Series(
        0.0,
        index=x.index
    )

    for factor, weight in weights.items():

        if factor not in x.columns:
            continue

        z = zscore(
            x[factor]
        )

        component = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100
        )

        score += (
            component
            * weight
        )

    x["score"] = score

    return x.sort_values(
        "score",
        ascending=False
    )


# ============================================================
# Daily evaluation
# ============================================================

def evaluate_day(
    panel,
    weights,
    top_n,
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

    d = (
        top["diff"]
        .astype(float)
    )

    return {

        "avg_diff":
            float(d.mean()),

        "median_diff":
            float(d.median()),

        "win_rate":
            float(
                (d > 0).mean()
                * 100
            ),

        "plus1000_rate":
            float(
                (d >= 1000).mean()
                * 100
            ),

        "plus2000_rate":
            float(
                (d >= 2000).mean()
                * 100
            ),

        "positive":
            int(
                d.sum() > 0
            ),

        "total_diff":
            float(
                d.sum()
            ),

        "machines":
            int(len(panel)),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.4.2 Rolling Walk-Forward OOS Test"
    )
    print("=" * 70)

    print()

    print("MODELS")
    print("-" * 70)

    for name, weights in MODELS.items():

        print(
            f"{name:<12} "
            f"factors={len(weights):2d} "
            f"weight_sum={sum(weights.values()):.6f}"
        )

    print()

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    print()

    # --------------------------------------------------------
    # Build all daily panels
    # --------------------------------------------------------

    print(
        "Building daily feature panels..."
    )

    panels = {}

    for target_date in pd.date_range(
        START + pd.Timedelta(days=1),
        END,
    ):

        panel = build_features(
            df,
            target_date
        )

        if panel.empty:
            continue

        panels[target_date] = panel

        print(
            f"{target_date.date()} "
            f"machines={len(panel)}"
        )

    print()

    daily_rows = []
    split_rows = []

    # --------------------------------------------------------
    # Rolling evaluation
    # --------------------------------------------------------

    for (
        split_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in ROLLING_SPLITS:

        print("=" * 70)
        print(split_name)
        print("=" * 70)

        print(
            f"TRAIN: "
            f"{train_start.date()} "
            f"to "
            f"{train_end.date()}"
        )

        print(
            f"TEST : "
            f"{test_start.date()} "
            f"to "
            f"{test_end.date()}"
        )

        print()

        test_dates = pd.date_range(
            test_start,
            test_end
        )

        for model_name, weights in MODELS.items():

            print(
                f"Evaluating {model_name}..."
            )

            for top_n in (
                5,
                10,
                20,
                30,
            ):

                results = []

                for target_date in test_dates:

                    panel = panels.get(
                        target_date
                    )

                    if (
                        panel is None
                        or panel.empty
                    ):

                        continue

                    result = evaluate_day(
                        panel,
                        weights,
                        top_n
                    )

                    if result is None:
                        continue

                    result.update({

                        "split":
                            split_name,

                        "model":
                            model_name,

                        "top_n":
                            top_n,

                        "date":
                            target_date,

                        "train_start":
                            train_start,

                        "train_end":
                            train_end,

                        "test_start":
                            test_start,

                        "test_end":
                            test_end,
                    })

                    daily_rows.append(
                        result
                    )

                    results.append(
                        result
                    )

                if not results:
                    continue

                rdf = pd.DataFrame(
                    results
                )

                split_rows.append({

                    "split":
                        split_name,

                    "model":
                        model_name,

                    "top_n":
                        top_n,

                    "days":
                        len(rdf),

                    "avg_diff":
                        rdf[
                            "avg_diff"
                        ].mean(),

                    "median_daily_avg":
                        rdf[
                            "avg_diff"
                        ].median(),

                    "win_rate":
                        (
                            rdf["win_rate"]
                            .mean()
                        ),

                    "plus1000_rate":
                        (
                            rdf[
                                "plus1000_rate"
                            ].mean()
                        ),

                    "plus2000_rate":
                        (
                            rdf[
                                "plus2000_rate"
                            ].mean()
                        ),

                    "positive_days":
                        (
                            rdf[
                                "positive"
                            ].mean()
                            * 100
                        ),

                    "total_diff":
                        rdf[
                            "total_diff"
                        ].sum(),
                })

    daily_df = pd.DataFrame(
        daily_rows
    )

    summary_df = pd.DataFrame(
        split_rows
    )

    if summary_df.empty:

        raise RuntimeError(
            "No rolling evaluation results."
        )

    # --------------------------------------------------------
    # Compare models against V4
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ROLLING TEST RESULT"
    )
    print("=" * 70)

    top10 = summary_df[
        summary_df["top_n"] == 10
    ].copy()

    base = top10[
        top10["model"] == "V4_BASE"
    ][
        [
            "split",
            "avg_diff",
            "total_diff",
        ]
    ].rename(
        columns={
            "avg_diff":
                "v4_avg_diff",

            "total_diff":
                "v4_total_diff",
        }
    )

    top10 = top10.merge(
        base,
        on="split",
        how="left"
    )

    top10[
        "avg_diff_change_vs_v4"
    ] = (
        top10["avg_diff"]
        - top10["v4_avg_diff"]
    )

    top10[
        "total_diff_change_vs_v4"
    ] = (
        top10["total_diff"]
        - top10["v4_total_diff"]
    )

    print(
        top10[
            [
                "split",
                "model",
                "avg_diff",
                "total_diff",
                "win_rate",
                "positive_days",
                "avg_diff_change_vs_v4",
                "total_diff_change_vs_v4",
            ]
        ].sort_values(
            [
                "split",
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

    # --------------------------------------------------------
    # Overall model stability ranking
    # --------------------------------------------------------

    ranking_rows = []

    for model_name in MODELS:

        m = top10[
            top10["model"]
            == model_name
        ].copy()

        if m.empty:
            continue

        better_count = int(
            (
                m[
                    "avg_diff_change_vs_v4"
                ] > 0
            ).sum()
        )

        ranking_rows.append({

            "model":
                model_name,

            "test_splits":
                len(m),

            "splits_better_than_v4":
                better_count,

            "mean_avg_diff":
                m[
                    "avg_diff"
                ].mean(),

            "mean_change_vs_v4":
                m[
                    "avg_diff_change_vs_v4"
                ].mean(),

            "total_diff_all_tests":
                m[
                    "total_diff"
                ].sum(),

            "total_change_vs_v4":
                m[
                    "total_diff_change_vs_v4"
                ].sum(),

            "mean_win_rate":
                m[
                    "win_rate"
                ].mean(),

            "mean_positive_days":
                m[
                    "positive_days"
                ].mean(),

            "min_avg_diff":
                m[
                    "avg_diff"
                ].min(),

            "max_avg_diff":
                m[
                    "avg_diff"
                ].max(),
        })

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    ranking_df = ranking_df.sort_values(
        [
            "splits_better_than_v4",
            "mean_avg_diff",
        ],
        ascending=[
            False,
            False,
        ]
    )

    print()
    print("=" * 70)
    print(
        "ROLLING STABILITY RANKING"
    )
    print("=" * 70)

    print(
        ranking_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Model by top_n comparison
    # --------------------------------------------------------

    model_topn = (
        summary_df
        .groupby(
            [
                "model",
                "top_n",
            ],
            as_index=False
        )
        .agg({

            "avg_diff":
                "mean",

            "median_daily_avg":
                "mean",

            "win_rate":
                "mean",

            "plus1000_rate":
                "mean",

            "plus2000_rate":
                "mean",

            "positive_days":
                "mean",

            "total_diff":
                "sum",
        })
    )

    print()
    print("=" * 70)
    print(
        "MODEL / TOP-N OVERALL"
    )
    print("=" * 70)

    print(
        model_topn.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Diagnostic
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "DIAGNOSTIC"
    )
    print("=" * 70)

    for model_name in (
        "V4_BASE",
        "V4.2_A",
        "V4.2_B",
        "V4.2_C",
    ):

        m = ranking_df[
            ranking_df["model"]
            == model_name
        ]

        if m.empty:
            continue

        row = m.iloc[0]

        print(
            f"{model_name:<12} "
            f"mean={row['mean_avg_diff']:+.2f} "
            f"vs_v4={row['mean_change_vs_v4']:+.2f} "
            f"better="
            f"{int(row['splits_better_than_v4'])}/"
            f"{int(row['test_splits'])} "
            f"min={row['min_avg_diff']:+.2f} "
            f"max={row['max_avg_diff']:+.2f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    out_daily = (
        OUT_DIR
        / "24_Ver4_2_rolling_daily.csv"
    )

    out_summary = (
        OUT_DIR
        / "24_Ver4_2_rolling_summary.csv"
    )

    out_ranking = (
        OUT_DIR
        / "24_Ver4_2_rolling_ranking.csv"
    )

    out_top10 = (
        OUT_DIR
        / "24_Ver4_2_rolling_top10_compare.csv"
    )

    out_topn = (
        OUT_DIR
        / "24_Ver4_2_rolling_model_topn.csv"
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

    ranking_df.to_csv(
        out_ranking,
        index=False,
        encoding="utf-8-sig"
    )

    top10.to_csv(
        out_top10,
        index=False,
        encoding="utf-8-sig"
    )

    model_topn.to_csv(
        out_topn,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print(
        "FILES SAVED"
    )
    print("=" * 70)

    print(out_daily)
    print(out_summary)
    print(out_ranking)
    print(out_top10)
    print(out_topn)

    print()
    print(
        "Ver.4.2 rolling OOS test complete."
    )


if __name__ == "__main__":
    main()