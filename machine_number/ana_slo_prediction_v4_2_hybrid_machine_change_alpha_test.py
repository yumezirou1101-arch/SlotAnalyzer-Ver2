from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
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

INPUT_CSV = (
    DATA_DIR
    / "ana_slo_20260711_20260818.csv"
)

OUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "50_Ver4_2_hybrid_machine_change_alpha_test"
)

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-18")

# Old-machine-history inheritance ratio.
# 0.00 = clean/current-machine history only
# 1.00 = full same-slot history inheritance
ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

TOP_NS = [
    5,
    10,
    20,
    30,
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

V42_A = V4_WEIGHTS.copy()
V42_A.pop("recent7_win")

V42_B = V4_WEIGHTS.copy()
V42_B.pop("bounce_signal")

V42_C = V4_WEIGHTS.copy()
V42_C.pop("recent7_win")
V42_C.pop("bounce_signal")


def normalize_weights(weights):

    total = sum(weights.values())

    if total <= 0:
        raise ValueError(
            "Weight sum must be positive."
        )

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
    (
        "ROLL6",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-08-10"),
        pd.Timestamp("2026-08-11"),
        pd.Timestamp("2026-08-14"),
    ),
    (
        "ROLL7",
        pd.Timestamp("2026-07-11"),
        pd.Timestamp("2026-08-14"),
        pd.Timestamp("2026-08-15"),
        pd.Timestamp("2026-08-18"),
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

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = read_csv(
        INPUT_CSV
    )

    def find(cols):

        for col in cols:

            if col in df.columns:
                return col

        return None

    date_col = find([
        "date",
        "\u65e5\u4ed8",
    ])

    no_col = find([
        "machine_no",
        "\u53f0\u756a\u53f7",
    ])

    name_col = find([
        "machine_name",
        "\u6a5f\u7a2e\u540d",
    ])

    diff_col = find([
        "diff",
        "\u5dee\u679a",
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
            "machine_name",
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

    df = (
        df.sort_values(
            [
                "date",
                "machine_no",
            ]
        )
        .drop_duplicates(
            [
                "date",
                "machine_no",
            ],
            keep="last"
        )
        .reset_index(drop=True)
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
# Feature helpers
# ============================================================

BLEND_FACTORS = [
    "avg31",
    "recent7_avg",
    "recent7_win",
    "last_diff",
    "prev_change",
    "weekday_avg",
    "plus1000_rate",
    "plus2000_rate",
    "bounce_signal",
]


def machine_stats(
    m,
    target_weekday,
    defaults,
):

    if m is None or m.empty:

        return defaults.copy()

    m = m.sort_values(
        "date"
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

        prev_change = (
            last_diff
            - prev_diff
        )

    else:

        prev_change = 0.0

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

    if last_diff <= -1000:

        bounce_signal = 1.0

    elif last_diff <= -500:

        bounce_signal = 0.5

    elif last_diff >= 1000:

        bounce_signal = -0.25

    else:

        bounce_signal = 0.0

    return {
        "avg31": avg31,
        "recent7_avg": recent7_avg,
        "recent7_win": recent7_win,
        "last_diff": last_diff,
        "prev_change": prev_change,
        "weekday_avg": weekday_avg,
        "plus1000_rate": plus1000_rate,
        "plus2000_rate": plus2000_rate,
        "bounce_signal": bounce_signal,
    }


def blend_stats(
    clean_stats,
    legacy_stats,
    alpha,
):

    out = {}

    for factor in BLEND_FACTORS:

        clean = float(
            clean_stats[factor]
        )

        legacy = float(
            legacy_stats[factor]
        )

        out[factor] = (
            (1.0 - alpha) * clean
            + alpha * legacy
        )

    return out


# ============================================================
# Hybrid feature construction
# ============================================================

def build_features(
    df,
    target_date,
    alpha,
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
        .sort_values(
            "machine_no"
        )
        .drop_duplicates(
            "machine_no",
            keep="last"
        )
        .set_index(
            "machine_no"
        )
    )

    type_stats = (
        hist.groupby(
            "machine_name"
        )["diff"]
        .mean()
        .to_dict()
    )

    # --------------------------------------------------------
    # Store-level priors
    # --------------------------------------------------------

    store_avg = float(
        hist["diff"].mean()
    )

    store_recent = hist[
        hist["date"]
        >= latest_date - pd.Timedelta(days=6)
    ]

    if not store_recent.empty:

        store_recent_avg = float(
            store_recent["diff"].mean()
        )

        store_recent_win = float(
            store_recent["win"].mean()
        )

    else:

        store_recent_avg = store_avg

        store_recent_win = float(
            hist["win"].mean()
        )

    store_wd = hist[
        hist["date"].dt.dayofweek
        == target_weekday
    ]

    if not store_wd.empty:

        store_weekday_avg = float(
            store_wd["diff"].mean()
        )

    else:

        store_weekday_avg = store_avg

    defaults = {
        "avg31": store_avg,
        "recent7_avg": store_recent_avg,
        "recent7_win": store_recent_win,
        "last_diff": store_recent_avg,
        "prev_change": 0.0,
        "weekday_avg": store_weekday_avg,
        "plus1000_rate": float(
            hist["plus1000"].mean()
        ),
        "plus2000_rate": float(
            hist["plus2000"].mean()
        ),
        "bounce_signal": 0.0,
    }

    hist_by_no = {
        int(no): m.sort_values(
            "date"
        ).copy()
        for no, m in hist.groupby(
            "machine_no"
        )
    }

    rows = []

    # Always start from all target-day machines.
    for row in actual.itertuples(
        index=False
    ):

        no = int(
            row.machine_no
        )

        current_name = str(
            row.machine_name
        ).strip()

        actual_diff = float(
            row.diff
        )

        all_no_hist = hist_by_no.get(
            no
        )

        if (
            all_no_hist is not None
            and not all_no_hist.empty
        ):

            previous_name = str(
                all_no_hist.iloc[-1][
                    "machine_name"
                ]
            ).strip()

            machine_changed = int(
                previous_name
                != current_name
            )

            current_hist = all_no_hist[
                all_no_hist[
                    "machine_name"
                ]
                .astype(str)
                .str.strip()
                == current_name
            ].copy()

        else:

            previous_name = ""
            machine_changed = 1
            all_no_hist = pd.DataFrame(
                columns=hist.columns
            )
            current_hist = pd.DataFrame(
                columns=hist.columns
            )

        clean_stats = machine_stats(
            current_hist,
            target_weekday,
            defaults,
        )

        legacy_stats = machine_stats(
            all_no_hist,
            target_weekday,
            defaults,
        )

        blended = blend_stats(
            clean_stats,
            legacy_stats,
            alpha,
        )

        type_avg = float(
            type_stats.get(
                current_name,
                store_avg
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

        rows.append({

            "machine_no":
                no,

            "machine_name":
                current_name,

            "previous_machine_name":
                previous_name,

            "machine_changed":
                machine_changed,

            "same_machine_history_n":
                int(len(current_hist)),

            "slot_history_n":
                int(len(all_no_hist)),

            "alpha_old_history":
                float(alpha),

            "avg31":
                blended["avg31"],

            "recent7_avg":
                blended["recent7_avg"],

            "recent7_win":
                blended["recent7_win"],

            "last_diff":
                blended["last_diff"],

            "prev_change":
                blended["prev_change"],

            "weekday_avg":
                blended["weekday_avg"],

            "type_avg":
                type_avg,

            "plus1000_rate":
                blended["plus1000_rate"],

            "plus2000_rate":
                blended["plus2000_rate"],

            "neighbor_avg":
                neighbor_avg,

            "bounce_signal":
                blended["bounce_signal"],

            "diff":
                actual_diff,
        })

    feat = pd.DataFrame(
        rows
    )

    if feat.empty:

        return feat

    feat = (
        feat.sort_values(
            "machine_no"
        )
        .drop_duplicates(
            "machine_no",
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    return feat


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
        [
            "score",
            "machine_no",
        ],
        ascending=[
            False,
            True,
        ]
    )


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

        "changed_candidates":
            int(
                panel[
                    "machine_changed"
                ].sum()
            ),

        "changed_selected":
            int(
                top[
                    "machine_changed"
                ].sum()
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)
    print(
        "Ana-Slo Ver.4.2 Hybrid Machine-Change Alpha OOS Test"
    )
    print("=" * 78)

    print()
    print(
        "ALPHAS:",
        ALPHAS
    )

    print()
    print(
        "IMPORTANT: 102 change observations exist on only 4 dates, "
        "including 87 on 2026-08-03."
    )
    print(
        "Treat the best alpha as exploratory, not production tuning."
    )
    print()

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    print()

    # --------------------------------------------------------
    # Build panels for every alpha/date
    # --------------------------------------------------------

    panels = {}

    for alpha in ALPHAS:

        print("-" * 78)
        print(
            f"Building panels alpha={alpha:.2f}"
        )
        print("-" * 78)

        for target_date in pd.date_range(
            START + pd.Timedelta(days=1),
            END,
        ):

            panel = build_features(
                df,
                target_date,
                alpha,
            )

            if panel.empty:
                continue

            panels[
                (
                    float(alpha),
                    target_date,
                )
            ] = panel

        counts = [
            len(
                panels[
                    (
                        float(alpha),
                        d,
                    )
                ]
            )
            for d in pd.date_range(
                START + pd.Timedelta(days=1),
                END,
            )
            if (
                float(alpha),
                d,
            ) in panels
        ]

        print(
            f"panel days={len(counts)} "
            f"min_machines={min(counts)} "
            f"max_machines={max(counts)}"
        )

    daily_rows = []
    summary_rows = []

    # --------------------------------------------------------
    # Rolling OOS evaluation
    # --------------------------------------------------------

    for (
        split_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in ROLLING_SPLITS:

        test_dates = pd.date_range(
            test_start,
            test_end
        )

        for alpha in ALPHAS:

            for model_name, weights in MODELS.items():

                for top_n in TOP_NS:

                    results = []

                    for target_date in test_dates:

                        panel = panels.get(
                            (
                                float(alpha),
                                target_date,
                            )
                        )

                        if (
                            panel is None
                            or panel.empty
                        ):
                            continue

                        result = evaluate_day(
                            panel,
                            weights,
                            top_n,
                        )

                        if result is None:
                            continue

                        result.update({
                            "split":
                                split_name,

                            "alpha":
                                float(alpha),

                            "model":
                                model_name,

                            "top_n":
                                int(top_n),

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

                    summary_rows.append({
                        "split":
                            split_name,

                        "alpha":
                            float(alpha),

                        "model":
                            model_name,

                        "top_n":
                            int(top_n),

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
                            rdf[
                                "win_rate"
                            ].mean(),

                        "plus1000_rate":
                            rdf[
                                "plus1000_rate"
                            ].mean(),

                        "plus2000_rate":
                            rdf[
                                "plus2000_rate"
                            ].mean(),

                        "positive_days":
                            rdf[
                                "positive"
                            ].mean()
                            * 100,

                        "total_diff":
                            rdf[
                                "total_diff"
                            ].sum(),

                        "changed_selected":
                            rdf[
                                "changed_selected"
                            ].sum(),
                    })

    daily_df = pd.DataFrame(
        daily_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if summary_df.empty:

        raise RuntimeError(
            "No evaluation results."
        )

    # --------------------------------------------------------
    # Overall alpha/model/topN summary
    # --------------------------------------------------------

    overall_df = (
        summary_df
        .groupby(
            [
                "alpha",
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

            "changed_selected":
                "sum",
        })
    )

    # --------------------------------------------------------
    # Split stability
    # --------------------------------------------------------

    stability_df = (
        summary_df
        .groupby(
            [
                "alpha",
                "model",
                "top_n",
            ],
            as_index=False
        )
        .agg(
            test_splits=(
                "split",
                "nunique"
            ),

            mean_avg_diff=(
                "avg_diff",
                "mean"
            ),

            min_split_avg_diff=(
                "avg_diff",
                "min"
            ),

            max_split_avg_diff=(
                "avg_diff",
                "max"
            ),

            positive_split_rate=(
                "total_diff",
                lambda s:
                    float(
                        (s > 0).mean()
                        * 100
                    )
            ),
        )
    )

    # --------------------------------------------------------
    # Key candidates
    # --------------------------------------------------------

    key_df = overall_df[
        (
            (
                overall_df["model"]
                == "V4.2_A"
            )
            & (
                overall_df["top_n"]
                == 10
            )
        )
        |
        (
            (
                overall_df["model"]
                == "V4.2_C"
            )
            & (
                overall_df["top_n"]
                == 5
            )
        )
        |
        (
            (
                overall_df["model"]
                == "V4.2_C"
            )
            & (
                overall_df["top_n"]
                == 10
            )
        )
    ].copy()

    key_df = key_df.sort_values(
        [
            "model",
            "top_n",
            "alpha",
        ]
    )

    print()
    print("=" * 78)
    print(
        "KEY CANDIDATES BY ALPHA"
    )
    print("=" * 78)

    print(
        key_df[
            [
                "alpha",
                "model",
                "top_n",
                "avg_diff",
                "total_diff",
                "win_rate",
                "positive_days",
                "changed_selected",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "BEST ALPHA PER KEY CANDIDATE "
        "(EXPLORATORY ONLY)"
    )
    print("=" * 78)

    best_rows = []

    for (
        model_name,
        top_n,
    ), g in key_df.groupby(
        [
            "model",
            "top_n",
        ]
    ):

        best = g.sort_values(
            [
                "total_diff",
                "avg_diff",
            ],
            ascending=[
                False,
                False,
            ]
        ).iloc[0]

        best_rows.append({
            "model":
                model_name,

            "top_n":
                int(top_n),

            "best_alpha":
                float(
                    best["alpha"]
                ),

            "avg_diff":
                float(
                    best["avg_diff"]
                ),

            "total_diff":
                float(
                    best["total_diff"]
                ),

            "win_rate":
                float(
                    best["win_rate"]
                ),

            "positive_days":
                float(
                    best["positive_days"]
                ),
        })

    best_df = pd.DataFrame(
        best_rows
    )

    print(
        best_df.to_string(
            index=False
        )
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
        / "50_Ver4_2_hybrid_alpha_daily.csv"
    )

    out_summary = (
        OUT_DIR
        / "50_Ver4_2_hybrid_alpha_summary.csv"
    )

    out_overall = (
        OUT_DIR
        / "50_Ver4_2_hybrid_alpha_overall.csv"
    )

    out_stability = (
        OUT_DIR
        / "50_Ver4_2_hybrid_alpha_stability.csv"
    )

    out_key = (
        OUT_DIR
        / "50_Ver4_2_hybrid_alpha_key_candidates.csv"
    )

    out_best = (
        OUT_DIR
        / "50_Ver4_2_hybrid_alpha_best_exploratory.csv"
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

    overall_df.to_csv(
        out_overall,
        index=False,
        encoding="utf-8-sig"
    )

    stability_df.to_csv(
        out_stability,
        index=False,
        encoding="utf-8-sig"
    )

    key_df.to_csv(
        out_key,
        index=False,
        encoding="utf-8-sig"
    )

    best_df.to_csv(
        out_best,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 78)
    print("FILES SAVED")
    print("=" * 78)

    for p in (
        out_daily,
        out_summary,
        out_overall,
        out_stability,
        out_key,
        out_best,
    ):
        print(p)

    print()
    print(
        "Hybrid alpha OOS test complete."
    )

    print(
        "Do NOT adopt the best alpha yet: "
        "machine changes occurred on only 4 dates."
    )


if __name__ == "__main__":
    main()
