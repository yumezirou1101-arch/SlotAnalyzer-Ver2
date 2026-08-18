from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Ana-Slo Ver.4.1
# Fallback Penalty Comparison Test
#
# 目的:
# Ver.4固定ウェイトを変更せず、
# ・機種変更台へのペナルティ
# ・Global Fallbackへのペナルティ
# のみを変更してOOS比較する。
#
# 比較モデル:
# BASE
# CHANGE_5
# CHANGE_10
# GLOBAL_10
# GLOBAL_20
# GLOBAL_EXCLUDE
#
# OOS:
# 2026-07-26 ～ 2026-08-10
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

OUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CSV1 = DATA_DIR / "ana_slo_20260711.csv"
CSV2 = DATA_DIR / "ana_slo_20260712_20260810.csv"


START = pd.Timestamp("2026-07-11")
TEST_START = pd.Timestamp("2026-07-26")
TEST_END = pd.Timestamp("2026-08-10")


# ============================================================
# Ver.4 FIXED WEIGHTS
# ============================================================

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


# ============================================================
# MODEL SETTINGS
# ============================================================

MODELS = {
    "BASE": {
        "change_penalty": 1.00,
        "global_penalty": 1.00,
        "exclude_global": False,
    },

    "CHANGE_5": {
        "change_penalty": 0.95,
        "global_penalty": 1.00,
        "exclude_global": False,
    },

    "CHANGE_10": {
        "change_penalty": 0.90,
        "global_penalty": 1.00,
        "exclude_global": False,
    },

    "GLOBAL_10": {
        "change_penalty": 1.00,
        "global_penalty": 0.90,
        "exclude_global": False,
    },

    "GLOBAL_20": {
        "change_penalty": 1.00,
        "global_penalty": 0.80,
        "exclude_global": False,
    },

    "GLOBAL_EXCLUDE": {
        "change_penalty": 1.00,
        "global_penalty": 1.00,
        "exclude_global": True,
    },
}


# ============================================================
# CSV READ
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    frames = []

    for path in (
        CSV1,
        CSV2,
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

    # --------------------------------------------------------
    # Column detection
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Rename
    # --------------------------------------------------------

    df = df.rename(
        columns={
            date_col: "date",
            no_col: "machine_no",
            name_col: "machine_name",
            diff_col: "diff",
        }
    )


    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Date filter
    # --------------------------------------------------------

    df = df[
        (df["date"] >= START)
        & (df["date"] <= TEST_END)
    ].copy()


    # --------------------------------------------------------
    # Sort / duplicate removal
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Target flags
    # --------------------------------------------------------

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
# MACHINE CHANGE / FEATURE SOURCE
# ============================================================

def get_feature_history(
    df,
    machine_no,
    machine_name,
    target_date,
):

    # --------------------------------------------------------
    # Same machine history
    # --------------------------------------------------------

    machine_hist = df[
        (df["machine_no"] == machine_no)
        & (df["date"] < target_date)
    ].copy()


    if not machine_hist.empty:

        latest_machine_name = str(
            machine_hist.sort_values(
                "date"
            ).iloc[-1]["machine_name"]
        )

        if latest_machine_name == machine_name:

            return (
                machine_hist,
                "same_machine",
                False,
            )


    # --------------------------------------------------------
    # Machine changed
    # --------------------------------------------------------

    type_hist = df[
        (df["machine_name"] == machine_name)
        & (df["date"] < target_date)
    ].copy()


    if not type_hist.empty:

        return (
            type_hist,
            "type_fallback",
            True,
        )


    # --------------------------------------------------------
    # Global fallback
    # --------------------------------------------------------

    global_hist = df[
        df["date"] < target_date
    ].copy()


    return (
        global_hist,
        "global_fallback",
        True,
    )


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    df,
    target_date,
):

    actual = df[
        df["date"] == target_date
    ][
        [
            "machine_no",
            "machine_name",
            "diff",
        ]
    ].copy()


    if actual.empty:

        return (
            pd.DataFrame(),
            {
                "same_machine": 0,
                "type_fallback": 0,
                "global_fallback": 0,
                "machine_change": 0,
            }
        )


    target_weekday = (
        target_date.dayofweek
    )


    # --------------------------------------------------------
    # Latest day for neighbor information
    # --------------------------------------------------------

    hist_all = df[
        df["date"] < target_date
    ].copy()


    if hist_all.empty:

        return (
            pd.DataFrame(),
            {
                "same_machine": 0,
                "type_fallback": 0,
                "global_fallback": 0,
                "machine_change": 0,
            }
        )


    latest_date = hist_all["date"].max()


    latest_day = (
        hist_all[
            hist_all["date"] == latest_date
        ]
        .set_index("machine_no")
    )


    # --------------------------------------------------------
    # Global type statistics
    # --------------------------------------------------------

    type_stats = (
        hist_all
        .groupby("machine_name")["diff"]
        .mean()
        .to_dict()
    )


    rows = []


    diagnostics = {
        "same_machine": 0,
        "type_fallback": 0,
        "global_fallback": 0,
        "machine_change": 0,
    }


    # --------------------------------------------------------
    # Machine loop
    # --------------------------------------------------------

    for _, actual_row in actual.iterrows():

        no = int(
            actual_row["machine_no"]
        )

        current_name = str(
            actual_row["machine_name"]
        )


        hist, source, changed = (
            get_feature_history(
                df,
                no,
                current_name,
                target_date,
            )
        )


        diagnostics[source] += 1

        if changed:

            diagnostics["machine_change"] += 1


        if hist.empty:

            continue


        hist = hist.sort_values(
            "date"
        )


        diffs = (
            hist["diff"]
            .astype(float)
            .to_numpy()
        )


        avg31 = float(
            hist["diff"].mean()
        )


        recent7 = hist.tail(7)


        recent7_avg = float(
            recent7["diff"].mean()
        )


        recent7_win = float(
            recent7["win"].mean()
        )


        last_diff = float(
            diffs[-1]
        )


        if len(diffs) >= 2:

            prev_diff = float(
                diffs[-2]
            )

        else:

            prev_diff = last_diff


        prev_change = (
            last_diff
            - prev_diff
        )


        # ----------------------------------------------------
        # Weekday
        # ----------------------------------------------------

        wd = hist[
            hist["date"].dt.dayofweek
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


        # ----------------------------------------------------
        # Rates
        # ----------------------------------------------------

        plus1000_rate = float(
            hist["plus1000"].mean()
        )


        plus2000_rate = float(
            hist["plus2000"].mean()
        )


        # ----------------------------------------------------
        # Type average
        # ----------------------------------------------------

        type_avg = float(
            type_stats.get(
                current_name,
                0.0
            )
        )


        # ----------------------------------------------------
        # Neighbor
        # ----------------------------------------------------

        neighbor_values = []


        for n2 in (
            no - 1,
            no + 1,
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


        # ----------------------------------------------------
        # Bounce signal
        # ----------------------------------------------------

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
                no,

            "machine_name":
                current_name,

            "feature_source":
                source,

            "machine_change":
                int(changed),

            "history_days":
                len(hist),

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


    return (
        pd.DataFrame(rows),
        diagnostics,
    )


# ============================================================
# Z SCORE
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
# SCORE
# ============================================================

def calculate_score(
    panel,
):

    x = panel.copy()


    score = pd.Series(
        0.0,
        index=x.index
    )


    for factor in FACTORS:

        z = zscore(
            x[factor]
        )


        normalized = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100
        )


        score += (
            normalized
            * V4_WEIGHTS[factor]
        )


    x["base_score"] = score


    return x


# ============================================================
# APPLY MODEL PENALTY
# ============================================================

def apply_model(
    panel,
    model_name,
):

    cfg = MODELS[
        model_name
    ]


    x = panel.copy()


    # --------------------------------------------------------
    # Global fallback exclusion
    # --------------------------------------------------------

    if cfg["exclude_global"]:

        x = x[
            x["feature_source"]
            != "global_fallback"
        ].copy()


    if x.empty:

        return x


    x["score"] = (
        x["base_score"]
    )


    # --------------------------------------------------------
    # Machine change penalty
    # --------------------------------------------------------

    if cfg["change_penalty"] != 1.0:

        x.loc[
            x["machine_change"] == 1,
            "score"
        ] *= cfg[
            "change_penalty"
        ]


    # --------------------------------------------------------
    # Global fallback penalty
    # --------------------------------------------------------

    if cfg["global_penalty"] != 1.0:

        x.loc[
            x["feature_source"]
            == "global_fallback",
            "score"
        ] *= cfg[
            "global_penalty"
        ]


    return x.sort_values(
        "score",
        ascending=False
    )


# ============================================================
# DAILY EVALUATION
# ============================================================

def evaluate_daily(
    ranked,
    top_n,
):

    if ranked.empty:

        return None


    top = ranked.head(
        min(
            top_n,
            len(ranked)
        )
    )


    d = (
        top["actual_diff"]
        .astype(float)
    )


    return {

        "avg_diff":
            float(
                d.mean()
            ),

        "median_diff":
            float(
                d.median()
            ),

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

        "total_diff":
            float(
                d.sum()
            ),

        "positive":
            int(
                d.sum() > 0
            ),

        "machine_change_count":
            int(
                (
                    top["machine_change"]
                    == 1
                ).sum()
            ),

        "global_fallback_count":
            int(
                (
                    top["feature_source"]
                    == "global_fallback"
                ).sum()
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.4.1 "
        "Fallback Penalty Comparison Test"
    )
    print("=" * 70)


    print()
    print("FIXED WEIGHTS")
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


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()


    print()
    print(
        f"records = {len(df):,}"
    )


    print(
        f"OOS period = "
        f"{TEST_START.date()} "
        f"to "
        f"{TEST_END.date()}"
    )


    # --------------------------------------------------------
    # Build panels
    # --------------------------------------------------------

    panels = {}

    diagnostics = []


    print()
    print(
        "Building daily feature panels..."
    )


    for target_date in pd.date_range(
        TEST_START,
        TEST_END,
    ):

        panel, diag = (
            build_features(
                df,
                target_date,
            )
        )


        if panel.empty:

            print(
                target_date.date(),
                "EMPTY"
            )

            continue


        panel = calculate_score(
            panel
        )


        panels[target_date] = panel


        diagnostics.append({

            "date":
                target_date.date(),

            "machines":
                len(panel),

            "same_machine":
                diag[
                    "same_machine"
                ],

            "type_fallback":
                diag[
                    "type_fallback"
                ],

            "global_fallback":
                diag[
                    "global_fallback"
                ],

            "machine_change":
                diag[
                    "machine_change"
                ],
        })


        print(
            f"{target_date.date()} "
            f"machines={len(panel)} "
            f"same_machine="
            f"{diag['same_machine']} "
            f"type_fallback="
            f"{diag['type_fallback']} "
            f"global_fallback="
            f"{diag['global_fallback']} "
            f"machine_change="
            f"{diag['machine_change']}"
        )


    # --------------------------------------------------------
    # Evaluate all models
    # --------------------------------------------------------

    daily_rows = []


    for model_name in MODELS:

        print()
        print(
            f"Evaluating {model_name}..."
        )


        for target_date, panel in (
            panels.items()
        ):

            ranked = apply_model(
                panel,
                model_name,
            )


            for top_n in (
                1,
                5,
                10,
                20,
                30,
            ):

                result = evaluate_daily(
                    ranked,
                    top_n,
                )


                if result is None:

                    continue


                daily_rows.append({

                    "model":
                        model_name,

                    "date":
                        target_date.date(),

                    "top_n":
                        top_n,

                    **result,
                })


    daily_df = pd.DataFrame(
        daily_rows
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = []


    for (
        model_name,
        top_n
    ), g in daily_df.groupby(
        [
            "model",
            "top_n",
        ]
    ):

        summary_rows.append({

            "model":
                model_name,

            "top_n":
                top_n,

            "days":
                len(g),

            "avg_diff":
                float(
                    g["avg_diff"].mean()
                ),

            "median_daily_avg":
                float(
                    g[
                        "avg_diff"
                    ].median()
                ),

            "win_rate":
                float(
                    g[
                        "win_rate"
                    ].mean()
                ),

            "plus1000_rate":
                float(
                    g[
                        "plus1000_rate"
                    ].mean()
                ),

            "plus2000_rate":
                float(
                    g[
                        "plus2000_rate"
                    ].mean()
                ),

            "positive_days":
                float(
                    g[
                        "positive"
                    ].mean()
                    * 100
                ),

            "total_diff":
                float(
                    g[
                        "total_diff"
                    ].sum()
                ),

            "avg_machine_change":
                float(
                    g[
                        "machine_change_count"
                    ].mean()
                ),

            "avg_global_fallback":
                float(
                    g[
                        "global_fallback_count"
                    ].mean()
                ),
        })


    summary_df = pd.DataFrame(
        summary_rows
    )


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VER.4.1 FALLBACK PENALTY RESULT"
    )
    print("=" * 70)


    print(
        summary_df[
            [
                "model",
                "top_n",
                "days",
                "avg_diff",
                "median_daily_avg",
                "win_rate",
                "plus1000_rate",
                "plus2000_rate",
                "positive_days",
                "total_diff",
            ]
        ]
        .sort_values(
            [
                "top_n",
                "avg_diff",
            ],
            ascending=[
                True,
                False,
            ]
        )
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # TOP10 comparison
    # --------------------------------------------------------

    top10 = summary_df[
        summary_df["top_n"] == 10
    ].copy()


    top10 = top10.sort_values(
        "total_diff",
        ascending=False
    )


    print()
    print("=" * 70)
    print(
        "TOP10 MODEL COMPARISON"
    )
    print("=" * 70)


    print(
        top10[
            [
                "model",
                "days",
                "avg_diff",
                "median_daily_avg",
                "win_rate",
                "plus1000_rate",
                "plus2000_rate",
                "positive_days",
                "total_diff",
                "avg_machine_change",
                "avg_global_fallback",
            ]
        ]
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    if not top10.empty:

        best = top10.iloc[0]


        print()
        print(
            "BEST TOP10 MODEL"
        )
        print("-" * 70)


        print(
            f"model              : "
            f"{best['model']}"
        )

        print(
            f"avg_diff           : "
            f"{best['avg_diff']:.2f}"
        )

        print(
            f"win_rate           : "
            f"{best['win_rate']:.2f}%"
        )

        print(
            f"positive_days      : "
            f"{best['positive_days']:.2f}%"
        )

        print(
            f"total_diff         : "
            f"{best['total_diff']:.0f}"
        )


    # --------------------------------------------------------
    # 8/3 special check
    # --------------------------------------------------------

    target_special = (
        pd.Timestamp("2026-08-03")
    )


    special = daily_df[
        (
            daily_df["date"]
            == target_special.date()
        )
        &
        (
            daily_df["top_n"]
            == 10
        )
    ].copy()


    print()
    print("=" * 70)
    print(
        "2026-08-03 TOP10 COMPARISON"
    )
    print("=" * 70)


    if not special.empty:

        print(
            special[
                [
                    "model",
                    "avg_diff",
                    "win_rate",
                    "plus1000_rate",
                    "plus2000_rate",
                    "total_diff",
                    "machine_change_count",
                    "global_fallback_count",
                ]
            ]
            .sort_values(
                "total_diff",
                ascending=False
            )
            .to_string(
                index=False
            )
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out_daily = (
        OUT_DIR
        / "20_Ver4_1_fallback_penalty_daily.csv"
    )


    out_summary = (
        OUT_DIR
        / "20_Ver4_1_fallback_penalty_summary.csv"
    )


    out_diag = (
        OUT_DIR
        / "20_Ver4_1_fallback_penalty_diagnostics.csv"
    )


    out_models = (
        OUT_DIR
        / "20_Ver4_1_fallback_penalty_models.csv"
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


    pd.DataFrame(
        diagnostics
    ).to_csv(
        out_diag,
        index=False,
        encoding="utf-8-sig"
    )


    model_rows = []


    for name, cfg in MODELS.items():

        model_rows.append({

            "model":
                name,

            "change_penalty":
                cfg[
                    "change_penalty"
                ],

            "global_penalty":
                cfg[
                    "global_penalty"
                ],

            "exclude_global":
                cfg[
                    "exclude_global"
                ],
        })


    pd.DataFrame(
        model_rows
    ).to_csv(
        out_models,
        index=False,
        encoding="utf-8-sig"
    )


    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)


    print(out_daily)
    print(out_summary)
    print(out_diag)
    print(out_models)


    print()
    print(
        "Ver.4.1 fallback penalty "
        "comparison complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()