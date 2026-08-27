from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 72 - V4.2_C Walk-Forward Rank-Band Backtest
#
# Purpose:
#   Extend the existing 48 rolling OOS logic with the same
#   rank-band evaluation used by 71 forward tracking.
#
# Important:
#   - V4.2_C weights are FIXED.
#   - Features for target_date use ONLY rows date < target_date.
#   - Actual target-date diff is joined only for evaluation.
#   - No 64 / 69 / 70 / 71 files are modified.
#   - This is a BACKTEST, kept separate from live forward results.
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
    / "72_Ver4_2_C_walk_forward_rank_band_backtest"
)

CSV1 = DATA_DIR / "ana_slo_20260711_20260818.csv"
CSV2 = DATA_DIR / "__unused__.csv"

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-18")

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
# V4.2_C weights
# Exact derivation preserved from existing 48 program:
# V4 weights -> remove recent7_win and bounce_signal -> normalize.
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


def normalize_weights(weights):
    total = sum(weights.values())

    if total <= 0:
        raise ValueError("Weight sum must be positive.")

    return {
        k: v / total
        for k, v in weights.items()
    }


V42_C_RAW = V4_WEIGHTS.copy()
V42_C_RAW.pop("recent7_win")
V42_C_RAW.pop("bounce_signal")
V42_C = normalize_weights(V42_C_RAW)

MODEL_NAME = "V4.2_C"


# ============================================================
# Same rolling OOS splits as existing 48
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


RANK_BANDS = {
    "TOP1": (1, 1),
    "TOP3": (1, 3),
    "TOP5": (1, 5),
    "TOP10": (1, 10),
    "PRIMARY_1_5": (1, 5),
    "NEXT_6_10": (6, 10),
    "RANK_7_9": (7, 9),
}

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260825


# ============================================================
# Helpers
# ============================================================

def header(title):
    print()
    print("=" * 116)
    print(title)
    print("=" * 116)


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
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
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
# Preserved from existing 48 program.
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

        if last_diff <= -1000:
            bounce_signal = 1.0
        elif last_diff <= -500:
            bounce_signal = 0.5
        elif last_diff >= 1000:
            bounce_signal = -0.25
        else:
            bounce_signal = 0.0

        rows.append({
            "machine_no": int(no),
            "machine_name": name,
            "avg31": avg31,
            "recent7_avg": recent7_avg,
            "recent7_win": recent7_win,
            "last_diff": last_diff,
            "prev_change": prev_change,
            "weekday_avg": weekday_avg,
            "type_avg": type_avg,
            "plus1000_rate": plus1000_rate,
            "plus2000_rate": plus2000_rate,
            "neighbor_avg": neighbor_avg,
            "bounce_signal": bounce_signal,
        })

    feat = pd.DataFrame(
        rows
    )

    if feat.empty:
        return feat

    # Same strict machine_no + machine_name merge as existing 48.
    # This intentionally excludes a target-day machine-name change
    # rather than silently evaluating it as the previous machine.
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

    x = x.sort_values(
        [
            "score",
            "machine_no",
        ],
        ascending=[
            False,
            True,
        ]
    ).reset_index(drop=True)

    x["prediction_rank"] = (
        np.arange(len(x)) + 1
    )

    return x


def safe_spearman(x, y):
    tmp = pd.DataFrame({
        "x": pd.to_numeric(
            x,
            errors="coerce"
        ),
        "y": pd.to_numeric(
            y,
            errors="coerce"
        ),
    }).dropna()

    if len(tmp) < 5:
        return np.nan

    if (
        tmp["x"].nunique() < 2
        or tmp["y"].nunique() < 2
    ):
        return np.nan

    return float(
        tmp[["x", "y"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )


def bootstrap_mean_ci(
    values,
    seed_offset=0,
):
    x = np.asarray(
        values,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) < 2:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
        + seed_offset
    )

    samples = rng.choice(
        x,
        size=(
            BOOTSTRAP_REPS,
            len(x),
        ),
        replace=True
    )

    means = samples.mean(
        axis=1
    )

    return (
        float(
            np.percentile(
                means,
                2.5
            )
        ),
        float(
            np.percentile(
                means,
                97.5
            )
        ),
    )


# ============================================================
# Build daily ranked predictions
# ============================================================

def build_ranked_days(df):
    rows = []
    quality_rows = []

    split_map = {}

    for (
        split_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in ROLLING_SPLITS:
        for target_date in pd.date_range(
            test_start,
            test_end
        ):
            split_map[target_date] = {
                "split": split_name,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }

    for target_date in sorted(
        split_map.keys()
    ):
        panel = build_features(
            df,
            target_date
        )

        if panel.empty:
            quality_rows.append({
                "target_date": target_date,
                "split": split_map[target_date]["split"],
                "status": "NO_PANEL",
                "machines_ranked": 0,
                "latest_history_date": pd.NaT,
                "future_leak_check": False,
            })
            continue

        hist = df[
            df["date"] < target_date
        ]

        latest_history_date = (
            hist["date"].max()
            if not hist.empty
            else pd.NaT
        )

        future_leak_check = bool(
            pd.notna(latest_history_date)
            and latest_history_date < target_date
        )

        ranked = rank_score(
            panel,
            V42_C
        )

        meta = split_map[
            target_date
        ]

        ranked["target_date"] = target_date
        ranked["split"] = meta["split"]
        ranked["train_start"] = meta["train_start"]
        ranked["train_end"] = meta["train_end"]
        ranked["test_start"] = meta["test_start"]
        ranked["test_end"] = meta["test_end"]
        ranked["latest_history_date"] = latest_history_date
        ranked["model"] = MODEL_NAME

        ranked["actual_win"] = (
            ranked["diff"] > 0
        ).astype(int)

        ranked["actual_plus1000"] = (
            ranked["diff"] >= 1000
        ).astype(int)

        ranked["actual_plus2000"] = (
            ranked["diff"] >= 2000
        ).astype(int)

        rows.append(
            ranked
        )

        quality_rows.append({
            "target_date": target_date,
            "split": meta["split"],
            "status": "OK",
            "machines_ranked": int(
                len(ranked)
            ),
            "latest_history_date": latest_history_date,
            "future_leak_check": future_leak_check,
            "rank_1_to_10_available": bool(
                len(ranked) >= 10
            ),
            "duplicate_machine_no": int(
                ranked.duplicated(
                    subset=["machine_no"]
                ).sum()
            ),
        })

    if not rows:
        raise RuntimeError(
            "No ranked walk-forward days were created."
        )

    detail = pd.concat(
        rows,
        ignore_index=True
    )

    quality = pd.DataFrame(
        quality_rows
    )

    return (
        detail,
        quality,
    )


# ============================================================
# Rank-band evaluation
# ============================================================

def summarize_selected(selected):
    d = (
        selected["diff"]
        .astype(float)
    )

    return {
        "selected_n": int(
            len(selected)
        ),
        "avg_diff": float(
            d.mean()
        ),
        "median_diff": float(
            d.median()
        ),
        "total_diff": float(
            d.sum()
        ),
        "win_rate": float(
            (d > 0).mean()
            * 100.0
        ),
        "plus1000_rate": float(
            (d >= 1000).mean()
            * 100.0
        ),
        "plus2000_rate": float(
            (d >= 2000).mean()
            * 100.0
        ),
        "min_diff": float(
            d.min()
        ),
        "max_diff": float(
            d.max()
        ),
    }


def build_daily_band(detail):
    rows = []

    for target_date, group in detail.groupby(
        "target_date",
        sort=True
    ):
        top10 = group[
            group["prediction_rank"] <= 10
        ].copy()

        score_spearman_top10 = safe_spearman(
            top10["score"],
            top10["diff"]
        )

        rank_spearman_top10 = safe_spearman(
            top10["prediction_rank"],
            top10["diff"]
        )

        store_avg_diff = float(
            group["diff"].mean()
        )

        for band, (
            rank_start,
            rank_end,
        ) in RANK_BANDS.items():
            selected = group[
                group["prediction_rank"].between(
                    rank_start,
                    rank_end
                )
            ].copy()

            if selected.empty:
                continue

            row = summarize_selected(
                selected
            )

            row.update({
                "target_date": target_date,
                "split": str(
                    group.iloc[0]["split"]
                ),
                "band": band,
                "rank_start": rank_start,
                "rank_end": rank_end,
                "store_avg_diff": store_avg_diff,
                "avg_diff_lift_vs_store":
                    row["avg_diff"]
                    - store_avg_diff,
                "excess_total_vs_store":
                    (
                        row["avg_diff"]
                        - store_avg_diff
                    )
                    * row["selected_n"],
                "score_spearman_top10":
                    score_spearman_top10,
                "rank_spearman_top10":
                    rank_spearman_top10,
            })

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def build_overall_band(
    daily,
    detail,
):
    rows = []

    for band, (
        rank_start,
        rank_end,
    ) in RANK_BANDS.items():
        selected = detail[
            detail["prediction_rank"].between(
                rank_start,
                rank_end
            )
        ].copy()

        bday = daily[
            daily["band"] == band
        ].copy()

        if selected.empty or bday.empty:
            continue

        d = selected[
            "diff"
        ].astype(float)

        machine_ci_low, machine_ci_high = (
            bootstrap_mean_ci(
                d.to_numpy(),
                seed_offset=(
                    rank_start * 100
                    + rank_end
                )
            )
        )

        daily_ci_low, daily_ci_high = (
            bootstrap_mean_ci(
                bday["avg_diff"].to_numpy(),
                seed_offset=(
                    rank_start * 1000
                    + rank_end
                )
            )
        )

        lift_ci_low, lift_ci_high = (
            bootstrap_mean_ci(
                bday[
                    "avg_diff_lift_vs_store"
                ].to_numpy(),
                seed_offset=(
                    rank_start * 10000
                    + rank_end
                )
            )
        )

        rows.append({
            "band": band,
            "rank_start": rank_start,
            "rank_end": rank_end,
            "evaluated_days": int(
                bday["target_date"].nunique()
            ),
            "selected_rows": int(
                len(selected)
            ),
            "avg_diff_per_machine": float(
                d.mean()
            ),
            "median_diff_per_machine": float(
                d.median()
            ),
            "total_diff": float(
                d.sum()
            ),
            "win_rate": float(
                (d > 0).mean()
                * 100.0
            ),
            "plus1000_rate": float(
                (d >= 1000).mean()
                * 100.0
            ),
            "plus2000_rate": float(
                (d >= 2000).mean()
                * 100.0
            ),
            "positive_day_rate": float(
                (
                    bday["total_diff"] > 0
                ).mean()
                * 100.0
            ),
            "mean_daily_avg_diff": float(
                bday["avg_diff"].mean()
            ),
            "mean_store_avg_diff": float(
                bday["store_avg_diff"].mean()
            ),
            "mean_lift_vs_store": float(
                bday[
                    "avg_diff_lift_vs_store"
                ].mean()
            ),
            "total_excess_vs_store": float(
                bday[
                    "excess_total_vs_store"
                ].sum()
            ),
            "machine_avg_diff_ci95_low":
                machine_ci_low,
            "machine_avg_diff_ci95_high":
                machine_ci_high,
            "daily_avg_diff_ci95_low":
                daily_ci_low,
            "daily_avg_diff_ci95_high":
                daily_ci_high,
            "daily_lift_ci95_low":
                lift_ci_low,
            "daily_lift_ci95_high":
                lift_ci_high,
        })

    return pd.DataFrame(
        rows
    )


def build_split_band(daily):
    rows = []

    for (
        split,
        band,
    ), g in daily.groupby(
        [
            "split",
            "band",
        ],
        sort=True
    ):
        rows.append({
            "split": split,
            "band": band,
            "days": int(
                g["target_date"].nunique()
            ),
            "selected_rows": int(
                g["selected_n"].sum()
            ),
            "mean_daily_avg_diff": float(
                g["avg_diff"].mean()
            ),
            "total_diff": float(
                g["total_diff"].sum()
            ),
            "mean_win_rate": float(
                g["win_rate"].mean()
            ),
            "mean_plus1000_rate": float(
                g["plus1000_rate"].mean()
            ),
            "mean_plus2000_rate": float(
                g["plus2000_rate"].mean()
            ),
            "positive_day_rate": float(
                (
                    g["total_diff"] > 0
                ).mean()
                * 100.0
            ),
            "mean_store_avg_diff": float(
                g["store_avg_diff"].mean()
            ),
            "mean_lift_vs_store": float(
                g[
                    "avg_diff_lift_vs_store"
                ].mean()
            ),
            "total_excess_vs_store": float(
                g[
                    "excess_total_vs_store"
                ].sum()
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_rank_summary(detail):
    rows = []

    top10 = detail[
        detail["prediction_rank"] <= 10
    ].copy()

    for rank, g in top10.groupby(
        "prediction_rank",
        sort=True
    ):
        d = g[
            "diff"
        ].astype(float)

        rows.append({
            "prediction_rank": int(rank),
            "n": int(
                len(g)
            ),
            "avg_actual_diff": float(
                d.mean()
            ),
            "median_actual_diff": float(
                d.median()
            ),
            "total_actual_diff": float(
                d.sum()
            ),
            "win_rate": float(
                (d > 0).mean()
                * 100.0
            ),
            "plus1000_rate": float(
                (d >= 1000).mean()
                * 100.0
            ),
            "plus2000_rate": float(
                (d >= 2000).mean()
                * 100.0
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_daily_rank_matrix(detail):
    top10 = detail[
        detail["prediction_rank"] <= 10
    ].copy()

    matrix = top10.pivot(
        index="prediction_rank",
        columns="target_date",
        values="diff"
    ).sort_index()

    matrix.columns = [
        pd.Timestamp(c).strftime(
            "%Y-%m-%d"
        )
        for c in matrix.columns
    ]

    matrix = matrix.reset_index()

    date_cols = [
        c
        for c in matrix.columns
        if c != "prediction_rank"
    ]

    matrix["total_diff"] = (
        matrix[date_cols]
        .sum(axis=1)
    )

    matrix["avg_diff"] = (
        matrix[date_cols]
        .mean(axis=1)
    )

    matrix["win_days"] = (
        matrix[date_cols] > 0
    ).sum(axis=1)

    matrix["loss_days"] = (
        matrix[date_cols] <= 0
    ).sum(axis=1)

    return matrix


def build_daily_order(detail):
    rows = []

    for target_date, group in detail.groupby(
        "target_date",
        sort=True
    ):
        top10 = group[
            group["prediction_rank"] <= 10
        ].copy()

        rows.append({
            "target_date": target_date,
            "split": str(
                group.iloc[0]["split"]
            ),
            "top10_n": int(
                len(top10)
            ),
            "score_spearman_vs_actual_diff":
                safe_spearman(
                    top10["score"],
                    top10["diff"]
                ),
            "rank_spearman_vs_actual_diff":
                safe_spearman(
                    top10["prediction_rank"],
                    top10["diff"]
                ),
            "top10_avg_diff": float(
                top10["diff"].mean()
            ),
            "top10_total_diff": float(
                top10["diff"].sum()
            ),
            "top10_win_rate": float(
                (
                    top10["diff"] > 0
                ).mean()
                * 100.0
            ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():
    header(
        "72 - V4.2_C Walk-Forward Rank-Band Backtest"
    )

    print(f"input CSV             : {CSV1}")
    print(f"output directory      : {OUT_DIR}")
    print(f"source period         : {START.date()} to {END.date()}")
    print(f"model                 : {MODEL_NAME}")
    print(f"factor count          : {len(V42_C)}")
    print(f"weight sum            : {sum(V42_C.values()):.12f}")

    print()
    print("V4.2_C normalized weights")
    print("-" * 116)

    for factor, weight in V42_C.items():
        print(
            f"{factor:<20} {weight:.12f}"
        )

    df = load_data()

    print()
    print(f"records loaded        : {len(df):,}")
    print(
        f"data date range       : "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    detail, quality = build_ranked_days(
        df
    )

    header("DATA QUALITY")
    print(
        quality.to_string(
            index=False
        )
    )

    if not quality[
        "future_leak_check"
    ].fillna(False).all():
        raise RuntimeError(
            "Future leakage check failed."
        )

    ok_quality = quality[
        quality["status"] == "OK"
    ].copy()

    if ok_quality.empty:
        raise RuntimeError(
            "No eligible backtest days."
        )

    if not ok_quality[
        "rank_1_to_10_available"
    ].fillna(False).all():
        raise RuntimeError(
            "At least one day has fewer than 10 ranked machines."
        )

    daily = build_daily_band(
        detail
    )

    overall = build_overall_band(
        daily,
        detail
    )

    split_band = build_split_band(
        daily
    )

    rank_summary = build_rank_summary(
        detail
    )

    rank_matrix = build_daily_rank_matrix(
        detail
    )

    daily_order = build_daily_order(
        detail
    )

    header("OVERALL RANK-BAND RESULTS")
    print(
        overall[
            [
                "band",
                "evaluated_days",
                "selected_rows",
                "avg_diff_per_machine",
                "total_diff",
                "win_rate",
                "plus1000_rate",
                "plus2000_rate",
                "positive_day_rate",
                "mean_store_avg_diff",
                "mean_lift_vs_store",
                "daily_lift_ci95_low",
                "daily_lift_ci95_high",
            ]
        ].to_string(
            index=False
        )
    )

    header("SPLIT / RANK-BAND RESULTS")
    print(
        split_band.to_string(
            index=False
        )
    )

    header("RANK SUMMARY")
    print(
        rank_summary.to_string(
            index=False
        )
    )

    header("DAILY SCORE / RANK ORDER")
    print(
        daily_order.to_string(
            index=False
        )
    )

    header("KEY COMPARISON")

    key = overall[
        overall["band"].isin(
            [
                "PRIMARY_1_5",
                "NEXT_6_10",
                "RANK_7_9",
                "TOP10",
            ]
        )
    ][
        [
            "band",
            "evaluated_days",
            "selected_rows",
            "avg_diff_per_machine",
            "total_diff",
            "win_rate",
            "plus2000_rate",
            "positive_day_rate",
            "mean_lift_vs_store",
            "daily_lift_ci95_low",
            "daily_lift_ci95_high",
        ]
    ]

    print(
        key.to_string(
            index=False
        )
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    paths = {
        "quality":
            OUT_DIR / "72_data_quality.csv",
        "detail":
            OUT_DIR / "72_backtest_detail.csv",
        "daily_band":
            OUT_DIR / "72_rank_band_daily.csv",
        "overall_band":
            OUT_DIR / "72_rank_band_overall.csv",
        "split_band":
            OUT_DIR / "72_rank_band_by_split.csv",
        "rank_summary":
            OUT_DIR / "72_rank_summary.csv",
        "rank_matrix":
            OUT_DIR / "72_daily_rank_matrix.csv",
        "daily_order":
            OUT_DIR / "72_daily_score_rank_order.csv",
    }

    quality.to_csv(
        paths["quality"],
        index=False,
        encoding="utf-8-sig"
    )

    detail.to_csv(
        paths["detail"],
        index=False,
        encoding="utf-8-sig"
    )

    daily.to_csv(
        paths["daily_band"],
        index=False,
        encoding="utf-8-sig"
    )

    overall.to_csv(
        paths["overall_band"],
        index=False,
        encoding="utf-8-sig"
    )

    split_band.to_csv(
        paths["split_band"],
        index=False,
        encoding="utf-8-sig"
    )

    rank_summary.to_csv(
        paths["rank_summary"],
        index=False,
        encoding="utf-8-sig"
    )

    rank_matrix.to_csv(
        paths["rank_matrix"],
        index=False,
        encoding="utf-8-sig"
    )

    daily_order.to_csv(
        paths["daily_order"],
        index=False,
        encoding="utf-8-sig"
    )

    header("FILES SAVED")

    for path in paths.values():
        print(path)

    print()
    print(
        "72 walk-forward rank-band backtest complete."
    )
    print(
        "V4.2_C weights were fixed; no live prediction files were modified."
    )
    print(
        "Compare PRIMARY_1_5 vs NEXT_6_10 vs RANK_7_9 "
        "with 71 live-forward results."
    )


if __name__ == "__main__":
    main()
