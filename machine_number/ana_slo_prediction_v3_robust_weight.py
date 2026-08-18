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

OPT_FILE = (
    OUT_DIR
    / "09_Ver3_weight_optimization_results.csv"
)

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


def load_optimization():

    if not OPT_FILE.exists():
        raise FileNotFoundError(
            "Optimization result not found: "
            + str(OPT_FILE)
        )

    df = pd.read_csv(
        OPT_FILE,
        encoding="utf-8-sig"
    )

    required = [
        "rank",
        "pattern",
        "objective",
    ]

    for factor in FACTORS:
        required.append(
            "w_" + factor
        )

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    return df


def analyze_group(
    df,
    group_name
):

    rows = []

    for factor in FACTORS:

        col = "w_" + factor

        s = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if s.empty:
            continue

        rows.append({
            "group": group_name,
            "factor": factor,
            "mean": float(
                s.mean()
            ),
            "median": float(
                s.median()
            ),
            "std": float(
                s.std(ddof=0)
            ),
            "min": float(
                s.min()
            ),
            "max": float(
                s.max()
            ),
            "range": float(
                s.max() - s.min()
            ),
            "cv": float(
                s.std(ddof=0)
                / s.mean()
            )
            if s.mean() > 0
            else np.nan,
            "high_weight_rate": float(
                (s >= 0.10).mean()
                * 100
            ),
            "very_high_weight_rate": float(
                (s >= 0.15).mean()
                * 100
            ),
        })

    return pd.DataFrame(rows)


def factor_name(factor):

    names = {
        "avg31": "31日平均",
        "recent7_avg": "直近7日平均",
        "recent7_win": "直近7日勝率",
        "last_diff": "前日差枚",
        "prev_change": "前々日→前日の変化",
        "weekday_avg": "曜日平均",
        "type_avg": "機種平均",
        "plus1000_rate": "+1000出率",
        "plus2000_rate": "+2000出率",
        "neighbor_avg": "隣接台平均",
        "bounce_signal": "リバウンド信号",
    }

    return names.get(
        factor,
        factor
    )


def make_rank_table(
    analysis
):

    x = analysis.copy()

    x["factor_name"] = (
        x["factor"]
        .map(factor_name)
    )

    return x[
        [
            "group",
            "factor",
            "factor_name",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "range",
            "cv",
            "high_weight_rate",
            "very_high_weight_rate",
        ]
    ]


def make_robust_score(
    analysis
):

    rows = []

    for factor in FACTORS:

        x = analysis[
            analysis["factor"] == factor
        ].copy()

        if x.empty:
            continue

        # 上位パターンで平均的に高い
        mean_weight = float(
            x["mean"].mean()
        )

        # グループ間で安定している
        group_std = float(
            x["mean"].std(ddof=0)
        )

        # 個々のパターン内でも安定
        pattern_std = float(
            x["std"].mean()
        )

        high_rate = float(
            x["high_weight_rate"].mean()
        )

        very_high_rate = float(
            x["very_high_weight_rate"].mean()
        )

        # 高ウェイト率を重視し、
        # ばらつきを減点する。
        robust_score = (
            mean_weight * 100
            + high_rate * 0.20
            + very_high_rate * 0.10
            - group_std * 100
            - pattern_std * 50
        )

        rows.append({
            "factor": factor,
            "factor_name": factor_name(
                factor
            ),
            "mean_weight": mean_weight,
            "group_mean_std": group_std,
            "pattern_std_mean": pattern_std,
            "high_weight_rate": high_rate,
            "very_high_weight_rate": very_high_rate,
            "robust_score": robust_score,
        })

    result = pd.DataFrame(
        rows
    )

    return result.sort_values(
        "robust_score",
        ascending=False
    ).reset_index(
        drop=True
    )


def compare_top_groups(
    opt
):

    groups = [
        ("TOP5", 5),
        ("TOP10", 10),
        ("TOP20", 20),
    ]

    rows = []

    for name, n in groups:

        x = opt.head(n)

        for factor in FACTORS:

            col = "w_" + factor

            s = pd.to_numeric(
                x[col],
                errors="coerce"
            )

            rows.append({
                "group": name,
                "factor": factor,
                "factor_name": factor_name(
                    factor
                ),
                "mean": float(
                    s.mean()
                ),
                "median": float(
                    s.median()
                ),
                "std": float(
                    s.std(ddof=0)
                ),
            })

    return pd.DataFrame(rows)


def make_recommendation(
    robust
):

    lines = []

    lines.append(
        "Ver.3 Robust Weight Analysis"
    )
    lines.append(
        "========================================"
    )
    lines.append("")
    lines.append(
        "Purpose:"
    )
    lines.append(
        "Identify factors that remain important "
        "across multiple high-performing "
        "weight patterns."
    )
    lines.append("")

    lines.append(
        "Robust factor ranking:"
    )

    for i, row in robust.iterrows():

        lines.append(
            "%2d. %-18s "
            "mean=%.3f "
            "high>=10%%=%.1f%% "
            "very_high>=15%%=%.1f%% "
            "robust=%.2f"
            % (
                i + 1,
                row["factor_name"],
                row["mean_weight"],
                row["high_weight_rate"],
                row["very_high_weight_rate"],
                row["robust_score"],
            )
        )

    lines.append("")
    lines.append(
        "Interpretation:"
    )
    lines.append(
        "Higher robust_score means that the "
        "factor tends to receive higher weights "
        "while remaining relatively stable "
        "across strong patterns."
    )

    lines.append("")
    lines.append(
        "Important:"
    )
    lines.append(
        "This is not proof that a factor causes "
        "machine performance. It only measures "
        "its stability within the tested model."
    )

    lines.append("")
    lines.append(
        "Recommended candidates for a robust "
        "Ver.3 model:"
    )

    # 上位5要因
    for i, row in robust.head(5).iterrows():

        lines.append(
            "%d. %s"
            % (
                i + 1,
                row["factor_name"]
            )
        )

    return "\n".join(lines)


def main():

    print("=" * 70)
    print(
        "Ver.3 Robust Weight Analysis"
    )
    print("=" * 70)

    opt = load_optimization()

    print(
        "optimization patterns = %d"
        % len(opt)
    )

    print()

    # -----------------------------------------
    # 上位20
    # -----------------------------------------

    top20 = opt.head(20).copy()

    print(
        "Analyzing TOP20 weight patterns..."
    )

    analysis20 = analyze_group(
        top20,
        "TOP20"
    )

    # -----------------------------------------
    # TOP10
    # -----------------------------------------

    top10 = opt.head(10).copy()

    analysis10 = analyze_group(
        top10,
        "TOP10"
    )

    # -----------------------------------------
    # TOP5
    # -----------------------------------------

    top5 = opt.head(5).copy()

    analysis5 = analyze_group(
        top5,
        "TOP5"
    )

    analysis = pd.concat(
        [
            analysis5,
            analysis10,
            analysis20,
        ],
        ignore_index=True
    )

    # -----------------------------------------
    # Robust score
    # -----------------------------------------

    robust = make_robust_score(
        analysis
    )

    # -----------------------------------------
    # Group comparison
    # -----------------------------------------

    comparison = compare_top_groups(
        opt
    )

    # -----------------------------------------
    # Pattern weight table
    # -----------------------------------------

    pattern_rows = []

    for _, row in top20.iterrows():

        for factor in FACTORS:

            pattern_rows.append({
                "rank": int(
                    row["rank"]
                ),
                "pattern": row["pattern"],
                "objective": float(
                    row["objective"]
                ),
                "factor": factor,
                "factor_name": factor_name(
                    factor
                ),
                "weight": float(
                    row["w_" + factor]
                ),
            })

    pattern_df = pd.DataFrame(
        pattern_rows
    )

    # -----------------------------------------
    # Save
    # -----------------------------------------

    out_stats = (
        OUT_DIR
        / "13_Ver3_robust_weight_stats.csv"
    )

    out_robust = (
        OUT_DIR
        / "13_Ver3_robust_factor_ranking.csv"
    )

    out_compare = (
        OUT_DIR
        / "13_Ver3_robust_top_group_compare.csv"
    )

    out_patterns = (
        OUT_DIR
        / "13_Ver3_robust_top20_patterns.csv"
    )

    out_readme = (
        OUT_DIR
        / "13_Ver3_robust_weight_README.txt"
    )

    analysis.to_csv(
        out_stats,
        index=False,
        encoding="utf-8-sig"
    )

    robust.to_csv(
        out_robust,
        index=False,
        encoding="utf-8-sig"
    )

    comparison.to_csv(
        out_compare,
        index=False,
        encoding="utf-8-sig"
    )

    pattern_df.to_csv(
        out_patterns,
        index=False,
        encoding="utf-8-sig"
    )

    readme = make_recommendation(
        robust
    )

    out_readme.write_text(
        readme,
        encoding="utf-8"
    )

    # -----------------------------------------
    # Display
    # -----------------------------------------

    print()
    print(
        "===== ROBUST FACTOR RANKING ====="
    )

    display_cols = [
        "factor_name",
        "mean_weight",
        "group_mean_std",
        "pattern_std_mean",
        "high_weight_rate",
        "very_high_weight_rate",
        "robust_score",
    ]

    print(
        robust[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "===== TOP20 WEIGHT STATISTICS ====="
    )

    print(
        make_rank_table(
            analysis20
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "===== TOP5 / TOP10 / TOP20 COMPARISON ====="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(out_stats)
    print(out_robust)
    print(out_compare)
    print(out_patterns)
    print(out_readme)

    print()
    print(
        "Robust weight analysis complete."
    )


if __name__ == "__main__":
    main()