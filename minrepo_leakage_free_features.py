from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# SlotAnalyzer - Min-Repo Leakage-Free Historical Features
# ============================================================
#
# Input:
#   data\maruhan_maebashi\external_features\minrepo\
#       store_summary_all_dates.csv
#       machine_summary_all_dates.csv
#       tail_summary_all_dates.csv
#
# Output:
#   data\maruhan_maebashi\external_features\minrepo\history_features\
#
# Important:
#   Every feature for date D is calculated ONLY from rows with date < D.
#   Same-day Min-Repo values are never used as predictors for date D.
# ============================================================


ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

SOURCE_DIR = (
    ROOT
    / "data"
    / "maruhan_maebashi"
    / "external_features"
    / "minrepo"
)

OUT_DIR = (
    SOURCE_DIR
    / "history_features"
)

STORE_FILE = (
    SOURCE_DIR
    / "store_summary_all_dates.csv"
)

MACHINE_FILE = (
    SOURCE_DIR
    / "machine_summary_all_dates.csv"
)

TAIL_FILE = (
    SOURCE_DIR
    / "tail_summary_all_dates.csv"
)


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )


def normalize_date_col(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    out["date"] = pd.to_datetime(
        out["date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    # Also support YYYY-MM-DD if input format changes later.
    missing = out["date"].isna()

    if missing.any():
        out.loc[
            missing,
            "date",
        ] = pd.to_datetime(
            df.loc[
                missing,
                "date",
            ],
            errors="coerce",
        )

    if out["date"].isna().any():
        bad = df.loc[
            out["date"].isna(),
            "date",
        ].tolist()

        raise RuntimeError(
            f"Invalid dates found: {bad[:10]}"
        )

    return out


def normalize_machine_name(value) -> str:
    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    # Remove common source-specific prefixes only for matching.
    text = re.sub(
        r"^(スマスロ|Lパチスロ|Lスロット|L|スロット)",
        "",
        text,
    )

    return text


def safe_mean(series: pd.Series) -> float:
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return float(
        x.mean()
    ) if len(x) else np.nan


def safe_std(series: pd.Series) -> float:
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return float(
        x.std(ddof=0)
    ) if len(x) else np.nan


def safe_last(series: pd.Series) -> float:
    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return float(
        x.iloc[-1]
    ) if len(x) else np.nan


def prior_window(
    hist: pd.DataFrame,
    target_date: pd.Timestamp,
    n: int | None = None,
) -> pd.DataFrame:

    x = (
        hist[
            hist["date"]
            < target_date
        ]
        .sort_values("date")
        .copy()
    )

    if n is not None:
        x = x.tail(n)

    return x


# ============================================================
# STORE FEATURES
# ============================================================

def build_store_features(
    store: pd.DataFrame,
) -> pd.DataFrame:

    target_dates = (
        store[
            "date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    rows = []

    for target_date in target_dates:
        prior_all = prior_window(
            store,
            target_date,
        )

        prior3 = prior_window(
            store,
            target_date,
            3,
        )

        prior5 = prior_window(
            store,
            target_date,
            5,
        )

        row = {
            "date":
                target_date,

            "store_hist_days":
                len(prior_all),

            "store_avg_diff_last":
                safe_last(
                    prior_all[
                        "avg_diff"
                    ]
                ),

            "store_avg_diff_mean3":
                safe_mean(
                    prior3[
                        "avg_diff"
                    ]
                ),

            "store_avg_diff_mean5":
                safe_mean(
                    prior5[
                        "avg_diff"
                    ]
                ),

            "store_avg_diff_std5":
                safe_std(
                    prior5[
                        "avg_diff"
                    ]
                ),

            "store_positive_rate_last":
                safe_last(
                    prior_all[
                        "positive_diff_rate_percent"
                    ]
                ),

            "store_positive_rate_mean3":
                safe_mean(
                    prior3[
                        "positive_diff_rate_percent"
                    ]
                ),

            "store_positive_rate_mean5":
                safe_mean(
                    prior5[
                        "positive_diff_rate_percent"
                    ]
                ),

            "store_total_diff_last":
                safe_last(
                    prior_all[
                        "total_diff"
                    ]
                ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MACHINE-GROUP FEATURES
# ============================================================

def build_machine_features(
    machine: pd.DataFrame,
) -> pd.DataFrame:

    df = machine.copy()

    df[
        "machine_key"
    ] = df[
        "machine_name"
    ].map(
        normalize_machine_name
    )

    target_dates = (
        df[
            "date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    rows = []

    for target_date in target_dates:
        today = df[
            df[
                "date"
            ]
            == target_date
        ].copy()

        for _, current in today.iterrows():
            key = current[
                "machine_key"
            ]

            hist = df[
                (
                    df[
                        "machine_key"
                    ]
                    == key
                )
                & (
                    df[
                        "date"
                    ]
                    < target_date
                )
            ].sort_values(
                "date"
            )

            hist3 = hist.tail(3)
            hist5 = hist.tail(5)

            rows.append(
                {
                    "date":
                        target_date,

                    "machine_name":
                        current[
                            "machine_name"
                        ],

                    "machine_key":
                        key,

                    "machine_hist_days":
                        len(hist),

                    "machine_avg_diff_last":
                        safe_last(
                            hist[
                                "avg_diff"
                            ]
                        ),

                    "machine_avg_diff_mean3":
                        safe_mean(
                            hist3[
                                "avg_diff"
                            ]
                        ),

                    "machine_avg_diff_mean5":
                        safe_mean(
                            hist5[
                                "avg_diff"
                            ]
                        ),

                    "machine_win_rate_last":
                        safe_last(
                            hist[
                                "win_rate_percent"
                            ]
                        ),

                    "machine_win_rate_mean3":
                        safe_mean(
                            hist3[
                                "win_rate_percent"
                            ]
                        ),

                    "machine_win_rate_mean5":
                        safe_mean(
                            hist5[
                                "win_rate_percent"
                            ]
                        ),

                    "machine_count_last":
                        safe_last(
                            hist[
                                "machine_count"
                            ]
                        ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# TAIL FEATURES
# ============================================================

def build_tail_features(
    tail: pd.DataFrame,
) -> pd.DataFrame:

    target_dates = (
        tail[
            "date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    rows = []

    for target_date in target_dates:
        today = tail[
            tail[
                "date"
            ]
            == target_date
        ].copy()

        for _, current in today.iterrows():
            tail_no = int(
                current[
                    "tail"
                ]
            )

            hist = tail[
                (
                    tail[
                        "tail"
                    ]
                    == tail_no
                )
                & (
                    tail[
                        "date"
                    ]
                    < target_date
                )
            ].sort_values(
                "date"
            )

            hist3 = hist.tail(3)
            hist5 = hist.tail(5)

            rows.append(
                {
                    "date":
                        target_date,

                    "tail":
                        tail_no,

                    "tail_hist_days":
                        len(hist),

                    "tail_avg_diff_last":
                        safe_last(
                            hist[
                                "avg_diff"
                            ]
                        ),

                    "tail_avg_diff_mean3":
                        safe_mean(
                            hist3[
                                "avg_diff"
                            ]
                        ),

                    "tail_avg_diff_mean5":
                        safe_mean(
                            hist5[
                                "avg_diff"
                            ]
                        ),

                    "tail_win_rate_last":
                        safe_last(
                            hist[
                                "win_rate_percent"
                            ]
                        ),

                    "tail_win_rate_mean3":
                        safe_mean(
                            hist3[
                                "win_rate_percent"
                            ]
                        ),

                    "tail_win_rate_mean5":
                        safe_mean(
                            hist5[
                                "win_rate_percent"
                            ]
                        ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# DIAGNOSTICS
# ============================================================

def coverage_report(
    store_f: pd.DataFrame,
    machine_f: pd.DataFrame,
    tail_f: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    dates = sorted(
        set(
            store_f[
                "date"
            ].tolist()
        )
    )

    for date in dates:
        s = store_f[
            store_f[
                "date"
            ]
            == date
        ]

        m = machine_f[
            machine_f[
                "date"
            ]
            == date
        ]

        t = tail_f[
            tail_f[
                "date"
            ]
            == date
        ]

        rows.append(
            {
                "date":
                    date,

                "store_hist_days":
                    int(
                        s[
                            "store_hist_days"
                        ].iloc[0]
                    )
                    if len(s)
                    else 0,

                "machine_rows":
                    len(m),

                "machine_rows_with_history":
                    int(
                        (
                            m[
                                "machine_hist_days"
                            ]
                            > 0
                        ).sum()
                    )
                    if len(m)
                    else 0,

                "tail_rows":
                    len(t),

                "tail_rows_with_history":
                    int(
                        (
                            t[
                                "tail_hist_days"
                            ]
                            > 0
                        ).sum()
                    )
                    if len(t)
                    else 0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "SlotAnalyzer - Min-Repo Leakage-Free Historical Features"
    )

    store = normalize_date_col(
        read_csv(
            STORE_FILE
        )
    )

    machine = normalize_date_col(
        read_csv(
            MACHINE_FILE
        )
    )

    tail = normalize_date_col(
        read_csv(
            TAIL_FILE
        )
    )

    print(
        f"store dates            : {store['date'].nunique()}"
    )

    print(
        f"machine dates          : {machine['date'].nunique()}"
    )

    print(
        f"tail dates             : {tail['date'].nunique()}"
    )

    store_f = build_store_features(
        store
    )

    machine_f = build_machine_features(
        machine
    )

    tail_f = build_tail_features(
        tail
    )

    coverage = coverage_report(
        store_f,
        machine_f,
        tail_f,
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    store_path = (
        OUT_DIR
        / "store_history_features.csv"
    )

    machine_path = (
        OUT_DIR
        / "machine_history_features.csv"
    )

    tail_path = (
        OUT_DIR
        / "tail_history_features.csv"
    )

    coverage_path = (
        OUT_DIR
        / "history_feature_coverage.csv"
    )

    store_f.to_csv(
        store_path,
        index=False,
        encoding="utf-8-sig",
    )

    machine_f.to_csv(
        machine_path,
        index=False,
        encoding="utf-8-sig",
    )

    tail_f.to_csv(
        tail_path,
        index=False,
        encoding="utf-8-sig",
    )

    coverage.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "COVERAGE"
    )

    show = coverage.copy()

    show["date"] = (
        pd.to_datetime(
            show[
                "date"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    print(
        show.to_string(
            index=False
        )
    )

    header(
        "FILES SAVED"
    )

    for path in (
        store_path,
        machine_path,
        tail_path,
        coverage_path,
    ):
        print(path)

    print()
    print(
        "LEAKAGE CHECK: every feature uses date < target_date only."
    )

    print(
        "Existing V4.2_C and Ana-Slo files were not modified."
    )

    print()
    print(
        "Important: with only 8 collected dates, this is still a "
        "feature-construction / coverage test, not enough evidence "
        "to promote any external feature into production."
    )


if __name__ == "__main__":
    main()
