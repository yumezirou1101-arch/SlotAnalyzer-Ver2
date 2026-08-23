from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

INPUT_CSV = (
    DATA_DIR
    / "ana_slo_20260711_20260818.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "51_Ver4_2_machine_change_predictive_diagnostic"
)

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-18")

PRE_WINDOWS = [3, 7, 14]
POST_WINDOWS = [1, 3, 7]

# The largest machine-change event in this sample.
# A sensitivity analysis excluding this date is always produced.
LARGE_EVENT_DATE = pd.Timestamp("2026-08-03")


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in (
        "utf-8-sig",
        "utf-8",
        "cp932",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=enc,
            )
        except Exception:
            pass

    raise RuntimeError(
        f"CSV read failed: {path}"
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_corr(
    x: pd.Series,
    y: pd.Series,
    method: str,
) -> float:
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(pair) < 3:
        return np.nan

    if pair["x"].nunique() < 2:
        return np.nan

    if pair["y"].nunique() < 2:
        return np.nan

    if method == "pearson":
        return float(
            pair["x"].corr(
                pair["y"],
                method="pearson",
            )
        )

    if method == "spearman":
        # SciPy-free Spearman correlation:
        # rank both variables, then calculate Pearson correlation
        # between the ranks.
        x_rank = pair["x"].rank(
            method="average"
        )

        y_rank = pair["y"].rank(
            method="average"
        )

        return float(
            x_rank.corr(
                y_rank,
                method="pearson",
            )
        )

    raise ValueError(
        f"Unsupported correlation method: {method}"
    )


def mean_or_nan(
    series: pd.Series,
) -> float:
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if s.empty:
        return np.nan

    return float(
        s.mean()
    )


def median_or_nan(
    series: pd.Series,
) -> float:
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if s.empty:
        return np.nan

    return float(
        s.median()
    )


def sum_or_nan(
    series: pd.Series,
) -> float:
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if s.empty:
        return np.nan

    return float(
        s.sum()
    )


def pct_positive(
    series: pd.Series,
) -> float:
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if s.empty:
        return np.nan

    return float(
        (s > 0).mean()
        * 100.0
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = read_csv_flexible(
        INPUT_CSV
    )

    date_col = find_column(
        df,
        [
            "date",
            "\u65e5\u4ed8",
        ],
    )

    no_col = find_column(
        df,
        [
            "machine_no",
            "\u53f0\u756a\u53f7",
        ],
    )

    name_col = find_column(
        df,
        [
            "machine_name",
            "\u6a5f\u7a2e\u540d",
        ],
    )

    diff_col = find_column(
        df,
        [
            "diff",
            "\u5dee\u679a",
        ],
    )

    if not all(
        [
            date_col,
            no_col,
            name_col,
            diff_col,
        ]
    ):
        raise ValueError(
            "Required columns not found: "
            f"date={date_col}, "
            f"machine_no={no_col}, "
            f"machine_name={name_col}, "
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
        errors="coerce",
    )

    df["machine_no"] = pd.to_numeric(
        df["machine_no"],
        errors="coerce",
    )

    df["diff"] = (
        df["diff"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "+",
            "",
            regex=False,
        )
        .str.strip()
    )

    df["diff"] = pd.to_numeric(
        df["diff"],
        errors="coerce",
    )

    df["machine_name"] = (
        df["machine_name"]
        .astype(str)
        .str.strip()
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

    df = df[
        (df["date"] >= START)
        & (df["date"] <= END)
    ].copy()

    df = (
        df.sort_values(
            [
                "machine_no",
                "date",
            ]
        )
        .drop_duplicates(
            [
                "date",
                "machine_no",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# EVENT DETECTION
# ============================================================

def detect_machine_changes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    x = df.sort_values(
        [
            "machine_no",
            "date",
        ]
    ).copy()

    x["previous_machine_name"] = (
        x.groupby(
            "machine_no"
        )["machine_name"]
        .shift(1)
    )

    x["previous_date"] = (
        x.groupby(
            "machine_no"
        )["date"]
        .shift(1)
    )

    changes = x[
        x[
            "previous_machine_name"
        ].notna()
        & (
            x["machine_name"]
            != x[
                "previous_machine_name"
            ]
        )
    ].copy()

    changes = changes.rename(
        columns={
            "date":
                "change_date",
            "machine_name":
                "new_machine_name",
            "previous_machine_name":
                "old_machine_name",
            "diff":
                "change_day_diff",
        }
    )

    changes = changes[
        [
            "change_date",
            "machine_no",
            "old_machine_name",
            "new_machine_name",
            "previous_date",
            "change_day_diff",
        ]
    ].sort_values(
        [
            "change_date",
            "machine_no",
        ]
    )

    changes = changes.reset_index(
        drop=True
    )

    changes["event_id"] = (
        np.arange(
            len(changes)
        )
        + 1
    )

    return changes


# ============================================================
# EVENT-LEVEL FEATURE TABLE
# ============================================================

def build_event_table(
    df: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    by_no = {
        int(no): g.sort_values(
            "date"
        ).copy()
        for no, g in df.groupby(
            "machine_no"
        )
    }

    rows = []

    for event in changes.itertuples(
        index=False
    ):
        no = int(
            event.machine_no
        )

        change_date = pd.Timestamp(
            event.change_date
        )

        old_name = str(
            event.old_machine_name
        )

        new_name = str(
            event.new_machine_name
        )

        m = by_no.get(
            no
        )

        if m is None or m.empty:
            continue

        pre_all = m[
            m["date"] < change_date
        ].copy()

        pre_old = pre_all[
            pre_all["machine_name"]
            == old_name
        ].copy()

        # Post-change segment:
        # include the change day and stop before the next machine-name change.
        post_all = m[
            m["date"] >= change_date
        ].copy()

        post_segment = []

        for row in post_all.itertuples(
            index=False
        ):
            if str(
                row.machine_name
            ) != new_name:
                break

            post_segment.append(
                row
            )

        if post_segment:
            post_df = pd.DataFrame(
                [
                    r._asdict()
                    for r in post_segment
                ]
            )
        else:
            post_df = pd.DataFrame(
                columns=m.columns
            )

        result = {
            "event_id":
                int(
                    event.event_id
                ),

            "change_date":
                change_date,

            "machine_no":
                no,

            "old_machine_name":
                old_name,

            "new_machine_name":
                new_name,

            "previous_date":
                pd.Timestamp(
                    event.previous_date
                ),

            "pre_all_n":
                int(
                    len(pre_all)
                ),

            "pre_old_n":
                int(
                    len(pre_old)
                ),

            "post_segment_n":
                int(
                    len(post_df)
                ),

            "change_day_diff":
                float(
                    event.change_day_diff
                ),
        }

        if not pre_old.empty:
            result[
                "pre_old_avg_all"
            ] = float(
                pre_old["diff"].mean()
            )

            result[
                "pre_old_total_all"
            ] = float(
                pre_old["diff"].sum()
            )

            result[
                "pre_old_positive_rate_all"
            ] = float(
                (
                    pre_old["diff"]
                    > 0
                ).mean()
                * 100.0
            )

        else:
            result[
                "pre_old_avg_all"
            ] = np.nan
            result[
                "pre_old_total_all"
            ] = np.nan
            result[
                "pre_old_positive_rate_all"
            ] = np.nan

        for window in PRE_WINDOWS:
            pre_w = pre_old.tail(
                window
            )

            result[
                f"pre_old_avg_{window}"
            ] = mean_or_nan(
                pre_w["diff"]
                if not pre_w.empty
                else pd.Series(
                    dtype=float
                )
            )

            result[
                f"pre_old_total_{window}"
            ] = sum_or_nan(
                pre_w["diff"]
                if not pre_w.empty
                else pd.Series(
                    dtype=float
                )
            )

            result[
                f"pre_old_positive_rate_{window}"
            ] = pct_positive(
                pre_w["diff"]
                if not pre_w.empty
                else pd.Series(
                    dtype=float
                )
            )

        for window in POST_WINDOWS:
            post_w = post_df.head(
                window
            )

            result[
                f"post_avg_{window}"
            ] = mean_or_nan(
                post_w["diff"]
                if not post_w.empty
                else pd.Series(
                    dtype=float
                )
            )

            result[
                f"post_total_{window}"
            ] = sum_or_nan(
                post_w["diff"]
                if not post_w.empty
                else pd.Series(
                    dtype=float
                )
            )

            result[
                f"post_positive_rate_{window}"
            ] = pct_positive(
                post_w["diff"]
                if not post_w.empty
                else pd.Series(
                    dtype=float
                )
            )

        result[
            "post_avg_full_segment"
        ] = mean_or_nan(
            post_df["diff"]
            if not post_df.empty
            else pd.Series(
                dtype=float
            )
        )

        result[
            "post_total_full_segment"
        ] = sum_or_nan(
            post_df["diff"]
            if not post_df.empty
            else pd.Series(
                dtype=float
            )
        )

        result[
            "post_positive_rate_full_segment"
        ] = pct_positive(
            post_df["diff"]
            if not post_df.empty
            else pd.Series(
                dtype=float
            )
        )

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CORRELATION DIAGNOSTICS
# ============================================================

def build_correlation_table(
    events: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    pre_metrics = [
        "pre_old_avg_all",
        "pre_old_avg_3",
        "pre_old_avg_7",
        "pre_old_avg_14",
        "pre_old_positive_rate_all",
        "pre_old_positive_rate_3",
        "pre_old_positive_rate_7",
        "pre_old_positive_rate_14",
    ]

    post_metrics = [
        "change_day_diff",
        "post_avg_3",
        "post_avg_7",
        "post_avg_full_segment",
    ]

    rows = []

    for pre_col in pre_metrics:
        if pre_col not in events.columns:
            continue

        for post_col in post_metrics:
            if post_col not in events.columns:
                continue

            valid = events[
                [
                    pre_col,
                    post_col,
                ]
            ].dropna()

            rows.append(
                {
                    "sample":
                        label,

                    "pre_metric":
                        pre_col,

                    "post_metric":
                        post_col,

                    "n":
                        int(
                            len(valid)
                        ),

                    "pearson":
                        safe_corr(
                            events[
                                pre_col
                            ],
                            events[
                                post_col
                            ],
                            "pearson",
                        ),

                    "spearman":
                        safe_corr(
                            events[
                                pre_col
                            ],
                            events[
                                post_col
                            ],
                            "spearman",
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# HIGH / LOW PRE-PERFORMANCE COMPARISON
# ============================================================

def build_group_comparison(
    events: pd.DataFrame,
    label: str,
    pre_col: str,
    post_col: str,
) -> pd.DataFrame:
    x = events[
        [
            "change_date",
            "machine_no",
            pre_col,
            post_col,
        ]
    ].dropna().copy()

    if len(x) < 4:
        return pd.DataFrame()

    threshold = float(
        x[pre_col].median()
    )

    x["pre_group"] = np.where(
        x[pre_col] >= threshold,
        "HIGH_PRE",
        "LOW_PRE",
    )

    rows = []

    for group_name, g in x.groupby(
        "pre_group"
    ):
        rows.append(
            {
                "sample":
                    label,

                "pre_metric":
                    pre_col,

                "post_metric":
                    post_col,

                "median_threshold":
                    threshold,

                "group":
                    group_name,

                "n":
                    int(
                        len(g)
                    ),

                "pre_mean":
                    float(
                        g[
                            pre_col
                        ].mean()
                    ),

                "post_mean":
                    float(
                        g[
                            post_col
                        ].mean()
                    ),

                "post_median":
                    float(
                        g[
                            post_col
                        ].median()
                    ),

                "post_positive_rate":
                    float(
                        (
                            g[
                                post_col
                            ]
                            > 0
                        ).mean()
                        * 100.0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# EVENT-DATE AGGREGATION
# ============================================================

def build_event_date_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    agg_map = {
        "event_id":
            "count",

        "pre_old_avg_all":
            "mean",

        "pre_old_avg_3":
            "mean",

        "pre_old_avg_7":
            "mean",

        "change_day_diff":
            "mean",

        "post_avg_3":
            "mean",

        "post_avg_7":
            "mean",

        "post_avg_full_segment":
            "mean",
    }

    available = {
        k: v
        for k, v in agg_map.items()
        if k in events.columns
    }

    out = (
        events.groupby(
            "change_date",
            as_index=False
        )
        .agg(
            available
        )
    )

    out = out.rename(
        columns={
            "event_id":
                "changed_machines",
        }
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_header(
        "Ana-Slo Ver.4.2 Machine-Change Predictive Diagnostic"
    )

    df = load_data()

    print(
        f"records              : {len(df):,}"
    )
    print(
        f"days                 : {df['date'].nunique()}"
    )
    print(
        f"unique machine nos   : {df['machine_no'].nunique()}"
    )
    print(
        f"date range           : "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    changes = detect_machine_changes(
        df
    )

    print_header(
        "MACHINE CHANGE EVENTS"
    )

    print(
        f"change observations  : {len(changes)}"
    )
    print(
        f"changed machine nos  : "
        f"{changes['machine_no'].nunique()}"
    )
    print(
        f"change dates         : "
        f"{changes['change_date'].nunique()}"
    )

    by_date = (
        changes.groupby(
            "change_date"
        )
        .size()
        .reset_index(
            name="changes"
        )
    )

    print()
    print(
        by_date.to_string(
            index=False
        )
    )

    events = build_event_table(
        df,
        changes,
    )

    print_header(
        "EVENT TABLE SUMMARY"
    )

    print(
        f"event rows           : {len(events)}"
    )

    if not events.empty:
        print(
            f"pre history min/max  : "
            f"{events['pre_old_n'].min()} / "
            f"{events['pre_old_n'].max()}"
        )

        print(
            f"post segment min/max : "
            f"{events['post_segment_n'].min()} / "
            f"{events['post_segment_n'].max()}"
        )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    corr_all = build_correlation_table(
        events,
        "ALL_EVENTS",
    )

    events_ex_large = events[
        events["change_date"]
        != LARGE_EVENT_DATE
    ].copy()

    corr_ex_large = build_correlation_table(
        events_ex_large,
        "EXCLUDE_2026_08_03",
    )

    corr_df = pd.concat(
        [
            corr_all,
            corr_ex_large,
        ],
        ignore_index=True,
    )

    print_header(
        "CORRELATION DIAGNOSTIC"
    )

    display_corr = corr_df[
        corr_df[
            "pre_metric"
        ].isin(
            [
                "pre_old_avg_all",
                "pre_old_avg_7",
            ]
        )
        & corr_df[
            "post_metric"
        ].isin(
            [
                "change_day_diff",
                "post_avg_3",
                "post_avg_7",
                "post_avg_full_segment",
            ]
        )
    ].copy()

    if display_corr.empty:
        print(
            "No valid correlation rows."
        )
    else:
        print(
            display_corr.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # High-vs-low comparison
    # --------------------------------------------------------

    group_frames = []

    for label, sample in (
        (
            "ALL_EVENTS",
            events,
        ),
        (
            "EXCLUDE_2026_08_03",
            events_ex_large,
        ),
    ):
        for pre_col in (
            "pre_old_avg_all",
            "pre_old_avg_7",
        ):
            for post_col in (
                "change_day_diff",
                "post_avg_3",
                "post_avg_7",
                "post_avg_full_segment",
            ):
                g = build_group_comparison(
                    sample,
                    label,
                    pre_col,
                    post_col,
                )

                if not g.empty:
                    group_frames.append(
                        g
                    )

    if group_frames:
        group_df = pd.concat(
            group_frames,
            ignore_index=True,
        )
    else:
        group_df = pd.DataFrame()

    print_header(
        "HIGH PRE vs LOW PRE"
    )

    key_groups = group_df[
        (
            group_df[
                "pre_metric"
            ]
            == "pre_old_avg_7"
        )
        & group_df[
            "post_metric"
        ].isin(
            [
                "change_day_diff",
                "post_avg_3",
                "post_avg_7",
            ]
        )
    ].copy()

    if key_groups.empty:
        print(
            "No valid high/low comparison."
        )
    else:
        print(
            key_groups.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Event-date summary
    # --------------------------------------------------------

    event_date_df = (
        build_event_date_summary(
            events
        )
    )

    print_header(
        "EVENT-DATE SUMMARY"
    )

    if event_date_df.empty:
        print(
            "No event-date summary."
        )
    else:
        print(
            event_date_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Assessment
    # --------------------------------------------------------

    print_header(
        "ASSESSMENT"
    )

    event_dates = int(
        changes[
            "change_date"
        ].nunique()
    )

    largest_event = int(
        by_date[
            "changes"
        ].max()
    ) if not by_date.empty else 0

    largest_share = (
        largest_event
        / len(changes)
        * 100.0
        if len(changes) > 0
        else 0.0
    )

    if event_dates < 8:
        status = (
            "INSUFFICIENT_INDEPENDENT_CHANGE_EVENTS"
        )
    else:
        status = (
            "EXPLORATORY_SIGNAL_CHECK"
        )

    print(
        f"status               : {status}"
    )
    print(
        f"independent dates    : {event_dates}"
    )
    print(
        f"largest event size   : {largest_event}"
    )
    print(
        f"largest event share  : "
        f"{largest_share:.2f}%"
    )
    print()
    print(
        "Interpret correlations as exploratory only. "
        "Rows within the same change date are clustered and "
        "are not independent observations."
    )
    print(
        "The 2026-08-03 exclusion analysis is especially important "
        "because that single event contains most of the changes."
    )
    print(
        "Do not adopt or tune a production machine-change rule "
        "from this diagnostic alone."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_changes = (
        OUTPUT_DIR
        / "51_machine_change_events.csv"
    )

    out_events = (
        OUTPUT_DIR
        / "51_machine_change_event_features.csv"
    )

    out_corr = (
        OUTPUT_DIR
        / "51_machine_change_correlations.csv"
    )

    out_groups = (
        OUTPUT_DIR
        / "51_machine_change_high_low_comparison.csv"
    )

    out_dates = (
        OUTPUT_DIR
        / "51_machine_change_event_date_summary.csv"
    )

    out_assessment = (
        OUTPUT_DIR
        / "51_machine_change_assessment.csv"
    )

    changes.to_csv(
        out_changes,
        index=False,
        encoding="utf-8-sig",
    )

    events.to_csv(
        out_events,
        index=False,
        encoding="utf-8-sig",
    )

    corr_df.to_csv(
        out_corr,
        index=False,
        encoding="utf-8-sig",
    )

    group_df.to_csv(
        out_groups,
        index=False,
        encoding="utf-8-sig",
    )

    event_date_df.to_csv(
        out_dates,
        index=False,
        encoding="utf-8-sig",
    )

    assessment_df = pd.DataFrame(
        [
            {
                "status":
                    status,

                "change_observations":
                    int(
                        len(changes)
                    ),

                "changed_machine_nos":
                    int(
                        changes[
                            "machine_no"
                        ].nunique()
                    ),

                "independent_change_dates":
                    event_dates,

                "largest_event_size":
                    largest_event,

                "largest_event_share_pct":
                    largest_share,

                "excluded_large_event_date":
                    LARGE_EVENT_DATE.date(),
            }
        ]
    )

    assessment_df.to_csv(
        out_assessment,
        index=False,
        encoding="utf-8-sig",
    )

    print_header(
        "FILES SAVED"
    )

    for path in (
        out_changes,
        out_events,
        out_corr,
        out_groups,
        out_dates,
        out_assessment,
    ):
        print(path)

    print()
    print(
        "Machine-change predictive diagnostic complete."
    )


if __name__ == "__main__":
    main()
