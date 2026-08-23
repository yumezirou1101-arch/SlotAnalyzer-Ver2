from __future__ import annotations

from pathlib import Path

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
    / "52_Ver4_2_machine_number_island_structure_diagnostic"
)

START = pd.Timestamp("2026-07-11")
END = pd.Timestamp("2026-08-18")

# A numeric jump larger than this is highlighted as a strong break candidate.
LARGE_GAP_THRESHOLD = 5


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


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


# ============================================================
# DATA
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
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
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
                "date",
                "machine_no",
            ]
        )
        .drop_duplicates(
            [
                "date",
                "machine_no",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# NUMBER STRUCTURE
# ============================================================

def build_number_structure(
    latest_day: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = (
        latest_day[
            [
                "machine_no",
                "machine_name",
            ]
        ]
        .sort_values("machine_no")
        .drop_duplicates(
            "machine_no",
            keep="last",
        )
        .reset_index(drop=True)
        .copy()
    )

    x["prev_machine_no"] = (
        x["machine_no"]
        .shift(1)
    )

    x["next_machine_no"] = (
        x["machine_no"]
        .shift(-1)
    )

    x["gap_from_prev"] = (
        x["machine_no"]
        - x["prev_machine_no"]
    )

    x["gap_to_next"] = (
        x["next_machine_no"]
        - x["machine_no"]
    )

    machine_set = set(
        x["machine_no"]
        .astype(int)
        .tolist()
    )

    x["has_minus1"] = (
        x["machine_no"]
        .apply(
            lambda n:
                int(n - 1 in machine_set)
        )
    )

    x["has_plus1"] = (
        x["machine_no"]
        .apply(
            lambda n:
                int(n + 1 in machine_set)
        )
    )

    x["neighbor_count_pm1"] = (
        x["has_minus1"]
        + x["has_plus1"]
    )

    x["same_name_prev"] = (
        (
            x["machine_name"]
            == x["machine_name"].shift(1)
        )
        & (
            x["gap_from_prev"]
            == 1
        )
    ).astype(int)

    x["same_name_next"] = (
        (
            x["machine_name"]
            == x["machine_name"].shift(-1)
        )
        & (
            x["gap_to_next"]
            == 1
        )
    ).astype(int)

    x["machine_name_boundary"] = (
        (
            x["same_name_prev"] == 0
        )
        | (
            x["same_name_next"] == 0
        )
    ).astype(int)

    # Consecutive-number run ID.
    run_break = (
        x["gap_from_prev"]
        .fillna(1)
        .ne(1)
    )

    x["number_run_id"] = (
        run_break.cumsum()
    )

    runs = (
        x.groupby(
            "number_run_id",
            as_index=False,
        )
        .agg(
            run_start=(
                "machine_no",
                "min",
            ),
            run_end=(
                "machine_no",
                "max",
            ),
            machines=(
                "machine_no",
                "count",
            ),
        )
    )

    runs["span"] = (
        runs["run_end"]
        - runs["run_start"]
        + 1
    )

    runs["fully_consecutive"] = (
        runs["span"]
        == runs["machines"]
    )

    return x, runs


def build_gap_table(
    structure: pd.DataFrame,
) -> pd.DataFrame:
    gaps = structure[
        structure["gap_from_prev"]
        .fillna(1)
        .gt(1)
    ].copy()

    if gaps.empty:
        return pd.DataFrame(
            columns=[
                "left_machine_no",
                "right_machine_no",
                "gap_size",
                "missing_count",
                "left_machine_name",
                "right_machine_name",
                "large_gap_candidate",
            ]
        )

    rows = []

    for row in gaps.itertuples(
        index=False
    ):
        right_no = int(
            row.machine_no
        )

        left_no = int(
            row.prev_machine_no
        )

        left_name_series = structure.loc[
            structure["machine_no"]
            == left_no,
            "machine_name",
        ]

        left_name = (
            str(
                left_name_series.iloc[0]
            )
            if not left_name_series.empty
            else ""
        )

        rows.append(
            {
                "left_machine_no":
                    left_no,

                "right_machine_no":
                    right_no,

                "gap_size":
                    int(
                        right_no
                        - left_no
                    ),

                "missing_count":
                    int(
                        right_no
                        - left_no
                        - 1
                    ),

                "left_machine_name":
                    left_name,

                "right_machine_name":
                    str(
                        row.machine_name
                    ),

                "large_gap_candidate":
                    int(
                        (
                            right_no
                            - left_no
                        )
                        > LARGE_GAP_THRESHOLD
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MACHINE-NAME BLOCKS
# ============================================================

def build_machine_name_blocks(
    structure: pd.DataFrame,
) -> pd.DataFrame:
    x = structure[
        [
            "machine_no",
            "machine_name",
        ]
    ].copy()

    prev_no = (
        x["machine_no"]
        .shift(1)
    )

    prev_name = (
        x["machine_name"]
        .shift(1)
    )

    block_break = (
        prev_no.isna()
        | (
            x["machine_no"]
            - prev_no
            != 1
        )
        | (
            x["machine_name"]
            != prev_name
        )
    )

    x["machine_block_id"] = (
        block_break.cumsum()
    )

    out = (
        x.groupby(
            "machine_block_id",
            as_index=False,
        )
        .agg(
            block_start=(
                "machine_no",
                "min",
            ),
            block_end=(
                "machine_no",
                "max",
            ),
            machines=(
                "machine_no",
                "count",
            ),
            machine_name=(
                "machine_name",
                "first",
            ),
        )
    )

    return out


# ============================================================
# CURRENT +/-1 NEIGHBOR DIAGNOSTIC
# ============================================================

def build_neighbor_diagnostic(
    structure: pd.DataFrame,
) -> pd.DataFrame:
    x = structure.copy()

    x["pm1_status"] = np.select(
        [
            (
                x["has_minus1"] == 1
            )
            & (
                x["has_plus1"] == 1
            ),
            (
                x["has_minus1"] == 1
            )
            | (
                x["has_plus1"] == 1
            ),
        ],
        [
            "TWO_PM1_NEIGHBORS",
            "ONE_PM1_NEIGHBOR",
        ],
        default="NO_PM1_NEIGHBOR",
    )

    x["potential_block_edge"] = (
        (
            x["gap_from_prev"]
            .fillna(1)
            .ne(1)
        )
        | (
            x["gap_to_next"]
            .fillna(1)
            .ne(1)
        )
    ).astype(int)

    return x


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_header(
        "Ana-Slo Ver.4.2 Machine-Number / Island-Structure Diagnostic"
    )

    df = load_data()

    latest_date = (
        df["date"].max()
    )

    latest_day = df[
        df["date"] == latest_date
    ].copy()

    print(
        f"records              : {len(df):,}"
    )
    print(
        f"days                 : {df['date'].nunique()}"
    )
    print(
        f"latest date          : {latest_date.date()}"
    )
    print(
        f"latest machines      : {latest_day['machine_no'].nunique()}"
    )

    structure, runs = (
        build_number_structure(
            latest_day
        )
    )

    gaps = build_gap_table(
        structure
    )

    machine_blocks = (
        build_machine_name_blocks(
            structure
        )
    )

    neighbor_diag = (
        build_neighbor_diagnostic(
            structure
        )
    )

    print_header(
        "NUMBER RANGE SUMMARY"
    )

    print(
        f"min machine no       : "
        f"{int(structure['machine_no'].min())}"
    )
    print(
        f"max machine no       : "
        f"{int(structure['machine_no'].max())}"
    )
    print(
        f"consecutive runs     : {len(runs)}"
    )
    print(
        f"numeric gaps         : {len(gaps)}"
    )

    if not gaps.empty:
        print(
            f"large gaps >{LARGE_GAP_THRESHOLD}: "
            f"{int(gaps['large_gap_candidate'].sum())}"
        )

    print_header(
        "CONSECUTIVE NUMBER RUNS"
    )

    print(
        runs.to_string(
            index=False
        )
    )

    print_header(
        "NUMBER GAPS"
    )

    if gaps.empty:
        print(
            "No numeric gaps."
        )
    else:
        print(
            gaps.to_string(
                index=False
            )
        )

    print_header(
        "CURRENT +/-1 NEIGHBOR COVERAGE"
    )

    coverage = (
        neighbor_diag[
            "pm1_status"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "status"
        )
        .reset_index(
            name="machines"
        )
    )

    coverage["rate_pct"] = (
        coverage["machines"]
        / len(neighbor_diag)
        * 100.0
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    no_two_neighbors = neighbor_diag[
        neighbor_diag[
            "neighbor_count_pm1"
        ] < 2
    ][
        [
            "machine_no",
            "machine_name",
            "prev_machine_no",
            "next_machine_no",
            "gap_from_prev",
            "gap_to_next",
            "has_minus1",
            "has_plus1",
            "neighbor_count_pm1",
        ]
    ].copy()

    print()
    print(
        f"machines without two +/-1 neighbors: "
        f"{len(no_two_neighbors)} / {len(neighbor_diag)} "
        f"({len(no_two_neighbors) / len(neighbor_diag) * 100:.2f}%)"
    )

    print_header(
        "MACHINE-NAME CONTIGUOUS BLOCKS"
    )

    print(
        machine_blocks.to_string(
            index=False
        )
    )

    print_header(
        "STRUCTURAL ASSESSMENT"
    )

    no_neighbor_count = int(
        (
            neighbor_diag[
                "neighbor_count_pm1"
            ]
            == 0
        ).sum()
    )

    one_neighbor_count = int(
        (
            neighbor_diag[
                "neighbor_count_pm1"
            ]
            == 1
        ).sum()
    )

    two_neighbor_count = int(
        (
            neighbor_diag[
                "neighbor_count_pm1"
            ]
            == 2
        ).sum()
    )

    print(
        f"two +/-1 neighbors   : {two_neighbor_count}"
    )
    print(
        f"one +/-1 neighbor    : {one_neighbor_count}"
    )
    print(
        f"zero +/-1 neighbors  : {no_neighbor_count}"
    )
    print(
        f"numeric gap count    : {len(gaps)}"
    )
    print(
        f"machine-name blocks  : {len(machine_blocks)}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This diagnostic detects only number structure and machine-name "
        "contiguity. It does NOT prove real physical island adjacency."
    )
    print(
        "A consecutive machine number may still cross an island edge, "
        "aisle, or opposite row."
    )
    print(
        "A numeric gap may still represent physical adjacency if the "
        "store numbering scheme skips numbers."
    )
    print(
        "Do not replace the production neighbor feature until an OOS "
        "comparison is run."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_structure = (
        OUTPUT_DIR
        / "52_machine_number_structure.csv"
    )

    out_runs = (
        OUTPUT_DIR
        / "52_consecutive_number_runs.csv"
    )

    out_gaps = (
        OUTPUT_DIR
        / "52_number_gaps.csv"
    )

    out_blocks = (
        OUTPUT_DIR
        / "52_machine_name_blocks.csv"
    )

    out_neighbor = (
        OUTPUT_DIR
        / "52_pm1_neighbor_diagnostic.csv"
    )

    out_missing = (
        OUTPUT_DIR
        / "52_machines_without_two_pm1_neighbors.csv"
    )

    out_summary = (
        OUTPUT_DIR
        / "52_structure_summary.csv"
    )

    structure.to_csv(
        out_structure,
        index=False,
        encoding="utf-8-sig",
    )

    runs.to_csv(
        out_runs,
        index=False,
        encoding="utf-8-sig",
    )

    gaps.to_csv(
        out_gaps,
        index=False,
        encoding="utf-8-sig",
    )

    machine_blocks.to_csv(
        out_blocks,
        index=False,
        encoding="utf-8-sig",
    )

    neighbor_diag.to_csv(
        out_neighbor,
        index=False,
        encoding="utf-8-sig",
    )

    no_two_neighbors.to_csv(
        out_missing,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame(
        [
            {
                "latest_date":
                    latest_date.date(),

                "latest_machines":
                    int(
                        latest_day[
                            "machine_no"
                        ].nunique()
                    ),

                "min_machine_no":
                    int(
                        structure[
                            "machine_no"
                        ].min()
                    ),

                "max_machine_no":
                    int(
                        structure[
                            "machine_no"
                        ].max()
                    ),

                "consecutive_runs":
                    int(
                        len(runs)
                    ),

                "numeric_gaps":
                    int(
                        len(gaps)
                    ),

                "large_gaps":
                    int(
                        gaps[
                            "large_gap_candidate"
                        ].sum()
                    )
                    if not gaps.empty
                    else 0,

                "two_pm1_neighbors":
                    two_neighbor_count,

                "one_pm1_neighbor":
                    one_neighbor_count,

                "zero_pm1_neighbors":
                    no_neighbor_count,

                "machine_name_blocks":
                    int(
                        len(machine_blocks)
                    ),

                "physical_layout_confirmed":
                    False,
            }
        ]
    )

    summary_df.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig",
    )

    print_header(
        "FILES SAVED"
    )

    for path in (
        out_structure,
        out_runs,
        out_gaps,
        out_blocks,
        out_neighbor,
        out_missing,
        out_summary,
    ):
        print(path)

    print()
    print(
        "Machine-number / island-structure diagnostic complete."
    )


if __name__ == "__main__":
    main()
