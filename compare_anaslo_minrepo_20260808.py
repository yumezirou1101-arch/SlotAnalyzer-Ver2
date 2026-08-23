from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# Ana-Slo vs Min-Repo Cross Check
# Test date: 2026-08-08
# Store: マルハンメガシティ前橋インター
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

ANA_CSV = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "ana_slo_20260808.csv"
)

MINREPO_HTML = (
    PROJECT_ROOT
    / "minrepo_20260808_allmachines.html"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "external_validation"
    / "minrepo_20260808"
)

EXPECTED_ANA_ROWS = 514


# ============================================================
# DISPLAY
# ============================================================

def header(
    title: str,
) -> None:

    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


# ============================================================
# HELPERS
# ============================================================

def read_csv_flexible(
    path: Path,
) -> pd.DataFrame:

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


def parse_number(
    value,
) -> float:

    if pd.isna(
        value
    ):
        return np.nan

    text = (
        str(
            value
        )
        .strip()
        .replace(
            ",",
            "",
        )
        .replace(
            "+",
            "",
        )
    )

    # Keep optional leading minus and digits.
    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return np.nan

    try:
        return float(
            match.group(0)
        )

    except Exception:
        return np.nan


def normalize_machine_name(
    value,
) -> str:

    if pd.isna(
        value
    ):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(
            value
        ),
    )

    # Normalize spaces.
    text = re.sub(
        r"\s+",
        "",
        text,
    )

    # A few harmless typography differences.
    replacements = {
        "‐": "-",
        "‑": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "～": "~",
        "〜": "~",
        "・": "・",
    }

    for src, dst in replacements.items():
        text = text.replace(
            src,
            dst,
        )

    return text.strip()


# ============================================================
# ANA-SLO
# ============================================================

def load_anaslo() -> pd.DataFrame:

    if not ANA_CSV.exists():
        raise FileNotFoundError(
            f"Ana-Slo CSV not found: {ANA_CSV}"
        )

    df = read_csv_flexible(
        ANA_CSV
    )

    required = [
        "台番号",
        "機種名",
        "差枚",
        "G数",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Ana-Slo missing columns: {missing}"
        )

    x = df[
        required
    ].copy()

    x = x.rename(
        columns={
            "台番号":
                "machine_no",

            "機種名":
                "ana_machine_name",

            "差枚":
                "ana_diff",

            "G数":
                "ana_g",
        }
    )

    x[
        "machine_no"
    ] = pd.to_numeric(
        x[
            "machine_no"
        ],
        errors="coerce",
    )

    x[
        "ana_diff"
    ] = x[
        "ana_diff"
    ].map(
        parse_number
    )

    x[
        "ana_g"
    ] = x[
        "ana_g"
    ].map(
        parse_number
    )

    x = x.dropna(
        subset=[
            "machine_no",
        ]
    ).copy()

    x[
        "machine_no"
    ] = x[
        "machine_no"
    ].astype(
        int
    )

    x[
        "ana_machine_name_norm"
    ] = x[
        "ana_machine_name"
    ].map(
        normalize_machine_name
    )

    return x


# ============================================================
# MIN-REPO
# ============================================================

def load_minrepo() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not MINREPO_HTML.exists():
        raise FileNotFoundError(
            f"Min-Repo HTML not found: {MINREPO_HTML}"
        )

    tables = pd.read_html(
        MINREPO_HTML
    )

    candidates = []

    for table_index, table in enumerate(
        tables
    ):

        cols = [
            str(
                col
            )
            for col in table.columns
        ]

        if all(
            key in cols
            for key in (
                "機種",
                "台番",
                "差枚",
                "G数",
            )
        ):

            candidates.append(
                (
                    len(
                        table
                    ),
                    table_index,
                    table,
                )
            )

    if not candidates:
        raise RuntimeError(
            "Min-Repo all-machine table not found."
        )

    candidates.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    raw = candidates[
        0
    ][
        2
    ].copy()

    x = raw[
        [
            "機種",
            "台番",
            "差枚",
            "G数",
        ]
    ].copy()

    x = x.rename(
        columns={
            "機種":
                "minrepo_machine_name",

            "台番":
                "machine_no",

            "差枚":
                "minrepo_diff",

            "G数":
                "minrepo_g",
        }
    )

    x[
        "machine_no"
    ] = x[
        "machine_no"
    ].map(
        parse_number
    )

    x[
        "minrepo_diff"
    ] = x[
        "minrepo_diff"
    ].map(
        parse_number
    )

    x[
        "minrepo_g"
    ] = x[
        "minrepo_g"
    ].map(
        parse_number
    )

    # Keep only rows that truly look like machine rows.
    x = x.dropna(
        subset=[
            "machine_no",
        ]
    ).copy()

    x[
        "machine_no"
    ] = x[
        "machine_no"
    ].astype(
        int
    )

    x[
        "minrepo_machine_name_norm"
    ] = x[
        "minrepo_machine_name"
    ].map(
        normalize_machine_name
    )

    return (
        raw,
        x,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "Ana-Slo vs Min-Repo Cross Check - 2026-08-08"
    )

    ana = load_anaslo()

    minrepo_raw, minrepo = (
        load_minrepo()
    )

    print(
        f"Ana-Slo rows              : {len(ana)}"
    )

    print(
        f"Ana-Slo unique machines   : {ana['machine_no'].nunique()}"
    )

    print(
        f"Min-Repo raw table rows    : {len(minrepo_raw)}"
    )

    print(
        f"Min-Repo numeric rows      : {len(minrepo)}"
    )

    print(
        f"Min-Repo unique machines   : {minrepo['machine_no'].nunique()}"
    )

    ana_dup = int(
        ana.duplicated(
            subset=[
                "machine_no",
            ]
        ).sum()
    )

    minrepo_dup = int(
        minrepo.duplicated(
            subset=[
                "machine_no",
            ]
        ).sum()
    )

    print(
        f"Ana-Slo duplicate rows     : {ana_dup}"
    )

    print(
        f"Min-Repo duplicate rows    : {minrepo_dup}"
    )

    if (
        len(
            ana
        )
        != EXPECTED_ANA_ROWS
    ):

        print()
        print(
            f"[WARNING] Ana-Slo rows != {EXPECTED_ANA_ROWS}"
        )

    # Preserve duplicate diagnostics before deduplication.
    minrepo_duplicate_rows = (
        minrepo[
            minrepo.duplicated(
                subset=[
                    "machine_no",
                ],
                keep=False,
            )
        ]
        .sort_values(
            [
                "machine_no",
            ]
        )
        .copy()
    )

    # For comparison use the first row for each number;
    # duplicates are separately reported and must be reviewed.
    ana_one = (
        ana.drop_duplicates(
            subset=[
                "machine_no",
            ],
            keep="first",
        )
        .copy()
    )

    minrepo_one = (
        minrepo.drop_duplicates(
            subset=[
                "machine_no",
            ],
            keep="first",
        )
        .copy()
    )

    merged = ana_one.merge(
        minrepo_one,
        on="machine_no",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    merged[
        "present_ana"
    ] = (
        merged[
            "_merge"
        ]
        != "right_only"
    )

    merged[
        "present_minrepo"
    ] = (
        merged[
            "_merge"
        ]
        != "left_only"
    )

    both = merged[
        merged[
            "_merge"
        ]
        == "both"
    ].copy()

    both[
        "machine_name_exact"
    ] = (
        both[
            "ana_machine_name"
        ].astype(
            str
        )
        == both[
            "minrepo_machine_name"
        ].astype(
            str
        )
    )

    both[
        "machine_name_normalized"
    ] = (
        both[
            "ana_machine_name_norm"
        ]
        == both[
            "minrepo_machine_name_norm"
        ]
    )

    both[
        "diff_delta"
    ] = (
        both[
            "minrepo_diff"
        ]
        - both[
            "ana_diff"
        ]
    )

    both[
        "g_delta"
    ] = (
        both[
            "minrepo_g"
        ]
        - both[
            "ana_g"
        ]
    )

    both[
        "diff_exact"
    ] = (
        both[
            "diff_delta"
        ]
        == 0
    )

    both[
        "g_exact"
    ] = (
        both[
            "g_delta"
        ]
        == 0
    )

    both[
        "all_core_exact"
    ] = (
        both[
            "machine_name_normalized"
        ]
        & both[
            "diff_exact"
        ]
        & both[
            "g_exact"
        ]
    )

    ana_only = merged[
        merged[
            "_merge"
        ]
        == "left_only"
    ].copy()

    minrepo_only = merged[
        merged[
            "_merge"
        ]
        == "right_only"
    ].copy()

    header(
        "COVERAGE"
    )

    print(
        f"matched machine numbers    : {len(both)}"
    )

    print(
        f"Ana-Slo only               : {len(ana_only)}"
    )

    print(
        f"Min-Repo only              : {len(minrepo_only)}"
    )

    coverage_ana = (
        len(
            both
        )
        / len(
            ana_one
        )
        * 100.0
        if len(
            ana_one
        )
        else np.nan
    )

    print(
        f"Ana-Slo coverage by MinRepo: {coverage_ana:.2f}%"
    )

    # --------------------------------------------------------
    # Agreement
    # --------------------------------------------------------

    header(
        "AGREEMENT"
    )

    def rate(
        series: pd.Series,
    ) -> float:

        return float(
            series.mean()
            * 100.0
        )

    print(
        f"machine name exact         : "
        f"{rate(both['machine_name_exact']):.2f}%"
    )

    print(
        f"machine name normalized    : "
        f"{rate(both['machine_name_normalized']):.2f}%"
    )

    print(
        f"diff exact                 : "
        f"{rate(both['diff_exact']):.2f}%"
    )

    print(
        f"G exact                    : "
        f"{rate(both['g_exact']):.2f}%"
    )

    print(
        f"all core exact             : "
        f"{rate(both['all_core_exact']):.2f}%"
    )

    diff_mae = float(
        both[
            "diff_delta"
        ].abs().mean()
    )

    g_mae = float(
        both[
            "g_delta"
        ].abs().mean()
    )

    print(
        f"diff MAE                   : {diff_mae:.2f}"
    )

    print(
        f"G MAE                      : {g_mae:.2f}"
    )

    print(
        f"diff max abs error         : "
        f"{both['diff_delta'].abs().max():.0f}"
    )

    print(
        f"G max abs error            : "
        f"{both['g_delta'].abs().max():.0f}"
    )

    # --------------------------------------------------------
    # Mismatch diagnostics
    # --------------------------------------------------------

    mismatch = both[
        ~both[
            "all_core_exact"
        ]
    ].copy()

    mismatch = mismatch[
        [
            "machine_no",
            "ana_machine_name",
            "minrepo_machine_name",
            "machine_name_exact",
            "machine_name_normalized",
            "ana_diff",
            "minrepo_diff",
            "diff_delta",
            "diff_exact",
            "ana_g",
            "minrepo_g",
            "g_delta",
            "g_exact",
        ]
    ].sort_values(
        [
            "machine_no",
        ]
    )

    header(
        "MISMATCH SAMPLE"
    )

    if mismatch.empty:

        print(
            "No core mismatches."
        )

    else:

        print(
            mismatch.head(
                30
            ).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        OUTPUT_DIR
        / "anaslo_vs_minrepo_20260808_all.csv"
    )

    mismatch_path = (
        OUTPUT_DIR
        / "anaslo_vs_minrepo_20260808_mismatches.csv"
    )

    ana_only_path = (
        OUTPUT_DIR
        / "anaslo_vs_minrepo_20260808_ana_only.csv"
    )

    minrepo_only_path = (
        OUTPUT_DIR
        / "anaslo_vs_minrepo_20260808_minrepo_only.csv"
    )

    duplicate_path = (
        OUTPUT_DIR
        / "anaslo_vs_minrepo_20260808_minrepo_duplicates.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "anaslo_vs_minrepo_20260808_summary.csv"
    )

    both.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    mismatch.to_csv(
        mismatch_path,
        index=False,
        encoding="utf-8-sig",
    )

    ana_only.to_csv(
        ana_only_path,
        index=False,
        encoding="utf-8-sig",
    )

    minrepo_only.to_csv(
        minrepo_only_path,
        index=False,
        encoding="utf-8-sig",
    )

    minrepo_duplicate_rows.to_csv(
        duplicate_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [
            {
                "date":
                    "2026-08-08",

                "ana_rows":
                    int(
                        len(
                            ana
                        )
                    ),

                "ana_unique_machines":
                    int(
                        ana[
                            "machine_no"
                        ].nunique()
                    ),

                "minrepo_raw_rows":
                    int(
                        len(
                            minrepo_raw
                        )
                    ),

                "minrepo_numeric_rows":
                    int(
                        len(
                            minrepo
                        )
                    ),

                "minrepo_unique_machines":
                    int(
                        minrepo[
                            "machine_no"
                        ].nunique()
                    ),

                "minrepo_duplicates":
                    minrepo_dup,

                "matched_machine_numbers":
                    int(
                        len(
                            both
                        )
                    ),

                "ana_only":
                    int(
                        len(
                            ana_only
                        )
                    ),

                "minrepo_only":
                    int(
                        len(
                            minrepo_only
                        )
                    ),

                "ana_coverage_percent":
                    coverage_ana,

                "machine_name_exact_percent":
                    rate(
                        both[
                            "machine_name_exact"
                        ]
                    ),

                "machine_name_normalized_percent":
                    rate(
                        both[
                            "machine_name_normalized"
                        ]
                    ),

                "diff_exact_percent":
                    rate(
                        both[
                            "diff_exact"
                        ]
                    ),

                "g_exact_percent":
                    rate(
                        both[
                            "g_exact"
                        ]
                    ),

                "all_core_exact_percent":
                    rate(
                        both[
                            "all_core_exact"
                        ]
                    ),

                "diff_mae":
                    diff_mae,

                "g_mae":
                    g_mae,
            }
        ]
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    for path in (
        summary_path,
        comparison_path,
        mismatch_path,
        ana_only_path,
        minrepo_only_path,
        duplicate_path,
    ):

        print(
            path
        )

    print()
    print(
        "Cross check complete."
    )


if __name__ == "__main__":
    main()
