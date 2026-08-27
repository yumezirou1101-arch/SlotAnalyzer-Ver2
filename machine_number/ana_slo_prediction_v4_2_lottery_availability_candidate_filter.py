from __future__ import annotations

from pathlib import Path
import argparse
import re
import pandas as pd


# ============================================================
# 78 - Lottery Number / Availability Practical Candidate Filter
# ============================================================
#
# Purpose
# -------
# Use the already-created 77 integrated prediction report and
# create a practical shortlist AFTER the lottery result is known.
#
# Important safety policy
# -------
# - This script does NOT create a new prediction score.
# - Lottery number does NOT automatically change model ranking.
# - Availability is handled explicitly by machine number.
# - 64 / 74 / 75 / 76 / 77 are never modified.
#
# Why:
# -------
# We do not yet have enough validated historical data linking
# lottery number -> machine availability. Therefore we record the
# lottery numbers now, but we do NOT invent arbitrary thresholds.
#
# This gives us a clean base for future analysis:
# lottery number + candidate availability + actual result.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
)

SOURCE_77_DIR = (
    ANALYSIS_DIR
    / "77_live_integrated_prediction_report"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "78_lottery_availability_candidate_filter"
)


# ============================================================
# Helpers
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []

    values = []

    for part in re.split(r"[,、\s]+", text.strip()):
        if not part:
            continue

        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(
                f"Invalid machine number: {part}"
            ) from exc

    return sorted(set(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a practical shortlist from 77 after lottery results "
            "and actual machine availability are known."
        )
    )

    parser.add_argument(
        "--target-date",
        default=None,
        help=(
            "Target date YYYY-MM-DD. "
            "If omitted, use the newest 77 integrated report."
        ),
    )

    parser.add_argument(
        "--lottery-1",
        type=int,
        default=None,
        help="Lottery number for person 1.",
    )

    parser.add_argument(
        "--lottery-2",
        type=int,
        default=None,
        help="Lottery number for person 2 (optional).",
    )

    parser.add_argument(
        "--unavailable",
        default="",
        help=(
            "Comma-separated machine numbers already unavailable/taken. "
            "Example: 912,911,698"
        ),
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=15,
        help="Maximum remaining candidates to display/save. Default: 15.",
    )

    return parser.parse_args()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None

    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CSV read failed: {path}\n"
        f"last_error={last_error}"
    )


def discover_reports() -> dict[pd.Timestamp, Path]:
    rx = re.compile(
        r"^77_integrated_prediction_(\d{8})\.csv$",
        re.IGNORECASE,
    )

    found = {}

    if not SOURCE_77_DIR.exists():
        return found

    for path in SOURCE_77_DIR.glob(
        "77_integrated_prediction_????????.csv"
    ):
        m = rx.fullmatch(path.name)

        if not m:
            continue

        dt = pd.to_datetime(
            m.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(dt):
            continue

        found[
            pd.Timestamp(dt).normalize()
        ] = path

    return found


def resolve_source(
    target_date: pd.Timestamp | None,
) -> tuple[pd.Timestamp, Path]:

    found = discover_reports()

    if not found:
        raise FileNotFoundError(
            f"No 77 integrated prediction reports found in:\n"
            f"{SOURCE_77_DIR}"
        )

    if target_date is None:
        target = sorted(found)[-1]
        return target, found[target]

    target = pd.Timestamp(
        target_date
    ).normalize()

    if target not in found:
        raise FileNotFoundError(
            f"No 77 integrated report for {target.date()}."
        )

    return target, found[target]


def candidate_group(row) -> str:
    normal_rank = row.get(
        "normal_rank",
        pd.NA,
    )

    a_type_rank = row.get(
        "a_type_rank",
        pd.NA,
    )

    juggler_rank = row.get(
        "juggler_rank",
        pd.NA,
    )

    if pd.notna(normal_rank):
        if int(normal_rank) <= 5:
            return "NORMAL_PRIMARY"
        return "NORMAL_NEXT"

    if (
        pd.notna(a_type_rank)
        and pd.notna(juggler_rank)
    ):
        return "A_TYPE_JUGGLER_OVERLAP"

    if pd.notna(a_type_rank):
        return "A_TYPE_ONLY"

    if pd.notna(juggler_rank):
        return "JUGGLER_ONLY"

    return "OTHER"


def main() -> None:
    args = parse_args()

    requested_target = None

    if args.target_date:
        requested_target = pd.Timestamp(
            pd.to_datetime(
                args.target_date,
                format="%Y-%m-%d",
                errors="raise",
            )
        )

    target_date, source_path = resolve_source(
        requested_target
    )

    unavailable = parse_int_list(
        args.unavailable
    )

    if args.max_candidates <= 0:
        raise ValueError(
            "--max-candidates must be >= 1"
        )

    df = read_csv_flexible(
        source_path
    )

    required = [
        "report_order",
        "machine_no",
        "machine_name",
        "selected_by",
        "overlap_count",
        "normal_rank",
        "a_type_rank",
        "juggler_rank",
        "score",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"77 report columns missing: {missing}"
        )

    df["machine_no"] = pd.to_numeric(
        df["machine_no"],
        errors="coerce",
    )

    df["report_order"] = pd.to_numeric(
        df["report_order"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "machine_no",
            "report_order",
        ]
    ).copy()

    df["machine_no"] = (
        df["machine_no"]
        .astype(int)
    )

    df["candidate_group"] = df.apply(
        candidate_group,
        axis=1,
    )

    df["availability_status"] = "AVAILABLE_OR_UNKNOWN"

    if unavailable:
        df.loc[
            df["machine_no"].isin(
                unavailable
            ),
            "availability_status",
        ] = "UNAVAILABLE"

    remaining = df[
        df["availability_status"]
        != "UNAVAILABLE"
    ].copy()

    # Keep 77's display order. We do NOT create another score.
    remaining = (
        remaining.sort_values(
            "report_order"
        )
        .head(
            args.max_candidates
        )
        .reset_index(
            drop=True
        )
    )

    remaining[
        "shortlist_order"
    ] = range(
        1,
        len(remaining) + 1,
    )

    remaining[
        "lottery_1"
    ] = args.lottery_1

    remaining[
        "lottery_2"
    ] = args.lottery_2

    # Blank operational columns for on-site use / later analysis.
    remaining[
        "actual_availability_confirmed"
    ] = ""

    remaining[
        "final_play_order"
    ] = ""

    remaining[
        "final_decision"
    ] = ""

    remaining[
        "decision_note"
    ] = ""

    header(
        "78 - Lottery / Availability Practical Candidate Filter"
    )

    print(
        f"target date           : {target_date.date()}"
    )

    print(
        f"source 77 report      : {source_path}"
    )

    print(
        f"lottery 1             : {args.lottery_1}"
    )

    print(
        f"lottery 2             : {args.lottery_2}"
    )

    print(
        f"unavailable machines  : "
        f"{','.join(map(str, unavailable)) if unavailable else '(none)'}"
    )

    print(
        f"remaining shown       : {len(remaining)}"
    )

    header(
        "PRACTICAL SHORTLIST"
    )

    display_cols = [
        "shortlist_order",
        "machine_no",
        "machine_name",
        "candidate_group",
        "selected_by",
        "overlap_count",
        "normal_rank",
        "a_type_rank",
        "juggler_rank",
        "score",
    ]

    print(
        remaining[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Important:"
    )
    print(
        "- Lottery number is recorded, but does NOT automatically change ranking."
    )
    print(
        "- Machines listed with --unavailable are removed from the shortlist."
    )
    print(
        "- shortlist_order preserves 77's practical display order; "
        "it is NOT a new model score."
    )
    print(
        "- final_play_order / final_decision are intentionally left blank "
        "for the real on-site decision."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ymd = target_date.strftime(
        "%Y%m%d"
    )

    shortlist_path = (
        OUTPUT_DIR
        / f"78_practical_shortlist_{ymd}.csv"
    )

    full_status_path = (
        OUTPUT_DIR
        / f"78_candidate_availability_status_{ymd}.csv"
    )

    metadata_path = (
        OUTPUT_DIR
        / f"78_practical_shortlist_{ymd}_metadata.csv"
    )

    remaining.to_csv(
        shortlist_path,
        index=False,
        encoding="utf-8-sig",
    )

    df.to_csv(
        full_status_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = pd.DataFrame(
        [
            {
                "target_date":
                    target_date.date(),
                "source_77_file":
                    source_path.name,
                "lottery_1":
                    args.lottery_1,
                "lottery_2":
                    args.lottery_2,
                "unavailable_machine_count":
                    len(unavailable),
                "unavailable_machine_nos":
                    ",".join(
                        map(
                            str,
                            unavailable,
                        )
                    ),
                "max_candidates":
                    args.max_candidates,
                "shortlist_rows":
                    len(remaining),
                "new_prediction_score_created":
                    False,
                "lottery_auto_reranking":
                    False,
                "model_changed":
                    False,
                "policy":
                    (
                        "record lottery; filter actual unavailable machines; "
                        "preserve source prediction information"
                    ),
            }
        ]
    )

    metadata.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    for path in (
        shortlist_path,
        full_status_path,
        metadata_path,
    ):
        print(path)

    print()
    print(
        "78 practical candidate filter complete."
    )
    print(
        "64 / 74 / 75 / 76 / 77 were not modified."
    )


if __name__ == "__main__":
    main()
