from __future__ import annotations

from pathlib import Path
import argparse
import re
from datetime import datetime

import pandas as pd


# ============================================================
# 78 - Lottery / Availability Practical Candidate Filter
#      HISTORY V2
# ============================================================
#
# - Every execution is saved under history/
# - *_latest.csv is refreshed each run
# - --note is supported
# - 64 / 74 / 75 / 76 / 77 are not modified
# - No new prediction score is created
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

HISTORY_DIR = OUTPUT_DIR / "history"
LOG_PATH = OUTPUT_DIR / "78_execution_log.csv"


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
            "Create a practical shortlist from 77. "
            "Every execution is preserved as history."
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
        help="Maximum candidates to display/save. Default: 15.",
    )

    parser.add_argument(
        "--note",
        default="",
        help=(
            "Optional short note for this update. "
            "Example: entrance_after_5min"
        ),
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
        match = rx.fullmatch(path.name)

        if not match:
            continue

        dt = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(dt):
            continue

        found[pd.Timestamp(dt).normalize()] = path

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

    target = pd.Timestamp(target_date).normalize()

    if target not in found:
        raise FileNotFoundError(
            f"No 77 integrated report for {target.date()}."
        )

    return target, found[target]


def candidate_group(row) -> str:
    normal_rank = row.get("normal_rank", pd.NA)
    a_type_rank = row.get("a_type_rank", pd.NA)
    juggler_rank = row.get("juggler_rank", pd.NA)

    if pd.notna(normal_rank):
        if int(normal_rank) <= 5:
            return "NORMAL_PRIMARY"
        return "NORMAL_NEXT"

    if pd.notna(a_type_rank) and pd.notna(juggler_rank):
        return "A_TYPE_JUGGLER_OVERLAP"

    if pd.notna(a_type_rank):
        return "A_TYPE_ONLY"

    if pd.notna(juggler_rank):
        return "JUGGLER_ONLY"

    return "OTHER"


def validate_source(df: pd.DataFrame) -> None:
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
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"77 report columns missing: {missing}"
        )


def append_execution_log(row: dict) -> None:
    new_row = pd.DataFrame([row])

    if LOG_PATH.exists():
        old = read_csv_flexible(LOG_PATH)
        out = pd.concat(
            [old, new_row],
            ignore_index=True,
        )
    else:
        out = new_row

    out.to_csv(
        LOG_PATH,
        index=False,
        encoding="utf-8-sig",
    )


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

    validate_source(
        df
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
        subset=["machine_no", "report_order"]
    ).copy()

    df["machine_no"] = df["machine_no"].astype(int)

    df["candidate_group"] = df.apply(
        candidate_group,
        axis=1,
    )

    df["availability_status"] = "AVAILABLE_OR_UNKNOWN"

    if unavailable:
        df.loc[
            df["machine_no"].isin(unavailable),
            "availability_status",
        ] = "UNAVAILABLE"

    remaining = df[
        df["availability_status"] != "UNAVAILABLE"
    ].copy()

    remaining = (
        remaining.sort_values("report_order")
        .head(args.max_candidates)
        .reset_index(drop=True)
    )

    remaining["shortlist_order"] = range(
        1,
        len(remaining) + 1,
    )

    remaining["lottery_1"] = args.lottery_1
    remaining["lottery_2"] = args.lottery_2
    remaining["update_note"] = args.note

    remaining["actual_availability_confirmed"] = ""
    remaining["final_play_order"] = ""
    remaining["final_decision"] = ""
    remaining["decision_note"] = ""

    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    run_time = now.isoformat(timespec="seconds")

    remaining["run_id"] = run_id
    remaining["run_time"] = run_time

    df["run_id"] = run_id
    df["run_time"] = run_time
    df["lottery_1"] = args.lottery_1
    df["lottery_2"] = args.lottery_2
    df["update_note"] = args.note

    header(
        "78 - Lottery / Availability Practical Candidate Filter HISTORY V2"
    )

    print(f"target date           : {target_date.date()}")
    print(f"run id                : {run_id}")
    print(f"source 77 report      : {source_path}")
    print(f"lottery 1             : {args.lottery_1}")
    print(f"lottery 2             : {args.lottery_2}")
    print(
        f"update note           : "
        f"{args.note if args.note else '(none)'}"
    )
    print(
        f"unavailable machines  : "
        f"{','.join(map(str, unavailable)) if unavailable else '(none)'}"
    )
    print(f"remaining shown       : {len(remaining)}")

    header("PRACTICAL SHORTLIST")

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
        remaining[display_cols].to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ymd = target_date.strftime("%Y%m%d")

    history_shortlist_path = (
        HISTORY_DIR
        / f"78_practical_shortlist_{ymd}_{run_id}.csv"
    )

    history_status_path = (
        HISTORY_DIR
        / f"78_candidate_availability_status_{ymd}_{run_id}.csv"
    )

    history_metadata_path = (
        HISTORY_DIR
        / f"78_practical_shortlist_{ymd}_{run_id}_metadata.csv"
    )

    latest_shortlist_path = (
        OUTPUT_DIR
        / f"78_practical_shortlist_{ymd}_latest.csv"
    )

    latest_status_path = (
        OUTPUT_DIR
        / f"78_candidate_availability_status_{ymd}_latest.csv"
    )

    latest_metadata_path = (
        OUTPUT_DIR
        / f"78_practical_shortlist_{ymd}_latest_metadata.csv"
    )

    metadata = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_time": run_time,
                "target_date": target_date.date(),
                "source_77_file": source_path.name,
                "lottery_1": args.lottery_1,
                "lottery_2": args.lottery_2,
                "update_note": args.note,
                "unavailable_machine_count": len(unavailable),
                "unavailable_machine_nos":
                    ",".join(map(str, unavailable)),
                "max_candidates": args.max_candidates,
                "shortlist_rows": len(remaining),
                "new_prediction_score_created": False,
                "lottery_auto_reranking": False,
                "model_changed": False,
                "history_preserved": True,
            }
        ]
    )

    remaining.to_csv(
        history_shortlist_path,
        index=False,
        encoding="utf-8-sig",
    )

    df.to_csv(
        history_status_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata.to_csv(
        history_metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    remaining.to_csv(
        latest_shortlist_path,
        index=False,
        encoding="utf-8-sig",
    )

    df.to_csv(
        latest_status_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata.to_csv(
        latest_metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    append_execution_log(
        metadata.iloc[0].to_dict()
    )

    header("HISTORY FILES SAVED")

    for path in (
        history_shortlist_path,
        history_status_path,
        history_metadata_path,
    ):
        print(path)

    header("LATEST FILES UPDATED")

    for path in (
        latest_shortlist_path,
        latest_status_path,
        latest_metadata_path,
    ):
        print(path)

    print()
    print(f"Execution log          : {LOG_PATH}")
    print()
    print("78 HISTORY V2 complete.")
    print("History preserved; only *_latest.csv files are refreshed.")
    print("64 / 74 / 75 / 76 / 77 were not modified.")


if __name__ == "__main__":
    main()
