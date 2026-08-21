from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

CONVERTER = (
    PROJECT_ROOT
    / "ana_slo_source_html_to_daily_csv_auto.py"
)

FORWARD_TEST = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_prediction_v4_2_forward_champion_challenger.py"
)


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def run_command(
    command: list[str],
    label: str,
) -> None:

    header(label)

    print(
        "COMMAND:"
    )
    print(
        " ".join(
            f'"{part}"'
            if " " in part
            else part
            for part in command
        )
    )
    print()

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"{label} failed "
            f"(returncode={result.returncode})"
        )


def find_latest_source_html() -> Path:

    candidates = sorted(
        PROJECT_ROOT.glob(
            "ana_slo_*_source.html"
        ),
        key=lambda p:
            p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No ana_slo_*_source.html files found "
            f"in {PROJECT_ROOT}"
        )

    return candidates[0]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "SlotAnalyzer Daily Forward Update"
    )

    if not CONVERTER.exists():
        raise FileNotFoundError(
            f"Converter not found: {CONVERTER}"
        )

    if not FORWARD_TEST.exists():
        raise FileNotFoundError(
            f"Forward-test script not found: {FORWARD_TEST}"
        )

    source_html = (
        find_latest_source_html()
    )

    print(
        f"Latest source HTML : {source_html.name}"
    )
    print(
        f"Converter          : {CONVERTER.name}"
    )
    print(
        f"Forward test       : {FORWARD_TEST.name}"
    )

    # --------------------------------------------------------
    # Step 1: HTML -> validated daily CSV
    # --------------------------------------------------------

    run_command(
        [
            sys.executable,
            str(CONVERTER),
            str(source_html),
        ],
        "STEP 1 / 2 - Convert latest source HTML",
    )

    # --------------------------------------------------------
    # Step 2: Champion / Challenger forward test
    # --------------------------------------------------------

    run_command(
        [
            sys.executable,
            str(FORWARD_TEST),
        ],
        "STEP 2 / 2 - Run Champion / Challenger forward test",
    )

    header(
        "DAILY UPDATE COMPLETE"
    )

    print(
        "Latest HTML was converted, quality-checked, "
        "and the forward test was updated."
    )
    print()
    print(
        "No model parameters were changed."
    )


if __name__ == "__main__":
    main()
