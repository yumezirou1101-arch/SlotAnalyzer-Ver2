from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import time
from datetime import datetime


# ============================================================
# SlotAnalyzer - Morning Runner
# ============================================================
#
# Purpose
# -------
# Run the normal morning update for the currently supported
# SlotAnalyzer stores.
#
# Current stores
# --------------
# 1) Maruhan Mega City Maebashi Inter
#    -> one-click daily update V2
#    -> includes automatic live prediction backtest
#
# 2) Big March Takasaki Oyagi
#    -> one-click daily update V3
#    -> includes frozen Top3 forward evaluation
#
# 3) Yasuda Maebashi
#    -> one-click daily update V1
#    -> acquisition, freshness, conversion, and quality validation
#
# Safety
# ------
# - Existing store pipelines are not modified here.
# - This script only calls the already-tested pipelines.
# - Each store is evaluated independently.
# - If one store fails, the other stores are still attempted.
# - Final summary clearly reports OK / FAILED.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

MACHINE_DIR = (
    PROJECT_ROOT
    / "machine_number"
)


MARUHAN_SCRIPT = (
    MACHINE_DIR
    / "ana_slo_maruhan_maebashi_one_click_daily_update_v2.py"
)

BIGMARCH_SCRIPT = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_one_click_daily_update_v3.py"
)

YASUDA_SCRIPT = (
    MACHINE_DIR
    / "ana_slo_yasuda_maebashi_one_click_daily_update_v1.py"
)


def header(
    title: str,
) -> None:

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "SlotAnalyzer morning runner for "
            "Maruhan Maebashi, Big March Oyagi, and Yasuda Maebashi."
        )
    )

    parser.add_argument(
        "--skip-maruhan",
        action="store_true",
        help=(
            "Skip Maruhan Mega City "
            "Maebashi Inter."
        ),
    )

    parser.add_argument(
        "--skip-bigmarch",
        action="store_true",
        help=(
            "Skip Big March Takasaki Oyagi."
        ),
    )

    parser.add_argument(
        "--skip-yasuda",
        action="store_true",
        help=(
            "Skip Yasuda Maebashi."
        ),
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help=(
            "Skip HTML fetch for all selected stores "
            "and use existing data."
        ),
    )

    parser.add_argument(
        "--maruhan-fetch-days",
        type=int,
        default=1,
        help=(
            "Number of newest Maruhan date links "
            "to inspect. Default: 1."
        ),
    )

    parser.add_argument(
        "--bigmarch-fetch-days",
        type=int,
        default=1,
        help=(
            "Number of newest Big March date links "
            "to inspect. Default: 1."
        ),
    )

    parser.add_argument(
        "--yasuda-fetch-days",
        type=int,
        default=1,
        help=(
            "Number of newest Yasuda Maebashi date links "
            "to inspect. Default: 1."
        ),
    )

    parser.add_argument(
        "--bigmarch-min-machines",
        type=int,
        default=200,
        help=(
            "Minimum acceptable Big March "
            "machine rows. Default: 200."
        ),
    )

    parser.add_argument(
        "--chrome-wait-sec",
        type=int,
        default=15,
        help=(
            "Seconds to wait for Chrome CDP. "
            "Default: 15."
        ),
    )

    return parser.parse_args()


def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"Required script not found:\n{path}"
        )


def compile_script(
    path: Path,
) -> None:

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(path),
        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"py_compile failed:\n{path}"
        )


def run_store(
    store_name: str,
    script_path: Path,
    args: list[str],
) -> dict:

    header(
        f"START - {store_name}"
    )

    command = [
        sys.executable,
        str(script_path),
        *args,
    ]

    print(
        "COMMAND:"
    )

    print(
        " ".join(
            f'"{item}"'
            if " " in item
            else item
            for item in command
        )
    )

    print()

    started = time.perf_counter()

    try:

        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        if result.returncode == 0:

            status = "OK"
            error = ""

        else:

            status = "FAILED"
            error = (
                f"exit code "
                f"{result.returncode}"
            )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - started
        )

        status = "FAILED"
        error = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    print()

    if status == "OK":

        print(
            f"{store_name}: OK"
        )

    else:

        print(
            f"{store_name}: FAILED"
        )

        print(
            f"reason: {error}"
        )

    return {
        "store": store_name,
        "status": status,
        "elapsed": elapsed,
        "error": error,
    }


def main() -> None:

    args = parse_args()

    header(
        "SlotAnalyzer - Morning Runner"
    )

    print(
        f"started at            : "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    print(
        f"project root          : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"python                : "
        f"{sys.executable}"
    )

    print(
        f"skip Maruhan          : "
        f"{args.skip_maruhan}"
    )

    print(
        f"skip Big March        : "
        f"{args.skip_bigmarch}"
    )

    print(
        f"skip Yasuda           : "
        f"{args.skip_yasuda}"
    )

    print(
        f"skip fetch            : "
        f"{args.skip_fetch}"
    )

    print(
        f"Maruhan fetch days    : "
        f"{args.maruhan_fetch_days}"
    )

    print(
        f"Big March fetch days  : "
        f"{args.bigmarch_fetch_days}"
    )

    print(
        f"Yasuda fetch days     : "
        f"{args.yasuda_fetch_days}"
    )

    print(
        f"Big March min machines: "
        f"{args.bigmarch_min_machines}"
    )

    # --------------------------------------------------------
    # PREFLIGHT
    # --------------------------------------------------------

    header(
        "PREFLIGHT"
    )

    scripts_to_check = []

    if not args.skip_maruhan:

        scripts_to_check.append(
            (
                "Maruhan",
                MARUHAN_SCRIPT,
            )
        )

    if not args.skip_bigmarch:

        scripts_to_check.append(
            (
                "Big March",
                BIGMARCH_SCRIPT,
            )
        )

    if not args.skip_yasuda:

        scripts_to_check.append(
            (
                "Yasuda",
                YASUDA_SCRIPT,
            )
        )

    if not scripts_to_check:

        print(
            "No stores selected."
        )

        return

    for label, path in scripts_to_check:

        require_file(
            path
        )

        print(
            f"{label:<20}: "
            f"{path.name}"
        )

    print()
    print(
        "Compiling store pipelines..."
    )

    for label, path in scripts_to_check:

        compile_script(
            path
        )

        print(
            f"py_compile OK         : "
            f"{path.name}"
        )

    # --------------------------------------------------------
    # RUN STORES
    # --------------------------------------------------------

    total_started = (
        time.perf_counter()
    )

    results = []

    # --------------------------------------------------------
    # MARUHAN
    # --------------------------------------------------------

    if not args.skip_maruhan:

        maruhan_args = [
            "--fetch-days",
            str(
                args.maruhan_fetch_days
            ),
            "--chrome-wait-sec",
            str(
                args.chrome_wait_sec
            ),
        ]

        if args.skip_fetch:

            maruhan_args.append(
                "--skip-fetch"
            )

        result = run_store(
            (
                "MARUHAN MEGA CITY "
                "MAEBASHI INTER"
            ),
            MARUHAN_SCRIPT,
            maruhan_args,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # BIG MARCH
    # --------------------------------------------------------

    if not args.skip_bigmarch:

        bigmarch_args = [
            "--fetch-days",
            str(
                args.bigmarch_fetch_days
            ),
            "--min-machines",
            str(
                args.bigmarch_min_machines
            ),
            "--chrome-wait-sec",
            str(
                args.chrome_wait_sec
            ),
        ]

        if args.skip_fetch:

            bigmarch_args.append(
                "--skip-fetch"
            )

        result = run_store(
            (
                "BIG MARCH "
                "TAKASAKI OYAGI"
            ),
            BIGMARCH_SCRIPT,
            bigmarch_args,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # YASUDA
    # --------------------------------------------------------

    if not args.skip_yasuda:

        yasuda_args = [
            "--fetch-days",
            str(
                args.yasuda_fetch_days
            ),
        ]

        if args.skip_fetch:

            yasuda_args.append(
                "--skip-fetch"
            )

        result = run_store(
            "YASUDA MAEBASHI",
            YASUDA_SCRIPT,
            yasuda_args,
        )

        results.append(
            result
        )

    total_elapsed = (
        time.perf_counter()
        - total_started
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    header(
        "MORNING RUNNER FINAL SUMMARY"
    )

    for result in results:

        print(
            f"{result['store']:<42} "
            f": {result['status']:<7} "
            f"({result['elapsed']:.2f} sec)"
        )

        if result["error"]:

            print(
                f"  error: "
                f"{result['error']}"
            )

    ok_count = sum(
        1
        for result in results
        if result["status"] == "OK"
    )

    failed_count = sum(
        1
        for result in results
        if result["status"] == "FAILED"
    )

    print()

    print(
        f"stores processed      : "
        f"{len(results)}"
    )

    print(
        f"OK                    : "
        f"{ok_count}"
    )

    print(
        f"FAILED                : "
        f"{failed_count}"
    )

    print(
        f"total elapsed sec     : "
        f"{total_elapsed:.2f}"
    )

    print(
        f"completed at          : "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    print()

    if failed_count == 0:

        print(
            "RESULT: MORNING UPDATE COMPLETE"
        )

        print(
            "All selected store pipelines "
            "completed successfully."
        )

    else:

        print(
            "RESULT: MORNING UPDATE COMPLETED "
            "WITH ERRORS"
        )

        print(
            "Check the FAILED store output above."
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":

    main()