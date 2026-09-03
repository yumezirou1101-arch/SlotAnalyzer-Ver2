from __future__ import annotations

import argparse
import ctypes
import sys
import time
import uuid
from ctypes import wintypes
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MACHINE_DIR = PROJECT_ROOT / "machine_number"
if str(MACHINE_DIR) not in sys.path:
    sys.path.insert(0, str(MACHINE_DIR))

from slotanalyzer_morning_automation_support import (  # noqa: E402
    append_history_csv,
    now_jst,
)


DEFAULT_DELAY_SEC = 10
HISTORY_PATH = PROJECT_ROOT / "logs" / "morning_automation" / "sleep_helper_history.csv"
HISTORY_FIELDS = [
    "helper_id",
    "parent_automation_run_id",
    "launched_at_jst",
    "requested_at_jst",
    "delay_sec",
    "status",
    "error",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Request normal Windows sleep after a short delay."
    )
    parser.add_argument(
        "--parent-automation-run-id",
        required=True,
        help="Morning automation run identifier used to correlate helper history.",
    )
    parser.add_argument(
        "--delay-sec",
        type=int,
        default=DEFAULT_DELAY_SEC,
        help="Seconds to wait before requesting sleep. Default: 10.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the delay and logging path without calling SetSuspendState.",
    )
    args = parser.parse_args(argv)
    args.parent_automation_run_id = args.parent_automation_run_id.strip()
    if not args.parent_automation_run_id:
        parser.error("--parent-automation-run-id must not be empty")
    if args.delay_sec < 0:
        parser.error("--delay-sec must be >= 0")
    return args


def request_windows_sleep(set_suspend_state=None) -> None:
    if set_suspend_state is None:
        if sys.platform != "win32":
            raise OSError("Windows sleep is only available on Windows.")
        powrprof = ctypes.WinDLL("PowrProf.dll", use_last_error=True)
        set_suspend_state = powrprof.SetSuspendState
        set_suspend_state.argtypes = [
            wintypes.BOOLEAN,
            wintypes.BOOLEAN,
            wintypes.BOOLEAN,
        ]
        set_suspend_state.restype = wintypes.BOOLEAN

    # SetSuspendState(Hibernate, ForceCritical, DisableWakeEvent).
    # False, False, False requests normal sleep, does not force a critical
    # suspend, and keeps configured wake events enabled.
    if not set_suspend_state(False, False, False):
        error_code = ctypes.get_last_error()
        detail = f" (Windows error {error_code})" if error_code else ""
        raise OSError(f"SetSuspendState returned FALSE{detail}.")


def _append_event(
    history_path: Path,
    history_appender,
    *,
    helper_id: str,
    parent_automation_run_id: str,
    launched_at_jst: str,
    requested_at_jst: str,
    delay_sec: int,
    status: str,
    error: str = "",
) -> bool:
    try:
        history_appender(
            history_path,
            {
                "helper_id": helper_id,
                "parent_automation_run_id": parent_automation_run_id,
                "launched_at_jst": launched_at_jst,
                "requested_at_jst": requested_at_jst,
                "delay_sec": delay_sec,
                "status": status,
                "error": error,
            },
            HISTORY_FIELDS,
        )
    except Exception as exc:
        print(
            f"WARNING: sleep helper history write failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def run_helper(
    parent_automation_run_id: str,
    delay_sec: int = DEFAULT_DELAY_SEC,
    dry_run: bool = False,
    *,
    sleep_function=time.sleep,
    sleep_request=request_windows_sleep,
    clock=now_jst,
    history_path: Path = HISTORY_PATH,
    history_appender=append_history_csv,
    helper_id: str | None = None,
) -> int:
    launched = clock()
    if helper_id is None:
        helper_id = f"sleep_{launched:%Y%m%dT%H%M%S%f%z}_{uuid.uuid4().hex[:8]}"
    launched_at = launched.isoformat()
    common = {
        "helper_id": helper_id,
        "parent_automation_run_id": parent_automation_run_id,
        "launched_at_jst": launched_at,
        "delay_sec": delay_sec,
    }
    _append_event(
        history_path,
        history_appender,
        **common,
        requested_at_jst="",
        status="STARTED",
    )

    sleep_function(delay_sec)
    if dry_run:
        _append_event(
            history_path,
            history_appender,
            **common,
            requested_at_jst="",
            status="DRY_RUN_COMPLETE",
        )
        return 0

    requested_at = clock().isoformat()
    _append_event(
        history_path,
        history_appender,
        **common,
        requested_at_jst=requested_at,
        status="REQUESTING",
    )
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        sleep_request()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _append_event(
            history_path,
            history_appender,
            **common,
            requested_at_jst=requested_at,
            status="FAILED",
            error=error,
        )
        print(f"ERROR: Windows sleep request failed: {error}", file=sys.stderr, flush=True)
        return 1

    _append_event(
        history_path,
        history_appender,
        **common,
        requested_at_jst=requested_at,
        status="RETURNED_AFTER_RESUME",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_helper(
        args.parent_automation_run_id,
        delay_sec=args.delay_sec,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
