from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import uuid
from ctypes import wintypes
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MACHINE_DIR = PROJECT_ROOT / "machine_number"
if str(MACHINE_DIR) not in sys.path:
    sys.path.insert(0, str(MACHINE_DIR))

from slotanalyzer_morning_automation_support import (  # noqa: E402
    JST,
    STORE_BIGMARCH,
    STORE_MARUHAN,
    STORE_ORDER,
    STORE_YASUDA,
    TERMINAL_STATES,
    LockUnavailableError,
    StateCorruptError,
    WindowsFileLock,
    append_history_csv,
    atomic_write_json,
    build_fetch_command,
    build_pipeline_command,
    check_source_readiness,
    classify_fetch_result,
    classify_pipeline_result,
    determine_operation_dates,
    ensure_cdp,
    load_json_state,
    maruhan_pipeline_start_allowed,
    now_jst,
    other_store_retry_allowed,
    run_logged_subprocess,
    verify_store_completion,
)


LOG_ROOT = PROJECT_ROOT / "logs" / "morning_automation"
GLOBAL_LOCK_PATH = LOG_ROOT / "morning_automation.lock"
STATE_DIR = LOG_ROOT / "state"
RUNS_DIR = LOG_ROOT / "runs"
HISTORY_PATH = LOG_ROOT / "automation_run_history.csv"
HISTORY_FIELDS = [
    "automation_run_id",
    "operation_date",
    "started_at_jst",
    "completed_at_jst",
    "elapsed_sec",
    "store",
    "stage",
    "attempt",
    "status",
    "returncode",
    "error_category",
    "error",
]

SCHEMA_VERSION = 1
DEFAULT_RETRY_INTERVAL_SEC = 300
DEFAULT_MAX_FETCH_ATTEMPTS = 20
DEFAULT_CHROME_WAIT_SEC = 15
PROVISIONAL_MARUHAN_LAST_START = datetime_time(8, 30)
OTHER_STORE_DEADLINE = datetime_time(9, 30)

STORE_LABELS = {
    STORE_MARUHAN: "Maruhan Mega City Maebashi Inter",
    STORE_BIGMARCH: "Big March Takasaki Oyagi",
    STORE_YASUDA: "Yasuda Maebashi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SlotAnalyzer Phase 2 morning automation wrapper."
    )
    parser.add_argument(
        "--retry-interval-sec",
        type=int,
        default=DEFAULT_RETRY_INTERVAL_SEC,
        help="Readiness retry interval. Default: 300 seconds.",
    )
    parser.add_argument(
        "--max-fetch-attempts",
        type=int,
        default=DEFAULT_MAX_FETCH_ATTEMPTS,
        help="Maximum Fetch-only attempts per store. Default: 20.",
    )
    parser.add_argument(
        "--chrome-wait-sec",
        type=int,
        default=DEFAULT_CHROME_WAIT_SEC,
        help="Seconds to wait for common Chrome/CDP preflight. Default: 15.",
    )
    parser.add_argument(
        "--sleep-on-success",
        action="store_true",
        help="Request normal Windows sleep after all stores complete successfully.",
    )
    args = parser.parse_args()
    if args.retry_interval_sec < 1:
        parser.error("--retry-interval-sec must be >= 1")
    if args.max_fetch_attempts < 1:
        parser.error("--max-fetch-attempts must be >= 1")
    if args.chrome_wait_sec < 1:
        parser.error("--chrome-wait-sec must be >= 1")
    return args


def new_automation_run_id(current: datetime) -> str:
    # Keep the existing validated morning_* format so Maruhan one-click and
    # Fetch timing histories retain the same identifier end-to-end.
    return (
        f"morning_{current.astimezone(JST):%Y%m%dT%H%M%S%f%z}_"
        f"{uuid.uuid4().hex[:8]}"
    )


def blank_store_state(store: str, operation_date: date) -> dict:
    return {
        "store": store,
        "status": "PENDING",
        "attempt_count": 0,
        "fetch_attempt_count": 0,
        "pipeline_attempt_count": 0,
        "current_stage": "",
        "last_readiness": {},
        "latest_data_date": "",
        "target_date": operation_date.isoformat(),
        "last_started_at_jst": "",
        "last_completed_at_jst": "",
        "next_retry_at_jst": "",
        "returncode": "",
        "error_category": "",
        "error": "",
        "verified_artifacts": [],
    }


def create_state(operation_date: date, automation_run_id: str, current: datetime) -> dict:
    timestamp = current.astimezone(JST).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "operation_date": operation_date.isoformat(),
        "automation_run_id": automation_run_id,
        "created_at_jst": timestamp,
        "updated_at_jst": timestamp,
        "stores": {
            store: blank_store_state(store, operation_date) for store in STORE_ORDER
        },
    }


def save_state(path: Path, state: dict, current: datetime | None = None) -> None:
    state["updated_at_jst"] = (current or now_jst()).astimezone(JST).isoformat()
    atomic_write_json(path, state)


def _record_history(
    state: dict,
    store: str,
    stage: str,
    attempt: int,
    status: str,
    result=None,
    error_category: str = "",
    error: str = "",
) -> None:
    try:
        append_history_csv(
            HISTORY_PATH,
            {
                "automation_run_id": state["automation_run_id"],
                "operation_date": state["operation_date"],
                "started_at_jst": result.started_at_jst if result else now_jst().isoformat(),
                "completed_at_jst": result.completed_at_jst if result else now_jst().isoformat(),
                "elapsed_sec": f"{result.elapsed_sec:.6f}" if result else "0.000000",
                "store": store,
                "stage": stage,
                "attempt": attempt,
                "status": status,
                "returncode": result.returncode if result else "",
                "error_category": error_category,
                "error": error,
            },
            HISTORY_FIELDS,
        )
    except Exception as exc:
        print(f"WARNING: automation history append failed: {exc}", file=sys.stderr)


def _deadline_open(store: str, current: datetime, operation_date: date) -> bool:
    if store == STORE_MARUHAN:
        return maruhan_pipeline_start_allowed(
            current, operation_date, PROVISIONAL_MARUHAN_LAST_START
        )
    return other_store_retry_allowed(current, operation_date, OTHER_STORE_DEADLINE)


def _mark_deadline(store_state: dict, store: str, current: datetime) -> None:
    store_state["status"] = "FAILED_FINAL"
    store_state["last_completed_at_jst"] = current.isoformat()
    if store == STORE_MARUHAN:
        store_state["error_category"] = "PROVISIONAL_LAST_START_EXPIRED"
        store_state["error"] = (
            "PROVISIONAL Maruhan one-click last-start time 08:30 JST has passed. "
            "The existing 09:00 JST Forward Guard remains authoritative."
        )
    else:
        store_state["error_category"] = "DEADLINE_EXPIRED"
        store_state["error"] = "09:30 JST retry deadline has passed."


def _has_big_or_yasuda_completion_artifact(
    store: str, project_root: Path, operation_date: date
) -> bool:
    expected = operation_date - timedelta(days=1)
    ymd = operation_date.strftime("%Y%m%d")
    if store == STORE_BIGMARCH:
        data_dir = project_root / "data/bigmarch_takasaki_oyagi/machine_number"
        analysis = data_dir / "analysis_31days_deep"
        candidates = [
            analysis
            / "09_juggler_recent7_future_ranking"
            / f"09_prediction_{ymd}_all_juggler.csv",
            analysis
            / "09_juggler_recent7_future_ranking"
            / f"09_prediction_{ymd}_top10.csv",
            analysis
            / "09_juggler_recent7_future_ranking"
            / f"09_prediction_{ymd}_metadata.csv",
            analysis
            / "12_nonjuggler_weekday_future_ranking"
            / f"12_prediction_{ymd}_all_nonjuggler.csv",
            analysis
            / "12_nonjuggler_weekday_future_ranking"
            / f"12_prediction_{ymd}_top10.csv",
            analysis
            / "12_nonjuggler_weekday_future_ranking"
            / f"12_prediction_{ymd}_metadata.csv",
        ]
    else:
        candidates = [
            project_root / "data/yasuda_maebashi/machine_number" / f"ana_slo_{expected:%Y%m%d}.csv"
        ]
    return any(path.exists() for path in candidates)


def reconcile_startup_state(
    state: dict,
    project_root: Path,
    operation_date: date,
    current: datetime,
) -> None:
    for store in STORE_ORDER:
        store_state = state["stores"][store]
        if store_state["status"] in {"SUCCESS", "ALREADY_COMPLETE"}:
            verification = verify_store_completion(store, project_root, operation_date)
            if verification.ok:
                store_state["verified_artifacts"] = verification.artifacts
            else:
                store_state.update(
                    status="NEEDS_MANUAL_REVIEW",
                    error_category="TERMINAL_ARTIFACT_RECHECK_FAILED",
                    error=verification.error or verification.status,
                    verified_artifacts=verification.artifacts,
                    last_completed_at_jst=current.isoformat(),
                )
            continue
        if store_state["status"] in TERMINAL_STATES:
            continue
        verification = verify_store_completion(store, project_root, operation_date)
        if store == STORE_MARUHAN:
            if verification.ok:
                store_state.update(
                    status="ALREADY_COMPLETE",
                    error_category="",
                    error="",
                    verified_artifacts=verification.artifacts,
                    last_completed_at_jst=current.isoformat(),
                )
                continue
            if verification.status != "NONE":
                store_state.update(
                    status="NEEDS_MANUAL_REVIEW",
                    error_category=verification.status,
                    error=verification.error,
                    verified_artifacts=verification.artifacts,
                    last_completed_at_jst=current.isoformat(),
                )
                continue
        elif _has_big_or_yasuda_completion_artifact(store, project_root, operation_date):
            store_state.update(
                status="NEEDS_MANUAL_REVIEW",
                error_category="PREEXISTING_COMPLETION_ARTIFACT",
                error=(
                    "Completion artifacts predate this automation state; automatic overwrite is forbidden."
                ),
                verified_artifacts=verification.artifacts,
                last_completed_at_jst=current.isoformat(),
            )
            continue
        if store_state["status"] == "RUNNING":
            if store_state.get("current_stage") in {"FETCH", "CDP_PREFLIGHT", "READINESS"}:
                store_state.update(
                    status="FAILED_RETRYABLE",
                    error_category="RECOVERED_PRE_PIPELINE_INTERRUPTION",
                    error="Previous process ended before pipeline start.",
                )
            else:
                store_state.update(
                    status="NEEDS_MANUAL_REVIEW",
                    error_category="INTERRUPTED_PIPELINE_UNKNOWN",
                    error="Previous state was RUNNING after pipeline start or at an unknown stage.",
                    last_completed_at_jst=current.isoformat(),
                )


def _process_store(
    store: str,
    state: dict,
    state_path: Path,
    operation_date: date,
    expected_data_date: date,
    args: argparse.Namespace,
    clock=now_jst,
) -> None:
    store_state = state["stores"][store]
    if store_state["status"] in TERMINAL_STATES:
        return
    current = clock().astimezone(JST)
    next_retry = store_state.get("next_retry_at_jst")
    if next_retry and current < datetime.fromisoformat(next_retry):
        return
    if not _deadline_open(store, current, operation_date):
        _mark_deadline(store_state, store, current)
        save_state(state_path, state, current)
        return

    readiness = check_source_readiness(store, PROJECT_ROOT, expected_data_date)
    store_state["current_stage"] = "READINESS"
    store_state["last_readiness"] = readiness.to_dict()
    store_state["latest_data_date"] = (
        expected_data_date.isoformat() if readiness.ready else ""
    )
    if readiness.source_exists and not readiness.ready:
        store_state.update(
            status="NEEDS_MANUAL_REVIEW",
            error_category="SOURCE_INVALID",
            error=readiness.error,
            last_completed_at_jst=current.isoformat(),
        )
        save_state(state_path, state, current)
        return

    if not readiness.ready:
        if store_state["fetch_attempt_count"] >= args.max_fetch_attempts:
            store_state.update(
                status="FAILED_FINAL",
                error_category="FETCH_ATTEMPT_LIMIT",
                error="Fetch-only attempt limit reached.",
                last_completed_at_jst=current.isoformat(),
            )
            save_state(state_path, state, current)
            return
        attempt = store_state["fetch_attempt_count"] + 1
        store_state.update(
            status="RUNNING",
            current_stage="CDP_PREFLIGHT",
            attempt_count=store_state["attempt_count"] + 1,
            fetch_attempt_count=attempt,
            last_started_at_jst=current.isoformat(),
            error_category="",
            error="",
        )
        save_state(state_path, state, current)
        try:
            ensure_cdp(PROJECT_ROOT, args.chrome_wait_sec)
        except Exception as exc:
            completed = clock().astimezone(JST)
            store_state.update(
                status="FAILED_RETRYABLE",
                current_stage="CDP_PREFLIGHT",
                next_retry_at_jst=(completed + timedelta(seconds=args.retry_interval_sec)).isoformat(),
                error_category="CDP_PREFLIGHT_FAILED",
                error=f"{type(exc).__name__}: {exc}",
                last_completed_at_jst=completed.isoformat(),
            )
            _record_history(
                state, store, "CDP_PREFLIGHT", attempt, "FAILED_RETRYABLE",
                error_category=store_state["error_category"], error=store_state["error"]
            )
            save_state(state_path, state, completed)
            return
        command = build_fetch_command(store, PROJECT_ROOT, sys.executable)
        log_path = (
            RUNS_DIR
            / operation_date.strftime("%Y%m%d")
            / state["automation_run_id"]
            / f"{store}_fetch_attempt{attempt:02d}.log"
        )
        store_state["current_stage"] = "FETCH"
        save_state(state_path, state, clock())
        environment = {
            **os.environ,
            "SLOTANALYZER_MORNING_RUN_ID": state["automation_run_id"],
        }
        result = run_logged_subprocess(
            command, PROJECT_ROOT, log_path, "FETCH", environment=environment, clock=clock
        )
        readiness = check_source_readiness(store, PROJECT_ROOT, expected_data_date)
        current = clock().astimezone(JST)
        store_state["last_readiness"] = readiness.to_dict()
        classification = classify_fetch_result(
            result.returncode, readiness, _deadline_open(store, current, operation_date)
        )
        _record_history(
            state,
            store,
            "FETCH",
            attempt,
            classification,
            result,
            "" if classification == "READY" else (
                "EXPECTED_SOURCE_NOT_READY" if result.returncode == 0 else "FETCH_NONZERO"
            ),
            "" if classification == "READY" else readiness.error,
        )
        if classification != "READY":
            store_state.update(
                status=classification,
                current_stage="FETCH",
                returncode=result.returncode,
                next_retry_at_jst=(current + timedelta(seconds=args.retry_interval_sec)).isoformat(),
                error_category=(
                    "EXPECTED_SOURCE_NOT_READY" if result.returncode == 0 else "FETCH_NONZERO"
                ),
                error=readiness.error,
                last_completed_at_jst=current.isoformat(),
            )
            if classification in TERMINAL_STATES:
                store_state["next_retry_at_jst"] = ""
            save_state(state_path, state, current)
            return
        store_state["latest_data_date"] = expected_data_date.isoformat()

    current = clock().astimezone(JST)
    if not _deadline_open(store, current, operation_date):
        _mark_deadline(store_state, store, current)
        save_state(state_path, state, current)
        return
    if store_state["pipeline_attempt_count"] > 0:
        store_state.update(
            status="NEEDS_MANUAL_REVIEW",
            error_category="PIPELINE_ALREADY_ATTEMPTED",
            error="Automatic pipeline retry is forbidden.",
            last_completed_at_jst=current.isoformat(),
        )
        save_state(state_path, state, current)
        return

    attempt = 1
    store_state.update(
        status="RUNNING",
        current_stage="PIPELINE",
        attempt_count=store_state["attempt_count"] + 1,
        pipeline_attempt_count=attempt,
        last_started_at_jst=current.isoformat(),
        next_retry_at_jst="",
        error_category="",
        error="",
    )
    save_state(state_path, state, current)
    command = build_pipeline_command(
        store, PROJECT_ROOT, sys.executable, operation_date, args.chrome_wait_sec
    )
    log_path = (
        RUNS_DIR
        / operation_date.strftime("%Y%m%d")
        / state["automation_run_id"]
        / f"{store}_pipeline_attempt{attempt:02d}.log"
    )
    environment = {
        **os.environ,
        "SLOTANALYZER_MORNING_RUN_ID": state["automation_run_id"],
    }
    result = run_logged_subprocess(
        command, PROJECT_ROOT, log_path, "PIPELINE", environment=environment, clock=clock
    )
    verification = verify_store_completion(store, PROJECT_ROOT, operation_date)
    classification = classify_pipeline_result(result.returncode, verification)
    completed = clock().astimezone(JST)
    store_state.update(
        status=classification,
        current_stage="PIPELINE",
        returncode=result.returncode,
        last_completed_at_jst=completed.isoformat(),
        error_category=("" if classification == "SUCCESS" else verification.status or "PIPELINE_FAILED"),
        error=("" if classification == "SUCCESS" else verification.error or "One-click returned non-zero."),
        verified_artifacts=verification.artifacts,
    )
    _record_history(
        state,
        store,
        "PIPELINE",
        attempt,
        classification,
        result,
        store_state["error_category"],
        store_state["error"],
    )
    save_state(state_path, state, completed)


def all_terminal(state: dict) -> bool:
    return all(
        store_state["status"] in TERMINAL_STATES
        for store_state in state["stores"].values()
    )


def print_summary(state: dict) -> None:
    print()
    print("=" * 100)
    print("SLOTANALYZER MORNING AUTOMATION SUMMARY")
    print("=" * 100)
    print(f"automation run id     : {state['automation_run_id']}")
    print(f"operation date        : {state['operation_date']}")
    for store in STORE_ORDER:
        item = state["stores"][store]
        print(
            f"{STORE_LABELS[store]:38}: {item['status']} "
            f"fetch={item['fetch_attempt_count']} pipeline={item['pipeline_attempt_count']}"
        )
        if item.get("error"):
            print(f"  {item.get('error_category')}: {item['error']}")


def should_sleep_on_success(
    enabled: bool,
    state: dict,
    wrapper_returncode: int,
) -> bool:
    try:
        required_store_states = [state["stores"][store] for store in STORE_ORDER]
    except (KeyError, TypeError):
        return False
    return (
        enabled
        and wrapper_returncode == 0
        and all_terminal(state)
        and all(
            item["status"] in {"SUCCESS", "ALREADY_COMPLETE"}
            for item in required_store_states
        )
    )


def request_windows_sleep(set_suspend_state=None) -> None:
    if set_suspend_state is None:
        if os.name != "nt":
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
    # False, False, False requests normal sleep (not hibernation), does not
    # force a critical suspend, and leaves wake events enabled.
    if not set_suspend_state(False, False, False):
        error_code = ctypes.get_last_error()
        detail = f" (Windows error {error_code})" if error_code else ""
        raise OSError(f"SetSuspendState returned FALSE{detail}.")


def maybe_sleep_on_success(
    enabled: bool,
    state: dict,
    wrapper_returncode: int,
    sleep_request=None,
) -> bool:
    if not should_sleep_on_success(enabled, state, wrapper_returncode):
        return False

    print("All stores completed successfully. Requesting normal Windows sleep...")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        if sleep_request is None:
            sleep_request = request_windows_sleep
        sleep_request()
    except Exception as exc:
        print(
            f"WARNING: Windows sleep request failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False
    return True


def main() -> int:
    args = parse_args()
    started = now_jst()
    operation_date, expected_data_date = determine_operation_dates(started)
    state_path = STATE_DIR / f"morning_automation_state_{operation_date:%Y%m%d}.json"
    print("SlotAnalyzer Phase 2 morning automation")
    print(f"operation date        : {operation_date}")
    print(f"expected data date    : {expected_data_date}")
    print("Maruhan last start    : 08:30 JST (PROVISIONAL)")
    print("Maruhan formal cutoff : 09:00 JST (existing Forward Guard)")
    print("other store deadline  : 09:30 JST")
    try:
        with WindowsFileLock(GLOBAL_LOCK_PATH):
            try:
                state = load_json_state(state_path)
            except StateCorruptError as exc:
                print(f"FATAL: {exc}", file=sys.stderr)
                print("No store processing was started.", file=sys.stderr)
                return 2
            if state is None:
                state = create_state(operation_date, new_automation_run_id(started), started)
            elif state.get("operation_date") != operation_date.isoformat():
                print("FATAL: state operation_date mismatch.", file=sys.stderr)
                return 2
            reconcile_startup_state(state, PROJECT_ROOT, operation_date, now_jst())
            save_state(state_path, state)
            while not all_terminal(state):
                for store in STORE_ORDER:
                    try:
                        _process_store(
                            store,
                            state,
                            state_path,
                            operation_date,
                            expected_data_date,
                            args,
                        )
                    except Exception as exc:
                        item = state["stores"][store]
                        current = now_jst()
                        if item.get("current_stage") in {"READINESS", "CDP_PREFLIGHT", "FETCH"}:
                            item.update(
                                status="FAILED_RETRYABLE",
                                error_category="PRE_PIPELINE_EXCEPTION",
                                error=f"{type(exc).__name__}: {exc}",
                                next_retry_at_jst=(current + timedelta(seconds=args.retry_interval_sec)).isoformat(),
                            )
                        else:
                            item.update(
                                status="NEEDS_MANUAL_REVIEW",
                                error_category="PIPELINE_EXCEPTION",
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        item["last_completed_at_jst"] = current.isoformat()
                        save_state(state_path, state, current)
                        print(f"{store}: {item['status']}: {item['error']}", file=sys.stderr)
                if all_terminal(state):
                    break
                current = now_jst()
                future_retries = []
                for item in state["stores"].values():
                    if item["status"] not in TERMINAL_STATES and item.get("next_retry_at_jst"):
                        future_retries.append(datetime.fromisoformat(item["next_retry_at_jst"]))
                sleep_sec = args.retry_interval_sec
                if future_retries:
                    sleep_sec = max(1, min(sleep_sec, int((min(future_retries) - current).total_seconds())))
                print(f"Waiting {sleep_sec} seconds before the next readiness pass...")
                time.sleep(sleep_sec)
            save_state(state_path, state, now_jst())
            print_summary(state)
            wrapper_returncode = 0 if all(
                item["status"] in {"SUCCESS", "ALREADY_COMPLETE"}
                for item in state["stores"].values()
            ) else 1
            sys.stdout.flush()
            sys.stderr.flush()
            maybe_sleep_on_success(
                args.sleep_on_success,
                state,
                wrapper_returncode,
            )
            return wrapper_returncode
    except LockUnavailableError as exc:
        print(f"ALREADY RUNNING: {exc}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
