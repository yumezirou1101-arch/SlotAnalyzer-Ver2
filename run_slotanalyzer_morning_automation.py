from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MACHINE_DIR = PROJECT_ROOT / "machine_number"
if str(MACHINE_DIR) not in sys.path:
    sys.path.insert(0, str(MACHINE_DIR))

from slotanalyzer_morning_automation_support import (  # noqa: E402
    JST,
    STORE_BIGMARCH,
    STORE_BICTSUBAME,
    STORE_MARUHAN,
    STORE_ORDER,
    STORE_YASUDA,
    TERMINAL_STATES,
    LockUnavailableError,
    StateCorruptError,
    WindowsFileLock,
    append_history_csv,
    atomic_write_json,
    assess_big_march_catchup,
    build_big_march_catchup_command,
    build_big_march_provisional_command,
    build_big_march_stale_reference_command,
    build_fetch_command,
    build_pipeline_command,
    bic_tsubame_expected_latest_data_date,
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
    sha256_file,
    verify_big_march_daily_for_date,
    verify_big_march_provisional_completion,
    verify_big_march_stale_reference_completion,
    verify_store_completion,
)
from slotanalyzer_morning_notification import notify_terminal_stores_best_effort  # noqa: E402
from slotanalyzer_inventory_guard import (  # noqa: E402
    assess_bic_tsubame_inventory_guard,
    assess_inventory_guard,
    assess_yasuda_inventory_guard,
)
from slotanalyzer_manual_review_auto_triage import (
    assess_manual_review_auto_triage,
    not_applicable_triage,
)
from slotanalyzer_inventory_monitor import (  # noqa: E402
    load_inventory_monitor_evidence,
    observe_big_march_inventory_best_effort,
)
from slotanalyzer_bigmarch_inventory_guard import (  # noqa: E402
    assess_big_march_inventory_guard,
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
MARUHAN_RETRY_INTERVAL_SEC = 60
DEFAULT_MAX_FETCH_ATTEMPTS = 20
MARUHAN_MAX_FETCH_ATTEMPTS = 30
DEFAULT_CHROME_WAIT_SEC = 15
DEFAULT_FETCH_TIMEOUT_SEC = 120
PROVISIONAL_MARUHAN_LAST_START = datetime_time(8, 30)
MORNING_FINALIZE_DEADLINE = datetime_time(8, 30)
OTHER_STORE_DEADLINE = MORNING_FINALIZE_DEADLINE
SLEEP_HELPER_PATH = PROJECT_ROOT / "sleep_windows_after_delay.py"
SLEEP_HELPER_DELAY_SEC = 10
MARUHAN_E2E_CHECKER_PATH = (
    MACHINE_DIR / "slotanalyzer_morning_natural_e2e_check.py"
)
MARUHAN_E2E_LOG_DIR = LOG_ROOT / "e2e_checks"
MARUHAN_E2E_TIMEOUT_SEC = 60
WRAPPER_SUCCESS_STATUSES = frozenset({"SUCCESS", "ALREADY_COMPLETE", "PROVISIONAL", "STALE_REFERENCE"})
SLEEP_ELIGIBLE_STATUSES = frozenset({"SUCCESS", "ALREADY_COMPLETE"})

STORE_LABELS = {
    STORE_MARUHAN: "Maruhan Mega City Maebashi Inter",
    STORE_BIGMARCH: "Big March Takasaki Oyagi",
    STORE_YASUDA: "Yasuda Maebashi",
    STORE_BICTSUBAME: "Bic Tsubame Takasaki",
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
        "--fetch-timeout-sec",
        type=int,
        default=DEFAULT_FETCH_TIMEOUT_SEC,
        help="Maximum seconds allowed for one Fetch child process. Default: 120.",
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
    if args.fetch_timeout_sec < 1:
        parser.error("--fetch-timeout-sec must be >= 1")
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
        "provisional_attempt_count": 0,
        "provisional_status": "",
        "provisional_reason": "",
        "provisional_latest_data_date": "",
        "provisional_artifacts": [],
        "stale_reference_attempt_count": 0,
        "stale_reference_status": "",
        "stale_reference_reason": "",
        "stale_reference_latest_data_date": "",
        "stale_reference_source_delay_days": "",
        "stale_reference_source_delay_bucket": "",
        "stale_reference_artifacts": [],
        "catchup_attempt_count": 0,
        "catchup_status": "",
        "catchup_reason": "",
        "catchup_target_date": "",
        "catchup_latest_data_date": "",
        "catchup_attempted_dates": [],
        "catchup_converted_dates": [],
        "catchup_source_path": "",
        "catchup_daily_path": "",
        "catchup_source_sha256": "",
        "catchup_daily_sha256": "",
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
        "inventory_guard": {},
        "inventory_monitor": {},
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
        store_state["error"] = "08:30 JST morning finalize deadline has passed."


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
    elif store == STORE_YASUDA:
        candidates = [
            project_root / "data/yasuda_maebashi/machine_number" / f"ana_slo_{expected:%Y%m%d}.csv"
        ]
    elif store == STORE_BICTSUBAME:
        expected, _ = bic_tsubame_expected_latest_data_date(operation_date)
        candidates = [
            project_root
            / "data/bic_tsubame_takasaki/machine_number"
            / f"ana_slo_bic_tsubame_takasaki_{expected:%Y%m%d}.csv"
        ]
    else:
        return False
    return any(path.exists() for path in candidates)


def reconcile_startup_state(
    state: dict,
    project_root: Path,
    operation_date: date,
    current: datetime,
) -> None:
    stores = state.setdefault("stores", {})
    for store in STORE_ORDER:
        if store not in stores:
            stores[store] = blank_store_state(store, operation_date)
        store_state = stores[store]
        store_state.setdefault("provisional_attempt_count", 0)
        store_state.setdefault("provisional_status", "")
        store_state.setdefault("provisional_reason", "")
        store_state.setdefault("provisional_latest_data_date", "")
        store_state.setdefault("provisional_artifacts", [])
        store_state.setdefault("stale_reference_attempt_count", 0)
        store_state.setdefault("stale_reference_status", "")
        store_state.setdefault("stale_reference_reason", "")
        store_state.setdefault("stale_reference_latest_data_date", "")
        store_state.setdefault("stale_reference_source_delay_days", "")
        store_state.setdefault("stale_reference_source_delay_bucket", "")
        store_state.setdefault("stale_reference_artifacts", [])
        store_state.setdefault("catchup_attempt_count", 0)
        store_state.setdefault("catchup_status", "")
        store_state.setdefault("catchup_reason", "")
        store_state.setdefault("catchup_target_date", "")
        store_state.setdefault("catchup_latest_data_date", "")
        store_state.setdefault("catchup_attempted_dates", [])
        store_state.setdefault("catchup_converted_dates", [])
        store_state.setdefault("catchup_source_path", "")
        store_state.setdefault("catchup_daily_path", "")
        store_state.setdefault("catchup_source_sha256", "")
        store_state.setdefault("catchup_daily_sha256", "")
        store_state.setdefault("inventory_guard", {})
        store_state.setdefault("inventory_monitor", {})
        if store == STORE_BIGMARCH:
            recovered_monitor = load_inventory_monitor_evidence(
                project_root / "data/bigmarch_takasaki_oyagi/machine_number",
                operation_date,
            )
            if recovered_monitor:
                store_state["inventory_monitor"] = recovered_monitor

            provisional = verify_big_march_provisional_completion(
                project_root, operation_date
            )
            stale_reference = verify_big_march_stale_reference_completion(
                project_root, operation_date
            )

            if provisional.status != "NONE":
                formal = verify_store_completion(store, project_root, operation_date)
                if formal.status != "NONE" or stale_reference.status != "NONE":
                    store_state.update(
                        status="NEEDS_MANUAL_REVIEW",
                        current_stage="PROVISIONAL_RECONCILIATION",
                        error_category="RANKING_CLASS_CONFLICT",
                        error=(
                            "Formal, PROVISIONAL, and STALE_REFERENCE artifacts "
                            "must be mutually exclusive for one operation date."
                        ),
                        last_completed_at_jst=current.isoformat(),
                        next_retry_at_jst="",
                    )
                elif provisional.ok:
                    inventory_guard = _assess_big_march_inventory_guard_parent(
                        project_root,
                        operation_date,
                        current,
                    )
                    store_state["inventory_guard"] = inventory_guard
                    if not inventory_guard["provisional_allowed"]:
                        store_state.update(
                            status="NEEDS_MANUAL_REVIEW",
                            current_stage="INVENTORY_GUARD",
                            provisional_status="BLOCKED_BY_INVENTORY_GUARD",
                            provisional_reason=inventory_guard["reason"],
                            provisional_latest_data_date=provisional.details.get(
                                "latest_data_date", ""
                            ),
                            provisional_artifacts=provisional.artifacts,
                            verified_artifacts=[],
                            error_category="INVENTORY_GUARD_BLOCKED",
                            error=inventory_guard["reason"],
                            next_retry_at_jst="",
                            last_completed_at_jst=current.isoformat(),
                        )
                    else:
                        store_state.update(
                            status="PROVISIONAL",
                            current_stage="PROVISIONAL",
                            provisional_status="ALREADY_PROVISIONAL",
                            provisional_reason="Recovered verified provisional artifacts.",
                            provisional_latest_data_date=provisional.details.get(
                                "latest_data_date", ""
                            ),
                            provisional_artifacts=provisional.artifacts,
                            verified_artifacts=[],
                            error_category="",
                            error="",
                            next_retry_at_jst="",
                            last_completed_at_jst=current.isoformat(),
                        )
                else:
                    store_state.update(
                        status="NEEDS_MANUAL_REVIEW",
                        current_stage="PROVISIONAL_RECONCILIATION",
                        provisional_status=provisional.status,
                        provisional_reason=provisional.error,
                        provisional_artifacts=provisional.artifacts,
                        error_category=provisional.status,
                        error=provisional.error,
                        next_retry_at_jst="",
                        last_completed_at_jst=current.isoformat(),
                    )
                continue

            if stale_reference.status != "NONE":
                formal = verify_store_completion(store, project_root, operation_date)
                if formal.status != "NONE":
                    store_state.update(
                        status="NEEDS_MANUAL_REVIEW",
                        current_stage="STALE_REFERENCE_RECONCILIATION",
                        error_category="FORMAL_STALE_REFERENCE_CONFLICT",
                        error=(
                            "Formal and STALE_REFERENCE artifacts both exist "
                            "for this operation date."
                        ),
                        last_completed_at_jst=current.isoformat(),
                        next_retry_at_jst="",
                    )
                elif stale_reference.ok:
                    inventory_guard = _assess_big_march_inventory_guard_parent(
                        project_root,
                        operation_date,
                        current,
                    )
                    store_state["inventory_guard"] = inventory_guard
                    if not inventory_guard["reference_allowed"]:
                        store_state.update(
                            status="NEEDS_MANUAL_REVIEW",
                            current_stage="INVENTORY_GUARD",
                            stale_reference_status="BLOCKED_BY_INVENTORY_GUARD",
                            stale_reference_reason=inventory_guard["reason"],
                            stale_reference_latest_data_date=stale_reference.details.get(
                                "latest_data_date", ""
                            ),
                            stale_reference_source_delay_days=stale_reference.details.get(
                                "source_delay_days", ""
                            ),
                            stale_reference_source_delay_bucket=stale_reference.details.get(
                                "source_delay_bucket", ""
                            ),
                            stale_reference_artifacts=stale_reference.artifacts,
                            verified_artifacts=[],
                            error_category="INVENTORY_GUARD_BLOCKED",
                            error=inventory_guard["reason"],
                            next_retry_at_jst="",
                            last_completed_at_jst=current.isoformat(),
                        )
                    else:
                        store_state.update(
                            status="STALE_REFERENCE",
                            current_stage="STALE_REFERENCE",
                            stale_reference_status="ALREADY_STALE_REFERENCE",
                            stale_reference_reason=(
                                "Recovered verified STALE_REFERENCE artifacts."
                            ),
                            stale_reference_latest_data_date=stale_reference.details.get(
                                "latest_data_date", ""
                            ),
                            stale_reference_source_delay_days=stale_reference.details.get(
                                "source_delay_days", ""
                            ),
                            stale_reference_source_delay_bucket=stale_reference.details.get(
                                "source_delay_bucket", ""
                            ),
                            stale_reference_artifacts=stale_reference.artifacts,
                            verified_artifacts=[],
                            error_category="",
                            error="",
                            next_retry_at_jst="",
                            last_completed_at_jst=current.isoformat(),
                        )
                else:
                    store_state.update(
                        status="NEEDS_MANUAL_REVIEW",
                        current_stage="STALE_REFERENCE_RECONCILIATION",
                        stale_reference_status=stale_reference.status,
                        stale_reference_reason=stale_reference.error,
                        stale_reference_artifacts=stale_reference.artifacts,
                        error_category=stale_reference.status,
                        error=stale_reference.error,
                        next_retry_at_jst="",
                        last_completed_at_jst=current.isoformat(),
                    )
                continue
        if store == STORE_YASUDA:
            yasuda_verification = verify_store_completion(
                store, project_root, operation_date
            )
            if (
                yasuda_verification.ok
                and store_state["status"] in {"SUCCESS", "ALREADY_COMPLETE"}
            ):
                inventory_guard = assess_yasuda_inventory_guard(
                    project_root / "data/yasuda_maebashi/machine_number",
                    operation_date,
                    operation_date - timedelta(days=1),
                )
                store_state["inventory_guard"] = inventory_guard.to_dict()
                if inventory_guard.blocked:
                    store_state.update(
                        status="NEEDS_MANUAL_REVIEW",
                        current_stage="INVENTORY_GUARD",
                        error_category="INVENTORY_GUARD_BLOCKED",
                        error=inventory_guard.summary(),
                        verified_artifacts=yasuda_verification.artifacts,
                        last_completed_at_jst=current.isoformat(),
                        next_retry_at_jst="",
                    )
                    continue
        if store == STORE_BICTSUBAME:
            bic_verification = verify_store_completion(store, project_root, operation_date)
            if (
                bic_verification.ok
                and store_state["status"] in {"SUCCESS", "ALREADY_COMPLETE"}
            ):
                inventory_guard = assess_bic_tsubame_inventory_guard(
                    project_root / "data/bic_tsubame_takasaki/machine_number",
                    operation_date,
                    operation_date - timedelta(days=1),
                )
                store_state["inventory_guard"] = inventory_guard.to_dict()
                if inventory_guard.blocked:
                    store_state.update(
                        status="NEEDS_MANUAL_REVIEW",
                        current_stage="INVENTORY_GUARD",
                        error_category="INVENTORY_GUARD_BLOCKED",
                        error=inventory_guard.summary(),
                        verified_artifacts=bic_verification.artifacts,
                        last_completed_at_jst=current.isoformat(),
                        next_retry_at_jst="",
                    )
                    continue
        if store == STORE_MARUHAN:
            inventory_guard = assess_inventory_guard(
                project_root / "data/maruhan_maebashi/machine_number",
                operation_date,
                operation_date - timedelta(days=1),
            )
            store_state["inventory_guard"] = inventory_guard.to_dict()
            existing_inventory_sensitive_output = False
            if inventory_guard.blocked:
                existing_inventory_sensitive_output = (
                    verify_store_completion(
                        store, project_root, operation_date
                    ).status
                    != "NONE"
                )
            if inventory_guard.blocked and (
                store_state["status"] in {"SUCCESS", "ALREADY_COMPLETE"}
                or existing_inventory_sensitive_output
            ):
                store_state.update(
                    status="NEEDS_MANUAL_REVIEW",
                    current_stage="INVENTORY_GUARD",
                    error_category="INVENTORY_GUARD_BLOCKED",
                    error=inventory_guard.summary(),
                    last_completed_at_jst=current.isoformat(),
                    next_retry_at_jst="",
                )
                continue
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
        elif store == STORE_BICTSUBAME and verification.ok:
            store_state.update(
                status="ALREADY_COMPLETE",
                error_category="",
                error="",
                verified_artifacts=verification.artifacts,
                latest_data_date=verification.details.get("latest_data_date", ""),
                last_completed_at_jst=current.isoformat(),
                next_retry_at_jst="",
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
            if store_state.get("current_stage") in {"FETCH", "CDP_PREFLIGHT", "READINESS", "CATCHUP_CONVERT"}:
                store_state.update(
                    status="FAILED_RETRYABLE",
                    error_category=(
                        "RECOVERED_CATCHUP_INTERRUPTION"
                        if store_state.get("current_stage") == "CATCHUP_CONVERT"
                        else "RECOVERED_PRE_PIPELINE_INTERRUPTION"
                    ),
                    error="Previous process ended before formal pipeline start.",
                )
            else:
                store_state.update(
                    status="NEEDS_MANUAL_REVIEW",
                    error_category="INTERRUPTED_PIPELINE_UNKNOWN",
                    error="Previous state was RUNNING after pipeline start or at an unknown stage.",
                    last_completed_at_jst=current.isoformat(),
                )


def _assess_big_march_inventory_guard_parent(
    project_root: Path,
    operation_date: date,
    current: datetime,
) -> dict:
    """Parent-side authoritative Big March Inventory Guard snapshot."""
    try:
        decision = assess_big_march_inventory_guard(
            project_root / "data/bigmarch_takasaki_oyagi/machine_number",
            operation_date,
            generated_at_jst=current.astimezone(JST),
        )
    except Exception as exc:
        return {
            "status": "BLOCKED_GUARD_ERROR",
            "blocked": True,
            "formal_allowed": False,
            "provisional_allowed": False,
            "reference_allowed": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "operation_date": operation_date.isoformat(),
            "expected_data_date": (operation_date - timedelta(days=1)).isoformat(),
            "latest_data_date": "",
            "source_delay_days": "",
            "known_change_date": False,
            "comparison_status": "",
            "monitor_status": "ERROR",
        }

    def _date_text(value) -> str:
        if value is None:
            return ""
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    return {
        "status": decision.status,
        "blocked": bool(decision.blocked),
        "formal_allowed": bool(decision.formal_allowed),
        "provisional_allowed": bool(decision.provisional_allowed),
        "reference_allowed": bool(decision.reference_allowed),
        "reason": decision.reason,
        "operation_date": _date_text(decision.operation_date),
        "expected_data_date": (
            operation_date - timedelta(days=1)
        ).isoformat(),
        "latest_data_date": _date_text(decision.latest_data_date),
        "source_delay_days": decision.source_delay_days,
        "known_change_date": bool(decision.known_change_date),
        "comparison_status": decision.comparison_status,
        "monitor_status": decision.monitor_status,
    }


def _observe_big_march_inventory(
    project_root: Path, operation_date: date, current: datetime
) -> dict:
    """Observe inventory without allowing monitor status to affect control flow."""
    result = observe_big_march_inventory_best_effort(
        project_root / "data/bigmarch_takasaki_oyagi/machine_number",
        operation_date,
        generated_at_jst=current.astimezone(JST),
    )
    if result.get("status") in {
        "ERROR", "SOURCE_CHANGED_AFTER_OBSERVATION", "COMPARISON_UNAVAILABLE"
    }:
        print(
            "WARNING: Big March inventory monitor: "
            f"{result.get('status')}: {result.get('error', '')}",
            file=sys.stderr,
        )
    return result


def _run_big_march_provisional(
    state: dict,
    state_path: Path,
    operation_date: date,
    current: datetime,
    clock=now_jst,
) -> None:
    store_state = state["stores"][STORE_BIGMARCH]
    if store_state.get("pipeline_attempt_count", 0) > 0:
        store_state.update(
            status="FAILED_FINAL",
            error_category="PROVISIONAL_FORBIDDEN_AFTER_FORMAL_PIPELINE",
            error="Formal pipeline was already attempted; provisional is forbidden.",
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        save_state(state_path, state, current)
        return
    inventory_guard = _assess_big_march_inventory_guard_parent(
        PROJECT_ROOT,
        operation_date,
        current,
    )
    store_state["inventory_guard"] = inventory_guard
    if not inventory_guard["provisional_allowed"]:
        store_state.update(
            status="NEEDS_MANUAL_REVIEW",
            current_stage="INVENTORY_GUARD",
            provisional_status="BLOCKED_BY_INVENTORY_GUARD",
            provisional_reason=inventory_guard["reason"],
            error_category="INVENTORY_GUARD_BLOCKED",
            error=inventory_guard["reason"],
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        save_state(state_path, state, current)
        return

    if store_state.get("provisional_attempt_count", 0) > 0:
        verification = verify_big_march_provisional_completion(
            PROJECT_ROOT, operation_date
        )
        store_state.update(
            status="PROVISIONAL" if verification.ok else "NEEDS_MANUAL_REVIEW",
            current_stage="PROVISIONAL",
            provisional_status=("ALREADY_PROVISIONAL" if verification.ok else verification.status),
            provisional_reason=("Verified existing provisional artifacts." if verification.ok else verification.error),
            provisional_latest_data_date=verification.details.get("latest_data_date", ""),
            provisional_artifacts=verification.artifacts,
            error_category="" if verification.ok else verification.status,
            error="" if verification.ok else verification.error,
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        save_state(state_path, state, current)
        return
    attempt = 1
    store_state.update(
        status="RUNNING",
        current_stage="PROVISIONAL",
        attempt_count=store_state.get("attempt_count", 0) + 1,
        provisional_attempt_count=attempt,
        provisional_status="RUNNING",
        provisional_reason="08:00 morning snapshot found the expected source missing.",
        last_started_at_jst=current.isoformat(),
        next_retry_at_jst="",
        error_category="",
        error="",
    )
    save_state(state_path, state, current)
    command = build_big_march_provisional_command(
        PROJECT_ROOT, sys.executable, operation_date
    )
    log_path = (
        RUNS_DIR
        / operation_date.strftime("%Y%m%d")
        / state["automation_run_id"]
        / "bigmarch_provisional_attempt01.log"
    )
    environment = {
        **os.environ,
        "SLOTANALYZER_MORNING_RUN_ID": state["automation_run_id"],
    }
    result = run_logged_subprocess(
        command, PROJECT_ROOT, log_path, "PROVISIONAL", environment=environment, clock=clock
    )
    verification = verify_big_march_provisional_completion(
        PROJECT_ROOT, operation_date
    )
    completed = clock().astimezone(JST)
    if result.returncode == 0 and verification.ok:
        store_state["inventory_monitor"] = _observe_big_march_inventory(
            PROJECT_ROOT, operation_date, completed
        )
        status = "PROVISIONAL"
        category = ""
        error = ""
    elif verification.status in {"PARTIAL_PROVISIONAL", "INVALID_PROVISIONAL"}:
        status = "NEEDS_MANUAL_REVIEW"
        category = verification.status
        error = verification.error
    else:
        status = "FAILED_FINAL"
        category = "PROVISIONAL_NOT_ELIGIBLE_OR_FAILED"
        error = verification.error or "Provisional generator returned non-zero."
    store_state.update(
        status=status,
        current_stage="PROVISIONAL",
        returncode=result.returncode,
        pipeline_attempt_count=0,
        provisional_status=("PROVISIONAL" if status == "PROVISIONAL" else verification.status),
        provisional_reason=("Expected source missing; isolated provisional ranking generated." if status == "PROVISIONAL" else error),
        provisional_latest_data_date=verification.details.get("latest_data_date", ""),
        provisional_artifacts=verification.artifacts,
        verified_artifacts=[],
        error_category=category,
        error=error,
        next_retry_at_jst="",
        last_completed_at_jst=completed.isoformat(),
    )
    _record_history(
        state, STORE_BIGMARCH, "PROVISIONAL", attempt, status, result, category, error
    )
    save_state(state_path, state, completed)


def _run_big_march_stale_reference(
    state: dict,
    state_path: Path,
    operation_date: date,
    current: datetime,
    clock=now_jst,
) -> None:
    store_state = state["stores"][STORE_BIGMARCH]

    if store_state.get("pipeline_attempt_count", 0) > 0:
        store_state.update(
            status="FAILED_FINAL",
            error_category="STALE_REFERENCE_FORBIDDEN_AFTER_FORMAL_PIPELINE",
            error="Formal pipeline was already attempted; STALE_REFERENCE is forbidden.",
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        save_state(state_path, state, current)
        return

    if store_state.get("provisional_attempt_count", 0) > 0:
        store_state.update(
            status="FAILED_FINAL",
            error_category="STALE_REFERENCE_FORBIDDEN_AFTER_PROVISIONAL",
            error="PROVISIONAL was already attempted; STALE_REFERENCE is forbidden.",
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        save_state(state_path, state, current)
        return

    inventory_guard = _assess_big_march_inventory_guard_parent(
        PROJECT_ROOT,
        operation_date,
        current,
    )
    store_state["inventory_guard"] = inventory_guard

    if not inventory_guard["reference_allowed"]:
        store_state.update(
            status="NEEDS_MANUAL_REVIEW",
            current_stage="INVENTORY_GUARD",
            stale_reference_status="BLOCKED_BY_INVENTORY_GUARD",
            stale_reference_reason=inventory_guard["reason"],
            error_category="INVENTORY_GUARD_BLOCKED",
            error=inventory_guard["reason"],
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        save_state(state_path, state, current)
        return

    if store_state.get("stale_reference_attempt_count", 0) > 0:
        verification = verify_big_march_stale_reference_completion(
            PROJECT_ROOT,
            operation_date,
        )
        store_state.update(
            status="STALE_REFERENCE" if verification.ok else "NEEDS_MANUAL_REVIEW",
            current_stage="STALE_REFERENCE",
            stale_reference_status=(
                "ALREADY_STALE_REFERENCE" if verification.ok else verification.status
            ),
            stale_reference_reason=(
                "Verified existing STALE_REFERENCE artifacts."
                if verification.ok
                else verification.error
            ),
            stale_reference_latest_data_date=verification.details.get(
                "latest_data_date", ""
            ),
            stale_reference_source_delay_days=verification.details.get(
                "source_delay_days", ""
            ),
            stale_reference_source_delay_bucket=verification.details.get(
                "source_delay_bucket", ""
            ),
            stale_reference_artifacts=verification.artifacts,
            verified_artifacts=[],
            error_category="" if verification.ok else verification.status,
            error="" if verification.ok else verification.error,
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        save_state(state_path, state, current)
        return

    attempt = 1
    store_state.update(
        status="RUNNING",
        current_stage="STALE_REFERENCE",
        attempt_count=store_state.get("attempt_count", 0) + 1,
        stale_reference_attempt_count=attempt,
        stale_reference_status="RUNNING",
        stale_reference_reason=(
            "08:00 morning snapshot found Big March source delay of two days or more."
        ),
        last_started_at_jst=current.isoformat(),
        next_retry_at_jst="",
        error_category="",
        error="",
    )
    save_state(state_path, state, current)

    command = build_big_march_stale_reference_command(
        PROJECT_ROOT,
        sys.executable,
        operation_date,
    )
    log_path = (
        RUNS_DIR
        / operation_date.strftime("%Y%m%d")
        / state["automation_run_id"]
        / "bigmarch_stale_reference_attempt01.log"
    )
    environment = {
        **os.environ,
        "SLOTANALYZER_MORNING_RUN_ID": state["automation_run_id"],
    }
    result = run_logged_subprocess(
        command,
        PROJECT_ROOT,
        log_path,
        "STALE_REFERENCE",
        environment=environment,
        clock=clock,
    )
    verification = verify_big_march_stale_reference_completion(
        PROJECT_ROOT,
        operation_date,
    )
    completed = clock().astimezone(JST)

    if result.returncode == 0 and verification.ok:
        store_state["inventory_monitor"] = _observe_big_march_inventory(
            PROJECT_ROOT,
            operation_date,
            completed,
        )
        status = "STALE_REFERENCE"
        category = ""
        error = ""
    elif verification.status in {
        "PARTIAL_STALE_REFERENCE",
        "INVALID_STALE_REFERENCE",
    }:
        status = "NEEDS_MANUAL_REVIEW"
        category = verification.status
        error = verification.error
    else:
        status = "FAILED_FINAL"
        category = "STALE_REFERENCE_NOT_ELIGIBLE_OR_FAILED"
        error = verification.error or "STALE_REFERENCE generator returned non-zero."

    store_state.update(
        status=status,
        current_stage="STALE_REFERENCE",
        returncode=result.returncode,
        pipeline_attempt_count=0,
        stale_reference_status=(
            "STALE_REFERENCE" if status == "STALE_REFERENCE" else verification.status
        ),
        stale_reference_reason=(
            "08:00 snapshot fixed an isolated STALE_REFERENCE ranking."
            if status == "STALE_REFERENCE"
            else error
        ),
        stale_reference_latest_data_date=verification.details.get(
            "latest_data_date", ""
        ),
        stale_reference_source_delay_days=verification.details.get(
            "source_delay_days", ""
        ),
        stale_reference_source_delay_bucket=verification.details.get(
            "source_delay_bucket", ""
        ),
        stale_reference_artifacts=verification.artifacts,
        verified_artifacts=[],
        error_category=category,
        error=error,
        next_retry_at_jst="",
        last_completed_at_jst=completed.isoformat(),
    )
    _record_history(
        state,
        STORE_BIGMARCH,
        "STALE_REFERENCE",
        attempt,
        status,
        result,
        category,
        error,
    )
    save_state(state_path, state, completed)


def _run_big_march_catchup(
    state: dict,
    state_path: Path,
    operation_date: date,
    expected_data_date: date,
    current: datetime,
    clock=now_jst,
) -> bool:
    """Convert only contiguous pre-expected sources; never run a formal stage."""
    store_state = state["stores"][STORE_BIGMARCH]
    assessment = assess_big_march_catchup(
        PROJECT_ROOT, operation_date, expected_data_date, 200
    )
    if not assessment.ok:
        if assessment.status == "CATCHUP_NO_DAILY_BASELINE":
            return True
        store_state.update(
            status="NEEDS_MANUAL_REVIEW",
            current_stage="CATCHUP_CONVERT",
            catchup_status=assessment.status,
            catchup_reason=assessment.error,
            catchup_latest_data_date=assessment.latest_daily_date,
            error_category=assessment.status,
            error=assessment.error,
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        _record_history(
            state, STORE_BIGMARCH, "CATCHUP_CONVERT",
            store_state.get("catchup_attempt_count", 0),
            "NEEDS_MANUAL_REVIEW", error_category=assessment.status,
            error=assessment.error,
        )
        save_state(state_path, state, current)
        return False

    store_state["catchup_latest_data_date"] = assessment.latest_daily_date
    if not assessment.candidate_dates:
        if assessment.latest_daily_date == (
            expected_data_date - timedelta(days=1)
        ).isoformat():
            verification = verify_big_march_daily_for_date(
                PROJECT_ROOT, date.fromisoformat(assessment.latest_daily_date), 200
            )
            if not verification.ok:
                store_state.update(
                    status="NEEDS_MANUAL_REVIEW",
                    current_stage="CATCHUP_CONVERT",
                    catchup_status=verification.status,
                    catchup_reason=verification.error,
                    error_category=verification.status,
                    error=verification.error,
                    last_completed_at_jst=current.isoformat(),
                    next_retry_at_jst="",
                )
                save_state(state_path, state, current)
                return False
            store_state.update(
                catchup_status="RECOVERED_CATCHUP",
                catchup_reason="Latest contiguous pre-expected daily is already valid.",
                catchup_target_date=assessment.latest_daily_date,
                catchup_converted_dates=list(dict.fromkeys([
                    *store_state.get("catchup_converted_dates", []),
                    assessment.latest_daily_date,
                ])),
                catchup_daily_path=verification.artifacts[0],
                catchup_daily_sha256=verification.details.get("sha256", ""),
            )
            store_state["inventory_monitor"] = _observe_big_march_inventory(
                PROJECT_ROOT, operation_date, current
            )
        return True

    for candidate_text in assessment.candidate_dates:
        candidate = date.fromisoformat(candidate_text)
        readiness = check_source_readiness(STORE_BIGMARCH, PROJECT_ROOT, candidate)
        if not readiness.ready:
            store_state.update(
                status="NEEDS_MANUAL_REVIEW",
                current_stage="CATCHUP_CONVERT",
                catchup_status="CATCHUP_SOURCE_INVALID",
                catchup_reason=readiness.error or "Catch-up source became unavailable.",
                error_category="CATCHUP_SOURCE_INVALID",
                error=readiness.error or "Catch-up source became unavailable.",
                last_completed_at_jst=current.isoformat(),
                next_retry_at_jst="",
            )
            save_state(state_path, state, current)
            return False

        attempt = store_state.get("catchup_attempt_count", 0) + 1
        attempted_dates = list(store_state.get("catchup_attempted_dates", []))
        if candidate_text not in attempted_dates:
            attempted_dates.append(candidate_text)
        source_path = Path(readiness.source_path)
        store_state.update(
            status="RUNNING",
            current_stage="CATCHUP_CONVERT",
            attempt_count=store_state.get("attempt_count", 0) + 1,
            catchup_attempt_count=attempt,
            catchup_status="RUNNING",
            catchup_reason="",
            catchup_target_date=candidate_text,
            catchup_attempted_dates=attempted_dates,
            catchup_source_path=str(source_path),
            catchup_source_sha256=sha256_file(source_path),
            error_category="",
            error="",
            last_started_at_jst=current.isoformat(),
        )
        save_state(state_path, state, current)
        command = build_big_march_catchup_command(
            PROJECT_ROOT, sys.executable, candidate, 200
        )
        log_path = (
            RUNS_DIR
            / operation_date.strftime("%Y%m%d")
            / state["automation_run_id"]
            / f"bigmarch_catchup_{candidate:%Y%m%d}_attempt{attempt:02d}.log"
        )
        environment = {
            **os.environ,
            "SLOTANALYZER_MORNING_RUN_ID": state["automation_run_id"],
        }
        result = run_logged_subprocess(
            command, PROJECT_ROOT, log_path, "CATCHUP_CONVERT",
            environment=environment, clock=clock,
        )
        verification = verify_big_march_daily_for_date(
            PROJECT_ROOT, candidate, 200
        )
        completed = clock().astimezone(JST)
        if result.returncode != 0 or not verification.ok:
            category = verification.status if not verification.ok else "CATCHUP_CONVERTER_NONZERO"
            error = verification.error or f"Catch-up converter returned {result.returncode}."
            store_state.update(
                status="NEEDS_MANUAL_REVIEW",
                current_stage="CATCHUP_CONVERT",
                catchup_status=category,
                catchup_reason=error,
                returncode=result.returncode,
                error_category=category,
                error=error,
                last_completed_at_jst=completed.isoformat(),
                next_retry_at_jst="",
            )
            _record_history(
                state, STORE_BIGMARCH, "CATCHUP_CONVERT", attempt,
                "NEEDS_MANUAL_REVIEW", result, category, error,
            )
            save_state(state_path, state, completed)
            return False

        store_state["inventory_monitor"] = _observe_big_march_inventory(
            PROJECT_ROOT, operation_date, completed
        )

        converted_dates = list(store_state.get("catchup_converted_dates", []))
        if candidate_text not in converted_dates:
            converted_dates.append(candidate_text)
        store_state.update(
            status="WAITING_FOR_DATA",
            current_stage="CATCHUP_CONVERT",
            catchup_status="CATCHUP_SUCCESS",
            catchup_reason="Catch-up daily validated; expected source wait continues.",
            catchup_latest_data_date=candidate_text,
            catchup_converted_dates=converted_dates,
            catchup_daily_path=verification.artifacts[0],
            catchup_daily_sha256=verification.details.get("sha256", ""),
            returncode=result.returncode,
            error_category="EXPECTED_SOURCE_NOT_READY",
            error="Expected source is still not ready.",
            last_completed_at_jst=completed.isoformat(),
        )
        _record_history(
            state, STORE_BIGMARCH, "CATCHUP_CONVERT", attempt,
            "CATCHUP_SUCCESS", result,
        )
        save_state(state_path, state, completed)
        current = completed
    if assessment.status == "CATCHUP_READY_THEN_SOURCE_INVALID":
        store_state.update(
            status="NEEDS_MANUAL_REVIEW",
            current_stage="CATCHUP_CONVERT",
            catchup_status="CATCHUP_SOURCE_INVALID",
            catchup_reason=assessment.error,
            error_category="CATCHUP_SOURCE_INVALID",
            error=assessment.error,
            last_completed_at_jst=current.isoformat(),
            next_retry_at_jst="",
        )
        _record_history(
            state, STORE_BIGMARCH, "CATCHUP_CONVERT",
            store_state.get("catchup_attempt_count", 0),
            "NEEDS_MANUAL_REVIEW",
            error_category="CATCHUP_SOURCE_INVALID", error=assessment.error,
        )
        save_state(state_path, state, current)
        return False
    return True


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
    if store == STORE_BICTSUBAME:
        expected_data_date, _ = bic_tsubame_expected_latest_data_date(operation_date)
    if store_state["status"] in TERMINAL_STATES:
        return
    current = clock().astimezone(JST)

    if store == STORE_BIGMARCH:
        # Big March is a one-shot morning snapshot. Once this branch runs,
        # the wrapper never performs a network fetch or schedules a later retry
        # for Big March on the same operation date.
        readiness = check_source_readiness(
            store,
            PROJECT_ROOT,
            expected_data_date,
        )
        store_state["current_stage"] = "READINESS"
        store_state["last_readiness"] = readiness.to_dict()
        store_state["next_retry_at_jst"] = ""

        if readiness.source_exists and not readiness.ready:
            store_state.update(
                status="NEEDS_MANUAL_REVIEW",
                error_category="SOURCE_INVALID",
                error=readiness.error,
                last_completed_at_jst=current.isoformat(),
                next_retry_at_jst="",
            )
            save_state(state_path, state, current)
            return

        if not readiness.ready:
            if not _run_big_march_catchup(
                state,
                state_path,
                operation_date,
                expected_data_date,
                current,
                clock,
            ):
                return

            current = clock().astimezone(JST)
            inventory_guard = _assess_big_march_inventory_guard_parent(
                PROJECT_ROOT,
                operation_date,
                current,
            )
            store_state["inventory_guard"] = inventory_guard
            store_state["latest_data_date"] = inventory_guard.get(
                "latest_data_date", ""
            )

            if inventory_guard["provisional_allowed"]:
                _run_big_march_provisional(
                    state,
                    state_path,
                    operation_date,
                    current,
                    clock,
                )
                return

            if inventory_guard["reference_allowed"]:
                _run_big_march_stale_reference(
                    state,
                    state_path,
                    operation_date,
                    current,
                    clock,
                )
                return

            store_state.update(
                status="NEEDS_MANUAL_REVIEW",
                current_stage="INVENTORY_GUARD",
                error_category="INVENTORY_GUARD_BLOCKED",
                error=inventory_guard["reason"],
                last_completed_at_jst=current.isoformat(),
                next_retry_at_jst="",
            )
            save_state(state_path, state, current)
            return

        store_state["latest_data_date"] = expected_data_date.isoformat()

    next_retry = store_state.get("next_retry_at_jst")
    if next_retry and current < datetime.fromisoformat(next_retry):
        return
    if not _deadline_open(store, current, operation_date):
        if store == STORE_BIGMARCH:
            readiness = check_source_readiness(store, PROJECT_ROOT, expected_data_date)
            store_state["current_stage"] = "READINESS"
            store_state["last_readiness"] = readiness.to_dict()
            if readiness.source_exists and not readiness.ready:
                store_state.update(
                    status="NEEDS_MANUAL_REVIEW",
                    error_category="SOURCE_INVALID",
                    error=readiness.error,
                    last_completed_at_jst=current.isoformat(),
                    next_retry_at_jst="",
                )
                save_state(state_path, state, current)
                return
            if not readiness.ready:
                if not _run_big_march_catchup(
                    state, state_path, operation_date, expected_data_date,
                    current, clock,
                ):
                    return
                _run_big_march_provisional(
                    state, state_path, operation_date, current, clock
                )
                return
        else:
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
        if store == STORE_BIGMARCH and not _run_big_march_catchup(
            state, state_path, operation_date, expected_data_date,
            current, clock,
        ):
            return
        if store_state["fetch_attempt_count"] >= (MARUHAN_MAX_FETCH_ATTEMPTS if store == STORE_MARUHAN else args.max_fetch_attempts):
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
                next_retry_at_jst=(completed + timedelta(seconds=(MARUHAN_RETRY_INTERVAL_SEC if store == STORE_MARUHAN else args.retry_interval_sec))).isoformat(),
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
        command = build_fetch_command(
            store, PROJECT_ROOT, sys.executable, expected_data_date
        )
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
            "PYTHONUNBUFFERED": "1",
        }
        result = run_logged_subprocess(
            command,
            PROJECT_ROOT,
            log_path,
            "FETCH",
            environment=environment,
            clock=clock,
            timeout_sec=args.fetch_timeout_sec,
        )
        readiness = check_source_readiness(store, PROJECT_ROOT, expected_data_date)
        current = clock().astimezone(JST)
        store_state["last_readiness"] = readiness.to_dict()
        fetch_window_open = _deadline_open(store, current, operation_date)
        classification = classify_fetch_result(
            result.returncode, readiness, fetch_window_open
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
            if store == STORE_BIGMARCH and not readiness.source_exists:
                if not _run_big_march_catchup(
                    state, state_path, operation_date, expected_data_date,
                    current, clock,
                ):
                    return
            store_state.update(
                status=classification,
                current_stage="FETCH",
                returncode=result.returncode,
                next_retry_at_jst=(current + timedelta(seconds=(MARUHAN_RETRY_INTERVAL_SEC if store == STORE_MARUHAN else args.retry_interval_sec))).isoformat(),
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
    if store == STORE_MARUHAN:
        inventory_guard = assess_inventory_guard(
            PROJECT_ROOT / "data/maruhan_maebashi/machine_number",
            operation_date,
            expected_data_date,
        )
        store_state["inventory_guard"] = inventory_guard.to_dict()
        store_state["auto_triage"] = not_applicable_triage(
            store=store, operation_date=operation_date,
            reason="Maruhan is excluded from MANUAL_REVIEW Auto-Triage V1.",
        )
        # The one-click child performs the authoritative stop after fetch,
        # conversion, and Freshness, but before 69/63/79. Keeping this
        # preflight result in state makes the reason visible while preserving
        # collection of the latest validated daily CSV.
    elif store == STORE_BIGMARCH:
        inventory_guard = _assess_big_march_inventory_guard_parent(
            PROJECT_ROOT,
            operation_date,
            current,
        )
        store_state["inventory_guard"] = inventory_guard
        store_state["auto_triage"] = not_applicable_triage(
            store=store, operation_date=operation_date,
            reason="Big March is excluded from MANUAL_REVIEW Auto-Triage V1.",
        )
        if not inventory_guard["formal_allowed"]:
            store_state.update(
                status="NEEDS_MANUAL_REVIEW",
                current_stage="INVENTORY_GUARD",
                error_category="INVENTORY_GUARD_BLOCKED",
                error=inventory_guard["reason"],
                last_completed_at_jst=current.isoformat(),
                next_retry_at_jst="",
            )
            save_state(state_path, state, current)
            return
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
    inventory_guard = None
    if store == STORE_MARUHAN:
        inventory_guard = assess_inventory_guard(
            PROJECT_ROOT / "data/maruhan_maebashi/machine_number",
            operation_date,
            expected_data_date,
        )
        store_state["inventory_guard"] = inventory_guard.to_dict()
        if inventory_guard.blocked:
            classification = "NEEDS_MANUAL_REVIEW"
    elif store == STORE_YASUDA and verification.ok:
        inventory_guard = assess_yasuda_inventory_guard(
            PROJECT_ROOT / "data/yasuda_maebashi/machine_number",
            operation_date,
            expected_data_date,
        )
        store_state["inventory_guard"] = inventory_guard.to_dict()
        store_state["auto_triage"] = assess_manual_review_auto_triage(
            store=store, data_dir=PROJECT_ROOT / "data/yasuda_maebashi/machine_number",
            operation_date=operation_date, latest_data_date=expected_data_date,
            inventory_guard=inventory_guard, freshness_quality_ok=verification.ok,
            baseline_count=320, generated_at=clock().astimezone(JST),
        )
        if inventory_guard.blocked:
            classification = "NEEDS_MANUAL_REVIEW"
    elif store == STORE_BICTSUBAME and verification.ok:
        inventory_guard = assess_bic_tsubame_inventory_guard(
            PROJECT_ROOT / "data/bic_tsubame_takasaki/machine_number",
            operation_date,
            expected_data_date,
        )
        store_state["inventory_guard"] = inventory_guard.to_dict()
        store_state["auto_triage"] = assess_manual_review_auto_triage(
            store=store, data_dir=PROJECT_ROOT / "data/bic_tsubame_takasaki/machine_number",
            operation_date=operation_date, latest_data_date=expected_data_date,
            inventory_guard=inventory_guard, freshness_quality_ok=verification.ok,
            baseline_count=517, generated_at=clock().astimezone(JST),
        )
        if inventory_guard.blocked:
            classification = "NEEDS_MANUAL_REVIEW"
    elif store == STORE_BIGMARCH and verification.ok:
        big_march_guard = _assess_big_march_inventory_guard_parent(
            PROJECT_ROOT,
            operation_date,
            clock().astimezone(JST),
        )
        store_state["inventory_guard"] = big_march_guard
        store_state["auto_triage"] = not_applicable_triage(
            store=store, operation_date=operation_date,
            reason="Big March is excluded from MANUAL_REVIEW Auto-Triage V1.",
        )
        if not big_march_guard["formal_allowed"]:
            classification = "NEEDS_MANUAL_REVIEW"
            inventory_guard = big_march_guard
        else:
            store_state["inventory_monitor"] = _observe_big_march_inventory(
                PROJECT_ROOT, operation_date, clock().astimezone(JST)
            )
    completed = clock().astimezone(JST)
    store_state.update(
        status=classification,
        current_stage=(
            "INVENTORY_GUARD"
            if (
                inventory_guard is not None
                and (
                    inventory_guard.get("blocked", False)
                    if isinstance(inventory_guard, dict)
                    else inventory_guard.blocked
                )
            )
            else "PIPELINE"
        ),
        returncode=result.returncode,
        last_completed_at_jst=completed.isoformat(),
        error_category=(
            "INVENTORY_GUARD_BLOCKED"
            if (
                inventory_guard is not None
                and (
                    inventory_guard.get("blocked", False)
                    if isinstance(inventory_guard, dict)
                    else inventory_guard.blocked
                )
            )
            else ("" if classification == "SUCCESS" else verification.status or "PIPELINE_FAILED")
        ),
        error=(
            (
                inventory_guard.get("reason", "")
                if isinstance(inventory_guard, dict)
                else inventory_guard.summary()
            )
            if (
                inventory_guard is not None
                and (
                    inventory_guard.get("blocked", False)
                    if isinstance(inventory_guard, dict)
                    else inventory_guard.blocked
                )
            )
            else ("" if classification == "SUCCESS" else verification.error or "One-click returned non-zero.")
        ),
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
            item["status"] in SLEEP_ELIGIBLE_STATUSES
            for item in required_store_states
        )
    )


def _wrapper_store_result_is_nonfatal(item: dict) -> bool:
    if item.get("status") in WRAPPER_SUCCESS_STATUSES:
        return True

    return (
        item.get("status") == "NEEDS_MANUAL_REVIEW"
        and item.get("current_stage") == "INVENTORY_GUARD"
        and item.get("error_category") == "INVENTORY_GUARD_BLOCKED"
    )


def wrapper_returncode_for(state: dict) -> int:
    try:
        required_store_states = [state["stores"][store] for store in STORE_ORDER]
    except (KeyError, TypeError):
        return 1

    return 0 if all(
        _wrapper_store_result_is_nonfatal(item)
        for item in required_store_states
    ) else 1


def launch_sleep_helper(
    parent_automation_run_id: str,
    delay_sec: int = SLEEP_HELPER_DELAY_SEC,
    popen=None,
    *,
    dry_run: bool = False,
) -> int:
    if os.name != "nt":
        raise OSError("Detached sleep helper is only available on Windows.")
    if popen is None:
        popen = subprocess.Popen
    command = [
        sys.executable,
        str(SLEEP_HELPER_PATH.resolve()),
        "--parent-automation-run-id",
        parent_automation_run_id,
        "--delay-sec",
        str(delay_sec),
    ]
    if dry_run:
        command.append("--dry-run")
    process = popen(
        command,
        cwd=str(PROJECT_ROOT),
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=(
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        ),
    )
    return process.pid


def maybe_sleep_on_success(
    enabled: bool,
    state: dict,
    wrapper_returncode: int,
    helper_launcher=None,
) -> bool:
    if not should_sleep_on_success(enabled, state, wrapper_returncode):
        return False

    print("All stores completed successfully. Launching detached sleep helper...")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        if helper_launcher is None:
            helper_launcher = launch_sleep_helper
        helper_pid = helper_launcher(state["automation_run_id"])
    except Exception as exc:
        print(
            f"WARNING: Windows sleep helper launch failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False
    print(
        f"Detached sleep helper launched: pid={helper_pid} "
        f"delay={SLEEP_HELPER_DELAY_SEC}s"
    )
    sys.stdout.flush()
    return True


def flush_output_streams() -> None:
    sys.stdout.flush()
    sys.stderr.flush()



def run_maruhan_natural_e2e_check_best_effort(
    operation_date: str,
    *,
    runner=subprocess.run,
) -> bool:
    """Run read-only Maruhan Natural E2E check and persist stdout/stderr."""
    try:
        target_date = date.fromisoformat(operation_date)
    except ValueError as exc:
        print(
            f"WARNING: Maruhan Natural E2E skipped: invalid operation_date: {exc}",
            file=sys.stderr,
        )
        return False

    if not MARUHAN_E2E_CHECKER_PATH.is_file():
        print(
            f"WARNING: Maruhan Natural E2E checker missing: "
            f"{MARUHAN_E2E_CHECKER_PATH}",
            file=sys.stderr,
        )
        return False

    MARUHAN_E2E_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MARUHAN_E2E_LOG_DIR / f"maruhan_e2e_{target_date:%Y%m%d}.log"
    command = [
        sys.executable,
        str(MARUHAN_E2E_CHECKER_PATH),
        "--operation-date",
        target_date.isoformat(),
        "--write-report",
    ]

    try:
        completed = runner(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=MARUHAN_E2E_TIMEOUT_SEC,
            check=False,
        )
    except Exception as exc:
        log_path.write_text(
            f"E2E_RUNNER_ERROR={type(exc).__name__}: {exc}\\n",
            encoding="utf-8",
        )
        print(
            f"WARNING: Maruhan Natural E2E runner failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False

    payload = (
        f"returncode={completed.returncode}\\n"
        "--- stdout ---\\n"
        f"{completed.stdout or ''}"
        "\\n--- stderr ---\\n"
        f"{completed.stderr or ''}"
    )
    log_path.write_text(payload, encoding="utf-8")

    if completed.returncode != 0:
        print(
            f"WARNING: Maruhan Natural E2E check returned "
            f"{completed.returncode}; see {log_path}",
            file=sys.stderr,
        )
        return False

    print(f"Maruhan Natural E2E check PASS; log={log_path}")
    return True
def finalize_automation_run(
    state_path: Path,
    state: dict,
    sleep_enabled: bool,
    *,
    save_function=None,
    summary_function=None,
    flush_function=None,
    helper_launcher=None,
    notification_function=None,
    clock=None,
) -> int:
    if save_function is None:
        save_function = save_state
    if summary_function is None:
        summary_function = print_summary
    if flush_function is None:
        flush_function = flush_output_streams
    if clock is None:
        clock = now_jst
    if notification_function is None:
        notification_function = notify_terminal_stores_best_effort

    save_function(state_path, state, clock())
    summary_function(state)
    wrapper_returncode = wrapper_returncode_for(state)
    flush_function()
    try:
        notification_function(state, PROJECT_ROOT)
    except Exception as exc:
        print(
            f"WARNING: Gmail notification failed unexpectedly: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    flush_function()
    try:
        run_maruhan_natural_e2e_check_best_effort(
            str(state.get("operation_date", ""))
        )
    except Exception as exc:
        print(
            f"WARNING: Maruhan Natural E2E check failed unexpectedly: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    flush_function()
    maybe_sleep_on_success(
        sleep_enabled,
        state,
        wrapper_returncode,
        helper_launcher,
    )
    return wrapper_returncode







def _freeze_maruhan_isolated_reference_best_effort(operation_date: date) -> None:
    """Independent reference-only freeze; no state mutation and no formal Guard approval."""
    try:
        now = now_jst().astimezone(JST)
        if now.date() != operation_date or now.time().replace(tzinfo=None) >= datetime_time(9, 0):
            return
        # Independent reference may run while Formal Inventory Guard remains BLOCKED.
        # The isolated freezer itself strictly verifies the exact previous-day CSV,
        # 514 machine identities, recorded incident, and freeze time before writing.
        # Do not gate this reference on formal completion artifacts.
        daily = (PROJECT_ROOT / 'data/maruhan_maebashi/machine_number'
                 / f'ana_slo_{operation_date - timedelta(days=1):%Y%m%d}.csv')
        if not daily.is_file():
            print('MARUHAN_ISOLATED: SKIP expected daily source missing')
            return
        script = PROJECT_ROOT / 'slotanalyzer_maruhan_isolated_prospective_freeze_v1.py'
        if not script.is_file():
            print('MARUHAN_ISOLATED: SKIP freezer missing', file=sys.stderr)
            return
        destination = (PROJECT_ROOT / 'data/maruhan_maebashi/machine_number'
                       / 'analysis_isolated_prospective_reference_v1' / f'{operation_date:%Y%m%d}')
        if destination.exists():
            return
        command = [sys.executable, str(script), '--root', str(PROJECT_ROOT),
                   '--target-date', operation_date.isoformat(), '--freeze']
        done = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True,
                              text=True, encoding='utf-8', errors='replace', timeout=150, check=False)
        if done.stdout:
            print('MARUHAN_ISOLATED_OUTPUT: ' + done.stdout.strip().replace('\n', ' | '))
        if done.returncode:
            print(f'MARUHAN_ISOLATED: FAILED code={done.returncode} '
                  + done.stderr[-1200:], file=sys.stderr)
        else:
            print('MARUHAN_ISOLATED: FROZEN_REFERENCE_NOT_FORMAL')
    except Exception as exc:
        print(f'MARUHAN_ISOLATED: ERROR {type(exc).__name__}: {exc}', file=sys.stderr)


def _evaluate_maruhan_isolated_reference_best_effort(operation_date: date) -> None:
    """Evaluate yesterday's pre-09:00 frozen isolated reference only; never Formal Forward."""
    try:
        previous = operation_date - timedelta(days=1)
        directory = (PROJECT_ROOT / 'data/maruhan_maebashi/machine_number'
                     / 'analysis_isolated_prospective_reference_v1' / f'{previous:%Y%m%d}')
        if not (directory / 'freeze_manifest.json').is_file():
            return
        if (directory / 'evaluation_receipt_v1.json').exists():
            return
        script = PROJECT_ROOT / 'slotanalyzer_maruhan_isolated_auto_evaluate_v1.py'
        if not script.is_file():
            print('MARUHAN_ISOLATED_EVAL: SKIP evaluator missing', file=sys.stderr)
            return
        command = [sys.executable, str(script), '--root', str(PROJECT_ROOT),
                   '--target-date', previous.isoformat(), '--save']
        done = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True,
                              text=True, encoding='utf-8', errors='replace', timeout=120, check=False)
        if done.stdout:
            print('MARUHAN_ISOLATED_EVAL: ' + done.stdout.strip().replace('\n', ' | '))
        if done.returncode:
            print(f'MARUHAN_ISOLATED_EVAL: FAILED code={done.returncode} '
                  + done.stderr[-1200:], file=sys.stderr)
    except Exception as exc:
        print(f'MARUHAN_ISOLATED_EVAL: ERROR {type(exc).__name__}: {exc}', file=sys.stderr)

def _evaluate_maruhan_pair_reference_best_effort(operation_date: date) -> None:
    """Evaluate yesterday's frozen reference after today's acquisition, never formal Forward."""
    try:
        previous = operation_date - timedelta(days=1)
        directory = (PROJECT_ROOT / 'data/maruhan_maebashi/machine_number'
                     / 'analysis_paired_prospective_reference_v1' / f'{previous:%Y%m%d}')
        if not (directory / 'freeze_manifest.json').is_file():
            return
        if (directory / 'evaluation_receipt_v1.json').exists():
            return
        script = PROJECT_ROOT / 'slotanalyzer_maruhan_pair_auto_evaluate_v1.py'
        if not script.is_file():
            print('MARUHAN_PAIR_EVAL: SKIP evaluator missing', file=sys.stderr)
            return
        cmd = [sys.executable, str(script), '--root', str(PROJECT_ROOT),
               '--target-date', previous.isoformat(), '--save']
        done = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                              text=True, encoding='utf-8', errors='replace', timeout=150)
        if done.stdout:
            print('MARUHAN_PAIR_EVAL: ' + done.stdout.strip().replace('\n',' | '))
        if done.returncode:
            print('MARUHAN_PAIR_EVAL: FAILED ' + done.stderr[-1000:], file=sys.stderr)
    except Exception as exc:
        print(f'MARUHAN_PAIR_EVAL: ERROR {type(exc).__name__}: {exc}', file=sys.stderr)

def _freeze_maruhan_pair_best_effort(state: dict, operation_date: date) -> None:
    """Independent reference-only frozen predictions; fail closed, never approve Guard."""
    try:
        item = state["stores"][STORE_MARUHAN]
        if item.get("status") not in {"SUCCESS", "ALREADY_COMPLETE"}:
            print("MARUHAN_PAIR: SKIP store not SUCCESS")
            return
        verified = verify_store_completion(STORE_MARUHAN, PROJECT_ROOT, operation_date)
        if not verified.ok:
            print("MARUHAN_PAIR: SKIP data verification failed")
            return
        guard = assess_inventory_guard(
            PROJECT_ROOT / "data/maruhan_maebashi/machine_number",
            operation_date, operation_date - timedelta(days=1),
        )
        if guard.blocked or guard.status != "PASS":
            print("MARUHAN_PAIR: SKIP inventory guard blocked")
            return
        now = now_jst().astimezone(JST)
        if now.date() != operation_date or now.time().replace(tzinfo=None) >= datetime_time(9, 0):
            print("MARUHAN_PAIR: SKIP outside 09:00 freeze window")
            return
        script = PROJECT_ROOT / "slotanalyzer_maruhan_prospective_pair_v1.py"
        if not script.is_file():
            print("MARUHAN_PAIR: SKIP script missing", file=sys.stderr)
            return
        directory = (PROJECT_ROOT / "data/maruhan_maebashi/machine_number"
                     / "analysis_paired_prospective_reference_v1" / f"{operation_date:%Y%m%d}")
        if directory.exists():
            print("MARUHAN_PAIR: SKIP prediction directory already exists")
            return
        command = [sys.executable, str(script), "--root", str(PROJECT_ROOT),
                   "--target-date", operation_date.isoformat(), "--freeze"]
        finished = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=90, check=False)
        if finished.stdout:
            print("MARUHAN_PAIR_OUTPUT: " + finished.stdout.strip().replace("\n", " | "))
        if finished.returncode:
            print(f"MARUHAN_PAIR: FAILED code={finished.returncode} "
                  + finished.stderr[-1200:], file=sys.stderr)
        else:
            print("MARUHAN_PAIR: FROZEN_PROSPECTIVE_REFERENCE_NOT_FORMAL")
    except Exception as exc:
        print(f"MARUHAN_PAIR: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

def _freeze_yasuda_v3_best_effort(state: dict, operation_date: date) -> None:
    """Nonfatal prospective-reference freeze, only after verified Yasuda SUCCESS.

    Does not modify store state, email status, or other stores. It cannot bypass
    Inventory Guard or freeze after 09:00 JST. V3 script is the final authority.
    """
    try:
        item = state["stores"][STORE_YASUDA]
        if item.get("status") not in {"SUCCESS", "ALREADY_COMPLETE"}:
            return
        v = verify_store_completion(STORE_YASUDA, PROJECT_ROOT, operation_date)
        if not v.ok:
            print("YASUDA_V3: SKIP quality verification failed")
            return
        guard = assess_yasuda_inventory_guard(
            PROJECT_ROOT / "data/yasuda_maebashi/machine_number",
            operation_date, operation_date - timedelta(days=1),
        )
        if guard.blocked:
            print("YASUDA_V3: SKIP inventory guard blocked")
            return
        current = now_jst().astimezone(JST)
        if current.date() != operation_date or current.time().replace(tzinfo=None) >= datetime_time(9, 0):
            print("YASUDA_V3: SKIP freeze window closed")
            return
        script = MACHINE_DIR / "slotanalyzer_yasuda_v3_forward_v1.py"
        if not script.is_file():
            print("YASUDA_V3: SKIP missing script", file=sys.stderr)
            return
        folder = PROJECT_ROOT / "data/yasuda_maebashi/machine_number/analysis_prospective_forward_v1" / f"{operation_date:%Y%m%d}"
        if folder.exists():
            print("YASUDA_V3: SKIP preexisting prediction; never overwrite")
            return
        command = [sys.executable, str(script), "--root", str(PROJECT_ROOT),
                   "--target-date", operation_date.isoformat(), "--freeze"]
        done = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=60,
                              check=False)
        if done.stdout:
            print("YASUDA_V3_OUTPUT: " + done.stdout.strip().replace("\n", " | "))
        if done.returncode != 0:
            print(f"YASUDA_V3: FAILED code={done.returncode} " + done.stderr[-1200:], file=sys.stderr)
        else:
            print("YASUDA_V3: FROZEN_PROSPECTIVE_REFERENCE")
    except Exception as exc:
        print(f"YASUDA_V3: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

def _refresh_bic_reference_ledger(operation_date: date) -> None:
    """Non-fatal, idempotent reference result snapshot, independent of email."""
    import subprocess as _bic_subprocess
    source = PROJECT_ROOT / "machine_number" / "slotanalyzer_bic_tsubame_reference_ledger_v1.py"
    if not source.is_file():
        print("BIC_LEDGER: MISSING_SCRIPT", file=sys.stderr)
        return
    command = [sys.executable, str(source), "--root", str(PROJECT_ROOT),
               "--operation-date", operation_date.isoformat(), "--write"]
    try:
        completed = _bic_subprocess.run(command, cwd=str(PROJECT_ROOT),
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        if completed.stdout:
            print("BIC_LEDGER_OUTPUT: " + completed.stdout.strip().replace("\n", " | "))
        if completed.returncode:
            print(f"BIC_LEDGER: FAILED returncode={completed.returncode} "
                  + completed.stderr[-1200:], file=sys.stderr)
        else:
            print("BIC_LEDGER: SUCCESS")
    except Exception as exc:
        print(f"BIC_LEDGER: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)

def main() -> int:
    args = parse_args()
    started = now_jst()
    operation_date, expected_data_date = determine_operation_dates(started)
    state_path = STATE_DIR / f"morning_automation_state_{operation_date:%Y%m%d}.json"
    print("SlotAnalyzer Phase 2 morning automation")
    print(f"operation date        : {operation_date}")
    print(f"expected data date    : {expected_data_date}")
    print("Morning finalize      : 08:30 JST")
    print("Maruhan last start    : 08:30 JST (ranking/pipeline)")
    print("Late Collection       : separate process; never run inside Morning wrapper")
    print("Maruhan formal cutoff : 09:00 JST (existing Forward Guard; unchanged)")
    print("Big March snapshot    : first morning pass (no post-snapshot fetch/retry)")
    print("other store deadline  : 08:30 JST")
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
            _refresh_bic_reference_ledger(operation_date)
            _freeze_yasuda_v3_best_effort(state, operation_date)
            _freeze_maruhan_pair_best_effort(state, operation_date)
            _evaluate_maruhan_pair_reference_best_effort(operation_date)
            _freeze_maruhan_isolated_reference_best_effort(operation_date)
            _evaluate_maruhan_isolated_reference_best_effort(operation_date)
            # Notification attempts are process-local, never part of store/run state.
            notification_attempts = set()
            notify_terminal_stores_best_effort(state, PROJECT_ROOT, notification_attempts)
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
                        if store == STORE_BIGMARCH:
                            item.update(
                                status="NEEDS_MANUAL_REVIEW",
                                error_category="BIGMARCH_SNAPSHOT_EXCEPTION",
                                error=f"{type(exc).__name__}: {exc}",
                                next_retry_at_jst="",
                            )
                        elif item.get("current_stage") in {"READINESS", "CDP_PREFLIGHT", "FETCH"}:
                            item.update(
                                status="FAILED_RETRYABLE",
                                error_category="PRE_PIPELINE_EXCEPTION",
                                error=f"{type(exc).__name__}: {exc}",
                                next_retry_at_jst=(current + timedelta(seconds=(MARUHAN_RETRY_INTERVAL_SEC if store == STORE_MARUHAN else args.retry_interval_sec))).isoformat(),
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
                    if store == STORE_BICTSUBAME:
                        _refresh_bic_reference_ledger(operation_date)
                    if store == STORE_YASUDA:
                        _freeze_yasuda_v3_best_effort(state, operation_date)
                    if store == STORE_MARUHAN:
                        _freeze_maruhan_pair_best_effort(state, operation_date)
                        _evaluate_maruhan_pair_reference_best_effort(operation_date)
                        _freeze_maruhan_isolated_reference_best_effort(operation_date)
                        _evaluate_maruhan_isolated_reference_best_effort(operation_date)
                    # Dispatch before processing the next store or waiting for data.
                    notify_terminal_stores_best_effort(state, PROJECT_ROOT, notification_attempts)
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

                # Never sleep through the Morning finalize boundary.
                finalize_at = datetime.combine(
                    operation_date, MORNING_FINALIZE_DEADLINE, tzinfo=JST
                )
                remaining_to_finalize = int((finalize_at - current).total_seconds())
                if remaining_to_finalize > 0:
                    sleep_sec = max(1, min(sleep_sec, remaining_to_finalize))
                else:
                    sleep_sec = 1

                print(f"Waiting {sleep_sec} seconds before the next readiness pass...")
                time.sleep(sleep_sec)
            return finalize_automation_run(
                state_path,
                state,
                args.sleep_on_success,
                notification_function=lambda value, root: notify_terminal_stores_best_effort(
                    value, root, notification_attempts
                ),
            )
    except LockUnavailableError as exc:
        print(f"ALREADY RUNNING: {exc}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

