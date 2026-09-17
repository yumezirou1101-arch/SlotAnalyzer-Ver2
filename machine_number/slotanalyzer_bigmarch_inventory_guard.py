from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from slotanalyzer_inventory_guard import (
    InventoryDiff,
    InventoryGuardPolicy,
    _atomic_write_state,
    _is_explicitly_approved,
    _read_persistent_state,
    _same_change,
    _state_for_change,
    _state_for_unconfirmed_known_change,
    inspect_inventory_transition,
    inventory_guard_state_path,
)
from slotanalyzer_inventory_monitor import observe_big_march_inventory


POLICY_VERSION = "BIGMARCH_GUARD_V1"

BIGMARCH_TAKASAKI_OYAGI_POLICY = InventoryGuardPolicy(
    store="BIGMARCH_TAKASAKI_OYAGI",
    known_change_dates=frozenset(),
    confirmed_inventory_prefix="bigmarch_oyagi_inventory",
    daily_inventory_pattern="ana_slo_bigmarch_oyagi_{ymd}.csv",
)


@dataclass(frozen=True)
class BigMarchInventoryGuardDecision:
    status: str
    blocked: bool
    formal_allowed: bool
    provisional_allowed: bool
    reason: str
    operation_date: str
    expected_data_date: str
    latest_data_date: str
    source_delay_days: int | None
    known_change_date: bool
    comparison_status: str
    monitor_status: str
    comparison: InventoryDiff | None = None
    persistent_state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["comparison"] = (
            self.comparison.to_dict()
            if self.comparison is not None
            else None
        )
        return value

    def summary(self) -> str:
        parts = [
            f"BIGMARCH_INVENTORY_GUARD: {self.status}",
            self.reason,
            f"formal_allowed={self.formal_allowed}",
            f"provisional_allowed={self.provisional_allowed}",
        ]

        if self.latest_data_date:
            parts.append(
                f"latest_data_date={self.latest_data_date}"
            )

        if self.source_delay_days is not None:
            parts.append(
                f"source_delay_days={self.source_delay_days}"
            )

        if self.comparison_status:
            parts.append(
                f"comparison_status={self.comparison_status}"
            )

        if self.monitor_status:
            parts.append(
                f"monitor_status={self.monitor_status}"
            )

        return "; ".join(parts)


class BigMarchInventoryGuardBlockedError(RuntimeError):
    def __init__(self, decision: BigMarchInventoryGuardDecision):
        self.decision = decision
        super().__init__(decision.summary())


def discover_big_march_daily_files(
    data_dir: Path,
    *,
    not_after: date | None = None,
) -> list[tuple[date, Path]]:
    data_dir = Path(data_dir)
    found: list[tuple[date, Path]] = []

    for path in data_dir.glob(
        "ana_slo_bigmarch_oyagi_????????.csv"
    ):
        stem = path.stem
        compact = stem.rsplit("_", 1)[-1]

        try:
            file_date = datetime.strptime(
                compact,
                "%Y%m%d",
            ).date()
        except ValueError:
            continue

        if not_after is not None and file_date > not_after:
            continue

        found.append((file_date, path))

    return sorted(found, key=lambda item: item[0])


def _evidence_to_diff(evidence) -> InventoryDiff | None:
    if not evidence.comparison_performed:
        return None

    return InventoryDiff(
        previous_date=evidence.previous_date,
        current_date=evidence.current_date,
        previous_machine_count=int(
            evidence.previous_machine_count
        ),
        current_machine_count=int(
            evidence.current_machine_count
        ),
        added_machine_numbers=list(
            evidence.added_machine_numbers or []
        ),
        removed_machine_numbers=list(
            evidence.removed_machine_numbers or []
        ),
        renamed_machine_numbers=list(
            evidence.renamed_machine_numbers or []
        ),
    )


def _blocked_decision(
    *,
    status: str,
    reason: str,
    operation_date: date,
    expected_data_date: date,
    latest_data_date: date | None,
    source_delay_days: int | None,
    known_change_date: bool,
    comparison_status: str,
    monitor_status: str,
    comparison: InventoryDiff | None,
    persistent_state: dict,
) -> BigMarchInventoryGuardDecision:
    return BigMarchInventoryGuardDecision(
        status=status,
        blocked=True,
        formal_allowed=False,
        provisional_allowed=False,
        reason=reason,
        operation_date=operation_date.isoformat(),
        expected_data_date=expected_data_date.isoformat(),
        latest_data_date=(
            latest_data_date.isoformat()
            if latest_data_date is not None
            else ""
        ),
        source_delay_days=source_delay_days,
        known_change_date=known_change_date,
        comparison_status=comparison_status,
        monitor_status=monitor_status,
        comparison=comparison,
        persistent_state=persistent_state,
    )


def _pass_decision(
    *,
    status: str,
    reason: str,
    operation_date: date,
    expected_data_date: date,
    latest_data_date: date,
    source_delay_days: int,
    known_change_date: bool,
    comparison_status: str,
    monitor_status: str,
    comparison: InventoryDiff | None,
    persistent_state: dict,
    formal_allowed: bool,
    provisional_allowed: bool,
) -> BigMarchInventoryGuardDecision:
    return BigMarchInventoryGuardDecision(
        status=status,
        blocked=False,
        formal_allowed=formal_allowed,
        provisional_allowed=provisional_allowed,
        reason=reason,
        operation_date=operation_date.isoformat(),
        expected_data_date=expected_data_date.isoformat(),
        latest_data_date=latest_data_date.isoformat(),
        source_delay_days=source_delay_days,
        known_change_date=known_change_date,
        comparison_status=comparison_status,
        monitor_status=monitor_status,
        comparison=comparison,
        persistent_state=persistent_state,
    )


def _persist_actual_change(
    data_dir: Path,
    comparison: InventoryDiff,
    evidence,
    persistent_state: dict,
) -> dict:
    state_path = inventory_guard_state_path(data_dir)

    if (
        persistent_state.get("status") == "BLOCKED"
        and _same_change(persistent_state, comparison)
    ):
        return persistent_state

    planned_now_confirmed = (
        persistent_state.get("status") == "BLOCKED"
        and persistent_state.get("incident_kind")
        == "KNOWN_CHANGE_UNCONFIRMED"
        and persistent_state.get("change_date")
        == comparison.current_date
    )

    if (
        persistent_state
        and not _is_explicitly_approved(persistent_state)
        and not planned_now_confirmed
    ):
        return persistent_state

    new_state = _state_for_change(
        BIGMARCH_TAKASAKI_OYAGI_POLICY,
        comparison,
    )

    new_state["policy_version"] = POLICY_VERSION
    new_state["comparison_status"] = (
        evidence.comparison_status
    )
    new_state["previous_daily_sha256"] = (
        evidence.previous_daily_sha256
    )
    new_state["current_daily_sha256"] = (
        evidence.current_daily_sha256
    )
    new_state["incident_reason"] = (
        "Actual Big March inventory change detected."
    )

    _atomic_write_state(state_path, new_state)
    return new_state


def _persist_known_change_unconfirmed(
    data_dir: Path,
    expected_data_date: date,
    latest_data_date: date,
    persistent_state: dict,
) -> dict:
    state_path = inventory_guard_state_path(data_dir)

    same_planned = (
        persistent_state.get("status") == "BLOCKED"
        and persistent_state.get("incident_kind")
        == "KNOWN_CHANGE_UNCONFIRMED"
        and persistent_state.get("change_date")
        == expected_data_date.isoformat()
    )

    if same_planned:
        return persistent_state

    if (
        persistent_state
        and not _is_explicitly_approved(persistent_state)
    ):
        return persistent_state

    new_state = _state_for_unconfirmed_known_change(
        BIGMARCH_TAKASAKI_OYAGI_POLICY,
        expected_data_date,
        latest_data_date,
    )

    new_state["policy_version"] = POLICY_VERSION
    new_state["incident_reason"] = (
        "Known Big March inventory change date is "
        "not yet confirmed by the expected daily inventory."
    )

    _atomic_write_state(state_path, new_state)
    return new_state


def assess_big_march_inventory_guard(
    data_dir: Path,
    operation_date: date,
    *,
    generated_at_jst=None,
) -> BigMarchInventoryGuardDecision:
    """
    Big March Inventory Guard V1.

    Rules:
    - Fresh expected daily:
        Formal may run only if the latest consecutive
        inventory transition is confirmed safe.
    - One-day source delay:
        Formal is forbidden.
        PROVISIONAL may run only if the latest available
        consecutive inventory transition is confirmed safe.
    - Delay >= 2 days:
        Formal and PROVISIONAL are blocked.
    - NON_CONSECUTIVE / invalid / missing comparison:
        Formal and PROVISIONAL are blocked.
    - ACTUAL_CHANGE:
        Persisted BLOCK until explicit approval.
    - Known change date not yet confirmed:
        Formal and PROVISIONAL are blocked.
    - SOURCE_CHANGED_AFTER_OBSERVATION / monitor ERROR:
        Fail closed.
    - Monitor Evidence:
        Preserve observational evidence whenever a daily
        baseline exists and the monitor can evaluate it.
        Guard remains authoritative for allow/block control.
    """
    data_dir = Path(data_dir)
    operation_date = date.fromisoformat(
        str(operation_date)
    )

    if generated_at_jst is None:
        generated_at_jst = datetime.now().astimezone()

    expected_data_date = (
        operation_date - timedelta(days=1)
    )

    state_path = inventory_guard_state_path(data_dir)
    persistent_state = _read_persistent_state(
        state_path,
        BIGMARCH_TAKASAKI_OYAGI_POLICY,
    )

    daily_files = discover_big_march_daily_files(
        data_dir,
        not_after=expected_data_date,
    )

    if not daily_files:
        return _blocked_decision(
            status="BLOCKED_NO_DAILY_BASELINE",
            reason=(
                "No Big March daily inventory is available "
                "on or before the expected data date."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=None,
            source_delay_days=None,
            known_change_date=(
                expected_data_date
                in BIGMARCH_TAKASAKI_OYAGI_POLICY.known_change_dates
            ),
            comparison_status="NO_DAILY_BASELINE",
            monitor_status="",
            comparison=None,
            persistent_state=persistent_state,
        )

    latest_data_date, latest_path = daily_files[-1]
    source_delay_days = (
        expected_data_date - latest_data_date
    ).days

    if source_delay_days < 0:
        return _blocked_decision(
            status="BLOCKED_FUTURE_DAILY",
            reason=(
                "Latest Big March daily inventory is later "
                "than the expected data date."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=False,
            comparison_status="FUTURE_DAILY",
            monitor_status="",
            comparison=None,
            persistent_state=persistent_state,
        )

    known_change_date = (
        expected_data_date
        in BIGMARCH_TAKASAKI_OYAGI_POLICY.known_change_dates
    )

    try:
        monitor_result = observe_big_march_inventory(
            data_dir,
            operation_date,
            generated_at_jst=generated_at_jst,
        )
    except Exception as exc:
        return _blocked_decision(
            status="BLOCKED_GUARD_ERROR",
            reason=(
                "Big March inventory monitor/evidence "
                "verification failed during authoritative "
                f"Guard evaluation: {type(exc).__name__}: "
                f"{exc}"
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=known_change_date,
            comparison_status="MONITOR_EVIDENCE_ERROR",
            monitor_status="ERROR",
            comparison=None,
            persistent_state=persistent_state,
        )

    monitor_status = str(
        monitor_result.get("status", "")
    )

    if monitor_status in {
        "SOURCE_CHANGED_AFTER_OBSERVATION",
        "ERROR",
    }:
        return _blocked_decision(
            status=(
                "BLOCKED_SOURCE_CHANGED"
                if monitor_status
                == "SOURCE_CHANGED_AFTER_OBSERVATION"
                else "BLOCKED_GUARD_ERROR"
            ),
            reason=(
                "Previously observed Big March source "
                "identity changed after observation."
                if monitor_status
                == "SOURCE_CHANGED_AFTER_OBSERVATION"
                else (
                    "Big March inventory monitor returned "
                    "an error during authoritative Guard "
                    "evaluation."
                )
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=known_change_date,
            comparison_status="MONITOR_EVIDENCE_BLOCK",
            monitor_status=monitor_status,
            comparison=None,
            persistent_state=persistent_state,
        )

    if source_delay_days >= 2:
        return _blocked_decision(
            status="BLOCKED_SOURCE_DELAY",
            reason=(
                "Big March source delay is two days or more; "
                "Formal and PROVISIONAL are both forbidden."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=known_change_date,
            comparison_status="SOURCE_DELAY_GE_2",
            monitor_status=monitor_status,
            comparison=None,
            persistent_state=persistent_state,
        )

    if source_delay_days == 1 and known_change_date:
        persistent_state = (
            _persist_known_change_unconfirmed(
                data_dir,
                expected_data_date,
                latest_data_date,
                persistent_state,
            )
        )

        return _blocked_decision(
            status="BLOCKED_KNOWN_CHANGE_UNCONFIRMED",
            reason=(
                "Expected data date is a known inventory "
                "change date, but its inventory is not yet "
                "confirmed; PROVISIONAL is forbidden."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=True,
            comparison_status="KNOWN_CHANGE_UNCONFIRMED",
            monitor_status=monitor_status,
            comparison=None,
            persistent_state=persistent_state,
        )

    previous_candidates = [
        item
        for item in daily_files
        if item[0] < latest_data_date
    ]

    if not previous_candidates:
        return _blocked_decision(
            status="BLOCKED_COMPARISON_UNAVAILABLE",
            reason=(
                "No previous Big March daily inventory is "
                "available for continuity verification."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=known_change_date,
            comparison_status="PREVIOUS_MISSING",
            monitor_status=monitor_status,
            comparison=None,
            persistent_state=persistent_state,
        )

    previous_date, previous_path = (
        previous_candidates[-1]
    )

    evidence = inspect_inventory_transition(
        BIGMARCH_TAKASAKI_OYAGI_POLICY.store,
        previous_path,
        latest_path,
        previous_date,
        latest_data_date,
        policy_version=POLICY_VERSION,
        mode="ENFORCE",
    )

    comparison = _evidence_to_diff(evidence)

    if not evidence.comparison_performed:
        reason = (
            "Big March inventory comparison is "
            "non-consecutive."
            if evidence.comparison_status
            == "NON_CONSECUTIVE"
            else (
                "Big March inventory comparison is "
                "unavailable or invalid."
            )
        )

        return _blocked_decision(
            status="BLOCKED_NON_CONSECUTIVE"
            if evidence.comparison_status
            == "NON_CONSECUTIVE"
            else "BLOCKED_COMPARISON_UNAVAILABLE",
            reason=(
                f"{reason} "
                f"comparison_status="
                f"{evidence.comparison_status}"
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=known_change_date,
            comparison_status=evidence.comparison_status,
            monitor_status=monitor_status,
            comparison=comparison,
            persistent_state=persistent_state,
        )

    if comparison is None:
        return _blocked_decision(
            status="BLOCKED_GUARD_ERROR",
            reason=(
                "Inventory evidence reported a completed "
                "comparison but no diff could be built."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=known_change_date,
            comparison_status=evidence.comparison_status,
            monitor_status=monitor_status,
            comparison=None,
            persistent_state=persistent_state,
        )

    if comparison.has_changes:
        approved_same_change = (
            _is_explicitly_approved(persistent_state)
            and _same_change(
                persistent_state,
                comparison,
            )
        )

        if not approved_same_change:
            already_different_unapproved = (
                persistent_state
                and not _is_explicitly_approved(
                    persistent_state
                )
                and not (
                    persistent_state.get("status")
                    == "BLOCKED"
                    and _same_change(
                        persistent_state,
                        comparison,
                    )
                )
                and not (
                    persistent_state.get(
                        "incident_kind"
                    )
                    == "KNOWN_CHANGE_UNCONFIRMED"
                    and persistent_state.get(
                        "change_date"
                    )
                    == comparison.current_date
                )
            )

            if already_different_unapproved:
                return _blocked_decision(
                    status="BLOCKED_PERSISTENT_INCIDENT",
                    reason=(
                        "A different unapproved Big March "
                        "inventory incident is already "
                        "persisted and was not overwritten."
                    ),
                    operation_date=operation_date,
                    expected_data_date=expected_data_date,
                    latest_data_date=latest_data_date,
                    source_delay_days=source_delay_days,
                    known_change_date=known_change_date,
                    comparison_status=evidence.comparison_status,
                    monitor_status=monitor_status,
                    comparison=comparison,
                    persistent_state=persistent_state,
                )

            persistent_state = _persist_actual_change(
                data_dir,
                comparison,
                evidence,
                persistent_state,
            )

            return _blocked_decision(
                status="BLOCKED_ACTUAL_CHANGE",
                reason=(
                    "Actual Big March inventory change "
                    "detected; explicit safety approval is "
                    "required."
                ),
                operation_date=operation_date,
                expected_data_date=expected_data_date,
                latest_data_date=latest_data_date,
                source_delay_days=source_delay_days,
                known_change_date=known_change_date,
                comparison_status=evidence.comparison_status,
                monitor_status=monitor_status,
                comparison=comparison,
                persistent_state=persistent_state,
            )

    if (
        persistent_state
        and not _is_explicitly_approved(
            persistent_state
        )
    ):
        return _blocked_decision(
            status="BLOCKED_PERSISTENT_INCIDENT",
            reason=(
                "A Big March inventory incident remains "
                "blocked because explicit approval has not "
                "been recorded."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=source_delay_days,
            known_change_date=known_change_date,
            comparison_status=evidence.comparison_status,
            monitor_status=monitor_status,
            comparison=comparison,
            persistent_state=persistent_state,
        )

    if source_delay_days == 0:
        return _pass_decision(
            status="PASS_FORMAL",
            reason=(
                "Expected Big March daily inventory is "
                "present and the latest consecutive "
                "inventory transition has no unapproved "
                "change."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=0,
            known_change_date=known_change_date,
            comparison_status=evidence.comparison_status,
            monitor_status=monitor_status,
            comparison=comparison,
            persistent_state=persistent_state,
            formal_allowed=True,
            provisional_allowed=False,
        )

    if source_delay_days == 1:
        return _pass_decision(
            status="PASS_PROVISIONAL_ONLY",
            reason=(
                "Big March source is exactly one day "
                "behind, but the latest available "
                "consecutive inventory transition is "
                "confirmed safe; Formal is forbidden and "
                "PROVISIONAL is allowed."
            ),
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            source_delay_days=1,
            known_change_date=known_change_date,
            comparison_status=evidence.comparison_status,
            monitor_status=monitor_status,
            comparison=comparison,
            persistent_state=persistent_state,
            formal_allowed=False,
            provisional_allowed=True,
        )

    return _blocked_decision(
        status="BLOCKED_GUARD_ERROR",
        reason=(
            "Unexpected Big March source-delay state."
        ),
        operation_date=operation_date,
        expected_data_date=expected_data_date,
        latest_data_date=latest_data_date,
        source_delay_days=source_delay_days,
        known_change_date=known_change_date,
        comparison_status=evidence.comparison_status,
        monitor_status=monitor_status,
        comparison=comparison,
        persistent_state=persistent_state,
    )

def enforce_big_march_formal_inventory_guard(
    data_dir: Path,
    operation_date: date,
    *,
    generated_at_jst=None,
) -> BigMarchInventoryGuardDecision:
    decision = assess_big_march_inventory_guard(
        data_dir,
        operation_date,
        generated_at_jst=generated_at_jst,
    )

    if decision.blocked or not decision.formal_allowed:
        raise BigMarchInventoryGuardBlockedError(
            decision
        )

    return decision


def enforce_big_march_provisional_inventory_guard(
    data_dir: Path,
    operation_date: date,
    *,
    generated_at_jst=None,
) -> BigMarchInventoryGuardDecision:
    decision = assess_big_march_inventory_guard(
        data_dir,
        operation_date,
        generated_at_jst=generated_at_jst,
    )

    if (
        decision.blocked
        or not decision.provisional_allowed
    ):
        raise BigMarchInventoryGuardBlockedError(
            decision
        )

    return decision
