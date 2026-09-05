from __future__ import annotations

import csv
import ctypes
import hashlib
import html
import os
import smtplib
import ssl
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Callable

from slotanalyzer_morning_automation_support import (
    JST,
    STORE_BIGMARCH,
    STORE_MARUHAN,
    STORE_YASUDA,
    append_history_csv,
    now_jst,
    verify_big_march_completion,
    verify_maruhan_completion,
    verify_yasuda_completion,
)


CREDENTIAL_TARGET = "SlotAnalyzer_Gmail_SMTP"
RECIPIENT_ENV = "SLOTANALYZER_GMAIL_RECIPIENT"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_TIMEOUT_SEC = 20
NOTIFICATION_TYPE = "MORNING_RESULT"
HISTORY_FIELDS = [
    "notification_id",
    "automation_run_id",
    "operation_date",
    "notification_type",
    "attempted_at_jst",
    "completed_at_jst",
    "overall_status",
    "channel",
    "masked_recipient",
    "status",
    "error_category",
    "error",
    "message_sha256",
]


@dataclass(frozen=True)
class Credential:
    username: str
    password: str


@dataclass(frozen=True)
class NotificationMessage:
    subject: str
    plain: str
    html: str
    overall_status: str


@dataclass(frozen=True)
class RankingBlock:
    label: str
    rows: list[dict[str, str]]
    rank_keys: tuple[str, ...]
    unavailable_message: str = "ランキング未生成"


@dataclass(frozen=True)
class StoreSection:
    heading: str
    summary_lines: list[str]
    rankings: list[RankingBlock]
    detail_lines: list[str]


class CredentialReadError(RuntimeError):
    pass


class NotificationConfigError(RuntimeError):
    pass


def read_windows_credential(target: str = CREDENTIAL_TARGET) -> Credential:
    """Read a generic Windows credential without persisting or displaying its secret."""
    if os.name != "nt":
        raise CredentialReadError("Windows Credential Manager is unavailable on this platform.")

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", ctypes.c_uint32),
            ("Type", ctypes.c_uint32),
            ("TargetName", ctypes.c_wchar_p),
            ("Comment", ctypes.c_wchar_p),
            ("LastWritten", FILETIME),
            ("CredentialBlobSize", ctypes.c_uint32),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", ctypes.c_uint32),
            ("AttributeCount", ctypes.c_uint32),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", ctypes.c_wchar_p),
            ("UserName", ctypes.c_wchar_p),
        ]

    pcredential = ctypes.POINTER(CREDENTIALW)()
    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    cred_read = advapi32.CredReadW
    cred_read.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
    cred_read.restype = ctypes.c_bool
    cred_free = advapi32.CredFree
    cred_free.argtypes = [ctypes.c_void_p]
    cred_free.restype = None
    if not cred_read(target, 1, 0, ctypes.byref(pcredential)):  # CRED_TYPE_GENERIC
        code = ctypes.get_last_error()
        raise CredentialReadError(f"Credential read failed for configured target (Windows error {code}).")
    try:
        item = pcredential.contents
        username = item.UserName or ""
        if item.CredentialBlobSize:
            raw = ctypes.string_at(item.CredentialBlob, item.CredentialBlobSize)
            password = raw.decode("utf-16-le")
        else:
            password = ""
        if not username or not password:
            raise CredentialReadError("Credential username or secret is empty.")
        return Credential(username=username, password=password)
    finally:
        cred_free(pcredential)


def mask_recipient(recipient: str) -> str:
    local, separator, domain = recipient.strip().partition("@")
    if not separator:
        return "***"
    visible = local[:2] if len(local) > 2 else local[:1]
    return f"{visible}***@{domain}"


def determine_overall_status(state: dict) -> str:
    statuses = [str(item.get("status", "")) for item in state.get("stores", {}).values()]
    success = {"SUCCESS", "ALREADY_COMPLETE"}
    manual = {"NEEDS_MANUAL_REVIEW", "MANUAL_REVIEW"}
    if statuses and all(value in success for value in statuses):
        return "SUCCESS"
    if any(value in success for value in statuses):
        return "PARTIAL"
    if any(value in manual for value in statuses):
        return "MANUAL_REVIEW"
    return "FAILED"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _same_dates(rows: list[dict[str, str]], operation_date: date, expected: date) -> bool:
    return bool(rows) and all(
        row.get("target_date", "") == operation_date.isoformat()
        and row.get("latest_data_date", "") == expected.isoformat()
        for row in rows
    )


def _ranking_values(row: dict[str, str], rank_keys: tuple[str, ...], index: int) -> tuple[str, str, str, str]:
    rank = next((row.get(key) for key in rank_keys if row.get(key)), str(index))
    machine_no = row.get("machine_no") or row.get("台番号") or "?"
    name = row.get("machine_name") or row.get("機種名") or "?"
    score = row.get("score", "")
    try:
        score_text = f"{float(score):.2f}" if score else ""
    except (TypeError, ValueError):
        score_text = ""
    return str(rank), str(machine_no), str(name), score_text


def _rank_lines(rows: list[dict[str, str]], rank_keys: tuple[str, ...], limit: int = 10) -> list[str]:
    output = []
    for index, row in enumerate(rows[:limit], 1):
        rank, machine_no, name, score_text = _ranking_values(row, rank_keys, index)
        values = [f"{rank}. {machine_no}番台", str(name)]
        if score_text:
            values.append(score_text)
        output.append("　".join(values))
    return output or ["  ランキング未生成"]


def _ranking_table_html(block: RankingBlock, limit: int = 10) -> str:
    if not block.rows:
        return (
            f'<h4 style="margin:14px 0 4px">{html.escape(block.label)}</h4>'
            f'<p style="margin:4px 0 12px;color:#a33">{html.escape(block.unavailable_message)}</p>'
        )
    header_style = "padding:8px 6px;border-bottom:2px solid #777;font-size:12px;color:#444;white-space:nowrap"
    cell_style = "padding:8px 6px;border-bottom:1px solid #ddd;vertical-align:top;font-size:14px"
    machine_boundary = "border-left:1px solid #ddd"
    parts = [
        f'<h4 style="margin:14px 0 4px">{html.escape(block.label)}</h4>',
        '<table width="100%" role="presentation" style="width:100%;table-layout:fixed;border-collapse:collapse">',
        "<thead><tr>",
        f'<th width="10%" style="{header_style};text-align:center">No.</th>',
        f'<th width="15%" style="{header_style};text-align:center">台</th>',
        f'<th width="57%" style="{header_style};{machine_boundary};text-align:left">機種</th>',
        f'<th width="18%" style="{header_style};text-align:right">Score</th>',
        "</tr></thead><tbody>",
    ]
    for index, row in enumerate(block.rows[:limit], 1):
        rank, machine_no, name, score_text = _ranking_values(row, block.rank_keys, index)
        parts.extend([
            "<tr>",
            f'<td style="{cell_style};text-align:center;white-space:nowrap">{html.escape(str(rank))}</td>',
            f'<td style="{cell_style};text-align:center;white-space:nowrap">{html.escape(str(machine_no))}</td>',
            f'<td style="{cell_style};{machine_boundary};text-align:left;white-space:normal;overflow-wrap:anywhere;word-break:break-word">{html.escape(str(name))}</td>',
            f'<td style="{cell_style};text-align:right;white-space:nowrap">{html.escape(score_text)}</td>',
            "</tr>",
        ])
    parts.append("</tbody></table>")
    return "".join(parts)


def _render_section_plain(section: StoreSection) -> str:
    lines = [section.heading, *section.summary_lines]
    for block in section.rankings:
        lines.append(f"{block.label}:")
        lines.extend(
            _rank_lines(block.rows, block.rank_keys)
            if block.rows
            else [f"  {block.unavailable_message}"]
        )
    if section.detail_lines:
        lines.extend(["詳細情報:", *section.detail_lines])
    return "\n".join(lines)


def _render_section_html(section: StoreSection) -> str:
    summary = "".join(f"<div>{html.escape(line)}</div>" for line in section.summary_lines)
    rankings = "".join(_ranking_table_html(block) for block in section.rankings)
    details = ""
    if section.detail_lines:
        detail_rows = "".join(f"<div>{html.escape(line)}</div>" for line in section.detail_lines)
        details = f'<h4 style="margin:18px 0 4px">詳細情報</h4>{detail_rows}'
    return (
        '<section style="margin:18px 0">'
        f'<h3 style="margin:0 0 6px">{html.escape(section.heading)}</h3>'
        f'{summary}{rankings}{details}</section>'
    )


def _store_state(state: dict, store: str) -> dict:
    return state.get("stores", {}).get(store, {})


def _maruhan_content(state: dict, root: Path, operation_date: date) -> tuple[StoreSection, list[str]]:
    expected = operation_date - timedelta(days=1)
    item = _store_state(state, STORE_MARUHAN)
    verification = verify_maruhan_completion(root, operation_date)
    base = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep"
    ymd = operation_date.strftime("%Y%m%d")
    metadata_rows = _read_rows(base / "64_Ver4_2_future_top10" / f"64_prediction_{ymd}_metadata.csv")
    metadata = metadata_rows[0] if len(metadata_rows) == 1 else {}
    inventory_guard = item.get("inventory_guard") or {}
    inventory_blocked = bool(inventory_guard.get("blocked"))
    formal = (
        not inventory_blocked
        and verification.ok
        and metadata.get("forward_valid", "").lower() in {"true", "1", "yes"}
    )
    warnings = [] if formal else [f"Maruhan: 非valid ({verification.status})"]
    if inventory_blocked:
        warnings.extend([
            "Maruhan: 台構成変更を検知",
            "Maruhan: 正式Forward停止",
            "Maruhan: MANUAL_REVIEW",
        ])
        persistent_state = inventory_guard.get("persistent_state") or {}
        if persistent_state.get("status") == "BLOCKED":
            warnings.extend([
                "Maruhan: 台構成変更による正式Forward停止中",
                "Maruhan: 未承認のため停止継続",
            ])
    summary_lines = [
        f"status: {item.get('status', 'UNKNOWN')}",
        f"target_date: {metadata.get('target_date', operation_date.isoformat())}",
        f"latest_data_date: {metadata.get('latest_data_date', item.get('latest_data_date', ''))}",
        f"Forward: {'正式Forward停止 (inventory guard)' if inventory_blocked else ('FORWARD_VALID (formal)' if formal else '非valid / formal表示不可')}",
    ]
    specs = [
        ("NORMAL Top10", base / "64_Ver4_2_future_top10" / f"64_prediction_{ymd}_top10.csv", ("prediction_rank",)),
        ("A-TYPE Top10", base / "74_Ver4_2_A_type_prediction" / f"74_A_type_prediction_{ymd}_top10.csv", ("a_type_rank", "prediction_rank")),
        ("JUGGLER Top10", base / "75_Ver4_2_Juggler_prediction" / f"75_Juggler_prediction_{ymd}_top10.csv", ("juggler_rank", "prediction_rank")),
        ("統合ランキング / 代替候補・NEXT", base / "77_live_integrated_prediction_report" / f"77_integrated_prediction_{ymd}.csv", ("report_order",)),
    ]
    rankings = []
    for label, path, keys in specs:
        rows = _read_rows(path)
        valid_rows = rows if formal and _same_dates(rows, operation_date, expected) else []
        rankings.append(RankingBlock(label, valid_rows, keys, "ランキング未生成または日付不一致"))
    details = [
        f"generated_at_jst: {metadata.get('generated_at_jst', '-')}",
        f"model: {metadata.get('model', '-')}",
        f"fingerprint: {metadata.get('weight_fingerprint', '-')}",
    ]
    if inventory_blocked:
        comparison = inventory_guard.get("comparison") or {}
        persistent_state = inventory_guard.get("persistent_state") or {}
        if not comparison.get("has_changes") and persistent_state:
            comparison = persistent_state
        details.append(f"inventory_guard: {inventory_guard.get('reason', 'blocked')}")
        if persistent_state:
            details.extend([
                f"persistent status: {persistent_state.get('status', '-')}",
                f"change date: {persistent_state.get('change_date', '-')}",
                f"detected at: {persistent_state.get('detected_at', '-')}",
            ])
        if comparison:
            details.extend([
                f"inventory previous/current: {comparison.get('previous_machine_count', '-')} / {comparison.get('current_machine_count', '-')}",
                f"added machine numbers: {comparison.get('added_machine_numbers', [])}",
                f"removed machine numbers: {comparison.get('removed_machine_numbers', [])}",
            ])
            for changed in comparison.get("renamed_machine_numbers", [])[:10]:
                details.append(
                    "renamed: "
                    f"{changed.get('machine_no')} "
                    f"{changed.get('previous_machine_name')} -> {changed.get('current_machine_name')}"
                )
    return StoreSection("【Maruhan 前橋インター】", summary_lines, rankings, details), warnings


def _maruhan_section(state: dict, root: Path, operation_date: date) -> tuple[list[str], list[str]]:
    section, warnings = _maruhan_content(state, root, operation_date)
    return _render_section_plain(section).splitlines(), warnings


def _forward_summary(path: Path) -> str:
    rows = _read_rows(path)
    if not rows:
        return "status=不明"
    row = rows[0]
    days = row.get("available_forward_days", row.get("forward_days", "-"))
    return (
        f"status={row.get('status', '-')} available_forward_days={days} "
        f"min_review_days={row.get('min_review_days', '-')} "
        f"automatic_promotion={row.get('automatic_promotion', '-')}"
    )


def _bigmarch_content(state: dict, root: Path, operation_date: date) -> tuple[StoreSection, list[str]]:
    expected = operation_date - timedelta(days=1)
    item = _store_state(state, STORE_BIGMARCH)
    verification = verify_big_march_completion(root, operation_date)
    base = root / "data/bigmarch_takasaki_oyagi/machine_number"
    analysis = base / "analysis_31days_deep"
    daily = base / f"ana_slo_bigmarch_oyagi_{expected:%Y%m%d}.csv"
    daily_rows = _read_rows(daily)
    fresh = bool(daily_rows) and all(row.get("date") == expected.isoformat() for row in daily_rows)
    warnings = []
    if not fresh:
        warnings.append("Big March: 最新1日未反映 / 更新待ち / Freshness失敗 / 本日ランキング未生成")
    summary_lines = [
        f"status: {item.get('status', 'UNKNOWN')}",
        "種別: Big March Forward検証ランキング",
        f"latest_data_date: {expected.isoformat() if fresh else item.get('latest_data_date', '-')}",
        "JUGGLER Forward: " + _forward_summary(analysis / "08_juggler_recent7_top3_forward/08_forward_status.csv"),
        "NON_JUGGLER Forward: " + _forward_summary(analysis / "11_nonjuggler_weekday_top1_forward/11_nonjuggler_weekday_top1_forward_summary.csv"),
    ]
    rankings = []
    for label, rel, keys in [
        ("JUGGLER", "09_juggler_recent7_future_ranking", ("prediction_rank", "rank")),
        ("NON_JUGGLER", "12_nonjuggler_weekday_future_ranking", ("prediction_rank", "rank")),
    ]:
        path = analysis / rel / f"{'09' if label == 'JUGGLER' else '12'}_prediction_{operation_date:%Y%m%d}_top10.csv"
        rows = _read_rows(path)
        if fresh and verification.ok and _same_dates(rows, operation_date, expected):
            rankings.append(RankingBlock(label, rows, keys))
        else:
            rankings.append(RankingBlock(label, [], keys, "本日ランキング未生成（古い生成分は本日分として表示しません）"))
    return StoreSection("【Big March 高崎おおやぎ】", summary_lines, rankings, []), warnings


def _bigmarch_section(state: dict, root: Path, operation_date: date) -> tuple[list[str], list[str]]:
    section, warnings = _bigmarch_content(state, root, operation_date)
    return _render_section_plain(section).splitlines(), warnings


def _yasuda_section(state: dict, root: Path, operation_date: date) -> tuple[list[str], list[str]]:
    expected = operation_date - timedelta(days=1)
    item = _store_state(state, STORE_YASUDA)
    verification = verify_yasuda_completion(root, operation_date)
    daily = root / "data/yasuda_maebashi/machine_number" / f"ana_slo_{expected:%Y%m%d}.csv"
    records = len(_read_rows(daily))
    warnings = [] if verification.ok else [f"Yasuda: Freshness/quality失敗 ({verification.status})"]
    return [
        "【Yasuda 前橋】",
        f"status: {item.get('status', 'UNKNOWN')}",
        f"expected/latest data date: {expected.isoformat()} / {item.get('latest_data_date') or ('確認済み' if verification.ok else '-')}",
        f"records: {records}",
        f"Freshness/quality: {'OK' if verification.ok else 'FAILED: ' + verification.status}",
        "ランキング機能: 未実装",
    ], warnings


def build_notification_message(state: dict, project_root: Path) -> NotificationMessage:
    operation_date = date.fromisoformat(state["operation_date"])
    overall = determine_overall_status(state)
    sections: list[StoreSection] = []
    warnings = []
    summaries = []
    for store, label, builder in [
        (STORE_MARUHAN, "Maruhan", _maruhan_content),
        (STORE_BIGMARCH, "Big March", _bigmarch_content),
        (STORE_YASUDA, "Yasuda", None),
    ]:
        if builder is None:
            lines, store_warnings = _yasuda_section(state, project_root, operation_date)
            section = StoreSection(lines[0], lines[1:], [], [])
        else:
            section, store_warnings = builder(state, project_root, operation_date)
        sections.append(section)
        warnings.extend(store_warnings)
        summaries.append(f"{label}: {_store_state(state, store).get('status', 'UNKNOWN')}")
    for store, item in state.get("stores", {}).items():
        status = str(item.get("status", ""))
        if status not in {"SUCCESS", "ALREADY_COMPLETE"}:
            warnings.append(f"{store}: {status} {item.get('error_category', '')} {item.get('error', '')}".strip())
    warning_text = "\n".join(f"⚠ {value}" for value in dict.fromkeys(warnings)) or "異常警告: なし"
    plain = (
        f"SlotAnalyzer 朝結果\n日付: {operation_date.isoformat()}\n"
        f"overall status: {overall}\n" + "\n".join(summaries) + "\n\n" + warning_text
        + "\n\n" + "\n\n".join(_render_section_plain(section) for section in sections)
    )
    summary_html = "".join(f"<div>{html.escape(value)}</div>" for value in summaries)
    if warnings:
        warnings_html = "".join(
            f'<div style="margin:3px 0">&#9888; {html.escape(value)}</div>'
            for value in dict.fromkeys(warnings)
        )
    else:
        warnings_html = "<div>異常警告: なし</div>"
    sections_html = "".join(_render_section_html(section) for section in sections)
    html_body = (
        '<html><body style="margin:0;padding:12px;font-family:sans-serif;line-height:1.45;color:#222">'
        f'<h2 style="margin:0 0 8px">SlotAnalyzer 朝結果</h2>'
        f'<div>日付: {operation_date.isoformat()}</div>'
        f'<div style="font-size:18px;font-weight:bold;margin:3px 0 8px">overall status: {html.escape(overall)}</div>'
        f'{summary_html}'
        f'<div style="margin:12px 0;padding:9px;background:#fff4e5;border-left:4px solid #e67e22">{warnings_html}</div>'
        f'{sections_html}'
        "</body></html>"
    )
    return NotificationMessage(
        subject=f"[SlotAnalyzer][{overall}] {operation_date.isoformat()} 朝結果",
        plain=plain,
        html=html_body,
        overall_status=overall,
    )


def _already_sent(history_path: Path, state: dict) -> bool:
    for row in _read_rows(history_path):
        if (
            row.get("operation_date") == state.get("operation_date")
            and row.get("automation_run_id") == state.get("automation_run_id")
            and row.get("notification_type") == NOTIFICATION_TYPE
            and row.get("status") == "SENT"
        ):
            return True
    return False


def _safe_error(exc: Exception, secrets: tuple[str, ...] = ()) -> str:
    value = f"{type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value[:1000]


def send_notification_best_effort(
    state: dict,
    project_root: Path,
    history_path: Path | None = None,
    *,
    credential_reader: Callable[[], Credential] = read_windows_credential,
    smtp_factory=None,
    recipient: str | None = None,
    clock: Callable[[], datetime] = now_jst,
) -> bool:
    history_path = history_path or project_root / "logs/morning_automation/notification_history.csv"
    if _already_sent(history_path, state):
        return True
    attempted = clock().astimezone(JST)
    message = build_notification_message(state, project_root)
    recipient = (recipient if recipient is not None else os.environ.get(RECIPIENT_ENV, "")).strip()
    masked = mask_recipient(recipient) if recipient else ""
    message_hash = hashlib.sha256(
        (message.subject + "\n" + message.plain + "\n" + message.html).encode("utf-8")
    ).hexdigest()
    status = "FAILED"
    category = ""
    error = ""
    credential = None
    try:
        if not recipient or "@" not in recipient:
            raise NotificationConfigError(f"{RECIPIENT_ENV} is not configured with a valid address.")
        credential = credential_reader()
        email = EmailMessage()
        email["Subject"] = message.subject
        email["From"] = credential.username
        email["To"] = recipient
        email.set_content(message.plain)
        email.add_alternative(message.html, subtype="html")
        factory = smtp_factory or smtplib.SMTP_SSL
        with factory(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SEC, context=ssl.create_default_context()) as smtp:
            smtp.login(credential.username, credential.password)
            smtp.send_message(email)
        status = "SENT"
    except Exception as exc:
        category = type(exc).__name__
        error = _safe_error(exc, (credential.password,) if credential else ())
    completed = clock().astimezone(JST)
    try:
        append_history_csv(history_path, {
            "notification_id": f"notification_{uuid.uuid4().hex}",
            "automation_run_id": state.get("automation_run_id", ""),
            "operation_date": state.get("operation_date", ""),
            "notification_type": NOTIFICATION_TYPE,
            "attempted_at_jst": attempted.isoformat(),
            "completed_at_jst": completed.isoformat(),
            "overall_status": message.overall_status,
            "channel": "GMAIL_SMTP_SSL",
            "masked_recipient": masked,
            "status": status,
            "error_category": category,
            "error": error,
            "message_sha256": message_hash,
        }, HISTORY_FIELDS)
    except Exception as exc:
        print(f"WARNING: notification history append failed: {_safe_error(exc)}", file=sys.stderr)
    if status != "SENT":
        print(f"WARNING: Gmail notification failed: {category}: {error}", file=sys.stderr)
        return False
    return True
