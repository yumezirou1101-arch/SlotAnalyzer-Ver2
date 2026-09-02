from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as datetime_time, timedelta
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd


JST = ZoneInfo("Asia/Tokyo")
CDP_VERSION_URL = "http://127.0.0.1:9222/json/version"
CDP_PORT = 9222
TERMINAL_STATES = {
    "SUCCESS",
    "ALREADY_COMPLETE",
    "FAILED_FINAL",
    "NEEDS_MANUAL_REVIEW",
}

STORE_MARUHAN = "maruhan"
STORE_BIGMARCH = "bigmarch"
STORE_YASUDA = "yasuda"
STORE_ORDER = (STORE_MARUHAN, STORE_BIGMARCH, STORE_YASUDA)

YASUDA_COLUMNS = [
    "日付",
    "台番号",
    "機種名",
    "G数",
    "差枚",
    "BB",
    "RB",
    "合成確率",
    "BB確率",
    "RB確率",
]


class LockUnavailableError(RuntimeError):
    pass


class StateCorruptError(RuntimeError):
    pass


@dataclass
class ReadinessResult:
    ready: bool
    source_exists: bool
    source_path: str
    expected_data_date: str
    category: str
    error: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    status: str
    ok: bool
    error: str = ""
    artifacts: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class ProcessResult:
    returncode: int
    started_at_jst: str
    completed_at_jst: str
    elapsed_sec: float
    log_path: str


class _VisibleTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if not value:
            return
        self.text_parts.append(value)
        if self._in_title:
            self.title_parts.append(value)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts)

    @property
    def text(self) -> str:
        return " ".join(self.text_parts)


def now_jst() -> datetime:
    return datetime.now(JST)


def determine_operation_dates(current: datetime) -> tuple[date, date]:
    if current.tzinfo is None:
        current = current.replace(tzinfo=JST)
    operation_date = current.astimezone(JST).date()
    return operation_date, operation_date - timedelta(days=1)


def deadline_at(operation_date: date, value: datetime_time) -> datetime:
    return datetime.combine(operation_date, value, tzinfo=JST)


def maruhan_pipeline_start_allowed(
    current: datetime,
    operation_date: date,
    provisional_last_start: datetime_time = datetime_time(8, 30),
) -> bool:
    return current.astimezone(JST) < deadline_at(
        operation_date, provisional_last_start
    )


def other_store_retry_allowed(
    current: datetime,
    operation_date: date,
    final_deadline: datetime_time = datetime_time(9, 30),
) -> bool:
    return current.astimezone(JST) < deadline_at(operation_date, final_deadline)


class WindowsFileLock(AbstractContextManager):
    """Hold a non-blocking one-byte Windows OS lock for this context."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None

    def __enter__(self):
        if os.name != "nt":
            raise RuntimeError("WindowsFileLock requires Windows/msvcrt.")
        import msvcrt

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        if self._handle.seek(0, os.SEEK_END) == 0:
            self._handle.write(b"\0")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        self._handle.seek(0)
        try:
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self._handle.close()
            self._handle = None
            raise LockUnavailableError(f"Lock is already held: {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            import msvcrt

            try:
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self._handle.close()
                self._handle = None
        return False


def atomic_write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json_state(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateCorruptError(f"State file is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("stores"), dict):
        raise StateCorruptError(f"State file has an invalid structure: {path}")
    return value


def _coerce_columns(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result.columns = [
        str(column[-1] if isinstance(column, tuple) else column).strip()
        for column in result.columns
    ]
    return result


def _find_machine_table(html: str) -> pd.DataFrame:
    required = {"機種名", "台番号", "G数", "差枚"}
    candidates = []
    for table in pd.read_html(StringIO(html)):
        table = _coerce_columns(table)
        if required.issubset(set(table.columns)):
            candidates.append(table)
    if not candidates:
        raise RuntimeError("Main machine table was not found.")
    return max(candidates, key=len)


def _static_source_readiness(
    source_path: Path,
    expected_data_date: date,
    store_names: Iterable[str],
    minimum_machines: int,
    expected_current_machines: int | None,
) -> ReadinessResult:
    source_path = Path(source_path)
    base = {
        "source_path": str(source_path),
        "expected_data_date": expected_data_date.isoformat(),
    }
    if not source_path.exists():
        return ReadinessResult(False, False, category="SOURCE_MISSING", **base)
    try:
        size = source_path.stat().st_size
    except OSError as exc:
        return ReadinessResult(
            False, True, category="SOURCE_INVALID", error=str(exc), **base
        )
    if size <= 0:
        return ReadinessResult(
            False, True, category="SOURCE_INVALID", error="Source is empty.", **base
        )
    try:
        html = source_path.read_text(encoding="utf-8", errors="replace")
        parser = _VisibleTextParser()
        parser.feed(html)
        title = parser.title
        body = parser.text
        slash_date = expected_data_date.strftime("%Y/%m/%d")
        iso_date = expected_data_date.isoformat()
        date_ok = slash_date in title or slash_date in body or iso_date in html
        store_ok = any(name in title or name in body or name in html for name in store_names)
        table = _find_machine_table(html)
        table = table.rename(
            columns={"機種名": "machine_name", "台番号": "machine_no", "G数": "G", "差枚": "diff"}
        )
        table["machine_name"] = table["machine_name"].astype(str).str.strip()
        table["machine_no"] = pd.to_numeric(table["machine_no"], errors="coerce")
        for column in ("G", "diff"):
            table[column] = pd.to_numeric(
                table[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.strip(),
                errors="coerce",
            )
        records = len(table)
        unique_machines = int(table["machine_no"].nunique(dropna=True))
        duplicate_rows = int(table["machine_no"].duplicated(keep=False).sum())
        missing_machine = int(table["machine_no"].isna().sum())
        missing_name = int(table["machine_name"].isin(["", "nan", "None"]).sum())
        invalid_g = int(table["G"].isna().sum())
        invalid_diff = int(table["diff"].isna().sum())
        negative_g = int(((table["G"] < 0).fillna(False)).sum())
        details = {
            "size_bytes": size,
            "date_ok": date_ok,
            "store_ok": store_ok,
            "records": records,
            "unique_machines": unique_machines,
            "duplicate_rows": duplicate_rows,
            "missing_machine": missing_machine,
            "missing_name": missing_name,
            "invalid_G": invalid_g,
            "invalid_diff": invalid_diff,
            "negative_G": negative_g,
            "minimum_machines": minimum_machines,
            "expected_current_machines": expected_current_machines,
            "current_count_match": (
                records == expected_current_machines
                if expected_current_machines is not None
                else None
            ),
        }
        ready = all(
            (
                date_ok,
                store_ok,
                records >= minimum_machines,
                unique_machines == records,
                duplicate_rows == 0,
                missing_machine == 0,
                missing_name == 0,
                invalid_g == 0,
                invalid_diff == 0,
                negative_g == 0,
            )
        )
        return ReadinessResult(
            ready,
            True,
            category="READY" if ready else "SOURCE_INVALID",
            error="" if ready else "Static source quality validation failed.",
            details=details,
            **base,
        )
    except Exception as exc:
        return ReadinessResult(
            False,
            True,
            category="SOURCE_INVALID",
            error=f"{type(exc).__name__}: {exc}",
            details={"size_bytes": size},
            **base,
        )


def source_path_for(store: str, project_root: Path, expected_data_date: date) -> Path:
    ymd = expected_data_date.strftime("%Y%m%d")
    if store == STORE_MARUHAN:
        return project_root / f"ana_slo_{ymd}_source.html"
    if store == STORE_BIGMARCH:
        return project_root / f"ana_slo_bigmarch_oyagi_{ymd}_source.html"
    if store == STORE_YASUDA:
        return (
            project_root
            / "data"
            / "yasuda_maebashi"
            / "source_html"
            / f"ana_slo_{ymd}_source.html"
        )
    raise ValueError(f"Unknown store: {store}")


def check_source_readiness(
    store: str, project_root: Path, expected_data_date: date
) -> ReadinessResult:
    path = source_path_for(store, project_root, expected_data_date)
    if store == STORE_MARUHAN:
        return _static_source_readiness(
            path, expected_data_date, ("マルハンメガシティ前橋インター",), 450, 514
        )
    if store == STORE_BIGMARCH:
        return _static_source_readiness(
            path,
            expected_data_date,
            ("ビックマーチ高崎おおやぎ店", "ビッグマーチ高崎おおやぎ店"),
            200,
            None,
        )
    if store == STORE_YASUDA:
        return _static_source_readiness(
            path, expected_data_date, ("やすだ前橋店",), 300, 320
        )
    raise ValueError(f"Unknown store: {store}")


def build_fetch_command(store: str, project_root: Path, python_executable: str) -> list[str]:
    scripts = {
        STORE_MARUHAN: "ana_slo_maruhan_maebashi_click_fetch_v3.py",
        STORE_BIGMARCH: "ana_slo_bigmarch_oyagi_click_fetch_31days_v3.py",
        STORE_YASUDA: "ana_slo_yasuda_maebashi_click_fetch_v1.py",
    }
    command = [
        python_executable,
        str(project_root / "machine_number" / scripts[store]),
        "--max-days",
        "1",
    ]
    if store == STORE_BIGMARCH:
        command += ["--min-machines", "200"]
    _assert_safe_command(command)
    return command


def build_pipeline_command(
    store: str,
    project_root: Path,
    python_executable: str,
    operation_date: date,
    chrome_wait_sec: int = 15,
) -> list[str]:
    if store == STORE_MARUHAN:
        command = [
            python_executable,
            str(project_root / "machine_number" / "ana_slo_maruhan_maebashi_one_click_daily_update_v2.py"),
            "--skip-fetch",
            "--target-date",
            operation_date.isoformat(),
            "--chrome-wait-sec",
            str(chrome_wait_sec),
        ]
    elif store == STORE_BIGMARCH:
        command = [
            python_executable,
            str(project_root / "machine_number" / "ana_slo_bigmarch_oyagi_one_click_daily_update_v3.py"),
            "--skip-fetch",
            "--min-machines",
            "200",
            "--chrome-wait-sec",
            str(chrome_wait_sec),
        ]
    elif store == STORE_YASUDA:
        command = [
            python_executable,
            str(project_root / "machine_number" / "ana_slo_yasuda_maebashi_one_click_daily_update_v1.py"),
            "--skip-fetch",
            "--target-date",
            operation_date.isoformat(),
        ]
    else:
        raise ValueError(f"Unknown store: {store}")
    _assert_safe_command(command)
    return command


def _assert_safe_command(command: Iterable[str]) -> None:
    forbidden = {"--allow-gap", "--overwrite"}
    present = forbidden.intersection(command)
    if present:
        raise ValueError(f"Forbidden automation argument(s): {sorted(present)}")


def try_get_cdp(timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(CDP_VERSION_URL, timeout=timeout) as response:
            info = json.loads(response.read().decode("utf-8"))
        return info if info.get("webSocketDebuggerUrl") else None
    except Exception:
        return None


def chrome_candidates() -> list[Path]:
    return [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
        / "Google/Chrome/Application/chrome.exe",
    ]


def ensure_cdp(
    project_root: Path,
    wait_sec: int = 15,
    probe: Callable[[float], dict | None] = try_get_cdp,
    popen: Callable = subprocess.Popen,
) -> dict:
    info = probe(2.0)
    if info is not None:
        return info
    chrome = next((path for path in chrome_candidates() if path.exists()), None)
    if chrome is None:
        raise FileNotFoundError("Google Chrome executable was not found.")
    profile = project_root / ".chrome_remote_profile_9222"
    profile.mkdir(parents=True, exist_ok=True)
    command = [
        str(chrome),
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank",
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    popen(
        command,
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    end = time.monotonic() + wait_sec
    while time.monotonic() < end:
        info = probe(1.0)
        if info is not None:
            return info
        time.sleep(0.5)
    raise RuntimeError(f"Chrome CDP did not become ready within {wait_sec} seconds.")


def run_logged_subprocess(
    command: list[str],
    cwd: Path,
    log_path: Path,
    stage: str,
    environment: dict[str, str] | None = None,
    clock: Callable[[], datetime] = now_jst,
) -> ProcessResult:
    _assert_safe_command(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = clock()
    started_perf = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write(f"stage={stage}\n")
        log.write(f"started_at_jst={started_at.isoformat()}\n")
        log.write(f"cwd={cwd}\n")
        log.write(f"command={subprocess.list2cmdline(command)}\n")
        log.write("--- child output ---\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode = process.wait()
        elapsed = time.perf_counter() - started_perf
        completed_at = clock()
        log.write("\n--- automation result ---\n")
        log.write(f"completed_at_jst={completed_at.isoformat()}\n")
        log.write(f"elapsed_sec={elapsed:.6f}\n")
        log.write(f"returncode={returncode}\n")
        log.flush()
        os.fsync(log.fileno())
    return ProcessResult(
        returncode,
        started_at.isoformat(),
        completed_at.isoformat(),
        elapsed,
        str(log_path),
    )


def append_history_csv(path: Path, row: dict, fields: list[str]) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    with WindowsFileLock(lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(paths: Iterable[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def verify_maruhan_completion(project_root: Path, operation_date: date) -> VerificationResult:
    ymd = operation_date.strftime("%Y%m%d")
    expected = operation_date - timedelta(days=1)
    analysis = project_root / "data/maruhan_maebashi/machine_number/analysis_31days_deep"
    dir64 = analysis / "64_Ver4_2_future_top10"
    files64 = [
        dir64 / f"64_prediction_{ymd}_all514.csv",
        dir64 / f"64_prediction_{ymd}_top10.csv",
        dir64 / f"64_prediction_{ymd}_metadata.csv",
    ]
    existing64 = [path for path in files64 if path.exists()]
    if not existing64:
        downstream = [
            analysis
            / "77_live_integrated_prediction_report"
            / f"77_integrated_prediction_{ymd}.csv",
            analysis
            / "77_live_integrated_prediction_report"
            / f"77_integrated_prediction_{ymd}_summary.csv",
            analysis
            / "79_one_click_prediction_pipeline"
            / f"79_pipeline_{ymd}_status.csv",
        ]
        existing_downstream = [path for path in downstream if path.exists()]
        if existing_downstream:
            return VerificationResult(
                "INCONSISTENT_PIPELINE_ARTIFACTS",
                False,
                "77/79 artifacts exist without a formal 64 set.",
                [str(path) for path in existing_downstream],
            )
        return VerificationResult("NONE", False)
    if len(existing64) != 3:
        return VerificationResult(
            "PARTIAL_64", False, "Only part of the formal 64 set exists.", [str(p) for p in existing64]
        )
    if not _nonempty(files64):
        return VerificationResult("INVALID_64", False, "A formal 64 file is empty.", [str(p) for p in files64])
    try:
        metadata = pd.read_csv(files64[2], encoding="utf-8-sig")
        if len(metadata) != 1:
            raise RuntimeError("64 metadata must contain exactly one row.")
        row = metadata.iloc[0]
        required = {
            "generated_at_jst", "target_date", "latest_data_date", "forward_guard_version",
            "forward_valid", "forward_cutoff_jst", "target_actual_absent_at_generation",
            "target_source_absent_at_generation", "daily_csv_sha256", "source_html_sha256",
            "all514_sha256", "top10_sha256", "machines_ranked", "target_actual_used",
            "model", "weight_fingerprint", "weight_sum",
        }
        missing = required - set(metadata.columns)
        if missing:
            raise RuntimeError(f"64 metadata columns missing: {sorted(missing)}")
        generated = datetime.fromisoformat(str(row["generated_at_jst"]))
        if generated.tzinfo is None:
            raise RuntimeError("generated_at_jst is timezone-naive.")
        generated_jst = generated.astimezone(JST)
        if generated_jst.date() != operation_date or generated_jst.time().replace(tzinfo=None) >= datetime_time(9, 0):
            raise RuntimeError("generated_at_jst is outside the valid formal window.")
        if pd.Timestamp(row["target_date"]).date() != operation_date:
            raise RuntimeError("metadata target_date mismatch.")
        if pd.Timestamp(row["latest_data_date"]).date() != expected:
            raise RuntimeError("metadata latest_data_date mismatch.")
        if not _truthy(row["forward_valid"]):
            raise RuntimeError("metadata forward_valid is false.")
        for key in ("forward_guard_version", "model", "weight_fingerprint"):
            if not str(row[key]).strip() or str(row[key]).strip().lower() == "nan":
                raise RuntimeError(f"metadata {key} is empty.")
        if str(row["forward_cutoff_jst"]).strip() != "09:00 Asia/Tokyo":
            raise RuntimeError("metadata forward_cutoff_jst mismatch.")
        if not math.isfinite(float(row["weight_sum"])):
            raise RuntimeError("metadata weight_sum is not finite.")
        if not _truthy(row["target_actual_absent_at_generation"]) or not _truthy(row["target_source_absent_at_generation"]):
            raise RuntimeError("metadata actual/source absence flags are false.")
        if _truthy(row["target_actual_used"]):
            raise RuntimeError("metadata says target actual was used.")
        if int(row["machines_ranked"]) != 514:
            raise RuntimeError("metadata machines_ranked is not 514.")
        if sha256_file(files64[0]) != str(row["all514_sha256"]):
            raise RuntimeError("all514 SHA-256 mismatch.")
        if sha256_file(files64[1]) != str(row["top10_sha256"]):
            raise RuntimeError("top10 SHA-256 mismatch.")
        daily = project_root / "data/maruhan_maebashi/machine_number" / f"ana_slo_{expected:%Y%m%d}.csv"
        source = project_root / f"ana_slo_{expected:%Y%m%d}_source.html"
        if not _nonempty((daily, source)):
            raise RuntimeError("Formal input daily/source file is missing or empty.")
        if sha256_file(daily) != str(row["daily_csv_sha256"]):
            raise RuntimeError("daily CSV SHA-256 mismatch.")
        if sha256_file(source) != str(row["source_html_sha256"]):
            raise RuntimeError("source HTML SHA-256 mismatch.")
    except Exception as exc:
        return VerificationResult(
            "INVALID_64", False, f"{type(exc).__name__}: {exc}", [str(p) for p in files64]
        )
    dir77 = analysis / "77_live_integrated_prediction_report"
    files77 = [
        dir77 / f"77_integrated_prediction_{ymd}.csv",
        dir77 / f"77_integrated_prediction_{ymd}_summary.csv",
    ]
    status79 = analysis / "79_one_click_prediction_pipeline" / f"79_pipeline_{ymd}_status.csv"
    if not _nonempty((*files77, status79)):
        return VerificationResult(
            "COMPLETE_64_INCOMPLETE_PIPELINE",
            False,
            "Formal 64 is complete but 77/79 is incomplete.",
            [str(p) for p in files64 if p.exists()] + [str(p) for p in (*files77, status79) if p.exists()],
        )
    try:
        status = pd.read_csv(status79, encoding="utf-8-sig")
        required_stages = {"64_NORMAL", "74_A_TYPE", "75_JUGGLER", "77_INTEGRATED"}
        if not {"stage", "pipeline_complete"}.issubset(status.columns):
            raise RuntimeError("79 status columns are incomplete.")
        if not required_stages.issubset(set(status["stage"].astype(str))):
            raise RuntimeError("79 status does not contain every required stage.")
        if not status["pipeline_complete"].map(_truthy).all():
            raise RuntimeError("79 pipeline_complete is not true for every stage.")
    except Exception as exc:
        return VerificationResult(
            "COMPLETE_64_INCOMPLETE_PIPELINE", False, f"{type(exc).__name__}: {exc}", [str(p) for p in files64 + files77 + [status79]]
        )
    artifacts = [str(path) for path in files64 + files77 + [status79]]
    return VerificationResult("COMPLETE", True, artifacts=artifacts)


def verify_big_march_completion(project_root: Path, operation_date: date) -> VerificationResult:
    ymd = operation_date.strftime("%Y%m%d")
    expected = operation_date - timedelta(days=1)
    data_dir = project_root / "data/bigmarch_takasaki_oyagi/machine_number"
    daily = data_dir / f"ana_slo_bigmarch_oyagi_{expected:%Y%m%d}.csv"
    analysis = data_dir / "analysis_31days_deep"
    files = [
        daily,
        analysis / "08_juggler_recent7_top3_forward/08_forward_status.csv",
        analysis / "11_nonjuggler_weekday_top1_forward/11_nonjuggler_weekday_top1_forward_summary.csv",
        analysis / "09_juggler_recent7_future_ranking" / f"09_prediction_{ymd}_all_juggler.csv",
        analysis / "09_juggler_recent7_future_ranking" / f"09_prediction_{ymd}_top10.csv",
        analysis / "09_juggler_recent7_future_ranking" / f"09_prediction_{ymd}_metadata.csv",
        analysis / "12_nonjuggler_weekday_future_ranking" / f"12_prediction_{ymd}_all_nonjuggler.csv",
        analysis / "12_nonjuggler_weekday_future_ranking" / f"12_prediction_{ymd}_top10.csv",
        analysis / "12_nonjuggler_weekday_future_ranking" / f"12_prediction_{ymd}_metadata.csv",
    ]
    if not any(path.exists() for path in files):
        return VerificationResult("NONE", False)
    if not _nonempty(files):
        return VerificationResult("PARTIAL", False, "Big March completion artifacts are partial.", [str(p) for p in files if p.exists()])
    try:
        frame = pd.read_csv(daily, encoding="utf-8-sig")
        required = {"date", "machine_name", "machine_no", "G", "diff"}
        if not required.issubset(frame.columns) or len(frame) < 200:
            raise RuntimeError("Big March daily CSV schema/count is invalid.")
        dates = pd.to_datetime(frame["date"], errors="raise").dt.date.unique().tolist()
        machines = pd.to_numeric(frame["machine_no"], errors="raise")
        games = pd.to_numeric(frame["G"], errors="raise")
        differences = pd.to_numeric(frame["diff"], errors="raise")
        missing_name = frame["machine_name"].astype(str).str.strip().isin(["", "nan", "None"]).any()
        if (
            dates != [expected]
            or machines.isna().any()
            or machines.nunique() != len(frame)
            or machines.duplicated().any()
            or games.isna().any()
            or differences.isna().any()
            or missing_name
            or (games < 0).any()
        ):
            raise RuntimeError("Big March daily CSV quality is invalid.")
        for metadata_path in (files[5], files[8]):
            metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
            if len(metadata) != 1:
                raise RuntimeError(f"Invalid metadata rows: {metadata_path}")
            row = metadata.iloc[0]
            if pd.Timestamp(row["target_date"]).date() != operation_date or pd.Timestamp(row["latest_data_date"]).date() != expected:
                raise RuntimeError(f"Metadata date mismatch: {metadata_path}")
    except Exception as exc:
        return VerificationResult("INVALID", False, f"{type(exc).__name__}: {exc}", [str(p) for p in files])
    return VerificationResult("COMPLETE", True, artifacts=[str(p) for p in files])


def verify_yasuda_completion(project_root: Path, operation_date: date) -> VerificationResult:
    expected = operation_date - timedelta(days=1)
    daily = project_root / "data/yasuda_maebashi/machine_number" / f"ana_slo_{expected:%Y%m%d}.csv"
    if not daily.exists():
        return VerificationResult("NONE", False)
    if daily.stat().st_size <= 0:
        return VerificationResult("INVALID", False, "Yasuda daily CSV is empty.", [str(daily)])
    try:
        frame = pd.read_csv(daily, encoding="utf-8-sig")
        if list(frame.columns) != YASUDA_COLUMNS or len(frame) != 320:
            raise RuntimeError("Yasuda CSV columns/count are invalid.")
        dates = pd.to_datetime(frame["日付"], errors="raise").dt.date.unique().tolist()
        machines = pd.to_numeric(frame["台番号"], errors="raise")
        games = pd.to_numeric(frame["G数"], errors="raise")
        differences = pd.to_numeric(frame["差枚"], errors="raise")
        if (
            dates != [expected]
            or machines.isna().any()
            or machines.nunique() != 320
            or machines.duplicated().any()
            or games.isna().any()
            or differences.isna().any()
        ):
            raise RuntimeError("Yasuda CSV date/machine uniqueness is invalid.")
        if frame["機種名"].astype(str).str.strip().isin(["", "nan", "None"]).any() or (games < 0).any():
            raise RuntimeError("Yasuda CSV values are invalid.")
    except Exception as exc:
        return VerificationResult("INVALID", False, f"{type(exc).__name__}: {exc}", [str(daily)])
    return VerificationResult("COMPLETE", True, artifacts=[str(daily)])


def verify_store_completion(store: str, project_root: Path, operation_date: date) -> VerificationResult:
    if store == STORE_MARUHAN:
        return verify_maruhan_completion(project_root, operation_date)
    if store == STORE_BIGMARCH:
        return verify_big_march_completion(project_root, operation_date)
    if store == STORE_YASUDA:
        return verify_yasuda_completion(project_root, operation_date)
    raise ValueError(f"Unknown store: {store}")


def classify_fetch_result(returncode: int, readiness: ReadinessResult, before_deadline: bool) -> str:
    if readiness.ready:
        return "READY"
    if readiness.source_exists:
        return "NEEDS_MANUAL_REVIEW"
    if not before_deadline:
        return "FAILED_FINAL"
    return "WAITING_FOR_DATA" if returncode == 0 else "FAILED_RETRYABLE"


def classify_pipeline_result(returncode: int, verification: VerificationResult) -> str:
    if returncode == 0 and verification.ok:
        return "SUCCESS"
    if verification.status in {
        "PARTIAL_64",
        "INVALID_64",
        "COMPLETE_64_INCOMPLETE_PIPELINE",
        "INCONSISTENT_PIPELINE_ARTIFACTS",
        "PARTIAL",
        "INVALID",
    }:
        return "NEEDS_MANUAL_REVIEW"
    return "FAILED_FINAL"
