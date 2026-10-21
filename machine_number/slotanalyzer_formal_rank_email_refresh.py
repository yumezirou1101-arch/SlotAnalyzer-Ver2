from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_SCRIPT = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_formal_rank_position_analysis_v1.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "formal_rank_position_analysis_v1"
)

TOPN_PATH = OUTPUT_DIR / "02_topn_summary.csv"
METADATA_PATH = OUTPUT_DIR / "05_analysis_metadata.csv"


def run_refresh(
    project_root: Path = PROJECT_ROOT,
    timeout_sec: int = 120,
) -> tuple[bool, str]:
    project_root = Path(project_root).resolve()

    script = (
        project_root
        / "machine_number"
        / "ana_slo_formal_rank_position_analysis_v1.py"
    )

    if not script.is_file():
        return False, f"analysis script missing: {script}"

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"analysis refresh timeout: {timeout_sec}s"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if result.returncode != 0:
        message = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"returncode={result.returncode}"
        )
        return False, message

    topn = (
        project_root
        / "data"
        / "maruhan_maebashi"
        / "machine_number"
        / "formal_rank_position_analysis_v1"
        / "02_topn_summary.csv"
    )
    metadata = topn.with_name("05_analysis_metadata.csv")

    if not topn.is_file() or not metadata.is_file():
        return False, "required analysis outputs were not generated"

    return True, "PASS"


def main() -> int:
    ok, message = run_refresh()

    print(
        "FORMAL_RANK_EMAIL_REFRESH="
        + ("PASS" if ok else "FAILED")
    )
    print(message)

    # Best Effort:
    # failure is reported to the caller, but this small wrapper itself
    # does not mutate Champion/ranking/Forward definitions.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
