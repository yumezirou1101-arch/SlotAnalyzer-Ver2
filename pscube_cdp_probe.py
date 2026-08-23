from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright, Response


# ============================================================
# SlotAnalyzer - P'sCUBE CDP Probe
# ============================================================
#
# Purpose:
#   Attach to an already-open Edge/Chrome instance via CDP
#   (default http://127.0.0.1:9222), locate the P'sCUBE tab,
#   and capture the JSON responses used by the currently opened
#   machine page.
#
# Important:
#   - This script does NOT launch a new browser.
#   - This script does NOT modify existing SlotAnalyzer data/models.
#   - It only observes responses from the already-open browser tab.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "dstation_takasaki_honten"
    / "pscube_cdp_probe"
)

CDP_URL = "http://127.0.0.1:9222"

TARGET_ENDPOINTS = (
    "nc-m06-001.php",
    "nc-m06-003.php",
)

PSCUBE_HOST_TOKEN = "pscube.jp"


def header(title: str) -> None:
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture P'sCUBE responses from an existing CDP browser."
    )

    parser.add_argument(
        "--cdp",
        default=CDP_URL,
        help="CDP endpoint. Default: http://127.0.0.1:9222",
    )

    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=8,
        help="Seconds to wait after reload.",
    )

    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Attach without reloading the selected P'sCUBE tab.",
    )

    return parser.parse_args()


def safe_filename(value: str) -> str:
    value = re.sub(
        r"[^0-9A-Za-z._-]+",
        "_",
        value,
    )
    return value[:180]


def summarize_json(data: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if not isinstance(data, dict):
        summary["root_type"] = type(data).__name__
        return summary

    summary["top_level_keys"] = list(
        data.keys()
    )

    if "YMD_biz" in data:
        summary["YMD_biz"] = data.get(
            "YMD_biz"
        )

    dai = data.get("Dai")
    if isinstance(dai, dict):
        summary["cd_dai"] = dai.get(
            "cd_dai"
        )
        summary["cd_dai_prev"] = dai.get(
            "cd_dai_prev"
        )
        summary["cd_dai_next"] = dai.get(
            "cd_dai_next"
        )

    ki = data.get("Ki")
    if isinstance(ki, dict):
        summary["cd_kisyu"] = ki.get(
            "cd_kisyu"
        )
        summary["machine_name"] = ki.get(
            "name"
        )
        summary["bai"] = ki.get(
            "bai"
        )
        summary["nm_ps"] = ki.get(
            "nm_ps"
        )

    data_rows = data.get("Data")
    if isinstance(data_rows, list):
        titles = []
        for item in data_rows:
            if (
                isinstance(item, dict)
                and item.get("title") is not None
            ):
                titles.append(
                    item.get("title")
                )
        summary["data_titles"] = titles

    graph_blocks = []

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            if (
                "datas" in obj
                and isinstance(
                    obj.get("datas"),
                    list,
                )
            ):
                graph_blocks.append(
                    {
                        "path": path,
                        "title": obj.get("title"),
                        "id": obj.get("id"),
                        "points": len(
                            obj.get("datas")
                        ),
                    }
                )

            for key, value in obj.items():
                next_path = (
                    f"{path}.{key}"
                    if path
                    else key
                )
                walk(
                    value,
                    next_path,
                )

        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(
                    value,
                    f"{path}[{i}]",
                )

    walk(data)

    summary["graph_blocks"] = graph_blocks

    return summary


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    header(
        "SlotAnalyzer - P'sCUBE CDP Probe"
    )

    print(
        f"cdp endpoint : {args.cdp}"
    )

    print(
        f"output dir   : {OUTPUT_DIR}"
    )

    captured: list[
        dict[str, Any]
    ] = []

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(
            args.cdp
        )

        pages = []

        for context in browser.contexts:
            for page in context.pages:
                pages.append(
                    page
                )

        print()
        print(
            f"browser pages : {len(pages)}"
        )

        for i, page in enumerate(
            pages,
            start=1,
        ):
            print(
                f"[{i}] {page.url}"
            )

        pscube_pages = [
            page
            for page in pages
            if PSCUBE_HOST_TOKEN in page.url
        ]

        if not pscube_pages:
            raise RuntimeError(
                "No P'sCUBE tab found in the CDP browser."
            )

        # Prefer a machine-detail page if multiple P'sCUBE pages exist.
        machine_pages = [
            page
            for page in pscube_pages
            if "nc-v06-001.php" in page.url
        ]

        target_page = (
            machine_pages[-1]
            if machine_pages
            else pscube_pages[-1]
        )

        header(
            "TARGET TAB"
        )

        print(
            f"url   : {target_page.url}"
        )

        print(
            f"title : {target_page.title()}"
        )

        def on_response(
            response: Response,
        ) -> None:
            url = response.url

            if not any(
                endpoint in url
                for endpoint in TARGET_ENDPOINTS
            ):
                return

            record: dict[str, Any] = {
                "url": url,
                "status": response.status,
                "headers": dict(
                    response.headers
                ),
            }

            print()
            print(
                f"CAPTURED: {url}"
            )

            print(
                f"status  : {response.status}"
            )

            try:
                body_text = response.text()
                record[
                    "body_text"
                ] = body_text

                try:
                    parsed = json.loads(
                        body_text
                    )

                    record[
                        "json"
                    ] = parsed

                    summary = summarize_json(
                        parsed
                    )

                    record[
                        "summary"
                    ] = summary

                    for key in (
                        "YMD_biz",
                        "cd_dai",
                        "cd_dai_prev",
                        "cd_dai_next",
                        "cd_kisyu",
                        "machine_name",
                        "bai",
                        "nm_ps",
                    ):
                        if key in summary:
                            print(
                                f"{key:14}: "
                                f"{summary[key]}"
                            )

                    if (
                        "data_titles"
                        in summary
                    ):
                        print(
                            "data titles   : "
                            + ", ".join(
                                map(
                                    str,
                                    summary[
                                        "data_titles"
                                    ],
                                )
                            )
                        )

                    if (
                        "graph_blocks"
                        in summary
                    ):
                        print(
                            "graph blocks  : "
                            f"{len(summary['graph_blocks'])}"
                        )

                except Exception as exc:
                    record[
                        "json_error"
                    ] = repr(
                        exc
                    )

            except Exception as exc:
                record[
                    "body_error"
                ] = repr(
                    exc
                )

            captured.append(
                record
            )

        target_page.on(
            "response",
            on_response,
        )

        if args.no_reload:
            print()
            print(
                "no-reload mode: waiting for network activity..."
            )
        else:
            print()
            print(
                "Reloading current P'sCUBE tab..."
            )

            target_page.reload(
                wait_until="domcontentloaded",
                timeout=60_000,
            )

        target_page.wait_for_timeout(
            args.wait_seconds
            * 1000
        )

        # Remove listener cleanly.
        target_page.remove_listener(
            "response",
            on_response,
        )

        # Important: do NOT call browser.close() on a CDP-attached browser.
        # Detaching occurs when the Playwright context exits.

    header(
        "CAPTURE SUMMARY"
    )

    print(
        f"captured responses : {len(captured)}"
    )

    if not captured:
        print(
            "No target response captured."
        )
        print(
            "Retry with the P'sCUBE machine-detail tab active, "
            "or use --no-reload and manually switch/reload the page."
        )
        return

    index_rows = []

    for i, record in enumerate(
        captured,
        start=1,
    ):
        url = record[
            "url"
        ]

        endpoint = next(
            (
                endpoint
                for endpoint in TARGET_ENDPOINTS
                if endpoint in url
            ),
            "unknown",
        )

        stem = safe_filename(
            f"{i:02d}_{endpoint}"
        )

        raw_path = (
            OUTPUT_DIR
            / f"{stem}_raw.txt"
        )

        raw_path.write_text(
            record.get(
                "body_text",
                "",
            ),
            encoding="utf-8",
        )

        json_path = None

        if "json" in record:
            json_path = (
                OUTPUT_DIR
                / f"{stem}.json"
            )

            json_path.write_text(
                json.dumps(
                    record[
                        "json"
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        summary = record.get(
            "summary",
            {}
        )

        index_rows.append(
            {
                "endpoint": endpoint,
                "status": record.get(
                    "status"
                ),
                "url": url,
                "YMD_biz": summary.get(
                    "YMD_biz"
                ),
                "cd_dai": summary.get(
                    "cd_dai"
                ),
                "cd_kisyu": summary.get(
                    "cd_kisyu"
                ),
                "machine_name": summary.get(
                    "machine_name"
                ),
                "raw_file": str(
                    raw_path
                ),
                "json_file": (
                    str(
                        json_path
                    )
                    if json_path
                    else ""
                ),
            }
        )

    index_path = (
        OUTPUT_DIR
        / "pscube_cdp_probe_index.json"
    )

    index_path.write_text(
        json.dumps(
            index_rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"index saved        : {index_path}"
    )

    print()
    print(
        "CDP probe complete."
    )

    print(
        "Existing SlotAnalyzer prediction/data files were not modified."
    )


if __name__ == "__main__":
    main()
