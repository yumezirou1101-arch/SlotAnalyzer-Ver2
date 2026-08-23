from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright, Response


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "dstation_takasaki_honten"
    / "pscube_probe"
)

DEFAULT_URL = (
    "https://www.pscube.jp/h/a718768/"
    "cgi-bin/nc-v06-001.php?cd_dai=0810"
)

TARGET_ENDPOINTS = (
    "nc-m06-001.php",
    "nc-m06-003.php",
)


def header(title: str) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture P'sCUBE JSON responses for one machine page."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="P'sCUBE machine-page URL.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser headless.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=8,
        help="Seconds to wait after page load.",
    )
    return parser.parse_args()


def safe_filename(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    return value[:180]


def summarize_json(data: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if not isinstance(data, dict):
        summary["root_type"] = type(data).__name__
        return summary

    summary["top_level_keys"] = list(data.keys())

    if "YMD_biz" in data:
        summary["YMD_biz"] = data.get("YMD_biz")

    dai = data.get("Dai")
    if isinstance(dai, dict):
        summary["cd_dai"] = dai.get("cd_dai")
        summary["cd_dai_prev"] = dai.get("cd_dai_prev")
        summary["cd_dai_next"] = dai.get("cd_dai_next")

    ki = data.get("Ki")
    if isinstance(ki, dict):
        summary["cd_kisyu"] = ki.get("cd_kisyu")
        summary["machine_name"] = ki.get("name")
        summary["bai"] = ki.get("bai")
        summary["nm_ps"] = ki.get("nm_ps")

    data_rows = data.get("Data")
    if isinstance(data_rows, list):
        titles = []
        for item in data_rows:
            if isinstance(item, dict) and item.get("title") is not None:
                titles.append(item.get("title"))
        summary["data_titles"] = titles

    graph_blocks = []

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            if "datas" in obj and isinstance(obj.get("datas"), list):
                graph_blocks.append(
                    {
                        "path": path,
                        "title": obj.get("title"),
                        "id": obj.get("id"),
                        "points": len(obj.get("datas")),
                    }
                )
            for key, value in obj.items():
                next_path = f"{path}.{key}" if path else key
                walk(value, next_path)

        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]")

    walk(data)
    summary["graph_blocks"] = graph_blocks

    return summary


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    header("SlotAnalyzer - P'sCUBE Probe")
    print(f"url          : {args.url}")
    print(f"headless     : {args.headless}")
    print(f"output dir   : {OUTPUT_DIR}")

    captured: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(locale="ja-JP")
        page = context.new_page()

        def on_response(response: Response) -> None:
            url = response.url

            if not any(endpoint in url for endpoint in TARGET_ENDPOINTS):
                return

            record: dict[str, Any] = {
                "url": url,
                "status": response.status,
                "headers": dict(response.headers),
            }

            print()
            print(f"CAPTURED: {url}")
            print(f"status  : {response.status}")

            try:
                body_text = response.text()
                record["body_text"] = body_text

                try:
                    parsed = json.loads(body_text)
                    record["json"] = parsed
                    record["summary"] = summarize_json(parsed)

                    summary = record["summary"]

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
                            print(f"{key:14}: {summary[key]}")

                    if "data_titles" in summary:
                        print(
                            "data titles   : "
                            + ", ".join(map(str, summary["data_titles"]))
                        )

                    if "graph_blocks" in summary:
                        print(
                            "graph blocks  : "
                            f"{len(summary['graph_blocks'])}"
                        )

                except Exception as exc:
                    record["json_error"] = repr(exc)

            except Exception as exc:
                record["body_error"] = repr(exc)

            captured.append(record)

        page.on("response", on_response)

        page.goto(
            args.url,
            wait_until="domcontentloaded",
            timeout=60_000,
        )

        page.wait_for_timeout(args.wait_seconds * 1000)

        print()
        print(f"page title    : {page.title()}")
        print(f"current url   : {page.url}")

        browser.close()

    header("CAPTURE SUMMARY")
    print(f"captured responses : {len(captured)}")

    if not captured:
        print("No target endpoint response was captured.")
        print(
            "Retry with the full machine-page URL copied from Edge "
            "if the default URL is not sufficient."
        )
        return

    index_rows = []

    for i, record in enumerate(captured, start=1):
        url = record["url"]

        endpoint = next(
            (
                endpoint
                for endpoint in TARGET_ENDPOINTS
                if endpoint in url
            ),
            "unknown",
        )

        stem = safe_filename(f"{i:02d}_{endpoint}")

        raw_path = OUTPUT_DIR / f"{stem}_raw.txt"
        raw_path.write_text(
            record.get("body_text", ""),
            encoding="utf-8",
        )

        json_path = None

        if "json" in record:
            json_path = OUTPUT_DIR / f"{stem}.json"
            json_path.write_text(
                json.dumps(
                    record["json"],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        summary = record.get("summary", {})

        index_rows.append(
            {
                "endpoint": endpoint,
                "status": record.get("status"),
                "url": url,
                "YMD_biz": summary.get("YMD_biz"),
                "cd_dai": summary.get("cd_dai"),
                "cd_kisyu": summary.get("cd_kisyu"),
                "machine_name": summary.get("machine_name"),
                "raw_file": str(raw_path),
                "json_file": str(json_path) if json_path else "",
            }
        )

    index_path = OUTPUT_DIR / "pscube_probe_index.json"
    index_path.write_text(
        json.dumps(
            index_rows,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"index saved        : {index_path}")
    print()
    print("Probe complete.")
    print(
        "Existing SlotAnalyzer prediction/data files were not modified."
    )


if __name__ == "__main__":
    main()
