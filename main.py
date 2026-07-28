# main.py
"""SlotAnalyzer の実行ファイル。"""

from __future__ import annotations

import argparse
import sys

from config import (
    DEFAULT_STORE_KEY,
    OUTPUT_DIR,
    OUTPUT_FILE_NAME,
    STORES,
)
from excel_writer import save_workbook
from scraper import ScrapingError, scrape_store


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を読み取る。"""
    parser = argparse.ArgumentParser(
        description="アナスロの全台データをExcelへ保存します。"
    )

    parser.add_argument(
        "--store",
        choices=STORES.keys(),
        default=DEFAULT_STORE_KEY,
        help="取得する店舗キー",
    )

    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="ブラウザ画面を表示して取得する",
    )

    return parser.parse_args()


def main() -> int:
    """データ取得からExcel保存までを実行する。"""
    args = parse_arguments()
    store = STORES[args.store]

    print(f"{store.name} の最新営業日データを取得します。")

    try:
        rows = scrape_store(
            store=store,
            headless=not args.show_browser,
        )

        output_path = save_workbook(
            rows=rows,
            output_path=OUTPUT_DIR / OUTPUT_FILE_NAME,
        )

    except ScrapingError as error:
        print(f"取得に失敗しました: {error}", file=sys.stderr)
        return 1

    except Exception as error:
        print(f"予期しないエラーです: {error}", file=sys.stderr)
        return 1

    print(f"{len(rows)}台分のデータを保存しました。")
    print(f"保存先: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())