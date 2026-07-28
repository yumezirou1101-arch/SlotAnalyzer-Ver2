# config.py
"""SlotAnalyzer の設定ファイル。"""

from dataclasses import dataclass, field
from pathlib import Path


BASE_URL = "https://ana-slo.com"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_FILE_NAME = "slot_data.xlsx"

PAGE_TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class StoreConfig:
    """店舗ごとの取得設定。"""

    name: str
    url: str
    selectors: dict[str, tuple[str, ...]] = field(default_factory=dict)


STORES: dict[str, StoreConfig] = {
    "maruhan_megacity_maebashi_inter": StoreConfig(
        name="マルハンメガシティ前橋インター",
        url=(
            f"{BASE_URL}/%E3%83%9B%E3%83%BC%E3%83%AB%E3%83%87%E3%83%BC%E3%82%BF/"
            "%E7%BE%A4%E9%A6%AC%E7%9C%8C/"
            "%E3%83%9E%E3%83%AB%E3%83%8F%E3%83%B3%E3%83%A1%E3%82%AC%E3%82%B7%E3%83%86%E3%82%A3"
            "%E5%89%8D%E6%A9%8B%E3%82%A4%E3%83%B3%E3%82%BF%E3%83%BC-%E3%83%87%E3%83%BC%E3%82%BF"
            "%E4%B8%80%E8%A6%A7/"
        ),
        selectors={
            "date_links": (
    'div.table-data-cell > a[href*="-data/"]',
),
            "machine_links": (
                "main a[href]",
                "article a[href]",
                ".entry-content a[href]",
            ),
            "data_tables": (
                "main table",
                "article table",
                ".entry-content table",
            ),
        },
    ),
}

DEFAULT_STORE_KEY = "maruhan_megacity_maebashi_inter"