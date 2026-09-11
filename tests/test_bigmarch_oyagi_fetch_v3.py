from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MACHINE_DIR = PROJECT_ROOT / "machine_number"
for path in (PROJECT_ROOT, MACHINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ana_slo_bigmarch_oyagi_click_fetch_31days_v3 as fetch


class FakePage:
    def __init__(self, url: str, titles: list[str]):
        self.url = url
        self._titles = list(titles)
        self.goto_calls: list[tuple[str, dict]] = []
        self.go_back_calls: list[dict] = []
        self.wait_calls: list[int] = []

    async def title(self) -> str:
        if not self._titles:
            return ""
        if len(self._titles) == 1:
            return self._titles[0]
        return self._titles.pop(0)

    async def goto(self, url: str, **kwargs):
        self.goto_calls.append((url, kwargs))
        self.url = url
        return None

    async def go_back(self, **kwargs):
        self.go_back_calls.append(kwargs)
        raise AssertionError(
            "ensure_list_page() must not use browser history recovery."
        )

    async def wait_for_timeout(self, milliseconds: int):
        self.wait_calls.append(milliseconds)


class BigMarchListUrlTests(unittest.TestCase):
    def test_exact_store_list_url_is_recognized(self):
        self.assertTrue(fetch.is_store_list_url(fetch.STORE_LIST_URL))

    def test_query_and_fragment_are_ignored(self):
        url = fetch.STORE_LIST_URL + "?example=1#google_vignette"
        self.assertTrue(fetch.is_store_list_url(url))

    def test_fragment_only_is_ignored(self):
        url = fetch.STORE_LIST_URL + "#google_vignette"
        self.assertTrue(fetch.is_store_list_url(url))

    def test_daily_page_is_not_store_list_url(self):
        url = (
            "https://ana-slo.com/"
            "%e3%83%9b%e3%83%bc%e3%83%ab%e3%83%87%e3%83%bc%e3%82%bf/"
            "%e7%be%a4%e9%a6%ac%e7%9c%8c/"
            "2026-09-10-bigmarch-example/"
        )
        self.assertFalse(fetch.is_store_list_url(url))

    def test_different_host_is_not_store_list_url(self):
        url = fetch.STORE_LIST_URL.replace(
            "https://ana-slo.com/",
            "https://example.com/",
            1,
        )
        self.assertFalse(fetch.is_store_list_url(url))


class BigMarchEnsureListPageTests(unittest.IsolatedAsyncioTestCase):
    async def test_blank_title_on_correct_list_url_does_not_navigate(self):
        page = FakePage(
            fetch.STORE_LIST_URL + "#google_vignette",
            [""],
        )

        await fetch.ensure_list_page(page)

        self.assertEqual(page.goto_calls, [])
        self.assertEqual(page.go_back_calls, [])
        self.assertEqual(page.wait_calls, [])

    async def test_wrong_page_recovers_with_explicit_store_list_url(self):
        valid_title = f"{fetch.STORE_TEXTS[0]} {fetch.LIST_TITLE_TEXT}"
        page = FakePage(
            "https://ana-slo.com/some-other-page/",
            ["not the list page", valid_title],
        )

        await fetch.ensure_list_page(page)

        self.assertEqual(len(page.goto_calls), 1)
        url, kwargs = page.goto_calls[0]
        self.assertEqual(url, fetch.STORE_LIST_URL)
        self.assertEqual(kwargs["wait_until"], "domcontentloaded")
        self.assertEqual(kwargs["timeout"], 30000)
        self.assertEqual(page.go_back_calls, [])
        self.assertEqual(page.wait_calls, [1200])

    async def test_explicit_recovery_still_fails_closed_if_title_is_invalid(self):
        page = FakePage(
            "https://ana-slo.com/some-other-page/",
            ["not the list page", "still invalid"],
        )

        with self.assertRaises(RuntimeError):
            await fetch.ensure_list_page(page)

        self.assertEqual(len(page.goto_calls), 1)
        self.assertEqual(page.go_back_calls, [])


if __name__ == "__main__":
    unittest.main()