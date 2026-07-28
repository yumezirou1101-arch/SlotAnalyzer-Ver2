from playwright.sync_api import sync_playwright
from config import STORE


def get_latest_day():

    with sync_playwright() as p:

        browser = p.firefox.launch(headless=False)

        page = browser.new_page()

        print("店舗ページを開いています...")

        page.goto(STORE["url"])

        page.wait_for_timeout(5000)

        print("ページタイトル")
        print(page.title())

        links = page.locator("a")

        latest_text = None
        latest_url = None

        for i in range(links.count()):

            text = links.nth(i).inner_text().strip()
            href = links.nth(i).get_attribute("href")

            if text.startswith("2026/"):

                latest_text = text
                latest_url = href
                break

                print()
                print()
        print("最新日")
        print(latest_text)
        print(latest_url)

        return latest_text, latest_url
def open_latest_page(url):

    with sync_playwright() as p:

        browser = p.firefox.launch(headless=False)

        page = browser.new_page()

        print("\n最新日のページを開いています...")

        page.goto(url)

        page.wait_for_timeout(5000)

        print("\nページタイトル")
        print(page.title())

        browser.close()

if __name__ == "__main__":

    day, url = get_latest_day()

    print()
    print("取得成功")
    print(day)
    print(url)

    open_latest_page(url)