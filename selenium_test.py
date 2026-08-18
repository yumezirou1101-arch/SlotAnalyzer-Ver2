import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time

url = "https://ana-slo.com/ホールデータ/群馬県/マルハンメガシティ前橋インター-データ一覧/"

options = uc.ChromeOptions()

driver = uc.Chrome(options=options)

driver.get(url)

print("5秒待ちます...")
time.sleep(5)

print("タイトル")
print(driver.title)

print()

links = driver.find_elements(By.TAG_NAME, "a")

print("aタグ数:", len(links))

for link in links[:20]:
    print(link.text, "=>", link.get_attribute("href"))

input("Enterで終了")

driver.quit()