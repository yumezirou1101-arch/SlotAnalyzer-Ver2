import pyautogui
import time

print("3秒後に画像を探します...")
time.sleep(3)

pos = pyautogui.locateCenterOnScreen(
    "images/date.png",
    confidence=0.7
)

print("検索結果:", pos)

if pos:
    pyautogui.moveTo(pos, duration=1)
    pyautogui.click()
    print("クリック成功！")
else:
    print("画像が見つかりませんでした")