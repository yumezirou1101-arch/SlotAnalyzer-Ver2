import pyautogui
import time

print("5秒後に開始します。")
time.sleep(5)

print("マウス位置を表示します。")
print("終了は Ctrl + C")

while True:
    x, y = pyautogui.position()
    print(f"\rX={x:4}  Y={y:4}", end="")
    time.sleep(0.1)