import time
import pyautogui

print("Tu as 5 secondes pour cliquer sur le jeu dans Chrome...")
time.sleep(5)

pyautogui.press("right")
print("Touche RIGHT envoyée")

time.sleep(1)

pyautogui.press("up")
print("Touche UP envoyée")

time.sleep(1)

pyautogui.press("down")
print("Touche DOWN envoyée")