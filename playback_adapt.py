from pyautogui import press

def threshold_control(metric: float, threshold: float, margin: float) -> None:
    if metric > threshold + margin:
        press('s')
    elif metric < threshold - margin:
        press('d')
