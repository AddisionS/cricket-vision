import cv2 as cv
import numpy as np

from mss import mss


class WindowCapture:

    def __init__(self, monitor_number, top, left, width, height):

        self.sct = mss()

        monitor = self.sct.monitors[monitor_number]

        self.monitor = {
            "top": monitor["top"] + top,
            "left": monitor["left"] + left,
            "width": width,
            "height": height
        }

    def get_screenshot(self):

        raw = self.sct.grab(self.monitor)

        screenshot = np.frombuffer(
            raw.rgb,
            dtype=np.uint8
        ).reshape(
            raw.height,
            raw.width,
            3
        )

        screenshot = cv.cvtColor(
            screenshot,
            cv.COLOR_RGB2BGR
        )

        return screenshot