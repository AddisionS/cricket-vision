import os
import cv2 as cv
import numpy as np
from vision.wicket_detector import WicketDetector
from vision.out_screen_detector import OutDetector
from vision.bat_detector import BatDetector
from vision.ball_detector import BallDetector
from vision.tracker_manager import TrackerManager


class VisionSystem:

    def __init__(self):

        BASE_DIR = os.path.dirname(
            os.path.abspath(__file__)
        )

        out_template_path = os.path.join(
            BASE_DIR,
            "..",
            "assets",
            "out_template.png"
        )

        self.tracker_manager = TrackerManager()

        self.wicket_detector = WicketDetector(
            threshold=120
        )

        self.out_detector = OutDetector(
            out_template_path,
            threshold=0.8
        )

        # x1, y1, x2, y2
        self.wicket_roi = (
            65,
            190,
            105,
            260
        )

        self.bat_detector = BatDetector()
        self.bat_roi = (
            95,
            100,
            330,
            310
        )

        self.ball_detector = BallDetector()

        # Horizontal pixel columns to black out before
        # ball detection — covers the left and right
        # border strips that bleed into the HSV range
        self.mask_columns = [
            (0, 65),      # left border strip
            (590, 640),   # right border strip
        ]

    def crop(self, frame, roi):

        x1, y1, x2, y2 = roi

        return frame[y1:y2, x1:x2]

    def apply_border_mask(self, frame):

        masked = frame.copy()

        for x1, x2 in self.mask_columns:

            masked[:, x1:x2] = 0

        return masked

    def check_out(self, frame):

        return self.out_detector.detect(frame)

    def update(self, frame):

        wicket_view = self.crop(
            frame,
            self.wicket_roi
        )

        wicket_result = self.wicket_detector.detect(
            wicket_view
        )

        bat_view = self.crop(
            frame,
            self.bat_roi
        )

        bat_result = self.bat_detector.detect(bat_view)

        if bat_result["detected"]:

            roi_x1, roi_y1, _, _ = self.bat_roi

            if bat_result["glove_center"] is not None:
                gx, gy = bat_result["glove_center"]
                bat_result["glove_center"] = (
                    gx + roi_x1,
                    gy + roi_y1
                )

            if bat_result["tip"] is not None:
                tx, ty = bat_result["tip"]
                bat_result["tip"] = (
                    tx + roi_x1,
                    ty + roi_y1
                )

            if bat_result["handle"] is not None:
                hx, hy = bat_result["handle"]
                bat_result["handle"] = (
                    hx + roi_x1,
                    hy + roi_y1
                )

            bat_result["box"] = (
                bat_result["box"] + np.array([roi_x1, roi_y1])
            )

        # Mask border strips before searching for the ball
        ball_frame = self.apply_border_mask(frame)

        ball_result = self.ball_detector.detect(ball_frame)

        tracker_balls = self.tracker_manager.update(
            ball_result["balls"]
        )

        return {
            "wicket": wicket_result,
            "bat": bat_result,
            "balls": tracker_balls
        }
    
    def read_score(self, frame):
        score_roi = frame[315:340, 260:310]
        gray = cv.cvtColor(score_roi, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        text = pytesseract.image_to_string(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )

        try:
            return int(text.strip().split()[0])
        except: 
            return None
        
    def reset(self):
        self.tracker_manager.reset()