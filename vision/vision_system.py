import os
import numpy as np
from vision.wicket_detector import WicketDetector
from vision.out_screen_detector import OutDetector
from vision.bat_detector import BatDetector
from vision.ball_detector import BallDetector


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

    def crop(self, frame, roi):

        x1, y1, x2, y2 = roi

        return frame[y1:y2, x1:x2]

    def check_out(self, frame):

        return self.out_detector.detect(
            frame
        )

    def update(self, frame):

        wicket_view = self.crop(
            frame,
            self.wicket_roi
        )

        wicket_result = (
            self.wicket_detector.detect(
                wicket_view
            )
        )

        bat_view = self.crop(
            frame,
            self.bat_roi
        )

        bat_result = self.bat_detector.detect(
            bat_view
        )   

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

            bat_result["box"] = ( bat_result["box"] + np.array([roi_x1, roi_y1]) )

        ball_result = self.ball_detector.detect(
            frame
        )

        return {
            "wicket": wicket_result,
            "bat": bat_result,
            "balls": ball_result
        }