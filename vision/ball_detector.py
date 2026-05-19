import cv2 as cv
import numpy as np


class BallDetector:

    def __init__(self):

        self.lower = np.array([
            0,
            35,
            71
        ])

        self.upper = np.array([
            20,
            160,
            230
        ])

        self.hand_lower = np.array([
            0,
            80,
            193
        ])

        self.hand_upper = np.array([
            20,
            160,
            242
        ])

    def detect(self, frame):

        hsv = cv.cvtColor(
            frame,
            cv.COLOR_BGR2HSV
        )

        ball_mask = cv.inRange(
            hsv,
            self.lower,
            self.upper
        )

        hand_mask = cv.inRange(
            hsv,
            self.hand_lower,
            self.hand_upper
        )

        mask = cv.subtract(
            ball_mask,
            hand_mask
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        mask = cv.morphologyEx(
            mask,
            cv.MORPH_OPEN,
            kernel
        )

        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        balls = []

        for contour in contours:

            area = cv.contourArea(
                contour
            )

            if area < 15:
                continue

            perimeter = cv.arcLength(
                contour,
                True
            )

            if perimeter == 0:
                continue

            circularity = (

                4 * np.pi * area

            ) / (

                perimeter * perimeter
            )

            if circularity < 0.65:
                continue

            (x, y), radius = cv.minEnclosingCircle(
                contour
            )

            x = int(x)
            y = int(y)

            radius = int(radius)

            if radius < 3:
                continue

            x1 = x - radius
            y1 = y - radius

            x2 = x + radius
            y2 = y + radius

            balls.append({

                "contour": contour,

                "center": (
                    x,
                    y
                ),

                "radius": radius,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "area": area,

                "circularity": circularity
            })

        return {

            "detected": len(balls) > 0,

            "balls": balls,

            "mask": mask
        }