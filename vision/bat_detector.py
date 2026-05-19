import cv2 as cv
import numpy as np


class BatDetector:

    def __init__(self):
        self.lower = np.array([
            7,
            95,
            215
        ])

        self.upper = np.array([
            41,
            134,
            255
        ])

        self.glove_lower = np.array([
            30,
            30,
            128
        ])

        self.glove_upper = np.array([
            105,
            255,
            235
        ])

    def detect(self, bat_roi):
        hsv = cv.cvtColor(
            bat_roi,
            cv.COLOR_BGR2HSV
        )

        bat_mask = cv.inRange(
            hsv,
            self.lower,
            self.upper
        )

        glove_mask = cv.inRange(
            hsv,
            self.glove_lower,
            self.glove_upper
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        bat_mask = cv.morphologyEx(
            bat_mask,
            cv.MORPH_OPEN,
            kernel
        )

        glove_mask = cv.morphologyEx(
            glove_mask,
            cv.MORPH_OPEN,
            kernel
        )

        contours, _ = cv.findContours(
            bat_mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:

            return {
                "detected": False,
                "rect": None,
                "box": None,
                "mask": bat_mask,

                "glove_center": None,

                "handle": None,
                "tip": None
            }

        largest = max(
            contours,
            key=cv.contourArea
        )

        area = cv.contourArea(
            largest
        )

        if area < 20:

            return {
                "detected": False,
                "rect": None,
                "box": None,
                "mask": bat_mask,

                "glove_center": None,

                "handle": None,
                "tip": None
            }

        rect = cv.minAreaRect(
            largest
        )

        box = cv.boxPoints(
            rect
        )

        box = np.int32(box)

        p0 = box[0]
        p1 = box[1]
        p2 = box[2]
        p3 = box[3]

        sides = [

            (
                np.linalg.norm(p1 - p0),
                (p0, p1)
            ),

            (
                np.linalg.norm(p2 - p1),
                (p1, p2)
            ),

            (
                np.linalg.norm(p3 - p2),
                (p2, p3)
            ),

            (
                np.linalg.norm(p0 - p3),
                (p3, p0)
            )
        ]

        longest_side = max(
            sides,
            key=lambda x: x[0]
        )

        (_, (a, b)) = longest_side

        glove_contours, _ = cv.findContours(
            glove_mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        glove_center = None

        if len(glove_contours) > 0:

            glove = max(
                glove_contours,
                key=cv.contourArea
            )

            glove_area = cv.contourArea(
                glove
            )

            if glove_area > 5:

                M = cv.moments(
                    glove
                )

                if M["m00"] != 0:

                    cx = int(
                        M["m10"] / M["m00"]
                    )

                    cy = int(
                        M["m01"] / M["m00"]
                    )

                    glove_center = (
                        cx,
                        cy
                    )

        handle = None
        tip = None

        if glove_center is not None:

            dist_a = np.linalg.norm(
                np.array(glove_center) - a
            )

            dist_b = np.linalg.norm(
                np.array(glove_center) - b
            )

            # closer endpoint = handle

            if dist_a < dist_b:

                handle = a
                tip = b

            else:

                handle = b
                tip = a

        extended_box = box.copy().astype(
            np.float32
        )

        if handle is not None and tip is not None:

            extension = 12

            vec = tip - handle

            vec_length = np.linalg.norm(
                vec
            )

            if vec_length != 0:

                vec = vec / vec_length

            for i in range(4):

                point = box[i]

                dist_to_tip = np.linalg.norm(
                    point - tip
                )

                dist_to_handle = np.linalg.norm(
                    point - handle
                )
                if dist_to_tip < dist_to_handle:

                    extended_box[i] = (

                        point + vec * extension
                    )

        extended_box = np.int32(
            extended_box
        )

        return {
            "detected": True,
            "rect": rect,
            "box": extended_box,
            "mask": bat_mask,
            "glove_center": glove_center,
            "handle": handle,
            "tip": tip
        }