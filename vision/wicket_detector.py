import cv2 as cv
import numpy as np


class WicketDetector:

    def __init__(self, threshold=120):

        # more edges = wicket exists
        self.threshold = threshold

    def detect(self, wicket_roi):

        gray = cv.cvtColor(
            wicket_roi,
            cv.COLOR_RGB2GRAY
        )

        edges = cv.Canny(
            gray,
            100,
            200
        )

        edge_count = np.count_nonzero(edges)

        detected = edge_count > self.threshold

        confidence = min(
            1.0,
            edge_count / (self.threshold * 2)
        )

        return {
            "detected": detected,
            "confidence": confidence,
            "edge_count": edge_count
        }