import cv2 as cv


class OutDetector:

    def __init__(
        self,
        template_path,
        threshold=0.8
    ):

        self.template = cv.imread(
            template_path,
            cv.IMREAD_COLOR
        )

        if self.template is None:

            raise ValueError(
                f'Failed to load template: {template_path}'
            )

        self.threshold = threshold

    def detect(self, frame):

        result = cv.matchTemplate(
            frame,
            self.template,
            cv.TM_CCOEFF_NORMED
        )

        _, max_val, _, max_loc = cv.minMaxLoc(
            result
        )

        detected = max_val >= self.threshold

        return {
            "detected": detected,
            "confidence": max_val,
            "location": max_loc
        }