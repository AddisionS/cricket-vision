import cv2 as cv
import numpy as np


class BallKalman:

    def __init__(self, initial_position):

        x, y = initial_position

        # 4 state values: x, y, vx, vy
        # 2 measurement values: x, y

        self.kalman = cv.KalmanFilter(4, 2)

        # State transition: x += vx, y += vy

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # We only observe x and y

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # Higher process noise = filter reacts faster
        # to sudden direction/speed changes (e.g. bounce)
        self.kalman.processNoiseCov = (
            np.eye(4, dtype=np.float32) * 0.5
        )

        # Moderate trust in detections
        self.kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 1.0
        )

        # Initialise both state matrices so
        # get_velocity() is safe before first correct()

        self.kalman.statePre = np.array(
            [[x], [y], [0], [0]], np.float32
        )

        self.kalman.statePost = np.array(
            [[x], [y], [0], [0]], np.float32
        )

    def predict(self):

        prediction = self.kalman.predict()

        px = int(prediction[0][0])
        py = int(prediction[1][0])

        return (px, py)

    def correct(self, measured_position):

        mx, my = measured_position

        measurement = np.array([
            [np.float32(mx)],
            [np.float32(my)]
        ])

        self.kalman.correct(measurement)

    def get_velocity(self):

        vx = self.kalman.statePost[2][0]
        vy = self.kalman.statePost[3][0]

        return (float(vx), float(vy))