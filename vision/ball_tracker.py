import numpy as np

from vision.kalman import BallKalman


class BallTrack:

    def __init__(self, track_id, detection):

        self.id = track_id

        self.center = detection["center"]

        self.radius = detection["radius"]

        self.history = [detection["center"]]

        self.missed_frames = 0

        # Starts at 1 since we have one detection on creation
        self.confirmed_frames = 1

        self.kalman = BallKalman(detection["center"])

        # Fix: initialise statePost so get_velocity() is safe on frame 1
        x, y = detection["center"]
        self.kalman.kalman.statePost = np.array(
            [[x], [y], [0], [0]], np.float32
        )

        self.predicted_position = detection["center"]

    def update(self, detection):

        self.center = detection["center"]

        self.radius = detection["radius"]

        self.history.append(detection["center"])

        if len(self.history) > 8:
            self.history.pop(0)

        self.kalman.correct(detection["center"])

        self.missed_frames = 0

        self.confirmed_frames += 1

    def predict_position(self):

        prediction = self.kalman.predict()

        px = int(prediction[0])
        py = int(prediction[1])

        self.predicted_position = (px, py)

        return self.predicted_position