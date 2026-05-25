import numpy as np

from scipy.optimize import linear_sum_assignment

from vision.ball_tracker import BallTrack


class TrackerManager:

    def __init__(self):

        self.tracks = []

        self.next_id = 0

        # Base search radius — scales up with ball velocity
        self.base_max_distance = 40

        self.max_missed_frames = 3

        self.min_confirm_frames = 2

    def _search_radius(self, track):

        vx, vy = track.kalman.get_velocity()

        speed = np.sqrt(vx ** 2 + vy ** 2)

        # Allow up to 3 frames worth of travel
        # so frame drops don't break the match
        return self.base_max_distance + speed * 3

    def update(self, detections):

        for track in self.tracks:
            track.predict_position()

        matched_track_ids = set()
        matched_detection_indices = set()

        if self.tracks and detections:

            cost = np.array([
                [
                    np.linalg.norm(
                        np.array(track.predicted_position)
                        -
                        np.array(detection["center"])
                    )
                    for detection in detections
                ]
                for track in self.tracks
            ])

            row_indices, col_indices = linear_sum_assignment(cost)

            for row, col in zip(row_indices, col_indices):

                track = self.tracks[row]

                # Each track gets its own dynamic search radius
                radius = self._search_radius(track)

                if cost[row, col] < radius:

                    track.update(detections[col])

                    matched_track_ids.add(track.id)

                    matched_detection_indices.add(col)

        for i, detection in enumerate(detections):

            if i not in matched_detection_indices:

                new_track = BallTrack(self.next_id, detection)

                self.tracks.append(new_track)

                matched_track_ids.add(new_track.id)

                self.next_id += 1

        alive_tracks = []

        for track in self.tracks:

            if track.id not in matched_track_ids:
                track.missed_frames += 1

            if track.missed_frames <= self.max_missed_frames:
                alive_tracks.append(track)

        self.tracks = alive_tracks

        confirmed = [
            t for t in self.tracks
            if t.confirmed_frames >= self.min_confirm_frames
        ]

        return confirmed
    
    def reset(self):
        self.tracks = []
        self.next_id = 0