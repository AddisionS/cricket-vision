import cv2 as cv
from time import time
import numpy as np

from window_capture import WindowCapture
from vision.vision_system import VisionSystem


wincap = WindowCapture(
    monitor_number=2,
    top=185,
    left=638,
    width=640,
    height=400
)

vision_system = VisionSystem()

loop_time = time()

frame_count = 0


while True:

    screenshot = wincap.get_screenshot()

    output = vision_system.update(
        screenshot
    )

    # =========================
    # BAT VISUALIZATION
    # =========================

    bat = output["bat"]

    if bat["detected"]:

        box = bat["box"]

        cv.drawContours(
            screenshot,
            [box],
            0,
            (255, 0, 0),
            2
        )

        # HANDLE

        if bat["handle"] is not None:

            hx, hy = bat["handle"]

            cv.circle(
                screenshot,
                (hx, hy),
                5,
                (0, 255, 255),
                -1
            )

        # TIP

        if bat["tip"] is not None:

            tx, ty = bat["tip"]

            cv.circle(
                screenshot,
                (tx, ty),
                5,
                (0, 0, 255),
                -1
            )

    # =========================
    # BALL TRACKING VISUALIZATION
    # =========================

    balls = output["balls"]

    for ball in balls:

        # CURRENT TRACKED POSITION

        x, y = ball.center

        cv.circle(
            screenshot,
            (x, y),
            ball.radius,
            (0, 255, 0),
            2
        )

        # TRACK ID

        cv.putText(
            screenshot,
            f"ID {ball.id}",
            (x, y - 12),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # PREDICTED POSITION

        px, py = ball.predicted_position

        cv.circle(
            screenshot,
            (px, py),
            3,
            (0, 0, 255),
            -1
        )

        # HISTORY TRAIL

        for point in ball.history:

            hx, hy = point

            cv.circle(
                screenshot,
                (hx, hy),
                1,
                (255, 255, 255),
                -1
            )

        # VELOCITY VECTOR

        vx, vy = ball.kalman.get_velocity()

        vx = np.clip(vx, -40, 40)

        vy = np.clip(vy, -40, 40)

        end_x = int(x + vx * 2)

        end_y = int(y + vy * 2)

        cv.line(
            screenshot,
            (x, y),
            (end_x, end_y),
            (255, 0, 255),
            2
        )
    # =========================
    # FPS DISPLAY
    # =========================

    frame_count += 1

    now = time()

    elapsed = now - loop_time

    if elapsed >= 1:

        fps = frame_count / elapsed

        cv.putText(
            screenshot,
            f"FPS: {fps:.2f}",
            (10, 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        print(
            f"FPS: {fps:.2f} | "
            f"Edges: "
            f'{output["wicket"]["edge_count"]}'
        )

        frame_count = 0

        loop_time = now

    # =========================
    # DISPLAY
    # =========================

    cv.imshow(
        "Computer Vision",
        screenshot
    )

    if cv.waitKey(1) == ord('q'):

        cv.destroyAllWindows()

        break