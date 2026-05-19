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


        if bat["handle"] is not None:

            hx, hy = bat["handle"]



            cv.circle(
                screenshot,
                (hx, hy),
                5,
                (0, 255, 255),
                -1
            )

        if bat["tip"] is not None:

            tx, ty = bat["tip"]


            cv.circle(
                screenshot,
                (tx, ty),
                5,
                (0, 0, 255),
                -1
            )

    balls = output["balls"]

    if balls["detected"]:

        for ball in balls["balls"]:

            x1, y1, x2, y2 = (
                ball["box"]
            )

            center_x, center_y = (
                ball["center"]
            )

            radius = ball["radius"]

            cv.rectangle(
                screenshot,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv.circle(
                screenshot,
                (center_x, center_y),
                3,
                (0, 255, 0),
                -1
            )

    cv.imshow(
        'Computer Vision',
        screenshot
    )

    frame_count += 1

    now = time()

    elapsed = now - loop_time

    if frame_count % 30 == 0:

        fps = 30 / elapsed

        loop_time = now

        print(
            f'FPS: {fps:.2f} | '
            f'Balls: {len(balls["balls"])} | '
            f'Edges: '
            f'{output["wicket"]["edge_count"]}'
        )
    if cv.waitKey(1) == ord('q'):

        cv.destroyAllWindows()

        break
