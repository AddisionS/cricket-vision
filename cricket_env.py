import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pyautogui
import time
import cv2 as cv
from mss import mss
import pytesseract

from window_capture import WindowCapture
from vision.vision_system import VisionSystem


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 8 directions x 3 speeds = 24 actions
DIRECTIONS = [
    (-1,  0),   # left
    ( 1,  0),   # right
    ( 0, -1),   # up
    ( 0,  1),   # down
    (-1, -1),   # up-left
    ( 1, -1),   # up-right
    (-1,  1),   # down-left
    ( 1,  1),   # down-right
]
SPEEDS = [15, 30, 50]


def zone_score(x, y):
    if 0 <= x <= 510 and 0 <= y <= 10:     return 0   # out
    if 511 <= x <= 640 and 0 <= y <= 10:   return 6   # six
    if 0 <= x <= 12 and 10 <= y <= 58:     return 6   # six
    if 0 <= x <= 12 and 59 <= y <= 105:    return 4   # four
    if 0 <= x <= 12 and 106 <= y <= 280:   return 0   # out
    if 628 <= x <= 640 and 10 <= y <= 34:  return 6   # six
    if 628 <= x <= 640 and 35 <= y <= 153: return 4   # four
    if 628 <= x <= 640 and 154 <= y <= 224:return 2   # two
    if 628 <= x <= 640 and 225 <= y <= 280:return 1   # one
    return -1  # not in any zone


class CricketEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.vision = VisionSystem()
        self.capture = WindowCapture.__new__(WindowCapture)
        self.capture.sct = mss()
        self.capture.monitor = {
            "top": 65,
            "left": 2558,
            "width": 640,
            "height": 400
        }

        # 8 directions x 3 speeds
        self.action_space = spaces.Discrete(24)

        self.N = 3
        # N balls * 5 values (x, y, vx, vy, predicted_zone_score)
        # + bat 4 values (handle_x, handle_y, tip_x, tip_y)
        # + mouse position 2 values (mx, my) in game coords
        obs_size = (self.N * 5) + 4 + 2
        self.observation_space = spaces.Box(
            low=-640.0,
            high=3200.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.roi = (95, 100, 330, 310)

        # Absolute mouse offset — verified working
        self.mouse_offset_x = 3200
        self.mouse_offset_y = 80

        self.prev_score = 0
        self.current_score = 0
        self.step_count = 0
        self.episode_count = 0
        self.done = False

        self.last_frame = None
        self.last_vision = None
        self.last_action = 0

    def _get_state(self, vision_result):

        balls = vision_result["balls"]
        bat = vision_result["bat"]

        def tti(track):
            if bat["tip"] is None:
                return float("inf")
            bx, by = bat["tip"]
            cx, cy = track.center
            vx, vy = track.kalman.get_velocity()
            dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
            speed = np.sqrt(vx**2 + vy**2)
            return dist / speed if speed > 0 else float("inf")

        balls_sorted = sorted(balls, key=tti)[:self.N]

        state = []

        for track in balls_sorted:
            vx, vy = track.kalman.get_velocity()
            cx, cy = track.center

            # Predict where ball will be in 10 frames
            pred_x = cx + vx * 10
            pred_y = cy + vy * 10
            predicted_zone = zone_score(int(pred_x), int(pred_y))

            state.extend([
                cx,
                cy,
                vx,
                vy,
                float(predicted_zone)
            ])

        # Zero pad if fewer than N balls
        while len(state) < self.N * 5:
            state.extend([0.0, 0.0, 0.0, 0.0, -1.0])

        if bat["handle"] and bat["tip"]:
            state.extend([
                bat["handle"][0],
                bat["handle"][1],
                bat["tip"][0],
                bat["tip"][1]
            ])
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])

        # Mouse position in game coords — agent knows where it is
        mx_abs, my_abs = pyautogui.position()
        mx_game = float(np.clip(mx_abs - self.mouse_offset_x, 0, 640))
        my_game = float(np.clip(my_abs - self.mouse_offset_y, 0, 400))
        state.extend([mx_game, my_game])

        return np.array(state, dtype=np.float32)

    def _action_to_mouse(self, action):

        dir_idx = action % 8
        speed_idx = action // 8

        dx, dy = DIRECTIONS[dir_idx]
        speed = SPEEDS[speed_idx]

        # Move relative to current mouse position
        cx, cy = pyautogui.position()

        new_x = cx + dx * speed
        new_y = cy + dy * speed

        # Clamp to bat ROI in absolute screen coords
        x1, y1, x2, y2 = self.roi
        new_x = max(x1 + self.mouse_offset_x, min(x2 + self.mouse_offset_x, new_x))
        new_y = max(y1 + self.mouse_offset_y, min(y2 + self.mouse_offset_y, new_y))

        return int(new_x), int(new_y)

    def _calculate_reward(self, vision_result, prev_score, curr_score):

        reward = 0.0

        # Main signal — actual runs scored this step
        runs_scored = curr_score - prev_score
        if runs_scored > 0:
            reward += runs_scored * 2.0

        # Contact bonus — encourage making contact with ball
        balls = vision_result["balls"]
        bat = vision_result["bat"]
        if bat["tip"] is not None:
            for track in balls:
                bx, by = bat["tip"]
                cx, cy = track.center
                dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
                if dist < 15:
                    reward += 0.3

        # Small survival reward
        reward += 0.05

        # Death penalty
        if self.done:
            reward -= 10.0

        return reward

    def step(self, action):

        self.last_action = action
        self.step_count += 1

        mx, my = self._action_to_mouse(action)

        # Smooth move so game registers bat velocity
        pyautogui.moveTo(mx, my, duration=0.03)

        frame = self.capture.get_screenshot()
        vision_result = self.vision.update(frame)

        self.last_frame = frame
        self.last_vision = vision_result

        self.done = False

        wicket_edges = vision_result["wicket"]["edge_count"]
        bat = vision_result["bat"]
        balls = vision_result["balls"]

        wicket_hit = wicket_edges < 30

        heading_to_zero = False
        if bat["tip"] is not None:
            for track in balls:
                bx, by = bat["tip"]
                cx, cy = track.center
                dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
                if dist < 15:
                    vx, vy = track.kalman.get_velocity()
                    if vy < -8:
                        heading_to_zero = True
                    if cx < 12 and 106 < cy < 280:
                        heading_to_zero = True

        if wicket_hit or heading_to_zero or self.step_count % 25 == 0:
            out_result = self.vision.check_out(frame)
            self.done = out_result["detected"]

        # Only run OCR every 10 steps — expensive at ~300ms
        if self.step_count % 10 == 0:
            score = self.vision.read_score(frame)
            if score is not None:
                self.current_score = score

        curr_score = self.current_score

        reward = self._calculate_reward(
            vision_result,
            self.prev_score,
            curr_score
        )

        self.prev_score = curr_score

        if self.done:
            pyautogui.click()
            time.sleep(0.5)


        state = self._get_state(vision_result)

        return state, reward, self.done, False, {}

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.episode_count += 1

        # Refresh browser every 30 episodes to prevent game lag
        if self.episode_count % 30 == 0:
            pyautogui.click(self.mouse_offset_x - 200, self.mouse_offset_y - 10)
            time.sleep(0.5)
            pyautogui.hotkey('ctrl', 'r')
            time.sleep(4)

        pyautogui.click(self.mouse_offset_x + 320, self.mouse_offset_y + 250)
        time.sleep(1.0)
        self.vision.reset()
        self.prev_score = 0
        self.current_score = 0
        self.step_count = 0
        self.done = False

        time.sleep(0.5)

        frame = self.capture.get_screenshot()
        vision_result = self.vision.update(frame)

        self.last_frame = frame
        self.last_vision = vision_result

        state = self._get_state(vision_result)

        return state, {}

    def render(self):
        pass