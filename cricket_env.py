import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pyautogui
import time
from mss import mss

from window_capture import WindowCapture
from vision.vision_system import VisionSystem


pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ---------------------------------------------------------------------------
# Action space — 8 directions x 4 speeds
# Speed 60 and 80 are "big swing" speeds, weighted heavier in sampling
# so the model is naturally nudged toward aggressive shots.
# The action indices 16-31 (speeds 60 and 80) cover the big swings.
# ---------------------------------------------------------------------------
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
SPEEDS = [15, 30, 60, 80]   # two slow, two aggressive

# Pre-built action probability weights — big swings (speed idx 2,3) get
# 3x the sampling weight of defensive moves (speed idx 0,1).
# PPO will override this via its own policy but this shapes early exploration.
_N_DIRS = len(DIRECTIONS)
_N_SPEEDS = len(SPEEDS)
_N_ACTIONS = _N_DIRS * _N_SPEEDS   # 32

_ACTION_WEIGHTS = np.array([
    1.0 if (a // _N_DIRS) < 2 else 3.0
    for a in range(_N_ACTIONS)
], dtype=np.float32)
_ACTION_WEIGHTS /= _ACTION_WEIGHTS.sum()

# ---------------------------------------------------------------------------
# Boundary zones — (x1, y1, x2, y2) -> score value
# score == 0  →  out
# score  > 0  →  runs scored
# ---------------------------------------------------------------------------
BOUNDARY_ZONES = [
    # top strip
    ((  0,  0, 510, 10), 0),   # out
    ((511,  0, 640, 10), 6),   # six

    # left strip
    ((0, 10,  12,  58), 6),    # six
    ((0, 59,  12, 105), 4),    # four
    ((0, 106, 12, 280), 0),    # out

    # right strip
    ((628, 10,  640,  34), 6), # six
    ((628, 35,  640, 153), 4), # four
    ((628, 154, 640, 224), 2), # two
    ((628, 225, 640, 280), 1), # one
]

# How much to expand each ROI inward so we catch the ball
# slightly before it fully exits frame
_ZONE_INSET = 4   # pixels


def _point_in_rect(px, py, x1, y1, x2, y2, inset=0):
    return (x1 - inset) <= px <= (x2 + inset) and \
           (y1 - inset) <= py <= (y2 + inset)


def zone_score(x, y):
    """
    Returns the score value for a ball at (x, y), or -1 if not in any zone.
    """
    for (x1, y1, x2, y2), score in BOUNDARY_ZONES:
        if _point_in_rect(x, y, x1, y1, x2, y2, inset=_ZONE_INSET):
            return score
    return -1   # not in any zone


def ball_exiting(x, y, vx, vy):
    """
    Returns True only if the ball is moving AWAY from the centre of the
    frame — i.e. it is exiting through a boundary, not entering.

    Left boundaries  (x ~ 0)   : exiting when vx < 0
    Right boundaries (x ~ 640) : exiting when vx > 0
    Top boundary     (y ~ 0)   : exiting when vy < 0
    """
    if x < 20:    return vx < 0      # moving left  → exiting left
    if x > 620:   return vx > 0      # moving right → exiting right
    if y < 15:    return vy < 0      # moving up    → exiting top
    return False


# Reward magnitudes — kept proportional so 6 > 4 > 2 > 1 > 0
SCORE_REWARD = {
    6: 12.0,
    4:  8.0,
    2:  4.0,
    1:  2.0,
    0:  0.0,   # out — handled separately as death penalty
}
DEATH_PENALTY = -15.0

# Observation normalisation constants
_MAX_POS   = 640.0
_MAX_VEL   = 40.0    # pixels/frame — Kalman velocity rarely exceeds this
_MAX_ZONE  = 6.0
_MAX_MOUSE = 640.0


class CricketEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.vision  = VisionSystem()
        self.capture = WindowCapture.__new__(WindowCapture)
        self.capture.sct = mss()
        self.capture.monitor = {
            "top":    65,
            "left":   2558,
            "width":  640,
            "height": 400,
        }

        self.action_space = spaces.Discrete(_N_ACTIONS)

        self.N = 3
        # Per ball: x, y, vx, vy, predicted_zone   (5 values)
        # Bat:      handle_x, handle_y, tip_x, tip_y (4 values)
        # Mouse:    mx, my                            (2 values)
        # All normalised to roughly [0, 1] or [-1, 1]
        obs_size = (self.N * 5) + 4 + 2
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.roi = (95, 100, 330, 310)   # bat movement clamp region

        # Absolute screen offset for mouse → game coords
        self.mouse_offset_x = 3200
        self.mouse_offset_y = 80

        self.step_count   = 0
        self.episode_count = 0
        self.done = False

        self.last_frame  = None
        self.last_vision = None
        self.last_action = 0

        # Cooldown tracker — prevents same ball firing reward multiple frames
        # key: track_id, value: steps remaining in cooldown
        self.seen_tracks = {}
        self._REWARD_COOLDOWN = 8

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _norm_pos(self, v):
        return float(np.clip(v / _MAX_POS, -1.0, 1.0))

    def _norm_vel(self, v):
        return float(np.clip(v / _MAX_VEL, -1.0, 1.0))

    def _norm_zone(self, v):
        # zone is -1 (none) … 6 (six) → map to [-1, 1]
        return float(np.clip(v / _MAX_ZONE, -1.0, 1.0))

    def _norm_mouse(self, v):
        return float(np.clip(v / _MAX_MOUSE, 0.0, 1.0))

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_state(self, vision_result):
        balls = vision_result["balls"]
        bat   = vision_result["bat"]

        # Sort by time-to-impact so the most dangerous ball is first
        def tti(track):
            if bat["tip"] is None:
                return float("inf")
            bx, by   = bat["tip"]
            cx, cy   = track.center
            vx, vy   = track.kalman.get_velocity()
            dist     = np.sqrt((cx - bx)**2 + (cy - by)**2)
            speed    = np.sqrt(vx**2 + vy**2)
            return dist / speed if speed > 0 else float("inf")

        balls_sorted = sorted(balls, key=tti)[:self.N]

        state = []

        for track in balls_sorted:
            vx, vy = track.kalman.get_velocity()
            cx, cy = track.center

            # Predict 10 frames ahead to give the model a heads-up on
            # which zone the ball is heading toward
            pred_x = cx + vx * 10
            pred_y = cy + vy * 10
            pzone  = zone_score(int(pred_x), int(pred_y))

            state.extend([
                self._norm_pos(cx),
                self._norm_pos(cy),
                self._norm_vel(vx),
                self._norm_vel(vy),
                self._norm_zone(pzone),
            ])

        # Zero-pad if fewer than N balls visible
        while len(state) < self.N * 5:
            state.extend([0.0, 0.0, 0.0, 0.0, self._norm_zone(-1)])

        if bat["handle"] and bat["tip"]:
            state.extend([
                self._norm_pos(bat["handle"][0]),
                self._norm_pos(bat["handle"][1]),
                self._norm_pos(bat["tip"][0]),
                self._norm_pos(bat["tip"][1]),
            ])
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])

        mx_abs, my_abs = pyautogui.position()
        mx_game = float(np.clip(mx_abs - self.mouse_offset_x, 0, 640))
        my_game = float(np.clip(my_abs - self.mouse_offset_y, 0, 400))
        state.extend([
            self._norm_mouse(mx_game),
            self._norm_mouse(my_game),
        ])

        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _action_to_mouse(self, action):
        dir_idx   = action % _N_DIRS
        speed_idx = action // _N_DIRS

        dx, dy = DIRECTIONS[dir_idx]
        speed  = SPEEDS[speed_idx]

        cx, cy = pyautogui.position()

        new_x = cx + dx * speed
        new_y = cy + dy * speed

        x1, y1, x2, y2 = self.roi
        new_x = int(np.clip(new_x, x1 + self.mouse_offset_x, x2 + self.mouse_offset_x))
        new_y = int(np.clip(new_y, y1 + self.mouse_offset_y, y2 + self.mouse_offset_y))

        return new_x, new_y

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _calculate_reward(self, vision_result):
        """
        Pure boundary-based reward — no contact bonus, no survival reward.

        Logic:
          1. For every tracked ball check if it is inside a boundary zone
             AND moving away from centre (exiting, not entering).
          2. If zone score > 0  → positive reward proportional to runs.
          3. If zone score == 0 → the ball hit a zero boundary; done=True
             and death penalty is applied.
          4. Wicket hit → done=True + death penalty (handled in step()).
        """
        reward = 0.0
        balls  = vision_result["balls"]

        # Tick down all cooldowns each step
        self.seen_tracks = {k: v - 1 for k, v in self.seen_tracks.items() if v > 1}

        for track in balls:
            cx, cy = track.center
            vx, vy = track.kalman.get_velocity()

            score = zone_score(int(cx), int(cy))

            if score == -1:
                continue   # not near any boundary

            if not ball_exiting(cx, cy, vx, vy):
                continue   # ball is entering frame, not exiting — ignore

            if track.id in self.seen_tracks:
                continue   # already rewarded this ball recently

            # Mark this track as seen and start cooldown
            self.seen_tracks[track.id] = self._REWARD_COOLDOWN

            if score == 0:
                # Zero boundary — out
                self.done = True
                reward += DEATH_PENALTY
            else:
                reward += SCORE_REWARD[score]

        # Death penalty also applied when done is set by wicket check in step()
        if self.done and reward == 0.0:
            reward += DEATH_PENALTY

        return reward

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def step(self, action):
        self.last_action  = action
        self.step_count  += 1

        mx, my = self._action_to_mouse(action)

        # Smooth move so the game engine registers bat velocity
        pyautogui.moveTo(mx, my, duration=0.03)

        frame         = self.capture.get_screenshot()
        vision_result = self.vision.update(frame)

        self.last_frame  = frame
        self.last_vision = vision_result

        self.done = False

        wicket_edges = vision_result["wicket"]["edge_count"]
        balls        = vision_result["balls"]

        # ----------------------------------------------------------
        # Out detection
        # Condition 1: wicket edge count drops below threshold
        # Condition 2: ball is detected inside a zero boundary zone
        # Both use the template-match out_checker as confirmation
        # (cheap because we only call it when suspicious)
        # ----------------------------------------------------------
        ball_in_zero_zone = any(
            zone_score(int(t.center[0]), int(t.center[1])) == 0
            and ball_exiting(t.center[0], t.center[1],
                             *t.kalman.get_velocity())
            for t in balls
        )

        wicket_hit = wicket_edges < 30

        if wicket_hit or ball_in_zero_zone or self.step_count % 25 == 0:
            out_result = self.vision.check_out(frame)
            if out_result["detected"]:
                self.done = True

        reward = self._calculate_reward(vision_result)

        if self.done:
            pyautogui.click()
            time.sleep(0.5)

        state = self._get_state(vision_result)

        return state, reward, self.done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_count += 1

        # Refresh browser every 30 episodes to prevent game drift/lag
        if self.episode_count % 30 == 0:
            pyautogui.click(self.mouse_offset_x - 200, self.mouse_offset_y - 10)
            time.sleep(0.5)
            pyautogui.hotkey('ctrl', 'r')
            time.sleep(4)

        pyautogui.click(self.mouse_offset_x + 320, self.mouse_offset_y + 250)
        time.sleep(1.0)
        self.vision.reset()
        self.step_count  = 0
        self.done        = False
        self.seen_tracks = {}

        time.sleep(0.5)

        frame         = self.capture.get_screenshot()
        vision_result = self.vision.update(frame)

        self.last_frame  = frame
        self.last_vision = vision_result

        state = self._get_state(vision_result)

        return state, {}

    def render(self):
        pass

    def sample_action(self):
        """
        Weighted random action for exploration scripts / curriculum warm-up.
        Big swing actions (speed 60, 80) are 3x more likely than defensive ones.
        PPO's own policy takes over during training — this is just for manual testing.
        """
        return int(np.random.choice(_N_ACTIONS, p=_ACTION_WEIGHTS))