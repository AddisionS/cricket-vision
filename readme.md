# RL Cricket Agent

A vision-based reinforcement learning cricket agent trained using PPO and desktop automation.

The project combines computer vision, Kalman tracking, and a custom Gymnasium environment to teach an AI agent how to play a browser cricket game entirely from screen pixels and mouse movement.

The agent learns:

* ball tracking
* bat positioning
* shot timing
* wicket avoidance
* scoring strategy

No game API or memory access is used.

---

# Features

* Vision-only gameplay
* PPO training with Stable-Baselines3
* Real-time ball tracking using Kalman filters
* Multi-ball tracking support
* Boundary-aware reward system
* Dynamic action space with aggressive shot bias
* TensorBoard logging
* Automatic checkpoint saving
* Desktop automation using PyAutoGUI

---

# Project Structure

```text
.
├── cricket_env.py          # Main Gymnasium environment
├── train.py                # PPO training script
├── play.py                 # Run trained model
├── window_capture.py       # Fast screen capture using MSS
│
├── vision/
│   ├── vision_system.py
│   ├── ball_detector.py
│   ├── ball_tracker.py
│   ├── tracker_manager.py
│   ├── kalman.py
│   ├── bat_detector.py
│   ├── wicket_detector.py
│   └── out_screen_detector.py
│
├── checkpoints/
├── logs/
└── assets/
```

---

# Setup Notes

This project is heavily tied to my personal hardware setup.

The cricket game runs on a Samsung S9 FE tablet used as a second monitor, while the training code runs on the main machine. Screen coordinates, capture regions, and several detection thresholds are currently configured specifically for this setup.

Because of this:

* screen dimensions are not universal
* capture coordinates may need adjustment
* HSV thresholds may vary depending on display calibration
* mouse movement scaling may differ across systems

If you plan to run this project on another setup, expect to recalibrate the environment configuration.

---

# How It Works

## 1. Screen Capture

The game window is captured in real time using MSS.

```python
WindowCapture.get_screenshot()
```

The capture system is optimized for low-latency PPO training.

---

## 2. Vision Pipeline

The `VisionSystem` processes each frame and extracts:

* visible cricket balls
* ball trajectories
* bat position
* wicket state
* out-screen detection

### Ball Detection

HSV masking and contour filtering are used to isolate the ball.

The detector filters based on:

* size
* circularity
* contour area

to reject noise and false detections.

### Ball Tracking

Each detected ball is assigned a Kalman filter tracker.

Tracking supports:

* multiple simultaneous balls
* velocity estimation
* trajectory prediction
* frame-drop recovery

### Bat Detection

The bat is detected using HSV segmentation and contour analysis.

The system estimates:

* bat handle
* bat tip
* bat orientation
* glove position

---

# Reinforcement Learning

The project currently uses PPO from Stable-Baselines3.

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
)
```

The action space controls bat movement using:

* 8 directions
* 4 swing speeds

Aggressive shots are weighted more heavily during exploration.

At the moment, the agent behaves more like a traditional test batsman:

* defensive
* survival-focused
* patient

The current reward shaping favors staying alive and building innings consistency over aggressive scoring.

---

# Reward System

The reward system is boundary-based.

Rewards:

* 1 run  -> +2
* 2 runs -> +4
* 4 runs -> +8
* 6 runs -> +12

Wickets:

* Out -> -15

The agent learns to balance:

* survival
* timing
* shot direction
* scoring aggression

The current reward function is still experimental and will likely be redesigned in future training runs.

---

# Training

Run training:

```bash
python train.py
```

Training logs are written to:

```text
./logs/
```

Checkpoints are automatically saved every 10k timesteps.

---

# TensorBoard

Launch TensorBoard:

```bash
tensorboard --logdir ./logs
```

Open:

```text
http://localhost:6006
```

Useful metrics:

* episode reward
* episode length
* entropy
* KL divergence
* explained variance

---

# Running a Trained Model

```bash
python play.py
```

The model loads:

```text
cricket_ppo_1M
```

and plays deterministically.

---

# Current Results

After 1 million timesteps:

* stable batting behavior emerged
* the agent learned defensive survival strategies
* the model achieved half-century innings
* maximum observed score exceeded 70 runs
* episode length increased from ~200 to ~1000 timesteps

The current agent is consistent but not highly aggressive. It prioritizes wicket preservation over maintaining a high strike rate.

---

# Tech Stack

* Python
* OpenCV
* Stable-Baselines3
* Gymnasium
* NumPy
* SciPy
* MSS
* PyAutoGUI

---

# Future Improvements

Planned improvements include:

* redesigned reward shaping
* strike-rate optimized rewards
* aggressive T20-style batting behavior
* recurrent PPO (LSTM)
* improved temporal awareness
* trajectory-aware recovery behavior
* better power transfer mechanics
* multi-scale observations
* improved multi-ball awareness and tracking performance
* finding a more optimal ball count that the model can reliably handle

Future training runs will likely focus on creating a far more aggressive batter capable of maintaining a high scoring rate instead of simply surviving long innings.

There are also plans to eventually replace Stable-Baselines3 entirely and implement PPO from scratch for greater control over:

* rollout collection
* policy updates
* entropy scheduling
* advantage estimation
* custom architectures
* training diagnostics

---

# Notes

This project intentionally avoids direct game memory access or APIs.

The agent learns purely through:

* pixels
* rewards
* interaction

which makes the environment significantly harder than traditional RL benchmarks.

This project is also highly experimental and serves as both a reinforcement learning playground and a computer vision systems project.

---

# Disclaimer

PyAutoGUI controls the real mouse cursor.

Do not use the machine normally while training unless you enjoy fighting a cricket-playing neural network for control of your desktop.