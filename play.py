from stable_baselines3 import PPO
from cricket_env import CricketEnv

env = CricketEnv()
model = PPO.load("cricket_ppo_1M", env=env)

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()