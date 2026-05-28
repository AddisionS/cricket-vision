from stable_baselines3 import PPO
from cricket_env import CricketEnv
from stable_baselines3.common.callbacks import CheckpointCallback

env = CricketEnv()

checkpoint = CheckpointCallback(
    save_freq=10000,
    save_path='./checkpoints/',
    name_prefix='cricket_ppo'
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=1_000_000, callback=checkpoint)
model.save("cricket_ppo_1M")