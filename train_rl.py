import numpy as np
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb

if_wandb = True
if_video = True
num_envs = 8
env_name = "HalfCheetah-v3"
device = "cuda"
log_root = "logs"
seed = 2023
max_episode_steps = 1000
train_steps = 100_000

import os
from datetime import datetime

t = datetime.now()
log_dir_expert = os.path.join(log_root, f"expert-{env_name}-{t.month}{t.day}{t.hour}{t.minute}{t.second}", str(seed))
os.makedirs(log_dir_expert, exist_ok=True)

if if_wandb:
    wandb.login()
    run = wandb.init(
        project="img-gen-rl", entity="rowing0914", name="motion-track", group="motion-track", dir="/tmp/wandb",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        )
    # wandb.config.update(config)

venv = make_vec_env(env_name, n_envs=num_envs)
eval_env = make_vec_env(env_name, n_envs=num_envs)

if if_video and if_wandb:
    # video_length = venv.envs[0].unwrapped.spec.max_episode_steps
    video_length = max_episode_steps
    eval_env = VecVideoRecorder(
        venv=eval_env,
        # video_folder=f"{log_dir_expert}/videos",
        video_folder=f"/tmp/-videos",
        name_prefix=f"SAC-expert-{env_name}",
        record_video_trigger=lambda x: x % 10000 == 0,
        video_length=video_length,
    )

expert = SAC("MlpPolicy", venv, device=device, verbose=1, tensorboard_log=log_dir_expert)
eval_callback = EvalCallback(eval_env, eval_freq=1000 * 2, best_model_save_path=f"{log_dir_expert}/best_model", log_path=log_dir_expert)
callbacks = [eval_callback]
if if_wandb:
    callbacks.append(WandbCallback(verbose=2))
expert.learn(train_steps, callback=CallbackList(callbacks))

if if_wandb:
    run.finish()
