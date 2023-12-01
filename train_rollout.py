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

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import wandb


if_wandb = True
if_video = True
num_envs = 8
env_name = "HalfCheetah-v3"
device = "cuda"
log_root = "logs"
seed = 2023
max_episode_steps = 1000
log_dir_expert = "./logs/expert-HalfCheetah-v3-1210841/2023"

rng = np.random.default_rng(seed)
venv = make_vec_env(env_name, n_envs=num_envs, rng=rng,)
expert = SAC("MlpPolicy", venv, device=device, verbose=1)
expert.load(f"{log_dir_expert}/best_model/best_model.zip")

rollouts = rollout.rollout(
    expert,
    make_vec_env(env_name, n_envs=num_envs, post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], rng=rng,),
    rollout.make_sample_until(min_timesteps=1000, min_episodes=60),
    rng=rng,
)
# import pudb; pudb.start()
rews = list()
ts_cnt = 0
for episode in rollouts:
    ts_cnt += episode.obs.shape[0]
    rews.append(episode.rews.sum())
print(f"Collected: {ts_cnt} transitions, Avg. Ep-Return: {np.mean(rews)}")

import pickle
with open(f"{log_dir_expert}/rollouts.pkl", "wb") as handle:
    pickle.dump(rollouts, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"{log_dir_expert}/rollouts_result.txt", "w") as handle:
    handle.write(f"Collected: {ts_cnt} transitions, Avg. Ep-Return: {np.mean(rews)}")
