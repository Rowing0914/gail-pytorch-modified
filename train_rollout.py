import numpy as np
import gym, argparse, os
from stable_baselines3 import SAC, PPO, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from imitation.src.imitation.algorithms.adversarial.gail import GAIL
from imitation.src.imitation.data import rollout
from imitation.src.imitation.data.wrappers import RolloutInfoWrapper
from imitation.src.imitation.rewards.reward_nets import BasicRewardNet
from imitation.src.imitation.util.networks import RunningNorm
from imitation.src.imitation.util.util import make_vec_env
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
parser.add_argument("--policy_name", default="sac")  # OpenAI gym environment name
parser.add_argument("--env_name", default="HalfCheetah-v3")  # OpenAI gym environment name
parser.add_argument("--seed", default=2023, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--num_envs", default=8, type=int)
parser.add_argument("--gen_train_timesteps", default=1000, type=int)
parser.add_argument("--max_episode_steps", default=1000, type=int)
parser.add_argument("--train_steps", default=10_000_000, type=int)
parser.add_argument("--num_eval_episodes", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--log_dir_expert", default="./logs/expert-sac-Humanoid-v3-seed1-12124112/")
# parser.add_argument("--log_dir_expert", default="./logs/expert-sac-HalfCheetah-v3-seed1-12124048/")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
venv = make_vec_env(args.env_name, n_envs=args.num_envs, rng=rng,)
venv_eval = make_vec_env(args.env_name, n_envs=args.num_envs, rng=rng,)
expert = SAC("MlpPolicy", venv, device=args.device, verbose=1)
# import pudb; pudb.start()
expert = expert.load(f"{args.log_dir_expert}/best_model/best_model.zip")

print("Test pretrained expert!")
from stable_baselines3.common.evaluation import evaluate_policy
rewards, _ = evaluate_policy(expert, venv_eval, 10, return_episode_rewards=True)
print("Pretrained expert: ", np.mean(rewards))

rollouts = rollout.rollout(
    expert,
    make_vec_env(args.env_name, n_envs=args.num_envs, post_wrappers=[lambda env, _: RolloutInfoWrapper(env)], rng=rng,),
    rollout.make_sample_until(min_timesteps=10000, min_episodes=60),
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
