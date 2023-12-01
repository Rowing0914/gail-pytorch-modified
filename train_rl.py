import numpy as np
import gym
import os, argparse
from datetime import datetime
from stable_baselines3 import SAC, PPO, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import wandb

parser = argparse.ArgumentParser()
# misc
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wb_project", type=str, default="img-gen-rl")
parser.add_argument("--wb_entity", type=str, default="rowing0914")
parser.add_argument("--wb_group", type=str, default="vanilla")
parser.add_argument("--device", default="cuda")
parser.add_argument("--log_root", default="logs")

# training setup
parser.add_argument("--policy_name", default="sac")  # OpenAI gym environment name
parser.add_argument("--env_name", default="HalfCheetah-v3")  # OpenAI gym environment name
parser.add_argument("--seed", default=2023, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--num_envs", default=8, type=int)
parser.add_argument("--max_episode_steps", default=1000, type=int)
parser.add_argument("--train_steps", default=10_000_000, type=int)
parser.add_argument("--num_eval_episodes", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
parser.add_argument("--if_save_weight", action="store_true", default=False)

# video related
parser.add_argument("--if_video", action="store_true", default=False)
parser.add_argument("--video_freq", default=10000, type=int)  # How often (time steps) we evaluate
args = parser.parse_args()

t = datetime.now()
run_name = f"expert-{args.policy_name}-{args.env_name}-seed{args.seed}-{t.month}{t.day}{t.hour}{t.minute}{t.second}"
log_dir_expert = os.path.join(args.log_root, run_name)
os.makedirs(log_dir_expert, exist_ok=True)

if args.wandb:
    wandb.login()
    run = wandb.init(
        project=args.wb_project, entity=args.wb_entity, name=run_name, group=args.wb_group, dir="/tmp/wandb",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        )
    wandb.config.update(args)

venv = make_vec_env(args.env_name, n_envs=args.num_envs)
eval_env = make_vec_env(args.env_name, n_envs=args.num_envs)

if args.if_video:
    # video_length = venv.envs[0].unwrapped.spec.max_episode_steps
    video_length = args.max_episode_steps
    eval_env = VecVideoRecorder(
        venv=eval_env,
        # video_folder=f"{log_dir_expert}/videos",
        video_folder=f"/tmp/-videos",
        name_prefix=f"SAC-expert-{args.env_name}",
        record_video_trigger=lambda x: x % args.video_freq == 0,
        video_length=video_length,
    )

if args.policy_name == "sac":
    expert = SAC("MlpPolicy", venv, device=args.device, verbose=1, tensorboard_log=log_dir_expert)
elif args.policy_name == "ppo":
    expert = PPO("MlpPolicy", venv, device=args.device, verbose=1, tensorboard_log=log_dir_expert)
best_model_save_path = f"{log_dir_expert}/best_model" if args.if_save_weight else None
eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=args.eval_freq, log_path=log_dir_expert, best_model_save_path=best_model_save_path)
callbacks = [eval_callback]
if args.wandb:
    callbacks.append(WandbCallback(verbose=2))
expert.learn(args.train_steps, callback=CallbackList(callbacks))

if args.wandb:
    run.finish()
