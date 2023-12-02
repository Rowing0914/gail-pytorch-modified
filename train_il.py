import numpy as np
import gym, argparse, os
from stable_baselines3 import SAC
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
parser.add_argument("--gen_train_timesteps", default=1000, type=int)
parser.add_argument("--max_episode_steps", default=1000, type=int)
parser.add_argument("--train_steps", default=10_000_000, type=int)
parser.add_argument("--num_eval_episodes", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--eval_freq_by_round", default=100, type=int)  # How often (round) we evaluate
parser.add_argument("--if_save_weight", action="store_true", default=False)
parser.add_argument("--if_no_use_state", action="store_true", default=False)
parser.add_argument("--if_no_use_action", action="store_true", default=False)

# video related
parser.add_argument("--if_video", action="store_true", default=False)
parser.add_argument("--video_freq", default=10000, type=int)  # How often (time steps) we evaluate
args = parser.parse_args()

from datetime import datetime
t = datetime.now()
run_name = f"agent-{args.policy_name}-{args.env_name}-seed{args.seed}-{t.month}{t.day}{t.hour}{t.minute}{t.second}"
log_dir_agent = os.path.join(args.log_root, run_name)
os.makedirs(log_dir_agent, exist_ok=True)

if args.wandb:
    wandb.login()
    run = wandb.init(
        project=args.wb_project, entity=args.wb_entity, name=run_name, group=args.wb_group, dir="/tmp/wandb",
        # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        )
    wandb.config.update(args)

rng = np.random.default_rng(args.seed)
log_dir_expert = "./logs/expert-sac-HalfCheetah-v3-seed1-12124048/"

import pickle
with open(f"{log_dir_expert}/rollouts.pkl", "rb") as handle:
    rollouts = pickle.load(handle)

venv = make_vec_env(args.env_name, n_envs=args.num_envs, rng=rng)
venv_eval = make_vec_env(args.env_name, n_envs=args.num_envs, rng=rng)
venv.my_args = args
venv.venv_eval = venv_eval
# learner = SAC(env=venv, policy=MlpPolicy)
learner = SAC("MlpPolicy", venv, device=args.device, verbose=0, tensorboard_log=log_dir_agent)
reward_net = BasicRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
    use_state=not args.if_no_use_state,
    use_action=not args.if_no_use_action,
    use_next_state=False,
    use_done=False,
)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=int(args.gen_train_timesteps * 0.5),
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    gen_train_timesteps=args.gen_train_timesteps,
)
# import pudb; pudb.start()
if args.wandb:
    import wandb
    rewards, _ = evaluate_policy(learner, venv.venv_eval, 1, return_episode_rewards=True)
    wandb.log(data={"rollout/ep_rew_mean": np.mean(rewards)}, step=0)
    wandb.log(data={"eval/mean_reward": np.mean(rewards)}, step=0)
gail_trainer.train(args.train_steps)
rewards, _ = evaluate_policy(learner, venv.venv_eval, 10, return_episode_rewards=True)
if args.wandb:
    import wandb
    wandb.log(data={"rollout/ep_rew_mean": np.mean(rewards)}, step=gail_trainer._global_step + 1)
    wandb.log(data={"eval/mean_reward": np.mean(rewards)}, step=gail_trainer._global_step + 1)
print("Rewards:", rewards)

if if_wandb:
    run.finish()