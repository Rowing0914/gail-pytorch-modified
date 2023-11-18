import os
import json
import pickle
import argparse
import wandb
import torch
import gym

from rl.models.pg import PolicyGradient
from rl.models.ac import ActorCritic
from rl.models.trpo import TRPO
from rl.models.gae import GAE
from rl.models.ppo import PPO


def main(env_name, model_name):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    if args.if_discrete_action:
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if model_name == "pg":
        model = PolicyGradient(state_dim, action_dim, discrete, **config).to(device)
    elif model_name == "ac":
        model = ActorCritic(state_dim, action_dim, discrete, **config).to(device)
    elif model_name == "trpo":
        model = TRPO(state_dim, action_dim, discrete, **config).to(device)
    elif model_name == "gae":
        model = GAE(state_dim, action_dim, discrete, **config).to(device)
    elif model_name == "ppo":
        model = PPO(state_dim, action_dim, discrete, **config).to(device)

    results = model.train(env)

    env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt"))
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--model_name", type=str, default="trpo", help="[pg, ac, trpo, gae, ppo]")

    parser.add_argument("--if_discrete_action", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_steps_per_iter", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)

    parser.add_argument("--gae_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.01)

    parser.add_argument("--max_kl", type=float, default=0.01)
    parser.add_argument("--cg_damping", type=float, default=0.1)

    parser.add_argument("--ppo_vf_coeff", type=int, default=1)
    parser.add_argument("--ppo_entropy_coeff", type=float, default=0.01)
    args = parser.parse_args()
    args.normalize_advantage = True
    args.use_baseline = True
    if args.horizon == 0:
        args.horizon = None
    
    main(**vars(args))

