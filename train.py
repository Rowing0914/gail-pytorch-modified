import os
import pickle
import argparse
import wandb
import torch
import gym

from models.nets import Expert
from models.gail import GAIL


def main(args):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, args.env_name)

    ckpt_path = os.path.join(ckpt_path, args.env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    env = gym.make(args.env_name)
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

    expert = Expert(state_dim, action_dim, args.if_discrete_action).to(device)
    expert.pi.load_state_dict(torch.load(os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device))

    # for Pendulum
    config = {
        "num_iters": 500,
        "num_steps_per_iter": 2000,
        "horizon": None,
        "lambda": 1e-3,
        "gae_gamma": 0.99,
        "gae_lambda": 0.99,
        "epsilon": 0.01,
        "max_kl": 0.01,
        "cg_damping": 0.1,
        "normalize_advantage": True
    }
    model = GAIL(state_dim, action_dim, args.if_discrete_action, config, args).to(device)
    results = model.train(env, expert)
    env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt"))
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt"))
    if hasattr(model, "d"):
        torch.save(model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="[CartPole-v1, Pendulum-v0, BipedalWalker-v3]")

    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wb_project", type=str, default="img-gen-rl")
    parser.add_argument("--wb_entity", type=str, default="rowing0914")
    parser.add_argument("--wb_run", type=str, default="vanilla")
    parser.add_argument("--wb_group", type=str, default="vanilla")

    parser.add_argument("--if_discrete_action", action="store_true", default=False)
    parser.add_argument("--if_obs_only", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--num_steps_per_iter", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--lambda", type=float, default=1e-3)
    parser.add_argument("--gae_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--max_kl", type=float, default=0.01)
    parser.add_argument("--cg_damping", type=float, default=0.1)
    args = parser.parse_args()
    args.normalize_advantage = True
    if args.horizon == 0:
        args.horizon = None

    if args.wandb:
        wandb.login()
        wandb.init(project=args.wb_project, entity=args.wb_entity, name=args.wb_run, group=args.wb_group, dir="/tmp/wandb")
        wandb.config.update(args)

    main(args)
