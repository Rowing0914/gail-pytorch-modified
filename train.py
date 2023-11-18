import os
import json
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

    if args.env_name not in ["CartPole-v1", "Pendulum-v1", "BipedalWalker-v3"]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, args.env_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)

    ckpt_path = os.path.join(ckpt_path, args.env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[args.env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(args.env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    if args.env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    expert = Expert(state_dim, action_dim, discrete, **expert_config).to(device)
    expert.pi.load_state_dict(torch.load(os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device))

    model = GAIL(state_dim, action_dim, discrete, config, args).to(device)
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
    args = parser.parse_args()

    if args.wandb:
        wandb.login()
        wandb.init(project=args.wb_project, entity=args.wb_entity, name=args.wb_run, group=args.wb_group, dir="/tmp/wandb")
        wandb.config.update(args)

    main(args)
