import argparse
import os

import gym
import numpy as np
import torch
import wandb

if torch.cuda.is_available():
    from torch.cuda import FloatTensor

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

from models.nets import Expert
from models.gail_simple import GAIL


def train(args, env, expert, gail):
    opt_d = torch.optim.Adam(gail.d.parameters())

    # if args.wandb:
    #     wandb.log(data={"eval/ep_return": eval_return}, step=0)

    # collect demo
    exp_rwd_iter, exp_obs, exp_acts = [], [], []
    steps = 0
    while steps < args.num_steps_per_iter:
        ep_obs, ep_rwds, t = [], [], 0
        (ob, _), done = env.reset(), False

        while not done and steps < args.num_steps_per_iter:
            act = expert.act(ob)
            ep_obs.append(ob);
            exp_obs.append(ob);
            exp_acts.append(act)

            if args.render:
                env.render()
            ob, rwd, done, truncated, _ = env.step(act)
            done = truncated or done

            ep_rwds.append(rwd)
            t += 1
            steps += 1

        if done:
            exp_rwd_iter.append(np.sum(ep_rwds))

    exp_rwd_mean = np.mean(exp_rwd_iter)
    print("Expert Reward Mean: {}".format(exp_rwd_mean))
    exp_obs, exp_acts = FloatTensor(np.array(exp_obs)), FloatTensor(np.array(exp_acts))

    # Train agent
    rwd_iter_means = []
    for i in range(args.num_iters):
        rwd_iter, obs, acts, rets, advs, gms = [], [], [], [], [], []
        steps = 0
        while steps < args.num_steps_per_iter:
            ep_obs, ep_acts, ep_rwds, ep_costs, ep_disc_costs, ep_gms, ep_lmbs = [], [], [], [], [], [], []
            t = 0
            (ob, _), done = env.reset(), False

            while not done and steps < args.num_steps_per_iter:
                act = gail.act(ob)
                ep_obs.append(ob);
                obs.append(ob);
                ep_acts.append(act);
                acts.append(act)

                if args.render:
                    env.render()
                ob, rwd, done, truncated, _ = env.step(act)
                done = truncated or done

                ep_rwds.append(rwd);
                ep_gms.append(args.gae_gamma ** t);
                ep_lmbs.append(args.gae_lambda ** t)
                t += 1
                steps += 1

            if done:
                rwd_iter.append(np.sum(ep_rwds))

            # gail computing discriminator costs
            ep_obs = FloatTensor(np.array(ep_obs))
            ep_acts = FloatTensor(np.array(ep_acts))
            ep_costs = (-1) * torch.log(gail.d(ep_obs, ep_acts)).squeeze().detach()

            # TRPO thing
            ep_gms = FloatTensor(ep_gms)
            ep_lmbs = FloatTensor(ep_lmbs)
            ep_disc_costs = ep_gms * ep_costs
            ep_disc_rets = FloatTensor([sum(ep_disc_costs[i:]) for i in range(t)])
            ep_rets = ep_disc_rets / ep_gms
            rets.append(ep_rets)
            ep_advs = gail.trpo_adv_eval(ep_obs, ep_costs, ep_gms, ep_lmbs, t)
            advs.append(ep_advs)
            gms.append(ep_gms)

        rwd_iter_means.append(np.mean(rwd_iter))
        print("Iterations: {},   Reward Mean: {}".format(i + 1, np.mean(rwd_iter)))

        obs = FloatTensor(np.array(obs))
        acts = FloatTensor(np.array(acts))
        rets = torch.cat(rets)
        advs = torch.cat(advs)
        gms = torch.cat(gms)

        if args.normalize_advantage:
            advs = (advs - advs.mean()) / advs.std()

        # Update Discriminator
        gail.d.train()
        exp_scores = gail.d.get_logits(exp_obs, exp_acts)
        nov_scores = gail.d.get_logits(obs, acts)
        opt_d.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(exp_scores, torch.zeros_like(exp_scores))
        loss += torch.nn.functional.binary_cross_entropy_with_logits(nov_scores, torch.ones_like(nov_scores))
        loss.backward()
        opt_d.step()

        gail.trpo_update(obs, rets, acts, advs, gms)


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

    model = GAIL(state_dim, action_dim, args).to(device)

    env.close()

    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt"))
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt"))
    if hasattr(model, "d"):
        torch.save(model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="CartPole-v1",
                        help="[CartPole-v1, Pendulum-v0, BipedalWalker-v3]")

    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wb_project", type=str, default="img-gen-rl-useless")
    parser.add_argument("--wb_entity", type=str, default="rowing0914")
    parser.add_argument("--wb_run", type=str, default="vanilla")
    parser.add_argument("--wb_group", type=str, default="vanilla")

    parser.add_argument("--if_discrete_action", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--num_steps_per_iter", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--lambda_update", type=float, default=1e-3)
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
        wandb.init(project=args.wb_project, entity=args.wb_entity, name=args.wb_run, group=args.wb_group,
                   dir="/tmp/wandb")
        wandb.config.update(args)

    main(args)
