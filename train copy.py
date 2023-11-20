import os
import json
import pickle
import argparse
import wandb
import torch
import gym

from models.nets import Expert
from models.gail import GAIL

def train():
    num_iters = self.train_config["num_iters"]
    num_steps_per_iter = self.train_config["num_steps_per_iter"]
    horizon = self.train_config["horizon"]
    lambda_ = self.train_config["lambda"]
    gae_gamma = self.train_config["gae_gamma"]
    gae_lambda = self.train_config["gae_lambda"]
    eps = self.train_config["epsilon"]
    max_kl = self.train_config["max_kl"]
    cg_damping = self.train_config["cg_damping"]
    normalize_advantage = self.train_config["normalize_advantage"]

    opt_d = torch.optim.Adam(self.d.parameters())
    
    # if self.args.wandb:
    #     wandb.log(data={"eval/ep_return": eval_return}, step=0)

    # collect demo
    exp_rwd_iter, exp_obs, exp_acts = [], [], []
    steps = 0
    while steps < num_steps_per_iter:
        ep_obs, ep_rwds, t = [], [], 0
        (ob, _), done = env.reset(), False

        while not done and steps < num_steps_per_iter:
            act = expert.act(ob)
            ep_obs.append(ob); exp_obs.append(ob); exp_acts.append(act)

            if render:
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
    for i in range(num_iters):
        rwd_iter, obs, acts, rets, advs, gms = [], [], [], [], [], []
        steps = 0
        while steps < num_steps_per_iter:
            ep_obs, ep_acts, ep_rwds, ep_costs, ep_disc_costs, ep_gms, ep_lmbs = [], [], [], [], [], [], []
            t = 0
            (ob, _), done = env.reset(), False

            while not done and steps < num_steps_per_iter:
                act = self.act(ob)
                ep_obs.append(ob); obs.append(ob); ep_acts.append(act); acts.append(act)

                if render:
                    env.render()
                ob, rwd, done, truncated, _ = env.step(act)
                done = truncated or done

                ep_rwds.append(rwd); ep_gms.append(gae_gamma ** t); ep_lmbs.append(gae_lambda ** t)
                t += 1
                steps += 1

            if done:
                rwd_iter.append(np.sum(ep_rwds))

            ep_obs = FloatTensor(np.array(ep_obs))
            ep_acts = FloatTensor(np.array(ep_acts))
            ep_rwds = FloatTensor(ep_rwds)
            ep_gms = FloatTensor(ep_gms)
            ep_lmbs = FloatTensor(ep_lmbs)

            ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
            ep_disc_costs = ep_gms * ep_costs
            ep_disc_rets = FloatTensor([sum(ep_disc_costs[i:]) for i in range(t)])
            ep_rets = ep_disc_rets / ep_gms

            rets.append(ep_rets)

            self.v.eval()
            curr_vals = self.v(ep_obs).detach()
            next_vals = torch.cat((self.v(ep_obs)[1:], FloatTensor([[0.]]))).detach()
            ep_deltas = ep_costs.unsqueeze(-1) + gae_gamma * next_vals - curr_vals
            ep_advs = FloatTensor([((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:]).sum() for j in range(t)])
            advs.append(ep_advs)
            gms.append(ep_gms)

        rwd_iter_means.append(np.mean(rwd_iter))
        print("Iterations: {},   Reward Mean: {}".format(i + 1, np.mean(rwd_iter)))

        obs = FloatTensor(np.array(obs))
        acts = FloatTensor(np.array(acts))
        rets = torch.cat(rets)
        advs = torch.cat(advs)
        gms = torch.cat(gms)

        if normalize_advantage:
            advs = (advs - advs.mean()) / advs.std()

        self.d.train()
        exp_scores = self.d.get_logits(exp_obs, exp_acts)
        nov_scores = self.d.get_logits(obs, acts)
        opt_d.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(exp_scores, torch.zeros_like(exp_scores))
        loss += torch.nn.functional.binary_cross_entropy_with_logits(nov_scores, torch.ones_like(nov_scores))
        loss.backward()
        opt_d.step()

        # === TRPO update
        self.v.train()
        old_params = get_flat_params(self.v).detach()
        old_v = self.v(obs).detach()

        def constraint():
            return ((old_v - self.v(obs)) ** 2).mean()

        grad_diff = get_flat_grads(constraint(), self.v)

        def Hv(v):
            hessian = get_flat_grads(torch.dot(grad_diff, v), self.v).detach()
            return hessian

        g = get_flat_grads(((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v).detach()
        s = conjugate_gradient(Hv, g).detach()

        Hs = Hv(s).detach()
        alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

        new_params = old_params + alpha * s

        set_params(self.v, new_params)

        self.pi.train()
        old_params = get_flat_params(self.pi).detach()
        old_distb = self.pi(obs)

        def L():
            distb = self.pi(obs)
            return (advs * torch.exp(distb.log_prob(acts) - old_distb.log_prob(acts).detach())).mean()

        def kld():
            distb = self.pi(obs)

            if self.discrete:
                old_p = old_distb.probs.detach()
                p = distb.probs
                return (old_p * (torch.log(old_p) - torch.log(p))).sum(-1).mean()

            else:
                old_mean = old_distb.mean.detach()
                old_cov = old_distb.covariance_matrix.sum(-1).detach()
                mean = distb.mean
                cov = distb.covariance_matrix.sum(-1)

                return (0.5) * (
                        (old_cov / cov).sum(-1)
                        + (((old_mean - mean) ** 2) / cov).sum(-1)
                        - self.action_dim
                        + torch.log(cov).sum(-1)
                        - torch.log(old_cov).sum(-1)
                    ).mean()

        grad_kld_old_param = get_flat_grads(kld(), self.pi)

        def Hv(v):
            hessian = get_flat_grads(torch.dot(grad_kld_old_param, v), self.pi).detach()
            return hessian + cg_damping * v

        g = get_flat_grads(L(), self.pi).detach()
        s = conjugate_gradient(Hv, g).detach()
        Hs = Hv(s).detach()

        new_params = rescale_and_linesearch(g, s, Hs, max_kl, L, kld, old_params, self.pi)

        disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts)).mean()
        grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
        new_params += lambda_ * grad_disc_causal_entropy
        set_params(self.pi, new_params)


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

    expert = Expert(state_dim, action_dim, args.if_discrete_action, **expert_config).to(device)
    expert.pi.load_state_dict(torch.load(os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device))

    model = GAIL(state_dim, action_dim, args.if_discrete_action, config, args).to(device)
    
    
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
