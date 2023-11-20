import numpy as np
import torch
import wandb

from torch.nn import Module

from rl.models.nets import PolicyNetwork, ValueNetwork
from rl.utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class TRPO(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        args=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.args = args

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        if self.args.use_baseline:
            self.v = ValueNetwork(self.state_dim)

    def get_networks(self):
        if self.args.use_baseline:
            return [self.pi, self.v]
        else:
            return [self.pi]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action

    def train(self, env, render=False):
        lr = self.args.lr
        num_iters = self.args.num_iters
        num_steps_per_iter = self.args.num_steps_per_iter
        horizon = self.args.horizon
        discount = self.args.discount
        max_kl = self.args.max_kl
        cg_damping = self.args.cg_damping
        normalize_advantage = self.args.normalize_advantage
        use_baseline = self.args.use_baseline

        if use_baseline:
            opt_v = torch.optim.Adam(self.v.parameters(), lr)

        rwd_iter_means = []
        global_ts = 0
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            disc = []

            steps = 0
            while steps < num_steps_per_iter:
                global_ts += 1
                ep_rwds = []
                ep_disc_rwds = []
                ep_disc = []

                t = 0
                done = False

                (ob, _) = env.reset()

                while not done and steps < num_steps_per_iter:
                    act = self.act(ob)

                    obs.append(ob)
                    acts.append(act)

                    if render:
                        env.render()
                    ob, rwd, done, truncated, _ = env.step(act)
                    done = truncated or done

                    ep_rwds.append(rwd)
                    ep_disc_rwds.append(rwd * (discount ** t))
                    ep_disc.append(discount ** t)

                    t += 1
                    steps += 1

                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break

                ep_disc = FloatTensor(ep_disc)

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_rwds[i:]) for i in range(t)]
                )
                ep_rets = ep_disc_rets / ep_disc

                rets.append(ep_rets)
                disc.append(ep_disc)

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

            rwd_iter_means.append(np.mean(rwd_iter))
            print(
                "Iterations: {},   Reward Mean: {}"
                .format(i + 1, np.mean(rwd_iter))
            )
            if self.args.wandb:
                wandb.log(data={"train/ep_return": np.mean(rwd_iter)}, step=global_ts)

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            disc = torch.cat(disc)

            if normalize_advantage:
                rets = (rets - rets.mean()) / rets.std()

            if use_baseline:
                self.v.eval()
                delta = (rets - self.v(obs).squeeze()).detach()

                self.v.train()

                opt_v.zero_grad()
                loss = (-1) * disc * delta * self.v(obs).squeeze()
                loss.mean().backward()
                opt_v.step()

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                if use_baseline:
                    return (disc * delta * torch.exp(
                                distb.log_prob(acts)
                                - old_distb.log_prob(acts).detach()
                            )).mean()
                else:
                    return (disc * rets * torch.exp(
                                distb.log_prob(acts)
                                - old_distb.log_prob(acts).detach()
                            )).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

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
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            set_params(self.pi, new_params)

        return rwd_iter_means
