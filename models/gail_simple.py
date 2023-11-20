import torch
from torch.nn import Module

if torch.cuda.is_available():
    from torch.cuda import FloatTensor

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, conjugate_gradient, rescale_and_linesearch


class GAIL(Module):
    def __init__(self, state_dim, action_dim, args):
        super().__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()
        state = FloatTensor(state)
        distb = self.pi(state)
        action = distb.sample().detach().cpu().numpy()
        return action

    def trpo_adv_eval(self, ep_obs, ep_costs, ep_gms, ep_lmbs, t):
        self.v.eval()
        curr_vals = self.v(ep_obs).detach()
        next_vals = torch.cat((self.v(ep_obs)[1:], FloatTensor([[0.]]))).detach()
        ep_deltas = ep_costs.unsqueeze(-1) + self.args.gae_gamma * next_vals - curr_vals
        ep_advs = FloatTensor([((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:]).sum() for j in range(t)])
        return ep_advs

    def trpo_update(self, obs, rets, acts, advs, gms):
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
        alpha = torch.sqrt(2 * self.args.epsilon / torch.dot(s, Hs))

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

            if self.args.discrete:
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
            return hessian + self.args.cg_damping * v

        g = get_flat_grads(L(), self.pi).detach()
        s = conjugate_gradient(Hv, g).detach()
        Hs = Hv(s).detach()

        new_params = rescale_and_linesearch(g, s, Hs, self.args.max_kl, L, kld, old_params, self.pi)

        disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts)).mean()
        grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
        new_params += self.args.lambda_update * grad_disc_causal_entropy
        set_params(self.pi, new_params)
