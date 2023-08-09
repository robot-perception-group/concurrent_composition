import torch
import functorch
from torch.distributions import Normal
from common.util import pile_sa_pairs

Epsilon = 1e-6


class Compositions:
    def __init__(self, agent_cfg, policy, sf, prev_impact, n_heads, action_dim) -> None:
        self.policy = policy
        self.sf = sf
        self.prev_impact = prev_impact
        self.n_heads = n_heads
        self.feature_dim = n_heads
        self.action_dim = action_dim

        self.impact_x_idx = []  # record primitive impact on x-axis
        self.policy_idx = []  # record primitive summon frequency

        self.agent_name = agent_cfg["name"]
        if self.agent_name == "sfgpi":
            self.composition_fn = self.sfgpi
            self.record_impact = False
        elif self.agent_name == "msf":
            self.composition_fn = self.msf
            self.record_impact = False
        elif self.agent_name == "sfcpi":
            self.composition_fn = self.sfcpi
            self.record_impact = False
        elif self.agent_name == "dac":
            self.composition_fn = self.dac
            self.record_impact = True
        elif self.agent_name == "dacgpi":
            self.composition_fn = self.dacgpi
            self.record_impact = True
        else:
            raise NotImplementedError

    def sfgpi(self, s, w, mode):
        if mode == "explore":
            acts, _, _ = self.policy(s)  # [N, H, A]  <-- [N, S]
        elif mode == "exploit":
            _, _, acts = self.policy(s)  # [N, H, A]  <-- [N, S]

        qs = self.gpe(s, acts, w)  # [N, H, H] <-- [N, S], [N, H, A], [N, F]
        a = self.gpi(acts, qs)  # [N, A] <-- [N, H, A], [N, H, H]
        return a

    def msf(self, s, w, mode):
        means, log_stds = self.policy.get_mean_std(s)  # [N, H, A]
        composed_mean, composed_std = self.cpi(means, log_stds, w, rule="mcp")
        if mode == "explore":
            a = torch.tanh(Normal(composed_mean, composed_std).rsample())
        elif mode == "exploit":
            a = composed_mean
        return a

    def sfcpi(self, s, w, mode):
        means, log_stds = self.policy.get_mean_std(s)  # [N, H, A]
        # [N, Ha, Hsf] <-- [N, S], [N, H, A], [N, F]
        qs = self.gpe(s, means, w)
        qs = qs.mean(2)  # [N, H]
        composed_mean, composed_std = self.cpi(means, log_stds, qs, rule="mcp")
        if mode == "explore":
            a = torch.tanh(Normal(composed_mean, composed_std).rsample())
        elif mode == "exploit":
            a = composed_mean
        return a

    def dac(self, s, w, mode):
        means, log_stds = self.policy.get_mean_std(s)  # [N, H, A]
        kappa = self.cpe(s, means, w)
        composed_mean, composed_std = self.cpi(means, log_stds, kappa, rule="mca")
        if mode == "explore":
            a = torch.tanh(Normal(composed_mean, composed_std).rsample())
        elif mode == "exploit":
            a = composed_mean
        return a

    def dacgpi(self, s, w, mode):
        if mode == "explore":
            acts, _, _ = self.policy(s)  # [N, H, A]  <-- [N, S]
        elif mode == "exploit":
            _, _, acts = self.policy(s)  # [N, H, A]  <-- [N, S]

        kappa = self.cpe(s, acts, w)
        a = self.gpi(acts, kappa, rule="k")
        return a

    def gpe(self, s, acts, w):
        # [N, Ha, Hsf, F] <-- [N, S], [N, Ha, A]
        curr_sf = self.calc_curr_sf(s, acts)
        # [N,Ha,Hsf]<--[N,Ha,Hsf,F],[N,F]
        qs = torch.einsum("ijkl,il->ijk", curr_sf.float(), w.float())
        return qs  # [N,Ha,Hsf]

    def gpi(self, acts, value, rule="q"):
        if rule == "q":
            value_flat = value.flatten(1)  # [N, HH] <-- [N, H, Ha]
            idx = torch.div(value_flat.argmax(1), self.n_heads, rounding_mode="floor")
            idx = idx[:, None].repeat(1, self.action_dim).unsqueeze(1)  # [N,1,A]<-[N]
        elif rule == "k":
            idx = value.argmax(1).unsqueeze(1)  # [N, 1, A] <-- [N, H, A]
        a = torch.gather(acts, 1, idx).squeeze(1)  # [N, A] <-- [N, H, A]
        # record policy freqeuncy
        self.policy_idx.extend(idx.reshape(-1).cpu().numpy())
        return a  # [N, A]

    def cpe(self, s, a, w):
        curr_sf = self.calc_curr_sf(s, a)  # [N,Ha,Hsf,F]<--[N,S],[N,Ha,A]
        impact = self.calc_impact(s, a)  # [N,F,A]<--[N,Ha,Hsf,F],[N,Ha,A]
        kappa = self.calc_advantage(curr_sf)  # [N,Hsf,F] <-- [N, Ha, Hsf, F]
        kappa = torch.relu(kappa)  # filterout advantage below 0
        # [N,H,A]<--[N,Hsf,F], [N,F], [N,F,A]
        kappa = torch.einsum("ijk,ik,ikl->ijl", kappa, w, impact)
        return kappa

    def cpi(self, means, log_stds, gating, rule="mcp"):
        gating = self.scale_gating(gating)  # [N, Ha], H=F

        if rule == "mcp":
            # [N, H, A] <-- [N,F], [N,H,A], F=H
            w_div_std = torch.einsum("ij, ijk->ijk", gating, (-log_stds).exp())
        elif rule == "mca":
            # [N, H, A] <-- [N,F,H], [N,H,A], F=H
            w_div_std = torch.einsum("ijk, ijk->ijk", gating, (-log_stds).exp())

        composed_std = 1 / (w_div_std.sum(1) + Epsilon)  # [N,A]
        # [N,A]<--[N,A]*[N,A]<-- [N,H,A], [N,H,A]
        composed_mean = composed_std * torch.einsum("ijk,ijk->ik", means, w_div_std)
        return composed_mean, composed_std

    def calc_curr_sf(self, s, a):
        s_tiled, a_tiled = pile_sa_pairs(s, a)
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]

        curr_sf1, curr_sf2 = self.sf(s_tiled, a_tiled)
        # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]
        curr_sf = torch.min(curr_sf1, curr_sf2)
        curr_sf = curr_sf.view(-1, self.n_heads, self.n_heads, self.feature_dim)
        return curr_sf  # [N, Ha, Hsf, F]

    def scale_gating(self, gating):
        return torch.softmax(gating / gating.shape[1], 1)

    def calc_advantage(self, value):  # [N,Ha,Hsf,F]
        adv = value.mean(1, keepdim=True) - value.mean((1, 2), keepdim=True)
        return adv.squeeze(1)

    def calc_impact(self, s, a):
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]
        s, a = pile_sa_pairs(s, a)

        self.sf = self.sf.eval()

        def func(s, a):
            return torch.min(*self.sf(s, a))  # [NHa, Hsf, F]

        j = functorch.vmap(functorch.jacrev(func, argnums=1))(s, a)  # [NHa,Hsf,F,A]
        j = j.view(-1, self.n_heads, self.n_heads, self.feature_dim, self.action_dim)
        cur_impact = j.mean((1, 2)).abs()  # [N,F,A]<-[N,Ha,Hsf,F,A]
        self.sf = self.sf.train()

        impact = (self.prev_impact + cur_impact) / 2
        self.prev_impact = impact

        # record primitive impact
        idx = impact.argmax(1)
        self.impact_x_idx.extend(idx[:, 0].reshape(-1).cpu().numpy())
        return impact
