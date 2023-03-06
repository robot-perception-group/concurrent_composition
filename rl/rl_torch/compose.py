from pathlib import Path

import functorch
import numpy as np
import torch
import wandb
from rl.rl_torch.common.agent import RaisimAgent
from rl.rl_torch.common.policy import MultiheadGaussianPolicy
from rl.rl_torch.common.util import (
    grad_false,
    hard_update,
    pile_sa_pairs,
    soft_update,
    update_params,
)
from rl.rl_torch.common.value_function import TwinnedMultiheadSFNetwork
from torch.distributions import Normal
from torch.optim import Adam

Epsilon = 1e-6


class CompositionAgent(RaisimAgent):
    @classmethod
    def default_config(cls):
        default_cfg = super().default_config()
        env_cfg = default_cfg["env_cfg"]
        agent_cfg = default_cfg["agent_cfg"]
        buffer_cfg = default_cfg["buffer_cfg"]

        env_cfg.update(
            dict(
                env_name="pointmass3d",  # pointmassXd[,_simple,_augment]
                num_envs=int(200),
                episode_max_step=int(200),
                eval_interval=10,
                total_episodes=int(10),
                random_robot_state=True,
                random_target_state=True,
                num_threads=10,
                save_model=True,
                render=True,  # render env
                record=True,  # dump config, record video only works if env is rendered
            )
        )
        agent_cfg.update(
            dict(
                # record folder name, options: [sfgpi, msf, sfcpi, dac, dacgpi, pickplace]
                name="sfgpi",
                importance_sampling=True,
                is_clip_max=1.0,
                entropy_tuning=True,
                alpha=0.2,
                alpha_lr=3e-4,
                lr=0.006589,
                policy_lr=0.006425,
                gamma=0.99,
                tau=5e-3,
                td_target_update_interval=2,
                updates_per_step=3,
                reward_scale=1.0,
                grad_clip=None,
                value_net_kwargs={
                    "sizes": [32, 32],
                    "activation": "selu",
                    "layernorm": True,
                    "droprate": 0.05,
                },
                policy_net_kwargs={
                    "sizes": [32, 32],
                    "layernorm": False,
                },
            )
        )
        buffer_cfg.update(
            dict(
                capacity=int(1e6),
                mini_batch_size=128,
                min_n_experience=1024,
                multi_step=5,
                n_env=env_cfg["num_envs"],
                prioritize_replay=False,
            )
        )
        return {"env_cfg": env_cfg, "agent_cfg": agent_cfg, "buffer_cfg": buffer_cfg}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lr = self.agent_cfg["lr"]
        self.policy_lr = self.agent_cfg["policy_lr"]
        self.value_net_kwargs = self.agent_cfg["value_net_kwargs"]
        self.policy_net_kwargs = self.agent_cfg["policy_net_kwargs"]
        self.gamma = self.agent_cfg["gamma"]
        self.tau = self.agent_cfg["tau"]
        self.td_target_update_interval = int(
            self.agent_cfg["td_target_update_interval"]
        )
        self.entropy_tuning = self.agent_cfg["entropy_tuning"]
        self.importance_sampling = self.agent_cfg["importance_sampling"]
        self.is_clip_range = (0, self.agent_cfg["is_clip_max"])
        self.updates_per_step = self.agent_cfg["updates_per_step"]
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.n_heads = self.feature_dim
        self.droprate = self.value_net_kwargs["droprate"]

        self.agent_name = self.agent_cfg["name"]
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
        elif self.agent_name == "pickplace":
            self.composition_fn = self.pickplace
        else:
            raise NotImplementedError

        self.sf = TwinnedMultiheadSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.value_net_kwargs,
        ).to(self.device)

        self.sf_target = TwinnedMultiheadSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.value_net_kwargs,
        ).to(self.device)
        if self.droprate <= 0.0:
            self.sf_target = self.sf_target.eval()

        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)

        self.policy = MultiheadGaussianPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.policy_net_kwargs,
        ).to(self.device)

        self.sf1_optimizer = Adam(self.sf.SF1.parameters(), lr=self.lr)
        self.sf2_optimizer = Adam(self.sf.SF2.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=self.policy_lr)

        if self.entropy_tuning:
            self.alpha_lr = self.agent_cfg["alpha_lr"]
            self.target_entropy = (
                -torch.prod(torch.tensor(self.action_shape,
                            device=self.device))
                .tile(self.n_heads)
                .unsqueeze(1)
            )  # -|A|, [H,1]
            # # optimize log(alpha), instead of alpha
            self.log_alpha = torch.zeros(
                (self.n_heads, 1), requires_grad=True, device=self.device
            )
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(self.agent_cfg["alpha"]).to(self.device)

        self.pseudo_w = torch.eye(self.feature_dim).to(
            self.device)  # base tasks
        self.prev_impact = torch.zeros((self.n_env, self.n_heads, self.action_dim)).to(
            self.device
        )

        self.impact_x_idx = []  # record primitive impact on x-axis
        self.policy_idx = []  # record primitive summon frequency

        self.learn_steps = 0

    def explore(self, s, w):
        # [N, A] <-- [N, S], [N, H, A], [N, F]
        a = self.composition_fn(s, w, mode="explore")
        return a  # [N, A]

    def exploit(self, s, w):
        # [N, A] <-- [N, S], [N, H, A], [N, F]
        a = self.composition_fn(s, w, mode="exploit")
        return a  # [N, A]

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
        composed_mean, composed_std = self.cpi(
            means, log_stds, kappa, rule="mca")
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

    def pickplace(self, s, w, mode):  # only works for 2d simple env as baseline
        if mode == "explore":
            acts, _, _ = self.policy(s)  # [N, H, A]  <-- [N, S]
        elif mode == "exploit":
            _, _, acts = self.policy(s)  # [N, H, A]  <-- [N, S]

        axx = acts[:, 0, 0].unsqueeze(1)  # [N,1] <-- [N, H, A]
        ayy = acts[:, 1, 1].unsqueeze(1)  # [N,1] <-- [N, H, A]
        return torch.cat([axx, ayy], 1)  # [N,2]

    def gpe(self, s, acts, w):
        # [N, Ha, Hsf, F] <-- [N, S], [N, Ha, A]
        curr_sf = self.calc_curr_sf(s, acts)
        # [N,Ha,Hsf]<--[N,Ha,Hsf,F],[N,F]
        qs = torch.einsum("ijkl,il->ijk", curr_sf, w)
        return qs  # [N,Ha,Hsf]

    def gpi(self, acts, value, rule="q"):
        if rule == "q":
            value_flat = value.flatten(1)  # [N, HH] <-- [N, H, Ha]
            idx = torch.div(value_flat.argmax(
                1), self.n_heads, rounding_mode="floor")
            idx = idx[:, None].repeat(
                1, self.action_dim).unsqueeze(1)  # [N,1,A]<-[N]
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
            w_div_std = torch.einsum(
                "ijk, ijk->ijk", gating, (-log_stds).exp())

        composed_std = 1 / (w_div_std.sum(1) + Epsilon)  # [N,A]
        # [N,A]<--[N,A]*[N,A]<-- [N,H,A], [N,H,A]
        composed_mean = composed_std * \
            torch.einsum("ijk,ijk->ik", means, w_div_std)
        return composed_mean, composed_std

    def scale_gating(self, gating):
        return torch.softmax(gating / gating.shape[1], 1)

    def calc_advantage(self, value):  # [N,Ha,Hsf,F]
        adv = value.mean(1, keepdim=True) - value.mean((1, 2), keepdim=True)
        return adv.squeeze(1)

    def calc_impact(self, s, a):
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]
        s, a = pile_sa_pairs(s, a)

        self.sf = self.sf.eval()
        def func(s, a): return torch.min(*self.sf(s, a))  # [NHa, Hsf, F]
        j = functorch.vmap(functorch.jacrev(
            func, argnums=1))(s, a)  # [NHa,Hsf,F,A]
        j = j.view(-1, self.n_heads, self.n_heads,
                   self.feature_dim, self.action_dim)
        cur_impact = j.mean((1, 2)).abs()  # [N,F,A]<-[N,Ha,Hsf,F,A]
        self.sf = self.sf.train()

        impact = (self.prev_impact + cur_impact) / 2
        self.prev_impact = impact

        # record primitive impact
        idx = impact.argmax(1)
        self.impact_x_idx.extend(idx[:, 0].reshape(-1).cpu().numpy())
        return impact

    def reset_env(self):
        self.prev_impact = torch.zeros((self.n_env, self.n_heads, self.action_dim)).to(
            self.device
        )
        self.impact_x_idx = []
        self.policy_idx = []
        return super().reset_env()

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.sf_target, self.sf, self.tau)

        if self.per:
            batch, indices, priority_weights = self.replay_buffer.sample(
                self.mini_batch_size
            )
        else:
            batch = self.replay_buffer.sample(self.mini_batch_size)
            priority_weights = torch.ones(
                (self.mini_batch_size, 1)).to(self.device)

        sf1_loss, sf2_loss, errors, mean_sf1, mean_sf2, target_sf = self.calc_sf_loss(
            batch, priority_weights
        )
        policy_loss, entropies = self.calc_policy_loss(batch, priority_weights)

        update_params(self.policy_optimizer, self.policy,
                      policy_loss, self.grad_clip)
        update_params(self.sf1_optimizer, self.sf.SF1,
                      sf1_loss, self.grad_clip)
        update_params(self.sf2_optimizer, self.sf.SF2,
                      sf2_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, priority_weights)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        if self.per:
            self.replay_buffer.update_priority(
                indices, errors.detach().cpu().numpy())

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/SF1": sf1_loss.detach().item(),
                "loss/SF2": sf2_loss.detach().item(),
                "loss/policy": policy_loss.detach().item(),
                "state/mean_SF1": mean_sf1,
                "state/mean_SF2": mean_sf2,
                "state/target_sf": target_sf.detach().mean().item(),
                "state/lr": self.lr,
                "state/entropy": entropies.detach().mean().item(),
                "state/policy_idx": wandb.Histogram(self.policy_idx),
            }
            if self.record_impact:
                metrics.update(
                    {
                        "state/impact_x_idx": wandb.Histogram(self.impact_x_idx),
                    }
                )
            if self.entropy_tuning:
                metrics.update(
                    {
                        "loss/alpha": entropy_loss.detach().item(),
                        "state/alpha": self.alpha.mean().detach().item(),
                    }
                )

            wandb.log(metrics)

    def calc_sf_loss(self, batch, priority_weights):
        (s, f, a, _, s_next, dones) = batch

        curr_sf1, curr_sf2 = self.sf(s, a)  # [N, H, F] <-- [N, S], [N,A]
        target_sf = self.calc_target_sf(f, s_next, dones)  # [N, H, F]

        # importance sampling
        if self.importance_sampling:
            ratio = self.calc_importance_ratio(
                s, self.is_clip_range)  # [N, H, F]
        else:
            ratio = torch.ones_like(target_sf)

        loss1 = (ratio * (curr_sf1 - target_sf)).pow(2)  # [N, H, F]
        loss2 = (ratio * (curr_sf2 - target_sf)).pow(2)  # [N, H, F]

        loss1 = loss1 * priority_weights.unsqueeze(
            1
        )  # [N, H, F] <-- [N, H, F] * [N,1,1]
        loss2 = loss2 * priority_weights.unsqueeze(
            1
        )  # [N, H, F] <-- [N, H, F] * [N,1,1]

        sf1_loss = torch.mean(loss1)
        sf2_loss = torch.mean(loss2)

        # TD errors for updating priority weights
        errors = torch.mean(torch.abs(curr_sf1.detach() - target_sf), (1, 2))

        # log means to monitor training.
        mean_sf1 = curr_sf1.detach().mean().item()
        mean_sf2 = curr_sf2.detach().mean().item()

        return sf1_loss, sf2_loss, errors, mean_sf1, mean_sf2, target_sf

    def calc_policy_loss(self, batch, priority_weights):
        (s, f, a, r, s_next, dones) = batch

        a_heads, entropies, _ = self.policy(s)  # [N,H,A], [N, H, 1] <-- [N,S]

        qs = self.calc_qs_from_sf(s, a_heads)
        qs = qs.unsqueeze(2)  # [N,H,1]

        loss = -qs - self.alpha * entropies
        # [N, H, 1] <--  [N, H, 1], [N,1,1]
        loss = loss * priority_weights.unsqueeze(1)
        policy_loss = torch.mean(loss)

        return policy_loss, entropies

    def calc_entropy_loss(self, entropy, priority_weights):
        loss = self.log_alpha * (self.target_entropy - entropy).detach()
        # [N, H, 1] <--  [N, H, 1], [N,1,1]
        loss = loss * priority_weights.unsqueeze(1)
        entropy_loss = -torch.mean(loss)
        return entropy_loss

    def calc_qs_from_sf(self, s, a):
        qs = torch.stack(
            [
                self.calc_q_from_sf(s, a[:, i, :], self.pseudo_w[i], i)
                for i in range(self.n_heads)
            ],
            1,
        )
        return qs  # [N,H]

    def calc_q_from_sf(self, s, a, w, head_idx):
        curr_sf1 = self.sf.SF1.forward_head(
            s, a, head_idx)  # [N, F] <-- [N, S], [N, A]
        curr_sf2 = self.sf.SF2.forward_head(
            s, a, head_idx)  # [N, F] <-- [N, S], [N, A]
        q1 = torch.einsum("ij,j->i", curr_sf1, w)  # [N]<-- [N,F]*[F]
        q2 = torch.einsum("ij,j->i", curr_sf2, w)  # [N]<-- [N,F]*[F]
        if self.droprate > 0.0:
            q = 0.5 * (q1 + q2)
        else:
            q = torch.min(q1, q2)
        return q

    def calc_curr_sf(self, s, a):
        s_tiled, a_tiled = pile_sa_pairs(s, a)
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]

        curr_sf1, curr_sf2 = self.sf(s_tiled, a_tiled)
        # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]
        curr_sf = torch.min(curr_sf1, curr_sf2)
        curr_sf = curr_sf.view(-1, self.n_heads,
                               self.n_heads, self.feature_dim)
        return curr_sf  # [N, Ha, Hsf, F]

    def calc_target_sf(self, f, s_next, dones):
        _, _, a_next = self.policy(s_next)  # [N, H, 1],[N, H, A] <-- [N, S]

        with torch.no_grad():
            next_sf1 = torch.stack(
                [
                    self.sf_target.SF1.forward_head(s_next, a_next[:, i, :], i)
                    for i in range(self.n_heads)
                ],
                1,
            )  # [N,H,F]  <-- [N, S], [N, A]
            next_sf2 = torch.stack(
                [
                    self.sf_target.SF2.forward_head(s_next, a_next[:, i, :], i)
                    for i in range(self.n_heads)
                ],
                1,
            )  # [N,H,F]  <-- [N, S], [N, A]
            next_sf = torch.min(next_sf1, next_sf2)  # [N, H, F]

        f = torch.tile(f[:, None, :], (self.n_heads, 1))  # [N,H,F] <-- [N,F]
        target_sf = f + torch.einsum(
            "ijk,il->ijk", next_sf, (1.0 - dones) * self.gamma
        )  # [N, H, F] <-- [N, H, F]+ [N, H, F]

        return target_sf  # [N, H, F]

    def calc_priority_error(self, batch):
        (s, f, a, _, s_next, dones) = batch

        with torch.no_grad():
            curr_sf1, curr_sf2 = self.sf(s, a)
        target_sf = self.calc_target_sf(f, s_next, dones)
        error = torch.mean(torch.abs(curr_sf1 - target_sf), (1, 2))
        return error.unsqueeze(1).cpu().numpy()

    def calc_importance_ratio(self, s, clip_range=(0.0, 1.3)):
        with torch.no_grad():
            means, log_stds = self.policy.get_mean_std(s)  # [N, 2*H*A]
            normals = Normal(means, log_stds.exp())
            logp = normals.log_prob(torch.tanh(means))
            logp = logp.sum(dim=2, keepdim=True).squeeze()  # [N, H]

            ratio = [logp[:, i, None] - logp for i in range(logp.shape[1])]
            ratio = torch.stack(ratio, 2).exp()  # [N, H, F]
            ratio = ratio.clip(*clip_range)
        return ratio  # [N, H, F]

    def save_torch_model(self):
        path = self.log_path + f"model{self.episodes}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        self.policy.save(path + "policy")
        self.sf.SF1.save(path + "sf1")
        self.sf.SF2.save(path + "sf2")

    def load_torch_model(self, path):
        self.policy.load(path + "policy")
        self.sf.SF1.load(path + "sf1")
        self.sf.SF2.load(path + "sf2")
        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)


if __name__ == "__main__":
    import pprint

    from rl.rl_torch.common.util import fix_config

    default_cfg = CompositionAgent.default_config()
    wandb.init(config=default_cfg)
    cfg = fix_config(wandb.config)
    pprint.pprint(cfg)

    agent = CompositionAgent(cfg)
    agent.run()
    wandb.finish()
