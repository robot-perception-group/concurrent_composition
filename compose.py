from pathlib import Path

import torch
import wandb
from common.agent import IsaacAgent
from common.policy import MultiheadGaussianPolicy
from common.util import (
    grad_false,
    hard_update,
    pile_sa_pairs,
    soft_update,
    update_params,
)
from common.value_function import TwinnedMultiheadSFNetwork
from common.compositions import Compositions
from torch.optim import Adam


class CompositionAgent(IsaacAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        try:
            self.lr = self.agent_cfg["lr"][0]
            self.policy_lr = self.agent_cfg["lr"][1]
        except:
        # bad fix
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
        self.is_clip_range = (0, self.agent_cfg["is_clip_max"])
        self.updates_per_step = self.agent_cfg["updates_per_step"]
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.n_heads = self.feature_dim
        self.droprate = self.value_net_kwargs["droprate"]

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

        # self.sf1_optimizer = Adam(self.sf.SF1.parameters(), lr=self.lr)
        # self.sf2_optimizer = Adam(self.sf.SF2.parameters(), lr=self.lr)
        self.sf_optimizer = Adam(self.sf.parameters(), lr=self.lr, betas=[0.9, 0.999])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=self.policy_lr, betas=[0.9, 0.999]
        )

        if self.entropy_tuning:
            self.alpha_lr = self.agent_cfg["alpha_lr"]
            self.target_entropy = (
                -torch.prod(torch.tensor(self.action_shape, device=self.device))
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

        self.pseudo_w = torch.eye(self.feature_dim).to(self.device)  # base tasks
        self.prev_impact = torch.zeros((self.n_env, self.n_heads, self.action_dim)).to(
            self.device
        )

        self.comp = Compositions(
            self.agent_cfg,
            self.policy,
            self.sf,
            self.prev_impact,
            self.n_heads,
            self.action_dim,
        )

        self.learn_steps = 0

        # masks used for vectorizing functions
        self.mask = torch.eye(self.n_heads, device="cuda:0").unsqueeze(dim=-1)
        self.mask = self.mask.repeat(self.mini_batch_size, 1, self.n_heads)
        self.mask = self.mask.ge(0.5)

    def explore(self, s, w):
        # [N, A] <-- [N, S], [N, H, A], [N, F]
        a = self.comp.composition_fn(s, w, mode="explore")
        return a  # [N, A]

    def exploit(self, s, w):
        # [N, A] <-- [N, S], [N, H, A], [N, F]
        a = self.comp.composition_fn(s, w, mode="exploit")
        return a  # [N, A]

    def reset_env(self):
        self.prev_impact = torch.zeros((self.n_env, self.n_heads, self.action_dim)).to(
            self.device
        )
        self.comp.impact_x_idx = []
        self.comp.policy_idx = []
        return super().reset_env()

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.sf_target, self.sf, self.tau)

        # if self.per:
        #     batch, indices, priority_weights = self.replay_buffer.sample(
        #         self.mini_batch_size
        #     )
        # else:
        batch = self.replay_buffer.sample(self.mini_batch_size)
        priority_weights = torch.ones((self.mini_batch_size, 1)).to(self.device)

        sf_loss, errors, mean_sf1 = self.update_sf(
            batch, priority_weights
        )
        policy_loss, entropies = self.update_policy(batch, priority_weights)

        # update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)
        # update_params(self.sf1_optimizer, self.sf.SF1, sf1_loss, self.grad_clip)
        # update_params(self.sf2_optimizer, self.sf.SF2, sf2_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, priority_weights)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        # if self.per:
        #     self.replay_buffer.update_priority(indices, errors.detach().cpu().numpy())

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/SF": sf_loss,
                "loss/policy": policy_loss,
                "state/mean_SF1": mean_sf1,
                "state/entropy": entropies.detach().mean().item(),
                "state/policy_idx": wandb.Histogram(self.comp.policy_idx),
            }
            if self.comp.record_impact:
                metrics.update(
                    {
                        "state/impact_x_idx": wandb.Histogram(self.comp.impact_x_idx),
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

    def update_sf(self, batch, priority_weights):
        (s, f, a, _, s_next, dones) = batch

        curr_sf1, curr_sf2 = self.sf(s, a)  # [N, H, F] <-- [N, S], [N,A]
        target_sf = self.calc_target_sf(f, s_next, dones)  # [N, H, F]

        loss1 = (curr_sf1 - target_sf).pow(2)  # [N, H, F]
        loss2 = (curr_sf2 - target_sf).pow(2)  # [N, H, F]

        loss1 = loss1 * priority_weights.unsqueeze(
            1
        )  # [N, H, F] <-- [N, H, F] * [N,1,1]
        loss2 = loss2 * priority_weights.unsqueeze(
            1
        )  # [N, H, F] <-- [N, H, F] * [N,1,1]

        sf1_loss = torch.mean(loss1)
        sf2_loss = torch.mean(loss2)

        sf_loss = sf1_loss + sf2_loss

        self.sf_optimizer.zero_grad(set_to_none=True)
        sf_loss.backward()
        self.sf_optimizer.step()

        # TD errors for updating priority weights
        errors = torch.mean(torch.abs(curr_sf1.detach() - target_sf), (1, 2))

        # log means to monitor training.
        mean_sf1 = curr_sf1.detach().mean().item()

        return sf_loss.detach().item(), errors, mean_sf1

    def update_policy(self, batch, priority_weights):
        (s, f, a, r, s_next, dones) = batch

        a_heads, entropies, _ = self.policy(s)  # [N,H,A], [N, H, 1] <-- [N,S]

        qs = self.calc_qs_from_sf(s, a_heads)

        qs = qs.unsqueeze(2)  # [N,H,1]

        loss = -qs - self.alpha * entropies
        # [N, H, 1] <--  [N, H, 1], [N,1,1]
        loss = loss * priority_weights.unsqueeze(1)
        policy_loss = torch.mean(loss)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        return policy_loss.detach().item(), entropies

    def calc_entropy_loss(self, entropy, priority_weights):
        loss = self.log_alpha * (self.target_entropy - entropy).detach()
        # [N, H, 1] <--  [N, H, 1], [N,1,1]
        loss = loss * priority_weights.unsqueeze(1)
        entropy_loss = -torch.mean(loss)
        return entropy_loss

    def calc_qs_from_sf(self, s, a):
        s_tiled, a_tiled = pile_sa_pairs(s, a)
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]

        curr_sf1, curr_sf2 = self.sf(s_tiled, a_tiled)
        # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]

        curr_sf1 = torch.masked_select(curr_sf1, self.mask).view(
            self.mini_batch_size, self.n_heads, self.n_heads
        )
        # [N, Ha, F] <-- [NHa, Hsf, F]

        curr_sf2 = torch.masked_select(curr_sf2, self.mask).view(
            self.mini_batch_size, self.n_heads, self.n_heads
        )
        # [N, Ha, F] <-- [NHa, Hsf, F]

        q1 = torch.einsum(
            "ijk,kj->ij", curr_sf1, self.pseudo_w
        )  # [N,H]<-- [N,H,F]*[F, H]
        q2 = torch.einsum(
            "ijk,kj->ij", curr_sf2, self.pseudo_w
        )  # [N,H]<-- [N,H,F]*[F, H]
        if self.droprate > 0.0:
            qs = 0.5 * (q1 + q2)
        else:
            qs = torch.min(q1, q2)
        return qs

    def calc_target_sf(self, f, s_next, dones):
        _, _, a_next = self.policy(s_next)  # [N, H, 1],[N, H, A] <-- [N, S]

        with torch.no_grad():
            s_tiled, a_tiled = pile_sa_pairs(s_next, a_next)
            # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]

            next_sf1, next_sf2 = self.sf_target(s_tiled, a_tiled)
            # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]

            next_sf1 = torch.masked_select(next_sf1, self.mask).view(
                self.mini_batch_size, self.n_heads, self.n_heads
            )
            # [N, Ha, F] <-- [NHa, Hsf, F]

            next_sf2 = torch.masked_select(next_sf2, self.mask).view(
                self.mini_batch_size, self.n_heads, self.n_heads
            )
            # [N, Ha, F] <-- [NHa, Hsf, F]

            next_sf = torch.min(next_sf1, next_sf2)  # [N, H, F]

        f = torch.tile(f[:, None, :], (self.n_heads, 1))  # [N,H,F] <-- [N,F]
        target_sf = f + torch.einsum(
            "ijk,il->ijk", next_sf, (~dones) * self.gamma
        )  # [N, H, F] <-- [N, H, F]+ [N, H, F]

        return target_sf  # [N, H, F]

    def calc_priority_error(self, batch):
        (s, f, a, _, s_next, dones) = batch

        with torch.no_grad():
            curr_sf1, curr_sf2 = self.sf(s, a)
        target_sf = self.calc_target_sf(f, s_next, dones)
        error = torch.mean(torch.abs(curr_sf1 - target_sf), (1, 2))
        return error.unsqueeze(1).cpu().numpy()

    def save_torch_model(self):
        path = self.log_path + f"model{self.episodes}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        self.policy.save(path + "policy")
        self.sf.SF1.save(path + "sf1")
        self.sf.SF2.save(path + "sf2")
        print(f"save torch model in {path}")

    def load_torch_model(self, path):
        self.policy.load(path + "policy")
        self.sf.SF1.load(path + "sf1")
        self.sf.SF2.load(path + "sf2")
        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)
