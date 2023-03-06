import numpy as np
import torch
from torch.optim import Adam

import wandb
from rl.rl_torch.common.agent import RaisimAgent
from rl.rl_torch.common.kernel import adaptive_isotropic_gaussian_kernel
from rl.rl_torch.common.policy import StochasticPolicy
from rl.rl_torch.common.util import (
    assert_shape,
    get_sa_pairs,
    grad_false,
    hard_update,
    pile_sa_pairs,
    soft_update,
    update_params,
)
from rl.rl_torch.common.value_function import TwinnedQNetwork

EPS = 1e-6


class SQLAgent(RaisimAgent):
    """SQL
    Tuomas Haarnoja, Reinforcement Learning with Deep Energy-Based Policies
    see https://github.com/haarnoja/softqlearning
    """

    @classmethod
    def default_config(cls):
        default_cfg = super().default_config()
        env_cfg = default_cfg["env_cfg"]
        agent_cfg = default_cfg["agent_cfg"]
        buffer_cfg = default_cfg["buffer_cfg"]

        env_cfg.update(
            dict(
                env_name="pointmass1d",
                num_envs=200,
                episode_max_step=200,
                total_episodes=int(100),
                eval_interval=10,
                single_task=False,
                random_robot_state=True,
                random_target_state=True,
                num_threads=10,
                save_model=True,
                render=False,  # render env
                record=True,  # dump config, record video only works if env is rendered
            )
        )
        agent_cfg.update(
            dict(
                name="sql",
                lr=0.008961,
                n_value_particles=64,
                tau=5e-3,
                td_target_update_interval=1,
                gamma=0.99,
                policy_lr=0.0002,
                n_kernel_particles=64,
                kernel_update_ratio=0.5,
                alpha=0.2,
                updates_per_step=3,
                reward_scale=1.0,
                grad_clip=None,
                value_net_kwargs={
                    "sizes": [32, 32],
                    "activation": "selu",
                    "layernorm": True,
                    "droprate": 0.01,
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
                multi_step=9,
                n_env=env_cfg["num_envs"],
                prioritize_replay=True,
            )
        )
        return {"env_cfg": env_cfg, "agent_cfg": agent_cfg, "buffer_cfg": buffer_cfg}

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lr = self.agent_cfg["lr"]
        self.n_value_particles = int(self.agent_cfg["n_value_particles"])
        self.value_net_kwargs = self.agent_cfg["value_net_kwargs"]
        self.tau = self.agent_cfg["tau"]
        self.td_target_update_interval = int(
            self.agent_cfg["td_target_update_interval"]
        )
        self.gamma = self.agent_cfg["gamma"]

        self.policy_lr = self.agent_cfg["policy_lr"]
        self.policy_net_kwargs = self.agent_cfg["policy_net_kwargs"]
        self.kernel_fn = adaptive_isotropic_gaussian_kernel
        self.kernel_update_ratio = self.agent_cfg["kernel_update_ratio"]
        self.n_kernel_particles = int(self.agent_cfg["n_kernel_particles"])

        self.updates_per_step = self.agent_cfg["updates_per_step"]
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.alpha = torch.tensor(self.agent_cfg["alpha"]).to(self.device)
        self.droprate = self.value_net_kwargs["droprate"]

        self.critic = TwinnedQNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            **self.value_net_kwargs,
        ).to(self.device)

        self.critic_target = TwinnedQNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            **self.value_net_kwargs,
        ).to(self.device)
        if self.droprate <= 0.0:
            self.critic_target = self.critic_target.eval()

        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)

        self.policy = StochasticPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            **self.policy_net_kwargs,
        ).to(self.device)

        self.q1_optimizer = Adam(self.critic.Q1.parameters(), lr=self.lr)
        self.q2_optimizer = Adam(self.critic.Q2.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)

        self.learn_steps = 0

    def explore(self, s, w):  # act with randomness
        with torch.no_grad():
            a, _, _ = self.policy.sample(s)
        return a

    def exploit(self, s, w):  # act without randomness
        with torch.no_grad():
            _, _, a = self.policy.sample(s)
        return a

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.per:
            batch, indices, weights = self.replay_buffer.sample(self.mini_batch_size)
        else:
            batch = self.replay_buffer.sample(self.mini_batch_size)
            weights = 1

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(
            batch, weights
        )
        policy_loss = self.calc_policy_loss(batch, weights)

        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)
        update_params(self.q1_optimizer, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(self.q2_optimizer, self.critic.Q2, q2_loss, self.grad_clip)

        if self.per:
            self.replay_buffer.update_priority(indices, errors.cpu().numpy())

        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/Q1": q1_loss.detach().item(),
                "loss/Q2": q2_loss.detach().item(),
                "loss/policy": policy_loss.detach().item(),
                "state/mean_Q1": mean_q1,
                "state/mean_Q2": mean_q2,
            }
            wandb.log(metrics)

    def calc_critic_loss(self, batch, weights):
        (s, f, a, r, s_next, dones) = batch

        curr_q1, curr_q2 = self.calc_current_q(s, a)
        target_q = self.calc_target_q(r, s_next, dones)

        # Critic loss is mean squared TD errors.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        """Create a minimization operation for policy update (SVGD)."""
        (s, f, a, r, s_next, dones) = batch

        a_fixed, a_updated, n_fixed_a, n_updated_a = self.get_fix_update_a(s)

        grad_log_p = self.calc_grad_log_p(s, a_fixed)
        assert_shape(grad_log_p, [None, n_fixed_a, 1, self.action_dim])

        # kernel function in Equation 13:
        kernel_dict = self.kernel_fn(xs=a_fixed, ys=a_updated)
        kappa = kernel_dict["output"].unsqueeze(3)
        assert_shape(kappa, [None, n_fixed_a, n_updated_a, 1])

        # stein variational gradient in equation 13:
        action_gradients = torch.mean(
            kappa * grad_log_p + self.alpha * kernel_dict["gradient"], dim=1
        )
        assert_shape(action_gradients, [None, n_updated_a, self.action_dim])

        # Propagate the gradient through the policy network (Equation 14). action_gradients * df_{\phi}(.,s)/d(\phi)
        gradients = torch.autograd.grad(
            a_updated,
            self.policy.parameters(),
            grad_outputs=action_gradients,
        )

        # multiply weight since optimizer will differentiate it later so we can apply the gradient
        surrogate_loss = torch.sum(
            torch.stack(
                [
                    torch.sum(w * g.detach())
                    for w, g in zip(self.policy.parameters(), gradients)
                ],
                dim=0,
            )
        )
        assert surrogate_loss.requires_grad == True

        loss = -surrogate_loss
        return loss

    def calc_current_q(self, s, a):
        curr_q1, curr_q2 = self.critic(s, a)
        assert_shape(curr_q1, [None, 1])
        return curr_q1, curr_q2

    def calc_target_q(self, r, s_next, dones):
        with torch.no_grad():
            next_q = self.calc_next_q(s_next)

            # eqn10
            next_v = torch.logsumexp(next_q, 1)
            assert_shape(next_v, [None])

            # Importance weights add just a constant to the value.
            next_v -= np.log(self.n_value_particles)
            next_v += self.action_dim * np.log(2)
            next_v = next_v.unsqueeze(1)

        # eqn11: \hat Q(s,a)
        target_q = r + (1.0 - dones) * self.gamma * next_v
        assert_shape(target_q, [None, 1])

        return target_q

    def calc_next_q(self, s_next):
        # eqn10: sampling a for importance sampling
        a_rsampled = (
            torch.distributions.uniform.Uniform(-1, 1)
            .sample((self.n_value_particles, self.action_dim))
            .to(self.device)
        )

        # eqn10: Q(s,a)
        s, a = get_sa_pairs(s_next, a_rsampled)
        next_q1, next_q2 = self.critic_target(s, a)

        n_sample = s_next.shape[0]
        next_q1, next_q2 = next_q1.view(n_sample, -1), next_q2.view(n_sample, -1)
        next_q = torch.min(next_q1, next_q2)
        assert_shape(next_q, [None, self.n_value_particles])

        return next_q

    def get_fix_update_a(self, s):
        a_sampled = self.policy.get_actions(obs=s, n_act=self.n_kernel_particles)
        assert_shape(a_sampled, [None, self.n_kernel_particles, self.action_dim])

        n_updated_a = int(self.n_kernel_particles * self.kernel_update_ratio)
        n_fixed_a = self.n_kernel_particles - n_updated_a
        a_fixed, a_updated = torch.split(a_sampled, [n_fixed_a, n_updated_a], dim=1)
        assert_shape(a_fixed, [None, n_fixed_a, self.action_dim])
        assert_shape(a_updated, [None, n_updated_a, self.action_dim])
        return a_fixed, a_updated, n_fixed_a, n_updated_a

    def calc_grad_log_p(self, s, a_fixed):
        s_paired, a_paired = pile_sa_pairs(s, a_fixed)
        svgd_q1, svgd_q2 = self.critic(s_paired, a_paired)

        n_sample = s.shape[0]
        svgd_q1, svgd_q2 = svgd_q1.view(n_sample, -1), svgd_q2.view(n_sample, -1)
        if self.droprate > 0.0:
            svgd_q = 0.5 * (svgd_q1 + svgd_q2)
        else:
            svgd_q = torch.min(svgd_q1, svgd_q2)

        # Target log-density. Q_soft in Equation 13:
        squash_correction = torch.sum(torch.log(1 - a_fixed**2 + EPS), dim=-1)
        log_p = svgd_q + squash_correction

        grad_log_p = torch.autograd.grad(
            log_p, a_fixed, grad_outputs=torch.ones_like(log_p)
        )
        grad_log_p = grad_log_p[0].unsqueeze(2).detach()
        return grad_log_p

    def calc_priority_error(self, batch):
        (s, _, a, r, s_next, dones) = batch
        with torch.no_grad():
            curr_q1, curr_q2 = self.calc_current_q(s, a)
        target_q = self.calc_target_q(r, s_next, dones)
        error = torch.abs(curr_q1 - target_q).cpu().numpy()
        return error

    def save_torch_model(self):
        from pathlib import Path

        path = self.log_path + f"model{self.episodes}/"
        Path(path).mkdir(parents=True, exist_ok=True)
        self.policy.save(path + "policy")
        self.critic.Q1.save(path + "critic1")
        self.critic.Q2.save(path + "critic2")

    def load_torch_model(self, path):
        self.policy.load(path + "policy")
        self.critic.Q1.load(path + "critic1")
        self.critic.Q2.load(path + "critic2")
        hard_update(self.critic_target, self.critic)
        grad_false(self.critic_target)


if __name__ == "__main__":
    import pprint

    from rl.rl_torch.common.util import fix_config

    default_cfg = SQLAgent.default_config()
    wandb.init(config=default_cfg)
    cfg = fix_config(wandb.config)
    pprint.pprint(cfg)

    agent = SQLAgent(cfg)
    agent.run()
    wandb.finish()
