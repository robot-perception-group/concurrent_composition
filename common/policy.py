import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import get_sa_pairs
from .distribution import GaussianMixture
import numpy as np
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPS = 1e-2


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, obs):
        raise NotImplementedError


class StochasticPolicy(BaseNetwork):
    """Stochastic NN policy"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        squash=True,
        layernorm=False,
        activation=nn.SiLU,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash

        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], self.action_dim)
        self.activ = activation()
        self.tanh = nn.Tanh()

        self.layernorm = layernorm
        if self.layernorm:
            self.ln1 = nn.LayerNorm(self.sizes[0])
            self.ln2 = nn.LayerNorm(self.sizes[1])

        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.fc3.weight, 0.001)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, 1)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        if self.layernorm:
            x = self.activ(self.ln1(self.fc1(x)))
            x = self.activ(self.ln2(self.fc2(x)))
        else:
            x = self.activ(self.fc1(x))
            x = self.activ(self.fc2(x))

        x = self.fc3(x)
        x = self.tanh(x) if self.squash else x
        return x

    def sample(self, obs):
        acts = self.get_actions(obs)
        a = acts[:, 0, :]  # TODO: add some randomness to explorative action
        return a, 0, a  # TODO: compute entropy

    def get_action(self, obs):
        return self.get_actions(obs).squeeze(0)[0]

    def get_actions(self, obs, n_act=1):
        obs, n_obs = self.check_input(obs)

        latent_shape = (n_act, self.action_dim)
        latents = torch.normal(0, 1, size=latent_shape).to(device)

        s, a = get_sa_pairs(obs, latents)
        raw_actions = self.forward(torch.cat([s, a], -1)).view(
            n_obs, n_act, self.action_dim
        )

        return raw_actions

    def check_input(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(device)

        if obs.ndim > 1:
            n_obs = obs.shape[0]
        else:
            obs = obs[None, :]
            n_obs = 1
        return obs, n_obs


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[256, 256],
        squash=True,
        activation=nn.SiLU,
        layernorm=False,
    ):
        super(GaussianPolicy, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash

        self.fc1 = nn.Linear(self.observation_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], self.action_dim * 2)
        self.activ = activation()
        self.tanh = nn.Tanh()

        self.layernorm = layernorm
        if self.layernorm:
            self.ln1 = nn.LayerNorm(self.sizes[0])
            self.ln2 = nn.LayerNorm(self.sizes[1])

        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.fc3.weight, 0.001)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, 1)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, obs):
        x = self._forward_hidden(obs)
        means, log_stds = self.calc_mean_std(x)

        normals, xs, actions = self.get_distribution(means, log_stds)
        entropy = self.calc_entropy(normals, xs, actions)
        return actions, entropy, means

    def _forward_hidden(self, obs):
        if self.layernorm:
            x = self.activ(self.ln1(self.fc1(obs)))
            x = self.activ(self.ln2(self.fc2(x)))
        else:
            x = self.activ(self.fc1(obs))
            x = self.activ(self.fc2(x))

        x = self.fc3(x)
        x = self.tanh(x) if self.squash else x
        return x

    def calc_mean_std(self, x):
        means, log_stds = torch.chunk(x, 2, dim=-1)
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return means, log_stds

    def get_distribution(self, means, log_stds):
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        return normals, xs, actions

    def calc_entropy(self, normals, xs, actions, dim=1):
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropy = -log_probs.sum(dim=dim, keepdim=True)
        return entropy

    def sample(self, obs):
        return self.forward(obs)


class MultiheadGaussianPolicy(GaussianPolicy):
    def __init__(
        self,
        observation_dim,
        action_dim,
        n_heads,
        sizes=[256, 256],
        squash=True,
        activation=nn.SiLU,
        layernorm=False,
    ):
        super().__init__(
            observation_dim, action_dim, sizes, squash, activation, layernorm
        )
        self.n_heads = n_heads

        self.fc1 = nn.Linear(self.observation_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], 2 * self.action_dim * n_heads)

        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.fc3.weight, 0.001)

    def forward(self, obs):
        means, log_stds = self.get_mean_std(obs)  # [N, 2*H*A]
        normals, xs, actions = self.get_distribution(means, log_stds)
        entropies = self.calc_entropy(normals, xs, actions, dim=2)  # [N, H, 1]
        return actions, entropies, means  # [N, H, A], [N, H, 1], [N, H, A]

    def get_mean_std(self, obs):
        x = self._forward_hidden(obs)  # [N, 2*H*A]
        means, log_stds = self.calc_mean_std(x)  # [N, H, A]
        return means, log_stds

    def calc_mean_std(self, x):
        means, log_stds = torch.chunk(x, 2, dim=-1)  # [N, H*A], [N, H*A] <-- [N, H*2A]
        means = means.view([-1, self.n_heads, self.action_dim])  # [N, H, A]
        log_stds = log_stds.view([-1, self.n_heads, self.action_dim])  # [N, H, A]
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return means, log_stds

    def forward_head(self, obs, head_idx):
        actions, entropies, means = self.forward(obs)
        return actions[:, head_idx, :], entropies, means[:, head_idx, :]


class GaussianMixturePolicy(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        n_gauss=10,
        reg=0.001,
        reparameterize=True,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.n_gauss = n_gauss
        self.reg = reg
        self.reparameterize = reparameterize

        self.model = GaussianMixture(
            input_dim=self.observation_dim,
            output_dim=self.action_dim,
            hidden_layers_sizes=sizes,
            K=n_gauss,
            reg=reg,
            reparameterize=reparameterize,
        )

    def forward(self, obs):
        act, logp, mean = self.model(obs)
        act = torch.tanh(act)
        mean = torch.tanh(mean)
        logp -= self.squash_correction(act)
        entropy = -logp[:, None].sum(
            dim=1, keepdim=True
        )  # TODO: why [:, None] then sum(dim=1)?
        return act, entropy, mean

    def squash_correction(self, inp):
        return torch.sum(torch.log(1 - torch.tanh(inp) ** 2 + EPS), 1)

    def reg_loss(self):
        return self.model.reg_loss_t
