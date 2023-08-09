import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20
EPS = 1e-6


class GaussianMixture(torch.nn.Module):
    def __init__(
        self,
        K,
        input_dim,
        output_dim,
        hidden_layers_sizes=[64, 64],
        reg=0.001,
        reparameterize=True,
        activation=nn.SiLU,
    ):
        super().__init__()
        self._K = K
        self._input_dim = input_dim
        self._Dx = output_dim

        self._reg = reg
        self._layer_sizes = list(hidden_layers_sizes) + [self._K * (2 * self._Dx + 1)]
        self._reparameterize = reparameterize

        self.fc1 = nn.Linear(self._input_dim, self._layer_sizes[0])
        self.fc2 = nn.Linear(self._layer_sizes[0], self._layer_sizes[1])
        self.fc3 = nn.Linear(self._layer_sizes[1], self._layer_sizes[2])
        self.activ = activation()
        self.tanh = nn.Tanh()

        self._log_p_x_t = 0
        self._log_p_x_mono_t = 0
        self._reg_loss_t = 0
        self._x_t = 0
        self._mus_t = 0
        self._log_sigs_t = 0
        self._log_ws_t = 0
        self._N_pl = 0

        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.fc3.weight, 0.0001)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, 1)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_distribution(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, self._K, 2 * self._Dx + 1)

        log_w_t = x[..., 0]
        mu_t = x[..., 1 : 1 + self._Dx]
        log_sig_t = x[..., 1 + self._Dx :]
        log_sig_t = torch.clip(log_sig_t, LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
        return log_w_t, mu_t, log_sig_t

    @staticmethod
    def _create_log_gaussian(mu_t, log_sig_t, t):
        normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)  # ... x D
        quadratic = -0.5 * torch.sum(normalized_dist_t**2, -1)
        # ... x (None)

        log_z = torch.sum(log_sig_t, axis=-1)  # ... x (None)
        D_t = torch.tensor(mu_t.shape[-1], dtype=torch.float32)
        log_z += 0.5 * D_t * np.log(2 * np.pi)

        log_p = quadratic - log_z

        return log_p  # ... x (None)

    @staticmethod
    def _create_log_mono_gaussian(mu_t, log_sig_t, t):
        normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)  # ... x D
        quadratic = -0.5 * normalized_dist_t**2
        # ... x (D)

        log_z = log_sig_t  # ... x (D)
        D_t = torch.tensor(mu_t.shape[-1], dtype=torch.float32)
        log_z += 0.5 * D_t * np.log(2 * np.pi)

        log_p = quadratic - log_z

        return log_p  # ... x (D)

    def forward(self, obs):
        if obs.ndim > 1:
            N = obs.shape[0]
        else:
            obs = obs[None, :]
            N = 1

        Dx, K = self._Dx, torch.tensor(self._K).to(device)

        # create K gaussians
        log_ws_t, xz_mus_t, xz_log_sigs_t = self.get_distribution(obs)
        # (N x K), (N x K x Dx), (N x K x Dx)
        xz_sigs_t = torch.exp(xz_log_sigs_t)

        # Sample the latent code.
        log_ws_t = self.tanh(log_ws_t) + 1 + EPS  # me add this to make it logits
        z_t = torch.multinomial(log_ws_t, num_samples=1)  # N x 1

        # Choose mixture component corresponding to the latent.
        mask_t = F.one_hot(z_t[:, 0], K)
        mask_t = mask_t.bool()
        xz_mu_t = xz_mus_t[mask_t]  # N x Dx
        xz_sig_t = xz_sigs_t[mask_t]  # N x Dx

        # Sample x.
        x_t = xz_mu_t + xz_sig_t * torch.normal(0, 1, (N, Dx)).to(device)  # N x Dx

        if not self._reparameterize:
            x_t = x_t.detach()

        log_p_x_t = self.calc_log_p_x(log_ws_t, xz_mus_t, xz_log_sigs_t, x_t)

        reg_loss_t = 0
        reg_loss_t += self._reg * 0.5 * torch.mean(xz_log_sigs_t**2)
        reg_loss_t += self._reg * 0.5 * torch.mean(xz_mus_t**2)

        self._log_p_x_t = log_p_x_t
        self._reg_loss_t = reg_loss_t
        self._xz_mu_t = xz_mu_t
        self._x_t = x_t

        self._log_ws_t = log_ws_t
        self._mus_t = xz_mus_t
        self._mu_t = xz_mu_t
        self._log_sigs_t = xz_log_sigs_t

        return x_t, log_p_x_t, xz_mu_t

    def calc_log_p_x(self, log_ws_t, xz_mus_t, xz_log_sigs_t, x_t):
        # log p(x|z_k) = log N(x | mu_k, sig_k)
        log_p_xz_t = self._create_log_gaussian(
            xz_mus_t, xz_log_sigs_t, x_t[:, None, :]
        )  # N x K

        # log p(x) = log sum_k p(z_k)p(x|z_k)
        log_p_x_t = torch.logsumexp(log_p_xz_t + log_ws_t, axis=1)
        log_p_x_t = log_p_x_t - torch.logsumexp(log_ws_t, axis=1)  # N
        return log_p_x_t

    def calc_log_p_x_mono(self, log_ws_t, xz_mus_t, xz_log_sigs_t, x_t):
        # log p(x|z_k)
        log_p_xz_mono_t = self._create_log_mono_gaussian(
            xz_mus_t, xz_log_sigs_t, x_t[:, None, :]
        )  # N x K

        # log mono p(x)
        log_ws_mono_t = log_ws_t[..., None]
        log_p_x_mono_t = torch.logsumexp(log_p_xz_mono_t + log_ws_mono_t, axis=1)
        log_p_x_mono_t = log_p_x_mono_t - torch.logsumexp(log_ws_mono_t, axis=1)  # N
        return log_p_x_mono_t

    @property
    def log_p_x_mono_t(self):
        log_ws_t = self._log_ws_t
        xz_mus_t = self._mu_t
        xz_log_sigs_t = self._log_sigs_t
        x_t = self._x_t
        log_p_x_mono_t = self.calc_log_p_x_mono(log_ws_t, xz_mus_t, xz_log_sigs_t, x_t)
        return log_p_x_mono_t

    @property
    def log_p_t(self):
        return self._log_p_x_t

    @property
    def reg_loss_t(self):
        return self._reg_loss_t

    @property
    def x_t(self):
        return self._x_t

    @property
    def mus_t(self):
        return self._mus_t

    @property
    def mu_t(self):
        return self._mu_t

    @property
    def log_sigs_t(self):
        return self._log_sigs_t

    @property
    def log_ws_t(self):
        return self._log_ws_t

    @property
    def N_t(self):
        return self._N
