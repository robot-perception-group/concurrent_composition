import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H_min = torch.tensor(1e-3).unsqueeze(0).to(device)


def adaptive_isotropic_gaussian_kernel(xs, ys):
    """Gaussian kernel with dynamic bandwidth

    the bandwidth is adjusted dynamically to match median_distance / log(Kx)

    a pytorch implementation of
    https://github.com/haarnoja/softqlearning/blob/6f51eaca77d15b35c6443363c51a5a53ff4e9854/softqlearning/misc/kernel.py#L7

    Args:
        xs(`tf.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
            particles of dimension D. This is the first kernel argument.
        ys(`tf.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
            particles of dimension D. This is the second kernel argument.
        h_min(`float`): Minimum bandwidth.
    Returns:
        `dict`: Returned dictionary has two fields:
            'output': A `tf.Tensor` object of shape (N x Kx x Ky) representing
                the kernel matrix for inputs `xs` and `ys`.
            'gradient': A 'tf.Tensor` object of shape (N x Kx x Ky x D)
                representing the gradient of the kernel with respect to `xs`.

    Reference:
        [2] Qiang Liu,Dilin Wang, "Stein Variational Gradient Descent: A General
            Purpose Bayesian Inference Algorithm," Neural Information Processing
            Systems (NIPS), 2016.
    """
    Kx, D = xs.shape[-2:]
    Ky, D2 = ys.shape[-2:]
    assert D == D2

    leading_shape = xs.shape[:-2]

    # Compute the pairwise distance of left and right particles.
    diff = xs.unsqueeze(-2) - ys.unsqueeze(-3)
    # ... x Kx x Ky x D

    dist_sq = torch.sum(diff**2, dim=-1, keepdims=False)

    # get median
    input_shape = (leading_shape[0], Kx * Ky)
    values, _ = torch.topk(
        input=dist_sq.view(input_shape),
        k=(Kx * Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
        sorted=True,  # ... x floor(Ks*Kd/2)
    )

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)

    h = medians_sq / np.log(Kx)  # ... (shape)
    h = torch.max(torch.cat((h, H_min))).detach()
    h_expanded_twice = h[..., None, None]
    # ... x 1 x 1

    kappa = torch.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky

    # construct the gradient
    h_expanded_thrice = h_expanded_twice.unsqueeze(-1)
    # ... x 1 x 1 x 1
    kappa_expanded = kappa.unsqueeze(-1)  # ... x Kx x Ky x 1

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    # ... x Kx x Ky x D

    return {"output": kappa, "gradient": kappa_grad}
