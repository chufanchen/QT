import torch

EPS = 1e-8  # Epsilon for avoiding numerical issues.
CLIP_EPS = 1e-3  # Epsilon for clipping actions.


def soft_relu(x):
    """Compute log(1 + exp(x)) in a numerically stable way."""
    # Note: log(sigmoid(x)) = x - soft_relu(x) = - soft_relu(-x).
    #       log(1 - sigmoid(x)) = - soft_relu(x)
    return torch.log(1.0 + torch.exp(-torch.abs(x))) + torch.maximum(
        x, torch.zeros_like(x)
    )


def clip_by_eps(x, spec, eps=0.0):
    return torch.clamp(x, min=torch.tensor(spec.low).to(x.device) + eps, max=torch.tensor(spec.high).to(x.device) - eps)


def gradient_penalty(s, a_p, a_b, c_fn, gamma=5.0):
    """Calculates interpolated gradient penalty."""
    batch_size = s.shape[0]
    alpha = torch.rand(batch_size, device=s.device)
    a_intpl = a_p + alpha.unsqueeze(1) * (a_b - a_p)
    a_intpl.requires_grad_(True)
    c_intpl = c_fn(s, a_intpl)
    grad = torch.autograd.grad(
        outputs=c_intpl,
        inputs=a_intpl,
        grad_outputs=torch.ones_like(c_intpl),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    slope = torch.sqrt(EPS + torch.sum(torch.square(grad), dim=-1))
    grad_penalty = torch.mean(
        torch.square(torch.maximum(slope - 1.0, torch.zeros_like(slope)))
    )
    return grad_penalty * gamma


class Divergence(object):
    """Basic interface for divergence."""

    def dual_estimate(self, s, a_p, a_b, c_fn):
        raise NotImplementedError

    def dual_critic_loss(self, s, a_p, a_b, c_fn):
        return -torch.mean(self.dual_estimate(s, a_p, a_b, c_fn)) + gradient_penalty(
            s, a_p, a_b, c_fn
        )

    def primal_estimate(self, p_fn, b_fn, n_samples, action_spec=None):
        raise NotImplementedError


class FDivergence(Divergence):
    """Interface for f-divergence."""

    def dual_estimate(self, s, a_p, a_b, c_fn):
        logits_p = c_fn(s, a_p)
        logits_b = c_fn(s, a_b)
        return self._dual_estimate_with_logits(logits_p, logits_b)

    def _dual_estimate_with_logits(self, logits_p, logits_b):
        raise NotImplementedError

    def primal_estimate(self, p_fn, b_fn, n_samples, action_spec=None):
        _, apn, apn_logp = p_fn.sample_n(n_samples)
        _, abn, abn_logb = b_fn.sample_n(n_samples)
        # Clip actions here to avoid numerical issues.
        apn_logb = b_fn.log_prob(clip_by_eps(apn, action_spec, CLIP_EPS))
        abn_logp = p_fn.log_prob(clip_by_eps(abn, action_spec, CLIP_EPS))
        return self._primal_estimate_with_densities(
            apn_logp, apn_logb, abn_logp, abn_logb
        )

    def _primal_estimate_with_densities(self, apn_logp, apn_logb, abn_logp, abn_logb):
        raise NotImplementedError


class KL(FDivergence):
    """KL divergence."""

    def _dual_estimate_with_logits(self, logits_p, logits_b):
        return -soft_relu(logits_b) + torch.log(soft_relu(logits_p) + EPS) + 1.0

    def _primal_estimate_with_densities(self, apn_logp, apn_logb, abn_logp, abn_logb):
        return torch.mean(apn_logp - apn_logb, axis=0)


class W(FDivergence):
    """Wasserstein distance."""

    def _dual_estimate_with_logits(self, logits_p, logits_b):
        return logits_p - logits_b


def laplacian_kernel(x1, x2, sigma=20.0):
    """Computes the Laplacian kernel between two sets of vectors.

    Args:
        x1: First set of vectors
        x2: Second set of vectors
        sigma: Kernel bandwidth parameter

    Returns:
        Kernel matrix of shape [x1.shape[0], x2.shape[0]]
    """
    # Computing pairwise L1 distances between vectors in x1 and x2
    d12 = torch.sum(torch.abs(x1.unsqueeze(1) - x2.unsqueeze(0)), dim=-1)
    # Apply exponential kernel
    k12 = torch.exp(-d12 / sigma)
    return k12


def mmd(x1, x2, kernel, use_sqrt=False):
    """Calculates the Maximum Mean Discrepancy between two sets of samples.

    Args:
        x1: First set of samples
        x2: Second set of samples
        kernel: Kernel function to use (e.g., laplacian_kernel)
        use_sqrt: Whether to return sqrt of MMD

    Returns:
        MMD value (scalar)
    """
    k11 = torch.mean(kernel(x1, x1), dim=[0, 1])
    k12 = torch.mean(kernel(x1, x2), dim=[0, 1])
    k22 = torch.mean(kernel(x2, x2), dim=[0, 1])

    if use_sqrt:
        return torch.sqrt(k11 + k22 - 2 * k12 + EPS)
    else:
        return k11 + k22 - 2 * k12


class MMD(Divergence):
    """MMD."""

    def primal_estimate(
        self, p_fn, b_fn, n_samples, kernel=laplacian_kernel, action_spec=None
    ):
        apn = p_fn.sample_n(n_samples)[1]
        abn = b_fn.sample_n(n_samples)[1]
        return mmd(apn, abn, kernel)
