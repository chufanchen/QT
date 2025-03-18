import numpy as np

import math

import torch.nn.functional as F
from torch import distributions as pyd
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Sequence
from torch.distributions import Normal, TanhTransform, TransformedDistribution

# Base sequence model

class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
    
    
# Building Blocks

def miniblock(
    inp: int,
    oup: int,
    norm_layer: Optional[Callable[[int], nn.modules.Module]],
    relu=True
) -> List[nn.modules.Module]:
    """Construct a miniblock with given input/output-size and norm layer."""
    ret: List[nn.modules.Module] = [nn.Linear(inp, oup)]
    if norm_layer is not None:
        ret += [norm_layer(oup)]
    if relu:
        ret += [nn.ReLU(inplace=True)]
    else:
        ret += [nn.Tanh()]
    return ret

class Net(nn.Module):
    """Simple MLP backbone.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape.
    :param bool dueling: whether to use dueling network to calculate Q values
        (for Dueling DQN), defaults to False.
    :param norm_layer: use which normalization before ReLU, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``, defaults to None.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: tuple,
        action_shape: Optional[Union[tuple, int]] = 0,
        softmax: bool = False,
        concat: bool = False,
        hidden_layer_size: int = 128,
        output_shape: int = 0, 
        dueling: Optional[Tuple[int, int]] = None,
        norm_layer: Optional[Callable[[int], nn.modules.Module]] = None,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.softmax = softmax
        self.output_shape = output_shape
        input_size = np.prod(state_shape)
        if concat:
            input_size += np.prod(action_shape)

        model = miniblock(input_size, hidden_layer_size, norm_layer)

        for i in range(layer_num):
            model += miniblock(
                hidden_layer_size, hidden_layer_size, norm_layer)
            
        if dueling is None:
            if action_shape and not concat:
                model += [nn.Linear(hidden_layer_size, np.prod(action_shape))]
        else:  # dueling DQN
            q_layer_num, v_layer_num = dueling
            Q, V = [], []

            for i in range(q_layer_num):
                Q += miniblock(
                    hidden_layer_size, hidden_layer_size, norm_layer)
            for i in range(v_layer_num):
                V += miniblock(
                    hidden_layer_size, hidden_layer_size, norm_layer)

            if action_shape and not concat:
                Q += [nn.Linear(hidden_layer_size, np.prod(action_shape))]
                V += [nn.Linear(hidden_layer_size, 1)]

            self.Q = nn.Sequential(*Q)
            self.V = nn.Sequential(*V)
            
        if self.output_shape:
            model +=  [nn.Linear(hidden_layer_size, output_shape)]
        self.model = nn.Sequential(*model)
        
        self.s_mean, self.s_std = 0, 1

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten -> logits."""

        s = s.reshape(s.size(0), -1)
        try:
            s = (s - self.s_mean.to(s.device)) / self.s_std.to(s.device)
        except:
            pass
        logits = self.model(s)
        if self.dueling is not None:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            logits = q - q.mean(dim=1, keepdim=True) + v
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    
    @property
    def mode(self):
        return self.mean

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        std = 0.01
        return SquashedNormal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)
