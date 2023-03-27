import torch
import numpy as np
from enum import Enum, auto
from torch import nn
from typing import Iterable, Callable
from gym.spaces import Discrete, Box
from collections import OrderedDict


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_simple_nn_model(input_size, hidden_layers, output_size, activation, bias=True):
    layers = []
    in_shape = input_size
    for i in range(len(hidden_layers)):
        out_shape = hidden_layers[i]
        layers.append(nn.Linear(in_shape, out_shape, bias=bias))
        layers.append(activation())
        in_shape = out_shape
    layers.append(nn.Linear(in_shape, output_size))
    return nn.Sequential(*layers)


def get_space_dim(space):
    if isinstance(space, Discrete):
        return 1
    elif isinstance(space, Box):
        return np.array(space.shape).prod()


def get_space_io_size(space):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        return np.array(space.shape).prod()


def normc_initializer(tensor: torch.Tensor, std=1.0):
    out = torch.rand_like(tensor)
    out *= std / torch.sqrt((out ** 2).sum(axis=0, keepdims=True))
    tensor.data = out
    return tensor


def linnear_anneal(min_val, max_val, epoch, max_epochs):
    return max_val * (max_epochs - epoch) / max_epochs + min_val * epoch / max_epochs


def cumul_reward(steps, discount=0.95, end_reward=2, per_step_reward=-0.1):
    from scipy.signal import lfilter
    x = np.ones(steps) * per_step_reward
    x[steps-1] = end_reward
    filtered = lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return filtered[0]


def categorical_entropy(probs=None, logits=None):
    if probs is None and logits is None:
        raise RuntimeError("Probs or logits must be specified")
    if probs is None:
        probs = torch.exp(logits)
    if logits is None:
        logits = torch.log(probs)
    p_log_p = logits * probs
    return -p_log_p.sum(-1)


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def weighted_logsumexp(input, dim, weights, keepdim=False, out=None):
    # if torch.any(weights == 0):
    #     input = input + 0.  # promote to at least float
    #     input[weights == 0] = float('-inf')
    max_vals = torch.max(input, dim=dim, keepdim=True)[0].detach()

    # if max_vals.ndim > 0:
    #     max_vals[~torch.isfinite(max_vals)] = 0
    # elif not torch.isfinite(max_vals):
    #     max_vals = 0

    tmp = weights * torch.exp(input - max_vals)
    out = torch.log(torch.sum(tmp, dim=dim, keepdim=keepdim))
    if not keepdim:
        max_vals = torch.squeeze(max_vals, dim=dim)
    out += max_vals
    return out


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple(t.reshape(-1) for t in tensors))


def unflatten_tensors(flattened, tensor_shapes):
    """Unflatten a flattened tensors into a list of tensors.

    Args:
        flattened (numpy.ndarray): Flattened tensors.
        tensor_shapes (tuple): Tensor shapes.

    Returns:
        list[numpy.ndarray]: Unflattened list of tensors.

    """
    tensor_sizes = list(map(np.prod, tensor_shapes))
    # indices = np.cumsum(tensor_sizes)[:-1]
    return tuple(torch.reshape(pair[0], pair[1]) for pair in zip(torch.split(flattened, tensor_sizes), tensor_shapes))

class GradUpdateType(Enum):
    META = auto()
    SINGLE = auto()
    MULTI = auto()

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return GradUpdateType[s.upper()]
        except KeyError:
            return s
