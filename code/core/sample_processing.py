import torch
from scipy.signal import lfilter


def calculate_discounted_returns(traj_data, return_discount):
    return [discounted_cumsum(r, discount=return_discount) for r in traj_data["rewards"]]


def calculate_gae_advantages(traj_data, return_discount, gae_discount, normalize_adv=False, positive_adv=False):
    """
    Processes sampled paths. This involves:
        - Estimating the advantages using GAE or other advantage estimator

    Args:
        traj_data (dict): A dict with trajectory data with observations, actions, log_probs, rewards and dones

    Returns:
        (dict): Processed trajectory data
    """
    assert 0 <= return_discount <= 1.0, 'return_discount factor must be in [0,1]'
    assert 0 <= gae_discount <= 1.0, 'gae_discount must be in [0,1]'
    assert traj_data.keys() >= {'rewards', 'baselines'}, f"Need rewards and baselines but got {traj_data.keys()}"

    # compute advantages and stack path data
    traj_data["advantages"] = [gae(rewards=r, baselines=b, gamma=return_discount, lam=gae_discount)
                               for r, b in zip(traj_data["rewards"], traj_data["baselines"])]

    if normalize_adv:
        traj_data["advantages"] = normalize_advantages(traj_data["advantages"])
    if positive_adv:
        traj_data["advantages"] = [shift_advantages_to_positive(a) for a in traj_data["advantages"]]
    return traj_data


def discounted_cumsum(x, discount):
    """
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

    Returns:
        (float) : y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    """
    x = x.numpy()
    filtered = lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return torch.tensor(filtered.copy(), dtype=torch.get_default_dtype())[:, None]


def gae(rewards, baselines, gamma=0.99, lam=0.95):
    baselines = baselines.squeeze(-1)
    deltas = rewards[:] - baselines[:]
    deltas[:-1] += gamma * baselines[1:]
    gae_advantage = discounted_cumsum(deltas, discount=gamma * lam)
    return gae_advantage


def shift_advantages_to_positive(advantages):
    return (advantages - torch.min(advantages)) + 1e-8


def normalize_advantages(
    advantages: list[torch.Tensor, ...],
    epsilon=1e-8
) -> list[torch.Tensor, ...]:
    cat_advs = torch.cat(advantages, dim=0)
    mean = cat_advs.mean()
    std = cat_advs.std(unbiased=False)
    return [(adv - mean) / (std + epsilon) for adv in advantages]
