import numpy as np


def pareto_front(rewards, minimization=False):
    """Return a boolean mask of non-dominated (Pareto) points for maximization.
    If maximiziation=False, then treat this as a minimization problem, and rewards represent costs

    rewards: array-like (N, d)
    """
    rewards = np.asarray(rewards)
    if minimization:
        rewards = -rewards
    if rewards.size == 0:
        return np.zeros((0,), dtype=bool)
    n = rewards.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i, r in enumerate(rewards):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(rewards[is_efficient] > r, axis=1)
            is_efficient[i] = True
    return is_efficient



