import numpy as np


def crps_ensemble(observation, forecasts):
    """
    This function is currently unused, but is equivalent to properscoring's crps_ensemble.

    If numba is unavailable, properscoring will fall back to a more naive algorithm that is very inefficient in time and memory.  Use this instead!
    """

    fc = forecasts.copy()
    fc.sort(axis=-1)
    obs = observation
    fc_below = fc < obs[..., None]
    crps = np.zeros_like(obs)

    for i in range(fc.shape[-1]):
        below = fc_below[..., i]
        weight = ((i + 1) ** 2 - i**2) / fc.shape[-1] ** 2
        crps[below] += weight * (obs[below] - fc[..., i][below])

    for i in range(fc.shape[-1] - 1, -1, -1):
        above = ~fc_below[..., i]
        k = fc.shape[-1] - 1 - i
        weight = ((k + 1) ** 2 - k**2) / fc.shape[-1] ** 2
        crps[above] += weight * (fc[..., i][above] - obs[above])

    return crps
