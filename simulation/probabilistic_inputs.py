from __future__ import annotations

import numpy as np


def normal_variation(base: float, sd: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw values from a normal distribution around a baseline."""
    return rng.normal(loc=base, scale=sd, size=size)


def poisson_stress_events(rate_per_hour: float, duration_hours: float, rng: np.random.Generator) -> int:
    """Generate the number of random stress events using a Poisson model."""
    lam = max(0.0, rate_per_hour * duration_hours)
    return int(rng.poisson(lam=lam))


def lognormal_exercise_intensity(mu: float, sigma: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample exercise intensity from a log-normal distribution."""
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def bounded(value: np.ndarray | float, lower: float | None = None, upper: float | None = None):
    """Clip a value or array into a physiologic range."""
    return np.clip(value, a_min=lower, a_max=upper)
