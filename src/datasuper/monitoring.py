"""Small drift checks for logged prediction inputs."""

from __future__ import annotations

import numpy as np


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("PSI requires non-empty expected and actual samples")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("PSI inputs must be finite")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0 if np.allclose(expected[0], actual) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    expected_rate = np.histogram(expected, bins=edges)[0] / expected.size
    actual_rate = np.histogram(actual, bins=edges)[0] / actual.size
    expected_rate = np.clip(expected_rate, 1e-6, None)
    actual_rate = np.clip(actual_rate, 1e-6, None)
    return float(np.sum((actual_rate - expected_rate) * np.log(actual_rate / expected_rate)))
