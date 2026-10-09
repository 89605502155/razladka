"""Change points from a Markov regime-switching chain with a left-to-right
transition matrix (Ferubko & Kazakov, 2026, section 2, formula (4)).

The chain has ``K + 1`` regimes. From regime ``j`` it either stays in ``j``
or moves to ``j + 1``; the last regime is absorbing, so a realisation that
reaches it has exactly ``K`` change points. Each of the first ``K`` regimes
lasts ``m + Geom(q)`` steps: a minimum of ``m`` steps, after which the chain
leaves with probability ``q`` at every step. ``q`` is chosen so that the
expected duration is ``n / (K + 1)``. A realisation that does not reach the
last regime with at least ``m`` points left is discarded and drawn again.
"""

from __future__ import annotations

import numpy as np

MIN_SEGMENT = 2
"""Minimum number of points in a segment."""


def leave_probability(n_points: int, n_change_points: int) -> float:
    """Probability ``q`` of leaving a regime after its minimum duration.

    >>> leave_probability(1000, 4)
    0.005025125628140704
    """
    mean_extra = n_points / (n_change_points + 1) - MIN_SEGMENT
    return 1.0 if mean_extra <= 0 else 1.0 / (mean_extra + 1.0)


def draw_change_points(n_points: int, n_change_points: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of the first points of segments ``1 … K`` (int64, strictly increasing)."""
    k = n_change_points
    if k == 0:
        return np.empty(0, dtype=np.int64)
    q = leave_probability(n_points, k)
    limit = n_points - MIN_SEGMENT
    while True:
        # numpy's geometric counts trials (>= 1); minus one gives failures (>= 0)
        durations = MIN_SEGMENT + rng.geometric(q, size=k).astype(np.int64) - 1
        bounds = np.cumsum(durations)
        if bounds[-1] <= limit:
            return bounds
