"""Score a naive change-point detector against the ground truth on many series."""

import numpy as np

import razladka


def detect(values: np.ndarray, k: int, window: int = 25) -> np.ndarray:
    """Positions where the smoothed slope changes sign most strongly."""
    slope = np.convolve(np.diff(values), np.ones(window) / window, mode="same")
    turns = np.abs(np.diff(np.sign(slope)))
    return np.sort(np.argsort(turns)[-k:]) + 1


def recall(truth: np.ndarray, found: np.ndarray, margin: int) -> float:
    if truth.size == 0:
        return 1.0
    return float(np.mean([np.any(np.abs(found - t) <= margin) for t in truth]))


for roughness in (0.0, 0.3, 0.6, 0.9):
    scores = []
    for seed in range(20):
        s = razladka.generate(
            2_000, 4, start="2026-01-01", end="2026-02-01", lower=0, upper=1,
            roughness=roughness, roughness_scale=0.3, seed=seed,
        )  # fmt: skip
        scores.append(recall(s.change_points, detect(s.values, 4), margin=20))
    print(f"roughness {roughness:.1f}: recall {np.mean(scores):.2f}")
