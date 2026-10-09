"""The rough component has fractal dimension 1 + s (Higuchi estimate)."""

from __future__ import annotations

import numpy as np
import pytest

import razladka
from razladka._roughness import LAMBDA, fill_weierstrass, n_terms


def higuchi_dimension(x: np.ndarray, kmax: int = 8) -> float:
    """Higuchi's (1988) estimate of the fractal dimension of a curve."""
    n = x.size
    ks = np.arange(1, kmax + 1)
    lengths = []
    for k in ks:
        per_offset = []
        for m in range(k):
            sub = x[m::k]
            norm = (n - 1) / ((sub.size - 1) * k)
            per_offset.append(np.abs(np.diff(sub)).sum() * norm / k)
        lengths.append(np.mean(per_offset))
    slope = np.polyfit(np.log(ks), np.log(lengths), 1)[0]
    return float(-slope)


def weierstrass(n: int, s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    peak = fill_weierstrass(out, s, rng.uniform(0, 2 * np.pi, n_terms(n)))
    return out / peak


# Higuchi's method is biased low near D = 2 on sampled curves: for s = 0.8 it
# returns about 0.69 whatever n or kmax, hence the wider tolerance there.
@pytest.mark.parametrize(("s", "tolerance"), [(0.2, 0.1), (0.4, 0.1), (0.6, 0.1), (0.8, 0.15)])
def test_dimension(s: float, tolerance: float) -> None:
    estimates = [higuchi_dimension(weierstrass(16384, s, seed)) for seed in range(3)]
    assert np.mean(estimates) == pytest.approx(1 + s, abs=tolerance)


def test_dimension_grows_with_roughness() -> None:
    estimates = [higuchi_dimension(weierstrass(4096, s, 0)) for s in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert np.all(np.diff(estimates) > 0)


def test_blocks_match_direct_sum() -> None:
    """The blocked matrix evaluation equals the textbook sum of cosines."""
    n = 40_000  # spans several chunks and a partial block
    rng = np.random.default_rng(3)
    phases = rng.uniform(0, 2 * np.pi, n_terms(n))
    fast = np.empty(n)
    fill_weierstrass(fast, 0.5, phases)
    i = np.arange(n)
    direct = np.zeros(n)
    for k, phi in enumerate(phases):
        direct += LAMBDA ** (-k * 0.5) * np.cos(2 * np.pi * LAMBDA**k * i / n + phi)
    assert np.max(np.abs(fast - direct)) < 1e-9


def test_terms_reach_nyquist() -> None:
    for n in (2, 3, 10, 1000, 10**8):
        m = n_terms(n) - 1
        assert LAMBDA**m / n <= 0.5
        assert LAMBDA ** (m + 1) / n > 0.5 or n == 2


def test_roughness_amplitude() -> None:
    """The rough part reaches s * roughness_scale * (upper - lower) exactly."""
    kwargs = {"start": "2026-01-01", "end": "2026-02-01", "lower": 0.0, "upper": 10.0, "seed": 9}
    smooth = razladka.generate(5000, 2, roughness=0.0, **kwargs)
    s, scale = 0.6, 0.2
    rough = razladka.generate(5000, 2, roughness=s, roughness_scale=scale, **kwargs)
    amplitude = s * scale * 10.0
    # same seed, same change points; the skeleton of the rough series sits in a
    # narrower band, so compare the deviation from its own rescaled skeleton
    assert np.array_equal(smooth.change_points, rough.change_points)
    skeleton = amplitude + smooth.values * (10.0 - 2 * amplitude) / 10.0
    deviation = rough.values - skeleton
    assert np.max(np.abs(deviation)) == pytest.approx(amplitude, rel=1e-9)
