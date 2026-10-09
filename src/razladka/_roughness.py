"""Roughness: a Weierstrass-Mandelbrot sum with a given fractal dimension.

``W(i) = sum_{k=0}^{M} lambda^(-k H) cos(2 pi lambda^k i / n + phi_k)`` with
``H = 1 - s`` (Hölder exponent), ``D = 1 + s`` (fractal dimension of the
graph), ``lambda = 1.5`` and ``M = floor(log_lambda(n / 2))``, i.e. the
frequencies run from one cycle over the whole series up to the Nyquist
frequency.

The sum is evaluated in blocks with matrix products: for a chunk of ``C``
points starting at ``a``, ``cos(theta_k t + psi_k)`` with
``psi_k = theta_k a + phi_k`` expands into ``cos(theta_k t) cos(psi_k) -
sin(theta_k t) sin(psi_k)``, so one ``C x 2(M+1)`` table of cosines and
sines serves every chunk. Memory stays at a few tens of megabytes for any
``n``.
"""

from __future__ import annotations

import math

import numpy as np

LAMBDA = 1.5
"""Frequency ratio of neighbouring terms."""

_CHUNK = 1 << 14  # points per chunk (rows of the table)
_BLOCK = 64  # chunks per matrix product


def n_terms(n_points: int) -> int:
    """Number of terms ``M + 1``.

    >>> n_terms(1000)
    16
    """
    return max(0, math.floor(math.log(n_points / 2) / math.log(LAMBDA))) + 1


def fill_weierstrass(out: np.ndarray, roughness: float, phases: np.ndarray) -> float:
    """Writes the raw sum ``W(i)`` into ``out`` and returns ``max |W|``."""
    n = out.size
    m = phases.size
    k = np.arange(m, dtype=np.float64)
    hurst = 1.0 - roughness
    amplitude = LAMBDA ** (-k * hurst)
    cycles = LAMBDA**k / n  # cycles per step, at most 1/2
    c = min(n, _CHUNK)
    angle = 2 * np.pi * np.outer(np.arange(c, dtype=np.float64), cycles)
    table = np.concatenate([np.cos(angle), np.sin(angle)], axis=1)  # c x 2m
    del angle
    n_chunks = -(-n // c)
    peak = 0.0
    for first in range(0, n_chunks, _BLOCK):
        chunks = np.arange(first, min(n_chunks, first + _BLOCK), dtype=np.float64)
        # phase of each term at the first point of each chunk; the fractional
        # number of cycles keeps the argument small and exact enough
        turns = np.mod(np.outer(cycles, chunks * c), 1.0)
        psi = 2 * np.pi * turns + phases[:, None]
        coef = np.concatenate(
            [amplitude[:, None] * np.cos(psi), -amplitude[:, None] * np.sin(psi)], axis=0
        )  # 2m x chunks
        block = (table @ coef).T.reshape(-1)  # chunk-major
        start = first * c
        stop = min(n, start + block.size)
        piece = block[: stop - start]
        out[start:stop] = piece
        peak = max(peak, float(np.abs(piece).max()))
    return peak
