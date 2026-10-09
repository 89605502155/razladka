# razladka guide

[Русская версия](https://github.com/89605502155/razladka/blob/main/docs/guide.ru.md) ·
[API reference](https://github.com/89605502155/razladka/blob/main/docs/api.md)

## 1. The model

A series of `n` points with `K` change points has `K + 1` *segments*.
Segment `k` runs from index `change_points[k - 1]` (0 for `k = 0`) up to,
not including, `change_points[k]` (`n` for the last one). `segments[i]` is
the segment of point `i`.

### 1.1 Where the change points are: Markov regime switching

The regime `r_t ∈ {0, …, K}` is a Markov chain with a left-to-right
(upper bidiagonal) transition matrix

```
      ⎡ 1−q   q    0   …   0 ⎤
      ⎢  0   1−q   q   …   0 ⎥
P  =  ⎢  ⋮              ⋱   ⋮ ⎥
      ⎢  0    …    0  1−q  q ⎥
      ⎣  0    …    0   0   1 ⎦
```

with a minimum stay of `m = 2` points: regime `j < K` lasts
`d_j = m + G_j`, `G_j ~ Geom(q)` (failures before the first success). The
last regime is absorbing, so a chain that reaches it has exactly `K`
changes. `q` makes the expected stay `E d_j = n / (K + 1)`:

```
q = 1 / (n / (K + 1) − m + 1)        (q = 1 if n / (K + 1) ≤ m)
```

A realisation that does not reach the last regime with at least `m` points
left (`Σ d_j > n − m`) is discarded and drawn again, as in the review the
library follows. Conditioning on success shortens the accepted stays a
little: for `n = 1000, K = 4` the mean is about 160 instead of 200.

`n_points ≥ 2 · (n_change_points + 1)` is required; `K = 0` gives one
segment.

### 1.2 The smooth skeleton

Regime `j` has a drift `μ_j = σ_j · |μ_j|`, `|μ_j| ~ Uniform(0.5, 1.5)`.
The signs `σ_j` alternate and the last one is `last_trend` (random if
`None`). The skeleton is

```
x_0 = 0,    x_i = x_{i−1} + μ_{segments[i]}
```

a continuous piecewise-linear curve: linear (smooth) inside a segment, a
kink at a change point. It is mapped affinely, with a positive factor, onto
`[lower + A, upper − A]`, so the directions are kept and the smooth series
touches both ends of that band.

### 1.3 Roughness: the Weierstrass–Mandelbrot sum

The **roughness coefficient** `s ∈ [0, 1]` is `s = D − 1`, where `D` is the
fractal (box-counting) dimension of the graph: a smooth curve such as a sine
has `D = 1` (`s = 0`), a curve that almost fills the plane has `D → 2`
(`s → 1`). Equivalently `H = 1 − s` is the Hölder exponent.

```
W(i) = Σ_{k=0}^{M} λ^{−kH} · cos(2π · λ^k · i / n + φ_k),
λ = 1.5,   M = ⌊log_λ(n / 2)⌋,   φ_k ~ Uniform(0, 2π)
```

The frequencies run from one cycle over the whole series up to the Nyquist
frequency. `W` is divided by `max |W|`, multiplied by

```
A = s · roughness_scale · (upper − lower)
```

and added to the skeleton. Since the skeleton stays within
`[lower + A, upper − A]` and `|A · W / max|W|| ≤ A`, every value is within
`[lower, upper]` without clipping. This needs `s · roughness_scale < 0.5`.

`roughness_scale` (default 0.1) sets the size of the rough part at `s = 1`
as a fraction of the value range; `s` controls both how rough and how large
it is.

The Higuchi estimate of the dimension of `W` matches `1 + s` within 0.1 for
`s ∈ [0.2, 0.6]`; at `s = 0.8` it gives about 1.69, the known downward bias
of the Higuchi method near `D = 2` on sampled curves (see
`tests/test_roughness.py`).

### 1.4 Time axis

```
t_i = start + round(i · (end − start) / (n − 1))
```

computed in exact integer arithmetic. The bounds may come in any order: the
series always runs forward in time, and the "last" segment is the latest
one. Naive times are UTC; aware times are converted to UTC. The unit is the
finest of `ns`, `us`, `ms`, `s` in which both bounds fit into int64 and the
step is at least one tick, so the stamps are strictly increasing; otherwise
`ValueError`.

| Example | Unit |
|---|---|
| 10⁸ points in one day | `ns` |
| today back to 2008 | `ns` |
| year 5 to today | `us` |
| a million years ago to today | `ms` |
| a billion years ago to today | `s` |

Accepted bounds: `datetime.datetime`, `datetime.date`, ISO 8601 strings
(including nanosecond fractions and expanded years such as `-100000-01-01`),
`numpy.datetime64`, `pandas.Timestamp`. `end=None` means now.

### 1.5 Reproducibility

`seed` (an `int` or a `numpy.random.Generator`) feeds
`numpy.random.default_rng`. The same seed and arguments give the same
series bit for bit on the same platform.

## 2. Memory and speed

Everything is computed in chunks: the time stamps exactly in integers, the
Weierstrass–Mandelbrot sum with blocked matrix products (a table of a few
tens of megabytes serves all chunks). Apart from the result (16 bytes per
point: time and value) the extra memory is small; `segments` is computed on
access and is not stored. On a laptop, 10⁷ points take about 4 s and 250 MB;
10⁸ points need about 1.7 GB.

CSV, JSON, Excel, HDF5, NetCDF, Trosna and TsFile are written in chunks.
Parquet, Feather and Arrow IPC build one Arrow table, which needs another
copy of the series in memory.

## 3. Files

All writers overwrite an existing file. Time stamps are integer counts of
`unit` since 1970-01-01 UTC where the format stores integers, ISO 8601 UTC
text (`…Z`) in CSV, JSON and Excel.

* **CSV** — header `time,value,segment`; values in shortest round-trip form.
* **JSON** — an object with `n_points`, `change_points`, `change_times`,
  `trends`, `roughness`, `unit`, then the arrays `time`, `value`, `segment`.
* **Parquet / Feather / Arrow IPC** — columns `time`
  (`timestamp[unit, tz=UTC]`), `value`, `segment`; the ground truth as JSON
  in the schema metadata key `razladka`.
* **HDF5** — datasets `time` (int64, attributes `unit`, `epoch`), `value`,
  `segment`; file attributes `change_points`, `change_times`, `trends`,
  `roughness`, `unit`, `razladka` (all labels as JSON).
* **NetCDF** (via h5netcdf, NetCDF-4) — dimension `time`; variable `time`
  with CF `units = "<unit> since 1970-01-01T00:00:00Z"`, `value`, `segment`;
  the same global attributes as HDF5.
* **NPZ** — arrays `time`, `value`, `segment`, `change_points`, `trends`,
  `roughness`, `unit`.
* **Excel** — sheet `series` (`time` as ISO text, since Excel dates cannot
  hold years before 1900 or nanoseconds), sheet `change_points` (segment,
  first index, first time, trend), sheet `info`. openpyxl writes numbers with
  16 significant digits, so the last bit of a value may differ. At most
  1 048 575 points.
* **Trosna** — device `series` with columns `value` (float64) and `segment`
  (int64) in the chosen unit; one annotation per segment from its first to its
  last time stamp, label `up` / `down`, note `segment k`; file metadata
  `razladka.change_points`, `razladka.change_times`, `razladka.trends`,
  `razladka.roughness`, `razladka.unit`; one commit. Files are read
  identically by pytrosna and by the reference Rust `trosna` CLI.
* **TsFile** (Apache TsFile 2.x, table model) — table `series_<unit>`
  (e.g. `series_ns`; TsFile itself does not store a time unit), tag column
  `segment` (one device per segment), field `value`.

## 4. Scoring a detector

```python
import numpy as np

import razladka

s = razladka.generate(2_000, 4, start="2026-01-01", lower=0, upper=1, roughness=0.3, seed=0)

# a naive detector: the largest changes of the smoothed slope
slope = np.convolve(np.diff(s.values), np.ones(25) / 25, mode="same")
turns = np.abs(np.diff(np.sign(slope)))
found = np.sort(np.argsort(turns)[-4:]) + 1


def hits(truth: np.ndarray, found: np.ndarray, margin: int) -> int:
    return sum(bool(np.any(np.abs(found - t) <= margin)) for t in truth)


print(f"{hits(s.change_points, found, margin=20)} of {s.n_change_points} found")
```
