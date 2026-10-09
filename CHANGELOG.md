# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-09

### Added

- `generate()`: time series with exactly `K` change points placed by a Markov
  regime-switching chain with a left-to-right transition matrix; alternating
  piecewise-linear skeleton; Weierstrass–Mandelbrot roughness of fractal
  dimension `1 + roughness`; values always within `[lower, upper]`.
- Time axis from the year 5 (and earlier) to 10⁸ points per day: automatic
  choice of ns / µs / ms / s, exact integer spacing, bounds in any order.
- Results as `GeneratedSeries`, NumPy, pandas, Polars or PyArrow, with the
  ground truth (change points, their times, trends).
- Files: CSV, JSON, Parquet, Feather, Arrow IPC, HDF5, NetCDF, NPZ, Excel,
  Trosna (via pytrosna, verified against the reference Rust CLI), TsFile.

[0.1.0]: https://github.com/89605502155/razladka/releases/tag/v0.1.0
