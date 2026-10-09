"""Generate a series with three change points and look at the ground truth."""

import razladka
from razladka import Trend

s = razladka.generate(
    1_000,
    3,
    start="2026-01-01",
    end="2026-03-01",
    lower=0.0,
    upper=100.0,
    roughness=0.4,
    last_trend=Trend.UP,
    seed=42,
)

print(s)
for k, trend in enumerate(s.trends):
    first = 0 if k == 0 else int(s.change_points[k - 1])
    print(f"segment {k}: from point {first} ({s.times[first]}), {trend.value}")
