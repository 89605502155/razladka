"""Write one series to every supported file format (needs razladka[all])."""

import pathlib
import sys

import razladka
from razladka import FileFormat

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "razladka-out")
out.mkdir(exist_ok=True)

s = razladka.generate(
    2_000, 4, start="0005-01-01", end="2026-10-09", lower=-1, upper=1, roughness=0.5, seed=7
)
for fmt in FileFormat:
    path = out / f"series{fmt.extension}"
    s.save(path)
    print(f"{fmt.name:8} {path} ({path.stat().st_size} bytes)")
