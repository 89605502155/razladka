"""Write a Trosna file and read the segments back as annotations (needs razladka[trosna])."""

import pathlib
import sys
import tempfile

import pytrosna

import razladka

folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
path = folder / "series.trosna"
s = razladka.generate(500, 2, start="2026-01-01", lower=0, upper=1, roughness=0.3, seed=1)
s.save(path)

f = pytrosna.open(path)
print("verified:", f.verify().ok)
print("metadata:", f.metadata)
for a in f.annotations("series"):
    print(a.label, a.note, f.to_datetime(a.start, "series"), "…", f.to_datetime(a.end, "series"))
