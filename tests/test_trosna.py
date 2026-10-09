"""Trosna files written by razladka are read identically by pytrosna and by
the reference Rust implementation (the ``trosna`` CLI, set TROSNA_CLI).

Both readers are checked against the generated series: values bit for bit,
time stamps and unit, segment numbers, one annotation per segment, the
ground-truth metadata and the integrity of the file.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import subprocess
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import razladka
from razladka._io import TROSNA_DEVICE

pytrosna = pytest.importorskip("pytrosna")

UNIT_CASES = {
    "ns": ("2026-01-01", "2026-01-02"),
    "us": ("0005-01-01", "2026-10-09"),
    "ms": (np.datetime64(-(2**60), "ms"), "2026-01-01"),
    "s": (np.datetime64(-(2**62), "s"), "2026-01-01"),
}


def expected_annotations(s: razladka.GeneratedSeries) -> list[tuple[int, int, str, str]]:
    stamps = s.times.astype(np.int64)
    bounds = [0, *s.change_points.tolist(), len(s)]
    return [
        (int(stamps[bounds[k]]), int(stamps[bounds[k + 1] - 1]), trend.value, f"segment {k}")
        for k, trend in enumerate(s.trends)
    ]


def check_with_pytrosna(path: pathlib.Path, s: razladka.GeneratedSeries) -> None:
    f = pytrosna.open(path)
    assert f.verify().ok
    assert f.finalized
    assert list(f.devices) == [TROSNA_DEVICE]
    schema = f.device(TROSNA_DEVICE)
    assert schema.time_unit.label == s.unit
    assert [(c.name, c.data_type.label) for c in schema.columns] == [
        ("value", "float64"),
        ("segment", "int64"),
    ]
    batch = f.read(TROSNA_DEVICE)
    assert np.array_equal(batch.time, s.times.astype(np.int64))
    assert batch["value"].to_numpy().tobytes() == s.values.tobytes()
    assert np.array_equal(batch["segment"].to_numpy(), s.segments)
    annotations = [(a.start, a.end, a.label, a.note) for a in f.annotations(TROSNA_DEVICE)]
    assert annotations == expected_annotations(s)
    labels = s.labels()
    meta = f.metadata
    assert json.loads(meta["razladka.change_points"]) == labels["change_points"]
    assert json.loads(meta["razladka.change_times"]) == labels["change_times"]
    assert json.loads(meta["razladka.trends"]) == labels["trends"]
    assert float(meta["razladka.roughness"]) == s.roughness
    assert meta["razladka.unit"] == s.unit
    assert len(f.commits()) == 1


def run(cli: str, *args: object) -> str:
    result = subprocess.run(
        [cli, *map(str, args)], capture_output=True, text=True, check=False, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def check_with_rust(cli: str, path: pathlib.Path, s: razladka.GeneratedSeries) -> None:
    assert run(cli, "verify", path).startswith("OK")
    info = run(cli, "info", path)
    assert f"device '{TROSNA_DEVICE}': {len(s)} points" in info
    assert f"unit {s.unit}," in info
    assert "finalized" in info
    for key, value in s.labels().items():
        if key == "n_points":
            continue
        line = f"razladka.{key} = "
        stored = next(x for x in info.splitlines() if line in x).split(line, 1)[1]
        if key == "unit":
            assert stored == value
        elif key == "roughness":
            assert float(stored) == value
        else:
            assert json.loads(stored) == value
    # raw points: time stamps as integers, values in shortest round-trip form
    rows = list(csv.reader(io.StringIO(run(cli, "cat", "--raw-time", path))))
    assert rows[0] == ["time", "value", "segment"]
    body = rows[1:]
    assert len(body) == len(s)
    assert np.array_equal(np.array([int(r[0]) for r in body]), s.times.astype(np.int64))
    assert np.array([float(r[1]) for r in body]).tobytes() == s.values.tobytes()
    assert np.array_equal(np.array([int(r[2]) for r in body]), s.segments)
    # annotations: id, device, start, end (as dates), label, note
    lines = [x.split("\t") for x in run(cli, "annotations", path).splitlines() if x]
    assert [(x[1], x[4], x[5]) for x in lines] == [
        (TROSNA_DEVICE, label, note) for _, _, label, note in expected_annotations(s)
    ]


def check_rust_csv_export(
    cli: str, path: pathlib.Path, s: razladka.GeneratedSeries, *, dates: bool
) -> None:
    """``trosna convert x.trosna x.csv`` reproduces the series."""
    out = path.with_suffix(".rust.csv")
    run(cli, "convert", "--raw-time", "--force", path, out)
    with out.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["time", "value", "segment"]
    assert [int(r[0]) for r in rows[1:]] == s.times.astype(np.int64).tolist()
    assert [float(r[1]) for r in rows[1:]] == s.values.tolist()
    assert [int(r[2]) for r in rows[1:]] == s.segments.tolist()
    if not dates:
        return
    # the date-time export agrees with razladka's own CSV
    out_dates = path.with_suffix(".dates.csv")
    run(cli, "convert", "--force", path, out_dates)
    ours = path.with_suffix(".ours.csv")
    s.save(ours)
    with out_dates.open(newline="") as a, ours.open(newline="") as b:
        rust_rows = list(csv.reader(a))[1:]
        our_rows = list(csv.reader(b))[1:]
    for rust_row, our_row in zip(rust_rows, our_rows, strict=True):
        assert float(rust_row[1]) == float(our_row[1])
        assert rust_row[2] == our_row[2]
        assert np.datetime64(rust_row[0]) == np.datetime64(our_row[0].rstrip("Z"))


def make(unit_case: str, n: int = 400, k: int = 3, **kwargs: Any) -> razladka.GeneratedSeries:
    start, end = UNIT_CASES[unit_case]
    return razladka.generate(
        n, k, start=start, end=end, lower=-2, upper=3, **{"roughness": 0.5, "seed": 3, **kwargs}
    )


@pytest.mark.parametrize("unit", list(UNIT_CASES))
def test_pytrosna_reads_razladka(tmp_path: pathlib.Path, unit: str) -> None:
    s = make(unit)
    assert s.unit == unit
    path = tmp_path / "series.trosna"
    s.save(path)
    check_with_pytrosna(path, s)


def test_pytrosna_pandas_view(tmp_path: pathlib.Path) -> None:
    s = make("ns")
    path = tmp_path / "series.trosna"
    s.save(path)
    df = pytrosna.read_pandas(path, TROSNA_DEVICE)
    assert np.array_equal(df["value"].to_numpy(), s.values)
    assert np.array_equal(df["segment"].to_numpy(), s.segments)


def test_zero_change_points_single_annotation(tmp_path: pathlib.Path) -> None:
    s = make("ns", n=2, k=0)
    path = tmp_path / "one.trosna"
    s.save(path)
    check_with_pytrosna(path, s)


def test_several_blocks(tmp_path: pathlib.Path) -> None:
    """More points than one write chunk."""
    s = make("ns", n=(1 << 20) + 1000, k=6)
    path = tmp_path / "big.trosna"
    s.save(path)
    check_with_pytrosna(path, s)


@given(
    n_extra=st.integers(0, 300),
    k=st.integers(0, 8),
    unit=st.sampled_from(list(UNIT_CASES)),
    roughness=st.floats(0.0, 1.0),
    seed=st.integers(0, 2**32 - 1),
)
@settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_pytrosna(
    tmp_path: pathlib.Path, n_extra: int, k: int, unit: str, roughness: float, seed: int
) -> None:
    s = make(unit, n=2 * (k + 1) + n_extra, k=k, roughness=roughness, seed=seed)
    path = tmp_path / "prop.trosna"
    s.save(path)
    check_with_pytrosna(path, s)


# ---------------------------------------------------------------- reference Rust CLI


@pytest.mark.interop
@pytest.mark.parametrize("unit", list(UNIT_CASES))
def test_rust_cli_reads_razladka(tmp_path: pathlib.Path, rust_cli: str, unit: str) -> None:
    s = make(unit)
    path = tmp_path / "series.trosna"
    s.save(path)
    check_with_rust(rust_cli, path, s)


@pytest.mark.interop
@pytest.mark.parametrize("unit", list(UNIT_CASES))
def test_rust_csv_export_matches(tmp_path: pathlib.Path, rust_cli: str, unit: str) -> None:
    s = make(unit, n=300, k=4)
    path = tmp_path / "series.trosna"
    s.save(path)
    # the Rust CLI (chrono) prints dates only from the year -262144 on
    check_rust_csv_export(rust_cli, path, s, dates=unit in {"ns", "us"})


@pytest.mark.interop
def test_rust_cli_several_blocks(tmp_path: pathlib.Path, rust_cli: str) -> None:
    s = make("ns", n=(1 << 20) + 1000, k=6)
    path = tmp_path / "big.trosna"
    s.save(path)
    assert run(rust_cli, "verify", path).startswith("OK")
    rows = run(rust_cli, "cat", "--raw-time", path).splitlines()
    assert len(rows) == len(s) + 1
    last = rows[-1].split(",")
    assert int(last[0]) == int(s.times[-1].astype(np.int64))
    assert float(last[1]) == s.values[-1]


@pytest.mark.interop
@given(
    n_extra=st.integers(0, 200),
    k=st.integers(0, 6),
    unit=st.sampled_from(list(UNIT_CASES)),
    roughness=st.floats(0.0, 1.0),
    seed=st.integers(0, 2**32 - 1),
)
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_both_readers(
    tmp_path: pathlib.Path,
    rust_cli: str,
    n_extra: int,
    k: int,
    unit: str,
    roughness: float,
    seed: int,
) -> None:
    s = make(unit, n=2 * (k + 1) + n_extra, k=k, roughness=roughness, seed=seed)
    path = tmp_path / "prop.trosna"
    s.save(path)
    check_with_pytrosna(path, s)
    check_with_rust(rust_cli, path, s)


@pytest.mark.interop
def test_rust_rewrite_is_read_back_by_pytrosna(tmp_path: pathlib.Path, rust_cli: str) -> None:
    """A Rust round trip (Trosna → CSV → Trosna) keeps the data pytrosna reads."""
    s = make("us", n=200, k=2)
    path = tmp_path / "series.trosna"
    s.save(path)
    exported = tmp_path / "series.csv"
    run(rust_cli, "convert", "--raw-time", "--force", path, exported)
    rebuilt = tmp_path / "rebuilt.trosna"
    run(rust_cli, "convert", "--unit", "us", "--device", TROSNA_DEVICE, exported, rebuilt)
    batch = pytrosna.read(rebuilt, TROSNA_DEVICE)
    assert np.array_equal(batch.time, s.times.astype(np.int64))
    assert batch["value"].to_numpy().tobytes() == s.values.tobytes()
    assert np.array_equal(batch["segment"].to_numpy(), s.segments)
