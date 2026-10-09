from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import razladka


def _find_rust_cli() -> str | None:
    candidate = os.environ.get("TROSNA_CLI") or shutil.which("trosna")
    if candidate and os.access(candidate, os.X_OK):
        return candidate
    return None


@pytest.fixture(scope="session")
def rust_cli() -> str:
    """The reference ``trosna`` command (set TROSNA_CLI); skips the test if missing."""
    cli = _find_rust_cli()
    if cli is None:
        pytest.skip("the reference Rust CLI is not available (set TROSNA_CLI)")
    return cli


@pytest.fixture
def series() -> razladka.GeneratedSeries:
    """A small rough series with three change points."""
    return razladka.generate(
        500,
        3,
        start="2026-01-01T00:00:00Z",
        end="2026-01-02T00:00:00Z",
        lower=-5.0,
        upper=5.0,
        roughness=0.5,
        seed=7,
    )


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace: dict[str, object]) -> None:
    doctest_namespace["razladka"] = razladka
    doctest_namespace["np"] = np
