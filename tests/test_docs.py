"""The code in the README files, the guides and the examples runs."""

from __future__ import annotations

import os
import pathlib
import re
import runpy
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "README.ru.md", ROOT / "docs" / "guide.md",
        ROOT / "docs" / "guide.ru.md"]  # fmt: skip
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def blocks() -> list[tuple[str, int, str]]:
    found = []
    for doc in DOCS:
        for i, code in enumerate(BLOCK.findall(doc.read_text(encoding="utf-8"))):
            found.append((doc.name, i, code))
    return found


@pytest.mark.parametrize(("doc", "index", "code"), blocks(), ids=lambda x: str(x)[:20])
def test_doc_block(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, doc: str, index: int, code: str
) -> None:
    if "from razladka import generate, GeneratedSeries" in code:
        pytest.skip("signature listing")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, f"{doc}[{index}]", "exec"), {"__name__": "__main__"})  # noqa: S102


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, example: pathlib.Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(example), str(tmp_path)])
    runpy.run_path(str(example), run_name="__main__")


def test_readme_links_are_absolute() -> None:
    """Relative links break on PyPI."""
    for doc in [ROOT / "README.md", ROOT / "README.ru.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            assert target.startswith("https://"), (doc.name, target)


def test_api_docs_list_every_format() -> None:
    import razladka

    for doc in ["api.md", "api.ru.md"]:
        text = (ROOT / "docs" / doc).read_text(encoding="utf-8")
        for fmt in razladka.FileFormat:
            assert f"`{fmt.name}`" in text, (doc, fmt)
    assert os.fspath(ROOT)
