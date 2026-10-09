"""Imports of optional dependencies with a helpful error message."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Imports ``module`` or raises :class:`ImportError` naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        top = module.split(".", 1)[0]
        msg = (
            f"this feature needs the optional dependency {top!r}; "
            f'install it with: pip install "razladka[{extra}]"'
        )
        raise ImportError(msg) from exc
