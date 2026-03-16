"""src.ml - model selection package."""

from __future__ import annotations

__all__ = ["run_selection", "ARCHITECTURES"]


def __getattr__(name: str):
    if name == "run_selection":
        from src.ml.model_selector import run_selection
        return run_selection
    if name == "ARCHITECTURES":
        from src.ml.model_selector import ARCHITECTURES
        return ARCHITECTURES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
