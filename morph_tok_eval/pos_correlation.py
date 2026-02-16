"""Package wrapper for top-level pos_correlation.py."""

from importlib import import_module

_mod = import_module("pos_correlation")

main = getattr(_mod, "main", None)

__all__ = ["main"]
