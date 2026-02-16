"""Package wrapper for top-level get_vocabulary.py."""

from importlib import import_module

_mod = import_module("get_vocabulary")

main = getattr(_mod, "main", None)

__all__ = ["main"]
