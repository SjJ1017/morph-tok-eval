"""Package wrapper for top-level data.py."""

from importlib import import_module

_mod = import_module("data")

unimorph = _mod.unimorph
unisegments = _mod.unisegments
main = _mod.main

__all__ = ["unimorph", "unisegments", "main"]
