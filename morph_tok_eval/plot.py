"""Package wrapper for top-level plot.py."""

from importlib import import_module

_mod = import_module("plot")

process_file = _mod.process_file
main = _mod.main

__all__ = ["process_file", "main"]
