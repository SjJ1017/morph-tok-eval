"""Package wrapper for top-level dynamic_program_segment.py."""

from importlib import import_module

_mod = import_module("dynamic_program_segment")

segment = _mod.segment

__all__ = ["segment"]
