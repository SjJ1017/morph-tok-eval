"""Package wrapper for top-level align.py."""

from importlib import import_module

_mod = import_module("align")

IBM1 = _mod.IBM1
IBM2 = _mod.IBM2
MODEL_REGISTRY = _mod.MODEL_REGISTRY
read_data = _mod.read_data
compute_score = _mod.compute_score
evaluate_segmentations = _mod.evaluate_segmentations

__all__ = [
    "IBM1",
    "IBM2",
    "MODEL_REGISTRY",
    "read_data",
    "compute_score",
    "evaluate_segmentations",
]
