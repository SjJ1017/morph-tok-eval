"""Package wrapper for top-level pos_tagger.py."""

from importlib import import_module

_mod = import_module("pos_tagger")

train_pos_tagger = _mod.train_pos_tagger
transformers_closure = _mod.transformers_closure
legros_tokenizer_closure = _mod.legros_tokenizer_closure

__all__ = ["train_pos_tagger", "transformers_closure", "legros_tokenizer_closure"]
