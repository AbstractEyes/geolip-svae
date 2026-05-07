"""
geolip_svae.inference.vocabulary
=================================
Vocabulary-aware codebook tooling.

PLACEHOLDER — not yet implemented.

The intent of this module is to host vocabulary-bound projective codebook
infrastructure that pairs trained sphere-solver / freckles models with a
tokenizer's vocabulary. Once the experiment-001 results from
``svae_proto.exp_001_vocab_trigram_recall`` graduate, the dataset/eval
helpers and the per-token recovery metrics live here as a stable public
surface.

What's planned (subject to NOTES.md in exp_001):

    - VocabCodebook artifact: extends Codebook with per-axis token-id
      attribution from `top_corrupted_tokens` / `confusion_pairs`.
    - vocab_recovery_metrics(model, dataset, ...) — per-byte / per-token
      / per-token-prefix / per-token-id-recovery, graduated from
      svae_proto.exp_001_vocab_trigram_recall.eval.
    - VocabTrigramDataset graduation from svae_proto into the main
      DATASET_FACTORIES under name 'vocab_trigram'.

Until that happens, importing this module is harmless but unhelpful —
nothing is exported. Code that needs vocab-aware functionality should
import from svae_proto.exp_001_vocab_trigram_recall instead.

If you find yourself reaching for something that "should be in here,"
that's the signal to graduate it from the prototype rather than re-implement.
"""
from __future__ import annotations


def __getattr__(name: str):
    """Loud failure on attribute access so accidental imports surface clearly.

    Importing the module is fine (no side effects) but reaching for any
    name on it raises NotImplementedError pointing at the prototype.
    """
    raise NotImplementedError(
        f"geolip_svae.inference.vocabulary.{name} is not implemented yet. "
        f"Vocab-aware tooling lives in svae_proto.exp_001_vocab_trigram_recall "
        f"until the experiment graduates. See svae_proto/README.md."
    )


__all__: list = []   # explicit empty surface
