# 001 — Vocabulary Trigram Recall

## Hypothesis

The current `byte_trigram` dataset measures reconstruction at the
**byte** granularity — every pixel encodes 3 raw bytes via the RGB
channel mapping, and recovery is the per-byte exact-match rate. This
treats every byte as an independent atom and tells us nothing about
whether the model preserves the **structural units of language** —
sentencepiece tokens, words, ngrams.

For the downstream programs Phil has in mind (vocabulary-bound
codebooks; full elemental structural awareness; controllable
high-complexity vector differentiation; cross-objective ngram fusion),
the relevant question is:

> When the model reconstructs a region of pixels representing a
> multi-byte token, did the **whole token** come back, or did it come
> back as a near-miss (one byte off, but no longer a valid token)?

A near-miss byte-string corresponds to a different vocab id (or no
valid id at all) — for any system that consumes recovered bytes as
language, that's a categorical failure even at high per-byte accuracy.
The byte-level metric over-reports goodness when the bytes that flip
are exactly the ones that change a token's identity.

## What this experiment measures

Same encoding as `byte_trigram` (utf-8 bytes packed 3-per-pixel as RGB,
values mapped to [-1, +1]) — so the encoder/decoder math is identical
and any difference in measured performance comes from the **evaluation
lens**, not the input distribution.

The dataset additionally tracks token boundaries — the byte offsets
where each sentencepiece token starts in each image's flat byte
window. The eval module exposes:

- `per_byte_acc`             — baseline, comparable to existing byte_trigram
- `per_token_exact_acc`      — fraction of tokens whose every byte recovered
- `per_token_prefix_acc`     — average length of correct prefix per token,
                                normalized by token length (a "soft" measure
                                of how much structure survives)
- `per_token_id_recovery`    — for each token in each image, did the
                                recovered bytes round-trip through the
                                tokenizer back to the original token id?
- `top_corrupted_tokens`     — vocab ids most often lost to recon error
- `confusion_pairs`          — (orig_id → recovered_id) pairs that occur
                                most often when recon fails the exact match

## Success criteria

This experiment graduates if **any** of the following hold:

1. **Quantitative wedge**: `per_token_exact_acc` curves cleanly separate
   different cfg variants (h2-class arch, freckles arch, full geolip
   arch) in a way that `per_byte_acc` does not. I.e. the new metric is
   discriminative where the old one is saturated.
2. **Qualitative differential**: at the same `per_byte_acc`, two trained
   models produce meaningfully different `per_token_id_recovery`
   distributions — implying token-level error patterns are
   architecture-dependent and codebook tooling can exploit that.
3. **Codebook hook**: the trained model's projective-axis Codebook
   (extracted via `inference.train_codebook.create_codebook`) shows
   structure on the `per_token_id_recovery` axis when projected through
   it — i.e. axes correlate with vocab-id clusters. This would
   directly motivate the experiment 002+ "vocabulary_codebook" line.

## What this experiment does NOT change

- Encoder / decoder architecture (still PatchSVAE).
- Training loop (use existing `train()`).
- Loss (still per-pixel MSE — token-level is **eval-only**, not a loss
  term, in this experiment. Adding sequence-aware losses is experiment
  003 territory; this is the substrate for it).
- The byte-stream encoding (3 bytes/pixel as RGB).

## Followups gated on results

If this proves out:
- Experiment 002 — 1-gram + 2-gram parallel batteries with cross-fusion
  (this experiment provides the recall metric we'd compare against).
- Experiment 003 — sequence-strict mode in the trainer: when
  `cfg['sequence_strict']=True`, save_best_model uses
  `per_token_exact_acc` instead of per-byte MSE.
- `vocabulary_codebook` artifact — Codebook.metadata.token_id_axis_map
  derived from per-token recovery statistics aggregated across
  calibration samples.
