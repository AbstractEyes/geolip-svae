"""
geolip_svae.tests.byte_trigram_similarity_test
================================================
Runnable diagnostic battery for ``SentenceEncoder`` (geolip_svae.inference.text).

Auto-loads a byte-trigram-trained sphere-solver, resolves a codebook
(HuggingFace fetch with fresh-extraction fallback), and runs the four
diagnostic comparison classes from the OMEGA_CATALOG triage protocol
against all three signature modes. Companion to noise_stress_test.py
and sentencepiece_stress_test.py — same convention: a self-contained,
runnable test module exercising a specific substrate / wrapper.

Usage::

    python -m geolip_svae.tests.byte_trigram_similarity_test
    python -m geolip_svae.tests.byte_trigram_similarity_test --hf-version byte_trigram_proto_64_patch_2_v1
    python -m geolip_svae.tests.byte_trigram_similarity_test --calibration byte_trigram_wikitext103_val
    python -m geolip_svae.tests.byte_trigram_similarity_test --pad-strategy space --agg best_match
    python -m geolip_svae.tests.byte_trigram_similarity_test --quick           # one pair per group
    python -m geolip_svae.tests.byte_trigram_similarity_test --extract-fresh   # skip HF fetch

What this test exercises
------------------------
1. Idiomatic load → engine → codebook attach → SentenceEncoder.
2. Codebook resolution with graceful HF-fallback (covers the case where
   the model has been trained but no codebook has been uploaded yet).
3. Same-pair comparison across ``mode ∈ {omega, omega_orig, codebook}``
   so the caller can see what each representation captures.
4. Similarity matrix on a curated 8-sentence test set covering paraphrase,
   edit, domain-match-content-mismatch, and cross-domain pairs.

The test sets are deliberately small and hand-curated, not benchmark
datasets — they triangulate what kind of similarity each representation
produces. Use the matrix to read which mode separates which axis of
variation, then design the actual evaluation around that signal.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from geolip_svae.inference import (
    load_model,
    InferenceEngine,
    Codebook,
    SentenceEncoder,
    make_calibration,
    PAD_STRATEGIES,
    AGG_METHODS,
    CodebookMissingError,
    CodebookIncompatibleError,
    HF_REPO,
)


# ── Diagnostic test sets ─────────────────────────────────────────────
# Pairs are (text_a, text_b). The classes are defined relative to two
# axes: SEMANTIC similarity (do the two say the same thing?) and SURFACE
# similarity (do they share characters/n-grams?).
#
# Class            | semantic | surface  | what it tests
# ─────────────────┼──────────┼──────────┼─────────────────────────────
# Paraphrase       | same     | different| does the model see semantics?
# Edit             | same     | similar  | typo / minor edit robustness
# Same-domain      | different| similar  | content vs form discrimination
# Cross-domain     | different| different| trivial sanity floor

PARAPHRASE_PAIRS = [
    ("The cat sat on the mat.",
     "A feline rested upon the rug."),
    ("Many believe artificial intelligence will transform medicine.",
     "Numerous experts think AI will revolutionize healthcare."),
]

EDIT_PAIRS = [
    ("The cat sat on the mat.",
     "The cat sits on the mat."),
    ("Many believe artificial intelligence will transform medicine.",
     "Many beleive artificial intellgence will transform medecine."),  # typos
]

SAME_DOMAIN_PAIRS = [
    ("The cat sat on the mat.",
     "The dog ran across the park."),
    ("Wikipedia is a free online encyclopedia accessible to anyone.",
     "Britannica is a paid reference work edited by experts."),
]

CROSS_DOMAIN_PAIRS = [
    ("The cat sat on the mat.",
     "import torch.nn.functional as F"),
    ("Many believe artificial intelligence will transform medicine.",
     "ERROR: connection timeout after 30s on port 8443"),
]

TEST_GROUPS = [
    ("Paraphrase     (sem= surf!=)", PARAPHRASE_PAIRS),
    ("Edit           (sem= surf~)",  EDIT_PAIRS),
    ("Same-domain    (sem!= surf~)", SAME_DOMAIN_PAIRS),
    ("Cross-domain   (sem!= surf!=)", CROSS_DOMAIN_PAIRS),
]


# ── Codebook resolution ──────────────────────────────────────────────

def resolve_codebook(
    engine: InferenceEngine,
    hf_version: str,
    calibration_name: str,
    repo_id: str = HF_REPO,
    extract_fresh: bool = False,
) -> Tuple[Codebook, str]:
    """Load codebook from HF if available; fall back to fresh extraction.

    Returns:
        (codebook, source_description)
    """
    if not extract_fresh:
        try:
            from huggingface_hub import hf_hub_download
            st_path = hf_hub_download(
                repo_id=repo_id,
                filename=f'{hf_version}/codebooks/{calibration_name}.safetensors',
                repo_type='model',
            )
            # Force the JSON sidecar download (Codebook.load expects both)
            hf_hub_download(
                repo_id=repo_id,
                filename=f'{hf_version}/codebooks/{calibration_name}.json',
                repo_type='model',
            )
            stem = Path(st_path).with_suffix('')
            cb = Codebook.load(stem)
            return cb, f'hf://{repo_id}/{hf_version}/codebooks/{calibration_name}'
        except Exception as e:
            print(f"  [codebook] HF fetch failed: {type(e).__name__}: {e}")
            print(f"  [codebook] Falling back to fresh extraction.")

    # Fresh extraction on sixteen_noise — directly comparable to h2-64 banks.
    print(f"  [codebook] Extracting on sixteen_noise (n=64, size=64)...")
    calib = make_calibration('sixteen_noise', n=64, size=64)
    cb = engine.extract_codebook(
        calib,
        model_id=hf_version,
        calibration_name='sixteen_noise',
    )
    return cb, 'fresh-extract:sixteen_noise'


# ── Reporting helpers ────────────────────────────────────────────────

def _truncate(s: str, n: int = 50) -> str:
    return s if len(s) <= n else s[: n - 1] + '…'


PRINT_MODES = ('M_flat', 'codebook_codes', 'codebook_sum')
"""Three modes shown in the diagnostic table:
   - M_flat:         per-row preserved (recommended for similarity)
   - codebook_codes: per-row codebook quantization (recommended)
   - codebook_sum:   V-aggregated (diagnostic — should CLT-collapse to ~1)
The contrast between the first two and the last is the headline result:
  if M_flat / codebook_codes give meaningful spread but codebook_sum
  collapses to ~0.998, the per-row representations are doing real work.
"""


def print_pair_table(
    enc: SentenceEncoder,
    pairs: List[Tuple[str, str]],
    agg: str,
):
    """Three-mode table for a list of (a, b) pairs.

    Shows two per-row modes (M_flat, codebook_codes) plus codebook_sum
    as a V-aggregated diagnostic comparison.
    """
    print(f"  {'mode':<14s} {'M_flat':>8s}  {'cb_codes':>9s}  {'cb_sum_NB':>10s}")
    print(f"  {'─'*14} {'─'*8}  {'─'*9}  {'─'*10}")
    for text_a, text_b in pairs:
        sims = {}
        for mode in PRINT_MODES:
            try:
                sims[mode] = enc.similarity(
                    text_a, text_b, mode=mode, agg=agg,
                )
            except CodebookMissingError:
                sims[mode] = float('nan')
        print(f"  A: {_truncate(text_a, 80)!r}")
        print(f"  B: {_truncate(text_b, 80)!r}")
        print(f"  {'per-patch':<14s} "
              f"{sims['M_flat']:>+8.4f}  "
              f"{sims['codebook_codes']:>+9.4f}  "
              f"{sims['codebook_sum']:>+10.4f}")
        print()


def print_similarity_matrix(
    enc: SentenceEncoder,
    sentences: List[str],
    labels: List[str],
    mode: str,
    agg: str,
):
    """Compact pairwise similarity matrix with row labels."""
    sim = enc.similarity_matrix(
        sentences, mode=mode, agg=agg,
    )
    n = len(sentences)
    label_w = max(len(l) for l in labels) + 1
    # Header (column indices)
    print(f"  {'':<{label_w}s}  " + "  ".join(f"{i:>5d}" for i in range(n)))
    for i in range(n):
        row = "  ".join(f"{sim[i, j]:+.2f}" for j in range(n))
        print(f"  {labels[i]:<{label_w}s}  {row}")


# ── Main ──────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end demo of SentenceEncoder on a byte-trigram model. "
            "Auto-loads model + codebook, runs four diagnostic comparison "
            "classes across all three signature modes."
        )
    )
    parser.add_argument(
        '--hf-version',
        default='byte_trigram_proto_64_patch_2_v1',
        help=("Named HF version under AbstractPhil/geolip-SVAE. "
              "Default: byte_trigram_proto_64_patch_2_v1."),
    )
    parser.add_argument(
        '--calibration',
        default='byte_trigram_wikitext103_val',
        help=("Codebook calibration name to fetch from HF "
              "(<hf-version>/codebooks/<calibration>.safetensors). "
              "Default: byte_trigram_wikitext103_val."),
    )
    parser.add_argument(
        '--extract-fresh',
        action='store_true',
        help="Skip HF codebook fetch; extract fresh on sixteen_noise instead.",
    )
    parser.add_argument(
        '--pad-strategy',
        default='space',
        choices=PAD_STRATEGIES,
        help=("How to pad short text. Default: space (gives a meaningful "
              "patch_real_mask). Use 'repeat' only for diagnostics where "
              "you specifically want all patches to count as real."),
    )
    parser.add_argument(
        '--agg',
        default='patch_mean',
        choices=AGG_METHODS,
        help=("Per-patch cosine aggregation. Operates on cosine SCALARS "
              "after per-patch comparison; pre-cosine pooling on features "
              "is intentionally not supported (architecture-incompatible). "
              "Default: patch_mean."),
    )
    parser.add_argument(
        '--img-size', type=int, default=64,
        help="Image side length. Should match the model's training. Default: 64.",
    )
    parser.add_argument(
        '--quick', action='store_true',
        help="Use one pair per group for fast iteration.",
    )
    parser.add_argument(
        '--device', default=None,
        help="Override device ('cuda' or 'cpu'). Default: auto-detect.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("=" * 70)
    print(f"SentenceEncoder demo — model: {args.hf_version}")
    print("=" * 70)

    # ── 1. Load model ──
    print(f"\n[1/4] Loading model from HF…")
    try:
        model, cfg = load_model(args.hf_version, device=args.device)
    except Exception as e:
        print(f"  ERROR: model load failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: V={cfg['V']}, D={cfg['D']}, ps={cfg['patch_size']}, "
          f"hidden={cfg['hidden']}, depth={cfg['depth']}, "
          f"n_cross={cfg['n_cross_layers']}")
    print(f"  Architecture: linear_readout={cfg.get('linear_readout')}, "
          f"svd_mode={cfg.get('svd_mode')}, "
          f"smooth_mid={cfg.get('smooth_mid')}")
    print(f"  Params: {n_params:,}, "
          f"best_test_mse={cfg.get('_test_mse')!r} "
          f"@ ep {cfg.get('_epoch')!r}")
    device = next(model.parameters()).device
    print(f"  Device: {device}")

    engine = InferenceEngine(model)

    # ── 2. Resolve codebook ──
    print(f"\n[2/4] Resolving codebook ({args.calibration})…")
    try:
        cb, cb_source = resolve_codebook(
            engine,
            hf_version=args.hf_version,
            calibration_name=args.calibration,
            extract_fresh=args.extract_fresh,
        )
    except Exception as e:
        print(f"  ERROR: codebook resolution failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 3

    print(f"  Source: {cb_source}")
    print(f"  {cb}")
    try:
        engine.attach_codebook(cb)
        print(f"  Attached. compatible_with(model)=True (D={cb.D} == model.D={model.D})")
    except CodebookIncompatibleError as e:
        print(f"  ERROR: codebook incompatible with model: {e}")
        return 4

    # ── 3. Build encoder ──
    print(f"\n[3/4] Building SentenceEncoder…")
    try:
        enc = SentenceEncoder(
            engine,
            img_size=args.img_size,
            patch_size=cfg['patch_size'],
            pad_strategy=args.pad_strategy,
        )
    except ValueError as e:
        print(f"  ERROR: SentenceEncoder init failed: {e}")
        return 5
    print(f"  Encoder: img_size={enc.img_size}, patch_size={enc.patch_size}, "
          f"pad_strategy={enc.pad_strategy!r}")
    print(f"  Capacity: {enc.bytes_per_image:,} bytes/image, "
          f"{enc.n_patches} patches × {enc.bytes_per_patch} bytes/patch")
    print(f"  Per-patch aggregation: {args.agg!r}")

    # ── 4a. Step 0 round-trip sanity check ──
    print(f"\n[4a] Round-trip sanity check (text → image → recon → text)…")
    print(f"  If real_byte_acc << model's training byte recovery (~99.6%),")
    print(f"  per-patch features for these sentences are unreliable.\n")
    print(f"  {'sentence':<55s}  {'n_real':>6s}  {'real_acc':>8s}  "
          f"{'real_l1':>7s}  {'recon_text_real':<40s}")
    print(f"  {'─'*55}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*40}")
    all_real_acc = []
    for group_name, pairs in TEST_GROUPS:
        for text in [pairs[0][0], pairs[0][1]] if args.quick \
                else [t for p in pairs for t in p]:
            m = enc.roundtrip_metrics(text)
            all_real_acc.append(m['real_byte_acc'])
            print(f"  {_truncate(text, 55):<55s}  "
                  f"{m['n_real_bytes']:>6d}  "
                  f"{m['real_byte_acc']:>8.4f}  "
                  f"{m['real_byte_l1']:>7.3f}  "
                  f"{_truncate(m['recon_text_real'], 40):<40s}")
    mean_acc = sum(all_real_acc) / len(all_real_acc) if all_real_acc else 0.0
    print(f"\n  Mean real_byte_acc across test set: {mean_acc:.4f}")
    if mean_acc < 0.95:
        print(f"  WARNING: low real-byte recovery on test sentences. "
              f"Per-patch similarity below may be unreliable.")
    elif mean_acc < 0.99:
        print(f"  Note: real-byte recovery somewhat below training floor "
              f"(~99.6%); proceeding with similarity but with caution.")
    else:
        print(f"  Round-trip looks healthy; proceeding to similarity.")

    # ── 4b. Run diagnostics ──
    print(f"\n[4b] Diagnostic comparisons (per-patch cosine similarity)…")
    print(f"  Three signature modes per pair: omega, omega_orig, codebook")
    print(f"  Higher = more similar. Range [-1, 1] for cosine.\n")

    for group_name, pairs in TEST_GROUPS:
        if args.quick:
            pairs = pairs[:1]
        print(f"┌─ {group_name} {'─' * (66 - len(group_name))}")
        print_pair_table(enc, pairs, agg=args.agg)

    # ── Cross-set similarity matrix ──
    print(f"┌─ Pairwise similarity matrix (one example per group) {'─' * 16}")
    sentences: List[str] = []
    labels: List[str] = []
    for group_name, pairs in TEST_GROUPS:
        a, b = pairs[0]
        # Compact 4-char group prefix for matrix labels
        prefix = group_name.split()[0][:4].lower()
        sentences.extend([a, b])
        labels.extend([f"{prefix}-A", f"{prefix}-B"])

    for mode in ('M_flat', 'codebook_codes'):
        print(f"\n  Mode: {mode!r}  (per-row preserved)")
        try:
            print_similarity_matrix(
                enc, sentences, labels, mode=mode, agg=args.agg,
            )
        except CodebookMissingError as e:
            print(f"    skipped ({type(e).__name__}: {e})")
    # One V-aggregated diagnostic mode for contrast
    print(f"\n  Mode: 'codebook_sum'  (V-aggregated — diagnostic only)")
    try:
        print_similarity_matrix(
            enc, sentences, labels, mode='codebook_sum', agg=args.agg,
        )
    except CodebookMissingError as e:
        print(f"    skipped ({type(e).__name__}: {e})")

    # ── Reading guide ──
    print()
    print("─" * 70)
    print("Reading guide:")
    print("  M_flat: direct per-patch sphere-norm encoder rows (V*D dims).")
    print("    Most byte-faithful representation; cosine measures byte-level")
    print("    similarity between corresponding patches.")
    print("  codebook_codes: per-row argmax over codebook axes, one-hot flat.")
    print("    Cosine = Hamming-style overlap (fraction of rows landing on")
    print("    the same polytope axis). Quantized but interpretable.")
    print("  codebook_sum (DIAGNOSTIC): V-summed |projections|. Should")
    print("    CLT-collapse to ~0.998 across all sentence pairs because the")
    print("    sum-over-32 unit vectors is near-constant. Included for")
    print("    contrast — if M_flat / codebook_codes show real spread but")
    print("    codebook_sum collapses, per-row representations are working.")
    print("  cross-domain pairs should be the lowest similarity floor; if")
    print("    they're not, the model isn't separating distributions cleanly.")
    print("─" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
