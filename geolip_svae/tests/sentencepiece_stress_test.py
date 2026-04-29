# ════════════════════════════════════════════════════════════════════════════
# COLAB CELL — sentencepiece_proto bit-recovery stress test
# ════════════════════════════════════════════════════════════════════════════
#
# Loads the trained sentencepiece_proto_v1 checkpoint and runs three tests
# that the standard token_exact_rate metric can't distinguish:
#
#   TEST A — Full val-set evaluation (32K tokens, not 1024)
#           Same metric as training, just much bigger sample.
#
#   TEST B — Conditional per-bit accuracy
#           Splits per_bit_position_acc into two: accuracy-when-source-bit-is-1
#           vs accuracy-when-source-bit-is-0. If a bit position is structurally
#           skewed (e.g. bit 15 is always 0), one of these will be near-100%
#           while the other is near-50% (or undefined for vanishing class).
#           Reveals whether each position is actually discriminating, vs
#           trivially predicting the majority class.
#
#   TEST C — Adversarial uniform-random IDs
#           Hand-constructs batches with token IDs sampled uniform[0, vocab),
#           NOT from the corpus. Every bit position is now ~50/50. If the
#           substrate truly handles bit reconstruction non-trivially, accuracy
#           stays high. If it was exploiting Zipfian skew, accuracy collapses
#           at MSB positions.
#
#   TEST D — Rare-region IDs only (IDs in [vocab/2, vocab))
#           Forces bit 14 = 1 always (in a vocab=32000 setup with bit 14
#           thresholding at 16384). Tests whether the model can correctly
#           output bit 14 = +1 when it's used to outputting -1.
#
# Run this in a fresh Colab cell after installing geolip_svae and authenticating
# to HuggingFace. Expects the trained checkpoint at:
#   AbstractPhil/geolip-SVAE/sentencepiece_proto_v1/checkpoints/best.pt
#
# ════════════════════════════════════════════════════════════════════════════

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# ── Imports from the repo ──
from geolip_svae.model import PatchSVAE
from geolip_svae.train import (
    SentencePieceBitDataset,
    decode_image_to_tokens,
    token_bit_recovery_metrics,
)


HF_REPO = 'AbstractPhil/geolip-SVAE'
HF_VERSION = 'sentencepiece_proto_v1'
CKPT_PATH_IN_REPO = f'{HF_VERSION}/checkpoints/best.pt'
N_BITS = 16
VOCAB_SIZE = 32000  # t5-base spiece.model raw piece count

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ════════════════════════════════════════════════════════════════════════════
# 1. Load the trained model from HF
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[1] Loading checkpoint from {HF_REPO}/{CKPT_PATH_IN_REPO}...")
ckpt_local = hf_hub_download(repo_id=HF_REPO, filename=CKPT_PATH_IN_REPO)
ckpt = torch.load(ckpt_local, map_location='cpu', weights_only=False)
cfg = dict(ckpt['config'])  # mutable copy

# Backfill any architecture kwargs missing from older-format checkpoints.
# Pre-fix saver dropped n_heads / smooth_mid; final_report.json has them.
needs_backfill = ('n_heads' not in cfg) or ('smooth_mid' not in cfg)
if needs_backfill:
    print(f"  Older checkpoint format detected — backfilling missing arch "
          f"kwargs from final_report.json...")
    try:
        report_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=f'{HF_VERSION}/final_report.json')
        with open(report_path) as f:
            report_cfg = json.load(f)['config']
        for k in ('n_heads', 'smooth_mid', 'sp_tokenizer', 'sp_corpus',
                   'sp_n_bits'):
            if k not in cfg and k in report_cfg:
                cfg[k] = report_cfg[k]
                print(f"    backfilled {k} = {cfg[k]}")
    except Exception as e:
        print(f"  ⚠ couldn't read final_report.json ({e})")
        print(f"  ⚠ falling back to known sentencepiece_proto_v1 preset values")
        cfg.setdefault('n_heads', 4)
        cfg.setdefault('smooth_mid', 16)

# Rebuild the architecture from the saved config dict
n_cross = cfg.get('n_cross', cfg.get('n_cross_layers', 2))
model_kwargs = dict(
    V=cfg['V'], D=cfg['D'], ps=cfg['patch_size'],
    hidden=cfg['hidden'], depth=cfg['depth'], n_cross=n_cross,
)
for k in ('n_heads', 'smooth_mid'):
    if cfg.get(k) is not None:
        model_kwargs[k] = cfg[k]
if cfg.get('linear_readout', False):
    model_kwargs['linear_readout'] = True
if cfg.get('svd_mode', 'default') != 'default':
    model_kwargs['svd_mode'] = cfg['svd_mode']
if 'match_params' in cfg:
    model_kwargs['match_params'] = cfg['match_params']

print(f"  model_kwargs: {model_kwargs}")
model = PatchSVAE(**model_kwargs).to(device).eval()
model.load_state_dict(ckpt['model_state_dict'], strict=True)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params:,} params, V={model_kwargs['V']}, "
      f"D={model_kwargs['D']}, ps={model_kwargs['ps']}")
print(f"  Resumed from epoch {ckpt['epoch']}, best test_mse={ckpt['test_mse']:.6e}")


# ════════════════════════════════════════════════════════════════════════════
# 2. Helpers — encode arbitrary token IDs to images, evaluate, recover
# ════════════════════════════════════════════════════════════════════════════

def ids_to_image_batch(ids_per_image: np.ndarray, n_bits: int = N_BITS) -> torch.Tensor:
    """ids_per_image: [B, 16] int → image tensor [B, 3, 16, 16].

    Mirrors SentencePieceBitDataset.__getitem__ exactly so the model sees the
    same input layout as during training.
    """
    B, n_patches = ids_per_image.shape
    assert n_patches == 16, "expected 16 tokens per image"

    # Encode each ID to ±1 bits, LSB-first
    flat_ids = ids_per_image.reshape(-1)
    bits = SentencePieceBitDataset.ids_to_bits(flat_ids, n_bits)  # [B*16, n_bits]
    bits = bits.reshape(B, n_patches, n_bits)

    # Pad to 48 floats per patch
    padded = np.zeros((B, n_patches, 48), dtype=np.float32)
    padded[:, :, :n_bits] = bits

    # Reshape & stitch into (B, 3, 16, 16) image. Mirrors the dataset code:
    #   patches = padded.reshape(n_patches, 3, 4, 4)
    #   img = patches.reshape(gh, gw, 3, 4, 4).transpose(2, 0, 3, 1, 4).reshape(3, H, W)
    patches = padded.reshape(B, n_patches, 3, 4, 4)
    img = patches.reshape(B, 4, 4, 3, 4, 4)         # B, gh, gw, C, ph, pw
    img = img.transpose(0, 3, 1, 4, 2, 5)            # B, C, gh, ph, gw, pw
    img = img.reshape(B, 3, 16, 16)
    return torch.from_numpy(img)


@torch.no_grad()
def reconstruct_and_recover(ids_per_image: np.ndarray, batch_size: int = 256
                             ) -> tuple:
    """Run the full pipeline: ids → image → model → reconstructed image →
    decoded bits → recovered ids. Returns (orig_bits, recon_bits, recovered_ids).
    All as numpy arrays."""
    all_orig_bits = []
    all_recon_bits = []
    all_recovered = []
    n_imgs = len(ids_per_image)
    for start in range(0, n_imgs, batch_size):
        chunk_ids = ids_per_image[start:start + batch_size]
        img = ids_to_image_batch(chunk_ids).to(device)
        out = model(img)
        recon = out['recon']

        orig_bits = decode_image_to_tokens(img, n_bits=N_BITS)        # [B,16,16]
        recon_bits = decode_image_to_tokens(recon, n_bits=N_BITS)
        recovered = SentencePieceBitDataset.bits_to_ids(recon_bits.cpu().numpy())

        all_orig_bits.append(orig_bits.cpu().numpy())
        all_recon_bits.append(recon_bits.cpu().numpy())
        all_recovered.append(recovered)

    return (np.concatenate(all_orig_bits, axis=0),
            np.concatenate(all_recon_bits, axis=0),
            np.concatenate(all_recovered, axis=0))


def conditional_per_bit_accuracy(orig_bits: np.ndarray, recon_bits: np.ndarray
                                  ) -> dict:
    """Split per-bit accuracy by source-bit-value.

    For each bit position, compute:
      acc_when_1: accuracy when the source bit was +1
      acc_when_0: accuracy when the source bit was -1
      n_when_1, n_when_0: support counts
      worst_class_acc: min(acc_when_0, acc_when_1) — the metric that detects
                       trivial-majority-class predictors

    Returns dict[bit_idx] = {acc_when_0, acc_when_1, n_when_0, n_when_1,
                             worst_class_acc, marginal_freq_of_1}
    """
    orig_signs = (orig_bits > 0)   # [B, 16, 16] bool, True=+1
    recon_signs = (recon_bits > 0)
    correct = (orig_signs == recon_signs)

    # Flatten over [B, 16] (token instances)
    orig_flat = orig_signs.reshape(-1, N_BITS)        # [B*16, 16]
    correct_flat = correct.reshape(-1, N_BITS)

    out = {}
    for b in range(N_BITS):
        is_1 = orig_flat[:, b]
        is_0 = ~is_1
        n_1 = is_1.sum()
        n_0 = is_0.sum()
        acc_1 = float(correct_flat[is_1, b].mean()) if n_1 > 0 else float('nan')
        acc_0 = float(correct_flat[is_0, b].mean()) if n_0 > 0 else float('nan')
        worst = min(a for a in (acc_0, acc_1) if not np.isnan(a))
        out[b] = {
            'acc_when_0': acc_0,
            'acc_when_1': acc_1,
            'n_when_0': int(n_0),
            'n_when_1': int(n_1),
            'worst_class_acc': worst,
            'marginal_freq_of_1': float(n_1) / (n_0 + n_1),
        }
    return out


def report_conditional(label: str, cond: dict, n_tokens: int):
    print(f"\n=== {label} ===  (n_tokens={n_tokens:,})")
    print(f"  bit  freq(1)   acc|0    acc|1    n|0       n|1       worst")
    print(f"  ---  -------   -----    -----    -----     -----     -----")
    for b in range(N_BITS):
        c = cond[b]
        f1 = c['marginal_freq_of_1']
        a0 = c['acc_when_0']
        a1 = c['acc_when_1']
        n0 = c['n_when_0']
        n1 = c['n_when_1']
        w = c['worst_class_acc']
        # Format acc_when_X as "n/a" if no support
        s0 = f"{a0:.4f}" if not np.isnan(a0) else "  n/a "
        s1 = f"{a1:.4f}" if not np.isnan(a1) else "  n/a "
        # Flag genuinely failing positions (skipping "constant bit" cases
        # where one class has zero support — those are structurally trivial,
        # not a model failure)
        has_both_classes = n0 > 0 and n1 > 0
        flag = " ⚠" if (has_both_classes and w < 0.95) else ""
        marker = " *constant*" if not has_both_classes else ""
        print(f"  {b:>3d}  {f1:.4f}   {s0}   {s1}   {n0:>7d}   {n1:>7d}   "
              f"{w:.4f}{flag}{marker}")


# ════════════════════════════════════════════════════════════════════════════
# TEST A — Full val-set evaluation (32K tokens)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 76)
print("TEST A — full val-set evaluation (32K tokens)")
print("═" * 76)

print("Building val dataset (mirrors training-time val)...")
val_ds = SentencePieceBitDataset(
    size=2_000, img_size=16,
    tokenizer_id='google-t5/t5-base',
    corpus_id='wikitext-2-raw-v1',
    n_bits=N_BITS, seed=999,
)

# Pull all 2000 val images' source IDs by replaying __getitem__
val_ids = np.zeros((len(val_ds), 16), dtype=np.int32)
for i in range(len(val_ds)):
    img, _ = val_ds[i]
    decoded = decode_image_to_tokens(img.unsqueeze(0), n_bits=N_BITS).numpy()[0]
    val_ids[i] = SentencePieceBitDataset.bits_to_ids(decoded)
print(f"  {val_ids.shape[0]:,} images × {val_ids.shape[1]} tokens "
      f"= {val_ids.size:,} tokens to evaluate")

t0 = time.time()
orig_A, recon_A, recovered_A = reconstruct_and_recover(val_ids)
print(f"  forward pass: {time.time()-t0:.1f}s")

# Standard metrics
m_A = token_bit_recovery_metrics(
    torch.from_numpy(orig_A), torch.from_numpy(recon_A))
print(f"\n  per_bit_acc:       {m_A['per_bit_acc']:.6f}")
print(f"  token_exact_rate:  {m_A['token_exact_rate']:.6f}")
print(f"  exact ids match:   {(recovered_A == val_ids).all()}")
n_id_correct = (recovered_A == val_ids).sum()
n_id_total = val_ids.size
print(f"  ids correct:       {n_id_correct:,} / {n_id_total:,} "
      f"({100 * n_id_correct / n_id_total:.4f}%)")

cond_A = conditional_per_bit_accuracy(orig_A, recon_A)
report_conditional("TEST A conditional per-bit accuracy", cond_A, val_ids.size)


# ════════════════════════════════════════════════════════════════════════════
# TEST B — Conditional per-bit (already shown above — same numbers)
# ════════════════════════════════════════════════════════════════════════════
# Test B IS the conditional table from Test A. Repeated here as the explicit
# answer to "is each bit position non-trivially discriminating".
#
# Read the table:
#   - acc|0 = accuracy when source bit was -1 (zero in binary)
#   - acc|1 = accuracy when source bit was +1 (one in binary)
#   - worst = min of the two — this is the non-trivial accuracy
#   - flagged ⚠ if worst < 0.95
#
# Bit 15 with vocab=32000 should show n|1 = 0 (constant zero across all data).
# Bits with skewed marginal freq AND high worst-class accuracy = genuinely
# discriminating. Skewed marginal freq AND low worst-class accuracy = model
# is exploiting the skew.


# ════════════════════════════════════════════════════════════════════════════
# TEST C — Adversarial uniform-random IDs
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 76)
print("TEST C — Adversarial uniform-random IDs (every bit ~50/50)")
print("═" * 76)

# Generate 2000 images of 16 random IDs uniform in [0, vocab)
rng = np.random.default_rng(seed=12345)
adversarial_ids = rng.integers(0, VOCAB_SIZE, size=(2000, 16), dtype=np.int32)
print(f"  {adversarial_ids.size:,} uniformly-random tokens")
print(f"  ID range: [{adversarial_ids.min()}, {adversarial_ids.max()}]")
# Sanity: marginal bit freq should be ~0.5 across all bits except bit 15
adv_bits = SentencePieceBitDataset.ids_to_bits(
    adversarial_ids.reshape(-1), N_BITS)
adv_bits_signed = (adv_bits > 0).astype(np.float32)
print(f"  Marginal P(bit=1) per position:")
for b in range(N_BITS):
    print(f"    bit {b:>2d}: {adv_bits_signed[:, b].mean():.4f}")

t0 = time.time()
orig_C, recon_C, recovered_C = reconstruct_and_recover(adversarial_ids)
print(f"\n  forward pass: {time.time()-t0:.1f}s")

m_C = token_bit_recovery_metrics(
    torch.from_numpy(orig_C), torch.from_numpy(recon_C))
print(f"\n  per_bit_acc:       {m_C['per_bit_acc']:.6f}")
print(f"  token_exact_rate:  {m_C['token_exact_rate']:.6f}")
n_id_correct_C = (recovered_C == adversarial_ids).sum()
print(f"  ids correct:       {n_id_correct_C:,} / {adversarial_ids.size:,} "
      f"({100 * n_id_correct_C / adversarial_ids.size:.4f}%)")

cond_C = conditional_per_bit_accuracy(orig_C, recon_C)
report_conditional("TEST C conditional per-bit accuracy "
                   "(uniform-random IDs)", cond_C, adversarial_ids.size)


# ════════════════════════════════════════════════════════════════════════════
# TEST D — Rare-region IDs only (IDs in upper half of vocab)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 76)
print("TEST D — Rare-region IDs only (upper half of vocab, bit 14 forced ON)")
print("═" * 76)
# IDs in [16384, 32000) — bit 14 always 1 in this range
rare_ids = rng.integers(16384, VOCAB_SIZE, size=(500, 16), dtype=np.int32)
print(f"  {rare_ids.size:,} tokens with IDs in [16384, {VOCAB_SIZE})")
print(f"  All have bit 14 = 1 (where in real text it's almost always 0)")

t0 = time.time()
orig_D, recon_D, recovered_D = reconstruct_and_recover(rare_ids)
print(f"  forward pass: {time.time()-t0:.1f}s")

m_D = token_bit_recovery_metrics(
    torch.from_numpy(orig_D), torch.from_numpy(recon_D))
print(f"\n  per_bit_acc:       {m_D['per_bit_acc']:.6f}")
print(f"  token_exact_rate:  {m_D['token_exact_rate']:.6f}")
n_id_correct_D = (recovered_D == rare_ids).sum()
print(f"  ids correct:       {n_id_correct_D:,} / {rare_ids.size:,} "
      f"({100 * n_id_correct_D / rare_ids.size:.4f}%)")

cond_D = conditional_per_bit_accuracy(orig_D, recon_D)
report_conditional("TEST D conditional per-bit accuracy "
                   "(rare-region IDs)", cond_D, rare_ids.size)


# ════════════════════════════════════════════════════════════════════════════
# Summary verdict
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 76)
print("VERDICT")
print("═" * 76)

# A bit position is "trivially solved" if its worst-class accuracy is < 95%
# AND its marginal frequency is < 5% or > 95% (one class dominates).
def trivial_or_genuine(cond, label):
    print(f"\n{label}:")
    n_genuine = 0
    n_trivial = 0
    n_perfect = 0
    n_constant = 0
    for b, c in cond.items():
        has_both = c['n_when_0'] > 0 and c['n_when_1'] > 0
        if not has_both:
            n_constant += 1
            print(f"  bit {b:>2d}  freq(1)={c['marginal_freq_of_1']:.4f}  "
                  f"*constant in this test set — cannot evaluate*")
            continue
        is_skewed = c['marginal_freq_of_1'] < 0.05 or c['marginal_freq_of_1'] > 0.95
        worst = c['worst_class_acc']
        if worst >= 0.99:
            n_perfect += 1
            tag = "✓ perfect (genuine)"
        elif worst >= 0.95:
            n_genuine += 1
            tag = "✓ strong (genuine)"
        elif is_skewed:
            n_trivial += 1
            tag = "⚠ trivial-leaning (skewed input + weak minority recovery)"
        else:
            n_trivial += 1
            tag = "✗ failing"
        print(f"  bit {b:>2d}  worst={worst:.4f}  freq(1)={c['marginal_freq_of_1']:.4f}  {tag}")
    print(f"  → {n_perfect} perfect, {n_genuine} strong, "
          f"{n_trivial} trivial/failing, {n_constant} constant")

trivial_or_genuine(cond_A, "TEST A (real corpus)")
trivial_or_genuine(cond_C, "TEST C (uniform-random IDs)")
trivial_or_genuine(cond_D, "TEST D (rare-region IDs)")

print("\n" + "═" * 76)
print("Interpretation guide:")
print("  TEST A all-perfect: model handles real text bit reconstruction "
      "trivially — could be exploiting Zipfian skew, can't tell from this alone")
print("  TEST C all-perfect: model handles ALL bit patterns "
      "non-trivially — the substrate genuinely encodes bit information")
print("  TEST C drops at MSB: model exploits Zipfian skew, falls apart "
      "when bit distribution is uniform")
print("  TEST D 100% on bit 14: model can flip bit 14 to +1 when needed, "
      "not just always output -1")
print("═" * 76)