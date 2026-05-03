"""
prototypes.001_vocab_trigram_recall.eval
=========================================
Token-level recovery metrics for the vocabulary trigram experiment.

Computes:
    per_byte_acc          - baseline; matches byte_trigram_dataset's metric
    per_token_exact_acc   - fraction of tokens whose every byte recovered
    per_token_prefix_acc  - mean correct-prefix length / token length
    per_token_id_recovery - did recovered bytes round-trip via tokenizer
                             back to the same vocab id
    top_corrupted_tokens  - vocab ids most often lost to recon error
    confusion_pairs       - (orig_id -> recovered_id) pairs that occur
                             most often when recon fails the exact match

Usage
-----
    from prototypes.001_vocab_trigram_recall.eval import run_vocab_eval

    report = run_vocab_eval(
        model, dataset, device='cuda',
        n_samples=512, batch_size=32,
    )
    print(report.summary())
    report.save('vocab_eval_report.json')
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import VocabTrigramDataset


# ── Pixel <-> byte conversion (matches dataset encoding) ────────────

def _pixels_to_bytes(x: torch.Tensor) -> np.ndarray:
    """Invert the dataset's [0,255] -> [-1,+1] mapping. Returns uint8 array
    of shape (B, channels*H*W) in row-major (channel-first) order."""
    # x: (B, C, H, W) in [-1, +1]
    x_np = x.detach().cpu().numpy()
    b = ((x_np + 1.0) * (255.0 / 2.0)).round().clip(0, 255).astype(np.uint8)
    B = b.shape[0]
    return b.reshape(B, -1)


# ── Per-sample recovery summary ─────────────────────────────────────

@dataclass
class SampleRecovery:
    sample_idx: int
    n_bytes: int
    n_bytes_correct: int
    tokens: List[Dict[str, Any]] = field(default_factory=list)
    # tokens entries: {token_id, orig_bytes, recon_bytes, exact, prefix_len,
    #                  recovered_token_ids}


# ── Aggregated report ───────────────────────────────────────────────

@dataclass
class VocabEvalReport:
    n_samples: int
    n_tokens: int
    per_byte_acc: float
    per_token_exact_acc: float
    per_token_prefix_acc: float
    per_token_id_recovery: float
    top_corrupted_tokens: List[Tuple[int, int]]   # (token_id, miss_count)
    top_confusion_pairs: List[Tuple[Tuple[int, int], int]]  # ((orig, rec), count)
    tokenizer_name: str
    vocab_size: int
    notes: str = ''

    def summary(self) -> str:
        lines = [
            f"VocabEvalReport — {self.n_samples} samples, {self.n_tokens} tokens",
            f"  per_byte_acc          : {self.per_byte_acc * 100:.3f}%",
            f"  per_token_exact_acc   : {self.per_token_exact_acc * 100:.3f}%",
            f"  per_token_prefix_acc  : {self.per_token_prefix_acc * 100:.3f}%",
            f"  per_token_id_recovery : {self.per_token_id_recovery * 100:.3f}%",
            f"  vocab={self.vocab_size}, tokenizer={self.tokenizer_name}",
        ]
        if self.top_corrupted_tokens:
            lines.append("  Top corrupted tokens (id, miss_count):")
            for tid, n in self.top_corrupted_tokens[:8]:
                lines.append(f"    {tid:>6d}  ×{n}")
        if self.top_confusion_pairs:
            lines.append("  Top confusion pairs ((orig→recon), count):")
            for (a, b), n in self.top_confusion_pairs[:8]:
                lines.append(f"    {a:>6d} → {b:>6d}  ×{n}")
        return '\n'.join(lines)

    def save(self, path: str) -> None:
        d = asdict(self)
        # JSON wants list-of-lists, not list-of-tuples for confusion_pairs
        d['top_confusion_pairs'] = [
            [list(pair), n] for pair, n in self.top_confusion_pairs
        ]
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)


# ── Driver ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_vocab_eval(
    model,
    dataset: VocabTrigramDataset,
    device: str = 'cuda',
    n_samples: Optional[int] = None,
    batch_size: int = 32,
    notes: str = '',
) -> VocabEvalReport:
    """Run the model on `n_samples` from `dataset` and compute token-level
    recovery metrics. ``n_samples=None`` evaluates the full dataset.

    The model must accept (B, C, H, W) and return a dict with key 'recon'.
    """
    if n_samples is None:
        n_samples = len(dataset)
    n_samples = min(n_samples, len(dataset))

    # Lazy tokenizer load (only needed for token-id round-trip)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(dataset.tokenizer_name, use_fast=True)

    # Per-byte counters
    total_bytes = 0
    correct_bytes = 0
    # Per-token counters
    n_tokens = 0
    n_tokens_exact = 0
    prefix_sum = 0.0          # accumulator for per-token-prefix-acc
    n_tokens_id_recovered = 0
    miss_counter: Counter = Counter()
    confusion_counter: Counter = Counter()

    model.eval()
    model_device = next(model.parameters()).device

    # Iterate sequentially so per-sample metadata aligns with batches.
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_idx_list = list(range(batch_start, batch_end))

        # Stack inputs manually (avoid DataLoader shuffling re-ordering metadata)
        imgs = torch.stack([dataset[i][0] for i in batch_idx_list], dim=0)
        imgs = imgs.to(model_device, non_blocking=True)
        out = model(imgs)
        recon = out['recon']

        orig_bytes = _pixels_to_bytes(imgs)
        recon_bytes = _pixels_to_bytes(recon)

        for local_i, sample_i in enumerate(batch_idx_list):
            ob = orig_bytes[local_i]
            rb = recon_bytes[local_i]
            byte_match = (ob == rb)
            total_bytes += byte_match.size
            correct_bytes += int(byte_match.sum())

            bounds = dataset.get_boundaries(sample_i)
            tids = dataset.get_token_ids(sample_i)
            if len(tids) == 0:
                continue

            # bounds has length len(tids) + 1
            for ti, tid in enumerate(tids):
                start = int(bounds[ti])
                end = int(bounds[ti + 1])
                if end <= start:
                    continue
                ob_tok = ob[start:end]
                rb_tok = rb[start:end]

                exact = bool(np.array_equal(ob_tok, rb_tok))
                # Prefix length: longest matching head
                eq = (ob_tok == rb_tok)
                first_mismatch = int(np.argmin(eq)) if (not eq.all()) else len(eq)
                prefix_len = first_mismatch  # 0..len
                prefix_sum += prefix_len / max(1, len(eq))
                n_tokens += 1
                if exact:
                    n_tokens_exact += 1
                else:
                    miss_counter[int(tid)] += 1

                # Token-id recovery: round-trip recon bytes through tokenizer
                # and check if the FIRST recovered token id matches.
                try:
                    rb_str = bytes(rb_tok).decode('utf-8', errors='replace')
                    rec_ids = tok(rb_str, add_special_tokens=False)['input_ids']
                except Exception:
                    rec_ids = []
                if rec_ids and rec_ids[0] == int(tid):
                    n_tokens_id_recovered += 1
                else:
                    rec_id = rec_ids[0] if rec_ids else -1
                    confusion_counter[(int(tid), int(rec_id))] += 1

    per_byte_acc = correct_bytes / max(1, total_bytes)
    per_token_exact = n_tokens_exact / max(1, n_tokens)
    per_token_prefix = prefix_sum / max(1, n_tokens)
    per_token_id_rec = n_tokens_id_recovered / max(1, n_tokens)

    return VocabEvalReport(
        n_samples=n_samples,
        n_tokens=n_tokens,
        per_byte_acc=per_byte_acc,
        per_token_exact_acc=per_token_exact,
        per_token_prefix_acc=per_token_prefix,
        per_token_id_recovery=per_token_id_rec,
        top_corrupted_tokens=miss_counter.most_common(32),
        top_confusion_pairs=confusion_counter.most_common(32),
        tokenizer_name=dataset.tokenizer_name,
        vocab_size=dataset.vocab_size,
        notes=notes,
    )
