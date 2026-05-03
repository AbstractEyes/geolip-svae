"""
prototypes.001_vocab_trigram_recall.dataset
============================================
Vocabulary-aware byte-trigram dataset.

Encoding is identical to ``geolip_svae.dataset_presets.ByteTrigramDataset``
(utf-8 bytes packed 3-per-pixel as RGB, mapped from [0, 255] to [-1, +1]).
The novelty is structural awareness — each sample additionally carries the
list of byte offsets where each sentencepiece token starts in its window,
plus the originating vocab token ids. Eval can then ask "did the multi-byte
token come back as a unit" rather than just "did each byte come back".

The dataset is registered into ``DATASET_FACTORIES`` only at runtime by
the experiment's ``run.py`` — never by importing this module. See the
prototypes/ contract.

Usage
-----
    from prototypes.001_vocab_trigram_recall.dataset import (
        VocabTrigramDataset, vocab_trigram_factory,
    )

    ds = VocabTrigramDataset(
        corpus='wikitext-2-raw-v1',
        tokenizer='google-t5/t5-base',
        img_size=64, patch_size=4, channels=3,
        n_samples=10_000, seed=0,
    )
    img, _ = ds[0]                         # trainer-compatible: (image, label)
    boundaries = ds.get_boundaries(0)      # numpy array of byte offsets
    token_ids  = ds.get_token_ids(0)       # numpy array of vocab ids

The eval module consumes get_boundaries/get_token_ids alongside the model's
recon to compute per-token recovery metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Tokenizer + corpus loaders ──────────────────────────────────────

def _load_tokenizer(name: str):
    """Load an HF AutoTokenizer (handles sentencepiece, BPE, and wordpiece)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def _load_corpus_text(corpus_id: str, split: str = 'train',
                       max_chars: Optional[int] = None) -> str:
    """Load a corpus from HF datasets and return as a single concatenated string."""
    from datasets import load_dataset
    if '/' in corpus_id:
        repo, config = corpus_id.split('/', 1)
        ds = load_dataset(repo, config, split=split)
    else:
        # wikitext-2-raw-v1 etc. — known shorthand for the wikitext repo
        if corpus_id.startswith('wikitext'):
            ds = load_dataset('Salesforce/wikitext', corpus_id, split=split)
        else:
            ds = load_dataset(corpus_id, split=split)
    out_parts: List[str] = []
    total = 0
    for row in ds:
        text = row.get('text') or row.get('content') or ''
        if not text:
            continue
        out_parts.append(text)
        total += len(text)
        if max_chars is not None and total >= max_chars:
            break
    return '\n'.join(out_parts)


# ── Token-aware byte stream ─────────────────────────────────────────

@dataclass
class TokenizedStream:
    """A flat utf-8 byte stream paired with token-boundary metadata.

    Attributes
    ----------
    bytes_arr : np.ndarray (uint8)
        Concatenated utf-8 bytes of the corpus, in tokenizer order.
    token_starts : np.ndarray (int64)
        Byte offsets where each token begins. Length = n_tokens + 1
        (the trailing entry equals ``len(bytes_arr)`` so consecutive
        slicing always works).
    token_ids : np.ndarray (int64)
        Vocab id of each token. Length = n_tokens.
    vocab_size : int
    """
    bytes_arr: np.ndarray
    token_starts: np.ndarray
    token_ids: np.ndarray
    vocab_size: int


def _build_tokenized_stream(text: str, tokenizer) -> TokenizedStream:
    """Tokenize text and emit a byte stream plus per-token boundaries.

    Each token's byte representation is its decoded text in utf-8.
    Whitespace handling matches the tokenizer's own `decode` — i.e. for
    sentencepiece this includes the leading-space marker as a real space.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=False)
    ids: List[int] = enc['input_ids']
    pieces: List[str] = []
    for tid in ids:
        piece = tokenizer.decode([tid], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        pieces.append(piece)

    # Convert each piece to utf-8 bytes; record start offsets.
    byte_chunks: List[bytes] = []
    starts: List[int] = []
    cursor = 0
    for p in pieces:
        b = p.encode('utf-8')
        starts.append(cursor)
        byte_chunks.append(b)
        cursor += len(b)
    starts.append(cursor)  # sentinel for trailing slice
    bytes_arr = np.frombuffer(b''.join(byte_chunks), dtype=np.uint8).copy()
    return TokenizedStream(
        bytes_arr=bytes_arr,
        token_starts=np.asarray(starts, dtype=np.int64),
        token_ids=np.asarray(ids, dtype=np.int64),
        vocab_size=tokenizer.vocab_size,
    )


# ── Dataset ─────────────────────────────────────────────────────────

class VocabTrigramDataset(Dataset):
    """Vocabulary-aware byte-trigram image dataset.

    Each sample is a (channels, img_size, img_size) image whose flattened
    byte sequence (length = channels * img_size * img_size) is a contiguous
    window of utf-8 bytes from the tokenized corpus. Pixel values are
    ``2.0 * byte / 255.0 - 1.0`` to match ByteTrigramDataset's convention.

    Per-sample boundary / token-id metadata is stored separately and
    accessed via ``get_boundaries(idx)`` and ``get_token_ids(idx)`` —
    keeping ``__getitem__`` returning the standard ``(image, label)``
    tuple so the existing trainer's collate / loop work without changes.
    """

    def __init__(
        self,
        corpus: str = 'wikitext-2-raw-v1',
        tokenizer: str = 'google-t5/t5-base',
        img_size: int = 64,
        patch_size: int = 4,
        channels: int = 3,
        n_samples: int = 10_000,
        seed: int = 0,
        max_corpus_chars: Optional[int] = 4_000_000,
        split: str = 'train',
        stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        if channels != 3:
            # Could generalize, but the whole point of trigram packing is
            # 3-byte-per-pixel via RGB — non-3 deserves a different scheme.
            raise ValueError(
                f"VocabTrigramDataset only supports channels=3; got {channels}. "
                f"For non-3 channel experiments, design a separate encoding."
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        self.n_samples = n_samples
        self.seed = seed
        self.bytes_per_image = channels * img_size * img_size

        # Window stride defaults to one full image apart (no overlap),
        # which gives reproducible per-sample offset arithmetic.
        self.stride = stride if stride is not None else self.bytes_per_image

        # 1. Load corpus + tokenizer
        text = _load_corpus_text(corpus, split=split, max_chars=max_corpus_chars)
        tok = _load_tokenizer(tokenizer)
        self.tokenizer_name = tokenizer

        # 2. Tokenize and build the byte stream
        self.stream = _build_tokenized_stream(text, tok)
        self.vocab_size = self.stream.vocab_size
        if len(self.stream.bytes_arr) < self.bytes_per_image:
            raise RuntimeError(
                f"Corpus too small: {len(self.stream.bytes_arr)} bytes < "
                f"one image's capacity ({self.bytes_per_image} bytes). "
                f"Increase max_corpus_chars or shrink img_size/channels."
            )

        # 3. Pre-compute per-sample byte windows (random offsets, fixed seed)
        rng = np.random.default_rng(seed)
        max_offset = len(self.stream.bytes_arr) - self.bytes_per_image
        self._offsets = rng.integers(0, max_offset + 1, size=n_samples,
                                      dtype=np.int64)

        # 4. Pre-compute per-sample token boundary slices.
        # For each sample we record the indices into stream.token_starts of
        # the tokens whose first byte falls within [offset, offset+capacity).
        # The boundary array per sample is then
        # stream.token_starts[lo:hi+1] - offset (relative to the image).
        self._sample_lo = np.empty(n_samples, dtype=np.int64)
        self._sample_hi = np.empty(n_samples, dtype=np.int64)
        ts = self.stream.token_starts
        # token_starts is sorted. For each sample, find the first token
        # whose start >= offset, and the first token whose start >= offset+cap.
        offsets_end = self._offsets + self.bytes_per_image
        self._sample_lo[:] = np.searchsorted(ts, self._offsets, side='left')
        self._sample_hi[:] = np.searchsorted(ts, offsets_end, side='left')

    # ── Trainer-compatible interface ────────────────────────────────

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        offset = int(self._offsets[idx])
        b = self.stream.bytes_arr[offset:offset + self.bytes_per_image]
        # Map [0, 255] -> [-1, +1] (matches ByteTrigramDataset convention)
        x = (b.astype(np.float32) * (2.0 / 255.0)) - 1.0
        x = x.reshape(self.channels, self.img_size, self.img_size)
        return torch.from_numpy(x), 0

    # ── Token-awareness API (used by eval, NOT by trainer) ──────────

    def get_boundaries(self, idx: int) -> np.ndarray:
        """Byte offsets within sample ``idx`` where each contained token starts.

        The returned array is monotonically increasing, lies in
        ``[0, bytes_per_image]``, and includes a trailing sentinel equal to
        the offset where the LAST contained token *ends* (or
        ``bytes_per_image`` if the last token spills past the window — in
        which case it's clipped, see ``get_token_ids``).
        """
        offset = int(self._offsets[idx])
        lo = int(self._sample_lo[idx])
        hi = int(self._sample_hi[idx])
        if lo >= hi:
            return np.empty(0, dtype=np.int64)
        # token_starts[lo:hi+1] gives starts plus one trailing position
        # (the start of the next token after the window's last token).
        bounds = self.stream.token_starts[lo:hi + 1] - offset
        # Clip the trailing edge to the window (last token may overflow)
        bounds[-1] = min(int(bounds[-1]), self.bytes_per_image)
        return bounds

    def get_token_ids(self, idx: int) -> np.ndarray:
        """Vocab ids of the tokens fully contained in sample ``idx``.

        A token "in" a sample is one whose START offset falls inside the
        window — its trailing bytes may extend past the window end, in
        which case the last-token boundary returned by ``get_boundaries``
        is clipped to ``bytes_per_image``. Eval should treat such tokens as
        partial-recovery only (length-clipped prefix).
        """
        lo = int(self._sample_lo[idx])
        hi = int(self._sample_hi[idx])
        return self.stream.token_ids[lo:hi]


# ── Factory for runtime registration into DATASET_FACTORIES ─────────

def vocab_trigram_factory(cfg: Dict[str, Any], channels: int = 3):
    """Build a DatasetBundle for the VocabTrigramDataset.

    Cfg keys consumed (all optional except those marked):
        img_size           (REQUIRED, from preset)
        patch_size         (REQUIRED, from preset)
        batch_size         (REQUIRED, from preset)
        ds_size            train sample count             (default 10_000)
        val_size           test sample count              (default 1_000)
        vt_corpus          HF datasets id                 (default 'wikitext-2-raw-v1')
        vt_tokenizer       HF model id w/ tokenizer       (default 'google-t5/t5-base')
        vt_max_corpus_chars cap on corpus characters      (default 4_000_000)
        vt_split           HF split for train             (default 'train')
        vt_test_split      HF split for test              (default 'test')
        vt_seed            base RNG seed                  (default 0)
    """
    from geolip_svae.dataset_presets import DatasetBundle

    img_size   = cfg['img_size']
    patch_size = cfg['patch_size']
    batch_size = cfg['batch_size']

    train_ds = VocabTrigramDataset(
        corpus=cfg.get('vt_corpus', 'wikitext-2-raw-v1'),
        tokenizer=cfg.get('vt_tokenizer', 'google-t5/t5-base'),
        img_size=img_size, patch_size=patch_size, channels=channels,
        n_samples=cfg.get('ds_size', 10_000),
        seed=cfg.get('vt_seed', 0),
        max_corpus_chars=cfg.get('vt_max_corpus_chars', 4_000_000),
        split=cfg.get('vt_split', 'train'),
    )
    test_ds = VocabTrigramDataset(
        corpus=cfg.get('vt_corpus', 'wikitext-2-raw-v1'),
        tokenizer=cfg.get('vt_tokenizer', 'google-t5/t5-base'),
        img_size=img_size, patch_size=patch_size, channels=channels,
        n_samples=cfg.get('val_size', 1_000),
        seed=cfg.get('vt_seed', 0) + 9999,
        max_corpus_chars=cfg.get('vt_max_corpus_chars', 4_000_000),
        split=cfg.get('vt_test_split', 'test'),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.get('num_workers', 2),
        pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.get('num_workers', 2),
        pin_memory=True, drop_last=False,
    )

    bundle = DatasetBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        # Tag as byte_trigram so the trainer's existing per-batch byte-recovery
        # reporting fires; token-level metrics live in eval.py and run
        # post-train (or as a hook in run.py).
        is_byte_trigram=True,
        is_text=False, is_image=False, is_noise=False,
        is_tree=False, is_sentencepiece=False,
        extra={
            'tokenizer_name': train_ds.tokenizer_name,
            'vocab_size': train_ds.vocab_size,
        },
    )
    return bundle
