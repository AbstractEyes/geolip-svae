"""
svae_proto.exp_001_vocab_trigram_recall.dataset
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
svae_proto/ contract.

Usage
-----
    from svae_proto.exp_001_vocab_trigram_recall.dataset import (
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

import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Diagnostic timing + memory probe ────────────────────────────────
# psutil ships with Colab. Fall back to resource.getrusage on any machine
# where psutil is missing — both report process RSS.

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def _rss_gb() -> float:
        return _PROC.memory_info().rss / (1024 ** 3)
except Exception:
    import resource
    def _rss_gb() -> float:
        # Linux: ru_maxrss is kilobytes; macOS: bytes.
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (1024 ** 2)


class _Step:
    """Context manager that prints elapsed wall time + RSS delta around a
    block. Output goes to stderr with explicit flush so prints don't get
    buffered behind a long computation. Format::

        [vocab_trigram] <label>... done in 12.3s (RSS 5.42 GB, +1.20 GB)
    """
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> '_Step':
        self.t0 = time.time()
        self.rss0 = _rss_gb()
        sys.stderr.write(f'  [vocab_trigram] {self.label}... ')
        sys.stderr.flush()
        return self

    def __exit__(self, *exc) -> None:
        dt = time.time() - self.t0
        rss = _rss_gb()
        sys.stderr.write(
            f'done in {dt:.1f}s (RSS {rss:.2f} GB, '
            f'{"+"if rss>=self.rss0 else ""}{rss-self.rss0:+.2f} GB)\n'
        )
        sys.stderr.flush()


# ── Tokenizer + corpus loaders ──────────────────────────────────────

def _load_tokenizer(name: str):
    """Load an HF AutoTokenizer (handles sentencepiece, BPE, and wordpiece)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def _load_corpus_text(corpus_id: str, split: str = 'train',
                       max_chars: Optional[int] = None) -> str:
    """Load a corpus from HF datasets and return as a single concatenated string."""
    from datasets import load_dataset
    with _Step(f"load_dataset({corpus_id!r}, split={split!r})"):
        if '/' in corpus_id:
            repo, config = corpus_id.split('/', 1)
            ds = load_dataset(repo, config, split=split)
        else:
            if corpus_id.startswith('wikitext'):
                ds = load_dataset('Salesforce/wikitext', corpus_id, split=split)
            else:
                ds = load_dataset(corpus_id, split=split)

    with _Step(f"iterate {len(ds):,} rows -> str (max_chars={max_chars})"):
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
        sys.stderr.write(f'(parts={len(out_parts):,}, total_chars={total:,}) ')
        sys.stderr.flush()

    with _Step("join parts -> single str"):
        joined = '\n'.join(out_parts)
        sys.stderr.write(f'(len={len(joined):,}) ')
        sys.stderr.flush()
    del out_parts
    gc.collect()
    return joined


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


_VOCAB_BYTES_CACHE: Dict[str, List[bytes]] = {}
_STREAM_CACHE: Dict[Tuple[str, str, int, Optional[int]], 'TokenizedStream'] = {}


def _vocab_id_to_bytes(tokenizer) -> List[bytes]:
    """Build (or fetch from cache) the lookup [vocab_id -> utf-8 bytes].

    Decodes each vocab token EXACTLY ONCE. ~32128 calls for T5-base
    regardless of corpus size, ~1-2s. Cached at module level keyed on
    the tokenizer's name_or_path.
    """
    key = getattr(tokenizer, 'name_or_path', repr(tokenizer))
    cached = _VOCAB_BYTES_CACHE.get(key)
    if cached is not None:
        sys.stderr.write(f'  [vocab_trigram] vocab_id_to_bytes: cache hit ({key})\n')
        sys.stderr.flush()
        return cached

    with _Step(f"build vocab lookup ({tokenizer.vocab_size} tokens)"):
        vocab_size = tokenizer.vocab_size
        table: List[bytes] = [b''] * vocab_size
        for tid in range(vocab_size):
            s = tokenizer.decode([tid], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
            table[tid] = s.encode('utf-8')
    _VOCAB_BYTES_CACHE[key] = table
    return table


def _build_tokenized_stream(text: str, tokenizer,
                              chunk_chars: int = 2_000_000) -> TokenizedStream:
    """Build byte stream + token boundaries via SOURCE-BYTES + CHUNKED tokenize.

    Memory peak is bounded by the chunk size, NOT the corpus size — for
    full wikitext-103 (~540 MB text, ~150M tokens) peak stays around
    5-6 GB instead of the 30-65 GB the single-shot tokenize hit.

    Approach:
        1. byte_stream = text.encode('utf-8') ONCE — that's the byte_arr
           we return. Source text bytes ARE the byte stream; no per-token
           reconstruction needed.
        2. Tokenize text in ~2 MB chunks with return_offsets_mapping=True
           so each chunk gives us per-token CHAR offsets within the chunk.
        3. Per chunk: build a small char→byte map (handles multi-byte UTF-8
           chars correctly) and translate local char offsets to GLOBAL byte
           offsets in the full byte stream.
        4. Drop per-chunk intermediates aggressively; concatenate the
           accumulated arrays at the end.

    Why this avoids the 65 GB blowup:
        - No 150M-element Python list of ints (the BatchEncoding only ever
          holds ~600K ints per chunk, immediately copied to numpy).
        - No b''.join over 150M items + no precomputed vocab byte lookup.
        - Single 540 MB allocation for the byte stream, lives once, no
          intermediate bytes_blob copy.
    """
    # Stream-level cache — same tokenizer + same text → reuse.
    cache_key = (
        getattr(tokenizer, 'name_or_path', repr(tokenizer)),
        text[:256],
        len(text),
        None,
    )
    cached_stream = _STREAM_CACHE.get(cache_key)
    if cached_stream is not None:
        sys.stderr.write(
            f'  [vocab_trigram] _build_tokenized_stream: stream cache hit '
            f'({len(cached_stream.bytes_arr):,} bytes, '
            f'{len(cached_stream.token_ids):,} tokens)\n'
        )
        sys.stderr.flush()
        return cached_stream

    n_chars = len(text)
    n_chunks = (n_chars + chunk_chars - 1) // chunk_chars
    sys.stderr.write(
        f'  [vocab_trigram] === build_tokenized_stream START '
        f'(text_len={n_chars:,} chars, '
        f'tokenizer={getattr(tokenizer, "name_or_path", "?")}, '
        f'strategy=source_bytes+chunked_tokenize, '
        f'n_chunks={n_chunks} of ~{chunk_chars:,} chars each) ===\n'
        f'  [vocab_trigram] starting RSS: {_rss_gb():.2f} GB\n'
    )
    sys.stderr.flush()

    # 1. Source byte stream — single allocation.
    with _Step(f"text.encode('utf-8')"):
        text_bytes = text.encode('utf-8')
        n_total_bytes = len(text_bytes)
        sys.stderr.write(f'(total_bytes={n_total_bytes:,}) ')
        sys.stderr.flush()

    with _Step(f"np.frombuffer(text_bytes, uint8).copy()"):
        bytes_arr = np.frombuffer(text_bytes, dtype=np.uint8).copy()
    del text_bytes
    gc.collect()
    sys.stderr.write(
        f'  [vocab_trigram] post-byte-encode RSS: {_rss_gb():.2f} GB\n'
    )
    sys.stderr.flush()

    # 2. Chunked tokenization with offset mapping — bounded peak per chunk.
    chunk_ids: List[np.ndarray] = []
    chunk_byte_starts: List[np.ndarray] = []   # global byte offsets
    char_cursor = 0
    byte_cursor = 0
    t_chunked_start = time.time()

    for chunk_idx in range(n_chunks):
        start_char = char_cursor
        end_char = min(start_char + chunk_chars, n_chars)
        chunk_text = text[start_char:end_char]

        # Tokenize the chunk — small input, single Rust call, small output.
        enc = tokenizer(chunk_text, add_special_tokens=False,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                        return_token_type_ids=False)
        local_ids = np.asarray(enc['input_ids'], dtype=np.int64)
        local_offsets = np.asarray(enc['offset_mapping'], dtype=np.int64)
        # local_offsets[:, 0] = char start of each token within chunk_text

        # Per-chunk char→byte map. ~2 MB for a 2 MB chunk. UTF-8 char
        # starts are bytes that are NOT continuation bytes (mask 0xC0 != 0x80).
        chunk_bytes = chunk_text.encode('utf-8')
        chunk_byte_arr = np.frombuffer(chunk_bytes, dtype=np.uint8)
        is_char_start = (chunk_byte_arr & 0xC0) != 0x80
        local_char_byte_starts = np.flatnonzero(is_char_start).astype(np.int64)

        # Sanity: char count from byte-pattern walk must match Python char count
        if len(local_char_byte_starts) != len(chunk_text):
            raise RuntimeError(
                f"chunk {chunk_idx}: char count mismatch — "
                f"{len(local_char_byte_starts)} from utf-8 walk vs "
                f"{len(chunk_text)} from Python str"
            )

        # Translate local CHAR offsets → local BYTE offsets → global byte offsets
        local_char_starts = local_offsets[:, 0]
        # Clamp char offsets to valid range (guards against tokenizer edge cases)
        local_char_starts = np.clip(local_char_starts, 0, len(chunk_text) - 1)
        local_byte_starts = local_char_byte_starts[local_char_starts]
        global_byte_starts = local_byte_starts + byte_cursor

        chunk_ids.append(local_ids)
        chunk_byte_starts.append(global_byte_starts)

        char_cursor = end_char
        byte_cursor += len(chunk_bytes)

        # Drop everything chunk-local before next iteration.
        del enc, local_offsets, chunk_text, chunk_bytes, chunk_byte_arr
        del is_char_start, local_char_byte_starts, local_char_starts
        del local_byte_starts, global_byte_starts, local_ids
        if (chunk_idx + 1) % 50 == 0 or chunk_idx + 1 == n_chunks:
            gc.collect()
            elapsed = time.time() - t_chunked_start
            sys.stderr.write(
                f'  [vocab_trigram] chunked tokenize {chunk_idx+1}/{n_chunks} '
                f'({elapsed:.1f}s, RSS {_rss_gb():.2f} GB)\n'
            )
            sys.stderr.flush()

    sys.stderr.write(
        f'  [vocab_trigram] chunked tokenize DONE in {time.time()-t_chunked_start:.1f}s '
        f'(RSS {_rss_gb():.2f} GB)\n'
    )
    sys.stderr.flush()

    # 3. Concatenate per-chunk arrays into the final buffers.
    with _Step(f"np.concatenate chunk_ids ({len(chunk_ids)} chunks)"):
        ids_arr = np.concatenate(chunk_ids)
        n_tokens = len(ids_arr)
        sys.stderr.write(f'(n_tokens={n_tokens:,}) ')
        sys.stderr.flush()
    del chunk_ids
    gc.collect()

    with _Step(f"np.concatenate chunk_byte_starts + sentinel"):
        token_starts = np.empty(n_tokens + 1, dtype=np.int64)
        token_starts[:-1] = np.concatenate(chunk_byte_starts)
        token_starts[-1] = n_total_bytes
    del chunk_byte_starts
    gc.collect()

    sys.stderr.write(
        f'  [vocab_trigram] === build_tokenized_stream DONE — '
        f'final RSS: {_rss_gb():.2f} GB ===\n'
    )
    sys.stderr.flush()

    stream = TokenizedStream(
        bytes_arr=bytes_arr,
        token_starts=token_starts,
        token_ids=ids_arr,
        vocab_size=tokenizer.vocab_size,
    )
    _STREAM_CACHE[cache_key] = stream
    return stream


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

        sys.stderr.write(
            f'\n  [vocab_trigram] === VocabTrigramDataset.__init__ '
            f'(corpus={corpus!r}, split={split!r}, '
            f'n_samples={n_samples:,}, img={img_size}x{img_size}x{channels}, '
            f'bytes/image={self.bytes_per_image:,}) ===\n'
            f'  [vocab_trigram] starting RSS: {_rss_gb():.2f} GB\n'
        )
        sys.stderr.flush()

        # 1. Load corpus + tokenizer
        text = _load_corpus_text(corpus, split=split, max_chars=max_corpus_chars)
        with _Step(f"AutoTokenizer.from_pretrained({tokenizer!r})"):
            tok = _load_tokenizer(tokenizer)
        self.tokenizer_name = tokenizer

        # 2. Tokenize and build the byte stream
        self.stream = _build_tokenized_stream(text, tok)
        # Drop source text — _build_tokenized_stream copied what it needs.
        del text
        gc.collect()

        self.vocab_size = self.stream.vocab_size
        if len(self.stream.bytes_arr) < self.bytes_per_image:
            raise RuntimeError(
                f"Corpus too small: {len(self.stream.bytes_arr)} bytes < "
                f"one image's capacity ({self.bytes_per_image} bytes). "
                f"Increase max_corpus_chars or shrink img_size/channels."
            )

        # 3. Pre-compute per-sample byte windows (random offsets, fixed seed)
        with _Step(f"sample offsets (n_samples={n_samples:,})"):
            rng = np.random.default_rng(seed)
            max_offset = len(self.stream.bytes_arr) - self.bytes_per_image
            self._offsets = rng.integers(0, max_offset + 1, size=n_samples,
                                          dtype=np.int64)

        # 4. Pre-compute per-sample token boundary slices via two binary
        # searches. searchsorted on n_samples=1M against ~150M token_starts
        # is ~1-2s each; this is the last step that scales with n_samples.
        with _Step(f"searchsorted boundaries (n_samples={n_samples:,} vs "
                    f"n_tokens={len(self.stream.token_starts):,})"):
            self._sample_lo = np.empty(n_samples, dtype=np.int64)
            self._sample_hi = np.empty(n_samples, dtype=np.int64)
            ts = self.stream.token_starts
            offsets_end = self._offsets + self.bytes_per_image
            self._sample_lo[:] = np.searchsorted(ts, self._offsets, side='left')
            self._sample_hi[:] = np.searchsorted(ts, offsets_end, side='left')

        sys.stderr.write(
            f'  [vocab_trigram] === __init__ DONE — '
            f'final RSS: {_rss_gb():.2f} GB '
            f'(stream: {len(self.stream.bytes_arr):,} B, '
            f'{len(self.stream.token_ids):,} tokens; '
            f'{n_samples:,} sample windows pre-computed) ===\n'
        )
        sys.stderr.flush()

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
