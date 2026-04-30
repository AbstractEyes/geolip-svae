"""
geolip_svae.inference.text
==========================
Text-side I/O wrapper for byte-trigram-trained sphere-solvers.

Wraps ``InferenceEngine`` to expose string-level encode / signature /
similarity operations. The model itself operates on ``(3, H, W)`` image
tensors; this module handles the text → image conversion (using the
canonical ``ByteTrigramDataset.bytes_to_image`` layout) and the
patch-level → sentence-level pooling.

Three signature modes, each measuring a different aspect of what the
byte-trigram codebook captures:

    'omega':       Cross-attention-coordinated singular values (S),
                   pooled across patches → [D]. The model's "omega tokens."
                   Smallest signature; spectral fingerprint of the text.
    'omega_orig':  Pre-cross-attn singular values (S_orig),
                   pooled across patches → [D]. Useful for comparison
                   against 'omega' to diagnose cross-attn contribution.
    'codebook':    Codebook-axis activations (sum_abs over V rows per patch),
                   pooled across patches → [n_axes]. Activations on the
                   polytope-class axes the model uses to organize byte
                   structure. Requires ``engine.attach_codebook(cb)`` first.

What this kind of similarity captures
-------------------------------------
The byte_trigram model is trained on byte reconstruction from raw UTF-8
streams. Its codebook captures structural patterns in byte sequences —
UTF-8 byte distribution, common bigrams/trigrams, whitespace and
punctuation regularities, English orthographic patterns. Similarity in
this space is therefore character-n-gram structural similarity in a
learned polytope basis.

  Strong on: near-duplicate detection, character-edit similarity,
             typo / spelling robustness, orthographic style matching,
             plagiarism with reordering.
  Weak on:   paraphrase detection, synonym handling, semantic search,
             RAG retrieval (no semantic learning).

See OMEGA_CATALOG.md "Statute classes" section for context on the
polytope-class codebook this module projects against.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from geolip_svae.inference.engine import (
    InferenceEngine,
    CodebookMissingError,
)


# ── Module-level constants (validation surfaces) ────────────────────

PAD_STRATEGIES = ('repeat', 'space', 'zero', 'truncate')
SIGNATURE_MODES = ('omega', 'omega_orig', 'codebook')
POOL_METHODS = ('mean', 'max', 'masked_mean')
SIMILARITY_METRICS = ('cosine', 'l2')


# ── SentenceEncoder ──────────────────────────────────────────────────

class SentenceEncoder:
    """Text-side wrapper around InferenceEngine for sentence similarity.

    Args:
        engine: an ``InferenceEngine`` wrapping a sphere-solver model trained
            on byte-trigram-encoded text (e.g. ``byte_trigram_proto_64_patch_2_v1``).
        img_size: image height/width in pixels. Must be divisible by
            ``patch_size``. Should match the model's training config to
            stay in distribution.
        patch_size: patch size. Should match the model's training config.
        pad_strategy: how to pad short texts to fill the image:

            'repeat'    (default) - repeat text bytes cyclically. Best for
                                    short text; biases toward self-similarity
                                    within text content.
            'space'              - 0x20 whitespace pad. Closer to wikitext
                                    distribution than zeros.
            'zero'               - 0x00 null pad. Most OOD on padded patches —
                                    use only with ``pool='masked_mean'``.
            'truncate'           - never pad; raise if text shorter than
                                    image capacity.

    Image capacity (bytes per image):
        ``bytes_per_image = img_size * img_size * 3``
        At 64×64: 12,288 bytes (regardless of patch_size, since every
        pixel-channel is one byte). Far more than a typical sentence;
        ``pad_strategy`` decides how the slack is filled.

    The encoder is stateless aside from the engine reference and config —
    safe to reuse across calls.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        img_size: int = 64,
        patch_size: int = 2,
        pad_strategy: str = 'repeat',
    ):
        if pad_strategy not in PAD_STRATEGIES:
            raise ValueError(
                f"pad_strategy={pad_strategy!r} not in {PAD_STRATEGIES}"
            )
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size={img_size} must be divisible by patch_size={patch_size}"
            )
        # NOTE: encoder.patch_size and engine.model.patch_size are independent
        # concerns. The encoder's controls how bytes are laid out in the
        # (3, img_size, img_size) image via ByteTrigramDataset.bytes_to_image.
        # The model's controls how its forward carves the image into patches
        # for encoding (set at training time, immutable). These USUALLY match
        # for in-distribution behavior — when they don't, the byte layout is
        # off-distribution but the architecture still runs cleanly. Per
        # CLAUDE.md "no core limiters" rule for the inference layer, we
        # explicitly do NOT gate on this; the engine handles arbitrary
        # patch_size at forward time and we want diagnostic overrides to work.
        # The model will fail loudly if (img_size % model.patch_size != 0).

        self.engine = engine
        self.img_size = img_size
        self.patch_size = patch_size
        self.pad_strategy = pad_strategy

        # Capacity arithmetic
        self.gh = self.gw = img_size // patch_size
        self.cells_per_patch = patch_size * patch_size
        self.n_patches = self.gh * self.gw
        self.bytes_per_patch = self.cells_per_patch * 3
        self.bytes_per_image = self.n_patches * self.bytes_per_patch

    # ── Text → bytes → image ────────────────────────────────────────

    def _pad_bytes(self, text: str) -> tuple[np.ndarray, int]:
        """Encode text to UTF-8 and pad/truncate to ``bytes_per_image``.

        Returns:
            (chunk, n_real_bytes): ``chunk`` is shape ``(bytes_per_image,)``
                uint8; ``n_real_bytes`` is the count of original sentence
                bytes (before padding), capped at ``bytes_per_image``.
        """
        text_bytes = text.encode('utf-8')
        n_real = len(text_bytes)
        target = self.bytes_per_image

        if n_real >= target:
            return np.frombuffer(text_bytes[:target], dtype=np.uint8), target

        if self.pad_strategy == 'truncate':
            raise ValueError(
                f"Text is {n_real} bytes but image needs {target} bytes "
                f"(pad_strategy='truncate' refuses to pad)"
            )

        chunk = self._fill_padding(text_bytes, target)
        return chunk, n_real

    def _fill_padding(self, text_bytes: bytes, target: int) -> np.ndarray:
        """Pad text_bytes up to target according to pad_strategy."""
        n_real = len(text_bytes)
        if self.pad_strategy == 'zero':
            buf = bytearray(target)              # zero-init
            buf[:n_real] = text_bytes
            return np.frombuffer(bytes(buf), dtype=np.uint8)
        if self.pad_strategy == 'space':
            buf = bytearray(b' ' * target)
            buf[:n_real] = text_bytes
            return np.frombuffer(bytes(buf), dtype=np.uint8)
        if self.pad_strategy == 'repeat':
            if n_real == 0:
                # Can't repeat empty; degrade gracefully to spaces.
                return np.frombuffer(b' ' * target, dtype=np.uint8)
            n_repeats = (target + n_real - 1) // n_real
            full = (text_bytes * n_repeats)[:target]
            return np.frombuffer(full, dtype=np.uint8)
        # Should be unreachable due to __init__ validation
        raise ValueError(f"Unknown pad_strategy: {self.pad_strategy}")

    def encode_text(self, text: str) -> torch.Tensor:
        """text → ``(3, img_size, img_size)`` tensor in [-1, 1].

        Compatible with ``engine.model.forward(...)``. Single-text version.
        """
        # Lazy import: ByteTrigramDataset's module pulls torchvision and
        # opens the door to datasets/huggingface_hub. Defer until needed.
        from geolip_svae.train import ByteTrigramDataset

        chunk, _ = self._pad_bytes(text)
        img_np = ByteTrigramDataset.bytes_to_image(
            chunk, self.img_size, self.patch_size,
        )
        return torch.from_numpy(img_np)

    def encode_text_batch(self, texts: Sequence[str]) -> torch.Tensor:
        """Stack texts into ``(B, 3, img_size, img_size)`` batch."""
        if len(texts) == 0:
            return torch.empty(0, 3, self.img_size, self.img_size)
        return torch.stack([self.encode_text(t) for t in texts])

    def patch_real_mask(self, text: str) -> torch.Tensor:
        """Boolean mask of which patches contain real (un-padded) text bytes.

        For ``pad_strategy='repeat'`` all patches contain real content
        (just repeated), so the mask is all True. For 'zero' and 'space',
        only the prefix patches whose byte range is within the text length
        are marked True. Use with ``pool='masked_mean'`` to ignore padded
        patches when pooling.

        Returns: ``Tensor[n_patches]`` of bool.
        """
        if self.pad_strategy == 'repeat':
            return torch.ones(self.n_patches, dtype=torch.bool)
        n_real = min(len(text.encode('utf-8')), self.bytes_per_image)
        # Patch i covers bytes [i*bpp, (i+1)*bpp). Real if start < n_real.
        starts = torch.arange(self.n_patches) * self.bytes_per_patch
        return starts < n_real

    # ── Signature computation ───────────────────────────────────────

    @torch.no_grad()
    def signature(
        self,
        text: Union[str, Sequence[str]],
        mode: str = 'omega',
        pool: str = 'mean',
    ) -> torch.Tensor:
        """Compute sentence-level fingerprint.

        Args:
            text: a single string or sequence of strings.
            mode: 'omega' | 'omega_orig' | 'codebook'.

                'omega':       S_coordinated pooled over patches → [D].
                'omega_orig':  S_orig (pre-cross-attn) pooled → [D].
                'codebook':    axis activations (sum_abs over V per patch)
                               pooled over patches → [n_axes].
                               Requires ``engine.attach_codebook(cb)`` first.

            pool: 'mean' | 'max' | 'masked_mean'.

                'masked_mean' uses ``patch_real_mask`` to ignore padded
                patches. Only meaningful for ``pad_strategy != 'repeat'``.

        Returns:
            ``Tensor`` of shape ``[feat_dim]`` for single text input,
            ``[B, feat_dim]`` for sequence input. Always on CPU.
        """
        if mode not in SIGNATURE_MODES:
            raise ValueError(f"mode={mode!r} not in {SIGNATURE_MODES}")
        if pool not in POOL_METHODS:
            raise ValueError(f"pool={pool!r} not in {POOL_METHODS}")

        single = isinstance(text, str)
        texts = [text] if single else list(text)

        # Stage onto the engine's device
        device = next(self.engine.model.parameters()).device
        images = self.encode_text_batch(texts).to(device)

        # Per-patch features [B, n_patches, feat_dim]
        if mode == 'codebook':
            # engine.encode_axes returns activations [B, n_patches, V, n_axes]
            # Aggregate V → 1 via sum of |projections|.
            out = self.engine.encode_axes(images)
            acts = out['activations']                     # [B, P, V, n_axes]
            feat = acts.abs().sum(dim=2)                  # [B, P, n_axes]
        else:
            enc = self.engine.encode(images)
            if mode == 'omega':
                feat = enc['S']                           # [B, P, D]
            else:  # 'omega_orig'
                feat = enc['S_orig']                      # [B, P, D]

        feat = feat.to(device)

        # Pool patches → sentence-level
        if pool == 'mean':
            sig = feat.mean(dim=1)
        elif pool == 'max':
            sig = feat.amax(dim=1)
        else:  # 'masked_mean'
            masks = torch.stack([
                self.patch_real_mask(t) for t in texts
            ]).to(device)                                  # [B, P]
            mask_f = masks.float().unsqueeze(-1)           # [B, P, 1]
            denom = mask_f.sum(dim=1).clamp_min(1.0)       # [B, 1]
            sig = (feat * mask_f).sum(dim=1) / denom

        sig = sig.cpu()
        return sig.squeeze(0) if single else sig

    # ── Similarity ──────────────────────────────────────────────────

    @torch.no_grad()
    def similarity(
        self,
        text_a: str,
        text_b: str,
        mode: str = 'omega',
        metric: str = 'cosine',
        pool: str = 'mean',
    ) -> float:
        """Pairwise similarity between two strings.

        Args:
            metric: 'cosine' (default, range [-1, 1]) or 'l2' (negative
                L2 distance, larger = more similar; range (-inf, 0]).

        Returns:
            Scalar Python float.
        """
        sigs = self.signature([text_a, text_b], mode=mode, pool=pool)
        return _pair_metric(sigs[0], sigs[1], metric)

    @torch.no_grad()
    def similarity_matrix(
        self,
        texts: Sequence[str],
        mode: str = 'omega',
        metric: str = 'cosine',
        pool: str = 'mean',
    ) -> torch.Tensor:
        """Full pairwise similarity matrix.

        Returns: ``[N, N]`` tensor where ``M[i, j] = similarity(texts[i], texts[j])``.
        Diagonal is the identity (1.0 for cosine, 0.0 for negative L2).
        """
        sigs = self.signature(list(texts), mode=mode, pool=pool)  # [N, D]
        return _matrix_metric(sigs, metric)


# ── Metric helpers (module-level for testability) ───────────────────

def _pair_metric(a: torch.Tensor, b: torch.Tensor, metric: str) -> float:
    """Scalar pairwise metric between two 1-D tensors."""
    if metric == 'cosine':
        a_n = a / a.norm().clamp_min(1e-12)
        b_n = b / b.norm().clamp_min(1e-12)
        return float((a_n * b_n).sum())
    if metric == 'l2':
        return float(-(a - b).norm())
    raise ValueError(
        f"Unknown metric: {metric!r}. Supported: {SIMILARITY_METRICS}"
    )


def _matrix_metric(sigs: torch.Tensor, metric: str) -> torch.Tensor:
    """[N, N] pairwise metric matrix from a [N, feat_dim] stack of signatures."""
    if metric == 'cosine':
        norms = sigs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit = sigs / norms
        return unit @ unit.T
    if metric == 'l2':
        # Pairwise L2 via expansion: -‖x_i - x_j‖
        diffs = sigs.unsqueeze(0) - sigs.unsqueeze(1)         # [N, N, D]
        return -diffs.norm(dim=-1)
    raise ValueError(
        f"Unknown metric: {metric!r}. Supported: {SIMILARITY_METRICS}"
    )


__all__ = [
    'SentenceEncoder',
    'PAD_STRATEGIES',
    'SIGNATURE_MODES',
    'POOL_METHODS',
    'SIMILARITY_METRICS',
]
