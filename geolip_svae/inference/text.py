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

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from geolip_svae.inference.engine import (
    InferenceEngine,
    CodebookMissingError,
)


# ── Module-level constants (validation surfaces) ────────────────────

PAD_STRATEGIES = ('repeat', 'space', 'zero', 'truncate')

# Per-patch signature modes. The first two are the architecturally honest
# per-row representations and should be preferred for similarity work.
# The last three V-aggregate per-patch features and CLT-collapse to
# near-constant vectors that produce ~0.998 cosine across any natural
# inputs; they are kept for diagnostic comparison only.
#
#   'M_flat'         — per-patch flat M tensor [V*D].   PER-ROW preserved.
#                      Direct sphere-norm encoder rows. The model recons
#                      bytes through this, so it's the most byte-faithful
#                      per-patch representation available. Recommended
#                      default for sentence similarity.
#   'codebook_codes' — per-row argmax over codebook axes, one-hot encoded
#                      and flattened → [V*n_axes]. PER-ROW preserved.
#                      Each patch becomes a one-hot sequence over the
#                      learned polytope axes. Cosine on this measures
#                      Hamming-style code overlap (fraction of rows that
#                      land on the same axis). Requires attached codebook.
#   'omega'          — S (cross-attn-coordinated singular values), [D].
#                      V-aggregated via column norm of M_hat. CLT-collapse
#                      risk for similarity (sum over V=32 unit vectors
#                      → near-constant vector). Diagnostic only.
#   'omega_orig'     — S_orig (pre-cross-attn singular values), [D].
#                      Same V-aggregation problem. Diagnostic only.
#   'codebook_sum'   — sum_abs of codebook activations over V → [n_axes].
#                      V-aggregated; same CLT-collapse problem. Diagnostic
#                      only. Was named 'codebook' in earlier API; renamed
#                      to make the V-summing explicit.
SIGNATURE_MODES = (
    'M_flat',
    'codebook_codes',
    'omega',
    'omega_orig',
    'codebook_sum',
)
PER_ROW_MODES = ('M_flat', 'codebook_codes')         # preferred for similarity
V_AGGREGATED_MODES = ('omega', 'omega_orig', 'codebook_sum')  # diagnostic only
# Per-patch comparison aggregations. NO pre-cosine pooling — the model
# was never trained with a pool operation; pooling per-patch features
# into a sentence vector throws away the per-patch granularity that is
# the architecture's actual output unit. Aggregation happens AFTER
# per-patch cosine, on scalars.
#   'patch_mean'  — pairwise [K_a, K_b] cosine matrix → mean of all entries
#                   (order-agnostic; "average patch-pair similarity")
#   'best_match'  — symmetric mean of (max-over-B per A patch,
#                   max-over-A per B patch). Hausdorff-like; tolerates
#                   length mismatch and partial overlap.
#   'aligned'     — per-position cosine of real patches; requires
#                   K_a == K_b. Order-aware.
AGG_METHODS = ('patch_mean', 'best_match', 'aligned')
SIMILARITY_METRICS = ('cosine',)  # only metric currently supported per-patch


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

            'repeat'    (default) - repeat text bytes cyclically. Every patch
                                    sees real (cycled) content; the patch-real
                                    mask is all-True so all patches participate
                                    in similarity.
            'space'              - 0x20 whitespace pad. Closer to wikitext
                                    distribution than zeros. Real patches are
                                    only those whose pixel coverage includes
                                    at least one un-padded sentence byte.
            'zero'               - 0x00 null pad. Most OOD on padded patches.
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
        pad_strategy: str = 'space',
    ):
        if pad_strategy not in PAD_STRATEGIES:
            raise ValueError(
                f"pad_strategy={pad_strategy!r} not in {PAD_STRATEGIES}"
            )
        # 'repeat' makes patch_real_mask return all-True (every patch sees
        # cycled content, so by the byte-overlap criterion every patch is
        # "real"). That defeats the masking that per-patch similarity
        # depends on, since similarity then averages over ALL 1024 patches'
        # cycled content — which dominates any sentence-distinctive signal.
        # Default is 'space' so the mask actually filters padding patches.
        # Pass pad_strategy='repeat' explicitly only for diagnostics where
        # the all-patches-real behavior is wanted.
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

    @torch.no_grad()
    def decode_image(self, image: torch.Tensor) -> bytes:
        """``(3, H, W)`` or ``(B, 3, H, W)`` image → raw byte stream(s).

        Inverse of ``encode_text``'s byte→image step. Uses
        ``ByteTrigramDataset.image_to_bytes`` with this encoder's
        ``patch_size``. For single-image input, returns ``bytes`` of length
        ``bytes_per_image``. For batched input, returns a list of bytes
        objects, one per image.
        """
        from geolip_svae.train import ByteTrigramDataset

        single = image.ndim == 3
        if single:
            image = image.unsqueeze(0)
        # image_to_bytes: [B, 3, H, W] → [B, n_cells, 3] uint8 in cell-major
        # order (matches bytes_to_image's input order; cell c ↔ trigram c).
        bytes_t = ByteTrigramDataset.image_to_bytes(image, self.patch_size)
        # Flatten to [B, bytes_per_image] in the original byte stream order
        flat = bytes_t.reshape(bytes_t.shape[0], -1).cpu().numpy()
        out = [bytes(row.astype(np.uint8)) for row in flat]
        return out[0] if single else out

    @torch.no_grad()
    def reconstruct_text(self, text: str) -> str:
        """Full round-trip: text → image → model recon → image → text.

        Decodes the recon-image's REAL byte slice (the prefix that
        corresponds to the original text bytes, excluding padded patches)
        as UTF-8 with ``errors='replace'``. Use ``roundtrip_metrics`` for
        per-byte accuracy figures.
        """
        return self.roundtrip_metrics(text)['recon_text_real']

    @torch.no_grad()
    def roundtrip_metrics(self, text: str) -> Dict[str, Any]:
        """Verify text → image → model recon → image → text fidelity.

        This is the Step 0 sanity check that all per-patch similarity
        work depends on. If the model can't faithfully reconstruct the
        bytes underlying these specific sentences, then per-patch
        feature comparison over those sentences is comparing unreliable
        encodings.

        Returns a dict with:
            n_real_bytes:      bytes contributed by the original sentence
            recon_mse:         per-image MSE from engine.reconstruct
            full_byte_acc:     fraction of all 12,288 bytes recovered exactly
            full_byte_l1:      mean |orig - recon| in byte units (full image)
            real_byte_acc:     fraction of the real-byte prefix recovered
            real_byte_l1:      mean |orig - recon| over the real-byte prefix
            orig_text:         the input text
            recon_text_real:   UTF-8 decode of the real-byte prefix from
                               the recon image (errors='replace')
        """
        from geolip_svae.train import ByteTrigramDataset

        # Encode (same path encode_text uses)
        chunk_np, n_real = self._pad_bytes(text)
        img_np = ByteTrigramDataset.bytes_to_image(
            chunk_np, self.img_size, self.patch_size,
        )
        img = torch.from_numpy(img_np)

        # Forward through engine.reconstruct (resolution-aware, mode='direct'
        # to keep the natural patch grid; same reasoning as signature()).
        device = next(self.engine.model.parameters()).device
        img_b = img.unsqueeze(0).to(device)
        out = self.engine.reconstruct(img_b, mode='direct')
        recon_img = out['recon'].cpu()
        recon_mse = float(out['mse_per_image'][0]) if 'mse_per_image' in out \
            else float('nan')

        # Decode recon image back to bytes (same byte-stream order as input)
        recon_bytes_t = ByteTrigramDataset.image_to_bytes(
            recon_img, self.patch_size,
        )
        recon_bytes = recon_bytes_t[0].reshape(-1).cpu().numpy().astype(np.uint8)
        orig_bytes = chunk_np.astype(np.uint8)

        # Full-image byte accuracy
        full_eq = (recon_bytes == orig_bytes)
        full_byte_acc = float(full_eq.mean())
        full_byte_l1 = float(np.abs(
            recon_bytes.astype(np.int32) - orig_bytes.astype(np.int32)
        ).mean())

        # Real-byte-prefix accuracy (the part that actually encodes the sentence)
        if n_real > 0:
            real_eq = (recon_bytes[:n_real] == orig_bytes[:n_real])
            real_byte_acc = float(real_eq.mean())
            real_byte_l1 = float(np.abs(
                recon_bytes[:n_real].astype(np.int32)
                - orig_bytes[:n_real].astype(np.int32)
            ).mean())
            try:
                recon_text_real = bytes(recon_bytes[:n_real]).decode(
                    'utf-8', errors='replace',
                )
            except Exception:
                recon_text_real = '<decode_error>'
        else:
            real_byte_acc = float('nan')
            real_byte_l1 = float('nan')
            recon_text_real = ''

        return {
            'n_real_bytes': int(n_real),
            'recon_mse': recon_mse,
            'full_byte_acc': full_byte_acc,
            'full_byte_l1': full_byte_l1,
            'real_byte_acc': real_byte_acc,
            'real_byte_l1': real_byte_l1,
            'orig_text': text,
            'recon_text_real': recon_text_real,
        }

    def encode_text_batch(self, texts: Sequence[str]) -> torch.Tensor:
        """Stack texts into ``(B, 3, img_size, img_size)`` batch."""
        if len(texts) == 0:
            return torch.empty(0, 3, self.img_size, self.img_size)
        return torch.stack([self.encode_text(t) for t in texts])

    def patch_real_mask(self, text: str) -> torch.Tensor:
        """Boolean mask of which MODEL patches contain real (un-padded) text bytes.

        Returned mask is in MODEL patch space — sized
        ``(img_size // model.patch_size) ** 2`` — because that's the grid
        the model's forward (and engine.encode / engine.encode_axes in
        direct mode) emits per-patch features for. The encoder's own
        patch_size controls byte layout (via ``bytes_to_image``), not the
        feature grid. When the two patch_sizes differ (legitimate diagnostic
        config — see Critical lessons #3 / #4), each encoder logical patch
        decomposes into multiple model patches; this method handles both
        the matching and non-matching cases uniformly.

        Implementation: lay out a byte-mask (0xFF for real bytes, 0x00 for
        padded bytes) through ``bytes_to_image`` using the encoder's
        patch_size. Result is a ``(3, H, W)`` image with +1 pixels for
        real-byte content and -1 for padded. Aggregate to the model patch
        grid by taking max over (channel, ps_model, ps_model) per model
        patch — a model patch is "real" if ANY pixel within it has a real
        byte contribution.

        For ``pad_strategy='repeat'`` all model patches see (cycled) real
        content so the mask is all True. The mask drives which patches
        contribute to per-patch similarity; padded patches are skipped
        entirely (NOT averaged in — see signature/similarity docs).

        Returns: ``Tensor[n_model_patches]`` of bool.
        """
        model_ps = int(self.engine.model.patch_size)
        if self.img_size % model_ps != 0:
            raise ValueError(
                f"img_size={self.img_size} must be divisible by "
                f"model.patch_size={model_ps} for the model's forward to work"
            )
        gh_m = gw_m = self.img_size // model_ps
        n_model_patches = gh_m * gw_m

        if self.pad_strategy == 'repeat':
            return torch.ones(n_model_patches, dtype=torch.bool)

        n_real = min(len(text.encode('utf-8')), self.bytes_per_image)
        if n_real >= self.bytes_per_image:
            return torch.ones(n_model_patches, dtype=torch.bool)
        if n_real == 0:
            return torch.zeros(n_model_patches, dtype=torch.bool)

        # Build a byte mask: 0xFF for real bytes, 0x00 for padded.
        # Lay it out through the same path real bytes go through, so the
        # spatial pixel locations of "real content" exactly match.
        from geolip_svae.train import ByteTrigramDataset

        byte_mask = np.zeros(self.bytes_per_image, dtype=np.uint8)
        byte_mask[:n_real] = 0xFF
        mask_img_np = ByteTrigramDataset.bytes_to_image(
            byte_mask, self.img_size, self.patch_size,
        )
        # Pixel value: +1 if pixel comes from a real byte, -1 if from padding.
        mask_img = torch.from_numpy(mask_img_np)            # (3, H, W)

        # Aggregate to model patch grid: a model patch is "real" if any
        # pixel within it is positive.
        # (3, H, W) → (3, gh_m, ps_m, gw_m, ps_m); max over channel + ps_m × ps_m
        blocks = mask_img.reshape(3, gh_m, model_ps, gw_m, model_ps)
        patch_max = blocks.amax(dim=(0, 2, 4))              # (gh_m, gw_m)
        return (patch_max > 0).flatten()                     # (n_model_patches,)

    # ── Per-patch signature (NO pooling) ────────────────────────────

    @torch.no_grad()
    def signature(
        self,
        text: Union[str, Sequence[str]],
        mode: str = 'M_flat',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-patch features and real-patch mask. NO pooling.

        Returns per-patch features in the requested representation,
        unmodified along the patch axis, plus a mask indicating which
        patches contain real (un-padded) bytes. Aggregation, if any,
        happens AFTER cosine in similarity()/_per_patch_cosine — never
        on the feature side.

        Args:
            text: a single string or sequence of strings.
            mode: see ``SIGNATURE_MODES``. Two are recommended for
                similarity work because they preserve per-row info:

                'M_flat'         (default) — per-patch flat sphere-norm M
                    tensor. Shape [B, P, V*D]. The most byte-faithful
                    per-patch representation: the model recons bytes
                    through this exact tensor, so its 128-dim entries
                    encode the patch's bytes by construction.

                'codebook_codes' — per-row argmax over codebook axes,
                    one-hot encoded and flattened. Shape [B, P, V*n_axes].
                    Each patch becomes a one-hot sequence over the
                    polytope axes; cosine on this measures Hamming-style
                    code overlap (fraction of rows landing on the same
                    axis). Requires ``engine.attach_codebook(cb)``.

            And three diagnostic-only modes that V-aggregate and
            CLT-collapse to ~constant vectors (cosine ≈ 0.998 on any
            natural inputs — see scratchpad Critical lesson #9):

                'omega'         — coordinated S → [B, P, D]
                'omega_orig'    — pre-cross-attn S_orig → [B, P, D]
                'codebook_sum'  — sum_abs over V of codebook activations
                                  → [B, P, n_axes]

        Returns:
            features: ``[P, feat_dim]`` for single text, ``[B, P, feat_dim]``
                      for sequence input. On CPU.
            mask:     ``[P]`` or ``[B, P]`` of bool. On CPU.
        """
        if mode not in SIGNATURE_MODES:
            raise ValueError(f"mode={mode!r} not in {SIGNATURE_MODES}")

        single = isinstance(text, str)
        texts = [text] if single else list(text)

        # Stage onto the engine's device
        device = next(self.engine.model.parameters()).device
        images = self.encode_text_batch(texts).to(device)

        # mode='direct' through the engine: keeps the resolution-aware
        # path (no model() bypass) but disables tile-mode patch permuting.
        # See Critical lessons #4/#5 in the scratchpad.
        if mode == 'M_flat':
            # PER-ROW preserved: flat M tensor per patch.
            enc = self.engine.encode(images, mode='direct')
            M = enc['M'].to(device)                        # [B, P, V, D]
            B, P, V, D = M.shape
            feat = M.reshape(B, P, V * D)                  # [B, P, V*D]

        elif mode == 'codebook_codes':
            # PER-ROW preserved: per-row argmax over codebook axes,
            # one-hot encoded and flattened. Each patch becomes a
            # one-hot sequence over the polytope axes; cosine on
            # flat one-hot = Hamming overlap rate scaled by V.
            out = self.engine.encode_axes(images, mode='direct')
            acts = out['activations'].to(device)           # [B, P, V, n_axes]
            B, P, V, n_axes = acts.shape
            # Use absolute value because codebook axes are projective
            # (antipodal pairs collapsed); sign of projection is meaningless.
            codes = acts.abs().argmax(dim=-1)              # [B, P, V] int64
            one_hot = F.one_hot(codes, num_classes=n_axes).float()  # [B, P, V, n_axes]
            feat = one_hot.reshape(B, P, V * n_axes)       # [B, P, V*n_axes]

        elif mode == 'codebook_sum':
            # V-AGGREGATED — diagnostic only.
            out = self.engine.encode_axes(images, mode='direct')
            acts = out['activations'].to(device)           # [B, P, V, n_axes]
            feat = acts.abs().sum(dim=2)                   # [B, P, n_axes]

        else:
            # V-AGGREGATED column-norm modes — diagnostic only.
            enc = self.engine.encode(images, mode='direct')
            if mode == 'omega':
                feat = enc['S'].to(device)                 # [B, P, D]
            else:  # 'omega_orig'
                feat = enc['S_orig'].to(device)            # [B, P, D]

        masks = torch.stack([self.patch_real_mask(t) for t in texts])  # [B, P]

        feat = feat.cpu()
        masks = masks.cpu()
        if single:
            return feat.squeeze(0), masks.squeeze(0)
        return feat, masks

    # ── Per-patch similarity (NO pre-cosine pooling) ────────────────

    @torch.no_grad()
    def similarity(
        self,
        text_a: str,
        text_b: str,
        mode: str = 'M_flat',
        agg: str = 'patch_mean',
    ) -> float:
        """Per-patch sentence similarity. Cosine, no pre-cosine pooling.

        Pipeline:
            1. Compute per-patch features + real-patch mask for both texts.
            2. Subset to real patches only (drops padded patches entirely;
               they don't contribute to the comparison).
            3. Compute pairwise cosine matrix [K_a, K_b] between real
               patches.
            4. Aggregate to a scalar via ``agg`` (operates on cosine
               scalars, NOT on features).

        Args:
            agg: how to aggregate the per-patch cosine matrix:
                'patch_mean'  — mean over all entries of the [K_a, K_b]
                                matrix. Order-agnostic.
                'best_match'  — symmetric mean of (max-over-B per A patch,
                                max-over-A per B patch). Hausdorff-like.
                                Tolerates length mismatch.
                'aligned'     — per-position cosine of real patches; mean.
                                Requires K_a == K_b (raises otherwise).

        Returns:
            Scalar Python float in [-1, 1] (cosine range). Returns
            ``float('nan')`` when either sentence has zero real patches.
        """
        if agg not in AGG_METHODS:
            raise ValueError(f"agg={agg!r} not in {AGG_METHODS}")

        feat_a, mask_a = self.signature(text_a, mode=mode)
        feat_b, mask_b = self.signature(text_b, mode=mode)
        return _per_patch_cosine(feat_a[mask_a], feat_b[mask_b], agg)

    @torch.no_grad()
    def similarity_matrix(
        self,
        texts: Sequence[str],
        mode: str = 'M_flat',
        agg: str = 'patch_mean',
    ) -> torch.Tensor:
        """Full pairwise per-patch similarity matrix. NO pooling.

        Returns: ``[N, N]`` tensor where
        ``M[i, j] = similarity(texts[i], texts[j], mode, agg)``. Diagonal
        is 1.0 (a sentence is identical to itself under any per-patch
        cosine aggregation that respects the metric).
        """
        if agg not in AGG_METHODS:
            raise ValueError(f"agg={agg!r} not in {AGG_METHODS}")

        feats, masks = self.signature(list(texts), mode=mode)   # [N, P, D], [N, P]
        n = len(texts)
        out = torch.zeros(n, n)
        # Subset each text to its real patches up front
        real = [feats[i][masks[i]] for i in range(n)]            # list of [K_i, D]
        for i in range(n):
            for j in range(n):
                out[i, j] = _per_patch_cosine(real[i], real[j], agg)
        return out


# ── Per-patch cosine helpers (module-level for testability) ─────────

def _per_patch_cosine(
    real_a: torch.Tensor,
    real_b: torch.Tensor,
    agg: str,
) -> float:
    """Per-patch cosine aggregation.

    Args:
        real_a: ``[K_a, feat_dim]`` features of text A's real patches only.
        real_b: ``[K_b, feat_dim]`` features of text B's real patches only.
        agg: one of ``AGG_METHODS``.

    Returns:
        Scalar float, ``nan`` if either sentence has zero real patches.
    """
    if real_a.numel() == 0 or real_b.numel() == 0:
        return float('nan')

    a_n = real_a / real_a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b_n = real_b / real_b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = a_n @ b_n.T                                # [K_a, K_b]

    if agg == 'patch_mean':
        return float(sim.mean())
    if agg == 'best_match':
        a_best = sim.amax(dim=1).mean()              # mean over A's best matches in B
        b_best = sim.amax(dim=0).mean()              # mean over B's best matches in A
        return float((a_best + b_best) / 2)
    if agg == 'aligned':
        if real_a.shape[0] != real_b.shape[0]:
            raise ValueError(
                f"agg='aligned' requires same number of real patches; "
                f"got K_a={real_a.shape[0]} K_b={real_b.shape[0]}"
            )
        return float((a_n * b_n).sum(dim=-1).mean())
    raise ValueError(f"Unknown agg: {agg}. Supported: {AGG_METHODS}")


__all__ = [
    'SentenceEncoder',
    'PAD_STRATEGIES',
    'SIGNATURE_MODES',
    'PER_ROW_MODES',
    'V_AGGREGATED_MODES',
    'AGG_METHODS',
    'SIMILARITY_METRICS',
]
