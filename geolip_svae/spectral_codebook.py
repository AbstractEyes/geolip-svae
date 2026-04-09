"""
Spectral Codebook — Noise-Native Tokenizer for Alexandria
============================================================
Every token is a noise pattern. Every word is a waveform.

STRUCTURE:
  Special tokens → constant patches (outside noise spectrum, erank ≈ 1.0)
  Whitespace     → near-zero amplitude noise (spectrally quiet)
  1-grams        → single character → unique (noise_type, param) tuple
  2-grams        → common bigrams  → composite noise signatures
  3-grams        → common trigrams → composite noise signatures
  4-grams        → common 4-grams  → composite noise signatures
  5-grams        → common 5-grams  → composite noise signatures

PATCH FORMAT:
  Each token produces a (3, 16, 16) = 768-value patch.
  The SVD of the patch yields 16 omega tokens — the spectral fingerprint.
  After Johanna encode → decode, the fingerprint must be recoverable.

[PAD] = constant 0.0 (flat, rank 1, erank ≈ 1.0, OUTSIDE noise spectrum)
[BOS] = constant -2.0
[EOS] = constant +2.0
[UNK] = constant alternating ±0.5 rows

CODEBOOK FORMAT (JSON):
  {
    "version": "1.0",
    "patch_size": 16,
    "vocab_size": N,
    "tokens": {
      "[PAD]": {"id": 0, "type": "constant", "value": 0.0, "tier": "special"},
      "a":     {"id": 5, "type": "gaussian", "sigma": 0.15, "tier": "1gram"},
      "th":    {"id": 97, "type": "pink", "slope": 0.3, "scale": 0.8, "tier": "2gram"},
      ...
    }
  }
"""

import json
import math
import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# NOISE GENERATORS (matched to Johanna's 16 types)
# ═══════════════════════════════════════════════════════════════

def _pink(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(S / torch.sqrt(fx**2 + fy**2).clamp(min=1e-8), s=(h, ww))


def _brown(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(S / (fx**2 + fy**2).clamp(min=1e-8), s=(h, ww))


def generate_patch(token_def, ps=16, seed=None):
    """Generate a (3, ps, ps) noise patch from a token definition.

    Args:
        token_def: dict with 'type' and type-specific params
        ps: patch size (16)
        seed: optional seed for reproducibility

    Returns:
        (3, ps, ps) tensor
    """
    if seed is not None:
        torch.manual_seed(seed)

    t = token_def['type']
    s = ps

    if t == 'constant':
        return torch.full((3, s, s), token_def['value'])

    elif t == 'constant_rows':
        img = torch.zeros(3, s, s)
        for row in range(s):
            img[:, row, :] = token_def['value'] * (1 if row % 2 == 0 else -1)
        return img

    elif t == 'gaussian':
        sigma = token_def['sigma']
        return torch.randn(3, s, s) * sigma

    elif t == 'uniform':
        scale = token_def['scale']
        return (torch.rand(3, s, s) * 2 - 1) * scale

    elif t == 'pink':
        img = _pink((3, s, s))
        return img / (img.std() + 1e-8) * token_def.get('scale', 1.0)

    elif t == 'brown':
        img = _brown((3, s, s))
        return img / (img.std() + 1e-8) * token_def.get('scale', 1.0)

    elif t == 'block':
        bsize = token_def['block_size']
        small = torch.randn(3, s // bsize + 1, s // bsize + 1)
        img = F.interpolate(small.unsqueeze(0), size=s, mode='nearest').squeeze(0)
        return img * token_def.get('scale', 1.0)

    elif t == 'gradient':
        angle = token_def['angle']
        gy = torch.linspace(-2, 2, s).unsqueeze(1).expand(s, s)
        gx = torch.linspace(-2, 2, s).unsqueeze(0).expand(s, s)
        grad = math.cos(angle) * gx + math.sin(angle) * gy
        return grad.unsqueeze(0).expand(3, -1, -1) * token_def.get('scale', 1.0)

    elif t == 'checkerboard':
        cs = token_def['cell_size']
        cy = torch.arange(s) // cs
        cx = torch.arange(s) // cs
        checker = ((cy.unsqueeze(1) + cx.unsqueeze(0)) % 2).float() * 2 - 1
        return checker.unsqueeze(0).expand(3, -1, -1) * token_def.get('scale', 1.0)

    elif t == 'salt_pepper':
        density = token_def['density']
        base = torch.where(
            torch.rand(3, s, s) > 0.5,
            torch.ones(3, s, s) * 2, -torch.ones(3, s, s) * 2
        ) * density
        return base + torch.randn(3, s, s) * 0.05

    elif t == 'sparse':
        density = token_def['density']
        return torch.randn(3, s, s) * (torch.rand(3, s, s) > (1 - density)).float() * 2

    elif t == 'cauchy':
        scale = token_def['scale']
        return torch.tan(math.pi * (torch.rand(3, s, s) - 0.5)).clamp(-3, 3) * scale

    elif t == 'exponential':
        rate = token_def['rate']
        return (torch.empty(3, s, s).exponential_(rate) - 1.0 / rate)

    elif t == 'laplace':
        scale = token_def['scale']
        u = torch.rand(3, s, s) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs()) * scale

    elif t == 'poisson':
        lam = token_def['lambda']
        return torch.poisson(torch.full((3, s, s), lam)) / lam - 1.0

    elif t == 'composite':
        # Sum of two noise types — for n-gram encoding
        p1 = generate_patch(token_def['component_a'], ps, seed)
        p2 = generate_patch(token_def['component_b'], ps,
                           seed + 1 if seed else None)
        w = token_def.get('mix', 0.5)
        return w * p1 + (1 - w) * p2

    return torch.randn(3, s, s)


# ═══════════════════════════════════════════════════════════════
# SPECIAL TOKENS
# ═══════════════════════════════════════════════════════════════

SPECIAL_TOKENS = {
    '[PAD]':  {'id': 0, 'type': 'constant', 'value': 0.0,
               'tier': 'special', 'note': 'flat patch, erank≈1.0, OUTSIDE noise spectrum'},
    '[BOS]':  {'id': 1, 'type': 'constant', 'value': -2.0,
               'tier': 'special', 'note': 'start of sequence'},
    '[EOS]':  {'id': 2, 'type': 'constant', 'value': 2.0,
               'tier': 'special', 'note': 'end of sequence'},
    '[UNK]':  {'id': 3, 'type': 'constant_rows', 'value': 0.5,
               'tier': 'special', 'note': 'alternating rows ±0.5, unknown token'},
    '[SEP]':  {'id': 4, 'type': 'constant', 'value': -1.0,
               'tier': 'special', 'note': 'separator'},
}

WHITESPACE_TOKENS = {
    ' ':   {'type': 'gaussian', 'sigma': 0.01,
            'tier': 'whitespace', 'note': 'near-silent, spectrally minimal'},
    '\n':  {'type': 'gradient', 'angle': 0.0, 'scale': 0.02,
            'tier': 'whitespace', 'note': 'faint horizontal gradient'},
    '\t':  {'type': 'checkerboard', 'cell_size': 8, 'scale': 0.02,
            'tier': 'whitespace', 'note': 'faint checkerboard'},
}


# ═══════════════════════════════════════════════════════════════
# 1-GRAM ASSIGNMENTS
# ═══════════════════════════════════════════════════════════════

def build_1gram_vocab():
    """Map single characters to unique noise signatures.

    Lowercase a-z:  gaussian, σ spaced 0.10–0.88
    Uppercase A-Z:  uniform, scale spaced 0.20–1.20
    Digits 0-9:     pink noise, scale spaced 0.15–1.05
    Punctuation:    various noise types with distinct params
    """
    vocab = {}

    # Lowercase: gaussian with increasing σ
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
        vocab[c] = {'type': 'gaussian', 'sigma': 0.10 + i * 0.03,
                     'tier': '1gram'}

    # Uppercase: uniform with increasing scale
    for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        vocab[c] = {'type': 'uniform', 'scale': 0.20 + i * 0.04,
                     'tier': '1gram'}

    # Digits: pink noise with increasing scale
    for i, c in enumerate('0123456789'):
        vocab[c] = {'type': 'pink', 'scale': 0.15 + i * 0.10,
                     'tier': '1gram'}

    # Punctuation: each gets a distinct noise type
    punct_map = {
        '.':  {'type': 'block', 'block_size': 2, 'scale': 0.3},
        ',':  {'type': 'block', 'block_size': 4, 'scale': 0.3},
        '!':  {'type': 'salt_pepper', 'density': 0.3},
        '?':  {'type': 'salt_pepper', 'density': 0.5},
        ':':  {'type': 'checkerboard', 'cell_size': 2, 'scale': 0.4},
        ';':  {'type': 'checkerboard', 'cell_size': 4, 'scale': 0.4},
        "'":  {'type': 'gradient', 'angle': math.pi / 4, 'scale': 0.3},
        '"':  {'type': 'gradient', 'angle': math.pi / 2, 'scale': 0.3},
        '-':  {'type': 'gradient', 'angle': 0, 'scale': 0.4},
        '_':  {'type': 'gradient', 'angle': math.pi, 'scale': 0.4},
        '(':  {'type': 'cauchy', 'scale': 0.2},
        ')':  {'type': 'cauchy', 'scale': 0.3},
        '[':  {'type': 'laplace', 'scale': 0.3},
        ']':  {'type': 'laplace', 'scale': 0.4},
        '{':  {'type': 'exponential', 'rate': 1.0},
        '}':  {'type': 'exponential', 'rate': 2.0},
        '/':  {'type': 'gradient', 'angle': math.pi * 0.75, 'scale': 0.5},
        '\\': {'type': 'gradient', 'angle': math.pi * 0.25, 'scale': 0.5},
        '@':  {'type': 'sparse', 'density': 0.3},
        '#':  {'type': 'sparse', 'density': 0.5},
        '$':  {'type': 'poisson', 'lambda': 3.0},
        '%':  {'type': 'poisson', 'lambda': 8.0},
        '&':  {'type': 'brown', 'scale': 0.4},
        '*':  {'type': 'brown', 'scale': 0.7},
        '+':  {'type': 'block', 'block_size': 8, 'scale': 0.5},
        '=':  {'type': 'block', 'block_size': 16, 'scale': 0.5},
        '<':  {'type': 'cauchy', 'scale': 0.4},
        '>':  {'type': 'cauchy', 'scale': 0.5},
        '~':  {'type': 'laplace', 'scale': 0.5},
        '`':  {'type': 'laplace', 'scale': 0.2},
        '^':  {'type': 'exponential', 'rate': 0.5},
        '|':  {'type': 'gradient', 'angle': math.pi / 2, 'scale': 0.6},
    }

    for c, defn in punct_map.items():
        defn['tier'] = '1gram'
        vocab[c] = defn

    return vocab


# ═══════════════════════════════════════════════════════════════
# N-GRAM VOCABULARY BUILDER
# ═══════════════════════════════════════════════════════════════

# Noise type families for n-gram composite construction
NOISE_FAMILIES = [
    ('gaussian', lambda i, n: {'type': 'gaussian', 'sigma': 0.2 + (i / n) * 1.0}),
    ('uniform',  lambda i, n: {'type': 'uniform', 'scale': 0.3 + (i / n) * 1.2}),
    ('pink',     lambda i, n: {'type': 'pink', 'scale': 0.2 + (i / n) * 0.8}),
    ('brown',    lambda i, n: {'type': 'brown', 'scale': 0.2 + (i / n) * 0.8}),
    ('block',    lambda i, n: {'type': 'block', 'block_size': 2 + (i * 14 // n), 'scale': 0.3 + (i / n) * 0.7}),
    ('gradient', lambda i, n: {'type': 'gradient', 'angle': (i / n) * 2 * math.pi, 'scale': 0.3 + (i / n) * 0.5}),
    ('checker',  lambda i, n: {'type': 'checkerboard', 'cell_size': 2 + (i * 14 // n), 'scale': 0.3 + (i / n) * 0.7}),
    ('cauchy',   lambda i, n: {'type': 'cauchy', 'scale': 0.1 + (i / n) * 0.5}),
    ('laplace',  lambda i, n: {'type': 'laplace', 'scale': 0.2 + (i / n) * 0.8}),
    ('exponential', lambda i, n: {'type': 'exponential', 'rate': 0.5 + (i / n) * 2.0}),
    ('sparse',   lambda i, n: {'type': 'sparse', 'density': 0.1 + (i / n) * 0.5}),
    ('salt',     lambda i, n: {'type': 'salt_pepper', 'density': 0.1 + (i / n) * 0.6}),
    ('poisson',  lambda i, n: {'type': 'poisson', 'lambda': 1.0 + (i / n) * 15.0}),
]


def extract_ngrams(corpus, n, top_k=5000):
    """Extract top-k most common n-grams from a text corpus."""
    counts = Counter()
    for i in range(len(corpus) - n + 1):
        ngram = corpus[i:i + n]
        if all(32 <= ord(c) < 127 for c in ngram):  # printable ASCII
            counts[ngram] += 1
    return [gram for gram, _ in counts.most_common(top_k)]


def assign_ngram_noise(ngrams, tier_name):
    """Assign composite noise signatures to n-grams.

    Each n-gram gets a composite of two noise families.
    The primary family is determined by hash, the secondary
    provides the distinguishing modulation.
    """
    vocab = {}
    n_families = len(NOISE_FAMILIES)

    for i, gram in enumerate(ngrams):
        # Primary and secondary family selection
        primary_idx = i % n_families
        secondary_idx = (i // n_families + 7) % n_families  # offset to avoid same
        _, primary_fn = NOISE_FAMILIES[primary_idx]
        _, secondary_fn = NOISE_FAMILIES[secondary_idx]

        # Position within family determines parameters
        n_in_family = sum(1 for j in range(len(ngrams))
                         if j % n_families == primary_idx)
        pos = i // n_families

        vocab[gram] = {
            'type': 'composite',
            'component_a': primary_fn(pos, max(n_in_family, 1)),
            'component_b': secondary_fn(pos, max(n_in_family, 1)),
            'mix': 0.6,  # primary dominates
            'tier': tier_name,
        }

    return vocab


# ═══════════════════════════════════════════════════════════════
# FULL CODEBOOK BUILDER
# ═══════════════════════════════════════════════════════════════

def build_codebook(corpus_text=None, ngram_counts=None, save_path=None):
    """Build the complete spectral codebook.

    Args:
        corpus_text: text corpus for n-gram extraction (or use defaults)
        ngram_counts: dict of {n: top_k} for each n-gram tier
            default: {2: 2000, 3: 5000, 4: 3000, 5: 2000}
        save_path: path to save JSON codebook

    Returns:
        codebook: dict with full vocabulary
    """
    if ngram_counts is None:
        ngram_counts = {2: 2000, 3: 5000, 4: 3000, 5: 2000}

    # ── Special tokens ──
    all_tokens = dict(SPECIAL_TOKENS)

    # ── Whitespace ──
    next_id = len(all_tokens)
    for token, defn in WHITESPACE_TOKENS.items():
        defn['id'] = next_id
        all_tokens[token] = defn
        next_id += 1

    # ── 1-grams ──
    onegrams = build_1gram_vocab()
    for token, defn in onegrams.items():
        defn['id'] = next_id
        all_tokens[token] = defn
        next_id += 1

    # ── N-grams (from corpus or defaults) ──
    if corpus_text is None:
        # Use a representative English sample for default extraction
        corpus_text = _default_corpus()

    for n, top_k in sorted(ngram_counts.items()):
        tier = f'{n}gram'
        print(f"  Extracting top {top_k} {tier}s...")
        ngrams = extract_ngrams(corpus_text, n, top_k)
        # Skip ngrams that are already in vocab (e.g., single chars)
        ngrams = [g for g in ngrams if g not in all_tokens][:top_k]
        ngram_vocab = assign_ngram_noise(ngrams, tier)
        for token, defn in ngram_vocab.items():
            defn['id'] = next_id
            all_tokens[token] = defn
            next_id += 1
        print(f"    {len(ngrams)} {tier}s assigned")

    # ── Build codebook ──
    codebook = {
        'version': '1.0',
        'patch_size': 16,
        'vocab_size': len(all_tokens),
        'tiers': {
            'special': sum(1 for v in all_tokens.values() if v.get('tier') == 'special'),
            'whitespace': sum(1 for v in all_tokens.values() if v.get('tier') == 'whitespace'),
            '1gram': sum(1 for v in all_tokens.values() if v.get('tier') == '1gram'),
        },
        'tokens': all_tokens,
    }
    for n in ngram_counts:
        tier = f'{n}gram'
        codebook['tiers'][tier] = sum(1 for v in all_tokens.values() if v.get('tier') == tier)

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(codebook, f, indent=2, default=str)
        print(f"\n  Codebook saved: {save_path}")
        print(f"  Vocab size: {codebook['vocab_size']}")
        for tier, count in codebook['tiers'].items():
            print(f"    {tier}: {count}")

    return codebook


# ═══════════════════════════════════════════════════════════════
# ENCODER / DECODER
# ═══════════════════════════════════════════════════════════════

class SpectralTokenizer:
    """Encode text → noise patches, decode noise patches → text.

    Encoding: greedy longest-match tokenization, then generate patches.
    Decoding: compare patch omega tokens against codebook fingerprints.
    """

    def __init__(self, codebook):
        self.codebook = codebook
        self.tokens = codebook['tokens']
        self.ps = codebook['patch_size']

        # Build lookup structures
        self.id_to_token = {v['id']: k for k, v in self.tokens.items()}
        self.token_to_id = {k: v['id'] for k, v in self.tokens.items()}

        # Sort tokens by length (longest first) for greedy matching
        self.sorted_tokens = sorted(
            [k for k in self.tokens if self.tokens[k].get('tier') != 'special'],
            key=len, reverse=True)

        # Pre-compute spectral fingerprints for decoding
        self._fingerprints = None

    def encode_text(self, text, max_patches=64):
        """Tokenize text into token IDs using greedy longest-match.

        Args:
            text: input string
            max_patches: maximum patches (64 for 128×128)

        Returns:
            token_ids: list of int
            token_strings: list of matched strings
        """
        ids, strings = [], []
        i = 0
        while i < len(text) and len(ids) < max_patches:
            matched = False
            for token in self.sorted_tokens:
                if text[i:i + len(token)] == token:
                    ids.append(self.tokens[token]['id'])
                    strings.append(token)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                # Unknown character
                ids.append(self.tokens['[UNK]']['id'])
                strings.append(text[i])
                i += 1

        # Pad to max_patches
        while len(ids) < max_patches:
            ids.append(self.tokens['[PAD]']['id'])
            strings.append('[PAD]')

        return ids[:max_patches], strings[:max_patches]

    def ids_to_patches(self, token_ids, seed=42):
        """Convert token IDs to noise patches.

        Args:
            token_ids: list of int

        Returns:
            patches: (N, 3, ps, ps) tensor
        """
        patches = []
        for i, tid in enumerate(token_ids):
            token_str = self.id_to_token[tid]
            token_def = self.tokens[token_str]
            patch = generate_patch(token_def, self.ps, seed=seed + i)
            patches.append(patch)
        return torch.stack(patches)

    def text_to_image(self, text, max_patches=64, seed=42):
        """Full pipeline: text → token IDs → noise patches → (3, H, W) image.

        For 64 patches: 8×8 grid → (3, 128, 128)
        """
        ids, strings = self.encode_text(text, max_patches)
        patches = self.ids_to_patches(ids, seed)

        # Stitch into image
        n = len(ids)
        gh = gw = int(math.sqrt(n))
        assert gh * gw == n, f"Need square patch count, got {n}"

        patches_flat = patches.reshape(1, n, -1)
        from geolip_svae.model import stitch_patches
        image = stitch_patches(patches_flat, gh, gw, self.ps)
        return image.squeeze(0), ids, strings

    def compute_fingerprints(self, seed=42):
        """Pre-compute omega token fingerprints for all codebook entries."""
        from geolip_svae.model import gram_eigh_svd
        print("  Computing spectral fingerprints...")
        fps = {}
        for token_str, token_def in self.tokens.items():
            patch = generate_patch(token_def, self.ps, seed=seed)
            # Compute SVD → singular values = fingerprint
            M = patch.reshape(1, -1).unsqueeze(0)  # (1, 1, 768)
            # For proper fingerprint, use the SVD path
            mat = patch.reshape(1, 3 * self.ps, self.ps)  # (1, 48, 16)
            mat = F.normalize(mat, dim=-1)
            _, S, _ = gram_eigh_svd(mat)
            fps[token_def['id']] = S.squeeze(0)  # (16,)
        self._fingerprints = fps
        print(f"  {len(fps)} fingerprints computed")

    def decode_patches(self, omega_tokens):
        """Match omega tokens against codebook fingerprints.

        Args:
            omega_tokens: (N, 16) singular values per patch

        Returns:
            decoded_ids: list of int
            decoded_strings: list of str
        """
        if self._fingerprints is None:
            self.compute_fingerprints()

        fp_ids = list(self._fingerprints.keys())
        fp_stack = torch.stack([self._fingerprints[i] for i in fp_ids])  # (V, 16)

        ids, strings = [], []
        for i in range(omega_tokens.shape[0]):
            s = omega_tokens[i]  # (16,)
            # Cosine similarity against all fingerprints
            sim = F.cosine_similarity(s.unsqueeze(0), fp_stack, dim=1)
            best = fp_ids[sim.argmax().item()]
            ids.append(best)
            strings.append(self.id_to_token[best])

        return ids, strings


# ═══════════════════════════════════════════════════════════════
# DEFAULT CORPUS
# ═══════════════════════════════════════════════════════════════

def _default_corpus():
    """Representative English text for n-gram extraction."""
    return """
The quick brown fox jumps over the lazy dog. This sentence contains every letter
of the English alphabet. In the beginning was the Word, and the Word was with
God, and the Word was God. To be or not to be, that is the question. Whether
it is nobler in the mind to suffer the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles. The only thing we have to fear is
fear itself. I think therefore I am. All that glitters is not gold. A journey
of a thousand miles begins with a single step. Knowledge is power. Time flies
like an arrow, fruit flies like a banana. The unexamined life is not worth
living. In the middle of difficulty lies opportunity. Imagination is more
important than knowledge. The best way to predict the future is to create it.
Not all those who wander are lost. The pen is mightier than the sword. Actions
speak louder than words. Where there is a will there is a way. Practice makes
perfect. Fortune favors the bold. When in Rome do as the Romans do. The early
bird catches the worm. A picture is worth a thousand words. Better late than
never. Two wrongs do not make a right. Birds of a feather flock together.
The machine learning model was trained on a large dataset of images and text.
Neural networks consist of layers of interconnected nodes that process information.
The transformer architecture uses self-attention mechanisms to capture long-range
dependencies in sequential data. Convolutional neural networks are particularly
effective for image recognition tasks. The spectral decomposition of a matrix
reveals its fundamental structure through eigenvalues and eigenvectors. Singular
value decomposition provides a way to factorize any matrix into orthogonal
components. The geometric properties of high-dimensional spaces often defy our
three-dimensional intuition. Optimization algorithms search for the minimum of
a loss function by iteratively adjusting model parameters. Regularization
techniques help prevent overfitting by adding constraints to the learning process.
""" * 20  # repeat for statistical stability


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPECTRAL CODEBOOK GENERATOR")
    print("=" * 60)

    codebook = build_codebook(
        ngram_counts={2: 2000, 3: 5000, 4: 3000, 5: 2000},
        save_path='spectral_codebook_v1.json',
    )

    # Demo encoding
    tokenizer = SpectralTokenizer(codebook)
    text = "Hello, world! The quick brown fox."
    ids, strings = tokenizer.encode_text(text, max_patches=64)
    print(f"\n  Input:  '{text}'")
    print(f"  Tokens: {strings[:len(text.replace(' ', '')) + 10]}")
    print(f"  IDs:    {ids[:20]}...")

    # Demo patch generation
    image, ids, strings = tokenizer.text_to_image(text)
    print(f"\n  Image shape: {image.shape}")
    print(f"  Non-PAD tokens: {sum(1 for s in strings if s != '[PAD]')}")