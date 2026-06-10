# aleph_routed_attention.py
"""
Aleph-Routed Attention — routing attention through a learned projective codebook
=================================================================================

Two variants of attention whose routing medium is the aleph signed-projective
address (geolip-svae aleph_model.py lineage):

  HUB    : linear attention whose feature map IS the aleph address.
           score(i,j) = <addr(q_i), addr(k_j)> over 2K oriented axes [+A; -A].
           Factors through two K-wide memories (the antipodal closed-form trick:
           the 2K tensor is never materialized). O(n*K*d), PURE GEMM — no gathers.
           Denominator is a dot product of strictly positive distributions, so it
           cannot vanish or flip sign (structurally stabler than elu+1 feature maps).
           Attention-matrix rank is bounded by 2K: K is the bandwidth knob,
           tau is the hardness knob.

  BUCKET : hard address. Each token's winner oriented half-axis is its bucket;
           exact softmax attention within sorted equal-width blocks (Reformer-style
           sort-and-window), masked to same-bucket pairs. One gather-bound mode for
           the A/B against the GEMM mode. Codebook receives gradient through a
           differentiable address-agreement bias added to the scores (hard argmax
           alone is gradient-dead w.r.t. the codebook).

Shared geometric discipline (geolip-svae invariants honored):
  - q/k address rows are sphere-normalized onto S^(D_addr-1)  (geometric premise)
  - nn.init.orthogonal_ on the address projections                (load-bearing)
  - no BatchNorm, no Dropout on the geometric path, no GAP
  - codebook init: 'random' | 'fibonacci' (super-Fibonacci S^3 at D=4) | (K,D) array
    — 'custom' array supports TRANSPLANTING a trained AlephModel codebook.

Preregistered basin test (decide before running):
  Train the routing codebook from scratch on a sequence task, then run the
  geolip-svae antipodal-collapse extraction on export_codebook().
    CLEAN  (|deviation| < 0.05 on RP^(D-1)) -> cross-objective attractor evidence.
    DIRTY  -> the attractor is reconstruction-specific.
  Either answer is data.

Compile discipline (Phil's rule): forward() returns a single Tensor. All
diagnostics (perplexity, margin, bucket load, confidence) live in the separate
no-grad address_stats() method — never in the compiled hot path.

Prior-art honesty for the writeup: hub is the linear-transformer/Performer
family (kernel feature maps) crossed with Set-Transformer inducing points;
bucket rhymes with Reformer/Routing Transformer. Novel content: the signed
antipodal closed form as feature map, spherical D-space addresses, codebook
transplant from reconstruction alephs, and the attractor test.

Author: AbstractPhil + Mirel
Date: 2026-06-09
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AlephAttentionConfig:
    """Configuration for AlephRoutedAttention.

    Args:
        dim:            model dimension
        num_heads:      attention heads
        mode:           'hub' (linear, GEMM-only) | 'bucket' (hard-address cliques)
        K:              codebook axes (oriented axes = 2K). Rank/bandwidth knob.
        D_addr:         address-space dimension (rows live on S^(D_addr-1))
        tau:            address temperature. Small -> near-discrete routing,
                        large -> mean-pool collapse. aleph reference: 0.1
        codebook_init:  'random' | 'fibonacci' | (K, D_addr) tensor/array
                        (transplant a trained AlephModel codebook here)
        freeze_codebook: register codebook as a buffer (no gradient). Only safe
                        once a drift check confirms the init IS the attractor.
        causal:         autoregressive masking (both modes)
        chunk_size:     hub-causal chunk width (exact chunked linear attention)
        block_size:     bucket-mode sorted-window width W (keys window = 2W
                        via 1-block lookback)
        bucket_bias_scale_init: init of the learnable scale on the differentiable
                        address-agreement bias (the codebook's gradient path in
                        bucket mode)
        confidence_gate: multiply head outputs by aleph address confidence
                        ||(p+ - p-) @ A||  (experimental; default off)
        qkv_bias / out_bias: projection biases
        dropout:        output-projection dropout ONLY (never on the geometric path)
        eps:            numerical floor for denominators
    """
    dim: int = 512
    num_heads: int = 8
    mode: str = "hub"                      # 'hub' | 'bucket'
    K: int = 64
    D_addr: int = 4
    tau: float = 0.1
    codebook_init: object = "fibonacci"
    freeze_codebook: bool = False
    causal: bool = False
    chunk_size: int = 128
    block_size: int = 64
    bucket_bias_scale_init: float = 1.0
    confidence_gate: bool = False
    tied_address: bool = False        # share q/k address projection. EMPIRICAL (2026-06-09
                                      # CPU recall A/B): tying HURTS — sharp self-affinity
                                      # at low tau structurally biases routing to self
                                      # (same family as softmax(1/d) collapse). Keep False.
    qkv_bias: bool = False
    out_bias: bool = True
    dropout: float = 0.0
    eps: float = 1e-8

    def __post_init__(self):
        assert self.mode in ("hub", "bucket"), f"mode must be 'hub'|'bucket', got {self.mode!r}"
        assert self.dim % self.num_heads == 0, \
            f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.dim // self.num_heads
        assert self.K >= 2 and self.D_addr >= 2
        assert self.tau > 0 and self.eps > 0
        assert self.chunk_size > 0 and self.block_size > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Codebook init (ported from geolip-svae aleph_model.py — self-contained)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _super_fibonacci_s3(n: int, dtype=torch.float32) -> Tensor:
    """n near-uniform unit quaternions on S^3 via super-Fibonacci spirals
    (Alexa, CVPR 2022). Deterministic, low-discrepancy. Returns (n, 4)."""
    PHI = math.sqrt(2.0)
    PSI = 1.533751168755204288118041
    i = torch.arange(n, dtype=torch.float64) + 0.5
    s = i / n
    r = torch.sqrt(s)
    R = torch.sqrt(1.0 - s)
    alpha = 2.0 * math.pi * i / PHI
    beta = 2.0 * math.pi * i / PSI
    q = torch.stack([r * torch.sin(alpha), r * torch.cos(alpha),
                     R * torch.sin(beta), R * torch.cos(beta)], dim=-1)
    return q.to(dtype)


def _init_codebook(K: int, D: int, init, dtype=torch.float32) -> Tensor:
    """'random' Gaussian | 'fibonacci' near-uniform spread (exact at D=4,
    seeded-normalized fallback otherwise) | caller (K, D) array, row-normalized
    — the transplant path for a trained AlephModel codebook."""
    if isinstance(init, str):
        if init == "random":
            return torch.randn(K, D, dtype=dtype)
        if init == "fibonacci":
            if D == 4:
                return F.normalize(_super_fibonacci_s3(K, dtype=dtype), dim=-1)
            g = torch.Generator().manual_seed(0)
            return F.normalize(torch.randn(K, D, generator=g, dtype=dtype), dim=-1)
        raise ValueError(f"unknown codebook_init '{init}'")
    A = torch.as_tensor(init, dtype=dtype)
    if tuple(A.shape) != (K, D):
        raise ValueError(f"codebook_init array shape {tuple(A.shape)} != ({K}, {D})")
    return F.normalize(A, dim=-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main module
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AlephRoutedAttention(nn.Module):
    """Attention routed through a learned (K, D_addr) projective codebook.

    Per head, queries and keys are projected to D_addr and sphere-normalized.
    The aleph address  p(x) = softmax([u; -u]),  u = (x_hat @ A^T) / tau
    is the routing medium:

      mode='hub'    tokens communicate THROUGH the codebook — linear attention
                    with p as the feature map, antipodal-factored to K-wide ops.
      mode='bucket' tokens attend only to same-winner-half-axis peers — exact
                    softmax within sorted blocks.

    forward(x, attn_mask=None) -> Tensor (B, S, dim).   Diagnostics: address_stats().
    """

    def __init__(self, config: AlephAttentionConfig):
        super().__init__()
        self.cfg = config
        c = config
        self.dim, self.H, self.hd = c.dim, c.num_heads, c.head_dim
        self.K, self.Da, self.tau = c.K, c.D_addr, c.tau

        # ── projections ──
        # address projections: per-head D_addr rows for q and k (the routing space)
        self.q_addr = nn.Linear(c.dim, self.H * self.Da, bias=c.qkv_bias)
        nn.init.orthogonal_(self.q_addr.weight)            # load-bearing convention
        if c.tied_address:
            self.k_addr = self.q_addr                      # one routing space
        else:
            self.k_addr = nn.Linear(c.dim, self.H * self.Da, bias=c.qkv_bias)
            nn.init.orthogonal_(self.k_addr.weight)
        # value projection: full head_dim payload
        self.v_proj = nn.Linear(c.dim, c.dim, bias=c.qkv_bias)
        self.out_proj = nn.Linear(c.dim, c.dim, bias=c.out_bias)
        self.dropout = nn.Dropout(c.dropout)               # output path only

        # bucket mode additionally scores with full-width q/k (payload attention
        # inside the clique); hub routes purely through the address
        if c.mode == "bucket":
            self.q_proj = nn.Linear(c.dim, c.dim, bias=c.qkv_bias)
            self.k_proj = nn.Linear(c.dim, c.dim, bias=c.qkv_bias)
            nn.init.orthogonal_(self.q_proj.weight)
            nn.init.orthogonal_(self.k_proj.weight)
            self.bucket_bias_scale = nn.Parameter(
                torch.tensor(float(c.bucket_bias_scale_init)))
        self.scale = 1.0 / math.sqrt(self.hd)

        # ── the aleph codebook ──
        A0 = _init_codebook(c.K, c.D_addr, c.codebook_init)
        if c.freeze_codebook:
            self.register_buffer("codebook", A0)
        else:
            self.codebook = nn.Parameter(A0)

        # diversity-loss hook: stash the mean address (WITH grad) when armed
        self.emit_diversity: bool = False
        self._mean_address: Optional[Tensor] = None        # (2K,) when armed

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Address machinery (shared)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def oriented_codebook(self) -> Tensor:
        """(2K, D_addr) oriented half-axes [+A; -A], unit rows."""
        A = F.normalize(self.codebook, dim=-1)
        return torch.cat([A, -A], dim=0)

    def export_codebook(self) -> Tensor:
        """Normalized (K, D_addr) axes for the geolip-svae antipodal-collapse
        extraction — the preregistered basin test entry point."""
        return F.normalize(self.codebook.detach(), dim=-1).cpu()

    def _split_addr(self, t: Tensor, B: int, S: int) -> Tensor:
        """(B, S, H*Da) -> (B, H, S, Da), rows sphere-normalized."""
        t = t.view(B, S, self.H, self.Da).transpose(1, 2)
        return F.normalize(t, dim=-1)                      # S^(D_addr-1): the premise

    def _address(self, x_hat: Tensor) -> Tuple[Tensor, Tensor]:
        """Aleph address of unit rows x_hat (..., Da) against the codebook.

        Returns (p_plus, p_minus), each (..., K), with
            p_plus_k  = e^{ u_k} / Z,   p_minus_k = e^{-u_k} / Z,
            Z = sum_k (e^{u_k} + e^{-u_k}),    u = (x_hat @ A^T)/tau
        i.e. the exact softmax over the 2K oriented axes, antipodally factored:
        the 2K tensor is never materialized. Stable via max|u| subtraction
        (at least one exponent is exactly e^0, so Z' >= 1)."""
        A = F.normalize(self.codebook, dim=-1)             # (K, Da)
        u = (x_hat @ A.t()) * (1.0 / self.tau)             # (..., K) signed
        m = u.abs().amax(dim=-1, keepdim=True)
        ep = torch.exp(u - m)                              # ∝ e^{+u}
        en = torch.exp(-u - m)                             # ∝ e^{-u}
        Z = (ep + en).sum(dim=-1, keepdim=True)            # >= 1 by construction
        return ep / Z, en / Z

    def _confidence(self, pq_p: Tensor, pq_m: Tensor) -> Tensor:
        """Aleph address confidence ||(p+ - p-) @ A|| in (0, 1] — the norm of the
        soft codebook reconstruction (the hub analogue of ||M_hat||)."""
        A = F.normalize(self.codebook, dim=-1)
        return ((pq_p - pq_m) @ A).norm(dim=-1)            # (..., )

    def _stash_diversity(self, pk_p: Tensor, pk_m: Tensor,
                         mask: Optional[Tensor]) -> None:
        """Mean address over valid key rows -> (2K,) with grad, for diversity_loss()."""
        if not (self.emit_diversity and self.training):
            return
        if mask is not None:
            w = mask[:, None, :, None].to(pk_p.dtype)      # (B,1,S,1)
            n = w.sum().clamp_min(1.0) * self.H
            mp = (pk_p * w).sum(dim=(0, 1, 2)) / n
            mm = (pk_m * w).sum(dim=(0, 1, 2)) / n
        else:
            mp = pk_p.mean(dim=(0, 1, 2))
            mm = pk_m.mean(dim=(0, 1, 2))
        self._mean_address = torch.cat([mp, mm], dim=0)    # (2K,)

    def diversity_loss(self) -> Tensor:
        """Anti-collapse term (train_aleph div_weight semantics):
        log(2K) - H(mean address). Zero at uniform usage. Arm with
        model.emit_diversity = True; read after forward; weight ~0.01."""
        if self._mean_address is None:
            return torch.zeros((), device=self.codebook.device)
        p = self._mean_address.clamp_min(1e-12)
        H = -(p * p.log()).sum()
        return math.log(2 * self.K) - H

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HUB mode — linear attention through the codebook (pure GEMM)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _hub_full(self, pq_p, pq_m, pk_p, pk_m, v) -> Tensor:
        """Non-causal hub. p*: (B,H,S,K), v: (B,H,S,hd) -> (B,H,S,hd).

        score(i,j) = pq+(i)·pk+(j) + pq-(i)·pk-(j)  factors through two K-wide
        memories; out_i = num_i / den_i with den strictly positive."""
        Mp = torch.einsum('bhsk,bhsd->bhkd', pk_p, v)      # (B,H,K,hd)
        Mm = torch.einsum('bhsk,bhsd->bhkd', pk_m, v)
        zp = pk_p.sum(dim=2)                               # (B,H,K)
        zm = pk_m.sum(dim=2)
        num = torch.einsum('bhsk,bhkd->bhsd', pq_p, Mp) \
            + torch.einsum('bhsk,bhkd->bhsd', pq_m, Mm)
        den = torch.einsum('bhsk,bhk->bhs', pq_p, zp) \
            + torch.einsum('bhsk,bhk->bhs', pq_m, zm)
        return num / den.unsqueeze(-1).clamp_min(self.cfg.eps)

    def _hub_causal(self, pq_p, pq_m, pk_p, pk_m, v,
                    state: Optional[Tuple[Tensor, ...]] = None
                    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """Exact chunked causal hub: running K-wide state across chunks +
        lower-triangular intra-chunk correction. Loop count = S/chunk_size
        (the standard chunked linear-attention recurrence — not a per-token loop).

        `state` = (Mp, Mm, zp, zm) carried from previous segments. The state is
        constant-size — (B,H,K,hd)+(B,H,K) per sign — regardless of how much
        past it summarizes: Mp/Mm are what has been written to each oriented
        codebook axis so far. Returns (out, final_state) for streaming."""
        B, H, S, _ = v.shape
        C = min(self.cfg.chunk_size, S)
        if state is None:
            Mp = v.new_zeros(B, H, self.K, self.hd)
            Mm = v.new_zeros(B, H, self.K, self.hd)
            zp = v.new_zeros(B, H, self.K)
            zm = v.new_zeros(B, H, self.K)
        else:
            Mp, Mm, zp, zm = state
        outs = []
        tri_cache: Dict[int, Tensor] = {}
        for s0 in range(0, S, C):
            s1 = min(s0 + C, S)
            qp, qm = pq_p[:, :, s0:s1], pq_m[:, :, s0:s1]
            kp, km = pk_p[:, :, s0:s1], pk_m[:, :, s0:s1]
            vc = v[:, :, s0:s1]
            c = s1 - s0
            if c not in tri_cache:
                tri_cache[c] = torch.tril(
                    torch.ones(c, c, device=v.device, dtype=v.dtype))
            tri = tri_cache[c]
            # intra-chunk (causal) scores — strictly positive entries pre-mask
            intra = (torch.einsum('bhik,bhjk->bhij', qp, kp)
                     + torch.einsum('bhik,bhjk->bhij', qm, km)) * tri
            num = intra @ vc \
                + torch.einsum('bhsk,bhkd->bhsd', qp, Mp) \
                + torch.einsum('bhsk,bhkd->bhsd', qm, Mm)
            den = intra.sum(dim=-1) \
                + torch.einsum('bhsk,bhk->bhs', qp, zp) \
                + torch.einsum('bhsk,bhk->bhs', qm, zm)
            outs.append(num / den.unsqueeze(-1).clamp_min(self.cfg.eps))
            # state update (inclusive of this chunk, for the next one)
            Mp = Mp + torch.einsum('bhck,bhcd->bhkd', kp, vc)
            Mm = Mm + torch.einsum('bhck,bhcd->bhkd', km, vc)
            zp = zp + kp.sum(dim=2)
            zm = zm + km.sum(dim=2)
        return torch.cat(outs, dim=2), (Mp, Mm, zp, zm)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BUCKET mode — hard-address cliques (sort + windowed exact attention)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _take(t: Tensor, idx: Tensor) -> Tensor:
        """Gather along dim=2. t: (B,H,S,X) or (B,H,S); idx: (B,H,S)."""
        if t.dim() == 3:
            return torch.gather(t, 2, idx)
        return torch.gather(t, 2, idx.unsqueeze(-1).expand(-1, -1, -1, t.shape[-1]))

    @staticmethod
    def _window(t: Tensor, nb: int, W: int) -> Tensor:
        """Blocked tensor (B,H,nb,W,...) -> (B,H,nb,2W,...) keys window =
        [previous block ; this block]. Block 0's previous half is junk —
        callers must kill it via the validity window."""
        prev = torch.cat([torch.zeros_like(t[:, :, :1]), t[:, :, :-1]], dim=2)
        return torch.cat([prev, t], dim=3)

    def _bucket_attend(self, q, k, v, pq_p, pq_m, pk_p, pk_m,
                       mask: Optional[Tensor]) -> Tensor:
        """q,k,v: (B,H,S,hd); p*: (B,H,S,K); mask: (B,S) 1=valid or None.

        1. bucket = winner oriented half-axis (argmax |u|, sign-resolved)
        2. stable-sort tokens by bucket; pad S to a multiple of W
        3. exact softmax attention within [prev block ; block] windows,
           masked to same-bucket, valid, (and causal by original position)
        4. differentiable codebook path: scores += scale * address-agreement
        5. inverse-permute, un-pad."""
        cfg = self.cfg
        B, H, S, hd = q.shape
        W = min(cfg.block_size, max(8, S))
        dev = q.device

        # ── 1. hard bucket ids ── (recover signed u from the address: u = (log ep - log en)/2
        # is unnecessary — argmax of p_plus vs p_minus IS argmax |u| with sign)
        win_p, idx_p = pq_p.max(dim=-1)                    # query side unused for ids
        # bucket from the KEY/QUERY shared address rows: use each token's own address
        # (q-side and k-side addresses may differ; routing identity = q-address for
        # queries, k-address for keys — a token can listen in one clique and speak in
        # another. We bucket by the K-side address for keys and Q-side for queries,
        # then require equality — implemented by bucketing each side independently.)
        def hard_ids(pp: Tensor, pm: Tensor) -> Tensor:
            vp, ip = pp.max(dim=-1)
            vm, im = pm.max(dim=-1)
            plus_wins = vp >= vm
            return torch.where(plus_wins, ip, im + self.K)  # (B,H,S) in [0, 2K)

        bq = hard_ids(pq_p, pq_m)
        bk = hard_ids(pk_p, pk_m)
        valid = (mask if mask is not None
                 else torch.ones(B, S, device=dev, dtype=torch.bool))
        valid = valid.bool()[:, None, :].expand(B, H, S)
        JUNK = 2 * self.K + 1
        bq = torch.where(valid, bq, torch.full_like(bq, JUNK))
        bk = torch.where(valid, bk, torch.full_like(bk, JUNK))

        # ── 2. pad to multiple of W, sort by key-bucket ──
        pad = (-S) % W
        if pad:
            def padS(t, fill=0.0):
                shape = list(t.shape); shape[2] = pad
                return torch.cat([t, t.new_full(shape, fill)], dim=2)
            q, k, v = padS(q), padS(k), padS(v)
            pq_p, pq_m, pk_p, pk_m = padS(pq_p), padS(pq_m), padS(pk_p), padS(pk_m)
            bq, bk = padS(bq, JUNK), padS(bk, JUNK)
            valid = padS(valid, False)
        Sp = S + pad
        nb = Sp // W
        pos = torch.arange(Sp, device=dev).view(1, 1, Sp).expand(B, H, Sp)

        sort_idx = bk.argsort(dim=-1, stable=True)         # cluster keys by bucket
        inv_idx = sort_idx.argsort(dim=-1)
        gq, gk, gv = self._take(q, sort_idx), self._take(k, sort_idx), self._take(v, sort_idx)
        gpq_p, gpq_m = self._take(pq_p, sort_idx), self._take(pq_m, sort_idx)
        gpk_p, gpk_m = self._take(pk_p, sort_idx), self._take(pk_m, sort_idx)
        gbq, gbk = self._take(bq, sort_idx), self._take(bk, sort_idx)
        gvalid, gpos = self._take(valid.long(), sort_idx).bool(), self._take(pos, sort_idx)

        def blk(t):
            return t.view(B, H, nb, W, *t.shape[3:])
        q_b, v_b = blk(gq), blk(gv)
        k_w = self._window(blk(gk), nb, W)                 # (B,H,nb,2W,hd)
        v_w = self._window(blk(gv), nb, W)
        pkp_w = self._window(blk(gpk_p), nb, W)
        pkm_w = self._window(blk(gpk_m), nb, W)
        bq_b = blk(gbq)
        bk_w = self._window(blk(gbk).unsqueeze(-1), nb, W).squeeze(-1)
        val_w = self._window(blk(gvalid.long()).unsqueeze(-1), nb, W).squeeze(-1).bool()
        pos_b = blk(gpos)
        pos_w = self._window(blk(gpos).unsqueeze(-1), nb, W).squeeze(-1)
        val_w[:, :, 0, :W] = False                         # block 0 has no previous

        # ── 3. scores: payload q·k within the window ──
        scores = torch.einsum('bhnwd,bhnud->bhnwu', q_b, k_w) * self.scale

        # ── 4. differentiable address-agreement bias (codebook gradient path) ──
        pqp_b, pqm_b = blk(gpq_p), blk(gpq_m)
        agreement = torch.einsum('bhnwk,bhnuk->bhnwu', pqp_b, pkp_w) \
                  + torch.einsum('bhnwk,bhnuk->bhnwu', pqm_b, pkm_w)
        scores = scores + self.bucket_bias_scale * agreement

        # ── masks: same bucket, valid, causal ──
        same = bq_b.unsqueeze(-1) == bk_w.unsqueeze(-2)    # (B,H,nb,W,2W)
        keep = same & val_w.unsqueeze(-2)
        if cfg.causal:
            keep = keep & (pos_w.unsqueeze(-2) <= pos_b.unsqueeze(-1))
        scores = scores.masked_fill(~keep, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)             # all-masked rows = pads only
        out_b = torch.einsum('bhnwu,bhnud->bhnwd', attn, v_w)

        # ── 5. inverse permute, un-pad ──
        out = out_b.reshape(B, H, Sp, hd)
        out = self._take(out, inv_idx)
        return out[:, :, :S]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Forward (returns a single Tensor — compile-rule compliant)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """x: (B, S, dim); attn_mask: (B, S) with 1 = valid, 0 = padding.
        Returns (B, S, dim)."""
        B, S, _ = x.shape
        cfg = self.cfg

        qh = self._split_addr(self.q_addr(x), B, S)        # (B,H,S,Da) on the sphere
        kh = self._split_addr(self.k_addr(x), B, S)
        v = self.v_proj(x).view(B, S, self.H, self.hd).transpose(1, 2)

        pq_p, pq_m = self._address(qh)                     # (B,H,S,K) each
        pk_p, pk_m = self._address(kh)
        self._stash_diversity(pk_p, pk_m, attn_mask)

        if attn_mask is not None:
            mk = attn_mask[:, None, :, None].to(v.dtype)   # kill masked KEYS
            pk_p, pk_m, v_in = pk_p * mk, pk_m * mk, v * mk
        else:
            v_in = v

        if cfg.mode == "hub":
            if cfg.causal:
                out, _ = self._hub_causal(pq_p, pq_m, pk_p, pk_m, v_in)
            else:
                out = self._hub_full(pq_p, pq_m, pk_p, pk_m, v_in)
        else:  # bucket
            qf = self.q_proj(x).view(B, S, self.H, self.hd).transpose(1, 2)
            kf = self.k_proj(x).view(B, S, self.H, self.hd).transpose(1, 2)
            out = self._bucket_attend(qf, kf, v, pq_p, pq_m, pk_p, pk_m, attn_mask)

        if cfg.confidence_gate:
            out = out * self._confidence(pq_p, pq_m).unsqueeze(-1)

        out = out.transpose(1, 2).reshape(B, S, self.dim)
        return self.dropout(self.out_proj(out))

    def forward_stream(self, x: Tensor,
                       state: Optional[Tuple[Tensor, ...]] = None,
                       attn_mask: Optional[Tensor] = None
                       ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """Segment-recurrent forward (mode='hub', causal=True only).

        Processes a segment with the codebook memory carried in `state`
        (init None = empty past), returns (out, new_state). Context is
        unbounded at constant memory: state is (Mp, Mm, zp, zm), shape
        (B,H,K,hd)x2 + (B,H,K)x2, independent of total past length.
        TBPTT discipline: .detach() each state tensor between backward
        passes — graphs are freed per segment."""
        assert self.cfg.mode == "hub" and self.cfg.causal, \
            "forward_stream requires mode='hub', causal=True (bucket sorts globally)"
        B, S, _ = x.shape
        qh = self._split_addr(self.q_addr(x), B, S)
        kh = self._split_addr(self.k_addr(x), B, S)
        v = self.v_proj(x).view(B, S, self.H, self.hd).transpose(1, 2)
        pq_p, pq_m = self._address(qh)
        pk_p, pk_m = self._address(kh)
        self._stash_diversity(pk_p, pk_m, attn_mask)
        if attn_mask is not None:
            mk = attn_mask[:, None, :, None].to(v.dtype)
            pk_p, pk_m, v = pk_p * mk, pk_m * mk, v * mk
        out, new_state = self._hub_causal(pq_p, pq_m, pk_p, pk_m, v, state)
        if self.cfg.confidence_gate:
            out = out * self._confidence(pq_p, pq_m).unsqueeze(-1)
        out = out.transpose(1, 2).reshape(B, S, self.dim)
        return self.dropout(self.out_proj(out)), new_state

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Diagnostics (eval-only; never in the hot path)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @torch.no_grad()
    def address_stats(self, x: Tensor,
                      attn_mask: Optional[Tensor] = None,
                      max_rows: Optional[int] = None) -> Dict[str, float]:
        """Codebook-health monitors (train_aleph semantics):
          perplexity : exp(H(mean address)) — effective oriented axes in use,
                       in [1, 2K]. The collapse detector.
          margin     : mean (top1 - top2) of per-row address — decisiveness.
          confidence : mean ||(p+ - p-) @ A|| — address sharpness in (0, 1].
          bucket_cv  : coefficient of variation of hard-bucket occupancy
                       (load-balance; bucket mode's health metric).
        """
        B, S, _ = x.shape
        kh = self._split_addr(self.k_addr(x), B, S)
        pp, pm = self._address(kh)
        if attn_mask is not None:
            m = attn_mask.bool()[:, None, :].expand(B, self.H, S)
            pp = pp[m]; pm = pm[m]                          # (R, K)
        else:
            pp = pp.reshape(-1, self.K); pm = pm.reshape(-1, self.K)
        full = torch.cat([pp, pm], dim=-1)                  # (R, 2K)
        if max_rows is not None and full.shape[0] > max_rows:
            full = full[torch.randperm(full.shape[0])[:max_rows]]
            pp, pm = full[:, :self.K], full[:, self.K:]

        mean_addr = full.mean(dim=0).clamp_min(1e-12)
        H = -(mean_addr * mean_addr.log()).sum()
        perplexity = H.exp().item()

        top2 = full.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()

        A = F.normalize(self.codebook, dim=-1)
        confidence = ((pp - pm) @ A).norm(dim=-1).mean().item()

        ids = full.argmax(dim=-1)
        occ = torch.bincount(ids, minlength=2 * self.K).float()
        bucket_cv = (occ.std(unbiased=False) / occ.mean().clamp_min(1e-12)).item()

        return {"perplexity": perplexity, "margin": margin,
                "confidence": confidence, "bucket_cv": bucket_cv,
                "max_perplexity": float(2 * self.K)}

    def extra_repr(self) -> str:
        c = self.cfg
        return (f"dim={c.dim}, heads={c.num_heads}, mode={c.mode}, "
                f"K={c.K} (2K={2*c.K} oriented), D_addr={c.D_addr}, "
                f"tau={c.tau}, causal={c.causal}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reference baseline (for the harness A/B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StandardAttention(nn.Module):
    """Plain softmax attention, same I/O contract, for the A/B."""

    def __init__(self, dim: int, num_heads: int, causal: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.H, self.hd, self.causal = num_heads, dim // num_heads, causal
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = 1.0 / math.sqrt(self.hd)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.H, self.hd) \
                             .permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(
                ~attn_mask.bool()[:, None, None, :], float('-inf'))
        if self.causal:
            tri = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~tri, float('-inf'))
        out = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0) @ v
        return self.out_proj(out.transpose(1, 2).reshape(B, S, D))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Harness — associative recall (routing-sensitive synthetic task)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Sequence = [k1 v1 k2 v2 ... kn vn  Q kq]  -> predict the value paired with kq.
# Solvable only by routing the query token to the matching key token: a task
# where the routing medium IS the bottleneck. Trained with pure Adam (never
# AdamW — weight decay fights the geometric basin).

class TinyRecallModel(nn.Module):
    def __init__(self, vocab: int, dim: int, attn: nn.Module, n_layers: int = 2,
                 attn_factory=None):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(0.02 * torch.randn(1, 512, dim))
        layers = []
        for i in range(n_layers):
            a = attn if (i == 0 and attn_factory is None) else attn_factory()
            layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(dim), "attn": a,
                "norm2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                     nn.Linear(2 * dim, dim)),
            }))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.emb(ids) + self.pos[:, :ids.shape[1]]
        for L in self.layers:
            x = x + L["attn"](L["norm1"](x))
            x = x + L["mlp"](L["norm2"](x))
        return self.head(x[:, -1])                          # predict from final token


def make_recall_batch(B: int, n_pairs: int, n_keys: int, n_vals: int,
                      device) -> Tuple[Tensor, Tensor]:
    """Tokens: [0, n_keys) keys | [n_keys, n_keys+n_vals) values | Q = last id."""
    Q = n_keys + n_vals
    keys = torch.stack([torch.randperm(n_keys, device=device)[:n_pairs]
                        for _ in range(B)])                 # unique keys per row
    vals = torch.randint(0, n_vals, (B, n_pairs), device=device) + n_keys
    seq = torch.stack([keys, vals], dim=-1).reshape(B, 2 * n_pairs)
    qi = torch.randint(0, n_pairs, (B,), device=device)
    kq = keys.gather(1, qi[:, None])
    target = vals.gather(1, qi[:, None]).squeeze(1)
    ids = torch.cat([seq, torch.full((B, 1), Q, device=device), kq], dim=1)
    return ids, target


def run_harness(mode: str, steps: int = 300, device: str = "cpu",
                seed: int = 1234, log_every: int = 50,
                dim: int = 128, n_heads: int = 4, K: int = 32, D_addr: int = 4,
                n_pairs: int = 12, n_keys: int = 48, n_vals: int = 24,
                batch: int = 64, lr: float = 3e-4,
                div_weight: float = 0.0, tied_address: bool = False,
                codebook_init="fibonacci", lr_decay: bool = True,
                snapshot_codebook: bool = False) -> Dict[str, float]:
    torch.manual_seed(seed)
    vocab = n_keys + n_vals + 1
    if mode == "standard":
        attn_factory = lambda: StandardAttention(dim, n_heads)
        first = attn_factory()
    else:
        cfg = AlephAttentionConfig(dim=dim, num_heads=n_heads, mode=mode,
                                   K=K, D_addr=D_addr, tau=0.1,
                                   tied_address=tied_address,
                                   codebook_init=codebook_init)
        attn_factory = lambda: AlephRoutedAttention(cfg)
        first = attn_factory()
    model = TinyRecallModel(vocab, dim, first, n_layers=2,
                            attn_factory=attn_factory).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)       # pure Adam, never AdamW
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
             eta_min=lr * 0.1) if lr_decay else None)

    aleph_layers = [m for m in model.modules() if isinstance(m, AlephRoutedAttention)]
    for a in aleph_layers:
        a.emit_diversity = div_weight > 0

    print(f"\n=== mode={mode}  params={sum(p.numel() for p in model.parameters()):,} ===")
    final = {}
    snapshots = []                                          # (step, (K,D)) trajectory
    if snapshot_codebook and aleph_layers:
        snapshots.append((0, aleph_layers[0].export_codebook()))
    for step in range(1, steps + 1):
        ids, target = make_recall_batch(batch, n_pairs, n_keys, n_vals, device)
        logits = model(ids)
        loss = F.cross_entropy(logits, target)
        if div_weight > 0:
            loss = loss + div_weight * sum(a.diversity_loss() for a in aleph_layers)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max(loss.item(), 1.0))      # Phil's clip rule
        opt.step()
        if sched is not None:
            sched.step()

        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                acc = (logits.argmax(-1) == target).float().mean().item()
            line = f"  step {step:4d}  loss {loss.item():.4f}  acc {acc:.3f}  |g| {gnorm:.2f}"
            if aleph_layers:
                model.eval()
                x_probe = model.emb(ids) + model.pos[:, :ids.shape[1]]
                st = aleph_layers[0].address_stats(x_probe)
                model.train()
                line += (f"  ppl {st['perplexity']:.1f}/{st['max_perplexity']:.0f}"
                         f"  margin {st['margin']:.3f}  conf {st['confidence']:.3f}"
                         f"  bktCV {st['bucket_cv']:.2f}")
                final.update(st)
            print(line)
            final.update({"loss": loss.item(), "acc": acc})
            if snapshot_codebook and aleph_layers:
                snapshots.append((step, aleph_layers[0].export_codebook()))
    if snapshots:
        final["codebook_snapshots"] = snapshots
    return final


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Smoke tests + activation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _smoke():
    torch.manual_seed(0)
    print("=" * 70)
    print("AlephRoutedAttention — smoke tests")
    print("=" * 70)

    for mode in ("hub", "bucket"):
        for causal in (False, True):
            cfg = AlephAttentionConfig(dim=64, num_heads=4, mode=mode, K=16,
                                       D_addr=4, causal=causal, block_size=16,
                                       chunk_size=32)
            m = AlephRoutedAttention(cfg)
            x = torch.randn(2, 50, 64, requires_grad=True)   # odd S: pad path
            mask = torch.ones(2, 50); mask[1, 40:] = 0
            y = m(x, attn_mask=mask)
            assert y.shape == (2, 50, 64), y.shape
            assert torch.isfinite(y).all()
            y.sum().backward()
            assert torch.isfinite(x.grad).all()
            assert m.codebook.grad is not None and torch.isfinite(m.codebook.grad).all(), \
                f"codebook got no/bad gradient in mode={mode}"
            cb_g = m.codebook.grad.norm().item()
            print(f"  ✓ mode={mode:6s} causal={causal!s:5s}  out {tuple(y.shape)}  "
                  f"codebook |grad|={cb_g:.4f}")
            x.grad = None

    # hub causal == hub full restricted? sanity: causal output at position i must
    # not depend on tokens > i. Perturb a late token; early outputs must not move.
    cfg = AlephAttentionConfig(dim=64, num_heads=4, mode="hub", K=16, D_addr=4,
                               causal=True, chunk_size=16)
    m = AlephRoutedAttention(cfg).eval()
    x = torch.randn(1, 40, 64)
    y1 = m(x)
    x2 = x.clone(); x2[0, 35] += 10.0
    y2 = m(x2)
    assert torch.allclose(y1[0, :35], y2[0, :35], atol=1e-5), "causality leak!"
    print("  ✓ hub causal: no future leakage (perturbation test)")

    # stats sanity
    st = m.address_stats(x)
    assert 1.0 <= st["perplexity"] <= st["max_perplexity"] + 1e-3
    print(f"  ✓ stats: {st}")

    # diversity hook
    m2 = AlephRoutedAttention(AlephAttentionConfig(dim=64, num_heads=4, K=16))
    m2.train(); m2.emit_diversity = True
    _ = m2(torch.randn(2, 20, 64))
    d = m2.diversity_loss()
    assert d.requires_grad and torch.isfinite(d)
    print(f"  ✓ diversity_loss = {d.item():.4f} (grad-carrying)")

    # fibonacci init at D=4 is unit + deterministic
    A = _init_codebook(32, 4, "fibonacci")
    assert torch.allclose(A.norm(dim=-1), torch.ones(32), atol=1e-5)
    print("  ✓ super-Fibonacci codebook init (D=4) unit rows")
    print("All smoke tests passed.\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aleph-routed attention — smoke + A/B harness")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--modes", nargs="+", default=["hub", "bucket", "standard"])
    ap.add_argument("--div-weight", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--smoke-only", action="store_true")
    # parse_known_args: ignore foreign argv (e.g. Jupyter/Colab injects
    # `-f /.../kernel-*.json`), so the module runs in notebooks unchanged
    args, _unknown = ap.parse_known_args()

    _smoke()
    if not args.smoke_only:
        results = {}
        for mode in args.modes:
            results[mode] = run_harness(mode, steps=args.steps, device=args.device,
                                        K=args.K, div_weight=args.div_weight)
        print("\n" + "=" * 70)
        print("A/B summary (associative recall)")
        for mode, r in results.items():
            extra = (f"  ppl {r.get('perplexity', float('nan')):.1f}"
                     f"  margin {r.get('margin', float('nan')):.3f}"
                     if "perplexity" in r else "")
            print(f"  {mode:9s} loss {r['loss']:.4f}  acc {r['acc']:.3f}{extra}")
        print("=" * 70)
        print("\nBasin test entry point: model.export_codebook() -> feed to the")
        print("geolip-svae antipodal-collapse extraction. Preregistered criterion:")
        print("|deviation| < 0.05 on RP^(D-1) = cross-objective attractor evidence.")