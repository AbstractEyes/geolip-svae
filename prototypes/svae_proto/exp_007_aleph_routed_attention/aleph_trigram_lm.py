# aleph_trigram_lm.py
"""
Aleph-Routed Trigram LM — the basin-test vehicle
=================================================

Causal language model over the SAME substrate the geolip-svae aleph batteries
were fed: WikiText-103 as a raw UTF-8 byte stream, trigram-packed (3 bytes per
position, stride 3 — the sequence analogue of ByteTrigramDataset's 3-bytes-per-
cell RGB encoding). One token position = one trigram, embedded byte-factored
(sum of 3 byte embeddings + position) and predicted as 3 independent 256-way
byte heads — exactly the substrate, no 256^3 softmax.

Purpose (preregistered — CORRECTED per RESEARCH_HISTORY.md):
  The program established (Phase 2, discoveries #13/#15) at least TWO stable
  codebook statutes, SELECTED BY SUBSTRATE: uniform-class (|dev| < 0.05; the
  noise solvers) and polytope-class (dev > +0.05, pair fraction >= 45%;
  repulsive packing — the BYTE-TRIGRAM solvers, in-distribution dev +0.083).
  dev < -0.05 is degenerate (clumping) — the failure statute. Statute is a
  property of (model x calibration), not the model alone.

  Therefore, on THIS substrate, the basin question is statute-resolved:
    - random init  -> STATUTE-SELECTION test. Codebook settling polytope-class
      (the substrate-matched statute) or uniform-class under pure attention
      gradients = cross-objective attractor evidence. Degenerate = failure.
    - fibonacci init -> DRIFT test. Starts in the uniform basin; migration
      OUT toward polytope under the symbolic substrate = substrate-driven
      statute selection in a new objective (the stronger result).
  Run BOTH inits. Statute (deviation + pair fraction, computed per the
  program's own Sec 3.11 definitions) is logged per snapshot inline below;
  void-richness (beta_2/axis via ripser on projective angular distances,
  the symbolic-substrate fingerprint, discovery #20) is the deeper follow-up
  on the saved snapshots. Note the two non-interchangeable "margins"
  (top-1 softmax probability vs projective |<m,a>|): the inline monitors
  detect collapse only; structure evidence is deviation/statute/beta_2.

Substrate fidelity (mirrors geolip_svae.dataset_presets.ByteTrigramDataset):
  - corpus: 'wikitext-103-raw-v1' via the Salesforce/wikitext namespace
    (the bare 'wikitext' hub name is deprecated), or any local .txt path
  - raw bytes in a single uint8 numpy array (prototypes/CLAUDE.md trap #3:
    Python lists of ints are 5-7x the memory; never materialize them)
  - seeded window sampling over the stream
Repo invariants honored: pure Adam (never AdamW), no BatchNorm/Dropout on the
geometric path, grad clip = max(task_loss, 1.0).

Usage (Colab / Blackwell):
    from aleph_trigram_lm import TrigramLMConfig, train_trigram_lm
    cfg = TrigramLMConfig(steps=10_000, device='cuda',
                          attn_mode='hub', codebook_init='random')
    result = train_trigram_lm(cfg)
    # result['codebook_snapshots'] -> [(step, (K, D_addr) tensor), ...]
    # also saved to cfg.snapshot_path for the extraction pass

Baseline A/B: attn_mode='standard' runs the same LM with softmax attention.

Author: AbstractPhil + Mirel
Date: 2026-06-09
License: MIT
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aleph_routed_attention import (
    AlephRoutedAttention, AlephAttentionConfig, StandardAttention,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrigramLMConfig:
    """Everything for one basin run.

    Substrate:
        corpus_id:        HF dataset config name (default = the aleph batteries'
                          corpus) OR a local .txt/.text path
        max_corpus_bytes: cap on bytes loaded (None = whole corpus, ~520 MB for
                          wikitext-103). 50–100 MB is plenty for these runs.
        seq_len:          context length in TRIGRAMS (bytes seen = 3*seq_len)

    Model:
        dim/n_layers/n_heads: transformer shell
        attn_mode:        'hub' | 'bucket' | 'standard'
        K/D_addr/tau:     aleph routing knobs (ignored for 'standard')
        codebook_init:    'random' for the basin test (MANDATORY there) |
                          'fibonacci' | (K, D_addr) array transplant

    Training:
        pure Adam + cosine decay to 10%; loss reported in nats and bits/byte.
    """
    # substrate
    corpus_id: str = "wikitext-103-raw-v1"
    split: str = "train"
    max_corpus_bytes: Optional[int] = 100_000_000
    seq_len: int = 256                      # trigrams (= 768 bytes of context)
    seed: int = 1234

    # model
    dim: int = 384
    n_layers: int = 4
    n_heads: int = 6
    attn_mode: str = "hub"                  # 'hub' | 'bucket' | 'standard'
    K: int = 64
    D_addr: int = 4
    tau: float = 0.1
    codebook_init: object = "random"        # basin test requires 'random'
    div_weight: float = 0.0                 # anti-collapse; run 0 first, observe

    # paradigm + scale
    shared_codebook: bool = True            # ONE vocabulary, many speakers: all
                                            # layers address the same (K,D) param,
                                            # concentrating address pressure n_layers-x
    accum_steps: int = 1                    # gradient accumulation (effective batch
                                            # = batch_size * accum_steps)
    stream_segments: int = 1                # segments per sample, each seq_len long;
                                            # codebook-memory state carried across
                                            # (TBPTT, detached between segments).
                                            # context = seq_len * stream_segments
                                            # at CONSTANT attention memory. hub-only.
    probe_rows: int = 200_000               # address-stats sample size (ppl estimator)

    # training
    steps: int = 10_000                     # optimizer steps (micro-batches =
                                            # steps * accum_steps)
    batch_size: int = 32
    lr: float = 3e-4
    lr_decay: bool = True
    log_every: int = 250
    eval_batches: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = False                       # bf16 autocast on the shell (the
                                            # address stays fp32 inside)

    # outputs
    snapshot_codebook: bool = True
    snapshot_path: str = "aleph_lm_codebook_snapshots.pt"
    checkpoint_path: Optional[str] = "aleph_trigram_lm.pt"

    def __post_init__(self):
        assert self.attn_mode in ("hub", "bucket", "standard")
        assert self.dim % self.n_heads == 0
        assert self.accum_steps >= 1 and self.stream_segments >= 1
        if self.stream_segments > 1:
            assert self.attn_mode == "hub", \
                "streaming requires mode='hub' (bucket sorts globally)"




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Statute monitor — the program's own diagnostic geometry (Sec 3.11)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _canon(x: Tensor) -> Tensor:
    """Sign-canonicalize onto RP^(D-1): flip so the first nonzero coord is
    positive (antipodes map to one representative)."""
    x = F.normalize(x, dim=-1)
    first_nz = x[torch.arange(len(x)), x.abs().argmax(dim=-1)]
    return x * torch.sign(first_nz).unsqueeze(-1)


def _mean_projective_angle(X: Tensor) -> float:
    """Mean pairwise acos|cos| over distinct pairs (radians)."""
    c = (X @ X.t()).clamp(-1.0, 1.0).abs()
    iu = torch.triu_indices(len(X), len(X), offset=1)
    return torch.acos(c[iu[0], iu[1]]).mean().item()


_UNIFORM_BASELINE: Dict[int, float] = {}

def projective_deviation(axes: Tensor, n_ref: int = 4096,
                         seed: int = 0) -> float:
    """Uniformity deviation per the program definition: mean pairwise
    projective angle of the axes MINUS the same statistic for n_ref uniform
    random projective points at the same D. Signed; sign matters."""
    D = axes.shape[-1]
    if D not in _UNIFORM_BASELINE:
        g = torch.Generator().manual_seed(seed)
        ref = F.normalize(torch.randn(n_ref, D, generator=g), dim=-1)
        _UNIFORM_BASELINE[D] = _mean_projective_angle(ref)
    return _mean_projective_angle(F.normalize(axes.float(), dim=-1)) \
        - _UNIFORM_BASELINE[D]


def antipodal_pair_fraction(axes: Tensor, thresh: float = -0.9) -> float:
    """Fraction of rows in mutual most-negative pairs with cos < thresh
    (the antipodal-collapse acceptance rule)."""
    A = F.normalize(axes.float(), dim=-1)
    c = A @ A.t()
    c.fill_diagonal_(0.0)
    partner = c.argmin(dim=-1)
    val = c.gather(-1, partner.unsqueeze(-1)).squeeze(-1)
    mutual = partner[partner] == torch.arange(len(A))
    return ((val < thresh) & mutual).float().mean().item()


def statute(axes: Tensor) -> Dict[str, object]:
    """Classify per the program taxonomy: dev > +0.05 polytope-class
    (repulsive packing); |dev| < 0.05 uniform-class; dev < -0.05 degenerate
    (clumping, the failure statute)."""
    dev = projective_deviation(axes)
    pf = antipodal_pair_fraction(axes)
    cls = ("polytope" if dev > 0.05 else
           "degenerate" if dev < -0.05 else "uniform")
    return {"deviation": dev, "pair_fraction": pf, "statute": cls}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Substrate — trigram stream (mirrors ByteTrigramDataset's loading)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrigramStream:
    """WikiText-103 (or local .txt) as a uint8 byte stream, sampled as
    causal trigram sequences.

    __call__(batch, seq_len) -> (ids, targets):
        ids:     (B, S, 3) uint8->long — trigram t  = bytes[3t : 3t+3]
        targets: (B, S, 3)             — trigram t+1 (next-trigram prediction)
    Windows are sampled at byte offsets aligned to stride 3 so the trigram
    framing matches the image packing (cell i = bytes[3i : 3i+3])."""

    def __init__(self, corpus_id: str, split: str = "train",
                 max_corpus_bytes: Optional[int] = None, seed: int = 1234):
        if os.path.isfile(corpus_id) and corpus_id.endswith((".txt", ".text")):
            print(f"[TrigramStream] loading local corpus {corpus_id} ...")
            with open(corpus_id, "rb") as f:
                raw = f.read(max_corpus_bytes) if max_corpus_bytes else f.read()
            self.stream = np.frombuffer(raw, dtype=np.uint8).copy()
        else:
            print(f"[TrigramStream] loading HF corpus {corpus_id} ...")
            from datasets import load_dataset
            if corpus_id.startswith("wikitext"):
                ds = load_dataset("Salesforce/wikitext", corpus_id, split=split)
            else:
                ds = load_dataset(corpus_id, split=split)
            # accumulate utf-8 bytes directly into a byte buffer — never a
            # Python list of ints (prototypes/CLAUDE.md memory trap #3)
            buf = bytearray()
            cap = max_corpus_bytes or float("inf")
            for row in ds:
                t = row.get("text", "")
                if t:
                    buf.extend(t.encode("utf-8", errors="ignore"))
                if len(buf) >= cap:
                    break
            self.stream = np.frombuffer(
                bytes(buf[: max_corpus_bytes] if max_corpus_bytes else buf),
                dtype=np.uint8).copy()
        n_tri = len(self.stream) // 3
        print(f"[TrigramStream] {len(self.stream):,} bytes "
              f"= {n_tri:,} trigrams")
        assert n_tri > 0, "corpus too small"
        self._rng = np.random.default_rng(seed)

    def sample(self, batch: int, seq_len: int,
               device) -> Tuple[Tensor, Tensor]:
        need = 3 * (seq_len + 1)                       # +1 trigram for targets
        hi = len(self.stream) - need
        assert hi > 0, f"corpus shorter than one window ({need} bytes)"
        starts = self._rng.integers(0, hi // 3, size=batch) * 3   # stride-3 aligned
        idx = starts[:, None] + np.arange(need)[None, :]          # (B, need)
        window = self.stream[idx]                                  # (B, need) uint8
        tri = torch.from_numpy(window.astype(np.int64)) \
                   .view(batch, seq_len + 1, 3)
        ids, targets = tri[:, :-1], tri[:, 1:]
        return ids.to(device), targets.to(device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model — byte-factored trigram LM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrigramLM(nn.Module):
    """Causal LM over trigram positions. Embedding = sum of three per-slot
    byte embeddings (+ learned positions); head = three 256-way byte heads.
    Geometric-path hygiene: no BatchNorm/Dropout, pure pre-LN residual shell."""

    def __init__(self, cfg: TrigramLMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.byte_emb = nn.ModuleList([nn.Embedding(256, d) for _ in range(3)])
        self.pos = nn.Parameter(0.02 * torch.randn(1, cfg.seq_len, d))

        def make_attn() -> nn.Module:
            if cfg.attn_mode == "standard":
                return StandardAttention(d, cfg.n_heads, causal=True)
            return AlephRoutedAttention(AlephAttentionConfig(
                dim=d, num_heads=cfg.n_heads, mode=cfg.attn_mode,
                K=cfg.K, D_addr=cfg.D_addr, tau=cfg.tau, causal=True,
                codebook_init=cfg.codebook_init))

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(d), "attn": make_attn(),
                "norm2": nn.LayerNorm(d),
                "mlp": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                     nn.Linear(4 * d, d)),
            }) for _ in range(cfg.n_layers)
        ])
        self.norm_f = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, 256) for _ in range(3)])

        # one vocabulary, many speakers: tie every layer's codebook to layer 0's
        if cfg.shared_codebook and cfg.attn_mode in ("hub", "bucket"):
            shared = self.layers[0]["attn"].codebook
            for L in self.layers[1:]:
                L["attn"].codebook = shared

    def aleph_layers(self) -> List[AlephRoutedAttention]:
        return [m for m in self.modules() if isinstance(m, AlephRoutedAttention)]

    def backbone(self, ids: Tensor) -> Tensor:
        """ids: (B, S, 3) -> (B, S, dim)"""
        x = sum(emb(ids[..., i]) for i, emb in enumerate(self.byte_emb))
        x = x + self.pos[:, : ids.shape[1]]
        for L in self.layers:
            x = x + L["attn"](L["norm1"](x))
            x = x + L["mlp"](L["norm2"](x))
        return self.norm_f(x)

    def forward(self, ids: Tensor) -> Tensor:
        """(B, S, 3) -> byte logits (B, S, 3, 256)"""
        h = self.backbone(ids)
        return torch.stack([head(h) for head in self.heads], dim=2)

    def loss(self, ids: Tensor, targets: Tensor) -> Tensor:
        """Mean cross-entropy per byte (nats/byte). bpb = loss / ln 2."""
        logits = self(ids)                              # (B,S,3,256)
        return F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))

    # ── streaming: segment-recurrent backbone (codebook memory carried) ──
    def stream_loss(self, ids: Tensor, targets: Tensor,
                    states: Optional[List] = None
                    ) -> Tuple[Tensor, List]:
        """One SEGMENT with per-layer carried states. states[i] is layer i's
        (Mp, Mm, zp, zm) or None. Returns (loss, new_states) — caller detaches
        between backward passes (TBPTT-1: grads flow within segment; values
        flow forever)."""
        x = sum(emb(ids[..., i]) for i, emb in enumerate(self.byte_emb))
        x = x + self.pos[:, : ids.shape[1]]
        new_states: List = []
        states = states or [None] * len(self.layers)
        for L, st in zip(self.layers, states):
            a, ns = L["attn"].forward_stream(L["norm1"](x), state=st)
            x = x + a
            x = x + L["mlp"](L["norm2"](x))
            new_states.append(ns)
        h = self.norm_f(x)
        logits = torch.stack([head(h) for head in self.heads], dim=2)
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        return loss, new_states


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_trigram_lm(cfg: TrigramLMConfig,
                     stream: Optional[TrigramStream] = None) -> Dict:
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    stream = stream or TrigramStream(cfg.corpus_id, cfg.split,
                                     cfg.max_corpus_bytes, cfg.seed)
    model = TrigramLM(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)   # pure Adam, never AdamW
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.steps, eta_min=cfg.lr * 0.1) if cfg.lr_decay else None)

    alephs = model.aleph_layers()
    for a in alephs:
        a.emit_diversity = cfg.div_weight > 0

    snapshots: List[Tuple[int, Tensor]] = []
    if cfg.snapshot_codebook and alephs:
        snapshots.append((0, alephs[0].export_codebook()))

    print(f"\n=== trigram LM  mode={cfg.attn_mode}  params={n_params:,}  "
          f"ctx={cfg.seq_len}x{cfg.stream_segments} trigrams "
          f"({3*cfg.seq_len*cfg.stream_segments} bytes)  "
          f"eff.batch={cfg.batch_size*cfg.accum_steps}  "
          f"shared_cb={cfg.shared_codebook}  "
          f"device={dev} ===")
    autocast = (torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
                if cfg.amp and dev.type == "cuda" else None)
    result: Dict = {"mode": cfg.attn_mode, "params": n_params}
    t0 = time.time()

    segs = cfg.stream_segments
    micro_scale = 1.0 / (cfg.accum_steps * segs)
    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_sum, n_micro = 0.0, 0
        for _ in range(cfg.accum_steps):
            # one long sample, split into `segs` carried segments
            ids, targets = stream.sample(cfg.batch_size, cfg.seq_len * segs, dev)
            states = None
            for s in range(segs):
                sl = slice(s * cfg.seq_len, (s + 1) * cfg.seq_len)
                seg_ids, seg_tgt = ids[:, sl], targets[:, sl]
                if autocast:
                    with autocast:
                        if segs > 1:
                            loss, states = model.stream_loss(seg_ids, seg_tgt, states)
                        else:
                            loss = model.loss(seg_ids, seg_tgt)
                else:
                    if segs > 1:
                        loss, states = model.stream_loss(seg_ids, seg_tgt, states)
                    else:
                        loss = model.loss(seg_ids, seg_tgt)
                total = loss
                if cfg.div_weight > 0 and alephs:
                    total = total + cfg.div_weight * sum(
                        a.diversity_loss() for a in alephs)
                (total * micro_scale).backward()
                loss_sum += loss.item(); n_micro += 1
                if states is not None:                       # TBPTT boundary
                    states = [tuple(t.detach() for t in st) for st in states]
        loss_avg = loss_sum / n_micro
        gnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max(loss_avg, 1.0))           # the clip rule
        opt.step()
        if sched is not None:
            sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps:
            bpb = loss_avg / math.log(2)
            rate = (step * cfg.batch_size * cfg.seq_len * segs
                    * cfg.accum_steps) / (time.time() - t0)
            line = (f"  step {step:6d}  loss {loss_avg:.4f}  "
                    f"bpb {bpb:.3f}  |g| {gnorm:.2f}  {rate/1e3:.1f}k tri/s")
            if alephs:
                model.eval()
                with torch.no_grad():
                    x_probe = model.backbone(
                        ids[: min(8, cfg.batch_size), -cfg.seq_len:])
                st = alephs[0].address_stats(x_probe, max_rows=cfg.probe_rows)
                model.train()
                line += (f"  ppl {st['perplexity']:.1f}/{st['max_perplexity']:.0f}"
                         f"  margin {st['margin']:.4f}"
                         f"  conf {st['confidence']:.3f}")
                result.update(st)
                if cfg.snapshot_codebook:
                    snapshots.append((step, alephs[0].export_codebook()))
            print(line)
            result.update({"loss": loss_avg, "bpb": bpb, "step": step})

    if snapshots:
        result["codebook_snapshots"] = snapshots
        drift = (snapshots[-1][1] - snapshots[0][1]).norm().item()
        result["codebook_drift"] = drift
        traj = [(s, statute(cb)) for s, cb in snapshots]
        result["statute_trajectory"] = traj
        torch.save({"snapshots": snapshots, "statute_trajectory": traj,
                    "config": cfg.__dict__, "K": cfg.K, "D_addr": cfg.D_addr},
                   cfg.snapshot_path)
        print(f"\n[basin] {len(snapshots)} snapshots -> {cfg.snapshot_path}"
              f"   drift |A_end - A_0| = {drift:.4f}")
        print("[basin] statute trajectory (program taxonomy: polytope is the "
              "substrate-matched\n        statute for byte-trigram; uniform is "
              "the noise/OOD statute; degenerate = failure):")
        for s, st in traj:
            print(f"    step {s:6d}  dev {st['deviation']:+.4f}  "
                  f"pairs {st['pair_fraction']:.0%}  -> {st['statute']}")
        print("[basin] deeper follow-up on saved snapshots: beta_2/axis via "
              "ripser on projective\n        angular distances (the "
              "void/symbolic fingerprint, discovery #20).")
    if cfg.checkpoint_path:
        torch.save({"model_state_dict": model.state_dict(),
                    "config": cfg.__dict__}, cfg.checkpoint_path)
        print(f"[ckpt] saved -> {cfg.checkpoint_path}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Activation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _smoke(device: str = "cpu"):
    """End-to-end on a synthetic local corpus — no downloads."""
    print("=" * 70)
    print("aleph_trigram_lm — smoke (synthetic corpus)")
    print("=" * 70)
    path = "/tmp/_smoke_corpus.txt"
    rng = np.random.default_rng(0)
    words = [b"the", b"aleph", b"address", b"routes", b"attention",
             b"through", b"a", b"learned", b"projective", b"codebook"]
    with open(path, "wb") as f:
        f.write(b" ".join(words[i] for i in rng.integers(0, len(words), 60_000)))
    cfg = TrigramLMConfig(corpus_id=path, steps=30, log_every=10,
                          dim=96, n_layers=2, n_heads=4, K=16, seq_len=64,
                          batch_size=8, device=device,
                          snapshot_path="/tmp/_smoke_snaps.pt",
                          checkpoint_path=None)
    r = train_trigram_lm(cfg)
    assert "codebook_snapshots" in r and len(r["codebook_snapshots"]) >= 2
    assert math.isfinite(r["loss"]) and r["loss"] < math.log(256)
    print(f"\nsmoke OK — loss {r['loss']:.3f} (< ln256={math.log(256):.3f} prior), "
          f"drift {r['codebook_drift']:.4f}, "
          f"{len(r['codebook_snapshots'])} snapshots saved")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aleph trigram LM — basin run")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--mode", default="hub", choices=["hub", "bucket", "standard"])
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--corpus-mb", type=int, default=100)
    ap.add_argument("--codebook-init", default="random")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args, _unknown = ap.parse_known_args()      # notebook-safe (ignores -f kernel.json)

    if args.smoke_only:
        _smoke(device="cpu")
    else:
        cfg = TrigramLMConfig(attn_mode=args.mode, steps=args.steps,
                              max_corpus_bytes=args.corpus_mb * 1_000_000,
                              codebook_init=args.codebook_init,
                              device=args.device)
        train_trigram_lm(cfg)