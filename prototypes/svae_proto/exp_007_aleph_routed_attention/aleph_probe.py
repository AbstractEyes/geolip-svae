# aleph_probe_battery.py
"""
Aleph Probe Battery — second-order statistics for the attention-trained codebook
==================================================================================

First-order geometry (uniformity deviation) is exhausted: both the recon and
attention objectives land near-uniform, so deviation can no longer tell the
systems apart. This battery runs the program's second-order instruments, plus
the statistics the recon program structurally could not have:

  [B2]   beta_2/axis void trajectory   — ripser persistent homology on the
         projective angular distances of every saved codebook snapshot
         (Sec 3.11 recipe: maxdim 2, threshold 20 deg). The substrate
         fingerprint (discovery #20): recon-aleph 0.56/axis vs SVAE 0.08.
  [PM]   Projective margin |<m, a>|    — the assessor margin (the 0.967-vs-
         0.929 currency), computed on the q-cloud and k-cloud SEPARATELY
         over in-distribution text.
  [QK]   q/k asymmetry                 — per-cloud statute + margin, and the
         self-listening angle (each token's q-address vs its own k-address).
  [PR]   Procrustes residual           — attention codebook vs the hosted
         recon-aleph codebook (same K=64, D=4, same substrate): ICP-style
         sign-aware match + orthogonal alignment on RP^3, against a null of
         independent uniform codebook pairs. Sub-null residual = the two
         objectives found the same VOCABULARY up to rotation.
  [CV]   Raw-weight CV of q_addr/k_addr vs the CM band (0.13-0.30; the
         0.29154 boundary). Row- and column-norm CV both reported; the
         exact CM-pipeline definition may differ — treat as indicative.
  [TOPO] Percolation angle + local intrinsic dimension of the final codebook
         (completes the topology triple with beta_2).
  [TAU]  The 0.857 anomaly — confidence vs tau on trained / untrained /
         random rows. Determines whether confidence is a kernel invariant
         of (tau, K, D) or a learned quantity.
  [ER]   Streaming-state erank        — occupancy of the (M+, M-) codebook
         memory after streaming a real document vs noise (fp64 SVD per the
         house invariant). The capacity statistic of the recurrent regime.

Usage:
    python aleph_probe_battery.py \\
        --checkpoint aleph_trigram_lm.pt \\
        --snapshots  aleph_lm_codebook_snapshots.pt \\
        --recon-version aleph_byte_trigram_tied_hard_K64
    # or from a notebook:
    from aleph_probe_battery import ProbeConfig, run_battery
    results = run_battery(ProbeConfig(checkpoint='aleph_trigram_lm.pt',
                                      snapshots='aleph_lm_codebook_snapshots.pt'))

Outputs a printed report + probe_results.pt (all tensors and tables).
Requires: torch, numpy, ripser, scipy (assignment), huggingface_hub (Procrustes
reference download; section skips gracefully offline).

Author: AbstractPhil + Mirel
Date: 2026-06-09
License: MIT
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from aleph_routed_attention import AlephRoutedAttention
from aleph_trigram_lm import (
    TrigramLMConfig, TrigramLM, TrigramStream, statute, projective_deviation,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ProbeConfig:
    checkpoint: str = "aleph_trigram_lm.pt"
    snapshots: str = "aleph_lm_codebook_snapshots.pt"
    out_path: str = "probe_results.pt"

    # Procrustes reference (the hosted recon-aleph)
    recon_repo: str = "AbstractPhil/geolip-aleph-void"
    recon_version: str = "aleph_byte_trigram_tied_hard_K64"
    procrustes_null_seeds: int = 16

    # probes
    corpus_bytes: int = 5_000_000
    probe_batch: int = 16
    cloud_rows: int = 4000
    ripser_thresholds_deg: Tuple[float, ...] = (20.0, 30.0, 45.0, 60.0)
                                             # pipeline profile (phase curve)
    ripser_maxdim: int = 2
    knn_k: int = 10                          # local intrinsic dim (pipeline k)
    tau_sweep: Tuple[float, ...] = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
    stream_doc_segments: int = 8             # erank probe: 8 x seq_len context
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _canon(x: Tensor) -> Tensor:
    """Sign-canonicalize onto RP^(D-1) (first-largest-coord positive)."""
    x = F.normalize(x.float(), dim=-1)
    lead = x[torch.arange(len(x)), x.abs().argmax(dim=-1)]
    return x * torch.sign(lead).unsqueeze(-1)


def projective_angle_matrix(axes: Tensor) -> np.ndarray:
    """Pairwise acos|cos| (radians) — the TRUE RP metric (caps at 90 deg).
    Used for statute-side probes; NOT the topology instrument."""
    A = F.normalize(axes.float(), dim=-1)
    c = (A @ A.t()).clamp(-1.0, 1.0).abs()
    d = torch.acos(c)
    d.fill_diagonal_(0.0)
    return d.cpu().numpy().astype(np.float64)


def pipeline_angle_matrix(axes: Tensor) -> np.ndarray:
    """EXACT replication of train_codebook._pairwise_angular_dist: arccos(dot)
    on SIGN-CANONICALIZED unit axes (the program's topology instrument).
    Differs from the true RP metric near the canonical boundary; published
    beta_2 references (0.56 / 0.08) are in THIS metric — match it."""
    A = _canon(axes)
    dot = (A @ A.t()).clamp(-1.0, 1.0)
    ang = torch.acos(dot)
    ang.fill_diagonal_(0.0)
    return ang.cpu().numpy().astype(np.float64)


def load_lm(checkpoint: str, device: str) -> Tuple[TrigramLM, TrigramLMConfig]:
    d = torch.load(checkpoint, map_location=device, weights_only=False)
    fields = TrigramLMConfig.__dataclass_fields__
    cfg = TrigramLMConfig(**{k: v for k, v in d["config"].items() if k in fields})
    model = TrigramLM(cfg).to(device)
    model.load_state_dict(d["model_state_dict"])
    return model.eval(), cfg


def address_clouds(model: TrigramLM, cfg: TrigramLMConfig, ids: Tensor,
                   n_rows: int, seed: int = 0
                   ) -> Tuple[Tensor, Tensor]:
    """(q_cloud, k_cloud): unit address rows of layer 0 over the given ids,
    PAIRED (same subsample indices), each (n_rows, D_addr)."""
    with torch.no_grad():
        x = sum(emb(ids[..., i]) for i, emb in enumerate(model.byte_emb))
        x = x + model.pos[:, : ids.shape[1]]
        attn = model.layers[0]["attn"]
        xn = model.layers[0]["norm1"](x)
        B, S = ids.shape[0], ids.shape[1]
        qh = attn._split_addr(attn.q_addr(xn), B, S).reshape(-1, cfg.D_addr)
        kh = attn._split_addr(attn.k_addr(xn), B, S).reshape(-1, cfg.D_addr)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(qh.shape[0], generator=g)[: min(n_rows, qh.shape[0])]
    return qh[idx].cpu(), kh[idx].cpu()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [B2] Void trajectory — beta_2/axis over the snapshot sequence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def persistence_profile(axes: Tensor, thresholds_deg: Tuple[float, ...],
                        maxdim: int) -> Dict[float, Dict[str, float]]:
    """EXACT replication of the pipeline's _persistent_homology_profile:
    ripser on the canon'd arccos(dot) matrix at each threshold, finite and
    infinite features counted per H_dim. beta_2/axis = finite H2 / n_axes."""
    from ripser import ripser
    Dm = pipeline_angle_matrix(axes)
    prof: Dict[float, Dict[str, float]] = {}
    for deg in thresholds_deg:
        res = ripser(Dm, distance_matrix=True, maxdim=maxdim,
                     thresh=math.radians(deg))
        row: Dict[str, float] = {}
        for dim, dgm in enumerate(res["dgms"]):
            fin = int(np.isfinite(dgm[:, 1]).sum()) if len(dgm) else 0
            row[f"H{dim}_finite"] = fin
            row[f"H{dim}_infinite"] = int(len(dgm) - fin)
        row["beta2_per_axis"] = row.get("H2_finite", 0) / len(axes)
        prof[deg] = row
    return prof


def probe_void_trajectory(cfg: ProbeConfig) -> List[Tuple[int, Dict]]:
    d = torch.load(cfg.snapshots, map_location="cpu", weights_only=False)
    traj = []
    for step, cb in d["snapshots"]:
        traj.append((step, persistence_profile(cb, cfg.ripser_thresholds_deg,
                                               cfg.ripser_maxdim)))
    return traj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [PM]+[QK] Projective margin and q/k asymmetry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def projective_margin(rows: Tensor, codebook: Tensor) -> float:
    """Assessor margin: mean over rows of max_axis |<m, a>| (antipode-
    invariant). The 0.967-vs-0.929 currency — NOT the softmax-prob margin."""
    A = F.normalize(codebook.float(), dim=-1)
    m = F.normalize(rows.float(), dim=-1)
    return (m @ A.t()).abs().amax(dim=-1).mean().item()


def probe_qk(model: TrigramLM, cfg_lm: TrigramLMConfig, pcfg: ProbeConfig,
             stream: TrigramStream) -> Dict:
    ids, _ = stream.sample(pcfg.probe_batch, cfg_lm.seq_len, pcfg.device)
    q, k = address_clouds(model, cfg_lm, ids, pcfg.cloud_rows, pcfg.seed)
    cb = model.layers[0]["attn"].codebook.detach().cpu()
    # self-listening angle: each token's q-address vs its own k-address
    self_cos = (F.normalize(q, dim=-1) * F.normalize(k, dim=-1)).sum(-1)
    self_ang = torch.acos(self_cos.clamp(-1, 1)) * 180 / math.pi
    return {
        "q_margin": projective_margin(q, cb),
        "k_margin": projective_margin(k, cb),
        "q_statute": statute(q), "k_statute": statute(k),
        "self_listen_deg_mean": self_ang.mean().item(),
        "self_listen_deg_std": self_ang.std().item(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [PR] Procrustes residual vs the hosted recon-aleph codebook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_recon_codebook(repo: str, version: str) -> Optional[Tensor]:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo,
                            filename=f"{version}/checkpoints/best.pt")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        return sd["codebook"].float()
    except Exception as e:                                  # offline / layout drift
        print(f"  [PR] skipped — could not fetch reference codebook: {e}")
        return None


def procrustes_residual(A: Tensor, B: Tensor, iters: int = 12) -> float:
    """ICP-style projective alignment of two unordered axis sets (same K, D):
    iterate [sign-aware assignment on |cos| -> orthogonal map by SVD].
    Residual = mean projective angle (deg) between matched axes after
    alignment. Reflections allowed (projective)."""
    from scipy.optimize import linear_sum_assignment
    A = F.normalize(A.float(), dim=-1)
    B = F.normalize(B.float(), dim=-1)
    K, D = A.shape
    R = torch.eye(D)
    best = 1e9
    for _ in range(iters):
        BR = B @ R.t()
        C = A @ BR.t()                                      # (K, K) signed cos
        r, c = linear_sum_assignment(-C.abs().numpy())      # maximize |cos|
        sign = torch.sign(C[r, c]).unsqueeze(-1)
        Am, Bm = A[r], BR[c] * sign                         # matched, oriented
        resid = torch.acos((Am * Bm).sum(-1).clamp(-1, 1)
                           ).mean().item() * 180 / math.pi
        best = min(best, resid)
        # orthogonal update on the ORIGINAL B (compose fresh each round)
        M = (A[r] * 1.0).t() @ (B[c] * sign)                # (D, D)
        U, _, Vt = torch.linalg.svd(M.double())             # fp64 SVD (house rule)
        R = (U @ Vt).float()
    return best


def probe_procrustes(att_cb: Tensor, pcfg: ProbeConfig) -> Dict:
    ref = fetch_recon_codebook(pcfg.recon_repo, pcfg.recon_version)
    if ref is None:
        return {"skipped": True}
    if ref.shape != att_cb.shape:
        return {"skipped": True,
                "reason": f"shape mismatch {tuple(ref.shape)} vs {tuple(att_cb.shape)}"}
    resid = procrustes_residual(att_cb, ref)
    K, D = att_cb.shape
    null = []
    for s in range(pcfg.procrustes_null_seeds):             # uniform-vs-uniform null
        g1 = torch.Generator().manual_seed(1000 + s)
        g2 = torch.Generator().manual_seed(2000 + s)
        n1 = F.normalize(torch.randn(K, D, generator=g1), dim=-1)
        n2 = F.normalize(torch.randn(K, D, generator=g2), dim=-1)
        null.append(procrustes_residual(n1, n2))
    null = np.array(null)
    return {"residual_deg": resid, "null_mean_deg": float(null.mean()),
            "null_std_deg": float(null.std()),
            "z_vs_null": float((resid - null.mean()) / max(null.std(), 1e-9)),
            "reference": pcfg.recon_version}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [CV] Raw-weight CV of the address projections vs the CM band
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def weight_cv(W: Tensor) -> Dict[str, float]:
    rn = W.norm(dim=1); cn = W.norm(dim=0)
    return {"row_cv": (rn.std() / rn.mean()).item(),
            "col_cv": (cn.std() / cn.mean()).item()}


def probe_cv_band(model: TrigramLM) -> Dict:
    out = {}
    a0 = model.layers[0]["attn"]
    out["q_addr"] = weight_cv(a0.q_addr.weight.detach())
    out["k_addr"] = weight_cv(a0.k_addr.weight.detach())
    out["band"] = "CM band 0.13-0.30; boundary constant 0.29154 "\
                  "(definition indicative — verify against the CM pipeline)"
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [TOPO] Percolation angle + local intrinsic dimension (final codebook)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def probe_topology(axes: Tensor, knn_k: int) -> Dict:
    Dm = pipeline_angle_matrix(axes)                        # the instrument metric
    K = len(axes)
    # percolation: first theta where largest connected component >= 50%
    perc = None
    for deg in np.arange(1.0, 91.0, 1.0):
        adj = Dm <= math.radians(deg)
        # union-find via BFS over numpy
        seen = np.zeros(K, bool); largest = 0
        for s in range(K):
            if seen[s]:
                continue
            stack, comp = [s], 0
            seen[s] = True
            while stack:
                i = stack.pop(); comp += 1
                nbrs = np.where(adj[i] & ~seen)[0]
                seen[nbrs] = True; stack.extend(nbrs.tolist())
            largest = max(largest, comp)
        if largest >= K / 2:
            perc = float(deg)
            break
    # local intrinsic dim by kNN PCA (canon'd axes)
    A = _canon(axes).numpy()
    dims, prs = [], []
    order = np.argsort(Dm, axis=1)
    for i in range(K):
        nbr = A[order[i, 1: knn_k + 1]] - A[i]
        nbr = nbr - nbr.mean(axis=0)                        # pipeline: centered
        lam = np.sort((np.linalg.svd(nbr, compute_uv=False) ** 2) / knn_k)[::-1]
        lam = lam / max(lam.sum(), 1e-12)
        dims.append(int((lam > 0.05 * lam[0]).sum()))
        prs.append(float(1.0 / max((lam ** 2).sum(), 1e-12)))
    return {"percolation_deg": perc,
            "local_dim_mean": float(np.mean(dims)),
            "participation_ratio_mean": float(np.mean(prs))}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [TAU] The 0.857 anomaly — confidence vs tau, trained/untrained/random
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def confidence_at_tau(rows: Tensor, codebook: Tensor, tau: float) -> float:
    A = F.normalize(codebook.float(), dim=-1)
    u = (F.normalize(rows.float(), dim=-1) @ A.t()) / tau
    m = u.abs().amax(-1, keepdim=True)
    ep, en = torch.exp(u - m), torch.exp(-u - m)
    Z = (ep + en).sum(-1, keepdim=True)
    return (((ep - en) / Z) @ A).norm(dim=-1).mean().item()


def probe_tau(model: TrigramLM, cfg_lm: TrigramLMConfig, pcfg: ProbeConfig,
              stream: TrigramStream) -> Dict:
    ids, _ = stream.sample(pcfg.probe_batch, cfg_lm.seq_len, pcfg.device)
    q, _ = address_clouds(model, cfg_lm, ids, pcfg.cloud_rows, pcfg.seed)
    cb = model.layers[0]["attn"].codebook.detach().cpu()
    torch.manual_seed(pcfg.seed)
    fresh = TrigramLM(cfg_lm).eval()
    qf, _ = address_clouds(fresh, cfg_lm, ids.cpu(), pcfg.cloud_rows, pcfg.seed)
    cbf = fresh.layers[0]["attn"].codebook.detach()
    rnd_rows = F.normalize(torch.randn(pcfg.cloud_rows, cfg_lm.D_addr), dim=-1)
    rnd_cb = F.normalize(torch.randn_like(cb), dim=-1)
    table = {}
    for tau in pcfg.tau_sweep:
        table[tau] = {
            "trained": confidence_at_tau(q, cb, tau),
            "untrained": confidence_at_tau(qf, cbf, tau),
            "random": confidence_at_tau(rnd_rows, rnd_cb, tau),
        }
    return table


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [ER] Streaming-state erank — codebook-memory occupancy of a document
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def erank(M: Tensor) -> float:
    """exp(entropy of normalized singular values), fp64 SVD (house rule)."""
    s = torch.linalg.svdvals(M.double())
    p = (s / s.sum().clamp_min(1e-30)).clamp_min(1e-30)
    return float(torch.exp(-(p * p.log()).sum()))


def probe_state_erank(model: TrigramLM, cfg_lm: TrigramLMConfig,
                      pcfg: ProbeConfig, stream: TrigramStream) -> Dict:
    if cfg_lm.attn_mode != "hub":
        return {"skipped": "hub-only"}
    dev = pcfg.device
    segs = pcfg.stream_doc_segments

    def run(ids: Tensor) -> Dict[str, float]:
        states = None
        with torch.no_grad():
            for s in range(segs):
                sl = slice(s * cfg_lm.seq_len, (s + 1) * cfg_lm.seq_len)
                _, states = model.stream_loss(ids[:, sl], ids[:, sl], states)
        Mp, Mm, _, _ = states[0]                            # layer 0
        er_p = np.mean([erank(Mp[0, h]) for h in range(Mp.shape[1])])
        er_m = np.mean([erank(Mm[0, h]) for h in range(Mm.shape[1])])
        # rank ceiling is min(K, head_dim)
        ceil = min(model.layers[0]["attn"].K, model.layers[0]["attn"].hd)
        return {"erank_plus": float(er_p), "erank_minus": float(er_m),
                "ceiling": ceil, "occupancy": float((er_p + er_m) / (2 * ceil))}

    ids_text, _ = stream.sample(2, cfg_lm.seq_len * segs, dev)
    ids_noise = torch.randint(0, 256, ids_text.shape, device=dev)
    return {"text": run(ids_text), "noise": run(ids_noise),
            "context_bytes": 3 * cfg_lm.seq_len * segs}




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [XB2] Extraction-side voids — the apples-to-apples fingerprint probe
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The published 0.56 / 0.08 are EXTRACTED codebooks (antipodal collapse of
# rows), not learned parameters. The proper analogue here: the TYPE codebook —
# the mean k-address of each trigram TYPE is the empirical vocabulary the data
# writes against the frame. Collapse it with the program's exact rule
# (mutual-strongest matching, cos < -0.9, greedy by strength; pair ->
# (u_i - u_j)/||.||, all sign-canonicalized), then run the persistence
# profile. Prediction: voids follow the CONTENT — text type-codebook
# void-richer than noise type-codebook.

def identify_antipodal_pairs_repl(M: Tensor, threshold: float = -0.9):
    """Faithful replication of inference/codebook.py:identify_antipodal_pairs."""
    unit = F.normalize(M.detach().cpu().float(), dim=-1)
    cos = unit @ unit.t()
    cos.fill_diagonal_(1.0)
    V = unit.shape[0]
    claimed = [False] * V
    cands = []
    for i in range(V):
        j = int(cos[i].argmin())
        c = float(cos[i, j])
        if c < threshold:
            cands.append((c, i, j))
    cands.sort()
    pairs, unpaired = [], []
    for _c, i, j in cands:
        if claimed[i] or claimed[j]:
            continue
        if int(cos[j].argmin()) == i or float(cos[j, i]) < threshold:
            pairs.append((min(i, j), max(i, j)))
            claimed[i] = claimed[j] = True
    unpaired = [i for i in range(V) if not claimed[i]]
    return pairs, unpaired


def collapse_to_axes_repl(M: Tensor, pairs, unpaired) -> Tensor:
    """Faithful replication of inference/codebook.py:collapse_to_axes."""
    unit = F.normalize(M.detach().cpu().float(), dim=-1)
    reps = []
    for i, j in pairs:
        m = unit[i] - unit[j]
        reps.append(m / m.norm().clamp_min(1e-12))
    for i in unpaired:
        reps.append(unit[i].clone())
    out = torch.stack(reps) if reps else torch.empty(0, M.shape[1])
    return _canon(out)


def type_codebook(model: TrigramLM, cfg_lm: TrigramLMConfig, ids: Tensor,
                  n_types: int) -> Tensor:
    """Mean layer-0 k-address per trigram TYPE for the n_types most frequent
    types in ids — the empirical vocabulary the data writes on the frame."""
    with torch.no_grad():
        x = sum(emb(ids[..., i]) for i, emb in enumerate(model.byte_emb))
        x = x + model.pos[:, : ids.shape[1]]
        attn = model.layers[0]["attn"]
        kh = attn._split_addr(attn.k_addr(model.layers[0]["norm1"](x)),
                              ids.shape[0], ids.shape[1])
        kh = kh.mean(dim=1)                                # avg heads -> (B,S,Da)
        kh = F.normalize(kh, dim=-1).reshape(-1, cfg_lm.D_addr).cpu()
    tri = (ids[..., 0] * 65536 + ids[..., 1] * 256 + ids[..., 2]).reshape(-1).cpu()
    uniq, inv, counts = torch.unique(tri, return_inverse=True, return_counts=True)
    top = counts.argsort(descending=True)[:n_types]
    reps = []
    for t in top:
        reps.append(F.normalize(kh[inv == t].mean(dim=0), dim=0))
    return torch.stack(reps)


def probe_extraction_voids(model: TrigramLM, cfg_lm: TrigramLMConfig,
                           pcfg: ProbeConfig, stream: TrigramStream,
                           n_types: int = 64) -> Dict:
    ids_text, _ = stream.sample(pcfg.probe_batch, cfg_lm.seq_len, pcfg.device)
    ids_noise = torch.randint(0, 256, ids_text.shape, device=pcfg.device)
    out = {}
    for name, ids in (("text", ids_text), ("noise", ids_noise)):
        tc = type_codebook(model, cfg_lm, ids, n_types)
        pairs, unpaired = identify_antipodal_pairs_repl(tc)
        axes = collapse_to_axes_repl(tc, pairs, unpaired)
        prof = persistence_profile(axes, pcfg.ripser_thresholds_deg,
                                   pcfg.ripser_maxdim)
        out[name] = {"n_types": len(tc), "n_axes": len(axes),
                     "n_pairs": len(pairs),
                     "statute": statute(axes),
                     "profile": prof}
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Battery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_battery(pcfg: ProbeConfig) -> Dict:
    print("=" * 72)
    print("ALEPH PROBE BATTERY")
    print("=" * 72)
    model, cfg_lm = load_lm(pcfg.checkpoint, pcfg.device)
    stream = TrigramStream(cfg_lm.corpus_id, cfg_lm.split,
                           pcfg.corpus_bytes, pcfg.seed)
    cb = model.layers[0]["attn"].codebook.detach().cpu()
    results: Dict = {"checkpoint": pcfg.checkpoint}

    print("\n[B2] void profile trajectory (pipeline metric: canon'd arccos(dot);")
    print("     thresholds (20,30,45,60) deg; recon-aleph ref 0.56, SVAE 0.08)")
    try:
        vt = probe_void_trajectory(pcfg)
        results["void_trajectory"] = vt
        hdr = "    step    " + "".join(f"| th={d:>4.0f}: H1 H2 b2/ax  "
                                       for d in pcfg.ripser_thresholds_deg)
        print(hdr)
        for step, prof in vt[:: max(1, len(vt) // 10)]:
            line = f"    {step:6d}  "
            for d in pcfg.ripser_thresholds_deg:
                r = prof[d]
                line += (f"|  {r['H1_finite']:3d} {r['H2_finite']:3d} "
                         f"{r['beta2_per_axis']:.3f}  ")
            print(line)
        final = vt[-1][1]
        b2 = {d: final[d]["beta2_per_axis"] for d in pcfg.ripser_thresholds_deg}
        print(f"    FINAL b2/axis by threshold: " +
              "  ".join(f"{d:.0f}deg={v:.3f}" for d, v in b2.items()))
    except Exception as e:
        print(f"    skipped: {e}")

    print("\n[PM/QK] projective margin + q/k asymmetry (recon refs 0.967 / 0.929)")
    qk = probe_qk(model, cfg_lm, pcfg, stream)
    results["qk"] = qk
    print(f"    q-cloud margin {qk['q_margin']:.4f}   k-cloud margin {qk['k_margin']:.4f}")
    print(f"    q statute {qk['q_statute']}   k statute {qk['k_statute']}")
    print(f"    self-listening angle {qk['self_listen_deg_mean']:.1f} ± "
          f"{qk['self_listen_deg_std']:.1f} deg (90 = independent roles)")

    print("\n[PR] Procrustes residual vs hosted recon-aleph codebook")
    pr = probe_procrustes(cb, pcfg)
    results["procrustes"] = pr
    if not pr.get("skipped"):
        print(f"    residual {pr['residual_deg']:.2f} deg   "
              f"null {pr['null_mean_deg']:.2f} ± {pr['null_std_deg']:.2f} deg   "
              f"z = {pr['z_vs_null']:+.2f}")
        print("    (z << 0: same vocabulary up to rotation; z ~ 0: independent)")

    print("\n[CV] raw-weight CV of address projections (CM band 0.13-0.30)")
    cv = probe_cv_band(model)
    results["cv"] = cv
    for k in ("q_addr", "k_addr"):
        print(f"    {k}: row_cv {cv[k]['row_cv']:.4f}  col_cv {cv[k]['col_cv']:.4f}")

    print("\n[TOPO] percolation + local intrinsic dim (final codebook)")
    topo = probe_topology(cb, pcfg.knn_k)
    results["topology"] = topo
    print(f"    percolation {topo['percolation_deg']} deg   "
          f"local dim {topo['local_dim_mean']:.2f}   "
          f"PR {topo['participation_ratio_mean']:.2f}")

    print("\n[TAU] the 0.857 anomaly — confidence vs tau")
    tau = probe_tau(model, cfg_lm, pcfg, stream)
    results["tau_sweep"] = tau
    print("    tau      trained  untrained  random")
    for t, row in tau.items():
        print(f"    {t:<8} {row['trained']:.4f}   {row['untrained']:.4f}    "
              f"{row['random']:.4f}")
    print("    (all three columns equal => kernel invariant of (tau,K,D), not learned)")

    print("\n[ER] streaming-state erank (codebook-memory occupancy)")
    er = probe_state_erank(model, cfg_lm, pcfg, stream)
    results["state_erank"] = er
    if "text" in er:
        print(f"    context {er['context_bytes']} bytes;  ceiling {er['text']['ceiling']}")
        print(f"    text : M+ {er['text']['erank_plus']:.1f}  M- "
              f"{er['text']['erank_minus']:.1f}  occupancy {er['text']['occupancy']:.0%}")
        print(f"    noise: M+ {er['noise']['erank_plus']:.1f}  M- "
              f"{er['noise']['erank_minus']:.1f}  occupancy {er['noise']['occupancy']:.0%}")

    print("\n[XB2] extraction-side voids (TYPE codebook, program's collapse rule)")
    try:
        xb = probe_extraction_voids(model, cfg_lm, pcfg, stream)
        results["extraction_voids"] = xb
        for name in ("text", "noise"):
            r = xb[name]
            b2 = {d: r["profile"][d]["beta2_per_axis"]
                  for d in pcfg.ripser_thresholds_deg}
            print(f"    {name:5s}: types {r['n_types']}  pairs {r['n_pairs']}  "
                  f"axes {r['n_axes']}  statute {r['statute']['statute']} "
                  f"(dev {r['statute']['deviation']:+.4f})")
            print("           b2/axis: " +
                  "  ".join(f"{d:.0f}deg={v:.3f}" for d, v in b2.items()))
    except Exception as e:
        print(f"    skipped: {e}")

    torch.save(results, pcfg.out_path)
    print(f"\nresults -> {pcfg.out_path}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aleph probe battery")
    ap.add_argument("--checkpoint", default="aleph_trigram_lm.pt")
    ap.add_argument("--snapshots", default="aleph_lm_codebook_snapshots.pt")
    ap.add_argument("--recon-version", default="aleph_byte_trigram_tied_hard_K64")
    ap.add_argument("--out", default="probe_results.pt")
    args, _unknown = ap.parse_known_args()
    run_battery(ProbeConfig(checkpoint=args.checkpoint, snapshots=args.snapshots,
                            recon_version=args.recon_version, out_path=args.out))