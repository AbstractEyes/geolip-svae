"""
procrustes_frame.py — v2: gauge-correct commensuration of the three devices.

CRITICAL NOTE:
This is experimental and should not be taken into account for when
assessing the system as a whole.
============================================================================
PROTOTYPE CORE (self-contained Colab cell). SVAE + Aleph-Void ONLY — no Bertenstein
conduit, no GeometricHubLoss, no imported CV band. Those were different models; this is
rebuilt from the SVAE/Aleph-Void and the geofractal frame.

GOVERNING FRAME (geofractal): DIVERGENCE OVER ACCURACY. Alignment makes divergent devices
COMMENSURABLE so a fusion can triangulate them — it does NOT homogenize them into one space.
So we do NOT rotate the aleph into CLIP-G's 1280-d. Each device is addressed through its
OWN geometry; we then measure/relate them with the matching primitive.

DEVICES:
  G     : CLIP-G pooled (1280)            — Euclidean.
  QWEN  : Qwen rich-pooled (~1024)        — Euclidean.
  ALEPH : addressed M̂ (n_addr, V*D=128)  — SPHERICAL: V=32 rows on S^(D-1)=S^3, D=4.
          Signed-projective (double cover S^3 -> RP^3; +a and -a are the SAME axis, the
          sign is gauge). So the aleph's feature MUST be gauge-invariant.

THE ALEPH FIX (this is what v1 got wrong): v1 mean-pooled the 128-flat, padded to 1280, and
rotated it as a Euclidean vector — gauge-dependent (the ± freedom moves the mean), cancels
antipodes, and mixes axis identity. The correct, gauge-INVARIANT Euclidean reduction of an
axial/projective sphere code is the SECOND-MOMENT / orientation tensor  T = (1/V) Σ vᵢvᵢᵀ,
because (-v)(-v)ᵀ = vvᵀ kills the sign. T is D×D (=4×4); its upper-triangle (+ optionally
eigenvalues) is the device's gauge-correct Euclidean signature. (Same Gram/second-moment
idiom geo_zephyr uses for spherical structure.)

COMMENSURATION (the "align the devices" step), per-geometry, divergence-preserving:
  - Euclidean<->Euclidean (G,QWEN) and Euclidean<->aleph-signature: whitened Procrustes
    (FrozenAligner, fit once -> stored buffers -> apply), judged by LIFT = cos_after − a
    shuffled-caption floor (the probe lesson: absolute cos_after misleads — the whitened-
    rotation noise floor scales with dim/samples; only the lift over a shuffled control means
    anything). The maps commensurate; they don't collapse the full per-device representations.
  - Native D=4 structural metric: procrustes_distance(A,B) — VERBATIM from the Aleph-Void's
    own tests/framework.py (projective-abs, SVD(AᵀB)->R), for codebook/M-row comparison in D=4.

STRUCTURE: pentachoron CV (a pentachoron = 5 points in D=4 — the band is a D=4 quantity)
computed in D=4 over the addressed M-rows (reshape (32,128)->(32, V=32, D=4)) — NOT over the
128-flat. The CV penalty rides a LEARNED lift (a penalty on the frozen address has no grad).

Adam-only; loss fns return tensors (no multi-key dicts from compiled paths).
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# primitives (self-contained; match geolip-core / geovocab2 math)
# ---------------------------------------------------------------------------
def newton_schulz_invsqrt(G: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """G^{-1/2} via Newton-Schulz (G symmetric PSD). Matches geolip_core.linalg.newton_schulz."""
    d = G.shape[-1]
    I = torch.eye(d, device=G.device, dtype=G.dtype).expand_as(G)
    G = G + 1e-6 * I
    trace = G.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
    norm = trace.view(*trace.shape, 1, 1)
    Y = G / norm
    Z = I.clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * I - torch.matmul(Z, Y))
        Y = torch.matmul(Y, T)
        Z = torch.matmul(T, Z)
    return Z / norm.sqrt()


def harmonize_dim(x: torch.Tensor, hub: int) -> torch.Tensor:
    """Last dim -> hub by zero-pad-up / truncate-down. Legitimate here because it is applied to
    PROPER Euclidean features (pooled vectors, gauge-invariant signatures), never to raw sphere coords."""
    d = x.shape[-1]
    if d == hub:
        return x
    if d < hub:
        pad = list(x.shape); pad[-1] = hub - d
        return torch.cat([x, x.new_zeros(pad)], dim=-1)
    return x[..., :hub]


def procrustes_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    """VERBATIM from geolip_svae/tests/framework.py — the Aleph-Void's own codebook Procrustes.
    A,B: [n_axes, D] (same D). Min Frobenius distance after optimal rotation; aligns on the
    smaller n_axes. (Projective: callers may abs the cross-corr; kept faithful to source here.)"""
    A = A.detach().cpu().to(torch.float64)
    B = B.detach().cpu().to(torch.float64)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"D mismatch: A.D={A.shape[1]}, B.D={B.shape[1]}")
    n = min(A.shape[0], B.shape[0])
    if n < 2:
        return float('nan')
    A_n, B_n = A[:n], B[:n]
    M = A_n.T @ B_n
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    aligned = B_n @ R.T
    return float((A_n - aligned).norm()) / max(n ** 0.5, 1.0)


# ---------------------------------------------------------------------------
# ALEPH: reshape to S^3 rows -> gauge-invariant signature  (the v1 fix)
# ---------------------------------------------------------------------------
def aleph_rows(addr: torch.Tensor, V: int = 32, D: int = 4) -> torch.Tensor:
    """(B, n_addr, V*D) addressed M̂ -> (B, n_addr, V, D): V rows on S^(D-1) per patch-slot."""
    B, n_addr, vd = addr.shape
    assert vd == V * D, f"addr last dim {vd} != V*D={V*D}"
    return addr.float().reshape(B, n_addr, V, D)


def orientation_tensor(rows: torch.Tensor) -> torch.Tensor:
    """Gauge-invariant second moment of axial rows. rows (..., n, D) -> T (..., D, D),
    T = (1/n) Σ vᵢvᵢᵀ.  Sign-invariant: (-v)(-v)ᵀ = vvᵀ, so it ignores the double-cover gauge.
    Computed in fp64 (the matmul is tiny, D×D) so the invariance is bit-reproducible on GPU:
    float32 TF32 matmul is non-deterministic and makes two sign-flip-identical inputs disagree
    at ~1e-2 (a measurement artifact, not a real break — fp64/CPU gives ~1e-7)."""
    n = rows.shape[-2]
    r = rows.double()
    return (torch.matmul(r.transpose(-1, -2), r) / max(n, 1)).to(rows.dtype)


def _upper_tri(T: torch.Tensor) -> torch.Tensor:
    """Unique entries of a symmetric (..., D, D) -> (..., D(D+1)/2)."""
    D = T.shape[-1]
    iu = torch.triu_indices(D, D, device=T.device)
    return T[..., iu[0], iu[1]]


def aleph_signature(addr: torch.Tensor, V: int = 32, D: int = 4,
                    per_patch: bool = True, with_eigs: bool = False) -> torch.Tensor:
    """Gauge-correct Euclidean signature of the aleph device (the SphericalAddressComponent
    reduction). per_patch=True -> per-slot orientation tensors (B, n_addr*D(D+1)/2), better
    conditioned for commensuration; False -> one global tensor (B, D(D+1)/2). with_eigs appends
    eigenvalues (orientation spread, also gauge-free)."""
    rows = aleph_rows(addr, V, D)                          # (B, n_addr, V, D)
    if per_patch:
        T = orientation_tensor(rows)                       # (B, n_addr, D, D)
        feat = _upper_tri(T).reshape(rows.shape[0], -1)    # (B, n_addr*10)
        if with_eigs:
            ev = torch.linalg.eigvalsh(T).reshape(rows.shape[0], -1)
            feat = torch.cat([feat, ev], dim=-1)
    else:
        flat = rows.reshape(rows.shape[0], -1, D)          # (B, n_addr*V, D)
        T = orientation_tensor(flat)                       # (B, D, D)
        feat = _upper_tri(T)                               # (B, 10)
        if with_eigs:
            feat = torch.cat([feat, torch.linalg.eigvalsh(T)], dim=-1)
    return feat


# ---------------------------------------------------------------------------
# STRUCTURE: pentachoron CV in D=4 (over the M-rows / codebook), not the 128-flat
# ---------------------------------------------------------------------------
def _pdist2(P: torch.Tensor) -> torch.Tensor:
    g = torch.matmul(P, P.transpose(-1, -2))
    sq = g.diagonal(dim1=-2, dim2=-1)
    return (sq.unsqueeze(-1) + sq.unsqueeze(-2) - 2.0 * g).clamp_min(0.0)


def cayley_menger_vol2(P: torch.Tensor) -> torch.Tensor:
    """Squared volume of the (k-1)-simplex on k points P (..., k, D), Cayley-Menger det.
    Pentachoron = 5 points (4-simplex) — use in D=4."""
    with torch.amp.autocast('cuda', enabled=False):
        P = P.float(); k = P.shape[-2]; D2 = _pdist2(P); n = k + 1
        cm = torch.ones(*D2.shape[:-2], n, n, device=P.device, dtype=P.dtype)
        cm[..., 0, 0] = 0.0; cm[..., 1:, 1:] = D2
        det = torch.linalg.det(cm)
        import math
        coeff = ((-1.0) ** k) / (float(2 ** (k - 1)) * (math.factorial(k - 1) ** 2))
        return (coeff * det).clamp_min(0.0)


def _sample_quintuples(n_rows: int, n_samples: int, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.stack([torch.randperm(n_rows, generator=g)[:5] for _ in range(n_samples)]).to(device)


def pentachoron_cv(points: torch.Tensor, n_samples: int = 200, seed: int = 0) -> torch.Tensor:
    """CV (std/mean) of pentachoron volumes over sampled 5-subsets of `points` (..., n, D), D=4.
    `points` are M-rows on S^3 (or codebook axes). Differentiable. Returns mean CV over batch."""
    n_rows = points.shape[-2]
    idx = _sample_quintuples(n_rows, n_samples, points.device, seed)        # (n_samples, 5)
    vol = cayley_menger_vol2(points[..., idx, :]).clamp_min(1e-20).sqrt()   # (..., n_samples)
    return (vol.std(dim=-1) / vol.mean(dim=-1).clamp_min(1e-12)).mean()


@torch.no_grad()
def measured_cv(points: torch.Tensor, n_samples: int = 200, seed: int = 0) -> float:
    return float(pentachoron_cv(points, n_samples=n_samples, seed=seed).item())


def pentachoron_cv_penalty(points: torch.Tensor, target_cv: float,
                           n_samples: int = 200, seed: int = 0) -> torch.Tensor:
    """(cv − target_cv)² on a LEARNED constellation in D=4 (e.g. an adapter's lift of the
    M-rows). target_cv := measured_cv(frozen aleph rows). No grad on the frozen address itself."""
    return (pentachoron_cv(points, n_samples=n_samples, seed=seed) - target_cv) ** 2


# ---------------------------------------------------------------------------
# Euclidean commensuration: whitened Procrustes map (fit once -> buffers -> apply)
# ---------------------------------------------------------------------------
class FrozenAligner(nn.Module):
    """Directional whitened-Procrustes map source->target, fit once, applied to new batches:
    center -> whiten(src) -> rotate -> unwhiten(tgt) -> + tgt_mean. (geolip-core batched_procrustes
    whitened math, restructured for stored apply.) Used ONLY on proper Euclidean features."""

    def __init__(self, dim: int, whiten: bool = True, schulz_iters: int = 10):
        super().__init__()
        self.dim = dim; self.whiten = whiten; self.schulz_iters = schulz_iters
        z = torch.zeros(dim); I = torch.eye(dim)
        for name, init in [("src_mean", z), ("tgt_mean", z), ("R", I), ("src_W", I), ("tgt_W_inv", I)]:
            self.register_buffer(name, init.clone())
        self.register_buffer("fitted", torch.zeros(1))

    @torch.no_grad()
    def fit(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        src = src.float(); tgt = tgt.float(); N = src.shape[0]
        sm = src.mean(0, keepdim=True); tm = tgt.mean(0, keepdim=True)
        sc = src - sm; tc = tgt - tm
        if self.whiten:
            sW = newton_schulz_invsqrt((sc.T @ sc) / max(N - 1, 1), self.schulz_iters)
            tW = newton_schulz_invsqrt((tc.T @ tc) / max(N - 1, 1), self.schulz_iters)
            tW_inv = torch.linalg.pinv(tW)
            s_w = F.normalize(sc @ sW, dim=-1); t_w = F.normalize(tc @ tW, dim=-1)
        else:
            sW = torch.eye(self.dim, device=src.device); tW_inv = sW.clone()
            s_w, t_w = sc, tc
        U, _, Vh = torch.linalg.svd(s_w.T @ t_w)
        R = U @ Vh
        for buf, val in [(self.src_mean, sm.squeeze(0)), (self.tgt_mean, tm.squeeze(0)),
                         (self.src_W, sW), (self.tgt_W_inv, tW_inv), (self.R, R)]:
            buf.copy_(val)
        self.fitted.fill_(1.0)
        return float(F.cosine_similarity(s_w @ R, t_w, dim=-1).mean().item())

    def apply_map(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.float() - self.src_mean
        xw = F.normalize(xc @ self.src_W, dim=-1) if self.whiten else xc
        return (xw @ self.R) @ self.tgt_W_inv + self.tgt_mean

    forward = apply_map


def _pca_reduce(fit_X: torch.Tensor, all_X: torch.Tensor, k: int) -> torch.Tensor:
    """Project all_X onto the top-k principal directions of fit_X (centered). The probe's
    PCA-DOWN: concentrates a device's variance into k dims so a small shared signal isn't buried
    under zero-padding when commensurating devices of different dimensionality. PCA fit on the
    TRAIN rows only (no eval leakage)."""
    mu = fit_X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(fit_X - mu, full_matrices=False)
    P = Vh[:k].transpose(0, 1)                              # (D, k)
    return (all_X - mu) @ P


@torch.no_grad()
def commensuration_lift(src: torch.Tensor, tgt: torch.Tensor, whiten: bool = True,
                        n_shuffle: int = 3, eval_frac: float = 0.5, seed: int = 0) -> dict:
    """HELD-OUT commensurability of two device features. Reduce the larger device to min-dim by
    PCA (fit on train), fit the whitened-Procrustes map on a TRAIN split (true pairing), then
    score cos_after on a DISJOINT EVAL split; floor = same with the train pairing shuffled.
    LIFT = cos_after − floor on held-out data — >0 only if real alignable structure GENERALIZES.
    This is the fix for the fit-set metric, which saturated when N≲dim (an over-flexible rotation
    aligns ANY N≲dim points, so real and shuffled both hit ceiling and lift collapses to 0).
    NOTE: lift magnitude is the alignable FRACTION of variance (honestly small when the shared
    structure is a small part of a high-dim device); the SIGN-vs-floor is the discriminator."""
    src = src.float(); tgt = tgt.float()
    N = src.shape[0]
    k = min(src.shape[-1], tgt.shape[-1])
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    n_ev = max(2, int(N * eval_frac)); ev, tr = idx[:n_ev], idx[n_ev:]
    s = _pca_reduce(src[tr], src, k) if src.shape[-1] > k else src
    t = _pca_reduce(tgt[tr], tgt, k) if tgt.shape[-1] > k else tgt

    def _heldout(src_tr: torch.Tensor, tgt_tr: torch.Tensor) -> float:
        aln = FrozenAligner(k, whiten=whiten); aln.fit(src_tr, tgt_tr)
        return float(F.cosine_similarity(aln.apply_map(s[ev]), t[ev], dim=-1).mean().item())

    cos_after = _heldout(s[tr], t[tr])
    floors = [_heldout(s[tr][torch.randperm(tr.shape[0],
                       generator=torch.Generator().manual_seed(seed + 1 + j))], t[tr])
              for j in range(max(1, n_shuffle))]
    floor = float(sum(floors) / len(floors))
    return {"cos_after": cos_after, "floor": floor, "lift": cos_after - floor}


class DeviceFrame(nn.Module):
    """Commensurate the three devices (divergence-preserving). G is the natural Euclidean
    reference (SDXL's conditioning space); QWEN and the gauge-correct ALEPH signature are
    commensurated to it AND to each other. The maps are frozen (fit Stage-1); the full per-device
    representations are untouched. Use the lifts to judge whether a device is worth fusing."""

    def __init__(self, g_dim: int = 1280, qwen_dim: int = 1024,
                 aleph_V: int = 32, aleph_D: int = 4, aleph_per_patch: bool = True,
                 whiten: bool = True):
        super().__init__()
        self.g_dim, self.qwen_dim = g_dim, qwen_dim
        self.aleph_V, self.aleph_D, self.aleph_per_patch = aleph_V, aleph_D, aleph_per_patch
        # aleph signature dim: n_addr * D(D+1)/2 (per-patch) — set lazily on first fit (needs n_addr)
        self.whiten = whiten
        self.map_q = None      # QWEN -> G        (built lazily once dims are known)
        self.map_a = None      # ALEPHsig -> G
        self.hub = g_dim

    def aleph_feat(self, addr: torch.Tensor) -> torch.Tensor:
        return aleph_signature(addr, self.aleph_V, self.aleph_D, self.aleph_per_patch)

    @torch.no_grad()
    def fit(self, g: torch.Tensor, qwen: torch.Tensor, addr: torch.Tensor) -> dict:
        """Stage-1. g:(N,g_dim) CLIP-G pooled; qwen:(N,qwen_dim); addr:(N,n_addr,V*D).
        Builds frozen QWEN->G and ALEPHsig->G maps; returns commensuration lifts for all 3 pairs."""
        a = self.aleph_feat(addr)
        self.hub = max(self.g_dim, self.qwen_dim, a.shape[-1])
        self.map_q = FrozenAligner(self.hub, self.whiten)
        self.map_a = FrozenAligner(self.hub, self.whiten)
        self.map_q.fit(harmonize_dim(qwen, self.hub), harmonize_dim(g, self.hub))
        self.map_a.fit(harmonize_dim(a, self.hub),    harmonize_dim(g, self.hub))
        return {
            "G<->QWEN":  commensuration_lift(qwen, g, self.whiten),
            "G<->ALEPH": commensuration_lift(a,    g, self.whiten),
            "QWEN<->ALEPH": commensuration_lift(a, qwen, self.whiten),
            "aleph_sig_dim": int(a.shape[-1]), "hub": int(self.hub),
        }

    def align(self, g: torch.Tensor, qwen: torch.Tensor, addr: torch.Tensor):
        """Per-batch -> (g, q_in_G, a_in_G) all (B, hub). G kept native; the others commensurated."""
        assert self.map_q is not None, "call fit() (Stage-1) first"
        a = self.aleph_feat(addr)
        return (harmonize_dim(g, self.hub),
                self.map_q(harmonize_dim(qwen, self.hub)),
                self.map_a(harmonize_dim(a, self.hub)))


# ---------------------------------------------------------------------------
# self-test (python procrustes_frame.py): gauge-invariance, commensuration, D=4 CV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    N, n_addr, V, D = 4000, 32, 32, 4         # N >> dims so the held-out metric is well-posed
    GD, QD, SH = 512, 384, 192                # shared-latent dim SH (a clear fraction, for the demo)

    # synth devices: G/QWEN share a latent (Euclidean, rotated). aleph rows biased toward a
    # Z-dependent axis (so its orientation tensor carries real structure); a NULL aleph is random.
    Z = torch.randn(N, SH)
    Rq, _ = torch.linalg.qr(torch.randn(SH, SH))
    G = torch.randn(N, GD);  G[:, :SH] = Z + 0.05 * torch.randn(N, SH)
    QWEN = torch.randn(N, QD); QWEN[:, :SH] = Z @ Rq + 0.05 * torch.randn(N, SH)
    Wz = torch.randn(SH, D)                                            # Z -> a D=4 axis
    axis = F.normalize(Z @ Wz, dim=-1)[:, None, None, :]              # (N,1,1,D) per-sample axis
    rows = F.normalize(axis + 0.6 * torch.randn(N, n_addr, V, D), dim=-1)
    sign = torch.sign(torch.randn(N, n_addr, V, 1))
    addr = (sign * rows).reshape(N, n_addr, V * D)                    # signed -> the (32,128) address
    addr_null = (torch.sign(torch.randn(N, n_addr, V, 1)) *
                 F.normalize(torch.randn(N, n_addr, V, D), dim=-1)).reshape(N, n_addr, V * D)

    print("== aleph signature is gauge-invariant (flip the ± sign convention on same rows) ==")
    sig = aleph_signature(addr, V, D)
    g2 = torch.sign(torch.randn(N, n_addr, V, 1))
    sig_flip = aleph_signature((g2 * (sign * rows)).reshape(N, n_addr, V * D), V, D)
    print(f"   ||sig − sig_gaugeflip|| = {(sig - sig_flip).abs().max().item():.3e}  (≈0 => gauge-free)  dim={sig.shape[-1]}")

    print("\n== HELD-OUT commensuration lifts (cos_after − shuffled floor on a disjoint split) ==")
    print("   sign>0 vs floor = real alignable structure; magnitude = alignable variance fraction")
    a_struct = aleph_signature(addr, V, D)
    a_null = aleph_signature(addr_null, V, D)
    for name, d in [
        ("G<->QWEN  (shared)", commensuration_lift(QWEN, G)),
        ("G<->ALEPH (struct)", commensuration_lift(a_struct, G)),
        ("G<->ALEPH (null)  ", commensuration_lift(a_null, G)),
    ]:
        print(f"   {name}  cos_after={d['cos_after']:+.3f}  floor={d['floor']:+.3f}  LIFT={d['lift']:+.3f}")

    print("\n== pentachoron CV in D=4 over the M-rows (the correct space) ==")
    pr = rows[:8, 0]                                                 # (8, V=32, 4) one patch's rows
    tcv = measured_cv(pr)
    print(f"   measured target_cv={tcv:.4f}  penalty@target={pentachoron_cv_penalty(pr, tcv).item():.3e}")
    lyr = nn.Linear(D, D); pentachoron_cv_penalty(lyr(pr), tcv).backward()
    print(f"   backward OK; lift grad norm={lyr.weight.grad.norm().item():.3e}")

    print("\n== native D=4 structural metric (framework.py procrustes_distance, verbatim) ==")
    cbA = F.normalize(torch.randn(64, D), dim=-1); cbB = cbA + 0.1 * torch.randn(64, D)
    print(f"   procrustes_distance(A, A+noise) = {procrustes_distance(cbA, cbB):.4f}  "
          f"(self={procrustes_distance(cbA, cbA):.4f})")
    print("\nself-test done ✓")