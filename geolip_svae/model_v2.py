"""
PatchSVAE v2 — Inverse Cascade Spectral Autoencoder
=====================================================
The decoder mirrors the forward decomposition IN REVERSE.
Each forward FL phase has a corresponding inverse decoder phase.
Conduit evidence enters at the phase where it was created.

Forward FLEigh phases:
  1. FL polynomial:    M → G → Mstore[k], char_coeffs c[]
  2. Laguerre roots:   c[] → eigenvalues λ, friction, settle, order
  3. Adjugate vectors: λ + Mstore → raw eigenvectors
  4. Newton-Schulz:    raw V → orthogonalized V, refinement_residual
  5. Rayleigh refine:  V, G → final (λ, V)

Inverse decoder phases (reverse order):
  5'. InvRayleigh:     Vt + refinement_residual → h₅
  4'. InvNewtonSchulz: h₅ + eigenvalues → h₄
  3'. InvAdjugate:     h₄ + char_coeffs → h₃
  2'. InvLaguerre:     h₃ + friction + settle + order → h₂
  1'. InvFL:           h₂ + U → patch pixels

Each inverse phase is a residual block that takes:
  - Previous phase's hidden state (the cascade)
  - Its corresponding conduit evidence (the key)

The decoder CANNOT shortcut because:
  - Eigenvalues enter at phase 4', not alongside U
  - Friction enters at phase 2', not alongside eigenvectors
  - U enters LAST at phase 1'
  - Each phase transforms the hidden state before the next evidence arrives
  - The model must learn the inverse of each theorem to pass information through
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip_core.linalg.eigh import FLEigh, _FL_MAX_N
from geolip_core.linalg.conduit import FLEighConduit, ConduitPacket


# ── SVD Backends (same as v1) ───────────────────────────────────

def gram_eigh_svd(A: torch.Tensor):
    B, M, N = A.shape
    orig_dtype = A.dtype
    if N <= _FL_MAX_N and A.is_cuda:
        with torch.amp.autocast('cuda', enabled=False):
            A_d = A.double()
            G = torch.bmm(A_d.transpose(1, 2), A_d)
            eigenvalues, V = FLEigh()(G.float())
            eigenvalues = eigenvalues.double().flip(-1)
            V = V.double().flip(-1)
            S = torch.sqrt(eigenvalues.clamp(min=1e-24))
            U = torch.bmm(A_d, V) / S.unsqueeze(1).clamp(min=1e-16)
            Vh = V.transpose(-2, -1).contiguous()
        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)
    with torch.amp.autocast('cuda', enabled=False):
        A_d = A.double()
        G = torch.bmm(A_d.transpose(1, 2), A_d)
        G.diagonal(dim1=-2, dim2=-1).add_(1e-12)
        eigenvalues, V = torch.linalg.eigh(G)
        eigenvalues = eigenvalues.flip(-1)
        V = V.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-24))
        U = torch.bmm(A_d, V) / S.unsqueeze(1).clamp(min=1e-16)
        Vh = V.transpose(-2, -1).contiguous()
    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)


def gram_eigh_svd_conduit(A: torch.Tensor, conduit_solver: FLEighConduit):
    B, M, N = A.shape
    orig_dtype = A.dtype
    with torch.amp.autocast('cuda', enabled=False):
        A_d = A.double()
        G = torch.bmm(A_d.transpose(1, 2), A_d)
        packet = conduit_solver(G.float())
        eigenvalues = packet.eigenvalues.double().flip(-1)
        V = packet.eigenvectors.double().flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-24))
        U = torch.bmm(A_d, V) / S.unsqueeze(1).clamp(min=1e-16)
        Vh = V.transpose(-2, -1).contiguous()
    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype), packet


# ── Patch Utilities ──────────────────────────────────────────────

def extract_patches(images, patch_size=16):
    B, C, H, W = images.shape
    gh, gw = H // patch_size, W // patch_size
    p = images.reshape(B, C, gh, patch_size, gw, patch_size)
    p = p.permute(0, 2, 4, 1, 3, 5)
    return p.reshape(B, gh * gw, C * patch_size * patch_size), gh, gw


def stitch_patches(patches, gh, gw, patch_size=16):
    B = patches.shape[0]
    p = patches.reshape(B, gh, gw, 3, patch_size, patch_size)
    return p.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, gh * patch_size, gw * patch_size)


class BoundarySmooth(nn.Module):
    def __init__(self, channels=3, mid=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1), nn.GELU(),
            nn.Conv2d(mid, channels, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        return x + self.net(x)


# ── Spectral Cross-Attention (unchanged) ────────────────────────

class SpectralCrossAttention(nn.Module):
    def __init__(self, D, n_heads=4, max_alpha=0.2, alpha_init=-2.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        self.max_alpha = max_alpha
        assert D % n_heads == 0
        self.qkv = nn.Linear(D, 3 * D)
        self.out_proj = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.scale = self.head_dim ** -0.5
        self.alpha_logits = nn.Parameter(torch.full((D,), alpha_init))

    @property
    def alpha(self):
        return self.max_alpha * torch.sigmoid(self.alpha_logits)

    def forward(self, S):
        B, N, D = S.shape
        S_n = self.norm(S)
        qkv = self.qkv(S_n).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        gate = torch.tanh(self.out_proj(out))
        alpha = self.alpha.unsqueeze(0).unsqueeze(0)
        return S * (1.0 + alpha * gate)


# ── Inverse Phase Block ─────────────────────────────────────────

class InversePhaseBlock(nn.Module):
    """Single phase of the inverse cascade.

    Takes hidden state from previous phase + conduit evidence
    from this phase's corresponding forward step.
    Produces hidden state for the next inverse phase.

    input_dim = hidden (from previous phase) + evidence_dim (conduit for this phase)
    output_dim = hidden (for next phase)
    """
    def __init__(self, hidden: int, evidence_dim: int):
        super().__init__()
        self.inject = nn.Linear(hidden + evidence_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        """
        h: (BN, hidden) — cascade state from previous phase
        evidence: (BN, evidence_dim) — conduit evidence for this phase
        """
        fused = F.gelu(self.inject(torch.cat([h, evidence], dim=-1)))
        fused = self.norm(fused)
        return fused + self.block(fused)


# ── Inverse Cascade Decoder ─────────────────────────────────────

class InverseCascadeDecoder(nn.Module):
    """Decoder structured as the inverse of the forward FL decomposition.

    Five inverse phases, each receiving conduit evidence from its
    corresponding forward phase:

      5' InvRayleigh:     Vt(D²) + refinement_residual(1) → h₅
      4' InvNewtonSchulz: h₅ + eigenvalues(D) → h₄
      3' InvAdjugate:     h₄ + char_coeffs(D) → h₃
      2' InvLaguerre:     h₃ + friction(D) + settle(D) + order(D) → h₂
      1' InvFL:           h₂ + U(V×D) → patch pixels

    The cascade forces sequential processing. The decoder cannot
    see U until phase 1' — it must first process the spectral
    structure through phases 5'-2' using only conduit evidence.
    """
    def __init__(self, V: int, D: int, hidden: int, patch_dim: int):
        super().__init__()
        self.V = V
        self.D = D
        self.hidden = hidden

        # Phase 5': InvRayleigh — starts the cascade from eigenvectors
        # Input: Vt flattened (D²) + refinement_residual (1) = D²+1
        self.phase5 = InversePhaseBlock(hidden, evidence_dim=D * D + 1)
        # Need an initial projection since there's no previous hidden state
        self.phase5_init = nn.Linear(D * D + 1, hidden)

        # Phase 4': InvNewtonSchulz — adds eigenvalue magnitudes
        # Evidence: eigenvalues (D)
        self.phase4 = InversePhaseBlock(hidden, evidence_dim=D)

        # Phase 3': InvAdjugate — adds polynomial structure
        # Evidence: char_coeffs (D)
        self.phase3 = InversePhaseBlock(hidden, evidence_dim=D)

        # Phase 2': InvLaguerre — adds dynamic conditioning evidence
        # Evidence: friction(D) + settle(D) + extraction_order(D) = 3D
        self.phase2 = InversePhaseBlock(hidden, evidence_dim=3 * D)

        # Phase 1': InvFL — adds spatial content, produces pixels
        # Evidence: U flattened (V×D)
        self.phase1 = InversePhaseBlock(hidden, evidence_dim=V * D)

        # Final projection to patch pixel space
        self.out = nn.Linear(hidden, patch_dim)

    def forward(self, U: torch.Tensor, S: torch.Tensor,
                Vt: torch.Tensor, packet: ConduitPacket) -> torch.Tensor:
        """Inverse cascade reconstruction.

        Args:
            U:  (BN, V, D) — left singular vectors
            S:  (BN, D)    — singular values
            Vt: (BN, D, D) — right singular vectors
            packet: ConduitPacket with all telemetry

        Returns:
            patches: (BN, patch_dim)
        """
        BN = U.shape[0]
        device = U.device

        # Detach all conduit evidence — no gradients through piecewise dynamics
        friction = torch.log1p(packet.friction.detach().to(device))  # log-compress
        settle = packet.settle.detach().to(device)
        char_coeffs = packet.char_coeffs.detach().to(device)
        ext_order = packet.extraction_order.detach().to(device)
        refine_res = packet.refinement_residual.detach().to(device)

        # ── Phase 5': InvRayleigh ──
        # Start cascade from eigenvector structure + refinement quality
        vt_flat = Vt.reshape(BN, -1)                          # (BN, D²)
        ev5 = torch.cat([vt_flat, refine_res.unsqueeze(-1)], dim=-1)  # (BN, D²+1)
        h = F.gelu(self.phase5_init(ev5))                     # (BN, hidden) — initial state
        h = self.phase5(h, ev5)                                # (BN, hidden)

        # ── Phase 4': InvNewtonSchulz ──
        # Add eigenvalue magnitudes — the decoder now knows directions + magnitudes
        h = self.phase4(h, S)                                  # (BN, hidden)

        # ── Phase 3': InvAdjugate ──
        # Add polynomial invariants — connects eigenvalues to polynomial structure
        h = self.phase3(h, char_coeffs)                        # (BN, hidden)

        # ── Phase 2': InvLaguerre ──
        # Add dynamic conditioning — how the polynomial was solved
        ev2 = torch.cat([friction, settle, ext_order], dim=-1)  # (BN, 3D)
        h = self.phase2(h, ev2)                                # (BN, hidden)

        # ── Phase 1': InvFL ──
        # Add spatial content — the original M's left singular vectors
        u_flat = U.reshape(BN, -1)                             # (BN, V*D)
        h = self.phase1(h, u_flat)                             # (BN, hidden)

        # ── Reconstruct patch ──
        return self.out(h)


# ── PatchSVAE v2 ────────────────────────────────────────────────

class PatchSVAEv2(nn.Module):
    """Inverse Cascade Spectral Autoencoder.

    Encoder identical to v1. Decoder is an inverse cascade that
    mirrors the forward FL decomposition in reverse, with conduit
    evidence entering at each corresponding inverse phase.

    Args:
        V: rows per encoded matrix (default 48)
        D: spectral dimensions (default 4)
        ps: patch size (default 4)
        hidden: MLP hidden dimension (default 384)
        depth: encoder residual blocks (default 4)
        n_cross: spectral cross-attention layers (default 2)
        n_heads: attention heads (default: auto from D)
        smooth_mid: boundary smooth channels (default: auto from ps)
    """
    def __init__(self, V: int = 48, D: int = 4, ps: int = 4,
                 hidden: int = 384, depth: int = 4, n_cross: int = 2,
                 n_heads: int = None, smooth_mid: int = None):
        super().__init__()
        self.matrix_v = V
        self.D = D
        self.patch_size = ps
        self.patch_dim = 3 * ps * ps
        self.mat_dim = V * D

        if n_heads is None:
            n_heads = 2 if D <= 8 else min(4, D)
        if smooth_mid is None:
            smooth_mid = 16 if ps >= 16 else 8

        # ── Encoder (identical to v1) ──
        self.enc_in = nn.Linear(self.patch_dim, hidden)
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.enc_out = nn.Linear(hidden, self.mat_dim)
        nn.init.orthogonal_(self.enc_out.weight)

        # ── Conduit Solver (always active) ──
        self._conduit_solver = FLEighConduit()
        self.last_conduit_packet = None

        # ── Spectral Cross-Attention (unchanged) ──
        self.cross_attn = nn.ModuleList([
            SpectralCrossAttention(D, n_heads=n_heads)
            for _ in range(n_cross)
        ])

        # ── Inverse Cascade Decoder ──
        self.decoder = InverseCascadeDecoder(
            V=V, D=D, hidden=hidden, patch_dim=self.patch_dim)

        # ── Boundary Smoothing ──
        self.boundary_smooth = BoundarySmooth(channels=3, mid=smooth_mid)

    def _svd(self, A: torch.Tensor):
        solver = self._conduit_solver.to(A.device)
        U, S, Vh, packet = gram_eigh_svd_conduit(A, solver)
        self.last_conduit_packet = packet
        return U, S, Vh, packet

    def encode_patches(self, patches: torch.Tensor) -> dict:
        B, N, _ = patches.shape
        flat = patches.reshape(B * N, -1)

        h = F.gelu(self.enc_in(flat))
        for block in self.enc_blocks:
            h = h + block(h)

        M = self.enc_out(h).reshape(B * N, self.matrix_v, self.D)
        M = F.normalize(M, dim=-1)

        U, S, Vt, packet = self._svd(M)

        U = U.reshape(B, N, self.matrix_v, self.D)
        S = S.reshape(B, N, self.D)
        Vt = Vt.reshape(B, N, self.D, self.D)
        M = M.reshape(B, N, self.matrix_v, self.D)

        S_coordinated = S
        for layer in self.cross_attn:
            S_coordinated = layer(S_coordinated)

        return {
            'U': U, 'S_orig': S, 'S': S_coordinated,
            'Vt': Vt, 'M': M, 'conduit_packet': packet,
        }

    def decode_patches(self, U, S, Vt, packet):
        B, N, V, D = U.shape
        U_flat = U.reshape(B * N, V, D)
        S_flat = S.reshape(B * N, D)
        Vt_flat = Vt.reshape(B * N, D, D)

        decoded = self.decoder(U_flat, S_flat, Vt_flat, packet)
        return decoded.reshape(B, N, -1)

    def forward(self, images: torch.Tensor) -> dict:
        B, C, H, W = images.shape
        ps = self.patch_size
        patches, gh, gw = extract_patches(images, ps)

        svd = self.encode_patches(patches)

        decoded = self.decode_patches(
            svd['U'], svd['S'], svd['Vt'], svd['conduit_packet'])
        recon = stitch_patches(decoded, gh, gw, ps)
        recon = self.boundary_smooth(recon)

        return {'recon': recon, 'svd': svd}

    @staticmethod
    def from_v1(v1_model):
        """Initialize v2 from trained v1. Encoder copied, decoder is new."""
        v2 = PatchSVAEv2(
            V=v1_model.matrix_v, D=v1_model.D, ps=v1_model.patch_size,
            hidden=v1_model.enc_in.in_features,
            depth=len(v1_model.enc_blocks),
            n_cross=len(v1_model.cross_attn),
        )
        v2.enc_in.load_state_dict(v1_model.enc_in.state_dict())
        for v2b, v1b in zip(v2.enc_blocks, v1_model.enc_blocks):
            v2b.load_state_dict(v1b.state_dict())
        v2.enc_out.load_state_dict(v1_model.enc_out.state_dict())
        for v2c, v1c in zip(v2.cross_attn, v1_model.cross_attn):
            v2c.load_state_dict(v1c.state_dict())
        v2.boundary_smooth.load_state_dict(v1_model.boundary_smooth.state_dict())

        n_v1 = sum(p.numel() for p in v1_model.parameters())
        n_v2 = sum(p.numel() for p in v2.parameters())
        print(f"  PatchSVAEv2 from v1: encoder copied, decoder NEW")
        print(f"    v1: {n_v1:,}  v2: {n_v2:,}  delta: {n_v2-n_v1:+,}")
        return v2

    @staticmethod
    def effective_rank(S):
        p = S / (S.sum(-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        return (-(p * p.log()).sum(-1)).exp()

    @staticmethod
    def s_delta(S_orig, S_coord):
        return (S_coord - S_orig).abs().mean().item()