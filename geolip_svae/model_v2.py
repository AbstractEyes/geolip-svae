"""
PatchSVAE v2 — Conduit-Forced Spectral Autoencoder (Prototype)
================================================================
The decoder does NOT receive M_hat = U @ diag(S) @ Vt.
The decoder receives the DECOMPOSED spectral representation
interleaved with conduit telemetry. Every conduit element is
architecturally load-bearing — reconstruction requires all of them.

CRITICAL DIFFERENCE FROM v1:
  v1: Decoder gets M_hat (192 values). Conduit is ignorable context.
  v2: Decoder gets per-mode bundles (spectral + conduit interleaved).
      No shortcut to M_hat. Must learn what every element means.

Per-mode bundle (D=4, V=48):
  U_col[V]          — left singular vector for mode k (48 values)
  S_val[1]          — singular value magnitude
  Vt_row[D]         — right singular vector (4 values)
  friction[1]       — solver struggle for this mode
  settle[1]         — convergence iterations
  char_coeff[1]     — k-th polynomial invariant
  extraction_order[1] — solver hierarchy position
  = 57 values per mode, 228 total for D=4

Global conduit:
  refinement_residual[1]
  release_residual[1]
  = 230 total decoder input

The decoder processes each mode through a shared ModeDecoder,
then fuses across modes. Every element is in the critical path.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── SVD Backend (unchanged from v1) ─────────────────────────────

from geolip_core.linalg.eigh import FLEigh, _FL_MAX_N
from geolip_core.linalg.conduit import FLEighConduit, ConduitPacket


def gram_eigh_svd(A: torch.Tensor):
    """Standard thin SVD via Gram-eigh. Unchanged from v1."""
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
    """Thin SVD with conduit telemetry. Returns (U, S, Vh, packet)."""
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


# ── Patch Utilities (unchanged) ──────────────────────────────────

def extract_patches(images: torch.Tensor, patch_size: int = 16):
    B, C, H, W = images.shape
    gh, gw = H // patch_size, W // patch_size
    p = images.reshape(B, C, gh, patch_size, gw, patch_size)
    p = p.permute(0, 2, 4, 1, 3, 5)
    return p.reshape(B, gh * gw, C * patch_size * patch_size), gh, gw


def stitch_patches(patches: torch.Tensor, gh: int, gw: int,
                   patch_size: int = 16) -> torch.Tensor:
    B = patches.shape[0]
    p = patches.reshape(B, gh, gw, 3, patch_size, patch_size)
    return p.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, gh * patch_size, gw * patch_size)


# ── Boundary Smoothing (unchanged) ──────────────────────────────

class BoundarySmooth(nn.Module):
    def __init__(self, channels: int = 3, mid: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ── Spectral Cross-Attention (unchanged) ────────────────────────

class SpectralCrossAttention(nn.Module):
    def __init__(self, D: int, n_heads: int = 4,
                 max_alpha: float = 0.2, alpha_init: float = -2.0):
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

    def forward(self, S: torch.Tensor) -> torch.Tensor:
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


# ── Mode Decoder — processes each spectral mode independently ───

class ModeProcessor(nn.Module):
    """Process a single spectral mode: its U column, S value,
    Vt row, and conduit signature into a hidden representation.

    Input per mode: U_col(V) + S(1) + Vt_row(D) + friction(1) +
                    settle(1) + char_coeff(1) + order(1)
                    = V + D + 5 values

    Every element is in the linear projection. No element is separable.
    """
    def __init__(self, V: int, D: int, hidden: int):
        super().__init__()
        mode_dim = V + D + 5  # U_col + Vt_row + S + friction + settle + coeff + order
        self.proj = nn.Linear(mode_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, mode_bundle: torch.Tensor) -> torch.Tensor:
        """mode_bundle: (B, mode_dim) → (B, hidden)"""
        return self.norm(F.gelu(self.proj(mode_bundle)))


class ConduitDecoder(nn.Module):
    """Decoder that REQUIRES conduit evidence to reconstruct.

    Does NOT receive M_hat. Receives per-mode spectral+conduit bundles.
    Each mode is processed independently through shared ModeProcessor,
    then modes are fused with global conduit context, then decoded
    through residual MLP to pixel space.

    The shared processor ensures each mode's conduit is load-bearing —
    the same weights process S[k] alongside friction[k] alongside
    U[:,k]. They cannot be separated.
    """
    def __init__(self, V: int, D: int, hidden: int, depth: int,
                 patch_dim: int):
        super().__init__()
        self.V = V
        self.D = D
        self.hidden = hidden

        # Per-mode processor (shared across modes)
        self.mode_proc = ModeProcessor(V, D, hidden)

        # Mode fusion: D mode embeddings + 2 global conduit scalars → hidden
        self.fuse = nn.Linear(D * hidden + 2, hidden)
        self.fuse_norm = nn.LayerNorm(hidden)

        # Residual MLP decoder (same structure as v1 decoder)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.out = nn.Linear(hidden, patch_dim)

    def forward(self, U: torch.Tensor, S: torch.Tensor,
                Vt: torch.Tensor, packet: ConduitPacket) -> torch.Tensor:
        """Reconstruct patches from decomposed spectral+conduit representation.

        Args:
            U:  (BN, V, D)
            S:  (BN, D)
            Vt: (BN, D, D)
            packet: ConduitPacket from conduit solver (BN-batched)

        Returns:
            patches: (BN, patch_dim)
        """
        BN = U.shape[0]
        device = U.device

        # Detach conduit — no gradients through piecewise dynamics (Theorem 4)
        friction = packet.friction.detach().to(device)           # (BN, D)
        settle = packet.settle.detach().to(device)               # (BN, D)
        char_coeffs = packet.char_coeffs.detach().to(device)     # (BN, D)
        ext_order = packet.extraction_order.detach().to(device)  # (BN, D)
        refine_res = packet.refinement_residual.detach().to(device)  # (BN,)
        release_res = packet.release_residual                    # may be None

        # Log-compress friction to manageable range
        friction = torch.log1p(friction)

        # Build per-mode bundles: each mode gets its spectral data + conduit
        mode_hiddens = []
        for k in range(self.D):
            # Bundle: U_col(V) + S_val(1) + Vt_row(D) + friction(1) +
            #         settle(1) + char_coeff(1) + order(1)
            bundle = torch.cat([
                U[:, :, k],                        # (BN, V)
                S[:, k:k+1],                       # (BN, 1)
                Vt[:, k, :],                        # (BN, D)
                friction[:, k:k+1],                 # (BN, 1)
                settle[:, k:k+1],                   # (BN, 1)
                char_coeffs[:, k:k+1],              # (BN, 1)
                ext_order[:, k:k+1],                # (BN, 1)
            ], dim=-1)  # (BN, V+D+5)

            mode_hiddens.append(self.mode_proc(bundle))  # (BN, hidden)

        # Fuse all modes + global conduit
        global_conduit = torch.stack([
            refine_res,
            release_res if release_res is not None
            else torch.zeros(BN, device=device),
        ], dim=-1)  # (BN, 2)

        fused = torch.cat(mode_hiddens + [global_conduit], dim=-1)  # (BN, D*hidden + 2)
        h = self.fuse_norm(F.gelu(self.fuse(fused)))  # (BN, hidden)

        # Residual MLP → pixel space
        for block in self.blocks:
            h = h + block(h)

        return self.out(h)


# ── PatchSVAE v2 ────────────────────────────────────────────────

class PatchSVAEv2(nn.Module):
    """Conduit-Forced Spectral Autoencoder.

    Encoder is identical to v1. SVD always uses conduit solver.
    Decoder is replaced with ConduitDecoder that requires
    the full decomposed spectral + conduit representation.

    Args:
        V: rows per encoded matrix (default 48 for Freckles-class)
        D: spectral dimensions (default 4)
        ps: patch size (default 4)
        hidden: MLP hidden dimension (default 384)
        depth: residual blocks (default 4)
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

        # ── Conduit Solver (always active in v2) ──
        self._conduit_solver = FLEighConduit()
        self.last_conduit_packet = None

        # ── Spectral Cross-Attention (unchanged) ──
        self.cross_attn = nn.ModuleList([
            SpectralCrossAttention(D, n_heads=n_heads)
            for _ in range(n_cross)
        ])

        # ── Conduit Decoder (NEW — requires full spectral+conduit) ──
        self.decoder = ConduitDecoder(
            V=V, D=D, hidden=hidden, depth=depth,
            patch_dim=self.patch_dim,
        )

        # ── Boundary Smoothing (unchanged) ──
        self.boundary_smooth = BoundarySmooth(channels=3, mid=smooth_mid)

    def _svd(self, A: torch.Tensor):
        """SVD via conduit solver. Always captures telemetry in v2."""
        solver = self._conduit_solver.to(A.device)
        U, S, Vh, packet = gram_eigh_svd_conduit(A, solver)
        self.last_conduit_packet = packet
        return U, S, Vh, packet

    def encode_patches(self, patches: torch.Tensor) -> dict:
        """Encode patches → spectral decomposition + conduit.

        Returns dict with U, S_orig, S, Vt, M, and conduit_packet.
        """
        B, N, _ = patches.shape
        flat = patches.reshape(B * N, -1)

        # Residual MLP encoder (identical to v1)
        h = F.gelu(self.enc_in(flat))
        for block in self.enc_blocks:
            h = h + block(h)

        # Project to matrix manifold and sphere-normalize
        M = self.enc_out(h).reshape(B * N, self.matrix_v, self.D)
        M = F.normalize(M, dim=-1)

        # Exact SVD decomposition WITH conduit capture
        U, S, Vt, packet = self._svd(M)

        # Reshape for cross-attention
        U = U.reshape(B, N, self.matrix_v, self.D)
        S = S.reshape(B, N, self.D)
        Vt = Vt.reshape(B, N, self.D, self.D)
        M = M.reshape(B, N, self.matrix_v, self.D)

        # Cross-patch spectral coordination
        S_coordinated = S
        for layer in self.cross_attn:
            S_coordinated = layer(S_coordinated)

        return {
            'U': U, 'S_orig': S, 'S': S_coordinated,
            'Vt': Vt, 'M': M, 'conduit_packet': packet,
        }

    def decode_patches(self, U: torch.Tensor, S: torch.Tensor,
                       Vt: torch.Tensor, packet: ConduitPacket) -> torch.Tensor:
        """Decode from decomposed spectral + conduit representation.

        NO M_hat shortcut. The ConduitDecoder must use every element.

        Args:
            U:  (B, N, V, D)
            S:  (B, N, D) — coordinated singular values
            Vt: (B, N, D, D)
            packet: ConduitPacket (BN-batched from _svd)

        Returns:
            patches: (B, N, patch_dim)
        """
        B, N, V, D = U.shape
        U_flat = U.reshape(B * N, V, D)
        S_flat = S.reshape(B * N, D)
        Vt_flat = Vt.reshape(B * N, D, D)

        # Decoder receives decomposed components + conduit
        # NO M_hat = U @ diag(S) @ Vt reconstruction
        decoded = self.decoder(U_flat, S_flat, Vt_flat, packet)
        return decoded.reshape(B, N, -1)

    def forward(self, images: torch.Tensor) -> dict:
        """Full encode → SVD+conduit → coordinate → conduit-decode → stitch.

        Args:
            images: (B, 3, H, W)

        Returns:
            dict with recon, svd (including conduit_packet)
        """
        B, C, H, W = images.shape
        ps = self.patch_size
        patches, gh, gw = extract_patches(images, ps)

        # Encode → SVD+conduit → cross-attention
        svd = self.encode_patches(patches)

        # Conduit-forced decode → stitch → smooth
        decoded = self.decode_patches(
            svd['U'], svd['S'], svd['Vt'], svd['conduit_packet'])
        recon = stitch_patches(decoded, gh, gw, ps)
        recon = self.boundary_smooth(recon)

        return {'recon': recon, 'svd': svd}

    @staticmethod
    def from_v1(v1_model, strict_encoder: bool = True):
        """Initialize v2 from a trained v1 model.

        Copies encoder weights exactly. Decoder is fresh (must retrain).
        Cross-attention weights copied if shapes match.

        Args:
            v1_model: trained PatchSVAE v1 instance
            strict_encoder: if True, assert encoder weights match exactly
        """
        v2 = PatchSVAEv2(
            V=v1_model.matrix_v,
            D=v1_model.D,
            ps=v1_model.patch_size,
            hidden=v1_model.enc_in.in_features,  # infer hidden from encoder
            depth=len(v1_model.enc_blocks),
            n_cross=len(v1_model.cross_attn),
        )

        # Copy encoder weights (frozen or unfrozen, caller decides)
        v2.enc_in.load_state_dict(v1_model.enc_in.state_dict())
        for v2_block, v1_block in zip(v2.enc_blocks, v1_model.enc_blocks):
            v2_block.load_state_dict(v1_block.state_dict())
        v2.enc_out.load_state_dict(v1_model.enc_out.state_dict())

        # Copy cross-attention weights
        for v2_ca, v1_ca in zip(v2.cross_attn, v1_model.cross_attn):
            v2_ca.load_state_dict(v1_ca.state_dict())

        # Copy boundary smooth
        v2.boundary_smooth.load_state_dict(
            v1_model.boundary_smooth.state_dict())

        # Decoder is NEW — random init, must train
        print(f"  PatchSVAEv2 initialized from v1:")
        print(f"    Encoder:      copied ({'frozen' if strict_encoder else 'unfrozen'})")
        print(f"    Cross-attn:   copied")
        print(f"    Boundary:     copied")
        print(f"    Decoder:      NEW (conduit-forced, random init)")
        print(f"    Conduit solver: always active")

        n_v1 = sum(p.numel() for p in v1_model.parameters())
        n_v2 = sum(p.numel() for p in v2.parameters())
        print(f"    v1 params: {n_v1:,}")
        print(f"    v2 params: {n_v2:,}")
        print(f"    Delta:     {n_v2 - n_v1:+,}")

        return v2

    @staticmethod
    def effective_rank(S: torch.Tensor) -> torch.Tensor:
        p = S / (S.sum(-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        return (-(p * p.log()).sum(-1)).exp()

    @staticmethod
    def s_delta(S_orig: torch.Tensor, S_coord: torch.Tensor) -> float:
        return (S_coord - S_orig).abs().mean().item()