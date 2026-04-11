"""
PatchSVAE v2 — Hierarchical Spectral Cascade
==============================================
SAME INTERFACE AS V1. Different internals.

External API (identical to PatchSVAE v1):
    model.encode_patches(patches) → dict with U, S_orig, S, Vt, M
    model.decode_patches(U, S, Vt) → patches
    model.forward(images) → dict with recon, svd

Internal differences:
    - SVD uses FLEighConduit (captures conduit telemetry)
    - Hierarchy groups patches 2×2 at 3 levels, each with SVD+conduit
    - Decoder receives multi-scale conduit context per patch
    - Conduit evidence enters per-patch decoding at each cascade phase
    - Patches remain independent experts — hierarchy is structural context

256 patches → 64 cells → 16 blocks → 4 groups
Each level: GroupAttention + MLP + SVD+conduit
Conduit trickles from each level into per-patch decoder context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from geolip_core.linalg.eigh import FLEigh, _FL_MAX_N
from geolip_core.linalg.conduit import FLEighConduit, ConduitPacket


# ── SVD Backends ─────────────────────────────────────────────────

def gram_eigh_svd(A):
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


def gram_eigh_svd_conduit(A, conduit_solver):
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


# ── Spatial Grouping ─────────────────────────────────────────────

def spatial_group_2x2(x, gh, gw):
    """(B, gh*gw, C) → (B, gh//2*gw//2, 4, C)"""
    B, N, C = x.shape
    x = x.reshape(B, gh, gw, C)
    x = x.reshape(B, gh // 2, 2, gw // 2, 2, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(B, (gh // 2) * (gw // 2), 4, C)


def broadcast_to_patches(ctx, gh, gw, target_gh, target_gw):
    """Broadcast group-level context to per-patch.
    ctx: (B, gh*gw, C) at coarse level
    Returns: (B, target_gh*target_gw, C) by repeating spatially.
    """
    B, N, C = ctx.shape
    ctx = ctx.reshape(B, gh, gw, C)
    scale_h = target_gh // gh
    scale_w = target_gw // gw
    ctx = ctx.unsqueeze(2).unsqueeze(4)  # (B, gh, 1, gw, 1, C)
    ctx = ctx.expand(B, gh, scale_h, gw, scale_w, C)
    ctx = ctx.reshape(B, target_gh * target_gw, C)
    return ctx


# ── Patch Utilities ──────────────────────────────────────────────

def extract_patches(images, patch_size=4):
    B, C, H, W = images.shape
    gh, gw = H // patch_size, W // patch_size
    p = images.reshape(B, C, gh, patch_size, gw, patch_size)
    p = p.permute(0, 2, 4, 1, 3, 5)
    return p.reshape(B, gh * gw, C * patch_size * patch_size), gh, gw


def stitch_patches(patches, gh, gw, patch_size=4):
    B = patches.shape[0]
    p = patches.reshape(B, gh, gw, 3, patch_size, patch_size)
    return p.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, gh * patch_size, gw * patch_size)


class BoundarySmooth(nn.Module):
    def __init__(self, channels=3, mid=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1), nn.GELU(),
            nn.Conv2d(mid, channels, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


# ── Spectral Token ───────────────────────────────────────────────

TOKEN_DIM = 16  # S[4] + log_friction[4] + settle[4] + char_coeffs[4]


def make_spectral_token(S, packet):
    """Compact token: [S, log1p(friction), settle, char_coeffs]. Conduit detached."""
    friction = torch.log1p(packet.friction.detach())
    settle = packet.settle.detach()
    char_c = packet.char_coeffs.detach()
    return torch.cat([S, friction, settle, char_c], dim=-1)


# ── Internal Components ─────────────────────────────────────────

class GroupAttention(nn.Module):
    """Attention within groups of 4 elements."""
    def __init__(self, dim, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """x: (B, N_groups, 4, dim) → (B*N, 4, dim)"""
        B, N, four, C = x.shape
        x_flat = x.reshape(B * N, four, C)
        x_n = self.norm(x_flat)
        qkv = self.qkv(x_n).reshape(B * N, four, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * N, four, C)
        return (x_flat + self.out(out)).reshape(B, N, four, C)


class HierarchyStage(nn.Module):
    """One level: group 4 tokens → attend → MLP → SVD+conduit → token."""
    def __init__(self, V_stage, D, hidden):
        super().__init__()
        self.V = V_stage
        self.D = D
        self.attn = GroupAttention(TOKEN_DIM, n_heads=2)
        self.mlp_in = nn.Linear(4 * TOKEN_DIM, hidden)
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.mlp_out = nn.Linear(hidden, V_stage * D)
        nn.init.orthogonal_(self.mlp_out.weight)

    def forward(self, grouped, conduit_solver):
        """grouped: (B, N, 4, TOKEN_DIM) → tokens: (B, N, TOKEN_DIM), svd_state"""
        B, N, four, C = grouped.shape
        attended = self.attn(grouped)  # (B, N, 4, C)
        flat = attended.reshape(B * N, 4 * C)
        h = F.gelu(self.mlp_in(flat))
        h = h + self.mlp_block(h)
        M = F.normalize(self.mlp_out(h).reshape(B * N, self.V, self.D), dim=-1)
        U, S, Vt, packet = gram_eigh_svd_conduit(M, conduit_solver)
        token = make_spectral_token(S, packet)
        return token.reshape(B, N, TOKEN_DIM), packet


# ── Cross-Attention (v1 compatible) ──────────────────────────────

class SpectralCrossAttention(nn.Module):
    def __init__(self, D, n_heads=2, max_alpha=0.2, alpha_init=-2.0):
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


# ── PatchSVAE v2 ────────────────────────────────────────────────

class PatchSVAEv2(nn.Module):
    """Hierarchical Spectral Cascade — V1 INTERFACE.

    Patches are independent experts. Hierarchy is structural context.

    Constructor matches v1 with optional v2 params.
    encode_patches, decode_patches, forward match v1 signatures exactly.

    Args:
        V, D, ps, hidden, depth, n_cross, n_heads, smooth_mid:
            Same as PatchSVAE v1.
        stage_hidden: hierarchy stage MLP hidden (default 128)
        stage_V: hierarchy stage matrix rows (default 16)
    """
    def __init__(self, V: int = 48, D: int = 4, ps: int = 4,
                 hidden: int = 384, depth: int = 4, n_cross: int = 2,
                 n_heads: int = None, smooth_mid: int = None,
                 stage_hidden: int = 128, stage_V: int = 16):
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

        # ── Conduit solver (no learnable params) ──
        self._conduit_solver = FLEighConduit()
        self.last_conduit_packet = None
        self._hierarchy_context = None  # set during forward()

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

        # ── Cross-attention (identical to v1) ──
        self.cross_attn = nn.ModuleList([
            SpectralCrossAttention(D, n_heads=n_heads)
            for _ in range(n_cross)
        ])

        # ── Hierarchy stages (v2 internal) ──
        self.stage1 = HierarchyStage(stage_V, D, stage_hidden)
        self.stage2 = HierarchyStage(stage_V, D, stage_hidden)
        self.stage3 = HierarchyStage(stage_V, D, stage_hidden)

        # ── Decoder (v1 structure + hierarchy context injection) ──
        # hierarchy_ctx_dim = 3 levels × TOKEN_DIM per level
        hierarchy_ctx_dim = 3 * TOKEN_DIM  # 48

        self.dec_in = nn.Linear(self.mat_dim + hierarchy_ctx_dim, hidden)
        # Fallback for standalone decode (no hierarchy context)
        self.dec_in_fallback = nn.Linear(self.mat_dim, hidden)

        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.dec_out = nn.Linear(hidden, self.patch_dim)

        # ── Boundary smoothing ──
        self.boundary_smooth = BoundarySmooth(channels=3, mid=smooth_mid)

    def _svd(self, A):
        """SVD via FLEighConduit. Stores conduit packet."""
        solver = self._conduit_solver.to(A.device)
        U, S, Vh, packet = gram_eigh_svd_conduit(A, solver)
        self.last_conduit_packet = packet
        return U, S, Vh

    def encode_patches(self, patches: torch.Tensor) -> dict:
        """Encode patches to omega tokens. SAME AS V1.

        Args:
            patches: (B, N, patch_dim)

        Returns:
            dict with U, S_orig, S, Vt, M
            (v2 also stores conduit in self.last_conduit_packet)
        """
        B, N, _ = patches.shape
        flat = patches.reshape(B * N, -1)

        h = F.gelu(self.enc_in(flat))
        for block in self.enc_blocks:
            h = h + block(h)

        M = self.enc_out(h).reshape(B * N, self.matrix_v, self.D)
        M = F.normalize(M, dim=-1)

        U, S, Vt = self._svd(M)

        U = U.reshape(B, N, self.matrix_v, self.D)
        S = S.reshape(B, N, self.D)
        Vt = Vt.reshape(B, N, self.D, self.D)
        M = M.reshape(B, N, self.matrix_v, self.D)

        S_coordinated = S
        for layer in self.cross_attn:
            S_coordinated = layer(S_coordinated)

        return {
            'U': U, 'S_orig': S, 'S': S_coordinated,
            'Vt': Vt, 'M': M,
        }

    def _build_hierarchy(self, svd, gh, gw):
        """Build multi-scale conduit context per patch.

        Groups patches 2×2 at 3 levels, runs SVD+conduit at each.
        Broadcasts each level's spectral token back to per-patch.
        Returns: (B, N, 3*TOKEN_DIM) context per patch.
        """
        B = svd['S'].shape[0]
        solver = self._conduit_solver.to(svd['S'].device)
        packet0 = self.last_conduit_packet

        # Build level 0 tokens from patch SVD
        tok0 = make_spectral_token(
            svd['S_orig'].reshape(B * (gh * gw), self.D), packet0
        ).reshape(B, gh * gw, TOKEN_DIM)

        # Level 1: 2×2 → 64 cells
        gh1, gw1 = gh // 2, gw // 2
        grouped1 = spatial_group_2x2(tok0, gh, gw)
        tok1, pkt1 = self.stage1(grouped1, solver)

        # Level 2: 2×2 → 16 blocks
        gh2, gw2 = gh1 // 2, gw1 // 2
        grouped2 = spatial_group_2x2(tok1, gh1, gw1)
        tok2, pkt2 = self.stage2(grouped2, solver)

        # Level 3: 2×2 → 4 groups
        gh3, gw3 = gh2 // 2, gw2 // 2
        grouped3 = spatial_group_2x2(tok2, gh2, gw2)
        tok3, pkt3 = self.stage3(grouped3, solver)

        # Broadcast each level back to per-patch resolution
        ctx1 = broadcast_to_patches(tok1, gh1, gw1, gh, gw)  # (B, N, 16)
        ctx2 = broadcast_to_patches(tok2, gh2, gw2, gh, gw)  # (B, N, 16)
        ctx3 = broadcast_to_patches(tok3, gh3, gw3, gh, gw)  # (B, N, 16)

        # Per-patch context: concatenated multi-scale tokens
        return torch.cat([ctx1, ctx2, ctx3], dim=-1)  # (B, N, 48)

    def decode_patches(self, U: torch.Tensor, S: torch.Tensor,
                       Vt: torch.Tensor) -> torch.Tensor:
        """Decode omega tokens back to patches. SAME SIGNATURE AS V1.

        When called from forward(), uses hierarchy context internally.
        When called standalone, falls back to v1-style decoding.

        Args:
            U:  (B, N, V, D)
            S:  (B, N, D)
            Vt: (B, N, D, D)

        Returns:
            patches: (B, N, patch_dim)
        """
        B, N, V, D = U.shape
        U_flat = U.reshape(B * N, V, D)
        S_flat = S.reshape(B * N, D)
        Vt_flat = Vt.reshape(B * N, D, D)

        M_hat = torch.bmm(U_flat * S_flat.unsqueeze(1), Vt_flat)
        m_flat = M_hat.reshape(B * N, -1)

        if self._hierarchy_context is not None:
            ctx = self._hierarchy_context.reshape(B * N, -1).detach()
            h = F.gelu(self.dec_in(torch.cat([m_flat, ctx], dim=-1)))
        else:
            h = F.gelu(self.dec_in_fallback(m_flat))

        for block in self.dec_blocks:
            h = h + block(h)

        return self.dec_out(h).reshape(B, N, -1)

    def forward(self, images: torch.Tensor) -> dict:
        """Full pipeline. SAME RETURN FORMAT AS V1.

        Internally builds hierarchy context between encode and decode.
        """
        B, C, H, W = images.shape
        ps = self.patch_size
        patches, gh, gw = extract_patches(images, ps)

        # Encode (v1-identical per patch)
        svd = self.encode_patches(patches)

        # Build hierarchy context (v2 internal)
        self._hierarchy_context = self._build_hierarchy(svd, gh, gw)

        # Decode (v1 signature, uses hierarchy internally)
        decoded = self.decode_patches(svd['U'], svd['S'], svd['Vt'])

        # Clean internal state
        self._hierarchy_context = None

        recon = stitch_patches(decoded, gh, gw, ps)
        recon = self.boundary_smooth(recon)

        return {'recon': recon, 'svd': svd}

    @staticmethod
    def effective_rank(S):
        p = S / (S.sum(-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        return (-(p * p.log()).sum(-1)).exp()

    @staticmethod
    def s_delta(S_orig, S_coord):
        return (S_coord - S_orig).abs().mean().item()