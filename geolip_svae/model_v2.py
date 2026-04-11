"""
PatchSVAE v2 — Hierarchical Spectral Cascade
==============================================
SVD + conduit at EVERY level of the spatial hierarchy.
Conduit difficulty trickles stage by stage from patches to manifold.

Encoder hierarchy (16×16 grid, D=4 at every level):
  Level 0: 256 patches → encode → SVD+conduit₀ → 256 spectral tokens
  Level 1: group 2×2 → 64 cells  → attend → SVD+conduit₁ → 64 tokens
  Level 2: group 2×2 → 16 blocks → attend → SVD+conduit₂ → 16 tokens
  Level 3: group 2×2 → 4 groups  → attend → SVD+conduit₃ → 4 tokens

Each level captures decomposition difficulty at its spatial scale:
  Level 0: pixel-level spectral structure (edges, textures)
  Level 1: local patch interactions (2×2 neighborhoods)
  Level 2: meso-scale structure (4×4 regions)
  Level 3: global composition (full quadrants)

Decoder reverses the hierarchy (spectral U-Net):
  Each decoder level receives skip from its encoder level
  with the FULL stored SVD + conduit from that scale.
  Every conduit element is architecturally load-bearing at its level.

Spectral token (propagates between levels):
  S[4] + log_friction[4] + settle[4] + char_coeffs[4] = 16 values
  = the spectral signature + conditioning at that resolution
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


# ── Utilities ────────────────────────────────────────────────────

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


def spatial_group_2x2(x, gh, gw):
    """Group 2×2 spatial neighbors. (B, gh*gw, C) → (B, gh//2*gw//2, 4, C)"""
    B, N, C = x.shape
    x = x.reshape(B, gh, gw, C)
    x = x.reshape(B, gh // 2, 2, gw // 2, 2, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, gh//2, gw//2, 2, 2, C)
    return x.reshape(B, (gh // 2) * (gw // 2), 4, C)


def spatial_ungroup_2x2(x, gh, gw):
    """Ungroup back to spatial grid. (B, gh*gw, 4, C) → (B, gh*2*gw*2, C)"""
    B, N, four, C = x.shape
    x = x.reshape(B, gh, gw, 2, 2, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, gh, 2, gw, 2, C)
    return x.reshape(B, gh * 2 * gw * 2, C)


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
D = 4


def make_spectral_token(S, packet):
    """Build compact spectral token from SVD output + conduit.

    Token = [S, log1p(friction), settle, char_coeffs]
    All conduit values detached — no gradients through piecewise dynamics.
    S retains gradients for encoder training.

    Args:
        S: (BN, D) — singular values (gradient-carrying)
        packet: ConduitPacket from FLEighConduit

    Returns:
        token: (BN, TOKEN_DIM=16)
    """
    friction = torch.log1p(packet.friction.detach())  # (BN, D)
    settle = packet.settle.detach()                    # (BN, D)
    char_c = packet.char_coeffs.detach()               # (BN, D)
    return torch.cat([S, friction, settle, char_c], dim=-1)


# ── Local Group Attention ────────────────────────────────────────

class GroupAttention(nn.Module):
    """Attention within groups of 4 elements.

    Each group attends internally — 4 spectral tokens interact.
    Lightweight: operates on token_dim directly.
    """
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
        """x: (B, N_groups, 4, dim) → (B, N_groups, 4, dim)"""
        B, N, four, C = x.shape
        x_flat = x.reshape(B * N, four, C)
        x_n = self.norm(x_flat)
        qkv = self.qkv(x_n).reshape(B * N, four, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * N, four, C)
        return x_flat + self.out(out)  # residual
        # Reshape handled by caller


# ── Encoder Stage ────────────────────────────────────────────────

class EncoderStage(nn.Module):
    """One level of hierarchical encoding.

    Groups 4 tokens from previous level → attend →
    MLP → M matrix → SVD+conduit → spectral token.

    Stores full SVD decomposition for decoder skip connection.
    """
    def __init__(self, input_dim, V_stage, D, hidden):
        super().__init__()
        self.V = V_stage
        self.D = D

        # Attention over 4 grouped tokens
        self.group_attn = GroupAttention(input_dim, n_heads=2)

        # MLP: 4 tokens concatenated → hidden → matrix
        self.mlp_in = nn.Linear(4 * input_dim, hidden)
        self.mlp_block = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.mlp_out = nn.Linear(hidden, V_stage * D)
        nn.init.orthogonal_(self.mlp_out.weight)

    def forward(self, grouped, conduit_solver):
        """
        grouped: (B, N_groups, 4, input_dim) — groups of 4 tokens
        Returns:
            tokens: (B, N_groups, TOKEN_DIM) — spectral tokens
            svd_state: dict with U, S, Vt, packet for decoder
        """
        B, N, four, C = grouped.shape

        # Attend within each group
        attended = self.group_attn(grouped)  # (B*N, 4, C)

        # Flatten group → MLP → matrix
        flat = attended.reshape(B * N, 4 * C)
        h = F.gelu(self.mlp_in(flat))
        h = h + self.mlp_block(h)
        M = F.normalize(
            self.mlp_out(h).reshape(B * N, self.V, self.D), dim=-1)

        # SVD + conduit
        U, S, Vt, packet = gram_eigh_svd_conduit(M, conduit_solver)

        # Build spectral token for next level
        token = make_spectral_token(S, packet)  # (B*N, TOKEN_DIM)

        return token.reshape(B, N, TOKEN_DIM), {
            'U': U.reshape(B, N, self.V, self.D),
            'S': S.reshape(B, N, self.D),
            'Vt': Vt.reshape(B, N, self.D, self.D),
            'M': M.reshape(B, N, self.V, self.D),
            'packet': packet,
        }


# ── Decoder Stage ────────────────────────────────────────────────

class DecoderStage(nn.Module):
    """One level of hierarchical decoding.

    Takes parent tokens from above, expands to 4 children,
    injects stored SVD+conduit from corresponding encoder level,
    attends within group, refines.
    """
    def __init__(self, parent_dim, child_dim, V_stage, D, hidden):
        super().__init__()
        self.D = D
        self.V = V_stage

        # Expand 1 parent → 4 children
        self.expand = nn.Linear(parent_dim, 4 * hidden)

        # Inject stored conduit: hidden + S + friction + settle + char_c + Vt_flat
        conduit_dim = D + D + D + D + D * D  # S + fric + settle + char_c + Vt
        self.inject = nn.Linear(hidden + conduit_dim, hidden)
        self.inject_norm = nn.LayerNorm(hidden)

        # Attention over 4 children
        self.group_attn = GroupAttention(hidden, n_heads=2)

        # Refine → output token
        self.refine = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, child_dim),
        )

    def forward(self, parent_tokens, enc_svd_state):
        """
        parent_tokens: (B, N_parent, parent_dim)
        enc_svd_state: dict with U, S, Vt, packet from encoder

        Returns: (B, N_parent*4, child_dim) — expanded child tokens
        """
        B, N, _ = parent_tokens.shape
        packet = enc_svd_state['packet']

        # Expand each parent to 4 children
        expanded = self.expand(parent_tokens)  # (B, N, 4*hidden)
        hidden = expanded.shape[-1] // 4
        children = expanded.reshape(B * N, 4, hidden)  # (B*N, 4, hidden)

        # Build conduit evidence from stored encoder state
        S = enc_svd_state['S'].reshape(B * N, self.D)
        Vt = enc_svd_state['Vt'].reshape(B * N, self.D * self.D)
        friction = torch.log1p(packet.friction.detach()).reshape(B * N, self.D)
        settle = packet.settle.detach().reshape(B * N, self.D)
        char_c = packet.char_coeffs.detach().reshape(B * N, self.D)

        # Conduit is the SAME for all 4 children in a group
        # (it describes the group-level decomposition)
        conduit_ev = torch.cat([S, friction, settle, char_c, Vt], dim=-1)  # (B*N, conduit_dim)
        conduit_ev = conduit_ev.unsqueeze(1).expand(-1, 4, -1)  # (B*N, 4, conduit_dim)

        # Inject conduit into each child
        fused = torch.cat([children, conduit_ev], dim=-1)  # (B*N, 4, hidden+conduit_dim)
        fused = self.inject_norm(F.gelu(self.inject(fused)))  # (B*N, 4, hidden)

        # Attend within group of 4 children
        fused = fused.reshape(B, N, 4, hidden)
        attended = self.group_attn(fused)  # (B*N, 4, hidden)

        # Refine to child tokens
        out = self.refine(attended)  # (B*N, 4, child_dim)
        return out.reshape(B, N * 4, -1)


# ── Patch Encoder (Level 0) ─────────────────────────────────────

class PatchEncoder(nn.Module):
    """Level 0: individual patches → SVD+conduit → spectral tokens."""

    def __init__(self, patch_dim, V, D, hidden, depth):
        super().__init__()
        self.V = V
        self.D = D
        self.enc_in = nn.Linear(patch_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.enc_out = nn.Linear(hidden, V * D)
        nn.init.orthogonal_(self.enc_out.weight)

    def forward(self, patches, conduit_solver):
        """
        patches: (B, N, patch_dim)
        Returns:
            tokens: (B, N, TOKEN_DIM)
            svd_state: dict with U, S, Vt, M, packet
        """
        B, N, _ = patches.shape
        flat = patches.reshape(B * N, -1)

        h = F.gelu(self.enc_in(flat))
        for block in self.blocks:
            h = h + block(h)

        M = F.normalize(
            self.enc_out(h).reshape(B * N, self.V, self.D), dim=-1)
        U, S, Vt, packet = gram_eigh_svd_conduit(M, conduit_solver)

        token = make_spectral_token(S, packet)

        return token.reshape(B, N, TOKEN_DIM), {
            'U': U.reshape(B, N, self.V, self.D),
            'S': S.reshape(B, N, self.D),
            'Vt': Vt.reshape(B, N, self.D, self.D),
            'M': M.reshape(B, N, self.V, self.D),
            'packet': packet,
        }


# ── Patch Decoder (Level 0) ─────────────────────────────────────

class PatchDecoder(nn.Module):
    """Level 0 decoder: spectral tokens + stored conduit → pixel patches.

    Uses stored U, S, Vt, and conduit from encoder level 0
    to reconstruct patch pixels. Conduit enters as required context.
    """
    def __init__(self, token_dim, V, D, hidden, depth, patch_dim):
        super().__init__()
        self.V = V
        self.D = D

        # Input: token from level 1 decoder + stored SVD + conduit
        conduit_dim = D + D + D + D + D * D + V * D
        # S + friction + settle + char_c + Vt_flat + U_flat
        self.fuse_in = nn.Linear(token_dim + conduit_dim, hidden)
        self.fuse_norm = nn.LayerNorm(hidden)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.out = nn.Linear(hidden, patch_dim)

    def forward(self, child_tokens, enc_svd_state):
        """
        child_tokens: (B, N_patches, token_dim) — from level 1 decoder
        enc_svd_state: dict from level 0 encoder
        Returns: (B, N_patches, patch_dim)
        """
        B, N, C = child_tokens.shape
        packet = enc_svd_state['packet']

        # Build conduit evidence from stored level 0 state
        S = enc_svd_state['S'].reshape(B * N, self.D)
        Vt = enc_svd_state['Vt'].reshape(B * N, self.D * self.D)
        U = enc_svd_state['U'].reshape(B * N, self.V * self.D)
        friction = torch.log1p(packet.friction.detach()).reshape(B * N, self.D)
        settle = packet.settle.detach().reshape(B * N, self.D)
        char_c = packet.char_coeffs.detach().reshape(B * N, self.D)

        conduit_ev = torch.cat([S, friction, settle, char_c, Vt, U], dim=-1)

        # Fuse token + full conduit
        flat_tokens = child_tokens.reshape(B * N, C)
        fused = torch.cat([flat_tokens, conduit_ev], dim=-1)
        h = self.fuse_norm(F.gelu(self.fuse_in(fused)))

        for block in self.blocks:
            h = h + block(h)

        return self.out(h).reshape(B, N, -1)


# ── Cross-Attention (top level) ──────────────────────────────────

class SpectralCrossAttention(nn.Module):
    """Cross-attention over the final 4 spectral tokens at the top level."""
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


# ── PatchSVAE v2 — Hierarchical Spectral Cascade ────────────────

class PatchSVAEv2(nn.Module):
    """Hierarchical spectral autoencoder with conduit trickle.

    SVD + conduit at 4 spatial scales. Decoder uses stored
    conduit at each level for reconstruction.

    16×16 grid → 4 levels:
      256 patches → 64 cells → 16 blocks → 4 groups

    Args:
        V: patch matrix rows (48 for ps=4)
        D: spectral modes (4 — constant at all levels)
        ps: patch size (4)
        enc_hidden: patch encoder hidden (384)
        enc_depth: patch encoder residual blocks (4)
        stage_hidden: hierarchical stage hidden (128)
        stage_V: matrix rows at levels 1-3 (16)
        dec_depth: patch decoder residual blocks (3)
        n_cross: cross-attention layers at top (2)
        smooth_mid: boundary smooth channels (8)
    """
    def __init__(self, V=48, D=4, ps=4,
                 enc_hidden=384, enc_depth=4,
                 stage_hidden=128, stage_V=16,
                 dec_depth=3, n_cross=2, smooth_mid=8):
        super().__init__()
        self.matrix_v = V
        self.D = D
        self.patch_size = ps
        self.patch_dim = 3 * ps * ps

        # Shared conduit solver (no learnable params)
        self._conduit_solver = FLEighConduit()
        self.last_conduit_packet = None

        # ── Level 0: Patch encoder ──
        self.patch_encoder = PatchEncoder(
            self.patch_dim, V, D, enc_hidden, enc_depth)

        # ── Levels 1-3: Hierarchical stages ──
        self.stage1 = EncoderStage(TOKEN_DIM, stage_V, D, stage_hidden)
        self.stage2 = EncoderStage(TOKEN_DIM, stage_V, D, stage_hidden)
        self.stage3 = EncoderStage(TOKEN_DIM, stage_V, D, stage_hidden)

        # ── Top-level cross-attention (over 4 final tokens) ──
        self.cross_attn = nn.ModuleList([
            SpectralCrossAttention(D, n_heads=2)
            for _ in range(n_cross)
        ])

        # ── Decoder levels 3→0 ──
        self.dec_stage3 = DecoderStage(
            TOKEN_DIM, TOKEN_DIM, stage_V, D, stage_hidden)
        self.dec_stage2 = DecoderStage(
            TOKEN_DIM, TOKEN_DIM, stage_V, D, stage_hidden)
        self.dec_stage1 = DecoderStage(
            TOKEN_DIM, TOKEN_DIM, stage_V, D, stage_hidden)

        # ── Level 0 decoder: tokens → pixels ──
        self.patch_decoder = PatchDecoder(
            TOKEN_DIM, V, D, enc_hidden, dec_depth, self.patch_dim)

        # ── Boundary smoothing ──
        self.boundary_smooth = BoundarySmooth(channels=3, mid=smooth_mid)

    def forward(self, images):
        B, C, H, W = images.shape
        ps = self.patch_size
        gh0, gw0 = H // ps, W // ps  # 16, 16

        solver = self._conduit_solver.to(images.device)
        patches, gh0, gw0 = extract_patches(images, ps)

        # ═══════════════════════════════════════════
        # ENCODER — bottom up
        # ═══════════════════════════════════════════

        # Level 0: patches → spectral tokens
        tok0, svd0 = self.patch_encoder(patches, solver)
        # tok0: (B, 256, 16), svd0: full SVD+conduit at patch level
        self.last_conduit_packet = svd0['packet']

        # Level 1: group 2×2 → 64 cells
        gh1, gw1 = gh0 // 2, gw0 // 2  # 8, 8
        grouped1 = spatial_group_2x2(tok0, gh0, gw0)  # (B, 64, 4, 16)
        tok1, svd1 = self.stage1(grouped1, solver)     # (B, 64, 16)

        # Level 2: group 2×2 → 16 blocks
        gh2, gw2 = gh1 // 2, gw1 // 2  # 4, 4
        grouped2 = spatial_group_2x2(tok1, gh1, gw1)   # (B, 16, 4, 16)
        tok2, svd2 = self.stage2(grouped2, solver)      # (B, 16, 16)

        # Level 3: group 2×2 → 4 groups
        gh3, gw3 = gh2 // 2, gw2 // 2  # 2, 2
        grouped3 = spatial_group_2x2(tok2, gh2, gw2)    # (B, 4, 4, 16)
        tok3, svd3 = self.stage3(grouped3, solver)       # (B, 4, 16)

        # Top-level cross-attention on S values of the 4 final groups
        S_top = svd3['S']  # (B, 4, D)
        S_orig_top = S_top.clone()
        for layer in self.cross_attn:
            S_top = layer(S_top)

        # ═══════════════════════════════════════════
        # DECODER — top down with conduit skips
        # ═══════════════════════════════════════════

        # Level 3': 4 tokens → 16 block tokens
        dec3 = self.dec_stage3(tok3, svd3)              # (B, 16, 16)

        # Level 2': 16 tokens → 64 cell tokens
        dec2 = self.dec_stage2(dec3, svd2)              # (B, 64, 16)

        # Level 1': 64 tokens → 256 patch tokens
        dec1 = self.dec_stage1(dec2, svd1)              # (B, 256, 16)

        # Level 0': patch tokens + stored conduit → pixels
        recon_patches = self.patch_decoder(dec1, svd0)  # (B, 256, 48)

        # Stitch + smooth
        recon = stitch_patches(recon_patches, gh0, gw0, ps)
        recon = self.boundary_smooth(recon)

        return {
            'recon': recon,
            'svd': {
                # Level 0 (for compatibility / monitoring)
                'S_orig': svd0['S'].reshape(B, gh0 * gw0, self.D),
                'S': svd0['S'].reshape(B, gh0 * gw0, self.D),
                'U': svd0['U'],
                'Vt': svd0['Vt'],
                'M': svd0['M'],
                'conduit_packet': svd0['packet'],
                # Hierarchy
                'levels': {
                    0: svd0, 1: svd1, 2: svd2, 3: svd3,
                },
                'S_top': S_top,
                'S_top_orig': S_orig_top,
            },
        }

    @staticmethod
    def effective_rank(S):
        p = S / (S.sum(-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        return (-(p * p.log()).sum(-1)).exp()

    @staticmethod
    def s_delta(S_orig, S_coord):
        return (S_coord - S_orig).abs().mean().item()