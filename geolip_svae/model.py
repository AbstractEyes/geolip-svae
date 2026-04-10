"""
PatchSVAE — Patch-based Spectral Variational Autoencoder
==========================================================
Image → patches → encode → sphere normalize → SVD →
cross-patch spectral attention → decode → stitch → smooth.

Architecture:
    - Residual MLP encoder/decoder (hidden=768, depth=4)
    - Row-wise sphere normalization (F.normalize, dim=-1)
    - SVD via Gram-eigh in fp64 (exact decomposition)
    - Multiplicative spectral cross-attention (2 layers, 2272 params)
    - Zero-initialized boundary smoothing (~600 params)
    - Total: 16,942,419 parameters

Proven configurations:
    Fresnel-tiny   64×64:   16 patches, MSE=0.0005 (TinyImageNet)
    Fresnel-small 128×128:  64 patches, MSE=0.0000734 (ImageNet-128)
    Fresnel-base  256×256: 256 patches, MSE=0.0000610 (ImageNet-256)
    Johanna-small 128×128:  64 patches, MSE=0.029 (16 noise types)
    Johanna-base  256×256: 256 patches, MSE=0.027 (16 noise types, scheduled)

Solver modes:
    solver='default'  — standard FLEigh (no telemetry)
    solver='conduit'  — FLEighConduit (emits ConduitPacket via last_conduit_packet)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── SVD Backend ──────────────────────────────────────────────────

from geolip_core.linalg.eigh import FLEigh, _FL_MAX_N
from geolip_core.linalg.conduit import FLEighConduit, ConduitPacket


def gram_eigh_svd(A: torch.Tensor):
    """Thin SVD via Gram matrix eigendecomposition in fp64.

    Uses geolip-core FLEigh for N <= 12 on CUDA (optimized small-matrix eigh).
    Falls back to torch.linalg.eigh on CPU or large N.

    Args:
        A: (B, M, N) tensor, M >= N

    Returns:
        U: (B, M, N)  left singular vectors
        S: (B, N)     singular values (descending)
        Vh: (B, N, N)  right singular vectors (transposed)
    """
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
    """Thin SVD via Gram eigendecomposition WITH conduit telemetry.

    Identical arithmetic to gram_eigh_svd. Additionally returns the
    ConduitPacket capturing friction, settle, extraction_order, and
    other adjudication evidence from the ACTUAL decomposition.

    Args:
        A: (B, M, N) tensor, M >= N
        conduit_solver: FLEighConduit instance (on correct device)

    Returns:
        U:      (B, M, N)  left singular vectors
        S:      (B, N)     singular values (descending)
        Vh:     (B, N, N)  right singular vectors
        packet: ConduitPacket — telemetry from the real decomposition
    """
    B, M, N = A.shape
    orig_dtype = A.dtype

    with torch.amp.autocast('cuda', enabled=False):
        A_d = A.double()
        G = torch.bmm(A_d.transpose(1, 2), A_d)

        # FLEighConduit on the actual Gram matrix
        packet = conduit_solver(G.float())

        eigenvalues = packet.eigenvalues.double().flip(-1)
        V = packet.eigenvectors.double().flip(-1)

        S = torch.sqrt(eigenvalues.clamp(min=1e-24))
        U = torch.bmm(A_d, V) / S.unsqueeze(1).clamp(min=1e-16)
        Vh = V.transpose(-2, -1).contiguous()

    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype), packet


# ── Cayley-Menger Geometric Monitoring ───────────────────────────

def cayley_menger_vol2(points: torch.Tensor) -> torch.Tensor:
    """Squared simplex volume via Cayley-Menger determinant in fp64.

    Args:
        points: (B, N, D) — B sets of N points in D dimensions

    Returns:
        vol2: (B,) — squared volume of each simplex
    """
    B, N, D = points.shape
    pts = points.double()
    gram = torch.bmm(pts, pts.transpose(1, 2))
    norms = torch.diagonal(gram, dim1=1, dim2=2)
    d2 = F.relu(norms.unsqueeze(2) + norms.unsqueeze(1) - 2 * gram)
    cm = torch.zeros(B, N + 1, N + 1, device=points.device, dtype=torch.float64)
    cm[:, 0, 1:] = 1.0
    cm[:, 1:, 0] = 1.0
    cm[:, 1:, 1:] = d2
    k = N - 1
    sign = (-1.0) ** (k + 1)
    fact = math.factorial(k)
    return sign * torch.linalg.det(cm) / ((2 ** k) * (fact ** 2))


def cv_of(emb: torch.Tensor, n_samples: int = 200) -> float:
    """Coefficient of variation of pentachoron volumes.

    Measures geometric uniformity of the embedding space.
    CV ≈ 0.20-0.23 is the universal attractor band.

    Args:
        emb: (V, D) — rows of a sphere-normalized matrix
        n_samples: number of random 5-point subsets to sample

    Returns:
        CV value (float), or 0.0 if insufficient valid volumes
    """
    if emb.dim() != 2 or emb.shape[0] < 5:
        return 0.0
    N, D = emb.shape
    pool = min(N, 512)
    indices = torch.stack([
        torch.randperm(pool, device=emb.device)[:5]
        for _ in range(n_samples)
    ])
    vol2 = cayley_menger_vol2(emb[:pool][indices])
    valid = vol2 > 1e-20
    if valid.sum() < 10:
        return 0.0
    vols = vol2[valid].sqrt()
    return (vols.std() / (vols.mean() + 1e-8)).item()


# ── Patch Utilities ──────────────────────────────────────────────

def extract_patches(images: torch.Tensor, patch_size: int = 16):
    """Extract non-overlapping patches from images.

    Args:
        images: (B, C, H, W)
        patch_size: size of square patches

    Returns:
        patches: (B, N, C*patch_size*patch_size)
        gh, gw: grid dimensions
    """
    B, C, H, W = images.shape
    gh, gw = H // patch_size, W // patch_size
    p = images.reshape(B, C, gh, patch_size, gw, patch_size)
    p = p.permute(0, 2, 4, 1, 3, 5)
    return p.reshape(B, gh * gw, C * patch_size * patch_size), gh, gw


def stitch_patches(patches: torch.Tensor, gh: int, gw: int,
                   patch_size: int = 16) -> torch.Tensor:
    """Stitch patches back into images.

    Args:
        patches: (B, N, C*patch_size*patch_size)
        gh, gw: grid dimensions
        patch_size: size of square patches

    Returns:
        images: (B, 3, gh*patch_size, gw*patch_size)
    """
    B = patches.shape[0]
    p = patches.reshape(B, gh, gw, 3, patch_size, patch_size)
    return p.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, gh * patch_size, gw * patch_size)


# ── Boundary Smoothing ──────────────────────────────────────────

class BoundarySmooth(nn.Module):
    """Post-stitch boundary refinement. ~600 params, zero-initialized.

    Learns residual corrections at patch seams. Starts as identity
    (zero init on final conv) and gradually learns to smooth boundaries.
    """
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


# ── Spectral Cross-Attention ────────────────────────────────────

class SpectralCrossAttention(nn.Module):
    """Multiplicative spectral coordination with learnable per-mode alpha.

    S_out = S * (1 + α_d * tanh(attention_output_d))

    The alpha parameters are bounded by max_alpha (default 0.2) and
    initialized near zero (sigmoid(-2.0) * 0.2 ≈ 0.024). This ensures
    the cross-attention starts as near-identity and gradually learns
    to coordinate spectral modes across patches.

    Total parameters per layer: D*(3D + D + 2D + D + 1) ≈ 1136 for D=16
    """
    def __init__(self, D: int, n_heads: int = 4,
                 max_alpha: float = 0.2, alpha_init: float = -2.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        self.max_alpha = max_alpha
        assert D % n_heads == 0, f"D={D} must be divisible by n_heads={n_heads}"

        self.qkv = nn.Linear(D, 3 * D)
        self.out_proj = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.scale = self.head_dim ** -0.5
        self.alpha_logits = nn.Parameter(torch.full((D,), alpha_init))

    @property
    def alpha(self) -> torch.Tensor:
        """Bounded per-mode scaling: [0, max_alpha]."""
        return self.max_alpha * torch.sigmoid(self.alpha_logits)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """Coordinate singular values across patches.

        Args:
            S: (B, N, D) — singular values for N patches

        Returns:
            S_coordinated: (B, N, D) — spectrally coordinated values
        """
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


# ── PatchSVAE ───────────────────────────────────────────────────

class PatchSVAE(nn.Module):
    """Patch-based Spectral Variational Autoencoder.

    Encodes images as omega tokens — singular value vectors on S^{D-1}.

    Supports multiple regimes:
        Fresnel/Johanna: V=256, D=16, ps=16, hidden=768 (17M params)
        Freckles:        V=48,  D=4,  ps=4,  hidden=384 (2.5M params)

    Solver modes:
        solver='default'  — standard FLEigh (production, no telemetry)
        solver='conduit'  — FLEighConduit (captures ConduitPacket per forward)

    Args:
        V: rows per encoded matrix (default 256)
        D: columns / spectral dimensions (default 16)
        ps: patch size (default 16)
        hidden: MLP hidden dimension (default 768)
        depth: number of residual blocks (default 4)
        n_cross: number of spectral cross-attention layers (default 2)
        n_heads: attention heads (default: min(4, D) for D>=4, else 1)
        smooth_mid: BoundarySmooth hidden channels (default: 16 for ps>=16, else 8)
        solver: 'default' or 'conduit'
    """
    def __init__(self, V: int = 256, D: int = 16, ps: int = 16,
                 hidden: int = 768, depth: int = 4, n_cross: int = 2,
                 n_heads: int = None, smooth_mid: int = None,
                 solver: str = 'default'):
        super().__init__()
        self.matrix_v = V
        self.D = D
        self.patch_size = ps
        self.patch_dim = 3 * ps * ps
        self.mat_dim = V * D

        # Solver configuration
        self.solver = solver
        self.last_conduit_packet = None
        self._conduit_solver = None  # lazy init

        # Resolve regime-dependent defaults
        if n_heads is None:
            n_heads = 2 if D <= 8 else min(4, D)
        if smooth_mid is None:
            smooth_mid = 16 if ps >= 16 else 8

        # Encoder
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

        # Decoder
        self.dec_in = nn.Linear(self.mat_dim, hidden)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.dec_out = nn.Linear(hidden, self.patch_dim)

        # Spectral cross-attention
        self.cross_attn = nn.ModuleList([
            SpectralCrossAttention(D, n_heads=n_heads)
            for _ in range(n_cross)
        ])

        # Boundary smoothing
        self.boundary_smooth = BoundarySmooth(channels=3, mid=smooth_mid)

    def _get_conduit_solver(self):
        """Lazy-init conduit solver on correct device."""
        if self._conduit_solver is None:
            self._conduit_solver = FLEighConduit()
        # Ensure on same device as model
        device = self.enc_in.weight.device
        if next(iter([]), None) is None:  # no params to check on FLEighConduit
            self._conduit_solver = self._conduit_solver.to(device)
        return self._conduit_solver

    def _svd(self, A: torch.Tensor):
        """SVD via Gram-eigh. Routes to conduit if configured.

        solver='default': standard FLEigh, no telemetry
        solver='conduit': FLEighConduit, stores ConduitPacket in self.last_conduit_packet
        """
        if self.solver == 'conduit':
            conduit_solver = self._get_conduit_solver()
            U, S, Vh, packet = gram_eigh_svd_conduit(A, conduit_solver)
            self.last_conduit_packet = packet
            return U, S, Vh
        else:
            self.last_conduit_packet = None
            return gram_eigh_svd(A)

    def encode_patches(self, patches: torch.Tensor) -> dict:
        """Encode patches to omega tokens.

        Args:
            patches: (B, N, patch_dim)

        Returns:
            dict with keys:
                U:      (B, N, V, D)  left singular vectors
                S_orig: (B, N, D)     raw singular values
                S:      (B, N, D)     coordinated singular values (omega tokens)
                Vt:     (B, N, D, D)  right singular vectors
                M:      (B, N, V, D)  sphere-normalized encoding matrix
        """
        B, N, _ = patches.shape
        flat = patches.reshape(B * N, -1)

        # Residual MLP encoder
        h = F.gelu(self.enc_in(flat))
        for block in self.enc_blocks:
            h = h + block(h)

        # Project to matrix manifold and sphere-normalize
        M = self.enc_out(h).reshape(B * N, self.matrix_v, self.D)
        M = F.normalize(M, dim=-1)  # rows → S^{D-1}

        # Exact SVD decomposition (routes through _svd)
        U, S, Vt = self._svd(M)

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
            'Vt': Vt, 'M': M,
        }

    def decode_patches(self, U: torch.Tensor, S: torch.Tensor,
                       Vt: torch.Tensor) -> torch.Tensor:
        """Decode omega tokens back to patches.

        Args:
            U:  (B, N, V, D)
            S:  (B, N, D) — coordinated singular values
            Vt: (B, N, D, D)

        Returns:
            patches: (B, N, patch_dim)
        """
        B, N, V, D = U.shape
        U_flat = U.reshape(B * N, V, D)
        S_flat = S.reshape(B * N, D)
        Vt_flat = Vt.reshape(B * N, D, D)

        # Reconstruct matrix from SVD components
        M_hat = torch.bmm(U_flat * S_flat.unsqueeze(1), Vt_flat)

        # Residual MLP decoder
        h = F.gelu(self.dec_in(M_hat.reshape(B * N, -1)))
        for block in self.dec_blocks:
            h = h + block(h)

        return self.dec_out(h).reshape(B, N, -1)

    def forward(self, images: torch.Tensor) -> dict:
        """Full encode → SVD → coordinate → decode → stitch pipeline.

        Args:
            images: (B, 3, H, W) — H and W must be divisible by patch_size

        Returns:
            dict with keys:
                recon: (B, 3, H, W) — reconstructed images
                svd:   dict — full SVD decomposition (U, S, S_orig, Vt, M)
        """
        B, C, H, W = images.shape
        ps = self.patch_size
        gh, gw = H // ps, W // ps

        # Extract patches
        patches, gh, gw = extract_patches(images, ps)

        # Encode → SVD → cross-attention
        svd = self.encode_patches(patches)

        # Decode → stitch → smooth
        decoded = self.decode_patches(svd['U'], svd['S'], svd['Vt'])
        recon = stitch_patches(decoded, gh, gw, ps)
        recon = self.boundary_smooth(recon)

        return {'recon': recon, 'svd': svd}

    @staticmethod
    def effective_rank(S: torch.Tensor) -> torch.Tensor:
        """Effective rank of singular value distribution.

        erank = exp(-Σ p_i log p_i) where p_i = σ_i / Σσ

        Architectural constant ≈ 15.88 for D=16.
        """
        p = S / (S.sum(-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        return (-(p * p.log()).sum(-1)).exp()

    @staticmethod
    def s_delta(S_orig: torch.Tensor, S_coord: torch.Tensor) -> float:
        """Mean absolute spectral shift from cross-attention.

        Converges to modality-specific binding constants:
            Images:  ~0.238
            Noise:   ~0.350-0.407
            Text:    ~0.350
        """
        return (S_coord - S_orig).abs().mean().item()