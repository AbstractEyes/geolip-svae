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
from geolip_core.linalg import batched_svd as _gc_batched_svd


SVD_METHODS = ('auto', 'fl', 'gram_eigh', 'triton', 'torch')
SVD_COMPUTE_DTYPES = ('fp64', 'fp32')


def gram_eigh_svd(A: torch.Tensor, method: str = 'auto',
                   compute_dtype: str = 'fp64'):
    """Thin SVD — delegates to geolip_core.linalg.batched_svd (auto-dispatch).

    Routing performed by batched_svd (method='auto'):
        N=2..6, CUDA + Triton:    Fused Triton kernel (D=2..6 inclusive)
        N<=12, CUDA, fp32:        Gram + FL eigh
        N<=12, CUDA, fp64:        Gram + torch.linalg.eigh
                                  (FLEigh returns fp32 V which silently caps
                                   fp64 SVD orthogonality at ~1e-3 — the
                                   dispatcher avoids FL on fp64 paths)
        N>12 or CPU:              Gram + torch.linalg.eigh
        Wide shape (M<N):         Transparent transpose

    PatchSVAE inputs M to this function (B*N, V, D) with V >> D in the
    canonical case, so the tall path is taken; the dedicated Triton kernel
    fires for D ∈ {2,3,4,5,6} on CUDA when triton is installed.

    Args:
        A:             (B, M, N) tensor
        method:        'auto' | 'fl' | 'gram_eigh' | 'triton' | 'torch'
                       Forwarded to batched_svd. 'auto' picks the best path
                       per N/dtype/device. See SVD_METHODS for the full set.
        compute_dtype: 'fp64' (default) or 'fp32' — internal precision.
                       fp64 is required for stable eigenvector orthogonality;
                       fp32 is faster but ill-conditioning can corrupt V.

    Returns:
        U: (B, M, N)  left singular vectors
        S: (B, N)     singular values (descending)
        Vh: (B, N, N) right singular vectors (transposed)
    """
    return _gc_batched_svd(A, method=method, compute_dtype=compute_dtype)


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
                   patch_size: int = 16, channels: int = 3) -> torch.Tensor:
    """Stitch patches back into images.

    Args:
        patches: (B, N, C*patch_size*patch_size)
        gh, gw: grid dimensions
        patch_size: size of square patches
        channels: image channel count C (must match how the patches were
            extracted; defaults to 3 for back-compat with prior callers)

    Returns:
        images: (B, C, gh*patch_size, gw*patch_size)
    """
    B = patches.shape[0]
    p = patches.reshape(B, gh, gw, channels, patch_size, patch_size)
    return p.permute(0, 3, 1, 4, 2, 5).reshape(
        B, channels, gh * patch_size, gw * patch_size,
    )


# ── Boundary Smoothing ──────────────────────────────────────────

class BoundarySmooth(nn.Module):
    """Post-stitch boundary refinement. ~600 params, zero-initialized.

    Learns residual corrections at patch seams. Starts as identity
    (zero init on final conv) and gradually learns to smooth boundaries.
    """
    def __init__(self, channels: int = 3, mid: int = 16,
                 activation: str = 'gelu'):
        super().__init__()
        # Activation is parameterless — swapping does not change state_dict shape.
        # ACTIVATION_MODULES is defined later in the file; resolve lazily by name.
        act_factory = ACTIVATION_MODULES[activation]
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1),
            act_factory(),
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

# Ablation helpers — support F/G/H/L groups as parameterized toggles
# inside the PatchSVAE class.

# Functional activations — used at sites that call act_fn(tensor) directly
# (encoder/decoder outer activation in encode_patches / decode_patches).
#
# All entries are parameterless OR use safe defaults so that swapping does
# NOT alter state_dict shape — every existing checkpoint reloads identically
# regardless of which entry is selected. PReLU/RReLU intentionally excluded
# (trainable params would silently inflate model size).
ACTIVATIONS = {
    # Smooth, near-identity around 0
    'gelu':         F.gelu,
    'gelu_tanh':    lambda x: F.gelu(x, approximate='tanh'),
    'silu':         F.silu,
    'swish':        F.silu,                  # alias of silu
    'mish':         F.mish,

    # ReLU family
    'relu':         F.relu,
    'relu6':        F.relu6,
    'leaky_relu':   F.leaky_relu,            # negative_slope=0.01
    'elu':          F.elu,                   # alpha=1.0
    'selu':         F.selu,
    'celu':         F.celu,                  # alpha=1.0

    # Bounded
    'tanh':         torch.tanh,
    'sigmoid':      torch.sigmoid,
    'hardtanh':     F.hardtanh,              # min/max=-1/+1
    'hardsigmoid':  F.hardsigmoid,
    'hardswish':    F.hardswish,

    # Shaped / shifted
    'softplus':     F.softplus,              # beta=1, threshold=20
    'softsign':     F.softsign,
    'logsigmoid':   F.logsigmoid,
    'tanhshrink':   F.tanhshrink,

    # Pass-through
    'identity':     lambda x: x,
}

# nn.Module factories for the SAME activation set, used inside
# nn.Sequential blocks (encoder/decoder residual blocks, BoundarySmooth).
# Each entry is a zero-arg callable that returns a fresh module.
ACTIVATION_MODULES = {
    'gelu':         nn.GELU,
    'gelu_tanh':    lambda: nn.GELU(approximate='tanh'),
    'silu':         nn.SiLU,
    'swish':        nn.SiLU,                 # alias of silu
    'mish':         nn.Mish,

    'relu':         nn.ReLU,
    'relu6':        nn.ReLU6,
    'leaky_relu':   nn.LeakyReLU,
    'elu':          nn.ELU,
    'selu':         nn.SELU,
    'celu':         nn.CELU,

    'tanh':         nn.Tanh,
    'sigmoid':      nn.Sigmoid,
    'hardtanh':     nn.Hardtanh,
    'hardsigmoid':  nn.Hardsigmoid,
    'hardswish':    nn.Hardswish,

    'softplus':     nn.Softplus,
    'softsign':     nn.Softsign,
    'logsigmoid':   nn.LogSigmoid,
    'tanhshrink':   nn.Tanhshrink,

    'identity':     nn.Identity,
}

# Sanity invariant — keep the two registries aligned so that any name
# valid in one is valid in the other. This catches typos when extending.
assert set(ACTIVATIONS) == set(ACTIVATION_MODULES), (
    f"ACTIVATIONS / ACTIVATION_MODULES key sets diverged: "
    f"only-functional={set(ACTIVATIONS) - set(ACTIVATION_MODULES)}, "
    f"only-module={set(ACTIVATION_MODULES) - set(ACTIVATIONS)}"
)

# Per-site activation slots inside PatchSVAE. Each entry resolves to one
# of the keys in ACTIVATIONS / ACTIVATION_MODULES. Defaults preserve the
# pre-config behavior exactly: GELU on every site, with `enc_in` driven
# by the legacy `activation` kwarg (still F-group ablation surface).
ACTIVATION_SITES = (
    'enc_in',           # outer activation in encode_patches (between enc_in and blocks)
    'enc_block_inner',  # activation inside each encoder residual block
    'dec_in',           # outer activation in decode_patches (between dec_in and blocks)
    'dec_block_inner',  # activation inside each decoder residual block
    'boundary_smooth',  # activation inside BoundarySmooth (between two convs)
)
DEFAULT_ACTIVATIONS = {site: 'gelu' for site in ACTIVATION_SITES}


def _resolve_activations(activations, activation):
    """Build the per-site activation dict.

    Precedence:
      explicit `activations[site]` > legacy `activation` kwarg (enc_in only) > 'gelu'

    Args:
        activations: None, or a dict mapping site name -> activation name.
                     Unknown keys raise; missing keys fall through to defaults.
        activation:  Legacy F-group kwarg. Acts as a shortcut for the
                     'enc_in' slot only (matches pre-refactor behavior).

    Returns: dict[str, str] with every key in ACTIVATION_SITES.
    """
    resolved = dict(DEFAULT_ACTIVATIONS)
    # Legacy single-string kwarg: only affects enc_in (as before).
    if activation is not None:
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation!r}. "
                f"Valid: {sorted(ACTIVATIONS)}")
        resolved['enc_in'] = activation
    if activations:
        for site, name in activations.items():
            if site not in ACTIVATION_SITES:
                raise ValueError(
                    f"Unknown activation site: {site!r}. "
                    f"Valid: {ACTIVATION_SITES}")
            if name not in ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation: {name!r} for site {site!r}. "
                    f"Valid: {sorted(ACTIVATIONS)}")
            resolved[site] = name
    return resolved


def _row_normalize(M: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply row normalization to M (shape [*, V, D]).

    'sphere'     → F.normalize(dim=-1), rows on S^(D-1)  [default]
    'layernorm'  → per-row zero-mean / unit-variance
    'scale'      → per-row divide by max abs
    'none'       → identity
    """
    if mode == 'sphere':
        return F.normalize(M, dim=-1)
    elif mode == 'layernorm':
        mean = M.mean(dim=-1, keepdim=True)
        std = M.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (M - mean) / std
    elif mode == 'scale':
        scale = M.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        return M / scale
    elif mode == 'none':
        return M
    else:
        raise ValueError(f"Unknown row_norm mode: {mode}")


def _init_weights(module: nn.Module, scheme: str) -> None:
    """L-group init: override default nn.Linear init with one of the schemes."""
    if scheme == 'kaiming_normal':
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif scheme == 'xavier_uniform':
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif scheme == 'normal_0_02':
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif scheme == 'orthogonal':
        pass  # already the default for enc_out; let others fall through
    else:
        raise ValueError(f"Unknown init_scheme: {scheme}")


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
        channels: image channel count C (default 3). Sets patch_dim = C*ps*ps
            and the BoundarySmooth in/out channels. The geometric core
            (sphere-norm M, SVD, cross-attn, codebook) is channel-agnostic;
            channels only plumbs the encoder input dim, decoder output dim,
            and the post-stitch boundary smoother.
        solver: 'default' or 'conduit'
    """
    def __init__(self, V: int = 256, D: int = 16, ps: int = 16,
                 hidden: int = 768, depth: int = 4, n_cross: int = 2,
                 n_heads: int = None, smooth_mid: int = None,
                 channels: int = 3,
                 solver: str = 'default',
                 # ── Ablation toggles (F/G/H/L groups) ─────────────────
                 activation: str = 'gelu',
                 activations: 'Optional[Dict[str, str]]' = None,
                 row_norm: str = 'sphere',
                 svd_mode: str = 'default',
                 svd_method: str = 'auto',
                 svd_compute_dtype: str = 'fp64',
                 linear_readout: bool = False,
                 match_params: bool = True,
                 init_scheme: str = 'orthogonal'):
        """
        Ablation toggles:
            activation:     'gelu' (default) | 'relu' | 'silu' | 'tanh' | 'identity'
                            — Legacy F-group ablation. Now equivalent to setting
                            activations={'enc_in': activation} only; preserved
                            for back-compat with existing configs and presets.
            activations:    None (default) | dict[str,str] — fine-grained
                            per-site activation control. Keys must be drawn
                            from ACTIVATION_SITES:
                                'enc_in'           outer activation in encode_patches
                                'enc_block_inner'  activation inside each encoder block
                                'dec_in'           outer activation in decode_patches
                                'dec_block_inner'  activation inside each decoder block
                                'boundary_smooth'  activation inside BoundarySmooth
                            Values must be drawn from ACTIVATIONS keys (same
                            registry as `activation`). Missing keys fall through
                            to current defaults (all GELU; enc_in honors the
                            legacy `activation` kwarg). All activations are
                            parameterless modules — swapping does NOT change
                            state_dict shape, so existing checkpoints reload.
            row_norm:       'sphere' (default) | 'layernorm' | 'scale' | 'none'
                            — normalization applied to encoded M rows (G group)
            svd_mode:       'default' (use `solver` + svd_method) | 'fp32' | 'fp64' |
                            'batch_shared' | 'none' (H group)
                            When 'none' AND linear_readout=True: SVD is bypassed
                            entirely and a learned linear readout replaces it.
                            Column norms of readout output stand in as S;
                            Vt is identity. This is the sphere-solver path.
            svd_method:     'auto' (default) | 'fl' | 'gram_eigh' | 'triton' | 'torch'
                            — Routing for the geolip-core dispatcher when
                            svd_mode='default'. 'auto' picks the best path
                            per N/dtype/device; fused Triton kernel fires for
                            D∈{2,3,4,5,6} on CUDA when triton is installed.
                            Ignored when solver='conduit' (conduit owns dispatch).
            svd_compute_dtype: 'fp64' (default) | 'fp32' — internal SVD precision.
                            fp64 is required for stable V orthogonality on the
                            production sphere-norm path.
            linear_readout: False (default) | True — replace SVD with learned
                            nn.Linear(V*D → V*D) readout (H group)
            match_params:   True (default) | False — when linear_readout=True,
                            True uses nn.Linear(V*D, V*D), False uses Identity
                            (saves params but breaks geometric expressiveness)
            init_scheme:    'orthogonal' (default, on enc_out) | 'kaiming_normal'
                            | 'xavier_uniform' | 'normal_0_02' — initialization
                            scheme for Linear layers (L group). Orthogonal is
                            always re-applied to enc_out regardless.
        """
        super().__init__()
        self.matrix_v = V
        self.D = D
        self.patch_size = ps
        self.channels = channels
        self.patch_dim = channels * ps * ps
        self.mat_dim = V * D

        # Solver configuration
        self.solver = solver
        self.last_conduit_packet = None
        self._conduit_solver = None  # lazy init

        # Ablation mode storage
        self.activations = _resolve_activations(activations, activation)
        # Back-compat alias: external code reading activation_name still works.
        self.activation_name = self.activations['enc_in']
        self.row_norm_mode = row_norm
        self.svd_mode = svd_mode
        # SVD dispatcher config (forwarded to gram_eigh_svd via _svd)
        if svd_method not in SVD_METHODS:
            raise ValueError(
                f"Unknown svd_method: {svd_method!r}. Valid: {SVD_METHODS}")
        if svd_compute_dtype not in SVD_COMPUTE_DTYPES:
            raise ValueError(
                f"Unknown svd_compute_dtype: {svd_compute_dtype!r}. "
                f"Valid: {SVD_COMPUTE_DTYPES}")
        self.svd_method = svd_method
        self.svd_compute_dtype = svd_compute_dtype
        self.linear_readout = linear_readout
        self.match_params = match_params
        self.init_scheme = init_scheme

        # Resolve regime-dependent defaults
        if n_heads is None:
            n_heads = 2 if D <= 8 else min(4, D)
        if smooth_mid is None:
            smooth_mid = 16 if ps >= 16 else 8

        # Resolve module factories for inner-block / boundary activations.
        enc_block_act = ACTIVATION_MODULES[self.activations['enc_block_inner']]
        dec_block_act = ACTIVATION_MODULES[self.activations['dec_block_inner']]

        # Encoder
        self.enc_in = nn.Linear(self.patch_dim, hidden)
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                enc_block_act(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.enc_out = nn.Linear(hidden, self.mat_dim)
        nn.init.orthogonal_(self.enc_out.weight)

        # H group: optional learned readout replacing SVD
        if linear_readout:
            if match_params:
                self.readout = nn.Linear(self.mat_dim, self.mat_dim)
            else:
                self.readout = nn.Identity()

        # Decoder
        self.dec_in = nn.Linear(self.mat_dim, hidden)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                dec_block_act(),
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
        self.boundary_smooth = BoundarySmooth(
            channels=channels, mid=smooth_mid,
            activation=self.activations['boundary_smooth'],
        )

        # L group: optional init override
        if init_scheme != 'orthogonal':
            _init_weights(self, init_scheme)
            # Re-apply orthogonal to enc_out — load-bearing per the
            # architecture docs (validated in ablation Phase 1/2).
            nn.init.orthogonal_(self.enc_out.weight)

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

        solver='default': geolip-core batched_svd dispatcher, parameterized by
                          self.svd_method and self.svd_compute_dtype. No
                          telemetry; for D∈{2..6} on CUDA+Triton this hits
                          the fused N=4/etc. kernel.
        solver='conduit': FLEighConduit. Owns dispatch — svd_method ignored.
                          Stores ConduitPacket in self.last_conduit_packet.
        """
        if self.solver == 'conduit':
            conduit_solver = self._get_conduit_solver()
            U, S, Vh, packet = gram_eigh_svd_conduit(A, conduit_solver)
            self.last_conduit_packet = packet
            return U, S, Vh
        else:
            self.last_conduit_packet = None
            return gram_eigh_svd(
                A, method=self.svd_method,
                compute_dtype=self.svd_compute_dtype,
            )

    def encode_patches(self, patches: torch.Tensor) -> dict:
        """Encode patches to omega tokens.

        Args:
            patches: (B, N, patch_dim)

        Returns:
            dict with keys:
                U:      (B, N, V, D)  left singular vectors (or M_hat if linear_readout)
                S_orig: (B, N, D)     raw singular values (or column norms if linear_readout)
                S:      (B, N, D)     coordinated singular values (omega tokens)
                Vt:     (B, N, D, D)  right singular vectors (or identity if linear_readout)
                M:      (B, N, V, D)  row-normalized encoding matrix
        """
        B, N, _ = patches.shape
        flat = patches.reshape(B * N, -1)

        # F group: configurable activation on enc_in
        act_fn = ACTIVATIONS[self.activation_name]
        h = act_fn(self.enc_in(flat))
        for block in self.enc_blocks:
            # Inner block activations (GELU inside Sequential) remain.
            # F-group ablation only swaps the outer activation.
            h = h + block(h)

        # G group: configurable row normalization
        M = self.enc_out(h).reshape(B * N, self.matrix_v, self.D)
        M = _row_normalize(M, self.row_norm_mode)

        # H group: SVD decomposition or linear-readout replacement
        if self.linear_readout:
            # Sphere-solver path: learned linear readout replaces SVD.
            # This is the H2_linear_matched architecture used by the
            # h2-64 battery array (when combined with svd_mode='none').
            flat_M = M.reshape(B * N, -1)
            M_hat = self.readout(flat_M).reshape(B * N, self.matrix_v, self.D)
            U = M_hat
            # Column norms stand in as singular values
            S = M_hat.norm(dim=-2)
            # Vt is identity — decode reduces to U * S.unsqueeze(1)
            Vt = torch.eye(self.D, device=M.device, dtype=M.dtype
                            ).unsqueeze(0).expand(B * N, -1, -1)
        elif self.svd_mode == 'fp32':
            # Low-precision SVD path (ablation variant)
            G = torch.bmm(M.transpose(1, 2), M)
            G.diagonal(dim1=-2, dim2=-1).add_(1e-6)
            eigenvalues, Vmat = torch.linalg.eigh(G)
            eigenvalues = eigenvalues.flip(-1)
            Vmat = Vmat.flip(-1)
            S = torch.sqrt(eigenvalues.clamp(min=1e-12))
            U = torch.bmm(M, Vmat) / S.unsqueeze(1).clamp(min=1e-8)
            Vt = Vmat.transpose(-2, -1).contiguous()
        elif self.svd_mode == 'fp64':
            # Raw fp64 SVD (used by ablation_trainer, not the geolip-core
            # FLEigh path). Kept for ablation-reproducibility.
            with torch.amp.autocast('cuda', enabled=False):
                A_d = M.double()
                G = torch.bmm(A_d.transpose(1, 2), A_d)
                G.diagonal(dim1=-2, dim2=-1).add_(1e-12)
                eigenvalues, Vmat = torch.linalg.eigh(G)
                eigenvalues = eigenvalues.flip(-1)
                Vmat = Vmat.flip(-1)
                S_d = torch.sqrt(eigenvalues.clamp(min=1e-24))
                U_d = torch.bmm(A_d, Vmat) / S_d.unsqueeze(1).clamp(min=1e-16)
                Vt_d = Vmat.transpose(-2, -1).contiguous()
            U = U_d.to(M.dtype)
            S = S_d.to(M.dtype)
            Vt = Vt_d.to(M.dtype)
        elif self.svd_mode == 'batch_shared':
            # Single SVD per batch — S/Vt replicated across patches
            M_batched = M.reshape(B, N * self.matrix_v, self.D)
            U_b, S_b, Vt_b = self._svd(M_batched)
            S = S_b.unsqueeze(1).expand(-1, N, -1).reshape(B * N, self.D)
            Vt = Vt_b.unsqueeze(1).expand(-1, N, -1, -1).reshape(
                B * N, self.D, self.D)
            U = torch.bmm(M, Vt.transpose(-2, -1)) / S.unsqueeze(1
                                                                    ).clamp(min=1e-16)
        else:  # 'default' — production FLEigh path
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

 