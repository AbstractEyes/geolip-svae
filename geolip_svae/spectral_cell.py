"""
SpectralCell
============
Drop-in layer: (B, N, token_dim) → (B, N, token_dim).

Pipeline:
    tokens → Linear → residual MLP → Linear(hidden, V*D) → reshape(V, D)
    → F.normalize(dim=-1) → SVD(Gram-eigh, fp64) → U, S, Vt
    → cross-attention scales S per mode across all N tokens
    → recompose M_hat = U · diag(S_modified) · Vt
    → Linear → residual MLP → Linear(hidden, token_dim) → output

SVD is in the forward pass. Differentiable. Gradients flow through
U, S, Vt back to the input projection weights.

Cross-attention modifies S multiplicatively:
    S_out = S * (1 + α * tanh(attention_output))
    α per mode, bounded [0, max_alpha], initialized ~0.024.
    M_hat ≠ M after this step.

Sphere normalization enforces ||row||=1 for all V rows.
This constrains trace(M^T M) = V (fixed total spectral energy).
The SVD decomposes how that fixed energy distributes across D axes.

Cayley-Menger validation on M rows:
    Sample pentachora (5-point subsets) from the V rows on S^{D-1}.
    CM determinant → squared simplex volume.
    CV = std(vol) / mean(vol) over n_samples subsets.
    Measures geometric uniformity of the representation.

Author: AbstractPhil + Claude Opus
License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


# ── Cayley-Menger ───────────────────────────────────────────────

class CMValidator(nn.Module):
    """Batch-friendly Cayley-Menger determinant.
    Computes pairwise squared distances and simplex volume
    for (k+1)-point subsets in arbitrary embedding dimension.

    For k=4: 5 vertices → 10 pairwise d² + 1 vol².
    """
    def __init__(self, k):
        super().__init__()
        self._k = k
        self._nv = k + 1
        pairs = list(combinations(range(self._nv), 2))
        self._npairs = len(pairs)
        self.register_buffer('_pi', torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer('_pj', torch.tensor([p[1] for p in pairs], dtype=torch.long))
        sign = (-1.0) ** (k + 1)
        fact = math.factorial(k)
        self._prefactor = sign / ((2.0 ** k) * (fact ** 2))

    def forward(self, verts):
        """verts: (..., nv, edim) → d2_pairs: (..., npairs), vol2: (...)"""
        gram = torch.einsum('...ve,...we->...vw', verts, verts)
        norms = torch.diagonal(gram, dim1=-2, dim2=-1)
        d2_mat = norms.unsqueeze(-1) + norms.unsqueeze(-2) - 2 * gram
        d2_mat = F.relu(d2_mat)
        d2_pairs = d2_mat[..., self._pi, self._pj]
        shape = d2_mat.shape[:-2]
        Vn = d2_mat.shape[-1]
        cm = torch.zeros(*shape, Vn + 1, Vn + 1, device=d2_mat.device, dtype=d2_mat.dtype)
        cm[..., 0, 1:] = 1.0
        cm[..., 1:, 0] = 1.0
        cm[..., 1:, 1:] = d2_mat
        vol2 = self._prefactor * torch.linalg.det(cm.float())
        vol2 = vol2.to(d2_pairs.dtype)
        return d2_pairs, vol2


def cayley_menger_vol2(points: torch.Tensor) -> torch.Tensor:
    """Squared simplex volume via CM determinant in fp64.
    points: (B, N, D) → vol2: (B,)
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
    emb: (V, D) — rows of a sphere-normalized matrix.
    Samples random 5-point subsets, computes CM vol² for each,
    returns std(vol) / mean(vol).

    CV ≈ 0.20-0.23 is the empirically observed attractor band.
    Returns 0.0 if insufficient valid volumes.
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


# ── SVD via Gram-eigh (fp64 exact) ──────────────────────────────

def gram_eigh_svd(A: torch.Tensor):
    """Thin SVD via Gram eigendecomposition in fp64.

    Computes G = A^T A in fp64, eigendecomposes G, derives U, S, Vh.
    Diagonal perturbation 1e-12 for numerical stability.

    Args:
        A: (B, V, D) with V >= D

    Returns:
        U:  (B, V, D) left singular vectors
        S:  (B, D)    singular values, descending
        Vh: (B, D, D) right singular vectors transposed
    """
    B, V, D = A.shape
    orig = A.dtype
    with torch.amp.autocast('cuda', enabled=False):
        Ad = A.double()
        G = torch.bmm(Ad.transpose(1, 2), Ad)
        G.diagonal(dim1=-2, dim2=-1).add_(1e-12)
        eigenvalues, Vecs = torch.linalg.eigh(G)
        eigenvalues = eigenvalues.flip(-1)
        Vecs = Vecs.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-24))
        U = torch.bmm(Ad, Vecs) / S.unsqueeze(1).clamp(min=1e-16)
        Vh = Vecs.transpose(-2, -1).contiguous()
    return U.to(orig), S.to(orig), Vh.to(orig)


# ── Spectral Cross-Attention ────────────────────────────────────

class SpectralCrossAttention(nn.Module):
    """Multi-head attention on singular values across N tokens.

    Input S: (B, N, D) — one D-dim spectral profile per token.
    Attends across N positions (each token sees all others' spectra).
    Output: S * (1 + α * tanh(out_proj(attended)))

    α is per-mode, bounded [0, max_alpha] via sigmoid on learnable logits.
    Initialized at sigmoid(-2.0) * 0.2 ≈ 0.024 per mode.
    """
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
        Sn = self.norm(S)
        qkv = self.qkv(Sn).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        gate = torch.tanh(self.out_proj(out))
        alpha = self.alpha.unsqueeze(0).unsqueeze(0)
        return S * (1.0 + alpha * gate)


# ── SpectralCell ────────────────────────────────────────────────

class SpectralCell(nn.Module):
    """Processes N tokens through sphere-normalized SVD with spectral
    coordination and Cayley-Menger geometric validation.

    Shapes through the pipeline (for default V=16, D=4, hidden=128, token_dim=64):
        tokens:     (B, N, 64)
        enc_in:     Linear(64, 128)     → (B*N, 128)
        enc_blocks: 2× residual MLP     → (B*N, 128)
        enc_out:    Linear(128, 64)     → (B*N, 64) → reshape (B*N, 16, 4)
        normalize:  F.normalize(dim=-1) → each row has norm 1
        SVD:        Gram-eigh in fp64   → U(B*N,16,4), S(B*N,4), Vt(B*N,4,4)
        cross_attn: S reshaped (B,N,4)  → attention across N → S_coord (B,N,4)
        recompose:  U · diag(S_coord) · Vt → M_hat (B*N, 16, 4) → flatten (B*N, 64)
        out_in:     Linear(64, 128)     → (B*N, 128)
        out_blocks: 2× residual MLP     → (B*N, 128)
        out_proj:   Linear(128, 64)     → (B, N, 64)

    CM validation:
        M rows are V unit vectors on S^{D-1}.
        CMValidator(k=4) samples pentachora from the rows.
        vol² measures simplex volume. CV measures uniformity.
        cv_of() returns the coefficient of variation over random subsets.

    Args:
        token_dim: input and output dimension per token
        V:         matrix rows (each becomes a unit vector on S^{D-1})
        D:         matrix columns (spectral modes, eigenvalue count)
        hidden:    residual MLP width
        depth:     residual blocks in input and output projections
        n_cross:   SpectralCrossAttention layers applied to S
        n_heads:   attention heads in cross-attention (must divide D)
        max_alpha: upper bound on per-mode multiplicative scaling
    """
    def __init__(
        self,
        token_dim: int,
        V: int = 16,
        D: int = 4,
        hidden: int = 128,
        depth: int = 2,
        n_cross: int = 1,
        n_heads: int = 2,
        max_alpha: float = 0.2,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.V = V
        self.D = D
        self.mat_dim = V * D
        self.hidden = hidden

        # CM validator: k=min(4, D-1) for pentachoron on S^{D-1}
        # k=4 means 5 vertices, requires D >= 4
        self._cm_k = min(4, D - 1) if D >= 2 else 1
        self.cm = CMValidator(self._cm_k)

        # Input projection: token_dim → hidden → mat_dim
        self.enc_in = nn.Linear(token_dim, hidden)
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

        # Cross-attention on singular values across tokens
        self.cross_attn = nn.ModuleList([
            SpectralCrossAttention(D, n_heads=n_heads, max_alpha=max_alpha)
            for _ in range(n_cross)
        ])

        # Output projection: mat_dim → hidden → token_dim
        self.out_in = nn.Linear(self.mat_dim, hidden)
        self.out_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.out_proj = nn.Linear(hidden, token_dim)

    def format(self, tokens: torch.Tensor) -> dict:
        """Run full pipeline. Returns output tokens, SVD components, and CM metrics.

        Args:
            tokens: (B, N, token_dim)

        Returns:
            dict:
                output:   (B, N, token_dim) — processed tokens
                M:        (B, N, V, D)     — sphere-normalized matrix (rows on S^{D-1})
                U:        (B, N, V, D)     — left singular vectors from SVD
                S_orig:   (B, N, D)        — singular values before cross-attention
                S:        (B, N, D)        — singular values after cross-attention
                Vt:       (B, N, D, D)     — right singular vectors from SVD
                M_hat:    (B, N, V, D)     — U · diag(S_modified) · Vt (≠ M)
                cm_d2:    (B*N, npairs)    — pairwise squared distances from CM
                cm_vol2:  (B*N,)           — squared simplex volume from CM
        """
        B, N, _ = tokens.shape

        # Input projection → sphere-normalized V×D matrix
        flat = tokens.reshape(B * N, -1)
        h = F.gelu(self.enc_in(flat))
        for block in self.enc_blocks:
            h = h + block(h)
        M = self.enc_out(h).reshape(B * N, self.V, self.D)
        M = F.normalize(M, dim=-1)

        # CM validation on M rows — sample (k+1) rows per token
        # Use fixed evenly-spaced indices for deterministic CM
        nv = self._cm_k + 1
        cm_idx = torch.linspace(0, self.V - 1, nv).long().to(M.device)
        cm_verts = M[:, cm_idx, :]  # (B*N, nv, D)
        cm_d2, cm_vol2 = self.cm(cm_verts)

        # SVD decomposition (in compute graph, fp64)
        U, S, Vt = gram_eigh_svd(M)

        # Reshape for cross-attention over N tokens
        U = U.reshape(B, N, self.V, self.D)
        S = S.reshape(B, N, self.D)
        Vt = Vt.reshape(B, N, self.D, self.D)
        M = M.reshape(B, N, self.V, self.D)

        # Cross-attention multiplicatively scales S across tokens
        S_orig = S.clone()
        for layer in self.cross_attn:
            S = layer(S)

        # Recompose with modified S → M_hat ≠ M
        U_flat = U.reshape(B * N, self.V, self.D)
        S_flat = S.reshape(B * N, self.D)
        Vt_flat = Vt.reshape(B * N, self.D, self.D)
        M_hat = torch.bmm(U_flat * S_flat.unsqueeze(1), Vt_flat)

        # Output projection: M_hat → token_dim
        h = F.gelu(self.out_in(M_hat.reshape(B * N, -1)))
        for block in self.out_blocks:
            h = h + block(h)
        output = self.out_proj(h).reshape(B, N, self.token_dim)

        return {
            'output': output,
            'M': M,
            'U': U,
            'S_orig': S_orig,
            'S': S,
            'Vt': Vt,
            'M_hat': M_hat.reshape(B, N, self.V, self.D),
            'cm_d2': cm_d2,
            'cm_vol2': cm_vol2,
        }

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, token_dim) → (B, N, token_dim). Drop-in compatible."""
        return self.format(tokens)['output']

    # ── CM Diagnostics ───────────────────────────────────────────

    def cm_cv(self, M: torch.Tensor, n_samples: int = 200) -> float:
        """Compute CV of pentachoron volumes over random 5-point subsets.
        M: (B, N, V, D) — sphere-normalized matrices.
        Returns mean CV across all B*N matrices.
        """
        flat = M.reshape(-1, self.V, self.D)
        # Sample a few matrices to keep cost reasonable
        n_mats = min(flat.shape[0], 64)
        cvs = []
        for i in range(n_mats):
            c = cv_of(flat[i], n_samples=n_samples)
            cvs.append(c)
        return sum(cvs) / len(cvs) if cvs else 0.0

    def cm_vol2_stats(self, cm_vol2: torch.Tensor) -> dict:
        """Statistics on CM vol² from format() output.
        cm_vol2: (B*N,) — one vol² per token's sampled pentachoron.
        """
        valid = cm_vol2.abs() > 1e-20
        if valid.sum() < 2:
            return {'mean': 0.0, 'std': 0.0, 'frac_valid': 0.0}
        vols = cm_vol2[valid].abs().sqrt()
        return {
            'mean': vols.mean().item(),
            'std': vols.std().item(),
            'cv': (vols.std() / (vols.mean() + 1e-8)).item(),
            'frac_valid': valid.float().mean().item(),
        }

    # ── SVD Diagnostics ──────────────────────────────────────────

    @staticmethod
    def effective_rank(S: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of normalized singular values, exponentiated.
        erank = exp(-Σ p_i log p_i) where p_i = σ_i / Σσ.
        Returns 1.0 for rank-1, D for uniform spectrum.
        """
        p = S / (S.sum(-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        return (-(p * p.log()).sum(-1)).exp()

    @staticmethod
    def spectral_shift(S_orig, S_coord):
        """Mean |S_coord - S_orig| across all modes and tokens."""
        return (S_coord - S_orig).abs().mean().item()

    @staticmethod
    def trace_check(M):
        """trace(M^T M) should equal V (sum of squared unit row norms)."""
        flat = M.reshape(-1, M.shape[-2], M.shape[-1])
        G = torch.bmm(flat.transpose(1, 2), flat)
        return torch.diagonal(G, dim1=-2, dim2=-1).sum(-1).mean().item()

    def summary(self):
        """Print shapes, param count, DOF ratio, CM config."""
        n_params = sum(p.numel() for p in self.parameters())
        sphere_dof = self.V * (self.D - 1)
        ratio = sphere_dof / self.token_dim
        print(f"SpectralCell:")
        print(f"  token_dim={self.token_dim}, V={self.V}, D={self.D}")
        print(f"  mat_dim={self.mat_dim} ({self.V}×{self.D})")
        print(f"  sphere DOF={sphere_dof} (V rows × {self.D-1} free per row)")
        print(f"  CM: k={self._cm_k} ({self._cm_k+1} vertices, {self.cm._npairs} pairs)")
        print(f"  hidden={self.hidden}, depth={len(self.enc_blocks)}")
        print(f"  cross_attn={len(self.cross_attn)} layers")
        print(f"  params: {n_params:,}")
        print(f"  DOF ratio: {ratio:.2f}× "
              f"({'expand' if ratio > 1 else 'compress' if ratio < 1 else 'identity'})")


# ── Factory functions ────────────────────────────────────────────

def spectral_cell_tiny(token_dim: int) -> SpectralCell:
    """V=8, D=4, hidden=64, depth=1, 1 cross-attn."""
    return SpectralCell(token_dim, V=8, D=4, hidden=64, depth=1, n_cross=1)

def spectral_cell_small(token_dim: int) -> SpectralCell:
    """V=16, D=4, hidden=128, depth=2, 1 cross-attn."""
    return SpectralCell(token_dim, V=16, D=4, hidden=128, depth=2, n_cross=1)

def spectral_cell_base(token_dim: int) -> SpectralCell:
    """V=16, D=8, hidden=256, depth=2, 2 cross-attn."""
    return SpectralCell(token_dim, V=16, D=8, hidden=256, depth=2, n_cross=2, n_heads=4)

def spectral_cell_diamond(token_dim: int) -> SpectralCell:
    """V=16, D=16, hidden=256, depth=2, 1 cross-attn. Best sweep config."""
    return SpectralCell(token_dim, V=16, D=16, hidden=256, depth=2, n_cross=1, n_heads=4)


# ── Self-test ───────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for name, factory in [('tiny', spectral_cell_tiny),
                          ('small', spectral_cell_small),
                          ('diamond', spectral_cell_diamond)]:
        print(f"\n{'='*50}")
        cell = factory(token_dim=192).to(device)
        cell.summary()

        tokens = torch.randn(2, 16, 192, device=device)
        result = cell.format(tokens)

        print(f"\n  Input:  {tokens.shape}")
        print(f"  Output: {result['output'].shape}")
        print(f"  M:      {result['M'].shape}")
        print(f"  S:      {result['S'].shape}")
        print(f"  cm_d2:  {result['cm_d2'].shape}")
        print(f"  cm_vol2: {result['cm_vol2'].shape}")
        print(f"  trace:  {cell.trace_check(result['M']):.4f} (expect {cell.V})")
        print(f"  erank:  {cell.effective_rank(result['S_orig'].reshape(-1, cell.D)).mean():.2f}")
        print(f"  shift:  {cell.spectral_shift(result['S_orig'], result['S']):.6f}")

        # CM stats
        cm_stats = cell.cm_vol2_stats(result['cm_vol2'])
        print(f"  cm_vol: mean={cm_stats['mean']:.6f} cv={cm_stats.get('cv', 0):.4f} "
              f"valid={cm_stats['frac_valid']:.1%}")

        # Full CV (slower, samples 200 pentachora)
        with torch.no_grad():
            cv = cell.cm_cv(result['M'], n_samples=100)
        print(f"  cm_cv:  {cv:.4f}")

        # Gradient check
        loss = result['output'].sum()
        loss.backward()
        grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                      for p in cell.parameters() if p.requires_grad)
        print(f"  grads:  {'✓' if grad_ok else '✗'}")