"""aleph_model.py — the geolip-aleph-void model.

The natural evolution of PatchSVAE. The SVAE locks reconstruction into a deep
residual-MLP decoder; that accumulator carries recon, so the spherical latent M
is a "faux embedding" — recon flows through the MLP, not the geometry. AlephModel
keeps the SVAE's spherical encoder verbatim but makes the DECODER a pluggable
strategy, so M can be forced to carry reconstruction. This was validated: with a
single tied linear decoder, M reconstructs wikitext byte-trigram to cos≈0.9997
AND its extracted codebook addresses MORE sharply than the SVAE batteries
(aleph margin 0.967 vs 0.929), while staying projective-clean and becoming
markedly void-richer (betti2/axis ≈ 0.56 vs ≈ 0.08). So the aleph signed
address and the codebook void structure are both REAL on a codebook where the
geometry provably carries recon — the thing the faux-embedding SVAE could not
demonstrate.

Decoder strategies ("the avenues", vs the SVAE's MLP-only):
  'tied' : recon = single Linear(V*D -> patch_dim) of the sphere matrix M.
           No accumulator; M must linearly explain the patch. Recon-real,
           smallest. THE validated path.
  'dict' : recon = softmax code over a learned atom dictionary, code read off M.
           Sparse-coding; embedding-real by construction; non-conv/non-transformer.
  'mlp'  : the original SVAE deep residual decoder (decodes M_hat = U·S·Vt).
           Kept for direct comparison and SVAE continuity — selecting this
           recovers the faux-embedding regime on purpose.
  (room for 'rotor'/'cayley' — norm-preserving rotational decoders — later.)

Same forward contract as PatchSVAE:
    forward(images:(B,C,H,W)) -> {'recon':(B,C,H,W), 'svd':{U,S,S_orig,Vt,M}}
    svd['M']:(B,N,V,D) rows on S^(D-1) — the codebook source. So extract_codebook,
    the calibration registry, dataset bundles, and the training loop all work
    unchanged, and AlephModel codebooks are directly comparable to SVAE ones.

Readout (M -> U/S/Vt) is 'linear' for now (U=M, S=column norms, Vt=I — the
sphere-solver convention, svd_mode='none'); an 'svd' readout hook is reserved
for the geometric/Blackwell path and wired separately.

Reuses geolip_svae.model components so the encoder is byte-identical to the SVAE.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the SVAE's exact primitives — keeps the encoder identical and the patch
# grid / stitching consistent with every existing battery and dataset.
from geolip_svae.model import (
    extract_patches, stitch_patches, _row_normalize, BoundarySmooth,
    SpectralCrossAttention, ACTIVATIONS, ACTIVATION_MODULES,
)

ALEPH_MODEL_TYPE = "aleph"            # checkpoint marker (distinct from 'v1'/'v2')
DECODE_MODES = ("tied", "dict", "mlp")
ADDRESS_MODES = ("soft", "hard", "none")   # aleph address bottleneck; 'none' = recon-real AE (gate)
READOUTS = ("linear",)               # 'svd' reserved


class AlephModel(nn.Module):
    """geolip-aleph-void model: spherical encoder + learned projective codebook
    + aleph-address bottleneck + pluggable decoder.

    The aleph signed address is the core operation: each spherical M-row is
    addressed to a LEARNED codebook of K projective axes (the aleph logit — a
    softmax over 2K oriented axes), producing an addressed row M̂ that the decoder
    reconstructs from. Recon flows through the address into the codebook, so the
    logit is load-bearing. address='soft' (differentiable mixture) | 'hard'
    (straight-through discrete) | 'none' (M̂=M — recovers the recon-real tied
    autoencoder that gated this work). decode_mode controls how M̂ → patch.
    """

    MODEL_TYPE = ALEPH_MODEL_TYPE

    def __init__(self, V: int = 32, D: int = 4, ps: int = 4,
                 hidden: int = 64, depth: int = 1, channels: int = 3,
                 *,
                 address: str = "soft", K: int = 64, address_tau: float = 0.1,
                 decode_mode: str = "tied", n_atoms: int = 64, code_tau: float = 1.0,
                 readout: str = "linear", row_norm: str = "sphere",
                 n_cross: int = 0, n_heads: Optional[int] = None,
                 smooth_mid: Optional[int] = None, boundary_smooth: bool = True,
                 activation: str = "gelu", init_scheme: str = "orthogonal"):
        super().__init__()
        if decode_mode not in DECODE_MODES:
            raise ValueError(f"decode_mode must be in {DECODE_MODES}, got {decode_mode!r}")
        if address not in ADDRESS_MODES:
            raise ValueError(f"address must be in {ADDRESS_MODES}, got {address!r}")
        if readout not in READOUTS:
            raise ValueError(f"readout must be in {READOUTS} (got {readout!r}); "
                             "'svd' is reserved and wired separately")
        self.matrix_v = V
        self.D = D
        self.patch_size = ps
        self.channels = channels
        self.patch_dim = channels * ps * ps
        self.mat_dim = V * D
        self.decode_mode = decode_mode
        self.readout = readout
        self.row_norm_mode = row_norm
        self.n_atoms = n_atoms
        self.code_tau = code_tau
        self.activation_name = activation
        self.hidden = hidden
        self.depth = depth
        self.init_scheme = init_scheme
        self.boundary_smooth_on = bool(boundary_smooth)
        self.address = address
        self.n_axes = K
        self.address_tau = address_tau
        # emit (·,2K) aleph logits from forward only when needed. evaluate() runs
        # in eval mode (training=False) so logits/perplexity are available there;
        # set True to also emit during training (e.g. for the diversity term).
        self._emit_logits = False

        # ── learned aleph codebook: K projective axes in D-space ──
        # The address bottleneck snaps each M-row to this codebook (signed), so
        # recon flows through the address into the codebook — the aleph logit is
        # load-bearing, not a post-hoc read. address='none' disables it (M̂ = M),
        # recovering the recon-real tied autoencoder (the gate).
        if address != "none":
            self.codebook = nn.Parameter(torch.randn(K, D))
        else:
            self.register_parameter("codebook", None)

        if n_heads is None:
            n_heads = 2 if D <= 8 else min(4, D)
        if smooth_mid is None:
            smooth_mid = 16 if ps >= 16 else 8
        self.n_heads = n_heads          # resolved — stored for exact round-trip
        self.smooth_mid = smooth_mid

        inner_act = ACTIVATION_MODULES[activation]

        # ── encoder (byte-identical to PatchSVAE) ──
        self.enc_in = nn.Linear(self.patch_dim, hidden)
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                inner_act(),
                nn.Linear(hidden, hidden),
            ) for _ in range(depth)
        ])
        self.enc_out = nn.Linear(hidden, self.mat_dim)
        nn.init.orthogonal_(self.enc_out.weight)

        # ── decoder (the pluggable axis) ──
        if decode_mode == "mlp":
            # original SVAE deep residual accumulator (faux-embedding regime)
            self.dec_in = nn.Linear(self.mat_dim, hidden)
            self.dec_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    inner_act(),
                    nn.Linear(hidden, hidden),
                ) for _ in range(depth)
            ])
            self.dec_out = nn.Linear(hidden, self.patch_dim)
        elif decode_mode == "tied":
            self.dec = nn.Linear(self.mat_dim, self.patch_dim)
        else:  # 'dict'
            self.code_proj = nn.Linear(self.mat_dim, n_atoms)
            self.atoms = nn.Parameter(0.02 * torch.randn(n_atoms, self.patch_dim))

        # optional spectral cross-attention on S (default OFF; inert in practice)
        self.cross_attn = nn.ModuleList([
            SpectralCrossAttention(D, n_heads=n_heads) for _ in range(n_cross)
        ])

        self.boundary_smooth = (
            BoundarySmooth(channels=channels, mid=smooth_mid, activation=activation)
            if boundary_smooth else nn.Identity()
        )

    # ── aleph address: snap M-rows to the learned projective codebook ──
    def oriented_codebook(self) -> torch.Tensor:
        """The 2K oriented axes [+A; -A], A sign-canonicalized + unit. (2K, D)."""
        A = F.normalize(self.codebook, dim=-1)             # projective rep, unit rows
        return torch.cat([A, -A], dim=0)

    def aleph_address(self, M_rows: torch.Tensor, want_logits: bool = False):
        """Address each row (·, D) to the learned codebook. Returns (M_hat, logits).

        M_hat is the soft codebook reconstruction. The address is a softmax over
        the 2K oriented axes [+A; -A]; because the axes are antipodal, that mixture
        has a closed form that never materializes the 2K-wide tensor:

            M_hat = Σ_k 2·sinh(u_k)·A_k / Σ_k 2·cosh(u_k),   u = cos/τ

        computed stably by factoring out max|u| (so it holds for any τ). This is
        the hot path — only K-wide intermediates, no 2K softmax, no [+A;-A] concat.
        want_logits=True additionally returns the (·, 2K) oriented logits for
        diagnostics (perplexity/usage); skipped during training to avoid the large
        allocation. Differentiable; gradient trains the codebook via recon.
        """
        A = F.normalize(self.codebook, dim=-1)             # (K, D)
        cos = M_rows @ A.t()                               # (·, K) signed
        u = cos * (1.0 / self.address_tau)

        # stable antipodal soft mixture (exact softmax-over-[+A;-A], K-wide)
        m = u.abs().amax(dim=-1, keepdim=True)             # (·, 1) max oriented logit
        ep = torch.exp(u - m)                              # ∝ e^{+u}
        en = torch.exp(-u - m)                             # ∝ e^{-u}
        num = ep - en                                      # (·, K)  ∝ 2 sinh(u)
        den = (ep + en).sum(dim=-1, keepdim=True).clamp_min(1e-12)   # ∝ Σ 2 cosh(u)
        M_soft = (num @ A) / den                           # (·, D)

        if self.address == "hard":
            idx = cos.abs().argmax(dim=-1, keepdim=True)   # winner by |cos|
            win = cos.gather(-1, idx)                      # (·, 1) signed top
            M_hard = torch.sign(win) * A[idx.squeeze(-1)]  # (·, D) addressed axis
            M_hat = M_hard + (M_soft - M_soft.detach())    # straight-through
        else:
            M_hat = M_soft

        logits = torch.cat([cos, -cos], dim=-1) if want_logits else None
        return M_hat, logits

    # ── encode: patches -> spherical M, then aleph-address to the codebook ──
    def encode_patches(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _ = patches.shape
        act_fn = ACTIVATIONS[self.activation_name]
        h = act_fn(self.enc_in(patches.reshape(B * N, -1)))
        for block in self.enc_blocks:
            h = h + block(h)
        M = self.enc_out(h).reshape(B * N, self.matrix_v, self.D)
        M = _row_normalize(M, self.row_norm_mode)          # rows on S^(D-1)

        # aleph address bottleneck: M̂ = codebook reconstruction of M (load-bearing).
        # address='none' -> M̂ = M (the recon-real tied autoencoder / gate).
        # Logits (·,2K) are only built when needed (eval, or self._emit_logits for
        # the diversity term) — the training hot path skips that large allocation.
        aleph_logits = None
        if self.address != "none":
            want_logits = self._emit_logits or (not self.training)
            flat = M.reshape(B * N * self.matrix_v, self.D)
            M_hat_flat, logit_flat = self.aleph_address(flat, want_logits=want_logits)
            M_hat = M_hat_flat.reshape(B * N, self.matrix_v, self.D)
            if logit_flat is not None:
                aleph_logits = logit_flat.reshape(B, N, self.matrix_v, 2 * self.n_axes)
        else:
            M_hat = M

        # 'linear' readout on the addressed rows: U=M̂, S=col norms, Vt=I
        S = M_hat.norm(dim=-2)                             # (B*N, D)
        Vt = torch.eye(self.D, device=M.device, dtype=M.dtype).expand(
            B * N, self.D, self.D)

        out = {
            "U": M_hat.reshape(B, N, self.matrix_v, self.D),
            "S_orig": S.reshape(B, N, self.D),
            "Vt": Vt.reshape(B, N, self.D, self.D),
            "M": M.reshape(B, N, self.matrix_v, self.D),          # raw spherical rows
            "M_hat": M_hat.reshape(B, N, self.matrix_v, self.D),  # addressed rows (decode source)
        }
        if aleph_logits is not None:
            out["aleph_logits"] = aleph_logits                    # (B,N,V,2K) the address

        S_coord = out["S_orig"]
        for layer in self.cross_attn:
            S_coord = layer(S_coord)
        out["S"] = S_coord
        return out

    # ── decode: route recon through the ADDRESSED rows per the chosen strategy ──
    def decode_patches(self, svd: Dict[str, torch.Tensor]) -> torch.Tensor:
        Mh = svd["M_hat"]                                  # codebook-addressed rows
        B, N, V, D = Mh.shape
        if self.decode_mode == "mlp":
            U = svd["U"].reshape(B * N, V, D)
            S = svd["S"].reshape(B * N, D)
            Vt = svd["Vt"].reshape(B * N, D, D)
            M_hat = torch.bmm(U * S.unsqueeze(1), Vt)      # SVAE reconstruction (of M̂)
            act_fn = ACTIVATIONS[self.activation_name]
            h = act_fn(self.dec_in(M_hat.reshape(B * N, -1)))
            for block in self.dec_blocks:
                h = h + block(h)
            patch = self.dec_out(h)
        elif self.decode_mode == "tied":
            patch = self.dec(Mh.reshape(B * N, V * D))     # single linear of M̂
        else:  # 'dict'
            code = F.softmax(self.code_proj(Mh.reshape(B * N, V * D)) / self.code_tau, dim=-1)
            patch = code @ self.atoms
        return patch.reshape(B, N, -1)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = images.shape
        if C != self.channels:
            raise ValueError(f"input C={C} but model built channels={self.channels}")
        patches, gh, gw = extract_patches(images, self.patch_size)
        svd = self.encode_patches(patches)
        decoded = self.decode_patches(svd)
        recon = stitch_patches(decoded, gh, gw, self.patch_size, channels=self.channels)
        recon = self.boundary_smooth(recon)
        return {"recon": recon, "svd": svd}

    # ── provenance / checkpointing ──
    def get_config(self) -> dict:
        """Full reconstruction config (sufficient for build_aleph round-trip)."""
        return {
            "model_type": self.MODEL_TYPE,
            "V": self.matrix_v, "D": self.D, "ps": self.patch_size,
            "hidden": self.hidden, "depth": self.depth, "channels": self.channels,
            "address": self.address, "K": self.n_axes, "address_tau": self.address_tau,
            "decode_mode": self.decode_mode, "n_atoms": self.n_atoms,
            "code_tau": self.code_tau, "readout": self.readout,
            "row_norm": self.row_norm_mode, "n_cross": len(self.cross_attn),
            "n_heads": self.n_heads, "smooth_mid": self.smooth_mid,
            "boundary_smooth": self.boundary_smooth_on,
            "activation": self.activation_name, "init_scheme": self.init_scheme,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_aleph(config: dict) -> AlephModel:
    """Construct an AlephModel from a config dict (PatchSVAE-style keys)."""
    return AlephModel(
        V=config.get("V", config.get("matrix_v", 32)),
        D=config.get("D", 4),
        ps=config.get("ps", config.get("patch_size", 4)),
        hidden=config.get("hidden", 64),
        depth=config.get("depth", 1),
        channels=config.get("channels", 3),
        address=config.get("address", "soft"),
        K=config.get("K", config.get("n_axes", 64)),
        address_tau=config.get("address_tau", 0.1),
        decode_mode=config.get("decode_mode", "tied"),
        n_atoms=config.get("n_atoms", 64),
        code_tau=config.get("code_tau", 1.0),
        readout=config.get("readout", "linear"),
        row_norm=config.get("row_norm", "sphere"),
        n_cross=config.get("n_cross", config.get("n_cross_layers", 0)),
        n_heads=config.get("n_heads"),
        smooth_mid=config.get("smooth_mid"),
        boundary_smooth=config.get("boundary_smooth", True),
        activation=config.get("activation", "gelu"),
        init_scheme=config.get("init_scheme", "orthogonal"),
    )


def save_aleph_checkpoint(model: AlephModel, path: str, *,
                          epoch: Optional[int] = None,
                          test_mse: Optional[float] = None,
                          extra: Optional[dict] = None) -> None:
    """Write a checkpoint in the load_model-compatible format:
    {'config': <get_config>, 'model_state_dict', 'epoch', 'test_mse'}.
    geolip_svae.inference.load_model reconstructs it via build_aleph."""
    ckpt = {
        "config": model.get_config(),
        "model_state_dict": model.state_dict(),
        "epoch": epoch, "test_mse": test_mse,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


__all__ = ["AlephModel", "build_aleph", "save_aleph_checkpoint",
           "ALEPH_MODEL_TYPE", "DECODE_MODES", "ADDRESS_MODES"]