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
READOUTS = ("linear",)               # 'svd' reserved


class AlephModel(nn.Module):
    """geolip-aleph-void model: spherical encoder + pluggable decoder.

    AlephModel(decode_mode='tied', ...) is the validated recon-real configuration
    (the artifact previously prototyped as GeoSphereVAE), now a first-class model.
    """

    MODEL_TYPE = ALEPH_MODEL_TYPE

    def __init__(self, V: int = 32, D: int = 4, ps: int = 4,
                 hidden: int = 64, depth: int = 1, channels: int = 3,
                 *,
                 decode_mode: str = "tied", n_atoms: int = 64, code_tau: float = 1.0,
                 readout: str = "linear", row_norm: str = "sphere",
                 n_cross: int = 0, n_heads: Optional[int] = None,
                 smooth_mid: Optional[int] = None, boundary_smooth: bool = True,
                 activation: str = "gelu", init_scheme: str = "orthogonal"):
        super().__init__()
        if decode_mode not in DECODE_MODES:
            raise ValueError(f"decode_mode must be in {DECODE_MODES}, got {decode_mode!r}")
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

    # ── encode: patches -> spherical M + omega token S (sphere-solver readout) ──
    def encode_patches(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _ = patches.shape
        act_fn = ACTIVATIONS[self.activation_name]
        h = act_fn(self.enc_in(patches.reshape(B * N, -1)))
        for block in self.enc_blocks:
            h = h + block(h)
        M = self.enc_out(h).reshape(B * N, self.matrix_v, self.D)
        M = _row_normalize(M, self.row_norm_mode)          # rows on S^(D-1)

        # 'linear' readout convention (svd_mode='none'): U=M, S=col norms, Vt=I
        U = M
        S = M.norm(dim=-2)                                 # (B*N, D)
        Vt = torch.eye(self.D, device=M.device, dtype=M.dtype).expand(
            B * N, self.D, self.D)

        U = U.reshape(B, N, self.matrix_v, self.D)
        S = S.reshape(B, N, self.D)
        Vt = Vt.reshape(B, N, self.D, self.D)
        M = M.reshape(B, N, self.matrix_v, self.D)

        S_coord = S
        for layer in self.cross_attn:
            S_coord = layer(S_coord)

        return {"U": U, "S_orig": S, "S": S_coord, "Vt": Vt, "M": M}

    # ── decode: route recon through the geometry per the chosen strategy ──
    def decode_patches(self, svd: Dict[str, torch.Tensor]) -> torch.Tensor:
        M = svd["M"]
        B, N, V, D = M.shape
        if self.decode_mode == "mlp":
            U = svd["U"].reshape(B * N, V, D)
            S = svd["S"].reshape(B * N, D)
            Vt = svd["Vt"].reshape(B * N, D, D)
            M_hat = torch.bmm(U * S.unsqueeze(1), Vt)      # SVAE reconstruction
            act_fn = ACTIVATIONS[self.activation_name]
            h = act_fn(self.dec_in(M_hat.reshape(B * N, -1)))
            for block in self.dec_blocks:
                h = h + block(h)
            patch = self.dec_out(h)
        elif self.decode_mode == "tied":
            patch = self.dec(M.reshape(B * N, V * D))      # single linear of M
        else:  # 'dict'
            code = F.softmax(self.code_proj(M.reshape(B * N, V * D)) / self.code_tau, dim=-1)
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
           "ALEPH_MODEL_TYPE", "DECODE_MODES"]