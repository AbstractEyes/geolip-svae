
from __future__ import annotations

"""
ConduitBattery V17 — standalone structural relational encoder/decoder.

Purpose
-------
A portable GeoLIP-style battery built from the useful parts of geolip-svae
and geolip-core, but restructured for inline use inside existing models.

Key differences from the original PatchSVAE:
    1. Decomposition happens on actual local feature matrices, not on an
       oversized MLP-invented surrogate matrix.
    2. Conduit evidence is promoted into a real utility channel for gating,
       relation building, and downstream analysis.
    3. The module supports both token sequences and convolutional feature maps.
    4. The decoder is lightweight and optional; the primary product is a
       transferable relational representation, not pixel-perfect reconstruction.

Typical use:
    battery = ConduitBattery(BaseConfig(input_dim=768, model_dim=768))
    out = battery(tokens)                   # tokens: (B, N, C)
    y = out["output"]                       # same shape as input
    z = out["battery_tokens"]               # coarse relational tokens
    g = out["global_token"]                 # pooled geometric summary

    battery = ConduitBattery(BaseConfig(input_dim=256, model_dim=256, conv_window=4))
    out = battery(feature_map)              # feature_map: (B, C, H, W)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BaseConfig
# ============================================================================

@dataclass
class BaseConfig:
    """Configuration for ConduitBattery."""

    input_dim: int
    model_dim: Optional[int] = None
    out_dim: Optional[int] = None

    rank: int = 16
    geom_dim: int = 64
    relation_depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0

    conv_window: int = 4
    token_group_size: int = 4
    token_group_mode: str = "sequence"  # 'sequence' or 'grid'
    prefix_tokens: int = 0

    max_spectral_shift: float = 0.20
    max_scale_shift: float = 0.10
    residual_init: float = 0.0

    use_row_norm: bool = True
    use_local_conv_bias: bool = True
    use_local_token_bias: bool = True

    use_conduit: bool = True
    use_conduit_for_spectrum: bool = True
    detach_conduit_dynamic: bool = True
    compute_dtype: str = "fp64"

    conv_pad_mode: str = "replicate"
    norm_eps: float = 1e-5
    ema_momentum: float = 0.99
    track_ema: bool = True

    def __post_init__(self) -> None:
        if self.model_dim is None:
            self.model_dim = self.input_dim
        if self.out_dim is None:
            self.out_dim = self.input_dim
        if self.rank < 2:
            raise ValueError(f"rank must be >= 2, got {self.rank}")
        if self.conv_window < 1:
            raise ValueError(f"conv_window must be >= 1, got {self.conv_window}")
        if self.token_group_size < 1:
            raise ValueError(f"token_group_size must be >= 1, got {self.token_group_size}")
        if self.token_group_mode not in {"sequence", "grid"}:
            raise ValueError(f"token_group_mode must be 'sequence' or 'grid', got {self.token_group_mode}")
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")


# ============================================================================
# Optional GeoLIP imports with graceful fallback
# ============================================================================

try:
    from geolip_core.linalg.conduit import FLEighConduit, ConduitPacket  # type: ignore
    print("Has conduit")
    _HAS_GEO_CONDUIT = True
except Exception:  # pragma: no cover - fallback for standalone usage
    print("Has no conduit, defaulting to invalid code")
    _HAS_GEO_CONDUIT = False

    @dataclass
    class ConduitPacket:
        eigenvalues: torch.Tensor
        eigenvectors: torch.Tensor
        char_coeffs: torch.Tensor
        friction: torch.Tensor
        settle: torch.Tensor
        extraction_order: torch.Tensor
        refinement_residual: torch.Tensor
        release_residual: Optional[torch.Tensor] = None

        def eigenpairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.eigenvalues, self.eigenvectors

    class FLEighConduit(nn.Module):  # pragma: no cover - fallback only
        def forward(self, A: torch.Tensor) -> ConduitPacket:
            eigenvalues, eigenvectors = torch.linalg.eigh(A)
            B, n, _ = A.shape
            char_coeffs = torch.zeros(B, n, dtype=A.dtype, device=A.device)
            friction = torch.zeros(B, n, dtype=A.dtype, device=A.device)
            settle = torch.zeros(B, n, dtype=A.dtype, device=A.device)
            extraction_order = torch.arange(n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1)
            eye = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0)
            refinement_residual = (torch.bmm(eigenvectors.transpose(-2, -1), eigenvectors) - eye).norm(dim=(-2, -1))
            return ConduitPacket(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                char_coeffs=char_coeffs,
                friction=friction,
                settle=settle,
                extraction_order=extraction_order,
                refinement_residual=refinement_residual,
            )


# ============================================================================
# BatteryState
# ============================================================================

@dataclass
class BatteryState:
    """Structured state returned by encode()."""

    input_kind: str
    grouped_shape: Tuple[int, int, int, int]  # (B, G, M, D)

    U: torch.Tensor
    S: torch.Tensor
    S_shifted: torch.Tensor
    Vh: torch.Tensor

    row_scale: torch.Tensor
    spectral_entropy: torch.Tensor
    effective_rank: torch.Tensor
    novelty: torch.Tensor

    geom_tokens: torch.Tensor
    content_tokens: torch.Tensor
    battery_tokens: torch.Tensor
    global_token: torch.Tensor

    conduit: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


# ============================================================================
# Primary bulk
# ============================================================================

class ConduitBattery(nn.Module):
    """Standalone structural relational battery.

    The battery accepts either:
        - token sequences: (B, N, C)
        - spatial features: (B, C, H, W)

    It builds local feature matrices from real model activations, performs thin
    SVD with conduit telemetry, constructs geometric tokens, relates them across
    groups, and returns:
        - an inline output with the same shape as the input
        - relational battery tokens for downstream utility
        - a pooled global geometric summary
        - structured conduit diagnostics
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Shared spectral utilities
        self.conduit_solver = FLEighConduit() if config.use_conduit else None
        self.register_buffer("ema_s", torch.full((config.rank,), 1.0 / config.rank))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))

        # Conv pathway
        self.conv_in_norm = nn.GroupNorm(1, config.input_dim, eps=config.norm_eps)
        self.conv_to_rank = nn.Conv2d(config.input_dim, config.rank, kernel_size=1, bias=False)
        self.conv_out = nn.Conv2d(config.rank, config.out_dim, kernel_size=1, bias=False)
        self.conv_local_bias = nn.Conv2d(
            config.rank,
            config.rank,
            kernel_size=3,
            padding=1,
            groups=config.rank,
            bias=False,
        ) if config.use_local_conv_bias else nn.Identity()

        # Token pathway
        self.token_in_norm = nn.LayerNorm(config.input_dim, eps=config.norm_eps)
        self.token_to_rank = nn.Linear(config.input_dim, config.rank, bias=False)
        self.token_out = nn.Linear(config.rank, config.out_dim, bias=False)
        self.token_local_bias = nn.Conv1d(
            config.rank,
            config.rank,
            kernel_size=3,
            padding=1,
            groups=config.rank,
            bias=False,
        ) if config.use_local_token_bias else nn.Identity()

        # Geometric feature projection
        self.geom_feature_dim = (7 * config.rank) + 7
        self.geom_proj = nn.Sequential(
            nn.LayerNorm(self.geom_feature_dim, eps=config.norm_eps),
            nn.Linear(self.geom_feature_dim, config.geom_dim),
            build_activation("gelu"),
            nn.Linear(config.geom_dim, config.geom_dim),
        )

        # Content feature projection
        self.content_feature_dim = (3 * config.rank) + 3
        self.content_proj = nn.Sequential(
            nn.LayerNorm(self.content_feature_dim, eps=config.norm_eps),
            nn.Linear(self.content_feature_dim, config.model_dim),
            build_activation("gelu"),
            nn.Linear(config.model_dim, config.model_dim),
        )

        # Relational stacks
        self.geom_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.geom_dim,
                    heads=min(config.num_heads, max(1, config.geom_dim)),
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    eps=config.norm_eps,
                )
                for _ in range(config.relation_depth)
            ]
        )
        self.content_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.model_dim,
                    heads=min(config.num_heads, max(1, config.model_dim)),
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    eps=config.norm_eps,
                )
                for _ in range(config.relation_depth)
            ]
        )

        # Cross-stream fusion
        self.geom_to_film = nn.Linear(config.geom_dim, 2 * config.model_dim)
        self.battery_proj = nn.Sequential(
            nn.LayerNorm(config.model_dim + config.geom_dim, eps=config.norm_eps),
            nn.Linear(config.model_dim + config.geom_dim, config.model_dim),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm((2 * config.model_dim) + config.geom_dim + config.rank, eps=config.norm_eps),
            nn.Linear((2 * config.model_dim) + config.geom_dim + config.rank, config.model_dim),
            build_activation("gelu"),
            nn.Linear(config.model_dim, config.model_dim),
        )

        # Spectral and scale modulation
        self.spectral_gate = SpectralShiftHead(
            context_dim=config.geom_dim,
            rank=config.rank,
            max_shift=config.max_spectral_shift,
        )
        self.scale_gate = BoundedScalarHead(
            context_dim=config.geom_dim,
            max_shift=config.max_scale_shift,
        )

        # Row-wise refinement on reconstructed local matrices
        self.row_refine = nn.Sequential(
            nn.LayerNorm(config.rank + config.model_dim + config.geom_dim, eps=config.norm_eps),
            nn.Linear(config.rank + config.model_dim + config.geom_dim, config.rank * 2),
            build_activation("gelu"),
            nn.Linear(config.rank * 2, config.rank),
        )

        # Output scaling for inline residual use
        self.output_gain = nn.Parameter(torch.tensor(float(config.residual_init)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        *,
        spatial_shape: Optional[Tuple[int, int]] = None,
        input_kind: str = "auto",
        update_ema: Optional[bool] = None,
    ) -> BatteryState:
        """Encode input into a structured battery state."""
        if input_kind == "auto":
            if x.ndim == 4:
                input_kind = "conv"
            elif x.ndim == 3:
                input_kind = "tokens"
            else:
                raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

        if update_ema is None:
            update_ema = self.training and self.config.track_ema

        if input_kind == "conv":
            groups, row_scale, meta = self._prepare_conv_groups(x)
        elif input_kind == "tokens":
            groups, row_scale, meta = self._prepare_token_groups(x, spatial_shape=spatial_shape)
        else:
            raise ValueError(f"input_kind must be 'auto', 'conv', or 'tokens', got {input_kind}")

        B, G, M, D = groups.shape
        assert D == self.config.rank, f"Expected local feature width {self.config.rank}, got {D}"

        working = groups
        if self.config.use_row_norm:
            working = groups / row_scale.unsqueeze(-1)

        U, S, Vh, packet = self._decompose_groups(working.reshape(B * G, M, D))
        U = U.reshape(B, G, M, D)
        S = S.reshape(B, G, D)
        Vh = Vh.reshape(B, G, D, D)
        packet = self._reshape_packet(packet, B, G)

        spectral_entropy = self._spectral_entropy(S)
        effective_rank = self._effective_rank(S)
        s_norm = self._safe_normalize_s(S)
        novelty = s_norm - self.ema_s.view(1, 1, -1)

        geom_features = self._make_geom_features(S, Vh, row_scale, spectral_entropy, effective_rank, novelty, packet)
        geom_tokens = self.geom_proj(geom_features)

        content_features = self._make_content_features(groups, row_scale, s_norm)
        content_tokens = self.content_proj(content_features)

        for geom_block, content_block in zip(self.geom_blocks, self.content_blocks):
            geom_tokens = geom_block(geom_tokens)
            gamma, beta = self.geom_to_film(geom_tokens).chunk(2, dim=-1)
            content_tokens = content_tokens * (1.0 + torch.tanh(gamma)) + beta
            content_tokens = content_block(content_tokens)

        battery_tokens = self.battery_proj(torch.cat([content_tokens, geom_tokens], dim=-1))
        S_shifted = self.spectral_gate(S, geom_tokens)

        group_s_summary = s_norm.mean(dim=1)
        global_token = self.global_proj(
            torch.cat(
                [
                    content_tokens.mean(dim=1),
                    battery_tokens.mean(dim=1),
                    geom_tokens.mean(dim=1),
                    group_s_summary,
                ],
                dim=-1,
            )
        )

        state = BatteryState(
            input_kind=input_kind,
            grouped_shape=(B, G, M, D),
            U=U,
            S=S,
            S_shifted=S_shifted,
            Vh=Vh,
            row_scale=row_scale,
            spectral_entropy=spectral_entropy,
            effective_rank=effective_rank,
            novelty=novelty,
            geom_tokens=geom_tokens,
            content_tokens=content_tokens,
            battery_tokens=battery_tokens,
            global_token=global_token,
            conduit=self._packet_to_summary(packet),
            meta=meta,
        )

        if update_ema:
            self._update_ema(s_norm)

        return state

    def decode(self, state: BatteryState, *, residual: bool = True) -> torch.Tensor:
        """Decode a BatteryState back to the original feature domain."""
        B, G, M, D = state.grouped_shape

        local = torch.matmul(state.U * state.S_shifted.unsqueeze(-2), state.Vh)
        scale = state.row_scale * self.scale_gate(state.geom_tokens)
        local = local * scale.unsqueeze(-1)

        geom_rows = state.geom_tokens.unsqueeze(2).expand(B, G, M, -1)
        content_rows = state.battery_tokens.unsqueeze(2).expand(B, G, M, -1)
        refine_in = torch.cat([local, content_rows, geom_rows], dim=-1)
        local = local + 0.1 * self.row_refine(refine_in)

        if state.input_kind == "conv":
            projected = self._unpack_conv_groups(local, state.meta)
            decoded = self.conv_out(projected)
            base = state.meta["input"]
        elif state.input_kind == "tokens":
            projected = self._unpack_token_groups(local, state.meta)
            decoded = self.token_out(projected)
            base = state.meta["input"]
        else:  # pragma: no cover - guarded earlier
            raise ValueError(f"Unknown input_kind {state.input_kind}")

        if residual:
            gain = torch.tanh(self.output_gain)
            return base + gain * decoded
        return decoded

    def forward(
        self,
        x: torch.Tensor,
        *,
        spatial_shape: Optional[Tuple[int, int]] = None,
        input_kind: str = "auto",
        return_state: bool = False,
        residual: bool = True,
        update_ema: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full inline pass."""
        state = self.encode(
            x,
            spatial_shape=spatial_shape,
            input_kind=input_kind,
            update_ema=update_ema,
        )
        output = self.decode(state, residual=residual)

        result: Dict[str, torch.Tensor] = {
            "output": output,
            "battery_tokens": state.battery_tokens,
            "geom_tokens": state.geom_tokens,
            "content_tokens": state.content_tokens,
            "global_token": state.global_token,
            "S": state.S,
            "S_shifted": state.S_shifted,
            "spectral_entropy": state.spectral_entropy,
            "effective_rank": state.effective_rank,
            "novelty": state.novelty,
        }
        if "group_hw" in state.meta:
            gh, gw = state.meta["group_hw"]
            result["analysis_grid"] = state.battery_tokens.transpose(1, 2).reshape(state.grouped_shape[0], self.config.model_dim, gh, gw)
            result["geom_grid"] = state.geom_tokens.transpose(1, 2).reshape(state.grouped_shape[0], self.config.geom_dim, gh, gw)
        if return_state:
            result["state"] = state
        return result

    def auxiliary_terms(self, state: BatteryState) -> Dict[str, torch.Tensor]:
        """Small training-friendly summaries and regularizers."""
        shift_l1 = (state.S_shifted - state.S).abs().mean()
        novelty_l1 = state.novelty.abs().mean()
        entropy_mean = state.spectral_entropy.mean()
        erank_mean = state.effective_rank.mean()

        out: Dict[str, torch.Tensor] = {
            "shift_l1": shift_l1,
            "novelty_l1": novelty_l1,
            "entropy_mean": entropy_mean,
            "effective_rank_mean": erank_mean,
        }
        if "refinement_residual" in state.conduit:
            out["conduit_refinement"] = state.conduit["refinement_residual"].mean()
        if "release_residual" in state.conduit:
            out["conduit_release"] = state.conduit["release_residual"].mean()
        return out

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare_conv_groups(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cfg = self.config
        B, C, H, W = x.shape
        ws = cfg.conv_window

        z = self.conv_in_norm(x)
        z = self.conv_to_rank(z)
        z = z + self.conv_local_bias(z)

        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        if pad_h or pad_w:
            z = F.pad(z, (0, pad_w, 0, pad_h), mode=cfg.conv_pad_mode)

        Hp, Wp = z.shape[-2:]
        gh, gw = Hp // ws, Wp // ws
        groups = z.reshape(B, cfg.rank, gh, ws, gw, ws)
        groups = groups.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(B, gh * gw, ws * ws, cfg.rank)

        row_scale = groups.norm(dim=-1).clamp(min=1e-6)

        meta = {
            "input": x,
            "pad_hw": (pad_h, pad_w),
            "orig_hw": (H, W),
            "group_hw": (gh, gw),
            "window": ws,
        }
        return groups, row_scale, meta

    def _prepare_token_groups(
        self,
        x: torch.Tensor,
        *,
        spatial_shape: Optional[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        cfg = self.config
        B, N_total, C = x.shape
        p = cfg.prefix_tokens

        prefix = x[:, :p] if p > 0 else None
        tokens = x[:, p:]

        z = self.token_in_norm(tokens)
        z = self.token_to_rank(z)

        if isinstance(self.token_local_bias, nn.Conv1d):
            z = z + self.token_local_bias(z.transpose(1, 2)).transpose(1, 2)

        if cfg.token_group_mode == "grid":
            h, w = self._resolve_grid(tokens.shape[1], spatial_shape)
            ws = cfg.token_group_size
            pad_h = (ws - (h % ws)) % ws
            pad_w = (ws - (w % ws)) % ws
            Hp, Wp = h + pad_h, w + pad_w

            z_grid = z.reshape(B, h, w, cfg.rank)
            if pad_h or pad_w:
                z_grid = F.pad(z_grid.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h))
                z_grid = z_grid.permute(0, 2, 3, 1)

            gh, gw = Hp // ws, Wp // ws
            groups = z_grid.reshape(B, gh, ws, gw, ws, cfg.rank)
            groups = groups.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, gh * gw, ws * ws, cfg.rank)

            meta = {
                "input": x,
                "prefix": prefix,
                "prefix_tokens": p,
                "mode": "grid",
                "orig_n": tokens.shape[1],
                "orig_hw": (h, w),
                "pad_hw": (pad_h, pad_w),
                "group_hw": (gh, gw),
                "window": ws,
            }
        else:
            gs = cfg.token_group_size
            N = z.shape[1]
            pad_n = (gs - (N % gs)) % gs
            if pad_n:
                z = torch.cat([z, torch.zeros(B, pad_n, cfg.rank, device=z.device, dtype=z.dtype)], dim=1)
            groups = z.reshape(B, (N + pad_n) // gs, gs, cfg.rank)

            meta = {
                "input": x,
                "prefix": prefix,
                "prefix_tokens": p,
                "mode": "sequence",
                "orig_n": N,
                "pad_n": pad_n,
                "group_size": gs,
            }

        row_scale = groups.norm(dim=-1).clamp(min=1e-6)
        return groups, row_scale, meta

    # ------------------------------------------------------------------
    # Decomposition and conduit
    # ------------------------------------------------------------------

    def _decompose_groups(self, A):
        cfg = self.config
        B, M, D = A.shape
        orig_dtype = A.dtype

        compute_dtype = torch.float64 if cfg.compute_dtype.lower() in {"fp64", "float64"} else torch.float32

        with torch.amp.autocast("cuda", enabled=False):
            A_c = A.to(compute_dtype)
            G = torch.bmm(A_c.transpose(1, 2), A_c)
            G = 0.5 * (G + G.transpose(-2, -1))
            G.diagonal(dim1=-2, dim2=-1).add_(1e-8 if compute_dtype == torch.float64 else 1e-5)

            packet = None

            if self.conduit_solver is not None:
                packet = self.conduit_solver(G.float())
                eigenvalues = packet.eigenvalues.to(compute_dtype).flip(-1)
                V = packet.eigenvectors.to(compute_dtype).flip(-1)
            else:
                eigenvalues, V = torch.linalg.eigh(G)
                eigenvalues = eigenvalues.flip(-1)
                V = V.flip(-1)

            S = torch.sqrt(eigenvalues.clamp(min=1e-24 if compute_dtype == torch.float64 else 1e-12))
            U = torch.bmm(A_c, V) / S.unsqueeze(1).clamp(min=1e-16 if compute_dtype == torch.float64 else 1e-8)
            Vh = V.transpose(-2, -1).contiguous()

        return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype), packet

    def _reshape_packet(
        self,
        packet: Optional[ConduitPacket],
        batch_size: int,
        groups: int,
    ) -> Optional[ConduitPacket]:
        if packet is None:
            return None

        def reshape_vec(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            if t.ndim == 1:
                return t.reshape(batch_size, groups)
            return t.reshape(batch_size, groups, *t.shape[1:])

        return ConduitPacket(
            eigenvalues=reshape_vec(packet.eigenvalues),
            eigenvectors=reshape_vec(packet.eigenvectors),
            char_coeffs=reshape_vec(packet.char_coeffs),
            friction=reshape_vec(packet.friction),
            settle=reshape_vec(packet.settle),
            extraction_order=reshape_vec(packet.extraction_order),
            refinement_residual=reshape_vec(packet.refinement_residual),
            release_residual=reshape_vec(packet.release_residual),
        )

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _make_geom_features(
        self,
        S: torch.Tensor,
        Vh: torch.Tensor,
        row_scale: torch.Tensor,
        spectral_entropy: torch.Tensor,
        effective_rank: torch.Tensor,
        novelty: torch.Tensor,
        packet: Optional[ConduitPacket],
    ) -> torch.Tensor:
        B, G, D = S.shape
        s_norm = self._safe_normalize_s(S)
        vh_diag = Vh.diagonal(dim1=-2, dim2=-1)
        total_v_energy = Vh.pow(2).sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        diag_v_energy = vh_diag.pow(2).sum(dim=-1, keepdim=True)
        vh_offdiag = (total_v_energy - diag_v_energy).clamp(min=0.0).sqrt()

        energy_mean = row_scale.mean(dim=-1, keepdim=True)
        energy_std = row_scale.std(dim=-1, keepdim=True, unbiased=False)
        energy_max = row_scale.amax(dim=-1, keepdim=True)

        if packet is None:
            friction = torch.zeros_like(S)
            settle = torch.zeros_like(S)
            order = torch.zeros_like(S)
            char_coeffs = torch.zeros_like(S)
            refine = torch.zeros(B, G, 1, device=S.device, dtype=S.dtype)
        else:
            friction = packet.friction.to(S.dtype)
            settle = packet.settle.to(S.dtype)
            order = packet.extraction_order.to(S.dtype)
            char_coeffs = packet.char_coeffs.to(S.dtype)
            refine = packet.refinement_residual.to(S.dtype).unsqueeze(-1)

            if self.config.detach_conduit_dynamic:
                friction = friction.detach()
                settle = settle.detach()
                order = order.detach()
                char_coeffs = char_coeffs.detach()
                refine = refine.detach()

        friction = torch.log1p(friction.clamp(min=0))
        settle = settle / (settle.amax(dim=-1, keepdim=True) + 1.0)
        order = order / max(1, self.config.rank - 1)
        char_coeffs = torch.tanh(char_coeffs)

        return torch.cat(
            [
                s_norm,
                vh_diag,
                torch.log1p(vh_offdiag),
                torch.log1p(energy_mean),
                energy_std,
                torch.log1p(energy_max),
                spectral_entropy.unsqueeze(-1),
                effective_rank.unsqueeze(-1) / float(self.config.rank),
                novelty,
                friction,
                settle,
                order,
                char_coeffs,
                refine,
            ],
            dim=-1,
        )

    def _make_content_features(
        self,
        groups: torch.Tensor,
        row_scale: torch.Tensor,
        s_norm: torch.Tensor,
    ) -> torch.Tensor:
        group_mean = groups.mean(dim=2)
        group_max = groups.amax(dim=2)
        energy = row_scale.mean(dim=-1, keepdim=True)
        energy_std = row_scale.std(dim=-1, keepdim=True, unbiased=False)
        energy_max = row_scale.amax(dim=-1, keepdim=True)
        return torch.cat([group_mean, group_max, s_norm, energy, energy_std, energy_max], dim=-1)

    # ------------------------------------------------------------------
    # Unpacking
    # ------------------------------------------------------------------

    def _unpack_conv_groups(self, groups: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        B, G, M, D = groups.shape
        gh, gw = meta["group_hw"]
        ws = meta["window"]

        z = groups.reshape(B, gh, gw, ws, ws, D)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous().reshape(B, D, gh * ws, gw * ws)

        pad_h, pad_w = meta["pad_hw"]
        H, W = meta["orig_hw"]
        if pad_h or pad_w:
            z = z[:, :, :H, :W]
        return z

    def _unpack_token_groups(self, groups: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        B, G, M, D = groups.shape
        prefix = meta["prefix"]
        prefix_tokens = meta["prefix_tokens"]

        if meta["mode"] == "grid":
            gh, gw = meta["group_hw"]
            ws = meta["window"]
            h, w = meta["orig_hw"]
            pad_h, pad_w = meta["pad_hw"]

            z = groups.reshape(B, gh, gw, ws, ws, D)
            z = z.reshape(B, gh * ws, gw * ws, D)
            if pad_h or pad_w:
                z = z[:, :h, :w]
            z = z.reshape(B, h * w, D)
            z = z[:, :meta["orig_n"]]
        else:
            z = groups.reshape(B, G * M, D)
            z = z[:, :meta["orig_n"]]

        if prefix_tokens > 0 and prefix is not None:
            prefix_proj = self.token_to_rank(self.token_in_norm(prefix))
            z = torch.cat([prefix_proj, z], dim=1)
        return z

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _packet_to_summary(self, packet: Optional[ConduitPacket]) -> Dict[str, torch.Tensor]:
        if packet is None:
            return {}
        out: Dict[str, torch.Tensor] = {
            "friction": packet.friction,
            "settle": packet.settle,
            "extraction_order": packet.extraction_order,
            "char_coeffs": packet.char_coeffs,
            "refinement_residual": packet.refinement_residual,
        }
        if packet.release_residual is not None:
            out["release_residual"] = packet.release_residual
        return out

    def _update_ema(self, s_norm: torch.Tensor) -> None:
        with torch.no_grad():
            mean_s = s_norm.mean(dim=(0, 1))
            if not bool(self.ema_initialized):
                self.ema_s.copy_(mean_s)
                self.ema_initialized.fill_(True)
            else:
                m = self.config.ema_momentum
                self.ema_s.mul_(m).add_(mean_s, alpha=1.0 - m)

    def _resolve_grid(self, n_tokens: int, spatial_shape: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if spatial_shape is not None:
            h, w = spatial_shape
            if h * w != n_tokens:
                raise ValueError(f"spatial_shape={spatial_shape} does not match token count {n_tokens}")
            return h, w

        root = int(math.isqrt(n_tokens))
        if root * root != n_tokens:
            raise ValueError(
                "token_group_mode='grid' requires spatial_shape or a square token count; "
                f"got {n_tokens}"
            )
        return root, root

    @staticmethod
    def _safe_normalize_s(S: torch.Tensor) -> torch.Tensor:
        return S / (S.sum(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _spectral_entropy(S: torch.Tensor) -> torch.Tensor:
        p = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        return -(p * p.log()).sum(dim=-1)

    @staticmethod
    def _effective_rank(S: torch.Tensor) -> torch.Tensor:
        p = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
        p = p.clamp(min=1e-8)
        return (-(p * p.log()).sum(dim=-1)).exp()


# ============================================================================
# Activations at the base
# ============================================================================

class TransformerBlock(nn.Module):
    """Compact transformer block for relational mixing."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float, eps: float):
        super().__init__()
        if dim % max(1, heads) != 0:
            heads = 1

        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        hidden = max(dim, int(dim * mlp_ratio))
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            build_activation("gelu"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class SpectralShiftHead(nn.Module):
    """Bounded modulation of singular values."""

    def __init__(self, context_dim: int, rank: int, max_shift: float):
        super().__init__()
        self.max_shift = float(max_shift)
        self.shift = nn.Linear(context_dim, rank)
        self.alpha_logits = nn.Parameter(torch.full((rank,), -2.0))

    def forward(self, S: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.shift(context))
        alpha = self.max_shift * torch.sigmoid(self.alpha_logits).view(1, 1, -1)
        return S * (1.0 + alpha * delta)


class BoundedScalarHead(nn.Module):
    """Positive scalar gate with bounded deviation from identity."""

    def __init__(self, context_dim: int, max_shift: float):
        super().__init__()
        self.max_shift = float(max_shift)
        self.proj = nn.Linear(context_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.max_shift * torch.tanh(self.proj(x))


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "silu":
        return nn.SiLU(inplace=False)
    raise ValueError(f"Unknown activation '{name}'")


__all__ = [
    "BaseConfig",
    "BatteryState",
    "ConduitBattery",
]