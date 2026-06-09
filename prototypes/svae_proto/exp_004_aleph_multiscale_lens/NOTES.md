# 004 — Aleph Multiscale Lens (reconstruction fidelity vs D_lens)

First experiment of the multiscale aleph-void series. The other two
(`exp_005_aleph_address_statistics`, `exp_006_aleph_void_scaling`) reuse the
same frozen-aleph source and D_lens ladder but own the address and topology
metric families respectively.

## Hypothesis

`geolip-aleph-void` ships an `AlephTransformer` macro shell (RESEARCH_HISTORY.md
§3.12): a *frozen* aleph battery is read at the detached stem boundary, lifted
through a **fixed isometric lens** `E = QR(randn(D_lens, D_base))`
(⟨Ex,Ey⟩=⟨x,y⟩, zero params) to a higher spectral dimension `D_lens`, modulated
by a bounded-α spectral attention stack, and decoded to an **external recon**.
Only the shell (transformer + decoder) trains; the aleph is frozen.

`D_lens` is the multiscale knob. The claim under test:

> Lifting the frozen `D_base=4` aleph address to a larger `D_lens` lets a small
> spectral shell reconstruct the substrate **better than the frozen aleph's own
> internal recon**, and that external-recon fidelity improves with `D_lens`
> until it saturates.

This is "expand aleph fidelity multiscale" measured as reconstruction. The lens
being an *exact* isometry is what makes "improves" well-defined — the shell
cannot leave the geometric envelope it inherits.

## What this experiment measures

A `D_lens` ladder `[4, 8, 16, 32, 64, 128, 256, 512]` (`SingleLens` requires
`D_lens ≥ D_base`, so 4 is the near-identity baseline). Per rung the shell is
trained fresh via the **core** `train_aleph_transformer(...)`
(`geolip_svae/train_aleph.py`), then we record:

- **`external_mse` / `external_cos`** — shell reconstruction (final
  `evaluate_transformer` row).
- **`internal_mse`** — the frozen aleph's own recon on the same eval set (the
  fixed `D_base` floor the shell must beat). Constant across rungs.
- **`omega_cv`** — column-norm CV of `M_lens` (should track the uniform-sphere
  value for `D_lens`; a canary if it drifts).
- **`mean_alpha`** — spectral-attention engagement, bounded in [0, 0.2]
  (sanity: the shell modulates, never injects content).
- **`lens_isometry_err`** — `‖EᵀE − I_{D_base}‖∞`; must be ~1e-5 (the lift is
  exact or the whole premise is void).
- **`shell_params`** — trainable shell size per rung.

Headline curve: `external_mse(D_lens)` and `external_cos(D_lens)` vs the
constant `internal_mse`.

## Success criteria (graduation bar)

- `lens_isometry_err < 1e-4` at every rung (hard gate — non-negotiable).
- `mean_alpha ∈ [0, 0.2]` at every rung (the shell stays in-envelope).
- The headline graduates this into a core `sweep_d_lens` helper iff
  `external_cos` is **monotonic-then-saturating** in `D_lens` and at least one
  rung's `external_mse` is **below** `internal_mse` — i.e. the multiscale lift
  demonstrably adds reconstruction fidelity over the frozen aleph alone.
- If `external_mse` is flat or worse than `internal_mse` across the whole
  ladder, the lift does not add reconstruction fidelity on this substrate —
  record that and stop (do not graduate).

## What this experiment does NOT change

- No edits to `geolip_svae/`. The shell, lens, spectral stack, trainer, and
  eval are all called from core. Deleting this directory leaves core intact.
- No new latent math: only the documented isometric lens + bounded-α stack +
  the frozen aleph's closed-form address. No `'svd'` readout, no
  `'rotor'/'cayley'` decoder.
- The frozen aleph is hosted, hard-address, byte-trigram. Not retrained here.

## Followups gated on results

- If fidelity improves with `D_lens`: exp_005 asks whether the **address
  statistics** (byte-recovery through `external_recon`, perplexity) improve with
  it, under the `stem ∈ {m_hat, m}` and `lens_sign ∈ {signed, canon}` ablations.
- exp_006 asks whether the **void/topology** of the lifted `M_lens` axis cloud
  (β₂/axis) grows with `D_lens`.
- A saturating `D_lens*` becomes the default lift for the SDXL-aleph conditioning
  path (RESEARCH_HISTORY.md §6).
