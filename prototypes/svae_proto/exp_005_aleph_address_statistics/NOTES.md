# 005 — Aleph Address Statistics under the Multiscale Lens

Second of the multiscale aleph-void series. Reuses exp_004's frozen-aleph source
and D_lens ladder; owns the **address / recovery** metric family.

## Hypothesis

exp_004 asks whether the multiscale lift improves *reconstruction*. This asks
whether it improves the **utilizable address statistics** — i.e. whether the
information the aleph address carries survives (and strengthens) through the
lift, measured as byte-faithful round-trip recovery.

RESEARCH_HISTORY.md makes two falsifiable claims about the shell's geometry
(§3.12): `lens_sign='signed'` (keep the per-row sign channel) reconstructs ≈8.78%
better than `'canon'` (drop sign onto ℝP), and `stem='m_hat'` (lift the
*addressed* direction) strengthens the address vs `stem='m'` (lift the raw rows,
the SVAE-equivalent control). The claim under test:

> Round-trip byte recovery through the shell's `external_recon` rises with
> `D_lens`, and is highest for `stem='m_hat', lens_sign='signed'` — confirming
> the lift carries the address, not just a reconstruction.

## What this experiment measures

The byte-trigram substrate is exactly invertible (`bytes_to_image` /
`image_to_bytes`), so recovery is a real metric, not a proxy.

**Per rung (and per ablation arm):** a fixed evaluation passage is encoded to a
byte-trigram image, pushed through the trained `AlephTransformer`, and the
**`external_recon`** is decoded back to bytes (NOVEL: core's
`text_recovery_metrics` reads the frozen aleph's `out['recon']` via
`engine.reconstruct`; here we round-trip the *shell's* recon to measure recovery
*through the lift*). Reported: `real_byte_acc`, `real_byte_l1`, a recovered-text
snippet, plus `external_mse`/`external_cos` from the trainer.

**Fixed `D_base` baselines (computed once):**
- the frozen aleph's own recovery via core `text_recovery_metrics` — the floor;
- the aleph's address health via `_address_stats`: soft/hard **perplexity**
  (effective oriented axes in use, of 2K) and address margin — confirms the
  codebook is alive (hosted runs sit at soft ≈125–126 / hard ≈112–122 of 128)
  and that the shell never touches it (the aleph is frozen).

**Sweep shape (Colab-sane):** the primary arm `(m_hat, signed)` runs the full
ladder; the other three ablation arms `(m, signed)`, `(m_hat, canon)`,
`(m, canon)` run at a single `ablate_d_lens` (default 64). That is
`len(ladder) + 3` trainings, not the full 2×2×ladder grid.

## Success criteria (graduation bar)

- The two documented ablation directions reproduce at the ablation scale:
  `signed ≥ canon` and `m_hat ≥ m` on `real_byte_acc` (and `external_cos`).
- Shell recovery is **monotonic-then-saturating** in `D_lens` for the primary
  arm, and meets or beats the frozen-aleph baseline recovery at the saturating
  rung. If so, graduate a `shell_byte_recovery` helper into core's inference
  text tooling.
- Perplexity baselines are reported but not optimized — they are a frozen-aleph
  property; this experiment only confirms the shell does not degrade them
  (it cannot — the aleph is frozen).

## What this experiment does NOT change

- No core edits; the aleph is frozen; only the shell trains. Byte primitives,
  trainer, `text_recovery_metrics`, and `_address_stats` are all called from core.
- No new latent math (documented lens + bounded-α + closed-form address only).
- The frozen aleph is hosted, hard-address, byte-trigram (matched substrate).

## Followups gated on results

- A confirmed `signed/m_hat` win + recovery-vs-`D_lens` curve fixes the default
  shell config for the SDXL-aleph conditioning path.
- exp_006 measures whether the *void/topology* of the lifted `M_lens` cloud
  scales the same way the recovery does.
