# 006 — Aleph Void Scaling under the Multiscale Lens

Third of the multiscale aleph-void series. Reuses exp_004's frozen-aleph source
and D_lens ladder; owns the **void / topology** metric family.

## Hypothesis

RESEARCH_HISTORY.md establishes (discoveries #20, #23) that **H2 voids fingerprint
the substrate**: at fixed latent dimension, symbolic codebooks are void-rich and
continuous ones void-sparse, measured by persistent homology on ℝP^(D−1); the
aleph's learned codebook lands at β₂/axis ≈ 0.56, ~7× its SVAE ancestors. The
multiscale lens lifts the addressed rows to a larger `D_lens`. The claim:

> The void structure of the **lifted** `M_lens` axis cloud is preserved (and may
> be amplified) by the multiscale lift — β₂/axis stays finite and trends with
> `D_lens` rather than washing out — so the higher-dimensional spectral
> statistics remain a usable substrate fingerprint.

This is "fidelity into the utilizable statistics environment" measured
topologically: the lift must not destroy the void signal the program relies on
to tell symbolic substrates apart.

## What this experiment measures

Per `D_lens` rung the shell is trained (primary arm `stem='m_hat',
lens_sign='signed'`), then a batch of byte-trigram images is pushed through it to
collect `M_lens` (B,N,V,D_lens). NOVEL piece: core `extract_codebook` /
`create_codebook` operate on the frozen aleph's `M` at `D_base`; void *scaling*
needs the topology of the **lifted** cloud, so we:

1. aggregate `M_lens` → a `[V, D_lens]` codebook (mean over B·N rows);
2. antipodal-collapse to axes (`identify_antipodal_pairs` + `collapse_to_axes`);
3. run the core topology probes (`run_topology_analysis`) on those `D_lens` axes.

Recorded per rung (from `TopologyReport`):
- **`beta2_per_axis`** — `persistence_n_finite['H2'] / n_axes` (the void metric);
- `beta1`, `n_axes`, `percolation_thresh_deg`, `local_dim_pr_p50` (intrinsic dim
  proxy), `ripser_available`;
- **`deviation`** of the lifted axes from uniform ℝP^(D_lens−1)
  (`codebook_mean_projective_angle − uniform_projective_angle`) + statute class
  (uniform vs polytope).

Headline curve: **β₂/axis vs `D_lens`**.

## Success criteria (graduation bar)

- With ripser available, `beta2_per_axis` is **finite and > 0** on the
  byte-trigram (symbolic) lifted cloud at the mid/high rungs — the void signal
  survives the lift. If it collapses to 0 across the ladder, the lift destroys
  the fingerprint (record and stop).
- Report whether β₂/axis **grows, plateaus, or decays** with `D_lens`; a clear
  monotone-then-saturating trend graduates a `lifted_codebook_topology` helper
  into core's topology tooling.
- Degrade gracefully when ripser is absent: kNN connectivity (`percolation`) and
  PCA intrinsic dim still report; β₂ is marked unavailable, not silently 0.

## What this experiment does NOT change

- No core edits. Collapse helpers, topology probes, and the trainer are called
  from core; `M_lens` is read from the public `AlephTransformer.forward()` output.
- No new latent math; only the documented lens + the deterministic
  antipodal-collapse + the existing ripser/kNN/PCA probes.
- ripser is an OPTIONAL dep (`exp_006` extra); the experiment runs without it.

## Followups gated on results

- A confirmed β₂/axis-vs-`D_lens` curve, cross-referenced with exp_004
  (reconstruction) and exp_005 (recovery), answers whether reconstruction
  fidelity, address recovery, and void richness all scale together with `D_lens`
  — the joint "utilizable statistics" verdict for the series.
- If voids amplify with `D_lens`, the lifted cloud becomes the preferred
  substrate-discrimination probe over the `D_base` codebook.
