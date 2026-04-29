# Omega-class battery potentials — exhaustive catalog

Compiled 2026-04-29 from full scratchpad survey across sessions 000080-000113. Includes every documented config across all sweeps (P/Q/R/S/T plus the A-set verification probes), every published HF model, and the trained substrate prototypes from this week. Both Adam and LBFGS candidates are listed where the sweep ran both.

## Definition of omega-class

A trained or candidate battery is **omega-class** if it satisfies all four:

1. **Sphere-solver architecture**: PatchSVAE with sphere-norm M tensor, V rows on S^(D-1), output basis on ℝP^(D-1)
2. **Projective-clean codebook**: |deviation from uniform RP^(D-1) baseline| < 0.05, secondary antipodal pair count ≤ 3, axis utilization > 0.95
3. **In its natural CV band** for arch class (table below)
4. **Codebook-engaged or sphere-engaged**: cross-attn coupling moves off floor, recovery curve from random init, geometric stats leave passthrough signature

A "potential" is any trained instance OR sweep candidate that has run through the architecture and produced measurements against the criterion.

## Natural CV bands by architecture class

| arch class | V | D | natural CV band | source |
| --- | --- | --- | --- | --- |
| noise-substrate (Freckles, Johanna) | 256 | 16 | 0.20-0.23 (sweet spot), 0.13-0.30 (band) | ft1 attractor, ablation Phase 1 |
| h2-class | 32 | 4 | 0.80-1.05 | h2-64 measurement, 000111-112 |
| P-class | 32 | 3 | 0.029-0.036 (LOW band) | Q-sweep ranks 06/07/09 |
| Phase T D=5 V=16 | 16 | 5 | ~0.04 mean dev | Phase T sweet spot |
| Phase T D=5 V=32 | 32 | 5 | varies by noise (partial) | Phase T (qualified ft2 D=5 claim) |

## Diagnostic signature (for testing any candidate)

| dimension | passthrough | engaged (omega-class) |
| --- | --- | --- |
| α (cross-attn coupling) | stationary [0.020, 0.030] | rising monotonically off floor |
| α\_std/α\_mean | flat ~0.06 | climbs (>0.30 in byte-trigram) |
| row_cv | in arch's natural band | leaves natural band |
| ratio S0/SD | ≈ 1.0 (flat spectrum) | drifts (>1.05) |
| erank | flat at full rank | dips below |
| recovery from random init | near-100% from ep 1 (sign-recovery trivial) | curve from ~0% upward |
| codebook deviation | undefined / not measured | within ±0.05 of uniform RP^(D-1) |

---

## TIER 1 — Trained, verified omega-class on HuggingFace

| HF path | arch class | params | D | V | optimizer | natural CV | n_axes | dev | training content | verified by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `AbstractPhil/geolip-SVAE/v40` (Freckles 64×64) | Freckles | 2,557,539 | 4 | 48 | Adam | ~0.20-0.23 | — | <0.05 | 16-noise mixture | ft1, U5 cross-band |
| `AbstractPhil/geolip-SVAE/v41` (Freckles 256×256) | Freckles | 2,557,539 | 4 | 48 | Adam | same | — | — | resolution-scaled v40 | continuation of v40 |
| `AbstractPhil/geolip-SVAE/v50_fresnel_64` | Fresnel-base | 16,942,419 | 4 | — | Adam | — | — | — | 140M+ ImageNet random crops, sublens | streaming run, "phenomenal MSE recon" |
| `AbstractPhil/geolip-svae-h2-64` (192-bank array) | H2_linear_matched | 57,215 × 64 batteries × 3 phases | 4 | 32 | Adam | 0.80-1.05 | 24-27/bank | +0.010 mean, ±0.013 std | per-bank, see Tier 1a | A2 probe (ft2), 192-bank cosine sweep (000109) |
| `AbstractPhil/geolip-svae-implicit-solver-experiments/G-Cand` | H2-class | 28,899-45,852 | 3 | 32 | Adam | 0.03 (LOW) | 22 (10 pairs + 12 unpaired) | −0.004 | gaussian-only | A0 probe (ft2) |
| `AbstractPhil/geolip-svae-implicit-solver-experiments/H2a` | H2_linear_matched | 40,227-57,215 | 4 | 32 | Adam | 0.80-1.10 | 26 (6 pairs + 20 unpaired) | +0.002 | gaussian-only | A1 probe (ft2) |
| `AbstractPhil/geolip-svae-implicit-solver-experiments/A3` (3 runs, qualified) | H2 variant | varies | 5 | 16/32/64 | Adam | varies | 16/29/51 | −0.015 / +0.016 / +0.019 | gaussian-only single-arch | A3 probe (ft2). **QUALIFIED by 000106** — single-arch single-noise. Phase T showed D=5 partial. |
| `AbstractPhil/geolip-SVAE/byte_trigram_proto_v1` | h2-class | 57,215 | 4 | 32 | Adam | left band ep 6 → 1.31 | TBD | TBD | wikitext-103 byte trigrams | 000113 — engaged signature confirmed; codebook investigation pending |

## TIER 1a — h2-64 array decomposition (192 banks = 64 batteries × 3 phases)

All 16 single-noise experts (Group 1) verified PROJECTIVE-CLEAN in A2. Remaining 48 batteries share architecture and were measured in 192-bank cosine sweep but not individually probed against projective threshold.

### Group 1 — 16 single-noise experts (banks 0-15)

| bank | noise type | special | training distribution |
| --- | --- | --- | --- |
| 0 | gaussian | universal-pull center | standard normal noise |
| 1 | uniform | — | uniform[-1,1] noise |
| 2 | uniform_scaled | — | scaled uniform |
| 3 | poisson | — | poisson-distributed |
| 4 | pink | clone-pair with bank 5 | 1/f spectrum |
| 5 | brown | clone-pair with bank 4 | 1/f² spectrum |
| 6 | salt_pepper | hardest reconstruction (S-sweep), cleanest projective at D=5 (Phase T) | impulse-style |
| 7 | sparse_impulses | cosine outlier — heavy-tailed | sparse extreme-value |
| 8 | block_upsampled | — | block-correlated |
| 9 | gradient_gaussian | **most isolated battery on S³** — only non-stationary noise | spatial gradient |
| 10 | checker | structured noise | checkerboard |
| 11 | gauss_uniform_mix | — | mixture |
| 12 | four_quadrant | 0% projective-clean across all Phase T archs | structured |
| 13 | cauchy | heavy-tailed | cauchy-distributed |
| 14 | exponential | — | exponential |
| 15 | laplace | heavy-tailed | laplace-distributed |

### Group 2 — gaussian+one pairs + generalist (banks 16-31)

15 pair-trained banks (gaussian + each of noises 1-15), plus 1 generalist trained on all 16. Notable: pairs (19, 20) = (gaussian+pink, gaussian+brown) are clone-pair on S³ for the same noise-family adjacency reason as (4, 5). Pair (16, 26) = (gaussian+uniform, gaussian+gauss_uniform_mix) is a clone-pair (gaussian dominates the residual; both bounded distributions).

### Group 3 — gaussian-balanced quads (banks 32-47)

16 (gaussian, easy, medium, hard) covers via stride-7 deterministic enumeration over the EASY (uniform/uniform_scaled/cauchy/exponential/laplace) × MEDIUM (poisson/salt_pepper/sparse_impulses/gauss_uniform_mix) × HARD (pink/brown/block_upsampled/gradient_gaussian/checker/four_quadrant) product. All contain gaussian.

### Group 4 — no-gaussian quads (banks 48-63)

16 (easy, medium, hard, hard) covers via stride-19 deterministic enumeration. **No gaussian seen during training.** Banks 48 and 51 are kNN top-5 outliers because they solve the sphere problem from a fundamentally different starting distribution than Groups 1-3 (which all see gaussian as universal pull-toward-interior).

---

## TIER 2 — Q-sweep candidates (10 top-P configs × 1000 batches)

After the LBFGS Hessian-corruption fix (000099), the Q-sweep ran clean on 10/10 configs. **All 10 are listed; both Adam and LBFGS variants where present.**

| Q-rank | variant | params | optim | G-MSE | CV | D | V | depth | n_cross | class | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | Q_rank01_h64_V32_D4_dp1_nx0_lbfgs | 57,123 | LBFGS | 0.00421 | 0.954 | 4 | 32 | 1 | 0 | **H2a** | LBFGS Q-sweep best |
| 02 | Q_rank02_h64_V32_D4_dp0_nx0_adam | **40,227** | Adam | **0.00205** | 0.862 | 4 | 32 | 0 | 0 | **H2a** | **Smallest H2a, canonical sphere-solver. ≈ 1 h2-64 bank's capacity at 70% the params.** |
| 03 | Q_rank03_h64_V32_D4_dp0_nx1_adam | 40,319 | Adam | 0.00250 | 0.890 | 4 | 32 | 0 | 1 | **H2a** | Adam-vs-LBFGS twin of rank 04 |
| 04 | Q_rank04_h64_V32_D4_dp0_nx1_lbfgs | 40,319 | LBFGS | 0.00391 | 0.893 | 4 | 32 | 0 | 1 | **H2a** | LBFGS twin of rank 03; Adam wins 36% lower MSE |
| 05 | Q_rank05_h64_V16_D4_dp1_nx1_lbfgs | 36,607 | LBFGS | 0.03117 | 1.069 | 4 | 16 | 1 | 1 | **H2b** | Only V=16 candidate; CV slightly above HIGH ceiling. **Underexplored size class.** |
| 06 | Q_rank06_h64_V32_D3_dp1_nx1_adam | 45,852 | Adam | 0.02497 | 0.029 | 3 | 32 | 1 | 1 | **P-class (D=3)** | LOW-band; originally framed "polynomial," now confirmed projective-clean on RP² |
| 07 | Q_rank07_h64_V32_D3_dp0_nx1_adam | 28,956 | Adam | 0.03151 | 0.036 | 3 | 32 | 0 | 1 | **P-class (D=3)** | Smaller P-class variant |
| 08 | Q_rank08_h64_V32_D4_dp1_nx1_adam | **57,215** | Adam | 0.00231 | 0.960 | 4 | 32 | 1 | 1 | **H2a** | **Exact h2-64 single-bank arch** (depth=1+n_cross=1) |
| 09 | Q_rank09_h64_V32_D3_dp0_nx0_adam | **28,899** | Adam | 0.02782 | 0.035 | 3 | 32 | 0 | 0 | **P-class (D=3)** | **Smallest projective-clean omega-class candidate. Under 30K params.** 30% smaller than H2a at ~14× MSE cost. |
| 10 | Q_rank10_h64_V32_D2_dp0_nx1_adam | 19,649 | Adam | 0.16139 | 0.000 | 2 | 32 | 0 | 1 | **EXCLUDED** | D=2 cannot form pentachoron (needs ≥5 points), CV undefined |

**Per-architecture optimizer comparison** (where direct twin exists):

| arch | Adam Q-rank | Adam MSE | LBFGS Q-rank | LBFGS MSE | winner |
| --- | --- | --- | --- | --- | --- |
| h64_V32_D4_dp0_nx0 | 02 | 0.00205 | (no twin) | — | Adam |
| h64_V32_D4_dp0_nx1 | 03 | 0.00250 | 04 | 0.00391 | **Adam by 36%** |
| h64_V32_D4_dp1_nx0 | (no twin) | — | 01 | 0.00421 | LBFGS only |
| h64_V32_D4_dp1_nx1 | 08 | 0.00231 | (no twin) | — | Adam |

**Optimizer regime guidance** (000100, post-LBFGS-fix): Adam @ lr=3e-3 dominates at 1000-batch budgets. LBFGS retains niche for short-budget probing (≤100 batches) and floor-finding sweeps. For sphere-solver canonical training: **Adam is the recommended default** since 2026-04-24.

---

## TIER 3 — P-sweep parent grid (600 configs at 20 batches)

The Q-sweep was the top-10 from this. **Full grid: 5 × 5 × 3 × 2 × 2 × 2 = 600 configs.**

| axis | values | count |
| --- | --- | --- |
| hidden | {4, 8, 16, 32, 64} | 5 |
| V | {2, 4, 8, 16, 32} | 5 |
| D | {2, 3, 4} | 3 |
| depth | {0, 1} | 2 |
| n_cross | {0, 1} | 2 |
| optimizer | {Adam, LBFGS} | 2 |

**Pins**: H2_linear_matched baseline (svd='none', linear_readout=True, match_params=True), HIGH band (patch_size=4, img_size=64), batch_size=256, batch_limit=20, n_heads=1 (D varies are gone), grad_clip=1.0, soft_hand=False. Adam uses lr=3e-3; LBFGS uses lr=1.0 (default unit-Wolfe-step).

**Outcomes by optimizer**:

* **Adam configs** (300 total): all converged finite. Top 6 of Q-sweep are Adam (ranks 02, 03, 06, 07, 08, 09).
* **LBFGS configs** (300 total): **9 NaN/diverged** (the 000099 Hessian-corruption casualties — ALL 9 NaNs in the original P-sweep were LBFGS configs at depth=1+n_cross=1, exact bug profile). The 9 NaN configs were never re-run with the corrected trainer (parked open item from 000100). Surviving LBFGS configs: 4 in Q-sweep top 10 (ranks 01, 04, 05).

**Geometric attractor split observed in P-sweep + confirmed in Q**:

* D=4 configs → HIGH-band (CV 0.86-1.07) sphere-solver attractor (H2 family)
* D=3 configs → LOW-band (CV ~0.03) projective-clean attractor (P-class)
* D=2 configs → no pentachoron, fails geometric validity test

---

## TIER 4 — R-sweep polytope packing test (3 configs)

Test of the natural-axis-count hypothesis: V matched to known polytope vertex counts on S^(D-1) should produce **static** sphere-solver rows (no rotating antipodal frame).

| variant | V | D | polytope | predicted | params |
| --- | --- | --- | --- | --- | --- |
| R_h64_V16_D4_16cell_orthoplex_adam | 16 | 4 | 16-cell (4-orthoplex) | H2-LIKE static | — |
| R_h64_V8_D4_8cell_or_16cell_subset_adam | 8 | 4 | 8-cell (tesseract) | H2-LIKE static | — |
| R_h64_V20_D3_dodecahedron_adam | 20 | 3 | dodecahedron | H2-LIKE static | — |

**Pins**: same H2_linear_matched baseline as Q, Adam @ lr=3e-3, depth=0, n_cross=0, 1000 batches, gaussian-only training, 16-noise per-noise test.

**Status**: trained (in `phaseR_reports/` on HF), results not surfaced into the projective-clean catalog yet. Worth probing against the omega-class criterion since natural-axis-count framework predicts they should land cleanly.

---

## TIER 5 — Phase S D=5 architecture floor map (1600 configs at 20 batches)

| axis | values | count |
| --- | --- | --- |
| hidden | {4, 8, 16, 32, 64} | 5 |
| V | {2, 4, 8, 16, 32} | 5 |
| D | {5} | 1 |
| depth | {0, 1} | 2 |
| n_cross | {0, 1} | 2 |
| noise_type | {0..15} | 16 |
| optimizer | {Adam} | 1 (LBFGS too slow for sweep) |

**Total**: 5 × 5 × 1 × 2 × 2 × 16 × 1 = 1600 runs.

**Headline finding (000105)**: cross-noise rank correlation +0.954. Architectures rank near-identically across all 16 noise types — what changes per noise is achievable floor MSE, not which model achieves it. Top-4 universal architectures all hidden=64, V=32.

**Top-4 architectures from S analysis** (mean rank across 16 noise types):

| rank | architecture | mean rank |
| --- | --- | --- |
| 1 | h64_V32_dp0_nx1_D5 | 1.1 |
| 2 | h64_V32_dp1_nx0_D5 | (close to 1.1) |
| 3 | h64_V32_dp1_nx1_D5 | 1.9 |
| 5 | h64_V16_dp1_nx1_D5 | (the V<32 entry) |

**Note**: the 1391 individual config directories were lost to HF rate-limiting (87% of submitted commits failed). The 1600-config aggregate JSON survived; per-config artifacts mostly did not. Engineering invariant logged (000108): batch-sync uploads from this point forward.

---

## TIER 6 — Phase T D=5 convergence sweep (64 configs at 1000 batches)

Top-4 S architectures × 16 noise types, run at A3-reference budget. **The D=5 walk-back (000106) lives here.**

| arch | hidden | V | depth | n_cross | optimizer | % projective-clean across 16 noises |
| --- | --- | --- | --- | --- | --- | --- |
| h64_V16_dp1_nx1 | 64 | 16 | 1 | 1 | Adam | **62% (10/16)** — D=5 sweet spot |
| h64_V32_dp0_nx1 | 64 | 32 | 0 | 1 | Adam | 50% (8/16) |
| h64_V8_dp1_nx0 | 64 | 8 | 1 | 0 | Adam | 25% (4/16) |
| h64_V32_dp1_nx1 | 64 | 32 | 1 | 1 | Adam | 19% (3/16) |

**Headline**: 23/64 (~36%) configs converged within ±0.05 of uniform RP⁴ baseline. **V=16 was the geometric sweet spot at D=5, not V=32** — overturning the V=32 universality reading from A3's three runs.

**Per-V deviation summary**:

| V | mean dev | p25-p75 | in band? |
| --- | --- | --- | --- |
| 8 | +0.115 | [0.07, 0.18] | No |
| **16** | **+0.040** | **[0.01, 0.07]** | **Yes** (only V whose mean lands inside) |
| 32 | +0.057 | [0.04, 0.07] | Just outside |

**Salt_pepper anomaly** (000106): 100% projective-clean across all 4 archs in Phase T despite being the *worst* noise to reconstruct in S-sweep (best MSE 2.51, ~34× worse than pink). **Geometry decouples from MSE** — a bank that fails to reconstruct can still produce a clean projective codebook. This matters for downstream cross-bank analysis: "the worst-fitting bank" might still produce the most useful projective representation.

**Four_quadrant anomaly**: 0% projective-clean across all 4 archs. Spatially structured noise where no architecture in T converged. Open in ft3 §10 as a deeper-probe candidate.

---

## TIER 7 — A-set verification probes (the 19-model count from ft2)

These are the projective-codebook verification runs that produced the n=19 count cited in ft2's Section 5 table. **All entries are individual probes** with explicit deviation measurements.

| probe | model | D | V | n_axes | pairs | mean projective angle | uniform baseline | dev | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | G-Cand | 3 | 32 | 22 | 10 | 1.011 | 1.015 | **−0.004** | PROJECTIVE-CLEAN |
| A1 | H2a | 4 | 32 | 26 | 6 | 1.116 | 1.114 | **+0.002** | PROJECTIVE-CLEAN |
| A2 (×16 banks) | h2-64 single-noise (banks 0-15) | 4 | 32 | 24-27 | 5-8 | mean 1.115 | 1.105 | **+0.010 mean, ±0.013** | ALL 16 PROJECTIVE-CLEAN |
| A3 (×3 runs) | A3 D=5 (single-arch single-noise) | 5 | 16/32/64 | 16/29/51 | 0/3/13 | varies | varies | -0.015 / +0.016 / +0.019 | PROJECTIVE-CLEAN at A3, **QUALIFIED by Phase T** — generalization fails |

**Probe count: 1 (A0) + 1 (A1) + 16 (A2) + 3 (A3) = 21 individual probe runs.** ft2 cited "19 models" because the A3 three runs were treated as one architectural data point. Phase T (000106) re-classified A3 as a single-arch test, leaving 17 projective-clean instances at D=3/4 robustness level.

---

## TIER 8 — Substrate prototype trained models (this week's runs)

| HF path | params | D | V | content | regime | result |
| --- | --- | --- | --- | --- | --- | --- |
| `AbstractPhil/geolip-SVAE/bintree_proto_v1` | 57,215 | 4 | 32 | depth-4 binary tree, i.i.d. Bernoulli ±1, BFS-encoded | **PASSTHROUGH** | best test_mse 3.5e-5 ep 20, 100% bits/trees from ep 1, CV 0.80-1.00, erank 4.00, ratio 1.00 |
| `AbstractPhil/geolip-SVAE/sentencepiece_proto_v1` | 57,215 | 4 | 32 | t5-base SP token IDs as 16-bit ±1 floats | **PASSTHROUGH** | best test_mse 5.78e-6 ep 18, 100% bits/tokens from ep 1, α=0.023 throughout, CV 0.85-1.11, erank 4.00, ratio 0.99 |
| `AbstractPhil/geolip-SVAE/byte_trigram_proto_v1` | 57,215 | 4 | 32 | UTF-8 byte trigrams as RGB pixels at 256×256 | **ENGAGED** | best test_mse 1.7e-5 ep 19, 83.9% byte / 61.3% trigram from 0% floor, α 0.024→0.043, CV left band ep 6, ratio 1.07, erank 3.9955 dip |

The bintree + SP-bit pair establishes the **passthrough control**. Byte-trigram is the **first text-engaged omega-class candidate** but its codebook hasn't been formally probed against the projective threshold yet. Pending follow-up: byte_trigram_proto_128 (img_size decision pending, 100M sample-view target).

---

## TIER 9 — Architectural templates with no measured instance

Documented architectures the catalog does not yet cover. Each is omega-class *eligible* under the architecture criterion but lacks a verification probe.

| template | arch | params | D | V | status | what's needed |
| --- | --- | --- | --- | --- | --- | --- |
| Johanna D=16 | PatchSVAE-F | 8.7M (estimate) | 16 | 256 | not yet U5-tested | run extract_codebook on Johanna checkpoint, compute deviation from uniform RP^15 |
| Grandmaster omega tokens | concept | — | — | — | paper-level reference | needs trained instance + verification |
| `geolip-svae-nosvd-ablation` repo | svd_mode='none' variants | varies | varies | varies | independent repo, omega verification not surfaced into main catalog | inventory the trained checkpoints, run U5 across them |
| D=6, D=7, D=8 with V matched to natural axis count | predicted by Phase T framework | — | 6/7/8 | ~22/28/34 (predicted) | not run | sweep with natural-axis-count V matching |

---

## TIER 10 — Explicitly excluded from omega-class

| exclusion | reason | source |
| --- | --- | --- |
| D=2 configs | Cannot form pentachoron (needs ≥5 points), CV undefined | Q-rank 10 |
| Q-rank 10 (h64_V32_D2_dp0_nx1_adam) | D=2, MSE 0.16 = essentially failed reconstruction | Q-sweep |
| 9 P-sweep NaN configs | LBFGS Hessian-corruption casualties (000099 bug profile) | P-sweep, never re-run |
| bintree_proto_v1 | Passthrough regime, codebook not engaged | 000111 |
| sentencepiece_proto_v1 | Passthrough regime, cross-attn idle | 000112 |
| Phase T D=5 V=32 cells (most) | V over-counted vs natural axis count ~16, fails projective-clean | 000106 |
| Phase T D=5 four_quadrant (all 4 archs) | 0% projective-clean, spatial-structured noise | 000106 |
| ablation Group H, M, L SVD-removal variants | Spectrum-degenerate; not sphere-solvers in proper sense | Phase 1 ablation |

---

## Smallest-instance benchmarks across the catalog

For when minimal-parameter operation matters:

| tier | smallest config | params | D | V | MSE | regime | use case |
| --- | --- | --- | --- | --- | --- | --- | --- |
| absolute smallest projective-clean | **Q-rank09 (P-class)** | **28,899** | 3 | 32 | 0.02782 | LOW-band projective-clean on RP² | minimum-parameter omega |
| smallest H2a (canonical sphere-solver) | **Q-rank02** | **40,227** | 4 | 32 | 0.00205 | HIGH-band sphere-solver on RP³ | canonical sphere-solver baseline |
| Phase T D=5 V=16 sweet spot | h64_V16_dp1_nx1_D5 | ~36,607 | 5 | 16 | varies | D=5 partial projective-clean | D=5 representative |
| h2-64 single bank (production) | bank_idx 0..63 | 57,215 | 4 | 32 | varies | per-noise sphere-solver | bank-level training composition |

The 28,899-param P-class candidate (Q-rank09) is the absolute floor for projective-clean. The 40,227-param H2a (Q-rank02) is the floor for canonical sphere-solver behavior at D=4. Under 30K and under 41K respectively.

---

## What's missing from this catalog

1. **byte_trigram_proto_v1 codebook investigation** — engagement signature confirmed in trajectory, but `extract_codebook` against the trained checkpoint hasn't been run for the formal projective-clean verification. Tier 1 entry should move to "verified" once this is done.
2. **byte_trigram_proto_128** — pending img_size=64-vs-128 decision + 100M-sample-view run completion.
3. **9 P-sweep NaN re-runs** — never executed with the LBFGS-fixed trainer (parked open item from 000100). Could surface 9 additional Tier 2 candidates.
4. **R-sweep results probed against projective-clean criterion** — the polytope-packing predictions (16-cell, 8-cell, dodecahedron) trained but their codebooks weren't surfaced into the catalog.
5. **Johanna D=16 verification** — the only large-D representative of the noise-substrate line; not yet probed.
6. **Cross-substrate kNN graph** — bintree, SP-bit, byte-trigram, h2-64-noise codebook-similarity matrix. The "what survives the universal-substrate-hope death" finding (000111) requires this measurement.
7. **Disproof candidates for Omega** — the methodological pivot from 000108 demands negative-result candidates the catalog doesn't yet contain (non-spherical bottleneck variants, no-spatial-coherence content, byte-misaligned content).