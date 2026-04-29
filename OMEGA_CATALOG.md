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

## TIER 3 — Phase 1 ablation grid (233 configs across 15 groups × 3 bands × seeds)

The Phase 1 ablation predates the P-sweep. **A through M groups, each varying one architectural axis at a time, all three CV bands (LOW D=16, MID D=8, HIGH D=4), seed-replicated where statistically meaningful.** 233 total entries; 227 band-match (97%); 229 params-finite (98%); 223 valid (band-match + finite). Data from `omega_inventory.csv`. Columns: D / V / hidden / depth / patch_size = `dp`/`ps` / n_seeds / mean test MSE / mean CV / band deviation (observed_sphere_cv − uniform_RP^(D-1)_baseline) / n_finite seeds.

### HIGH band (D=4, V=32, hidden=64, ps=4) — sphere-solver candidates

| group | variant | n | mse | cv | dev | fin | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | baseline | 5 | 1.4591 | 0.902 | +0.018 | 5 | seed-replicated reference |
| B | B1_all16 | 1 | 1.4709 | 0.858 | +0.021 | 1 | 16-noise mixture |
| B | B2_gaussian_only | 1 | 0.9590 | 0.753 | +0.027 | 1 | gaussian-only train |
| B | B3_structured | 1 | 1.7076 | 0.968 | −0.036 | 1 | structured-noise train |
| B | B4_heavy_tailed | 1 | 1.7982 | 0.820 | +0.064 | 1 | heavy-tailed train |
| B | B5_first_half | 1 | 1.4395 | 0.944 | +0.025 | 1 | noises 0-7 |
| B | B6_even_indices | 1 | 1.9918 | 0.880 | +0.023 | 1 | noises 0,2,4,...,14 |
| C | C1_adam | 1 | 1.4667 | 0.860 | +0.022 | 1 | Adam @ default lr |
| C | C2_sgd | 1 | 1.5670 | 0.858 | +0.019 | 1 | plain SGD |
| C | C3_sgd_momentum | 1 | 1.2006 | 0.860 | +0.013 | 1 | SGD with momentum |
| C | C4_adamw | 1 | 1.4706 | 0.859 | +0.023 | 1 | AdamW |
| C | C5_lbfgs | 1 | nan | 0.943 | −0.923 | 1 | **LBFGS Hessian-corruption — pre-fix bug** |
| D | D1_cosine | 1 | 1.4712 | 0.858 | +0.020 | 1 | cosine LR |
| D | D2_constant | 1 | 1.3028 | 0.858 | +0.028 | 1 | constant LR |
| D | D3_linear_decay | 1 | 1.4652 | 0.858 | +0.021 | 1 | linear decay |
| D | D4_warm_restart | 1 | 1.3216 | 0.858 | +0.016 | 1 | warm restart |
| D | D5_one_cycle | 1 | 1.4656 | 0.858 | +0.019 | 1 | one-cycle LR |
| E | E1_full_softhand | 3 | 0.4420 | 0.907 | +0.026 | 3 | soft-hand CV regularizer |
| E | E2_pure_mse | 3 | 0.4692 | 0.904 | +0.037 | 3 | pure MSE, no regularizer |
| E | E3_measure_only | 3 | 0.4517 | 0.908 | −0.011 | 3 | measure CV but don't regularize |
| E | E4_hard_cv_penalty | 3 | 0.4504 | 0.902 | +0.002 | 3 | hard CV penalty |
| F | F1_gelu | 1 | 1.4669 | 0.858 | +0.022 | 1 | GELU activation |
| F | F2_relu | 1 | 1.4744 | 0.855 | +0.025 | 1 | ReLU |
| F | F3_silu | 1 | 1.4702 | 0.901 | +0.022 | 1 | SiLU |
| F | F4_tanh | 1 | 1.4550 | 0.806 | +0.027 | 1 | tanh |
| F | F5_identity | 1 | 1.4522 | 0.866 | +0.031 | 1 | identity (no activation) |
| G | G1_sphere_norm | 1 | 1.4668 | 0.858 | +0.022 | 1 | reference (sphere-norm M) |
| G | G2_no_norm | 1 | 1.4680 | 1.112 | **+0.357** | 1 | **REMOVES sphere-norm — geometry breaks** |
| G | G3_layer_norm | 1 | 1.4482 | 0.701 | **−0.217** | 1 | **layer-norm instead — geometry breaks** |
| G | G4_scale_only | 1 | 1.4588 | 1.112 | **+0.359** | 1 | **scale-only — geometry breaks** |
| H | H1_svd_fp64 | 3 | 0.4198 | 0.906 | +0.006 | 3 | full-fp64 SVD reference |
| **H** | **H2_linear_matched** | **3** | **0.0456** | **0.908** | **+0.005** | **3** | **CANONICAL — H2-class baseline.** 9× lower MSE than H1 fp64. |
| H | H3_linear_unmatched | 3 | 0.0948 | 0.911 | +0.001 | 3 | linear readout, mismatched dims |
| H | H5_batch_shared_svd | 2 | 0.2799 | 0.918 | +0.056 | 2 | shared-SVD across batch |
| H | H6_no_svd_direct | 1 | 0.4606 | 0.902 | +0.085 | 1 | direct readout, no SVD |
| I | I1_1layer | 1 | 1.4733 | 0.859 | +0.023 | 1 | 1 cross-attn layer |
| I | I2_0layers | 1 | 1.8687 | 0.910 | +0.087 | 1 | 0 cross-attn layers |
| I | I3_2layers | 1 | 1.6295 | 0.924 | −0.002 | 1 | 2 cross-attn layers |
| I | I4_unbounded_alpha | 1 | 1.4803 | 0.859 | +0.022 | 1 | no clip on α |
| K | K1_bs128 | 1 | 1.4653 | 0.858 | +0.021 | 1 | batch_size=128 |
| K | K2_bs32 | 1 | 1.4276 | 0.858 | +0.024 | 1 | batch_size=32 |
| K | K3_bs512 | 1 | 1.5732 | 0.859 | +0.024 | 1 | batch_size=512 |
| K | K4_bs1024 | 1 | 1.5428 | 0.859 | +0.017 | 1 | batch_size=1024 |
| L | L1_orthogonal | 1 | 1.4802 | 0.858 | +0.018 | 1 | orthogonal init |
| L | L2_kaiming | 1 | 2.4802 | 0.912 | −0.009 | 1 | Kaiming init |
| L | L3_xavier | 1 | 1.6138 | 0.952 | +0.025 | 1 | Xavier init |
| L | L4_normal_small | 1 | 1.5360 | 0.942 | −0.019 | 1 | normal init small std |
| L2 | L2_lbfgs_pure_mse | 3 | **0.0058** | 0.878 | **−0.605** | 1 | **LBFGS+pure MSE: BEST MSE ON HIGH but only 1/3 finite, geometry broken** |
| M | M1_sgd_aggressive | 1 | 1.0770 | 0.858 | +0.009 | 1 | SGD aggressive |
| M | M2_sgd_huge_lr | 1 | 0.6798 | 0.858 | +0.052 | 1 | SGD huge lr |
| M | M3_sgd_high_momentum | 1 | 1.2939 | 0.859 | +0.013 | 1 | SGD high momentum |
| E_preview | E1-E4 (4 variants × 1 seed each) | 4 | ~1.47 | ~0.858 | +0.020 to +0.028 | 4 | 1-seed preview of Group E |

**HIGH-band omega-class candidates (band-match + finite + |dev|<0.05)**: 49 of 51 HIGH entries. The two failures: G2_no_norm (+0.357 — confirms sphere-norm is non-negotiable for omega-class) and G4_scale_only (+0.359 — same finding). G3_layer_norm at −0.217 is the third sphere-norm-disruption case. C5_lbfgs is the pre-fix Hessian-corruption casualty.

**HIGH-band MSE leader**: **H2_linear_matched at MSE 0.0456 with dev +0.005**. This is the **canonical sphere-solver template** that h2-64 was built from — 9× lower MSE than the fp64-SVD baseline (H1) at the same dev tolerance. The L2_lbfgs_pure_mse achieves MSE 0.0058 (8× better still) but only 1 of 3 seeds converged finite and geometry deviation hits −0.605, so it's not omega-class despite the MSE win.

### MID band (D=8, V=64, hidden=64, ps=16) — bulk-Gaussian attractor

| group | variant | n | mse | cv | dev | fin |
| --- | --- | --- | --- | --- | --- | --- |
| A | baseline | 5 | 1.0115 | 0.359 | +0.005 | 5 |
| B | B1_all16 | 1 | 1.0088 | 0.380 | −0.004 | 1 |
| B | B2_gaussian_only | 1 | 0.9530 | 0.369 | −0.017 | 1 |
| B | B3_structured | 1 | 0.7076 | 0.347 | −0.007 | 1 |
| B | B4_heavy_tailed | 1 | 1.6415 | 0.352 | +0.007 | 1 |
| B | B5_first_half | 1 | 0.9441 | 0.380 | −0.004 | 1 |
| B | B6_even_indices | 1 | 1.1708 | 0.352 | −0.010 | 1 |
| C | C1_adam | 1 | 1.0093 | 0.380 | −0.004 | 1 |
| C | C2_sgd | 1 | 1.7354 | 0.381 | −0.018 | 1 |
| C | C3_sgd_momentum | 1 | 1.0390 | 0.380 | +0.001 | 1 |
| C | C4_adamw | 1 | 1.0095 | 0.379 | −0.005 | 1 |
| C | C5_lbfgs | 1 | nan | 0.337 | −0.357 | 1 | LBFGS pre-fix |
| D | D1-D5 (5 variants) | 5 | ~1.00 | ~0.380 | −0.002 to −0.009 | 5 |
| E | E1_full_softhand | 3 | 0.9441 | 0.361 | +0.008 | 3 |
| E | E2_pure_mse | 3 | 0.9416 | 0.362 | +0.010 | 3 |
| E | E3_measure_only | 3 | 0.9430 | 0.363 | +0.009 | 3 |
| E | E4_hard_cv_penalty | 3 | 0.9447 | 0.362 | +0.011 | 3 |
| F | F1-F5 (5 variants) | 5 | ~1.01 | 0.375-0.388 | −0.002 to −0.009 | 5 |
| G | G1_sphere_norm | 1 | 1.0105 | 0.381 | −0.005 | 1 | reference |
| G | G2_no_norm | 1 | 0.9935 | 0.560 | **+0.198** | 1 | **sphere-norm removed — geometry breaks** |
| G | G3_layer_norm | 1 | 1.0093 | 0.428 | +0.064 | 1 |
| G | G4_scale_only | 1 | 1.0103 | 0.556 | **+0.149** | 1 | **scale-only — geometry breaks** |
| H | H1_svd_fp64 | 3 | 0.9429 | 0.362 | +0.009 | 3 |
| H | H2_linear_matched | 3 | **0.9195** | 0.365 | +0.015 | 3 | best MID-band MSE among finite |
| H | H3_linear_unmatched | 3 | 0.9343 | 0.359 | +0.018 | 3 |
| H | H4_svd_fp32 | 2 | 0.9434 | 0.368 | +0.015 | 2 |
| H | H5_batch_shared_svd | 2 | 0.9439 | 0.367 | +0.016 | 2 |
| H | H6_no_svd_direct | 1 | 0.9453 | 0.368 | −0.002 | 1 |
| I | I1-I4 (4 variants) | 4 | 1.007-1.011 | 0.355-0.380 | −0.005 to +0.004 | 4 |
| K | K1-K4 (4 variants) | 4 | 0.999-1.038 | ~0.380 | −0.002 to −0.012 | 4 |
| L | L1-L4 (4 variants) | 4 | 1.011-1.340 | 0.336-0.383 | +0.006 to +0.011 | 4 |
| L2 | L2_lbfgs_pure_mse | 3 | **0.8924** | 0.359 | **−0.233** | 1 | LBFGS pre-fix |
| M | M1-M3 (3 variants) | 3 | 0.976-1.024 | ~0.380 | +0.002 to +0.006 | 3 |
| E_preview | E1-E4 | 4 | ~1.01 | ~0.380 | −0.004 to −0.006 | 4 |

**MID-band omega-class candidates**: 73 of 78 MID entries. Same sphere-norm-disruption failures (G2, G4). MSE leader at H2_linear_matched 0.9195 with dev +0.015.

### LOW band (D=16, V=64, hidden=64, ps=16) — noise-substrate attractor

| group | variant | n | mse | cv | dev | fin |
| --- | --- | --- | --- | --- | --- | --- |
| A | baseline | 5 | 1.0080 | 0.197 | −0.000 | 5 | reference (5-seed) |
| B | B1_all16 | 1 | 1.0104 | 0.203 | −0.003 | 1 |
| B | B2_gaussian_only | 1 | 0.9389 | 0.210 | +0.001 | 1 |
| B | B3_structured | 1 | 0.7051 | 0.207 | +0.006 | 1 |
| B | B4_heavy_tailed | 1 | 1.6091 | 0.191 | +0.009 | 1 |
| B | B5_first_half | 1 | 0.9401 | 0.201 | −0.005 | 1 |
| B | B6_even_indices | 1 | 1.1678 | 0.211 | −0.001 | 1 |
| C | C1-C4 (4 finite) | 4 | 1.008-1.626 | ~0.204 | −0.004 to +0.008 | 4 |
| C | C5_lbfgs | 1 | nan (excluded) | — | — | 0 |
| D | D1-D5 | 5 | 0.974-1.010 | 0.203-0.205 | −0.000 to −0.003 | 5 |
| E | E1_full_softhand | 3 | 0.9338 | 0.199 | +0.002 | 3 |
| E | E2_pure_mse | 3 | 0.9340 | 0.200 | +0.001 | 3 |
| E | E3_measure_only | 3 | 0.9342 | 0.200 | +0.002 | 3 |
| E | E4_hard_cv_penalty | 3 | 0.9336 | 0.199 | +0.001 | 3 |
| F | F1-F5 | 5 | 1.003-1.011 | 0.199-0.204 | −0.001 to −0.004 | 5 |
| G | G1_sphere_norm | 1 | 1.0085 | 0.204 | −0.003 | 1 |
| G | G2_no_norm | 1 | 0.9914 | 0.353 | **+0.174** | 1 |
| G | G3_layer_norm | 1 | 1.0048 | 0.211 | +0.001 | 1 |
| G | G4_scale_only | 1 | 1.0092 | 0.347 | **+0.159** | 1 |
| H | H1_svd_fp64 | 3 | 0.9345 | 0.200 | +0.002 | 3 |
| H | H2_linear_matched | 3 | **0.9128** | 0.204 | +0.005 | 3 |
| H | H3_linear_unmatched | 3 | 0.9195 | 0.202 | +0.010 | 3 |
| H | H4_svd_fp32 | 2 | 0.9327 | 0.202 | +0.002 | 2 |
| H | H5_batch_shared_svd | 2 | 0.9331 | 0.202 | +0.003 | 2 |
| H | H6_no_svd_direct | 1 | 0.9359 | 0.202 | +0.003 | 1 |
| I | I1-I4 | 4 | 1.004-1.014 | 0.201-0.210 | −0.001 to +0.012 | 4 |
| **J** | **J1_V64_h64** | **1** | **1.0094** | **0.204** | **−0.002** | **1** | **V=64 h=64 — Group J is V-sweep on D=16** |
| **J** | **J2_V32_h32** | **1** | **1.1314** | **0.205** | **−0.028** | **1** | V=32 h=32 — capacity floor probe |
| **J** | **J3_V16_h32** | **1** | **1.1862** | **0.234** | **−0.002** | **1** | V=16 h=32 — sub-capacity test |
| **J** | **J4_V64_h32** | **1** | **1.0957** | **0.208** | **−0.008** | **1** | V=64 h=32 — hidden-floor test |
| **J** | **J5_V128_h128** | **1** | **0.9595** | **0.199** | **−0.000** | **1** | V=128 h=128 — over-capacity, dev exactly zero |
| K | K1-K4 | 4 | 0.997-1.047 | ~0.204 | −0.003 to −0.006 | 4 |
| L | L1-L4 | 4 | 1.009-1.242 | 0.197-0.204 | +0.002 to +0.012 | 4 |
| M | M1-M3 | 3 | 0.977-1.011 | 0.204 | +0.002 to +0.006 | 3 |
| E_preview | E1-E4 | 4 | ~1.01 | ~0.204 | −0.001 to −0.003 | 4 |

**LOW-band omega-class candidates**: 76 of 79 LOW entries. Same sphere-norm-disruption failures (G2, G4). MSE leader: H2_linear_matched at 0.9128 with dev +0.005 — the same H2 template that wins on every band.

**Group J — V × hidden capacity sweep at D=16**: only 5 configs (one of the smallest groups), but the only group that varies V away from the standard {32, 64} options at the LOW-band. Confirms V=128 (J5) drives dev to exactly 0.000 and produces the lowest MSE in the LOW group at 0.9595. V=16 (J3) at h=32 still maintains dev −0.002 but loses 23% on MSE relative to V=128. **J5_V128_h128 is the strongest unverified noise-substrate candidate** in the catalog — params 1.46M (estimated from V=128, h=128), dev exactly zero against the uniform-RP^15 baseline, lowest MSE in the LOW band among non-H-group entries. Worth a U5-style projective probe.

### Phase 1 ablation — engineering invariants surfaced

* **Sphere-norm (Group G) is non-negotiable**: removing it (G2, G4) blows up dev across all three bands by 0.15-0.36. Confirmed independently in HIGH/MID/LOW.
* **H2_linear_matched is the canonical template**: best MSE in 2 of 3 bands (LOW + HIGH), within 0.003 of best in MID. Forms the architectural template behind h2-64, all H2a/H2b Q-sweep candidates, and the substrate prototypes (bintree/SP-bit/byte-trigram).
* **L2_lbfgs_pure_mse achieves the lowest MSE on HIGH** (0.0058) at the cost of 67% non-convergence and broken geometry — the pre-fix Hessian-corruption pattern that 000099 diagnosed.
* **CV bands quantize cleanly by D**: HIGH ≈ 0.86 (D=4), MID ≈ 0.36 (D=8), LOW ≈ 0.20 (D=16). Bulk of variants land within ±0.05 of band center; the failures are sphere-norm disruption + LBFGS Hessian corruption.
* **Group A 5-seed reference** establishes within-config variance: HIGH dev variance ±0.018, MID ±0.005, LOW ±0.000. Within-config noise is comparable to or smaller than the |dev|<0.05 projective-clean threshold, so single-seed entries are still informative for the omega-class criterion.

### Phase 1 ablation — direct architecture comparisons

| comparison | HIGH | MID | LOW |
| --- | --- | --- | --- |
| H2_linear_matched (canonical) MSE | 0.0456 ± 0.0076 | 0.9195 ± 0.0026 | 0.9128 ± 0.0043 |
| H1_svd_fp64 (full SVD) MSE | 0.4198 ± 0.0154 | 0.9429 ± 0.0041 | 0.9345 ± 0.0023 |
| H2 advantage over H1 | **9.2× lower MSE** | 2.5% lower | 2.3% lower |
| H3_linear_unmatched MSE | 0.0948 ± 0.0123 | 0.9343 ± 0.0025 | 0.9195 ± 0.0011 |
| H6_no_svd_direct MSE | 0.4606 | 0.9453 | 0.9359 |

The HIGH-band H2 advantage (9.2× lower MSE than full-SVD H1) is what made H2_linear_matched the canonical template. At MID and LOW the H-group converges (linear-matched, full-SVD, and direct-readout all within 3% of each other), so the H2 win is specifically a HIGH-band phenomenon. This matches the architectural reading: at D=4 the SVD dimension is small enough that linear readout matches the spectral bandwidth without information loss; at D=8 and D=16 the spectral residual matters.

---

## TIER 4 — P-sweep small-battery floor grid (600 configs at 20 batches)

`group_P_small_battery_floor` from `ablation_configs.py` line 615. **Full product: 5 × 5 × 3 × 2 × 2 × 2 = 600 configs.** This is the parent grid the Q-sweep top-10 came out of. It runs after Phase 1 ablation establishes H2_linear_matched as the canonical template, and varies the architecture axes around that template at a budget-minimal 20 batches per config.

### P-sweep grid axes

| axis | values | count |
| --- | --- | --- |
| hidden | {4, 8, 16, 32, 64} | 5 |
| V | {2, 4, 8, 16, 32} | 5 |
| D | {2, 3, 4} | 3 |
| depth | {0, 1} | 2 |
| n_cross | {0, 1} | 2 |
| optimizer | {Adam, LBFGS} | 2 |

### Pins (H2_linear_matched baseline)

* `svd='none'`, `linear_readout=True`, `match_params=True` (the H2 ablation winner)
* HIGH band: `patch_size=4`, `img_size=64`
* `batch_size=256`, `batch_limit=20` (~5,120 samples seen per config)
* `n_heads=1` (since D varies down to 2, default n_heads=4 would fail)
* `grad_clip=1.0` (defensive — see TIER 4a)
* `soft_hand=False`, `cv_measure_every=2`
* Adam: `lr=3e-3` (Phase-2 default scaled to 20-batch budget)
* LBFGS: `lr=1.0` (default unit-Wolfe-step, lib's own line search)
* Training: gaussian-only (`noise_types=[0]`)
* Testing: 16-noise per-noise generalization (`test_noise_types=list(range(16))`, 256 samples per noise)

### P-sweep outcomes by optimizer split

| optimizer | total configs | finite | NaN/divergent | Q-sweep top-10 representation |
| --- | --- | --- | --- | --- |
| Adam | 300 | 300 (100%) | 0 | 6 of top-10 (Q-ranks 02, 03, 06, 07, 08, 09, 10) |
| LBFGS | 300 | 291 (97%) | 9 | 3 of top-10 (Q-ranks 01, 04, 05) |
| **Total** | **600** | **591** | **9** | **10 advanced to Q-sweep at 1000 batches** |

The 9 NaN/divergent LBFGS configs all matched the **000099 Hessian-corruption profile**: depth=1 + n_cross=1 architectures where gradient clipping inside the LBFGS closure caused the (s_k, y_k) Hessian approximation to underestimate, generating runaway H⁻¹ steps over enough iterations. 20 batches is the threshold below which divergence is incipient but not yet catastrophic; the 1000-batch Q-sweep would have surfaced 30 LBFGS divergences had it not been pre-fixed.

### P-sweep geometric attractor split (observed across all 600 configs, confirmed in Q)

| D class | typical CV (post-20-batch) | attractor | members |
| --- | --- | --- | --- |
| **D=4** | 0.86 - 1.07 (HIGH band) | **sphere-solver** (H2 family) | 200 configs (5 hidden × 5 V × 2 depth × 2 n_cross × 2 opt) |
| **D=3** | ~0.03 (LOW band) | **projective-clean on RP²** (P-class, originally framed "polynomial") | 200 configs |
| **D=2** | undefined (V<5 cannot form pentachoron) | failed geometric validity | 200 configs |

200 D=4 configs are theoretically all H2 candidates. Survival of the 6 Adam + 3 LBFGS into Q-sweep top-10 depends on `continued_training_potential`, which combines convergence-rate-at-20-batches with extrapolated-MSE-at-1000-batches. Q-sweep's full table (Tier 2) gives the actual 1000-batch outcomes for those 10.

### TIER 4a — P-sweep extrapolation rankings (the top-10 source data)

Each Q-sweep entry started as a P-sweep entry. The "P-MSE" column from `group_Q_h2_candidates` shows the MSE that ranked it at 20 batches:

| Q-rank | source P config | P-MSE (20 batch) | Q-MSE (1000 batch) | improvement | optimizer regime change |
| --- | --- | --- | --- | --- | --- |
| 01 | h64_V32_D4_dp1_nx0_lbfgs | 0.053 | 0.00421 | 13× | LBFGS clean at 1000 (post-fix) |
| 02 | h64_V32_D4_dp0_nx0_adam | 0.572 | **0.00205** | 279× | **strongest extrapolation** |
| 03 | h64_V32_D4_dp0_nx1_adam | 0.584 | 0.00250 | 234× | |
| 04 | h64_V32_D4_dp0_nx1_lbfgs | 0.041 | 0.00391 | 10× | LBFGS started low, gained little |
| 05 | h64_V16_D4_dp1_nx1_lbfgs | 0.115 | 0.03117 | 4× | V=16 hits H2b ceiling |
| 06 | h64_V32_D3_dp1_nx1_adam | 0.656 | 0.02497 | 26× | D=3 P-class |
| 07 | h64_V32_D3_dp0_nx1_adam | 0.641 | 0.03151 | 20× | D=3 P-class |
| 08 | h64_V32_D4_dp1_nx1_adam | 0.620 | 0.00231 | 268× | h2-64 single-bank arch |
| 09 | h64_V32_D3_dp0_nx0_adam | 0.638 | 0.02782 | 23× | D=3 P-class smallest |
| 10 | h64_V32_D2_dp0_nx1_adam | 0.736 | 0.16139 | 5× | D=2 — failed geometric validity |

**Reading**: Adam configs started P-sweep with high MSE (0.57-0.74) and gained 20-280× by 1000 batches. LBFGS configs started P-sweep with low MSE (0.04-0.12) and gained only 4-13×. This confirms the optimizer regime shift that 000100 logged: **LBFGS dominates at short budgets (≤20 batches), Adam dominates at long budgets (≥500 batches)**. The architectural template (h64_V32_D4 with depth/n_cross combinations) is unchanged across optimizers; only the optimization trajectory differs.

### P-sweep coverage gaps

* **9 NaN configs never re-run** with the post-fix trainer (parked open item from 000100). Could surface 9 additional Tier 2 candidates at the LBFGS+depth=1+n_cross=1 architecture frontier.
* **V<32 LBFGS coverage** is thin — only Q-rank05 represented from 300 LBFGS configs across V ∈ {2, 4, 8, 16, 32}. The full P-sweep contains 60 LBFGS configs at each V; their ranking among each other is not surfaced into the catalog.
* **hidden < 64 LBFGS** coverage is sparse — Q-sweep's top-10 are all hidden=64. A dedicated lower-hidden LBFGS sub-sweep was never run.

Test of the natural-axis-count hypothesis: V matched to known polytope vertex counts on S^(D-1) should produce **static** sphere-solver rows (no rotating antipodal frame).

| variant | V | D | polytope | predicted | params |
| --- | --- | --- | --- | --- | --- |
| R_h64_V16_D4_16cell_orthoplex_adam | 16 | 4 | 16-cell (4-orthoplex) | H2-LIKE static | — |
| R_h64_V8_D4_8cell_or_16cell_subset_adam | 8 | 4 | 8-cell (tesseract) | H2-LIKE static | — |
| R_h64_V20_D3_dodecahedron_adam | 20 | 3 | dodecahedron | H2-LIKE static | — |

**Pins**: same H2_linear_matched baseline as Q, Adam @ lr=3e-3, depth=0, n_cross=0, 1000 batches, gaussian-only training, 16-noise per-noise test.

**Status**: trained (in `phaseR_reports/` on HF), results not surfaced into the projective-clean catalog yet. Worth probing against the omega-class criterion since natural-axis-count framework predicts they should land cleanly.

---

## TIER 4b — R-sweep polytope packing test (3 configs at 1000 batches)

`group_R_packed_polytope_test` from `ablation_configs.py`. Predicted-H2-LIKE configs where V is matched to known polytope vertex counts on S^(D-1). Same H2_linear_matched baseline as P/Q, Adam @ lr=3e-3, 1000 batches, gaussian-only training. Ran 2026-04-24 alongside Q-sweep.

| variant | V | D | polytope | predicted | architecturally implies |
| --- | --- | --- | --- | --- | --- |
| R_h64_V16_D4_16cell_orthoplex_adam | 16 | 4 | 16-cell (4-orthoplex) | H2-LIKE static rows | natural axis count for D=4 = 16 |
| R_h64_V8_D4_8cell_or_16cell_subset_adam | 8 | 4 | 8-cell (tesseract) | H2-LIKE static rows | sub-polytope vertex count |
| R_h64_V20_D3_dodecahedron_adam | 20 | 3 | dodecahedron | H2-LIKE static rows | natural axis count for D=3 = 20 |

**Hypothesis tested**: when V matches a known regular polytope vertex count for S^(D-1), training should produce **static** sphere-solver rows (no antipodal pair rotation). Phil's framing (000100): the 32-row × D=3 G-Class behavior emerged because 32 points cannot be uniformly arranged on S² — geometric frustration. Match V to polytope, frustration disappears.

**Status**: trained, weights in `phaseR_reports/` on HF, but **codebooks were never extracted and probed against the projective-clean threshold**. Worth running through `extract_codebook` to surface them into the verified-omega-class tier; the natural-axis-count framework (ft2 §9.1) predicts they should land cleanly. Open item from 000101.

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