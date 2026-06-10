# geolip-aleph-void: The First Relational Geometric Vocabulary Patchwork

*A spherical autoencoder whose latent bottleneck **is** a learned projective address. Every formula in this article was read directly from the shipping code in [`geolip-svae`](https://github.com/AbstractEyes/geolip-svae); every number was read from a `final_report.json`, a model card, or a logged experiment.*

*Revision 2026-06-09 22:30 UTC: §3.13 added (aleph-routed attention — the address as a feature map); discovery catalog extended #25–#31; dated updates in §2, §8, §10, §12 and catalog rows #15/#20. All new numbers are from logged 2026-06-09 session runs (RTX PRO 6000 Blackwell / A100-80GB); session artifacts: `aleph_routed_attention.py`, `aleph_trigram_lm.py`, `aleph_probe_battery.py`.*

**Model:** [AbstractPhil/geolip-aleph-void](https://huggingface.co/AbstractPhil/geolip-aleph-void) · **Code:** [AbstractEyes/geolip-svae](https://github.com/AbstractEyes/geolip-svae) · **Battery lineage:** [AbstractPhil/geolip-SVAE](https://huggingface.co/AbstractPhil/geolip-SVAE)

---

## Abstract

`geolip-aleph-void` is the point where three months of geometric autoencoder research collapses into a single load-bearing operation. The model — `AlephModel` in `geolip_svae/aleph_model.py` — keeps the PatchSVAE sphere-solver encoder byte-identical, but replaces the deep MLP decoder accumulator with a **learned projective codebook** and an **aleph signed-projective address**: each row of the spherical latent matrix `M` is snapped to one of `K` learned axes (with orientation), and the decoder reconstructs *from the addressed rows*. Reconstruction gradient flows through the address into the codebook, so the address is not a diagnostic readout bolted onto a trained model — it is the thing the model is.

Three facts justify the design, each established before the model was built:

1. **The spherical matrix `M` is reconstruction-real.** With the address disabled (`address='none'`), a *single tied linear map* off `M` reconstructs WikiText-103 byte-trigram images to cosine ≈ 0.9997. The geometry provably carries reconstruction — something the SVAE lineage, whose reconstruction lived in a deep residual decoder, could never demonstrate about its own latent.
2. **The recon-real codebook addresses more sharply than the faux-embedding batteries.** Codebooks extracted from this regime address real tokens with mean projective margin ≈ 0.967, versus ≈ 0.929 for the SVAE batteries.
3. **The codebook is void-rich.** It carries roughly 7× the 2-dimensional topological structure of the SVAE battery codebooks (β₂ per axis ≈ 0.56 vs ≈ 0.08). Voids — measured by persistent homology on the projective space the axes actually live on — had already been identified as the fingerprint that separates symbolic substrates from continuous ones. The model is named for them.

This article documents the model completely: the experimental chain that produced it, every formula and the purpose of each, the data substrates it accepts and why they are shaped the way they are, the patchwork system, the behavior it induces in other models (including the failure modes), its strengths as a tokenized backup structure, the implicitly learned structure revealed by the test batteries, and the full train/save/load/host system. It closes with the catalog of major discoveries across the geolip, geofractal, geovocab2, and WideCompiler systems that this model sits on top of.

---

## 1. What the model is, in one pass

```
images (B, C, H, W)
  └─ extract_patches(ps)                    → (B, N, patch_dim)        the patchwork
      └─ residual MLP encoder               → (B·N, V·D)
          └─ reshape + F.normalize(dim=-1)  → M: rows on S^(D-1)       the spherical matrix
              └─ aleph address vs learned codebook A (K, D)
                    M̂ = Σₖ sinh(uₖ)·Aₖ / Σₖ cosh(uₖ),  u = (M·Aᵀ)/τ   the bottleneck
                  └─ readout: U = M̂, S = column norms, Vt = I
                      └─ decoder (tied | dict | mlp) reads M̂          → patches
                          └─ stitch_patches + BoundarySmooth           → recon (B, C, H, W)
```

The forward contract is identical to PatchSVAE — `forward(images) → {'recon', 'svd': {U, S, S_orig, Vt, M, M_hat}}` — so the entire inference stack built for the SVAE batteries (codebook extraction, calibration registry, scaling engine, text wrappers, dataset bundles) operates on an `AlephModel` unchanged, and aleph codebooks are directly comparable to every battery codebook extracted since April.

The reference configuration is deliberately tiny: `V=32, D=4, ps=4, hidden=64, depth=1, K=64, address_tau=0.1, decode_mode='tied'` — **26,795 parameters**, a 114 KB checkpoint. The codebook itself adds only `K·D = 256` of those parameters. A scaled MLP-decode variant (`dec_hidden=512, dec_depth=8`) is 4,321,963 parameters. Both are hosted.

### Why "aleph", why "void"

The **aleph** is the address: a signed assignment of each spherical row to one of `2K` *oriented half-axes* `[+A; −A]` of a projective codebook. The name entered the program through the transfinite-cardinal machinery in the `geovocab2` formula library (`cantor.py` implements ℵ₀/ℵ₁ cardinal arithmetic for the lattice vocabulary), and was attached to this specific logit during the transformer experiments of late May 2026, where the "aleph logit" was first proposed as a discriminative readout for fractally-bound spaces where cosine saturates. The June 7 announcement put it directly: the architecture is intended as "the mathematical aleph procrustes geofractal addressed language latent" for the first large-scale distillation.

The **void** is the topology: *Reading the Voids* (May 31, 2026) established across 41 frozen batteries, six model classes, and three substrate types that H2 voids — finite β₂ features in the persistent homology of the extracted codebook, measured on ℝP^(D−1) after a metric-alignment fix — separate symbolic substrates from continuous ones at fixed latent dimension. Image codebooks are void-sparse; symbolic-vocabulary codebooks are void-rich. The aleph codebook, learned under a recon-real objective on a symbolic substrate, is the void-richest structure measured in the program so far. The model is the deliberate construction of the thing the diagnostics kept finding.

---

## 2. The experimental chain

The model was not designed; it was cornered. Twenty research articles, ~191 model repos, and 70 dataset repos under [AbstractPhil](https://huggingface.co/AbstractPhil) form the chain of custody. The arc, in phases:

### Phase 0 — Geometric memory and the pentachoron (Feb–Mar 2026)

- *QWEN 3.5 Residual Thinking Embeddings* (Mar 3), *Geometric Fusion: Cross-Modal Alignment Through Shared Pentachoron Geometry* (Mar 6), and the *Geometric Memory* trilogy (Mar 9, 11, 14) established the program's commitment: architecture supplies a geometric substrate; learned parameters are corrections and measurements on top. The pentachoron (5-vertex 4-simplex) and Cayley–Menger volume machinery from `lattice_vocabulary`/`geovocab2` date from here.
- *Procrustes ViT Shared Manifold Alignment* (Mar 15) introduced closed-form cross-model alignment — the mechanism the aleph distillation plan now names directly.
- *Constellation Relay, Geometric Bottleneck, and the Re-Emergence of the Potential 0.29154 Binding Constant* (Mar 18): the constant 0.29154 surfaces independently for the second time.
- Kernel infrastructure: *Fused Batched Thin SVD: a 5000× Speedup with Triton Kernels* (Mar 25, Part II May 1) and *FL Hybrid Eigendecomposition* (Apr 1) became `geolip-core`'s `batched_svd` / `FLEigh` — the exact solvers `geolip_svae.model` imports today.

### Phase 1 — The SVAE lineage and the sphere-norm breakthrough (Apr 2026)

- Prototypes V7→V8 hit the **scale-explosion "snap"**: S₀ and S_D blow up together (V8-amp epoch 30→32: S₀ 7.45→89.79, S_D 1.60→32.90), the spectral ratio collapses to near-init, effective rank jumps to near-max, and reconstruction breaks. Mechanism: Adam steps on an ill-conditioned Gram matrix.
- The fix is one line and it is the architecture: **`M = F.normalize(M, dim=-1)`**. Sphere-normalizing the rows of `M` bounds encoder output magnitude *by construction* — unit rows fix `‖M‖_F² = V`, so the total spectral energy `Σσᵢ² = V` no matter what the encoder weights do; under it, S₀ stays essentially constant (2.86→2.77 over 100 epochs at V=96; 4.12→4.12 over 400 epochs at V=256). In the maintainer's framing: "this is where we went from guessing to solving based on seeing what happened."
- The battery tier was trained on this template: **Johanna** (16-noise-type expert, 17M params, D=16), **Fresnel** (ImageNet, 99.993% reconstruction fidelity, 48:1 compression at latent `(B,16,8,8)`), **Freckles** (2.5M params, D=4, 4×4 patches, MSE 5e-6 at 64×64), **Grandmaster** (v30 denoiser: Johanna's decoder conditioned on Fresnel's omega tokens), **Alexandria** (v22, Wikipedia text). *Omega Tokens: Finding The Self Solving Frame* (Apr 8) and *Structural Attractors in Neural Network Weight Space* (Apr 6) document this tier.
- *Three Geometric Bands in a Sphere-Normalized Patch Autoencoder* (Apr 22): 149 training runs across 12 orthogonal hyperparameter dimensions. The headline result reproduces in 96% of runs and fails **only** when row normalization is ablated. Trained column-norm CV matches the uniform-sphere prediction at each D: D=16 → CV ≈ 0.20 (uniform S¹⁵ predicts 0.199, within ±0.003 across 5 seeds); D=8 → ≈ 0.36 (S⁷ predicts 0.357); D=4 → ≈ 0.90 (S³ predicts 0.923).

### Phase 2 — The projective-axis discovery (late Apr 2026)

- *The Polygonal Omega: Trained Sphere-Solvers Are Projective Codebooks* (Apr 25): every trained sphere-solver tested produces an `M` whose rows, after merging antipodal pairs by mutual-strongest matching, form a near-uniform codebook on ℝP^(D−1). Verified across 19 trained models at D=3, 4, 5, then across the **h2-64 battery array** (192 banks: 64 batteries × 3 epoch phases, 57,215 params per bank).
- *H2 Omega Confirmed, Paradigm Shift: Attempting to Disprove Omega As A Whole* (Apr 29) is the adversarial replication pass. The attempt to break the finding instead refined it into **statute classes**: a second stable codebook statute (polytope-class, repulsive packing, *more* spread than uniform) exists alongside the uniform class, and which statute appears is a property of *(model × calibration)*, not of the model alone.
- The byte-trigram substrate engaged here: `byte_trigram_proto_64_patch_2_v1` (52,571 params) measured uniform-class on an out-of-distribution noise probe (deviation +0.046, 39% pair fraction) and polytope-class in-distribution (deviation +0.083, 52% pair fraction).

### Phase 3 — Text recall, the transformer shell, and the voids (May 2026)

- The H2 batteries were turned on symbolic data: byte trigrams (99.6% n-gram recall byte-by-byte; a 16.77M-entry vocabulary potential over ~5M unicode byte combinations; channel-agnostic), SentencePiece bit-planes (valid, MSE 5.78e-6), binary trees ("a uniformly potent gating mechanism" under hierarchical ±1 encoding), and ternary codes (directly responsive to −1/0/+1).
- *The geolip-svae-transformer* (May 29): a frozen battery + fixed isometric lens + spectral transformer shell. v2 converges "semi-close to the SVAE spectrum in considerably fewer epochs" at ~90K shell params over a ~57K battery. The repo README also records the first, openly speculative naming of the **aleph logit** ("the conceptualization of an aleph logit is, a pipe dream" — flagged *Likely Faulty Experiment* at the time).
- *Reading the Voids: Topological Contribution Signals in Frozen Geometric Codebooks* (May 31): 15 independently-toggleable contribution signals, ablated across 41 frozen batteries spanning six model classes and three substrate types, after fixing a metric-alignment bug (persistence had been measured on the raw sphere instead of the projective space the axes live on). Finding: **at fixed latent dimension, H2 voids separate symbolic substrates from continuous ones** — a distinction the first-order geometry cannot make. Packaged as the `omega_phase_v2` two-axis taxonomy.

### Phase 4 — The gate, and the model (June 2026)

- The SVAE's reconstruction lives in a deep residual decoder; its spherical latent is therefore a "faux embedding" — the codebooks extracted from it are real, but the geometry was never *proven* to carry reconstruction. The gate experiment removed the accumulator: tied single-linear decode straight off `M`. Result: **cosine ≈ 0.9997** on byte-trigram. `M` is recon-real.
- With the gate passed, the address was made load-bearing: `AlephModel`, first committed May 31, 2026 (`added aleph model`, hours after the voids article) and iterated daily through June 4 (`added aleph transformer mirroring the tested svae transformer structure`), hosted at `AbstractPhil/geolip-aleph-void` (created June 1, updated June 4). The active experiment — whether reconstruction survives the *soft* address bottleneck versus the `none` gate — is exactly what the hosted `final_report.json` files track, and the published checkpoints already answer the harder discrete question: with `address='hard'` (straight-through, fully discrete codes), byte-trigram reconstruction holds at cosine 0.9967 (MLP decode, 4.32M params) and 0.9920 (tied decode, 26,795 params), with 125–126 of 128 oriented axes in active use and zero collapse.

### Phase 5 — The address leaves the autoencoder (2026-06-09)

- **[2026-06-09 update]** The aleph signed-projective address was lifted out of the
  reconstruction bottleneck and installed as the routing medium of an attention
  mechanism (`AlephRoutedAttention`, session artifact). The address distribution over
  the `2K` oriented axes IS the feature map of a linear attention: tokens communicate
  *through* the codebook rather than reconstructing from it. Same closed form, same
  `max|u|` stabilization, same K-wide intermediates — a new jar for the same lightning.
- Lineage closure: the Cantor fractal router (O(n) by position-geometry, killed by
  gather-bound hardware hostility) is formally authenticated as a breadcrumb — the
  aleph router keeps its premise (geometric routing replaces the O(n²) score matrix)
  and pays in pure GEMM. Measured on the trigram substrate: softmax-parity quality,
  linear wall-clock to 65,536 tokens where softmax attention OOMs at 16,384.
- The probe battery (statutes, projective margins, Procrustes, CM band, persistence,
  state erank) was pointed at the new system; the verdict is the **two-hemisphere
  law** (#27) and the **codebook-as-medium** synthesis of §3.13.

---

## 3. The formulas

Everything below is quoted against `geolip_svae/aleph_model.py` (650 lines), `geolip_svae/model.py`, `geolip_svae/train_aleph.py`, `geolip_svae/model_transformer.py`, and `geolip_svae/inference/`. Structure first, purpose second, for every operation in the model.

### 3.1 The patchwork: `extract_patches` / `stitch_patches`

An image `(B, C, H, W)` is cut into `N = (H/ps)·(W/ps)` non-overlapping patches, each flattened to `patch_dim = C·ps²`. Every patch is encoded **independently** — the model never sees global context except through the optional spectral cross-attention (§3.8) and the boundary smoother (§3.9). This is the "patchwork": the unit of representation, addressing, and vocabulary is the patch, not the image. It is why every weight in the model is dimensioned by `(V, D, ps, hidden)` and none by `N` — and therefore why the same weights run at any resolution (§5).

### 3.2 The spherical encoder (byte-identical to PatchSVAE)

```python
h = act(enc_in(patch))                  # Linear(patch_dim → hidden)
for block in enc_blocks:                # depth × [LayerNorm → Linear → act → Linear]
    h = h + block(h)                    # residual
M = enc_out(h).reshape(B·N, V, D)       # Linear(hidden → V·D), orthogonal init
M = F.normalize(M, dim=-1)              # rows onto S^(D-1)
```

**Purpose of each piece:**

- `nn.init.orthogonal_(enc_out.weight)` — load-bearing. Validated by the L-group init ablations; removing it regresses every variant tested.
- `F.normalize(M, dim=-1)` — **the** architectural fix. Each of the `V` rows of `M` becomes a unit vector on the sphere S^(D−1). Pins the total spectral energy by construction (`Σσᵢ² = ‖M‖_F² = V`), structurally breaks the scale-explosion chain, and is the precondition for everything geometric that follows: the projective codebooks, the statute classes, the voids, the address.
- No BatchNorm, no Dropout, no global average pooling, anywhere. Each was tried; each destabilizes or destroys the structure (GAP alone costs ~70% → 29% downstream accuracy).

### 3.3 The learned projective codebook

```python
self.codebook = nn.Parameter(A0)        # (K, D); buffer if freeze_codebook=True
```

`K` axes in D-space. An axis is a *line through the origin* — sign is resolved per-row by the address, which is what makes the codebook projective rather than spherical. It is trained **by reconstruction through the address**: no codebook loss, no commitment loss, no EMA — the recon gradient is the only pressure on it. It adds exactly `K·D` parameters (256 at the reference config).

Three initializations (`_init_codebook`):

- `'random'` — Gaussian `randn(K, D)`, the historical default (left un-normalized; the address normalizes per use).
- `'fibonacci'` — **super-Fibonacci spirals** (Alexa, CVPR 2022), D=4 exact:

  ```python
  PHI = √2;  PSI = 1.533751168755204288118041
  s = (i + 0.5)/n;  r = √s;  R = √(1−s)
  α = 2πi/PHI;  β = 2πi/PSI
  q_i = [r·sin α, r·cos α, R·sin β, R·cos β]      # (n, 4) unit quaternions
  ```

  Deterministic, low-discrepancy, near-uniform on S³. Purpose: *start the codebook inside the ℝP³ attractor basin* that trained sphere-solvers were repeatedly observed to converge to, instead of spending gradient learning into it. (For D≠4 a seeded, normalized Gaussian fallback is used.)
- a caller-supplied `(K, D)` array — row-normalized as given. This is the hook for geovocab pentachoron vertices and farmed/extracted codebooks.

`freeze_codebook=True` registers the axes as a fixed buffer — no moving pressure. The code comments are explicit about when that is safe: only after a drift check confirms the init *is* the attractor; otherwise the rest of the model converges around a wrong codebook.

### 3.4 The aleph address — the load-bearing operation

For every spherical row `m ∈ S^(D-1)` (there are `B·N·V` of them per batch):

```python
A   = F.normalize(codebook, dim=-1)     # (K, D) unit axes
cos = M_rows @ A.T                      # (·, K)  signed alignments
u   = cos / τ                           # address temperature τ = address_tau
```

The address is defined as the softmax over the **2K oriented half-axes** `[+A; −A]`. Because the axes are antipodal, that mixture has an exact closed form that never materializes a 2K-wide tensor:

```
M̂ = Σₖ 2·sinh(uₖ)·Aₖ  /  Σₖ 2·cosh(uₖ)
```

computed stably by factoring out `m = max|u|`:

```python
ep  = exp(u − m);  en = exp(−u − m)       # ∝ e^{+u}, e^{−u}
num = ep − en                              # ∝ 2 sinh(u)        (·, K)
den = (ep + en).sum(-1).clamp_min(1e-12)   # ∝ Σ 2 cosh(u)      (·, 1)
M̂  = (num @ A) / den                      # (·, D)
```

**Why this form, term by term:**

- *Why softmax over 2K oriented axes:* the codebook is projective (axes), but a reconstruction target needs orientation (vectors). Concatenating `[+A; −A]` and softmaxing scores every axis in both signs; the winning sign is the aleph's "± bit." The identity `softmax over [+u; −u] mixture of [+A; −A] = Σ sinh(u)A / Σ cosh(u)` follows from `e^u − e^{−u} = 2 sinh u` and `e^u + e^{−u} = 2 cosh u`; the implementation was verified equal to the explicit 2K softmax to ~1e-15 across τ.
- *Why the closed form and not the explicit softmax:* memory. At batch 2048 on byte-trigram there are ≈16.7M rows; the explicit `(rows, 2K)` tensor plus the `[+A; −A]` concat was the dominant allocation (~8.6 GB/step) and the training memory wall. The closed form keeps only K-wide intermediates.
- *Why the `max|u|` subtraction:* numerical stability at small τ. Raw `sinh/cosh` overflow fp32 below τ ≈ 0.02; factoring the max out makes the expression valid for any τ.
- *Why it trains the codebook:* `M̂` is what the decoder reads (§3.7). The reconstruction loss therefore backpropagates through `num @ A` and `cos = M @ Aᵀ` into the codebook parameter. The address is in the gradient path — the defining difference from every diagnostic codebook the program extracted before June.

**Hard mode** (`address='hard'`, the published checkpoints):

```python
idx    = cos.abs().argmax(-1)               # winner axis by |cos|
M_hard = sign(cos[idx]) · A[idx]            # the oriented winning axis
M̂     = M_hard + (M_soft − M_soft.detach())  # straight-through estimator
```

Forward emits the fully **discrete** oriented code; backward uses the soft mixture's gradient. This is the VQ-style regime — and the regime in which the hosted models hold cosine 0.992–0.997 reconstruction.

**`address='none'`** sets `M̂ = M` and registers no codebook parameter at all. This is the *recon-real tied autoencoder* — the gate experiment preserved inside the model class as its own control. It is not the aleph model; it is the proof that made the aleph model worth building.

### 3.5 Chunking and checkpointing the address (the VRAM contract)

Rows are independent, so the address is computed in row chunks with a closed-form chunk size (`_resolve_chunk`):

```python
LIVE   = 6                      # live (c, K) buffers: cos/u, ep, en, num + slack
TARGET = 2 GiB                  # target working set
c = TARGET // (LIVE · K · elem_bytes)
```

In the training hot path each chunk is gradient-checkpointed: forward stores only the `(c, D)` input, backward recomputes one chunk's `(c, K)` tensors at a time — peak address memory is ~one chunk in both directions, exactly, because rows are independent. The comment in code calls the chunk size "a wide plateau, not an optimum — no sweep needed." `address_chunk` overrides it explicitly. This is the mechanism that lets a 26K-parameter model with a 16.7M-row address train on a consumer GPU.

### 3.6 The readout: U, S, Vt without an SVD

`readout='linear'` (the sphere-solver convention, `svd_mode='none'` in SVAE terms):

```
U  = M̂                       (B, N, V, D)
S  = ‖M̂‖ over rows (dim=-2)  (B, N, D)    — column norms, the omega token
Vt = I_D
```

**Purpose:** `S` — the per-patch column-norm vector — is the **omega token**, the spectral summary every downstream consumer reads. Under the linear readout it is exactly the column norms of the addressed matrix; the decode path `U·diag(S)·Vt` collapses to `M̂` itself, so nothing is lost and nothing extra is learned. An `'svd'` readout is reserved in the code for the geometric/Blackwell path and wired separately; when SVAE models do run true SVD, it routes through `geolip-core`'s `batched_svd`/`FLEigh` Gram-eigendecomposition (`G = MᵀM`, `σᵢ = √λᵢ`, `uᵢ = M vᵢ/σᵢ`) with fp64 under a disabled autocast — SVD accuracy depends on fp64, and bypassing that is one of the lineage's known silent failure modes.

### 3.7 The decoder strategies ("the avenues")

`decode_mode` selects how `M̂ → patch`. All three read `svd['M_hat']` — never `svd['M']`. Reading `M` would silently cut the codebook out of the gradient path; the model would still train, but the address would stop being load-bearing.

- **`'tied'`** — `patch = Linear(V·D → patch_dim)(M̂)`. One linear map, no accumulator; `M̂` must *linearly* explain the patch. The validated path and the default. 26,795 params total at reference config.
- **`'dict'`** — sparse coding: `code = softmax(code_proj(M̂)/code_tau)`; `patch = code @ atoms` with `atoms ∈ (n_atoms, patch_dim)` initialized at `0.02·randn`. Embedding-real by construction; non-convolutional, non-transformer.
- **`'mlp'`** — the original SVAE deep residual decoder over `U·diag(S)·Vt`, with decoupled capacity (`dec_hidden`, `dec_depth`). With a hard/discrete address this is the principled VQ decoder reconstructing from codes; with a continuous `M̂`, a large MLP drifts back toward the faux-embedding regime — the code says so on purpose, and selecting it *is* the SVAE-continuity control.
- Room is explicitly reserved for `'rotor'`/`'cayley'` norm-preserving rotational decoders.

### 3.8 Spectral cross-attention (optional, bounded)

The only cross-patch operation. Default **off** for the aleph (`n_cross=0`; the hosted configs all train without it). When present, it acts multiplicatively on the omega tokens:

```python
S_out = S · (1 + α · tanh(out_proj(SDPA(S))))     # α = max_α · sigmoid(α_logits)
```

α initializes near 0.024 (`alpha_init=-2.0, max_alpha≈0.2` in the transformer stack), so the layer starts near-identity and can only *modulate* the spectrum, never inject content. Unbounded α was tested and poisons the spectrum — the bound is load-bearing.

### 3.9 BoundarySmooth (the stitch repair)

```python
net = Conv2d(C → mid, 3) → act → Conv2d(mid → C, 3);  zeros_(last.weight, last.bias)
recon = x + net(x)
```

A ~600-parameter residual conv applied once after `stitch_patches`. Zero-initialized — identity at init, gradually learns to erase patch seams. It is the only convolution in the model and it exists purely for the patchwork boundary.

### 3.10 Training objective and the health metrics (`train_aleph.py`)

**Reconstruction loss** (`loss_mode`):

```python
'mse'        : F.mse_loss(recon, images)
'cosine'     : (1 − cos(recon.flatten(1), images.flatten(1))).mean()
'cosine_mse' : cosine + mse
```

Cosine is per-image over the flattened `C·H·W` — it scores *direction*, the quantity a spherical latent encodes, and on zero-centered images it is not swamped by DC offset. Pure cosine is scale-blind, so `cosine_mse` restores amplitude when it matters (it is the mode used for `image_aleph_64`). Byte-trigram trains on `'mse'`. **There is no CV penalty anywhere** — column-norm CV is logged read-only:

```python
CV = std(‖M‖_col) / mean(‖M‖_col)
```

a cheap structural probe of spectrum self-organization. (The lineage learned this the hard way: an audit proved the historical "CV loss" had been gradient-free the entire time — `.item()` calls stripped the graph — so CV was always a *readout*, never a force. The aleph trainer makes that honest by design.)

**Address health** (`_address_stats`, computed in eval where logits are emitted):

```python
a      = softmax(aleph_logits / τ, dim=-1)          # (B, N, V, 2K)
usage  = a.reshape(-1, 2K).mean(0)                  # soft batch-mean usage
soft_ppl = exp(−Σₖ usageₖ · log usageₖ)             # effective oriented axes (soft field)
margin   = mean(max_k a)                            # mean top-1 address probability
hu       = histogram(argmax rows) / total           # discrete usage
hard_ppl = exp(−Σₖ huₖ · log huₖ)                   # effective oriented axes (argmax)
```

- **Perplexity** is the exponentiated entropy of codebook usage — "how many oriented axes are actually in use," out of `2K`. `hard_ppl` is the one that matters for `address='hard'`: it counts axes that *win*. Hosted runs sit at soft ppl ≈ 125–126 and hard ppl ≈ 112–122 of 128 — the codebook is fully alive.
- **Margin** here is the mean top-1 *softmax probability* — decisiveness of the soft field at the training temperature (0.225–0.232 at τ=0.1 across the three hosted runs). Note carefully: this is a different quantity from the *projective margin* `|⟨m, a⟩|` used by the assessor and the extracted-codebook comparisons (where the gate codebook scored ≈ 0.967 vs ≈ 0.929 for SVAE batteries). The first is a probability over 128 outcomes; the second is a cosine magnitude. Both are reported as "margin" in their own contexts; they are not interchangeable.

**Anti-collapse term** (off by default, `div_weight=0`):

```python
loss += div_weight · Σₖ usageₖ · log usageₖ        # negative usage entropy
```

Minimizing negative entropy pushes batch-mean usage toward uniform — raising perplexity directly. Enabling it sets `model._emit_logits = True`, because the term needs the `(·, 2K)` logits that the training hot path otherwise never materializes (that allocation is ~8.6 GB/step at batch 2048 — the reason logits are lazy). House discipline: run `div_weight=0` first and *watch* whether collapse actually happens before regularizing it away. On the hosted runs it never did.

**Optimizer:** pure `torch.optim.Adam` + `CosineAnnealingLR(T_max=total_steps)`. Never AdamW — weight decay fights the geometric structure (ablation-validated, a standing rule across the whole repo).

### 3.11 The diagnostic geometry (inference stack)

These operate on any model with the shared forward contract — SVAE batteries and aleph alike.

**Sign canonicalization onto ℝP^(D−1)** (`canon`): flip each vector so its first nonzero coordinate is positive; antipodes map to the same representative.

**Antipodal collapse** (`identify_antipodal_pairs` + `collapse_to_axes`): on the V unit rows, compute the cosine matrix; for each row take its most-negative partner; accept the pair if `cos < −0.9` and the match is mutual (or symmetric-below-threshold); greedily claim sorted by strength. Each pair merges to one axis:

```
axis = (uᵢ − uⱼ) / ‖uᵢ − uⱼ‖          (then sign-canonicalized)
```

unpaired rows pass through canonicalized. This deterministic tensor operation — not a learned property — is how every projective codebook in the program is read out of a trained model.

**Uniformity deviation**: mean pairwise projective angle `acos|cos|` of the axes, minus the same statistic for 4096 uniform random projective points at the same D. Statute classification: `dev > +0.05` polytope-class (repulsive packing, pair fraction ≥ 45%); `|dev| < 0.05` uniform-class; `dev < −0.05` degenerate (clumping — excluded).

**Topology probes** (`train_codebook.py`): (A) kNN-graph connected components swept over angular thresholds, with the percolation angle (first θ where the largest component holds ≥ 50%); (B) local intrinsic dimension by k-NN PCA — eigenvalue count above 5% of leading, plus participation ratio `(Σλ)²/Σλ²`; (C) **persistent homology via ripser** on the projective angular distance matrix up to a threshold (default 20°), maxdim 2 — finite H₀/H₁/H₂ birth–death features. **β₂ per axis** — finite H₂ features divided by axis count — is the void metric the model is named for.

**Effective rank** (spectral diversity): `erank = exp(−Σ pᵢ log pᵢ)`, `pᵢ = σᵢ/Σσ`.

**The aleph assessor** (`AlephAssessor`, read-only):

```python
a      = canon(normalize(M.detach()))
proj   = a @ codebookᵀ                  # signed projection
margin = |proj|                          # antipode-invariant
axis   = argmax(margin);  sign = sgn(proj[axis])
assign = softmax(margin / temp)
```

Never trains anything (M is detached) — it is the metric for a fractally-bound space where raw cosine saturates: two inputs that cosine collapses can still differ in their signed address pattern across the V rows. This *is* the aleph logit in read-only form; `AlephModel` is the same operation moved into the gradient path.

### 3.12 The AlephTransformer (the multiscale shell)

Mirrors the tested SVAE-transformer over a **frozen** aleph battery:

```
frozen aleph → stem rows (M̂ direction by default, or raw M as control)
            → SingleLens: fixed orthonormal E = QR(randn(D_lens, D_base)),  ⟨Ex, Ey⟩ = ⟨x, y⟩
            → omega = ‖M_lens‖ over rows → SpectralAlphaStack (bounded-α SDPA)
            → GeoDecoder → external recon
```

- The **lens is a buffer** — a fixed isometric lift onto a great subsphere of S^(D_lens−1). Zero parameters, geometry carried up exactly, "the wall the architecture's stability rests on." `lens_sign='signed'` preserves the per-row sign channel to the decoder (the address channel); `'canon'` drops it onto ℝP — the ablation.
- `stem='m_hat'` feeds the *addressed* direction, so the macro shell **strengthens the aleph address** natively at D_lens — no learned up-projection. `stem='m'` is the SVAE-equivalent control that ignores the codebook.
- The aleph is frozen and read under `no_grad` at a detached boundary; only the transformer + decoder train (`shell_parameters()`), with plain MSE on the external recon and the same Adam house rule. The assessor rides along read-only, scored against the aleph's *own* learned codebook.

### 3.13 Aleph-routed attention (added 2026-06-09): the address as a feature map

Session artifacts `aleph_routed_attention.py` (hub + bucket variants, streaming),
`aleph_trigram_lm.py` (causal trigram LM + statute trajectory), `aleph_probe_battery.py`
(the eight-probe battery). Every formula below shipped and every number was logged on
2026-06-09. Structure first, purpose second, and — because this system is structurally
different from the autoencoder — each entry states its **differentiation**: the role the
same machinery plays in routing versus reconstruction.

**The routing identity (hub mode).** Per head, queries and keys are projected to
`D_addr` and sphere-normalized (`q̂, k̂ ∈ S^(D_addr−1)`, orthogonal-init projections —
the encoder invariants apply unchanged). The aleph address `p(x) = softmax([u; −u])`,
`u = (x̂Aᵀ)/τ`, is computed in the antipodal-factored form:

```
m  = max|u|;  ep = e^(u−m);  en = e^(−u−m);  Z = Σ(ep+en)        # Z ≥ 1
p₊ = ep/Z;    p₋ = en/Z                                            # (·, K) each
```

and the attention is the kernel attention whose feature map IS the address:

```
score(i,j) = ⟨p(q̂ᵢ), p(k̂ⱼ)⟩ = p₊(q̂ᵢ)·p₊(k̂ⱼ) + p₋(q̂ᵢ)·p₋(k̂ⱼ)
M± = Σⱼ p±(k̂ⱼ) vⱼᵀ          (K, d_head)        z± = Σⱼ p±(k̂ⱼ)    (K,)
outᵢ = (p₊(q̂ᵢ)M₊ + p₋(q̂ᵢ)M₋) / (p₊(q̂ᵢ)z₊ + p₋(q̂ᵢ)z₋)
```

**Purpose of each piece:**

- *K-wide factorization* — the same antipodal closed-form trick as §3.4: the `2K`
  tensor is never materialized; only `(·, K)` intermediates exist.
- *Strictly positive denominator* — the address distributions are positive, so the
  normalizer can neither vanish nor flip sign: the instability of elu+1-style linear
  attention is structurally absent.
- *O(n·K·d), pure GEMM* — no gathers, no S×S matrix, no route tables. This is the
  hardware-sympathy answer to the Cantor router: the geometry and the silicon vote
  the same way. Measured: 1.4× vs softmax at S=1024, 4.1× at 4096, softmax OOMs at
  16,384, hub runs 65,536 tokens in 67 ms (A100-80GB, dim 512).
- *Rank bound 2K* — the attention matrix factors through the address, so its rank
  is at most `2K`: **K is the bandwidth knob, τ is the hardness knob** (τ→0 recovers
  discrete bucket routing; τ→∞ collapses to mean-pooling).
- *Quality* — softmax parity on associative recall and on the byte-trigram causal LM
  (bpb 2.835 vs 2.836 at 10k steps, 6.7M params, the only-delta control of #26).

**Differentiation:** in the autoencoder the address produces `M̂`, the decode source
(gradient: recon → M̂ → codebook). In routing the address produces a *communication
kernel* (gradient: task loss → score → codebook). Same formula, opposite residence
for the content — see the two-hemisphere law below.

**Bucket mode (the discrete sibling).** `bucket = argmax_k |u_k|` with sign — each
token's winner oriented half-axis is its clique; exact softmax attention inside
sorted equal-width windows, masked to same-bucket pairs; the codebook's gradient path
is a differentiable address-agreement bias added to the scores. Status: trains but is
gradient-starved relative to hub (hard assignment gives the address projections almost
no alignment signal from cold start); the principled rescue is a hub→bucket curriculum
with transplanted codebook + projections. Open (§12).

**The streaming recurrence (the codebook as memory).** The hub's causal form is a
recurrence with constant-size state:

```
state = (M₊, M₋, z₊, z₋)        # (B,H,K,d_head)×2 + (B,H,K)×2 — size independent of past
out_seg, state′ = recurrence(segment, state)
```

Verified **bit-exact** against the full causal pass (same chunk boundaries ⇒ identical
op order; max error 0.0e0). `M±` is literally *what has been written to each oriented
axis so far* — the document accumulates into the vocabulary. TBPTT-1 discipline:
gradients flow within a segment; values flow forever. Context is unbounded at constant
attention memory. **Differentiation:** the autoencoder has no temporal state; this
statistic class is routing-only.

**The occupancy statistic (the compressibility meter).**

```
occupancy = (erank(M₊) + erank(M₋)) / (2·min(K, d_head)),   erank = exp(H(σ/Σσ))
```

(fp64 singular values, per the house SVD invariant). Measured at 6,144 bytes of
streamed context: **text 16% (≈10/64 effective dimensions) vs noise 45% (≈29/64)** —
language concentrates into a handful of axis-directions; incompressible noise sprawls.
The streaming state is an entropy-rate proxy delivered by the architecture for free,
and it implies large context headroom for language at K=64. **Differentiation:** the
memory-side twin of the depth-funnel result (#31): the trained system *organizes* text
and *releases* noise, in both space (layers) and time (state).

**Confidence is a kernel invariant (the 0.857 monitor, closed).**

```
conf(x) = ‖(p₊(x) − p₋(x)) · A‖  =  f(τ, K, D)        # data-independent
```

τ-sweep on the trained model, an untrained twin, and pure random rows agree to three
decimals at every τ (0.8532 / 0.8543 / 0.8547 at τ=0.1). Confidence is address
*physics* — the norm of the soft codebook reconstruction of a generic unit row —
and carries zero learned information. Monitor reclassified: sanity check, not signal.

**The codebook under routing gradients (the restoring force, #26).** Shared across
all 4 layers at **zero quality cost** (the only-delta control: bpb 2.835 vs 2.836),
concentrating address pressure 4×. Under that pressure the random-init codebook's
uniformity deviation traced **−0.0022 → −0.0070 (step 2000) → −0.0004 (step 10000)** —
pushed out, pulled back, ending *closer to uniform than it started*, with the recovery
beginning at ~84% of base lr (not a decay artifact). Simultaneously the antipodal
parameter-pair fraction fell monotonically **53% → 38%** at constant uniform spread
with rising address margin: near-antipodal parameter pairs are projectively redundant
(`+A` already serves `−A` through the sign channel), and training prunes them —
vocabulary-coverage optimization. The learned routing codebook converges on the same
"near-uniform, alive, decisive" endpoint the recon-aleph reaches unforced.

**The type codebook (the extraction analogue, [XB2]).**

```
T_t = normalize( mean over occurrences of trigram type t of k̂ )    # the data's vocabulary
axes = collapse_to_axes(identify_antipodal_pairs(T))                # the program's exact rule
profile = persistence(axes; canon'd arccos(dot); θ ∈ {20,30,45,60}°)
```

The published β₂ references (0.56 recon-aleph / 0.08 SVAE) are **extracted** codebooks;
the type codebook is the routing-side analogue with the extraction performed by the
shipping `inference/codebook.py` rule, replicated verbatim. Measured: text types
**degenerate-class (dev −0.0624)** — the first formal statute-boundary crossing in the
routing system — vs noise types uniform (−0.0062); β₂ text 0.018 ≈ noise 0.019 (both
void-sparse). Caveats on record: head-averaged addresses may compress spread (per-head
variant cleaner); frequent English trigrams share bytes (compositional-correlation
confound); the type-level untrained control is pending (the cloud-level control was
decisive: untrained = flat zero at all layers, both calibrations).

**The two-hemisphere law (refines discovery #20).** Same substrate, same instrument,
opposite writing:

| | objective | in-dist extraction deviation | statute | β₂/axis | pressure |
|---|---|---|---|---|---|
| reconstruction | separate every code | **+0.083** | polytope | **0.56** (void-rich) | repulsive |
| routing | associate related tokens | **−0.062** | degenerate-ward | **≤0.02** (void-sparse) | attractive |

Mechanism: H₂ voids require points scaffolded *around* holes — repulsive polytope
packing builds that scaffolding; attractive clumping collapses it. Therefore: *within*
the reconstruction family (objective-sign constant) voids fingerprint the substrate
(#20 stands); *across* objective families voids fingerprint **substrate × objective-
sign**. The objective decides which way the language bends against the frame; both
learned frames inhabit the same near-uniform attractor. The shared scaffold physics
also extends quantitatively: the address projections' raw-weight column CV lands at
0.136/0.158 — inside the CM band (#4) on an architecture class that did not exist
before this session — and the projective margins (q 0.9457 / k 0.9476) sit between
the faux-embedding batteries (0.929) and the recon-real gate (0.967).

**Cross-jar identity (open).** ICP Procrustes (sign-aware assignment + orthogonal
alignment, fp64 SVD) of the routing codebook against the hosted
`aleph_byte_trigram_tied_hard_K64` codebook: residual 19.22° vs uniform-pair null
21.28 ± 1.43° — **z = −1.43, suggestive, not significant**. The sharpened successor
experiment is **type-matched**: the same 64 trigram types represented in both systems
with correspondence *known* (no assignment ambiguity); if the relational geometry
among the types transfers across jars beyond the rotation null while the surface signs
flip, the conserved object — the lightning — is the type relation structure. Queued.

---

## 4. The data types, and why they are shaped this way

The model accepts `(B, C, H, W)` float tensors in [−1, 1] with `H, W` divisible by the patch grid. What goes *into* that container is the interesting part — the program treats the image tensor as a universal byte-bus.

**Natural images** — tiny-imagenet 64×64 for the hosted `image_aleph_64`; ImageNet for the Fresnel ancestors. Trained with `cosine`/`cosine_mse` because a spherical latent encodes direction.

**Noise, 16 types** — the substrate the battery tier was raised on and the calibration registry serves (`gaussian`, `uniform`, `sixteen_noise` generators; the 16-mix covers gaussian, uniform, scaled-uniform, poisson, pink, brown, salt-pepper, sparse impulses, block-upsampled, gradient-gaussian, checker, gauss-uniform mix, four-quadrant, cauchy, exponential, laplace). Noise calibrations are the out-of-distribution probes for codebook extraction.

**Text as byte-trigram images** — the substrate this model was gated on. `ByteTrigramDataset.bytes_to_image`:

```
UTF-8 bytes, padded to img_size²·C
→ each consecutive C bytes = one cell's channel tuple
→ value = (byte − 127.5)/127.5  ∈ [−1, 1]
→ packed row-major across patches, row-major within patch
```

`channels=3` makes byte *trigrams* (an RGB-pixel-equivalent per cell); `channels=4` makes quadgrams — the trigram is channel-count, not a tokenizer. Why this shape: the failed first attempt (SentencePiece bit-planes padded with hard zeros) left patches ⅓-filled and starved the model — the standing rule extracted from that failure is that **every float in every patch must carry signal**; per-cell information cardinality should approach 256³. Byte-trigrams engaged on the first try, and the encoding is exactly invertible (`image_to_bytes`), which is what makes round-trip text recovery measurable.

**SentencePiece bits, binary trees, ternary codes** — proven on the H2 batteries: SentencePiece bit-planes are valid (MSE 5.78e-6); binary trees under hierarchical ±1 encoding train "a uniformly potent gating mechanism"; the models respond directly to −1/0/+1 ternary. These are the forward-looking substrates for the vocabulary-patchwork role: anything serializable to bytes is admissible, and the symbolic ones are precisely the substrates whose codebooks come out void-rich.

---

## 5. The structured patchwork system, and why

The patch is the atom of the whole system, and the consequences are measured, not asserted:

- **Resolution invariance is architectural.** Every component operates per-patch except the (optional) cross-attention — and even its weights (QKV, out-proj, α) are dimensioned by D, not by patch count N. Weights trained at N=256 run at N=4096 by construction; measured MSE varies 4.5% across 81 → 4096 patches, and reconstruction MSE is essentially constant (within ~1%) across a 36-config sweep of sizes 128/256/512 and tile sizes 32–512 on the h2-64 array. Inverted, this flatness is a debugging canary: any resolution-dependent MSE shift at fixed precision means something upstream broke.
- **Precision floors are mantissa-determined, not regime-determined.** bf16 reconstruction sits at ~7× the fp32 floor (8.08e-3 ± 0.04e-3, stable across all 36 configs) because the M rows live on the unit sphere and a 7-bit mantissa quantizes angular differences coarsely; fp16 sits ~5% above fp32.
- **The patch is the vocabulary slot.** Per patch, the encoder writes V=32 spherical rows; the address assigns each row an oriented axis; under `address='hard'` a patch literally *is* a V-tuple of discrete oriented codes from a 2K-symbol alphabet. That is the "relational geometric vocabulary patchwork" of the title: a lattice of patches, each carrying a small relational sentence of codebook addresses, with the relations (the codebook geometry — its angles, its statutes, its voids) learned once and shared by every patch at every resolution.
- **The inference layer enforces "no core limiters."** `patch_size`, `tile_size`, calibration distribution, codebook source, aggregation — all overridable at `InferenceEngine` construction or per call. The model fails loudly on overrides it can't tolerate; silent hard-coded assumptions are treated as regressions.

---

## 6. Behavioral responses the model invokes in other models

The aleph is built to be *consumed* — frozen, addressed, and read by hosts. What is known, both directions:

### Successes

- **Frozen-battery teaching (the precedent).** A frozen Freckles battery under a classifier head reached 76.0% CIFAR-10 (Omega v2, 935K classifier over 2.5M frozen params). The end-to-end SVDTransformer answer reached 67.7% at 452K params total — 5× smaller than the Omega v2 classifier alone — establishing the trade the shells now navigate.
- **The transformer shell strengthens the address.** The SVAE-transformer v2 converges semi-close to the SVAE spectrum in considerably fewer epochs at ~90K shell params; the `AlephTransformer` repeats that structure with the addressed rows as the stem, so shell gradient pressure lands *along the codebook directions*. The lens being an exact isometry is what makes "strengthens" well-defined — the shell cannot leave the envelope.
- **Per-bank MSE signatures as model-to-model language.** The `BatteryArrayModel` (192-bank h2-64) exposes a per-bank MSE signature through a single HuggingFace `AutoModel` — the array *responds* to an input with a fingerprint over experts, which downstream models read as a feature. The aleph's discrete addresses are the sharper successor to that signature.
- **Cross-model conditioning at scale (in progress).** `geolip-sdxl-aleph` Phase 0 trains a cross-attention LoRA on the SDXL UNet with Qwen replacing CLIP-G next to an encoder-invariant geometric address under a rectified-flow objective; Phase 1 (full finetune, 10 epochs at 86k images) is running as of June 7, 2026. The announced plan names the architecture as the address layer for the first large-scale distillation: "the mathematical aleph procrustes geofractal addressed language latent."
- **Read-only assessment of other latents.** The `AlephAssessor` attaches to any sphere-normalizable representation and produces axis/sign/margin/assign without training — it was designed as the metric for spaces where cosine saturates, and it is how address consistency is monitored inside the shells.

### Failpoints (each one cost real experiments to find)

- **Single-instance illegibility (S-class).** A lone Freckles instance fails to teach downstream — its omega tokens are not legible without the full array. Aggregation is part of the contract, not an optimization.
- **Global average pooling destroys the signal**: ~70% → 29% accuracy. Flatten or use spatial statistics; never GAP a geometric encoder.
- **`patch_idx=0` sampling** carried silently from training-time measurement into production probes and cost ~88% of the spatial signal in downstream classifiers (1/256 patches read). Patch aggregation now defaults to `'mean'`; the old path survives only as an opt-in reproduction flag.
- **Cross-model codebook misprojection** is a hard-to-debug silent failure; `engine.attach_codebook` therefore raises `CodebookIncompatibleError` on D mismatch unless compatibility checking is explicitly disabled for deliberate cross-model experiments.
- **Codebook collapse** is the aleph-specific failure mode: a few axes absorb all rows, perplexity craters. Monitored every eval; treated with `div_weight=0.01` only after observing it (it has not occurred on the hosted runs).
- **Architecture-identity mismatches on load** (`smooth_mid`, `n_heads`, missing ablation flags) silently partial-load under `strict=False` and produce garbage reconstructions; the repo's standing debug move is to diff instantiated state-dict keys/shapes against one real checkpoint before any loader work. The aleph reduces this surface — its config round-trips completely — but `address='none'` builds have *no* codebook key at all, so a "missing codebook" on load is a config mismatch, not corruption.
- **bf16 training/inference** elevates the reconstruction floor ~7×; acceptable under memory pressure, never for fidelity claims.

---

## 7. Strengths as a tokenized primary backup structure

The byte-trigram substrate makes the model a *reversible* store, and this was measured:

- **Round-trip text recovery is a first-class metric.** `text_recovery_metrics` runs text → bytes → image → model recon → bytes → text and reports `real_byte_acc`, `real_byte_l1`, and the recovered string itself. The H2 batteries hold **99.6% n-gram recall** byte-by-byte on this loop.
- **Vocabulary capacity outruns the parameter count.** The measured potential is a **16.77M-entry vocabulary** over ~5M unicode byte combinations, carried by models in the 26K–57K parameter range — because the store is the codebook geometry plus a linear map, not a lookup table.
- **The discrete address is the backup code.** Under `address='hard'`, each patch serializes to V oriented axis indices (a few hundred bits at reference config) while reconstructing at cosine 0.992+ — and `hard_ppl ≈ 112–122/128` says the code actually uses its alphabet.
- **The artifacts are absurdly portable.** The tied model is a 114 KB checkpoint, self-describing (`get_config` round-trips every constructor argument; no side files), loadable to eval in one call. A vocabulary's entire learned geometry travels as an email attachment.
- **Determinism where it matters.** Codebook *extraction* is a deterministic tensor operation on a frozen model; calibration generators are seeded; the trainer caches codebooks. Two parties holding the same checkpoint derive the same axes.
- **Compression context.** The ancestor Fresnel holds 99.993% fidelity at 48:1 compression on ImageNet-1K with a 1,024-value latent — the patchwork family was already a strong lossy store; the aleph adds the discrete, addressable, recoverable layer on top.

---

## 8. The implicit learned structure (what the test battery actually found)

Behind the architecture sits the empirical battery — hundreds of trained models in notebooks across the repos ([geolip-hypersphere-experiments](https://huggingface.co/AbstractPhil/geolip-hypersphere-experiments) alone ships 21 notebooks of the Omega-Aleph arc, including the massive battery sweeps and the Alexandria ablations). What they converge on:

- **Trained sphere-solvers ARE projective codebooks.** Antipodal-collapse reads a near-uniform ℝP^(D−1) codebook out of every healthy sphere-solver: 19 models at D=3/4/5, then the 192-bank array (per-bank deviation +0.010 ± 0.013, 24–27 axes per bank). Reading, not training, was the discovery.
- **There are (at least) two stable statutes, selected by substrate.** Uniform-class (|dev| < 0.05, the noise solvers) and polytope-class (dev > +0.05 with ≥45% antipodal pair fraction, the byte-trigram solvers, in-distribution dev +0.083 / 52% pairs). The statute is a property of *(model × calibration)* — the same weights probed OOD with noise read uniform-class (+0.046, 39%). Direct evidence of the "self-solving frame": one architecture, two tasks, two stable geometries.
- **Voids fingerprint the substrate.** Across 41 frozen batteries, six model classes, three substrate types: continuous-image codebooks are void-sparse, symbolic-vocabulary codebooks are void-rich (finite β₂ in projective persistent homology), at fixed D where first-order geometry cannot tell them apart. The aleph's learned codebook lands at β₂/axis ≈ 0.56, ~7× its SVAE ancestors — learning under a recon-real objective *amplified* the substrate fingerprint.
- **The CV bands are ambient-dimension physics, not training outcomes.** Raw-weight CV bands (0.13 < CV < 0.30 band-valid; D=24 the phase boundary at the binding constant **0.29154**; vocabulary size irrelevant from V=32 to 13M) were measured over 65,536 configs and do not move under training. Trained *activation* CV instead converges to the uniform-sphere value for its D (the tri-band result). And D=4 — the aleph's home — extrapolates to CV ≈ 0.9, far above the band: the model lives in the volatile regime *on purpose*, which is exactly why its defense stack (sphere-norm, bounded α, zero-init smoothing, fp64 spectral path) is load-bearing rather than cosmetic.
- **The same constant keeps surfacing.** 0.29154 appears as the CM CV phase boundary, the Nikola resonance gate, and the Constellation anchor-drift threshold — five architectures across three paradigms. The program records two honest framings: (A) a genuine constant of a universal geometric substrate, or (B) a strong shared prior many architectures converge to. The evidence tilts A but is not conclusive, and the published material says so plainly.
- **Perplexity says the codebook is alive.** 125–126 of 128 oriented axes in soft use, 112–122 by hard argmax, margin stable — across both substrates and all three hosted runs, with `div_weight=0`. The near-uniform attractor the diagnostics kept finding in *extracted* codebooks shows up unforced in the *learned* one.

**[2026-06-09 update — the statute mechanism extends to attention, with a depth
dynamic the per-patch systems could not show.]** The routing system's layer-0 address
clouds read uniform for both text and noise — but the trained-vs-untrained control is
decisive (untrained: dev ≈ 0 at *all* layers, both calibrations), so the in-distribution
deviation that exists (text −0.019 at L0) is 100% learned. With depth, the trained
tower is a **funnel**: text organizes monotonically (−0.019 → −0.043 → −0.057 → −0.081,
L0→L3) while noise is transiently clumped mid-tower and then *released* back toward
uniform (−0.0005 → −0.071 → −0.057 → −0.041). The substrate signature lives in the
trend direction at depth, not the level at any layer. The mundane component is known
transformer anisotropy; the content-dependent differential release is not. At the type
level the boundary formally breaks: text type codebook −0.0624 (degenerate-ward) vs
noise −0.0062 — the attractive hemisphere of the two-hemisphere law (§3.13, #27).

---

## 9. The ease-of-use system

The whole loop — install, train, save, host, load, probe — is four commands and is the same loop that produced every hosted artifact.

**Install** (one line; the cascade pulls the entire architecture):

```bash
pip install "git+https://github.com/AbstractEyes/geolip-svae.git"
```

`geolip-svae` (v0.9.6) declares `geolip-core` as a git dependency — the FLEigh/conduit/`batched_svd` kernel layer arrives automatically. The wider program (geofractal v1.2.0 → geometricvocab v0.1.2 + wide_compiler v0.7.0) shares the same single-install discipline; the components interoperate because they are one codebase with named surfaces, enforced-compatible through the dependency graph.

**Train** (`train_aleph.py` — Colab-first, tqdm progress, TensorBoard + HF streaming):

```python
from geolip_svae.train_aleph import train_aleph
model, report = train_aleph(
    dataset="byte_trigram",            # or "tiny_imagenet"
    decode_mode="tied",                # 'tied' | 'dict' | 'mlp'
    address="soft",                    # 'soft' | 'hard' | 'none' (the gate)
    cfg_overrides=dict(K=64, address_tau=0.1),   # div_weight=0.01 if collapse observed
    hf_repo="you/your-aleph",          # optional: streams checkpoints + tensorboard
)
```

Presets pin the *exact* dataset path that produced the SVAE batteries (`get_dataset_bundle`; WikiText-103 byte-trigram at 64×64, 1M train / 10k val, batch 256, lr 1e-3, Adam + cosine schedule), so every aleph MSE is directly comparable to the battery baselines (byte-trigram 3.8e-7, tiny-imagenet 8e-5). Every `report_every` steps the trainer logs `test_mse, test_cos, test_cv, perplexity, address_margin, hard_perplexity`, uploads TensorBoard, and tracks best-by-objective.

**Save** — `save_aleph_checkpoint(model, path, epoch=, test_mse=)` writes `{config, model_state_dict, epoch, test_mse}`; the config is self-complete (`get_config` round-trips every constructor argument — no `final_report.json` backfill step, unlike v1).

**Load** — `load_model` dispatches on `model_type`: `'v1'` → PatchSVAE, `'aleph'` → `build_aleph`, `'v2'` → a clean `UnsupportedCheckpointError`:

```python
from geolip_svae import load_model
model, cfg = load_model(hf_version="aleph_byte_trigram_tied_hard_K64",
                        repo_id="AbstractPhil/geolip-aleph-void")
out = model(images)                       # (B, 3, 64, 64)
M_hat   = out["svd"]["M_hat"]             # addressed rows (the decode source)
M       = out["svd"]["M"]                 # raw spherical rows (probe/extraction source)
axes    = model.codebook                  # learned (K, D) projective axes
axes2k  = model.oriented_codebook()       # (2K, D) oriented half-axes
```

**Host layout** (what the trainer streams to HF, per version):

```
<version>/
├── checkpoints/best.pt        # load_model(hf_version="<version>")
├── final_report.json          # config + full metric history
└── tensorboard/<run>/         # live curves
```

**Probe** — the inference engine and codebook tooling work unchanged:

```python
from geolip_svae.inference import InferenceEngine, extract_codebook, make_calibration
calib = make_calibration('sixteen_noise', n=64, size=64)
cb = extract_codebook(model, calib, model_id='aleph_bt_tied', calibration_name='sixteen_noise')
cb.save('codebooks/aleph_bt_tied__sixteen_noise')      # safetensors + JSON
engine = InferenceEngine(model); engine.attach_codebook(cb)
```

plus `SentenceEncoder` for text similarity/recovery (`mode='M'` raw rows or `mode='codes'` argmax addresses), `encode_at_scale`/`reconstruct_at_scale` for tiling, and the prototype harness under `prototypes/svae_proto/` (`exp_001_vocab_trigram_recall`, `exp_002_rigid_codebook_implementation`, `exp_003_occupancy_scaling`) for experiment-grade extensions without touching core.

---

## 10. Capabilities today, use cases tomorrow

**Today, with hosted weights:**

- Byte-faithful text encode/recover and sentence-level similarity over the byte-trigram substrate (99.6% n-gram recall lineage; recovery metrics built in).
- Discrete geometric tokenization: per-patch V-tuples of oriented axis codes at cosine-0.992+ reconstruction, 26K params, 114 KB artifact.
- Drop-in codebook research: extract, compare, and topologically profile codebooks against 41+ battery baselines with the same tooling and calibrations.
- Frozen-stem feature supply to shells (`AlephTransformer`) and assessors for any sphere-normalizable latent.

**Tomorrow, grounded in what is already running or announced:**

- **The aleph-addressed language latent for distillation.** The announced first large-scale distillation uses geolip-aleph-void as the address layer — Procrustes-aligned (closed-form, zero-gradient), geofractal-routed. This is Route B of the collective program: don't train experts to speak a language; train the translation codebook and align experts to it.
- **Encoder-invariant conditioning for diffusion.** `geolip-sdxl-aleph` Phase 1 is the live test: Qwen-for-CLIP-G swap held together by a geometric address under rectified flow.
- **The collective substrate (Beatrix).** geofractal v1.2.0 already ships the routers, ports, fusion strategies, and address components; WideCompiler already makes N frozen experts cost ~N/speedup in wall-time (primitive speedups measured up to 174× on A100-class hardware, near-linear tower scaling measured 4→32 towers). The aleph supplies what that machinery was waiting for: a compact, discrete, void-structured codebook that different experts can be aligned *to*.
- **Vocabulary patchworks beyond text.** SentencePiece bits, binary trees, and ternary codes are validated substrates; `codebook_init` accepts geovocab pentachoron vertices directly; `'rotor'`/`'cayley'` decoders and the `'svd'` readout are reserved seams in the shipping code.

- **[2026-06-09]** **The codebook as the medium of attention.** Aleph-routed
  attention (§3.13) is running: O(n·K) GEMM-only routing at softmax parity, with the
  codebook shared across layers at zero cost and a constant-size streaming memory whose
  occupancy meters compressibility. The type-matched cross-jar alignment (same 64
  trigram types in the recon and routing systems, correspondence known) is the queued
  experiment for locating what is conserved between jars.

---

## 11. The discovery catalog

The condensed historical record across the geometric program — geolip, geofractal, geovocab2/lattice_vocabulary, WideCompiler — each entry validated in the linked artifacts:

| # | Discovery | Where established |
|---|-----------|-------------------|
| 1 | The scale-explosion "snap" signature (S₀ and S_D blow up together; ratio collapses; erank jumps; recon breaks) and its mechanism (Adam on ill-conditioned Gram) | V8-amp logs, review session 2026-04-18 |
| 2 | **Sphere-normalization of M rows fixes it by construction** — the founding architectural fact | V8→Freckles arc |
| 3 | The historical CV "loss" was gradient-free all along; CV is a readout, not a force | nonbias code review, 2026-04-18 |
| 4 | CM CV band 0.13 < CV < 0.30; **D=24 phase boundary; binding constant 0.29154**; V irrelevant from 32 to 13M; training does not move raw-weight CV | geolip-deep-embedding-analysis (65,536-config sweep) |
| 5 | 0.29154 recurs independently: Nikola resonance gate, Constellation anchor drift, SVAE cross-attn gating — 5 architectures, 3 paradigms | constellation-diffusion-bottleneck (Mar 18) + temple record |
| 6 | Trained activation CV converges to the uniform-sphere value per D (0.20@16/0.36@8/0.90@4); reproduces in 96% of 149 runs; fails only when row-norm is ablated; CV at 1000 batches predicts final band | geolip-svae-ablations (Apr 22) |
| 7 | Global average pooling destroys geometric signal (70% → 29%) | omega processor experiments |
| 8 | Pure Adam, never AdamW: weight decay fights geometric structure; LBFGS niche only at ≤100-batch budgets | Phase-1 ablation group C |
| 9 | Bounded multiplicative α cross-attention; unbounded α poisons the spectrum | ablation program |
| 10 | Resolution invariance is architectural: 4.5% MSE variation 81→4096 patches; ~1% across the 36-config sweep; flatness is the debugging canary | Freckles v40/v41 + h2-64 sweeps |
| 11 | bf16 floor = ~7× fp32 (8.08e-3), mantissa-determined; fp16 ~5% above fp32 | 36-config precision sweep |
| 12 | fp64 SVD with autocast disabled is load-bearing; Triton fused thin-SVD (5000×) and FLEigh make it affordable | svd-triton articles, linalg-eigh-rehaul |
| 13 | **Trained sphere-solvers are projective codebooks**: antipodal mutual-strongest collapse reads near-uniform ℝP^(D−1) axes out of every healthy solver (19 models D=3/4/5; h2-64 array dev +0.010 ± 0.013) | geometric-tri-band-ft2, implicit-solver-experiments |
| 14 | `patch_idx=0` was a silent ~88% signal loss; patch aggregation must default to mean | inference rebuild postmortem |
| 15 | **Statute classes**: uniform vs polytope vs degenerate, sign of deviation matters; statute is (model × calibration) **[2026-06-09: extends to attention address clouds; first in-distribution crossing in the NEGATIVE direction at the type level (−0.062, degenerate-ward) — see #27, #31]** | tri-frequency-ft3 + 000115 session |
| 16 | Byte-trigram substrate engagement; "every float must carry signal" (hard-zero padding starves patches); channel count = n-gram order | byte_trigram sessions, geolip-svae-text |
| 17 | 99.6% byte n-gram recall; 16.77M vocabulary potential; SentencePiece/bintree/ternary validated | May 1 findings post |
| 18 | Frozen-battery transfer: Omega v2 76.0% CIFAR-10; end-to-end SVDTransformer 67.7% at 452K (5× smaller); S-class single instances don't teach | omega processor + SVD-transformer line |
| 19 | D=24 is simultaneously the geometric phase boundary and the computational SVD-kernel cliff — the recurring dimension choice is not arbitrary | CM CV framework + kernel sweeps |
| 20 | **Voids are a substrate fingerprint**: symbolic codebooks void-rich, continuous void-sparse, at fixed D; requires measuring persistence on ℝP, not S (metric-alignment fix); `omega_phase_v2` taxonomy **[2026-06-09: refined — within the reconstruction family voids fingerprint SUBSTRATE; across objective families voids fingerprint substrate × OBJECTIVE-SIGN; see #27]** | reading-voids-ft1 (May 31) |
| 21 | **The gate: M is recon-real** — tied linear decode at cosine ≈ 0.9997 on byte-trigram with no accumulator | aleph gate experiments (June) |
| 22 | **The aleph address**: exact antipodal-softmax closed form (sinh/cosh ratio) makes a learned 2K-oriented-axis bottleneck trainable at 16.7M rows/step; recon survives full discretization (hard cos 0.992–0.997); codebook stays near-uniformly alive (ppl 125+/128) without regularization | aleph_model.py + geolip-aleph-void final_reports |
| 23 | Recon-real codebooks address more sharply (margin 0.967 vs 0.929) and are ~7× void-richer (β₂/axis 0.56 vs 0.08) than faux-embedding ancestors | aleph gate vs battery comparison |
| 24 | Infrastructure tier: WideCompiler N-first fusion (24 primitives + 5 Flux blocks; primitive speedups up to 174× measured on A100) ; geofractal geometric routing (near-linear 4→32 towers); geovocab2 formula library (cantor/cayley-menger/nikola/euler/hooke/newton/einstein/hawking) with pentachoron lexical synthesis and transfinite (ℵ) arithmetic | WideCompiler v0.7.0, geofractal v1.2.0, lattice_vocabulary v0.1.2 |
| 25 | **The aleph address is a valid attention feature map** (2026-06-09): score = ⟨p(q̂), p(k̂)⟩ over 2K oriented axes, antipodally factored to K-wide GEMM; strictly positive denominator; softmax parity on recall and byte-trigram LM (bpb 2.835 vs 2.836); linear wall-clock to 65,536 tokens where softmax OOMs at 16,384 | session artifacts + logged A100/Blackwell runs 2026-06-09 |
| 26 | **One vocabulary, many speakers + the restoring force** (2026-06-09): one codebook shared across all layers at zero quality cost; under 4× concentrated pressure the random-init codebook is pushed out (dev → −0.0070) and pulled back to −0.0004 — uniform locally ATTRACTS under attention gradients; antipodal parameter redundancy pruned 53% → 38% with rising margin | shared-codebook only-delta run, statute trajectory log |
| 27 | **The two-hemisphere law** (2026-06-09): the objective sets the sign of in-distribution writing against the uniform frame — separation (recon) → repulsive → polytope (+0.083) + void-rich (0.56); association (routing) → attractive → degenerate-ward (−0.062) + void-sparse (≤0.02). Refines #20 to substrate × objective-sign | [XB2] type-codebook extraction vs aleph-gate comparison |
| 28 | **Streaming codebook memory** (2026-06-09): the hub causal form is a recurrence with constant-size state (M±, z±), bit-exact vs the full pass; erank occupancy is a compressibility meter — text 16% vs noise 45% of memory bandwidth at 6,144 bytes of context | forward_stream exactness test + [ER] probe |
| 29 | **Address confidence is a kernel invariant of (τ, K, D)** (2026-06-09): ‖(p₊−p₋)A‖ identical for trained / untrained / random rows at every τ — the 0.857 monitor closed as frame physics, zero learned information | [TAU] sweep, trained and control models |
| 30 | **The CM band extends to attention address projections** (2026-06-09): q_addr/k_addr column-norm CV 0.136 / 0.158, inside 0.13–0.30, on an architecture class outside the original 65,536-config sweep (definition indicative; row CV is an orthogonal-init artifact) | [CV] probe |
| 31 | **The trained depth funnel** (2026-06-09): mid-tower clumps all content, deep layers differentiate — text organizes monotonically (−0.019 → −0.081, L0→L3), noise is released back toward uniform; untrained control flat zero everywhere, so the organization is 100% learned | depth-profile probe + untrained control |

---

## 12. Limitations, honestly

- The hosted checkpoints are all `address='hard'` at K=64, τ=0.1, on two substrates at 64×64; the soft-address-vs-gate reconstruction comparison is the live experiment, tracked per-version in `final_report.json`.
- Pure-cosine training is scale-blind (use `cosine_mse` when amplitude matters). Codebook collapse is the structural failure mode; it is monitored (perplexity/margin every eval) and treatable (`div_weight`), and has not occurred in published runs — but K, τ, and substrate combinations beyond those published are unmeasured.
- Reconstruction targets the training substrates. This is a research artifact for geometric representation learning, not a general-purpose generative model.
- The β₂ and margin comparisons quote the gate-regime codebook analysis; topological statistics at other (K, D, substrate) corners remain to be mapped with the same 15-signal battery.
- **[2026-06-09]** Routing-side limitations, honestly: the bucket (hard-address) variant
  is gradient-starved from cold start and unrescued (curriculum untested); the codebook-vs-
  codebook Procrustes is suggestive only (z = −1.43); the type-codebook extraction carries
  two unresolved caveats (head-averaging, frequent-trigram byte-sharing) and its untrained
  control has not been run; the two "margins" of §3.10 remain non-interchangeable and the
  routing monitors detect collapse only — structure evidence lives in deviation/statute/β₂;
  the CM-band reading on address projections uses an indicative CV definition, not the
  original sweep pipeline.

---

## 13. Source index

**Model & code:** [geolip-aleph-void](https://huggingface.co/AbstractPhil/geolip-aleph-void) · [geolip-svae (GitHub)](https://github.com/AbstractEyes/geolip-svae) · [geolip-core](https://github.com/AbstractEyes/geolip-core) · [geofractal](https://github.com/AbstractEyes/geofractal) · [lattice_vocabulary](https://github.com/AbstractEyes/lattice_vocabulary) · [WideCompiler](https://github.com/AbstractEyes/pytorch-parallel-compiler)

**Battery & experiment repos (HF):** [geolip-SVAE](https://huggingface.co/AbstractPhil/geolip-SVAE) · [geolip-svae-h2-64](https://huggingface.co/AbstractPhil/geolip-svae-h2-64) · [geolip-svae-ablations](https://huggingface.co/AbstractPhil/geolip-svae-ablations) · [geolip-svae-implicit-solver-experiments](https://huggingface.co/AbstractPhil/geolip-svae-implicit-solver-experiments) · [geolip-svae-batteries](https://huggingface.co/AbstractPhil/geolip-svae-batteries) · [geolip-svae-text](https://huggingface.co/AbstractPhil/geolip-svae-text) · [geolip-svae-transformer](https://huggingface.co/AbstractPhil/geolip-svae-transformer) · [geolip-hypersphere-experiments](https://huggingface.co/AbstractPhil/geolip-hypersphere-experiments) · [geolip-deep-embedding-analysis](https://huggingface.co/AbstractPhil/geolip-deep-embedding-analysis) · [svae-fresnel-128](https://huggingface.co/AbstractPhil/svae-fresnel-128) · [svae-freckles-256](https://huggingface.co/AbstractPhil/svae-freckles-256) · [geolip-sdxl-aleph](https://huggingface.co/AbstractPhil/geolip-sdxl-aleph) — within an account of 191 model repos and 70 dataset repos serving this program.

**The article chain (HF blog):** Geometric Fusion (Mar 6) · Geometric Memory I–III (Mar 9/11/14) · Procrustes ViT (Mar 15) · Constellation & 0.29154 (Mar 18) · Geometric Encoder's Toolkit (Mar 18) · Fused Batched Thin SVD I/II (Mar 25 / May 1) · FL Hybrid Eigendecomposition (Apr 1) · Structural Attractors (Apr 6) · Omega Tokens (Apr 8) · Three Geometric Bands (Apr 22) · The Polygonal Omega (Apr 25) · H2 Omega Confirmed (Apr 29) · The geolip-svae-transformer (May 29) · Reading the Voids (May 31).

---

*Methodology note: this article was assembled by reading the shipping code (`aleph_model.py`, `train_aleph.py`, `model.py`, `model_transformer.py`, `inference/`), the hosted `final_report.json` files, the model cards, the OMEGA_CATALOG, and the program's session records — no formula or number herein is reconstructed from memory. Where two metrics share a name (the two "margins" of §3.10), the distinction is stated rather than smoothed over.*