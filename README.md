# geolip-svae

Spectral Variational Autoencoder — Omega Tokens on S^(D-1)

Patch-based SVD autoencoder with spectral cross-attention and sphere-normalized
encoding. Encodes images, noise, text, structured-substrate (binary tree,
sentencepiece bits, byte trigrams) as **omega tokens** — singular value
vectors on unit hyperspheres. Pluggable SVD backend with fused Triton
kernels at D ∈ {2..6}.

## Quick Start

```bash
pip install git+https://github.com/AbstractEyes/geolip-svae.git
```

```python
from geolip_svae import load_model
from geolip_svae.inference import (
    InferenceEngine, make_calibration, Codebook,
)

model, cfg = load_model(hf_version='v40_freckles_noise')
engine = InferenceEngine(model)

# Resolution-agnostic — same model, any size
recon_64  = engine.reconstruct(images_64x64)['recon']
recon_512 = engine.reconstruct(images_512x512, mode='auto')['recon']

# Projective-axis codebook on RP^(D-1)
calib = make_calibration('sixteen_noise', n=64, size=64)
codebook = engine.extract_codebook(calib, attach=True,
    model_id='v40_freckles_noise', calibration_name='sixteen_noise')
print(codebook)
# Codebook(D=4, n_axes=35, pairs=13, unpaired=22, dev=-0.0426, clean=True)

out = engine.encode_axes(test_images)
codebook.save('codebooks/freckles_v40__sixteen_noise')
```

## Architecture

```
Image → patches → MLP encode → sphere-normalize rows of M → SVD (fp64) →
spectral cross-attention (SDPA) → decode → stitch → boundary smooth
```

Six load-bearing properties (see `CLAUDE.md` for the longer story):

- Sphere-normalize rows of internal matrix M (`F.normalize(M, dim=-1)`) — geometric premise.
- SVD via Gram-eigh dispatcher in fp64 (autocast-shielded) — accuracy invariant.
- `nn.init.orthogonal_(self.enc_out.weight)` — applied unconditionally; load-bearing.
- No BatchNorm, no Dropout, no GAP — destabilize the spectral structure.
- Pure Adam, never AdamW — weight decay fights the geometric basin.
- All activations are parameterless — swapping them never changes `state_dict`.

### Two production regimes

| Regime | V | D | ps | hidden | params | compression | patches at 128² |
|--------|---|---|----|--------|--------|-------------|-----------------|
| **D=16** (Fresnel/Johanna/Alexandria) | 256 | 16 | 16 | 768 | 16.9M | 48:1 | 64 |
| **D=4** (Freckles) | 48 | 4 | 4 | 384 | 2.5M | 12:1 | 1024 |

Plus the **h2-class sphere-solver** variant (V=32, D=4) used by the h2-64
battery array — replaces SVD with a learned linear readout. Required to
load h2-64 weights; new D=4 work uses real SVD via the Triton kernel.

## Model Variants

`PatchSVAE` is one class; the variant is determined by the cfg kwargs.

| Variant | V | D | ps | hidden | depth | n_cross | linear_readout | svd_mode | smooth_mid |
|---------|---|---|----|--------|-------|---------|----------------|----------|------------|
| Fresnel  (image) | 256 | 16 | 16 | 768 | 4 | 2 | False | default | None (16) |
| Johanna  (noise) | 256 | 16 | 16 | 768 | 4 | 2 | False | default | None (16) |
| Alexandria (text) | 256 | 16 | 16 | 768 | 4 | 2 | False | default | None (16) |
| Freckles (D=4)   | 48  | 4  | 4  | 384 | 4 | 2 | False | default | 16 |
| h2-class single  | 32  | 4  | 4  | 384 | 4 | 2 | **True** | **none** | 16 |
| Bintree proto    | 32  | 4  | 4  | 384 | 4 | 2 | True  | none | 16 |
| SP-bits proto    | 32  | 4  | 4  | 384 | 4 | 2 | True  | none | 16 |
| ByteTri proto    | 32  | 4  | 4  | 384 | 4 | 2 | True  | none | 16 |

Identical class, identical `forward()`, three real ablation toggles:
`linear_readout` swaps SVD for a learned readout (sphere-solver), `svd_mode`
picks the SVD code path, `match_params` chooses readout sizing. State_dict
shapes change only when `linear_readout=True` (adds the `readout` module).

### Sphere-solver path

The triple `linear_readout=True, svd_mode='none', match_params=True` is the
sphere-solver variant. It replaces SVD with `nn.Linear(V*D, V*D)` and
treats the column norms as singular values, identity as Vt. Originated as
a workaround for D=4 being slow under the old FL-eigh path; with the new
fused N=4 Triton kernel real SVD is now competitive, but the variant is
preserved because the h2-64 battery array's checkpoints depend on it.

## Trainer Presets

23 named cfg dicts in `geolip_svae.train_presets.PRESETS`. Run any of
them with:

```bash
python -m geolip_svae.train --preset NAME
python -m geolip_svae.train --list-presets
python -m geolip_svae.train --preset NAME --epochs 5 --no-upload
```

### Image presets (D=16, V=256, ps=16)

| Preset | Dataset | Resolution | Batch | LR | Epochs |
|--------|---------|------------|-------|-----|--------|
| `fresnel_tiny`  | `tiny_imagenet` | 64²  | 256 | 1e-4 | 300 |
| `fresnel_small` | `imagenet_128`  | 128² | 128 | 1e-4 | 50 |
| `fresnel_base`  | `imagenet_256`  | 256² | 64  | 1e-4 | 20 |

### Noise presets (D=16, V=256, ps=16)

| Preset | Dataset | Resolution | Curriculum | Pretrained |
|--------|---------|------------|------------|------------|
| `johanna_tiny`  | `curriculum_noise` | 64²  | patience-based | scratch |
| `johanna_small` | `omega_noise`      | 128² | none           | from `v14_noise` |
| `johanna_base`  | `scheduled_noise`  | 256² | tier_schedule  | scratch |

### Text preset (D=16, V=256, ps=16)

| Preset | Dataset | Resolution | Pretrained |
|--------|---------|------------|------------|
| `alexandria_small` | `wikipedia` | 128² | from `v16_johanna_omega` |

### D=4 presets (V=48, ps=4)

| Preset | Dataset | Resolution | Notes |
|--------|---------|------------|-------|
| `freckles_64`  | `omega_noise`   | 64²  | 100 ep, 2.55M params (D=4 noise specialist) |
| `freckles_256` | `omega_noise`   | 256² | 1 ep, init from freckles_64 (resolution-transfer chain) |
| `freckles_512` | `omega_noise`   | 512² | 1 ep, init from freckles_256 |
| `fresnel_64`   | `tiny_imagenet` | 64²  | TinyImageNet w/ Freckles geometry (D=4, 2.55M) |

### h2-class sphere-solver presets (V=32, D=4, ps=4)

| Preset | Channels | Notes |
|--------|---------|-------|
| `h2_64_single`           | 3 | Reproduce one h2-64 battery from scratch (gaussian only) |
| `h2_64_1channel`         | 1 | Single-channel sphere-solver |
| `h2_64_5channel`         | 5 | Multi-channel sphere-solver |
| `h2_64_5channel_v40_d4_ps4_h80` | 5 | h=80 variant for capacity sweep |
| `h2_h64_v64_d16_ps16_single_full_noise_image64x64` | 3 | D=16 hybrid |
| `h2_64_dodecahedron_v1` / `_v2` | 3 | Polytope-class architecture studies |
| `h2_64_tesseract_v1`     | 3 | 4D polytope architecture |

### Substrate prototypes (h2-64 architecture)

| Preset | Dataset | Notes |
|--------|---------|-------|
| `bintree_proto`          | `binary_tree`        | Binary-tree bit substrate |
| `sentencepiece_proto`    | `sentencepiece_bits` | T5-base tokens as bit-images, one token per patch |
| `byte_trigram_proto`     | `byte_trigram`       | 3-byte-per-pixel text packing (1-gram) |
| `byte_trigram_proto_64`  | `byte_trigram`       | 64² variant |

### Streaming continuation

For long-running fine-tuning on streaming random crops (the "sublens
perspective" mode that produced v50_fresnel_64's 140M+ images):

```bash
python -m geolip_svae.train_streaming --hf-version v50_fresnel_64
```

## Dataset Registry

10 dataset factories in `geolip_svae.dataset_presets.DATASET_FACTORIES`.
Pick a name as `cfg['dataset']`; the trainer dispatches via
`get_dataset_bundle(cfg, channels)`.

| Name | Source | Reconstruction target | Cfg keys consumed |
|------|--------|------------------------|---------------------|
| `tiny_imagenet`     | HF `zh-plus/tiny-imagenet` | RGB images | `img_size`, `batch_size`, `ds_size` |
| `imagenet_128`      | HF imagenet (128²)         | RGB images | same |
| `imagenet_256`      | HF imagenet (256²)         | RGB images | same |
| `curriculum_noise`  | 16-type noise generators   | noise patches | `+ allowed_types`, `curriculum='patience'` |
| `omega_noise`       | 16-type noise generators   | noise patches | `+ allowed_types`, `ds_size`, `val_size` |
| `scheduled_noise`   | 16-type noise generators   | noise patches | `+ tier_schedule={epoch: tier}` |
| `wikipedia`         | HF wikipedia text          | char-grayscale images | `+ ds_size`, `val_size` |
| `binary_tree`       | Synthetic BFS binary trees | per-bit recovery | `+ tree_depth` |
| `sentencepiece_bits`| HF tokenizer + corpus      | per-bit and per-token recovery | `+ sp_tokenizer`, `sp_corpus`, `sp_n_bits` |
| `byte_trigram`      | HF text corpus, 3-byte/pixel | per-byte recovery | `+ bt_corpus`, `bt_max_corpus_bytes` |

All factories are channel-aware via the `channels=` kwarg threaded through
`get_dataset_bundle(cfg, channels=cfg['channels'])`. Noise / byte-trigram
emit C-channel tensors; image / text / tree / sentencepiece are
channel-agnostic and ignore the kwarg.

## Configuration Reference

Every cfg key the trainer recognizes is documented in
`geolip_svae.train_presets.TEMPLATE` — a fully-specified runnable cfg
dict where every entry has its default and accepted values inline.
Copy it when authoring a new preset:

```python
from geolip_svae.train_presets import TEMPLATE

my_cfg = dict(TEMPLATE)
my_cfg['hf_version'] = 'my_run'
my_cfg['dataset']    = 'omega_noise'
# delete keys you accept defaults for
```

The 51 keys are grouped:

| Group | Keys |
|-------|------|
| Required architecture | `V, D, patch_size, hidden, depth, n_cross` |
| Optional architecture | `n_heads, smooth_mid, channels` |
| Ablation toggles | `solver, activation, activations, row_norm, svd_mode, svd_method, svd_compute_dtype, linear_readout, match_params, init_scheme` |
| Required training | `dataset, img_size, batch_size, lr, epochs, target_cv, hf_version` |
| Optional training | `save_every, report_every` |
| Loss / band | `cv_weight, boost, sigma, cv_band_lo, cv_band_hi` |
| Schedule | `pretrained, curriculum, tier_schedule, allowed_types` |
| Dataset-specific | `ds_size, val_size, tree_depth, sp_tokenizer, sp_corpus, sp_n_bits, bt_corpus, bt_max_corpus_bytes` |
| Output / IO | `save_dir, hf_repo, tb_dir, upload` |
| Codebook hook | `build_codebook, build_topology` |

## SVD Dispatcher

`gram_eigh_svd` delegates to `geolip_core.linalg.batched_svd` with a 5-way
auto-dispatch that picks the best path per N (= D) / dtype / device:

```
N ∈ {2,3,4,5,6}, CUDA + Triton:    Fused Triton kernel  (per-N specialized)
N ≤ 12, CUDA, fp32:                Gram + FL eigh
N ≤ 12, CUDA, fp64:                Gram + torch.linalg.eigh
                                    (avoids FL fp64 — FLEigh returns fp32 V
                                     which silently caps fp64 orthogonality
                                     at ~1e-3)
N > 12 or CPU:                     Gram + torch.linalg.eigh
Wide shape (M < N):                Transparent transpose
```

D=4 in particular gets the fused `_svd4_kernel` (one program per batch,
6 Jacobi sweeps over the 4×4 Gram, fp32 / fp64 honored as kernel
constexpr). At a typical batch×patches workload (B=256, N=64 patches,
V=48, D=4) that's 16384 program instances per cross-attn forward.

Configurable from any preset:

```python
cfg['svd_method']        = 'auto'   # 'auto' | 'fl' | 'gram_eigh' | 'triton' | 'torch'
cfg['svd_compute_dtype'] = 'fp64'   # 'fp64' (recommended) | 'fp32'
```

The trainer's startup readout shows which path is actually engaged for
the current cfg + device:

```
  Device:       cuda:0 — NVIDIA RTX PRO 6000 Blackwell (sm_120, 188 SMs, 95.0 GiB)
  Torch:        2.10.0+cu128  cuda=12.8  cudnn=91002
  geolip-core:  triton=v3.6.0  use_triton=True  use_fl_eigh=True
  SVD path:     fused Triton N=4 kernel @ fp64 — BLOCK_M=128, JACOBI_ITERS=6
  SVD config:   solver='default', svd_mode='default', svd_method='auto', compute_dtype='fp64'
```

## Activation Registry

21 parameterless activations × 5 named sites in `PatchSVAE`. Default is
`gelu` everywhere; the legacy `activation='gelu'` shortcut sets `enc_in`
only for back-compat.

```python
from geolip_svae.model import ACTIVATIONS, ACTIVATION_SITES, ACTIVATION_MODULES
sorted(ACTIVATIONS)
# ['celu', 'elu', 'gelu', 'gelu_tanh', 'hardsigmoid', 'hardswish', 'hardtanh',
#  'identity', 'leaky_relu', 'logsigmoid', 'mish', 'relu', 'relu6', 'selu',
#  'sigmoid', 'silu', 'softplus', 'softsign', 'swish', 'tanh', 'tanhshrink']

ACTIVATION_SITES
# ('enc_in', 'enc_block_inner', 'dec_in', 'dec_block_inner', 'boundary_smooth')
```

Per-site override via cfg:

```python
cfg['activations'] = {
    'enc_in':           'silu',
    'enc_block_inner':  'silu',
    'dec_in':           'gelu',
    'dec_block_inner':  'gelu',
    'boundary_smooth':  'relu6',
}
```

PReLU/RReLU intentionally excluded - trainable params would silently
inflate model size. All entries are parameterless modules so swapping
never changes `state_dict` shape and existing checkpoints reload
identically.

## Channels

`PatchSVAE` is channel-aware. Setting `cfg['channels']` plumbs through the
encoder input dim (`patch_dim = C × ps × ps`), the decoder output dim, and
the post-stitch `BoundarySmooth`. The geometric core (sphere-norm M, SVD,
cross-attention, codebook) is channel-agnostic — channels only affects
the I/O layers.

Tested channel counts: 1, 3, 5. The `h2_64_1channel` and `h2_64_5channel*`
presets are reference points.

## Trained Models

All checkpoints on [HuggingFace: AbstractPhil/geolip-SVAE](https://huggingface.co/AbstractPhil/geolip-SVAE)

### D=16 family

| Version | Name | Resolution | Dataset | MSE | Epochs |
|---------|------|-----------|---------|-----|--------|
| v12 | Fresnel-small    | 128Â² | ImageNet-128 | 0.0000734 | 50  |
| v13 | Fresnel-base     | 256Â² | ImageNet-256 | 0.0000610 | 20  |
| v19 | Fresnel-tiny     | 64Â²  | TinyImageNet | 0.0005    | 300 |
| v16 | Johanna-small    | 128Â² | 16 noise types | 0.008  | 380 |
| v18 | Johanna-tiny     | 64Â²  | 16 noise types | —      | 300 |
| v20 | Johanna-base     | 256Â² | 16 noise types | 0.011  | 60  |
| v22 | Alexandria-small | 128Â² | Wikipedia text | 0.0016 | 100 |
| v30 | Grandmaster      | 128Â² | ImageNet (denoiser) | 0.042 | 50 |

### D=4 family

| Version | Name | Resolution | Dataset | MSE | Epochs |
|---------|------|-----------|---------|-----|--------|
| v40 | Freckles | 64Â² | 16 noise types | 0.000005 | 100 |

## Geometric Constants

The SVAE discovers universal geometric structure independent of training data:

### D=16 (Fresnel/Johanna/Alexandria)

- **erank**: 15.88 ± 0.04 / 16.0 (99.25%)
- **CV band**: 0.20–0.23
- **S_delta** (modality-dependent): images 0.238, noise 0.407, text 0.350
- **Compression**: 48:1

### D=4 (Freckles)

- **erank**: 3.82 / 4.0 (95.5%)
- **S0/SD ratio**: 2.32 (locked from ep40)
- **S_delta**: 0.055
- **Resolution invariant**: identical MSE from 32Â² to 4096Â²

## Key Results

**Freckles resolution invariance** (trained at 64×64, tested zero-shot):

```
 36×36    MSE=0.000002    |    512×512     27s, 31MB
 128×128  MSE=0.000002    |    2048×2048   6.7s, 31MB
 256×256  MSE=0.000002    |    4096×4096   27s, 31MB
```

**Freckles OOD noise** (16 untrained distributions):

```
All 16: ✓ handles (ratio ≤ 1.4× vs known types)
erank:  3.80–3.83 for all alien distributions
```

**Freckles tile-encode** (tiled vs native encoding):

```
All 16 types: 1.00× match, omega distance = 0.000000
4×4 patches are atomic — resolution-independent spectral descriptors
```

## Projective-Axis Codebooks

Every trained sphere-solver tested produces an M tensor whose rows, when
antipodal pairs are merged via mutual-strongest matching, form a
near-uniformly-distributed codebook on **ℝP^(D-1)**. The collapse method
is a deterministic tensor operation, not a learned property:

```python
from geolip_svae.inference import (
    InferenceEngine, extract_codebook, make_calibration,
)

calib = make_calibration('sixteen_noise', n=64, size=64)
cb = extract_codebook(model, calib, model_id='v40_freckles_noise',
                       calibration_name='sixteen_noise')

print(cb.metadata.deviation)         # signed distance from uniform RP^(D-1)
print(cb.is_projective_clean())      # |deviation| < 0.05
```

`Codebook` is a first-class artifact: extract once, save as a safetensors
+ JSON sidecar pair, reuse across inference runs.
`InferenceEngine.encode_axes()` projects M onto the codebook axes;
`InferenceEngine.quantize_axes()` returns nearest-axis indices.

### Auto-build from the trainer

The trainer hooks `create_codebook(model, cfg, ...)` at end-of-train. By
default this extracts the projective-axis Codebook artifact from the
final model and runs three topology probes (kNN-graph connectivity sweep,
local-PCA intrinsic dimension, optional ripser persistent homology).
Output lands under `save_dir/codebooks/` and uploads to HF when enabled.
Opt out with `cfg['build_codebook'] = False` (or just topology with
`cfg['build_topology'] = False`).

### Verification (Phase U / U5 — 6 cells, all projective-clean)

| Model | D | V | n_axes | pairs | deviation | clean |
|---|---|---|---|---|---|---|
| h2-64 battery_0 (gaussian)        | 4  | 32  | 27  | 5  | +0.012 | ✓ |
| h2-64 battery_0 (sixteen_noise)   | 4  | 32  | 27  | 5  | +0.012 | ✓ |
| Freckles v40 (gaussian)           | 4  | 48  | 35  | 13 | −0.043 | ✓ |
| Freckles v40 (sixteen_noise)      | 4  | 48  | 34  | 14 | −0.040 | ✓ |
| Johanna v18 (gaussian)            | 16 | 256 | 231 | 25 | +0.040 | ✓ |
| Johanna v18 (sixteen_noise)       | 16 | 256 | 229 | 27 | +0.040 | ✓ |

Calibration mismatch (gaussian vs sixteen_noise) shifts the codebook
metadata by less than 0.003 deviation in every case. The codebook is the
model's, not the input's. Direct extraction works at all tested D values
— no distillation training required.

Reproduce:

```bash
python -m geolip_svae.tests.u5_codebook_capacity --n-calib 64
```

## Battery Arrays

Bundle multiple independently-trained PatchSVAE batteries as a single
`AutoModel` that emits a per-bank MSE signature across all inputs.
Useful for ensembling, OOD detection, and signature-based classifiers.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("AbstractPhil/geolip-svae-h2-64")
signature = model(images)
# images:    [B, 3, H, W]
# signature: [B, n_batteries, n_epoch_phases] — MSE per bank
```

Build a new array from a completed training sweep:

```python
from geolip_svae.arrays import build_array
build_array(spec_name="h2_64", upload=True)
```

Array specs live under `geolip_svae.arrays.specs/` — each spec describes
its battery class, architecture kwargs, training-config layout, and
checkpoint path scheme. To add a new array, drop in a new spec module.

## Module Layout

```
geolip_svae/
├── model.py              PatchSVAE, SpectralCrossAttention (SDPA), BoundarySmooth,
│                         gram_eigh_svd dispatcher (geolip-core 0.3.0 batched_svd),
│                         ACTIVATIONS / ACTIVATION_MODULES / ACTIVATION_SITES,
│                         SVD_METHODS / SVD_COMPUTE_DTYPES
├── train.py              Unified trainer (CLI + train(cfg)) with codebook auto-build
├── train_presets.py      PRESETS registry (23 entries) + TEMPLATE (51 cfg keys)
├── dataset_presets.py    DATASET_FACTORIES (10 entries) + dataset classes,
│                         recovery metrics, eval_per_type, get_dataset_bundle
├── inference/            Production inference framework
│   ├── loading.py        load_model, VERSIONS, list_versions
│   ├── scaling.py        encode_at_scale / reconstruct_at_scale (direct/tile/auto)
│   ├── calibration.py    Calibration data generators (registry pattern)
│   ├── codebook.py       Codebook artifact, extract_codebook, antipodal-collapse
│   ├── engine.py         InferenceEngine — orchestrator with codebook lifecycle
│   ├── train_codebook.py create_codebook + topology probes (kNN/PCA/ripser)
│   ├── text.py           SentenceEncoder text-side wrapper for byte-trigram models
│   └── legacy.py         Back-compat shims (encode/decode/reconstruct/...)
├── arrays/               BatteryArrayConfig, BatteryArrayModel, build_array, specs/
├── experimental/         Preserved earlier variants — not the canonical path
└── tests/                Diagnostic + Phase U lens-scope tests

svae_proto/               SISTER PACKAGE — separate `pip install ./svae_proto`. Self-contained
                          experiment scaffolds (one-way dependency: svae_proto imports
                          from geolip_svae, never the reverse). Per-experiment opt-in
                          dependency groups via `pip install "./svae_proto[exp_NNN]"`.
                          See svae_proto/README.md for the contract.
```

The `inference/` package is the recommended public surface. Pre-v0.7.0
code that imports `encode`, `decode`, `reconstruct`, `batched_forward`,
or `compute_axis_codebook` directly continues to work via shims in
`inference/legacy.py`.

## Dependencies

- [geolip-core](https://github.com/AbstractEyes/geolip-core) ≥ 0.3.0 — batched_svd dispatcher with fused Triton kernels (hard dependency)
- torch ≥ 2.1.0 (≥ 2.5 recommended for SDPA backends)
- triton ≥ 3.0 (optional but strongly recommended — enables fused N∈{2..6} SVD kernels at D=4 and below)
- transformers ≥ 4.40.0 (battery arrays, AutoModel interface, sentencepiece tokenizer)
- huggingface-hub ≥ 0.20.0
- safetensors ≥ 0.4.0 (codebook persistence)
- ripser (optional — persistent-homology topology probe)

## Diagnostics

```bash
# Universal diagnostic (D=16 models)
python -m geolip_svae.tests.diagnostic --hf v12_imagenet128

# Freckles piecemeal test
python -m geolip_svae.tests.noise_diagnostic --model v40_freckles_noise

# Freckles extreme-resolution + OOD stress test
python -m geolip_svae.tests.noise_stress_test --model v40_freckles_noise

# Cross-band codebook capacity (Phase U5)
python -m geolip_svae.tests.u5_codebook_capacity --n-calib 64
```

## Spectral Codebook (experimental)

A pre-rebuild noise-native tokenizer mapping text characters to spectral
noise signatures. Lives in the `experimental/` subpackage and is distinct
from the projective-axis `Codebook` artifact — the spectral codebook does
NOT perform antipodal-pair collapse and reports different geometric
statistics. Preserved for the Alexandria text-as-noise pathway.

```python
from geolip_svae import SpectralTokenizer, build_codebook
codebook = build_codebook(save_path='codebook.json')
tokenizer = SpectralTokenizer(codebook)
image, ids, strings = tokenizer.text_to_image("Hello, world!")
```

For projective-axis codebooks (the canonical path for sphere-solver
inference), see [Projective-Axis Codebooks](#projective-axis-codebooks)
above.

## Prototypes — `svae-proto` sister package

Experimental scaffolding lives in **`svae_proto/`** as a separate
installable package, not as part of `geolip-svae`. The main package
stays slim — installing `geolip-svae` does NOT pull in `svae-proto` or
its experiment-specific deps. Production environments that just want to
run trained models pay zero cost for experiment infrastructure.

Neither package is on PyPI — both install from GitHub. svae-proto's
`pyproject.toml` lives in the `svae_proto/` subdirectory, so the URL
fragment `#subdirectory=svae_proto` is required. svae-proto declares
`geolip-svae` as a runtime dep (also via git URL), so installing
svae-proto pulls the core package along automatically — one URL is
enough for both.

### Terminal — set up an environment before running anything

```bash
# Main package only — production install
pip install "git+https://github.com/AbstractEyes/geolip-svae.git"

# Main + experimental scaffolding in one command (single URL — svae-proto
# pulls geolip-svae transitively):
pip install "svae-proto @ git+https://github.com/AbstractEyes/geolip-svae.git#subdirectory=svae_proto"

# With experiment 001's heavy deps (transformers / datasets / sentencepiece):
pip install "svae-proto[exp_001] @ git+https://github.com/AbstractEyes/geolip-svae.git#subdirectory=svae_proto"

# Everything across every experiment:
pip install "svae-proto[all] @ git+https://github.com/AbstractEyes/geolip-svae.git#subdirectory=svae_proto"

# Pin to a specific tag / branch / commit:
pip install "svae-proto @ git+https://github.com/AbstractEyes/geolip-svae.git@v0.9.0#subdirectory=svae_proto"
```

### Colab notebook — inline cell at top of notebook (or above an experiment call)

In Colab / Jupyter notebook cells, prefix shell commands with `!`. The
notebook environment persists installs across cells but not across
runtime restarts, so a self-contained install cell at the top of the
notebook (or directly above the experiment block) is the typical pattern.

The recommended Colab install cell — uninstall first to guarantee any
chained packages re-resolve from source, then a single pip command for
the dep tree:

```python
# ── Standalone install cell — paste at top of notebook ──────────────
!pip uninstall -y geolip-svae svae-proto geolip-core \
                  geofractal geometricvocab wide_compiler
!pip install --no-cache-dir \
    "svae-proto[exp_001] @ git+https://github.com/AbstractEyes/geolip-svae.git#subdirectory=svae_proto"
```

That single `pip install` does:
1. Clones `geolip-svae` repo once.
2. Reads `svae_proto/pyproject.toml` (because of `#subdirectory=svae_proto`).
3. Sees the dep `geolip-svae @ git+...` and installs the core from the same clone.
4. Sees the core's dep `geolip-core @ git+...` and installs that too.
5. Resolves `[exp_001]` extras (transformers, datasets, sentencepiece) plus the rest of the dep tree (torch, etc.) in a single resolution pass.

`--no-cache-dir` complements the explicit uninstall — it bypasses pip's
wheel cache so every install is a fresh build of the latest commit. The
uninstall list includes sibling packages (`geofractal`, `geometricvocab`,
`wide_compiler`) that may be in your stack — extend or trim it to match
what's actually installed.

Drop the `[exp_001]` to install just the proto core. Drop the entire
trailing URL term and use the `git+...geolip-svae.git` URL alone if you
want only the main package.

> **Colab notebook is not a terminal.** The `!` prefix invokes a fresh
> shell per cell; environment variables / `cd` / activated virtualenvs do
> not persist between cells. Within a single cell you can chain shell
> commands with `&&` or `\` line continuations as shown above. Use `%cd`
> (the magic, not the `!cd` shell builtin) if you actually need the
> working directory to persist across cells.

### Local clone — active development on the source

```bash
git clone https://github.com/AbstractEyes/geolip-svae.git
cd geolip-svae

pip install .                          # main package
pip install ./svae_proto               # add svae-proto
pip install -e ./svae_proto            # editable for live source edits
pip install "./svae_proto[exp_001]"    # with experiment 001 deps
```

Editable install (`-e`) reads source files directly from the clone, so
`git pull` is enough to pick up upstream changes — no reinstall needed.
In a notebook context, restart the runtime after `git pull` to clear
Python's module cache.

The contract: **prototypes depend on `geolip_svae`; the package never
imports prototypes**. Delete the entire `svae_proto/` directory at any
time without breaking anything in the core. When an experiment proves
out, its dataset / metric / etc. graduates into the appropriate
`geolip_svae` module and the prototype becomes a historical record.

Each experiment lives under `svae_proto/exp_NNN_<slug>/` and registers
its dataset into `DATASET_FACTORIES` at runtime (no source edits to the
core package).

Current experiments:

| ID  | Slug                       | What it tests | Extras key | Status |
|-----|----------------------------|----------------|------------|--------|
| 001 | `vocab_trigram_recall`     | multi-byte token recall (sentencepiece-aware) | `exp_001` | scaffolded |

```bash
python -m svae_proto.exp_001_vocab_trigram_recall.run --variant proto_64
python -m svae_proto.exp_001_vocab_trigram_recall.run --variant freckles_64
python -m svae_proto.exp_001_vocab_trigram_recall.run --variant fresnel_128
```

See `svae_proto/README.md` for the full contract, the directory naming
convention (note: `exp_` prefix is required since Python module names
cannot start with a digit), and how to author a new experiment with its
own opt-in dependency group.

## FOR CLAUDE

CLAUDE.md — required reading for any Claude (or person) entering the
repo fresh. Documents load-bearing invariants that look like free
choices but aren't, plus the debug move that catches every
architecture-identity mismatch in 5 lines.

## License

MIT

## Links

- Models: [huggingface.co/AbstractPhil/geolip-SVAE](https://huggingface.co/AbstractPhil/geolip-SVAE)
- Core: [github.com/AbstractEyes/geolip-core](https://github.com/AbstractEyes/geolip-core)
