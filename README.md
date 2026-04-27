# geolip-svae

Spectral Variational Autoencoder — Omega Tokens on S^15

Patch-based SVD autoencoder with spectral cross-attention and sphere-normalized encoding. Encodes images (and noise, and text) as omega tokens — singular value vectors on unit hyperspheres.

## Quick Start

```bash
pip install git+https://github.com/AbstractEyes/geolip-svae.git
```

```python
from geolip_svae import load_model
from geolip_svae.inference import (
    InferenceEngine, make_calibration, Codebook,
)

# Load a pretrained model
model, cfg = load_model(hf_version='v40_freckles_noise')

# Run inference through the engine — handles arbitrary resolution
engine = InferenceEngine(model)
recon = engine.reconstruct(images_64x64)['recon']
recon_large = engine.reconstruct(images_512x512, mode='auto')['recon']

# Extract the projective-axis codebook
calib = make_calibration('sixteen_noise', n=64, size=64)
codebook = engine.extract_codebook(
    calib, attach=True,
    model_id='v40_freckles_noise', calibration_name='sixteen_noise',
)
print(codebook)
# Codebook(D=4, n_axes=35, pairs=13, unpaired=22, dev=-0.0426, clean=True)

# Project test inputs onto the codebook axes
out = engine.encode_axes(test_images)
activations = out['activations']  # [B, n_patches, V, n_axes]

# Persist the codebook for reuse (safetensors + JSON sidecar)
codebook.save('codebooks/freckles_v40__sixteen_noise')

# ...and load it later, possibly into a different engine
cb_loaded = Codebook.load('codebooks/freckles_v40__sixteen_noise')
```

## Architecture

```
Image → patches → MLP encode → sphere normalize → SVD (fp64) →
spectral cross-attention → decode → stitch → boundary smooth
```

Two proven regimes:

| Regime | V | D | ps | hidden | params | compression | patches (128²) |
|--------|---|---|----|--------|--------|-------------|----------------|
| **Fresnel/Johanna** | 256 | 16 | 16 | 768 | 16.9M | 48:1 | 64 |
| **Freckles** | 48 | 4 | 4 | 384 | 2.5M | 12:1 | 1024 |

Both use FLEigh (geolip-core) for fast eigendecomposition on CUDA.

## Trained Models

All checkpoints on [HuggingFace: AbstractPhil/geolip-SVAE](https://huggingface.co/AbstractPhil/geolip-SVAE)

### D=16 Family (16×16 patches, 17M params)

| Version | Name | Resolution | Dataset | MSE | Epochs |
|---------|------|-----------|---------|-----|--------|
| v12 | Fresnel-small | 128×128 | ImageNet-128 | 0.0000734 | 50 |
| v13 | Fresnel-base | 256×256 | ImageNet-256 | 0.0000610 | 20 |
| v19 | Fresnel-tiny | 64×64 | TinyImageNet | 0.0005 | 300 |
| v16 | Johanna-small | 128×128 | 16 noise types | 0.008 | 380 |
| v18 | Johanna-tiny | 64×64 | 16 noise types | — | 300 |
| v20 | Johanna-base | 256×256 | 16 noise types | 0.011 | 60 |
| v22 | Alexandria-small | 128×128 | Wikipedia text | 0.0016 | 100 |
| v30 | Grandmaster | 128×128 | ImageNet (denoiser) | 0.042 | 50 |

### D=4 Family (4×4 patches, 2.5M params)

| Version | Name | Resolution | Dataset | MSE | Epochs |
|---------|------|-----------|---------|-----|--------|
| v40 | Freckles | 64×64 | 16 noise types | 0.000005 | 100 |

## Geometric Constants

The SVAE discovers universal geometric structure independent of training data:

### D=16 (Fresnel/Johanna)
- **erank**: 15.88 ± 0.04 / 16.0 (99.25%)
- **CV band**: 0.20–0.23
- **S_delta**: modality-dependent (images: 0.238, noise: 0.407, text: 0.350)
- **Compression**: 48:1

### D=4 (Freckles)
- **erank**: 3.82 / 4.0 (95.5%)
- **S0/SD ratio**: 2.32 (locked from ep40)
- **S_delta**: 0.055
- **Resolution invariant**: identical MSE from 32×32 to 4096×4096

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
erank: 3.80–3.83 for all alien distributions
```

**Freckles tile-encode** (tiled vs native encoding):
```
All 16 types: 1.00× match, omega distance = 0.000000
4×4 patches are truly atomic — resolution-independent spectral descriptors
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

# Verify the projective property on any sphere-solver
calib = make_calibration('sixteen_noise', n=64, size=64)
cb = extract_codebook(model, calib, model_id='v40_freckles_noise',
                       calibration_name='sixteen_noise')

print(cb.metadata.deviation)         # signed distance from uniform RP^(D-1)
print(cb.is_projective_clean())      # |deviation| < 0.05
```

`Codebook` is a first-class artifact: extract once, save as a
safetensors + JSON sidecar pair, reuse across inference runs.
`InferenceEngine.encode_axes()` projects M onto the codebook axes;
`InferenceEngine.quantize_axes()` returns nearest-axis indices.

### Verification (Phase U / U5 — 6 cells, all projective-clean)

| Model | D | V | n_axes | pairs | deviation | clean |
|---|---|---|---|---|---|---|
| h2-64 battery_0 (gaussian) | 4 | 32 | 27 | 5 | +0.012 | ✓ |
| h2-64 battery_0 (sixteen_noise) | 4 | 32 | 27 | 5 | +0.012 | ✓ |
| Freckles v40 (gaussian) | 4 | 48 | 35 | 13 | −0.043 | ✓ |
| Freckles v40 (sixteen_noise) | 4 | 48 | 34 | 14 | −0.040 | ✓ |
| Johanna v18 (gaussian) | 16 | 256 | 231 | 25 | +0.040 | ✓ |
| Johanna v18 (sixteen_noise) | 16 | 256 | 229 | 27 | +0.040 | ✓ |

Calibration mismatch (gaussian vs sixteen_noise) shifts the codebook
metadata by less than 0.003 deviation in every case. The codebook is
the model's, not the input's. Direct extraction works at all tested D
values — no distillation training required.

Reproduce:
```bash
python -m geolip_svae.tests.u5_codebook_capacity --n-calib 64
```

For the full discovery story see scratchpad entries 000099–000107
and the ft1/ft2 articles at the end of this README.

```
geolip_svae/
├── model.py              PatchSVAE, SpectralCrossAttention, BoundarySmooth, gram_eigh_svd
├── inference/            Production inference framework (v0.7.0+)
│   ├── loading.py        load_model, VERSIONS, list_versions
│   ├── scaling.py        encode_at_scale / reconstruct_at_scale (direct/tile/auto)
│   ├── calibration.py    Calibration data generators (registry pattern)
│   ├── codebook.py       Codebook artifact, extract_codebook, antipodal-collapse helpers
│   ├── engine.py         InferenceEngine — orchestrator with codebook lifecycle
│   └── legacy.py         Back-compat shims (encode/decode/reconstruct/compute_axis_codebook)
├── arrays/               BatteryArrayConfig, BatteryArrayModel, build_array, specs/
├── experimental/         Preserved earlier variants — not part of the canonical path
│   ├── spectral_cell.py
│   ├── spectral_battery.py
│   └── experimental_codebook.py   (formerly spectral_codebook.py)
├── tests/                Diagnostic + Phase U lens-scope tests
│   ├── framework.py      LensScopeTestCase base + 3 measurement axes
│   ├── u0_smoke_test.py  Framework integrity gate (15 tests, ~5s)
│   ├── u5_codebook_capacity.py   Cross-band codebook capacity test
│   ├── diagnostic.py     12-test universal diagnostic battery (D=16 models)
│   ├── noise_diagnostic.py       Freckles piecemeal resolution test (6 tests)
│   └── noise_stress_test.py      Freckles extreme-resolution + OOD stress test
├── train.py              Unified trainer (7 presets)
└── __init__.py           Package exports + back-compat surface
```

The `inference/` package is the recommended public surface. Pre-v0.7.0 code that
imports `encode`, `decode`, `reconstruct`, `batched_forward`, or
`compute_axis_codebook` directly from `geolip_svae.inference` continues to work
via shims in `inference/legacy.py`.

## Dependencies

- [geolip-core](https://github.com/AbstractEyes/geolip-core) — FLEigh fast eigendecomposition (hard dependency)
- torch >= 2.1.0
- transformers >= 4.40.0 (battery arrays, AutoModel interface)
- huggingface-hub >= 0.20.0
- safetensors >= 0.4.0 (codebook persistence)

## Training

```bash
# Fresnel (images)
python -m geolip_svae.train --preset fresnel_small

# Johanna (noise)
python -m geolip_svae.train --preset johanna_small

# List all presets
python -m geolip_svae.train --list-presets
```

## Diagnostics

```bash
# Universal diagnostic (D=16 models)
python -m geolip_svae.tests.diagnostic --hf v12_imagenet128

# Freckles piecemeal test
python -m geolip_svae.tests.noise_diagnostic --model v40_freckles_noise

# Freckles extreme-resolution + OOD stress test
python -m geolip_svae.tests.noise_stress_test --model v40_freckles_noise
```

## Spectral Codebook (experimental)

A pre-rebuild noise-native tokenizer mapping text characters to spectral
noise signatures. Lives in the `experimental/` subpackage and is
distinct from the projective-axis `Codebook` artifact described above —
the spectral codebook does NOT perform antipodal-pair collapse and
reports different geometric statistics. Preserved for the Alexandria
text-as-noise pathway.

```python
from geolip_svae import SpectralTokenizer, build_codebook
# (lazily re-exported from geolip_svae.experimental.experimental_codebook)

codebook = build_codebook(save_path='codebook.json')
tokenizer = SpectralTokenizer(codebook)
image, ids, strings = tokenizer.text_to_image("Hello, world!")
```

For projective-axis codebooks (the canonical path for sphere-solver
inference), see [Projective-Axis Codebooks](#projective-axis-codebooks)
above.

## Battery Arrays

Bundle multiple independently-trained PatchSVAE batteries as a single `AutoModel` that emits a per-bank MSE signature across all inputs. Useful for ensembling, OOD detection, and signature-based classifiers.

```python
from transformers import AutoModel

# Load a prebuilt array (geolip-svae must be installed, no trust_remote_code needed)
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

Array specs live under `geolip_svae.arrays.specs/` — each spec describes its battery class, architecture kwargs, training-config layout, and checkpoint path scheme. To add a new array, drop in a new spec module.

The packaged `PatchSVAE` supports ablation toggles used by array training:
`activation`, `row_norm`, `svd_mode`, `linear_readout`, `match_params`, `init_scheme`. Defaults preserve standard behavior.

## FOR CLAUDE

CLAUDE.md

## License

MIT

## Links

- Models: [huggingface.co/AbstractPhil/geolip-SVAE](https://huggingface.co/AbstractPhil/geolip-SVAE)
- Core: [github.com/AbstractEyes/geolip-core](https://github.com/AbstractEyes/geolip-core)
- Article: [Omega Tokens: Finding The Self Solving Frame](https://huggingface.co/blog/AbstractPhil/multimodal-geometric-fusion)