# geolip-svae

Spectral Variational Autoencoder — Omega Tokens on S^15

Patch-based SVD autoencoder with spectral cross-attention and sphere-normalized encoding. Encodes images (and noise, and text) as omega tokens — singular value vectors on unit hyperspheres.

## Quick Start

```bash
pip install git+https://github.com/AbstractEyes/geolip-svae.git
```

```python
from geolip_svae import PatchSVAE, load_model, reconstruct

# Load a pretrained model
model, cfg = load_model(hf_version='v12_imagenet128')  # Fresnel
out = model(images)  # out['recon'], out['svd']['S'], out['svd']['U'], out['svd']['Vt']

# Or Freckles (2.5M params, D=4, 4×4 patches)
model, cfg = load_model(hf_version='v40_freckles_noise')
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

**Grandmaster single-shot denoising** (noisy→clean, one pass):
```
σ=0.2: near-Fresnel quality
σ=0.5: full structure preserved
σ=1.0: recovers from visual destruction
```

## Package Structure

```
geolip_svae/
├── model.py             PatchSVAE, SpectralCrossAttention, BoundarySmooth, gram_eigh_svd
├── inference.py         load_model, encode, decode, reconstruct, VERSIONS
├── train.py             Unified trainer (7 presets)
├── diagnostic.py        12-test universal diagnostic battery
├── spectral_codebook.py Noise-native tokenizer for Alexandria
├── noise_diagnostic.py  Freckles piecemeal resolution test (6 tests)
└── __init__.py          Package exports
```

## Dependencies

- [geolip-core](https://github.com/AbstractEyes/geolip-core) — FLEigh fast eigendecomposition (hard dependency)
- torch >= 2.1.0
- huggingface-hub >= 0.20.0

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
python -m geolip_svae.diagnostic --hf v12_imagenet128

# Freckles piecemeal test
python -m geolip_svae.noise_diagnostic --model v40_freckles_noise
```

## Spectral Codebook

Noise-native tokenizer mapping text characters to spectral noise signatures:

```python
from geolip_svae import SpectralTokenizer, build_codebook

codebook = build_codebook(save_path='codebook.json')
tokenizer = SpectralTokenizer(codebook)
image, ids, strings = tokenizer.text_to_image("Hello, world!")
```

## License

MIT

## Links

- Models: [huggingface.co/AbstractPhil/geolip-SVAE](https://huggingface.co/AbstractPhil/geolip-SVAE)
- Core: [github.com/AbstractEyes/geolip-core](https://github.com/AbstractEyes/geolip-core)
- Article: [Omega Tokens: Finding The Self Solving Frame](https://huggingface.co/blog/AbstractPhil/multimodal-geometric-fusion)