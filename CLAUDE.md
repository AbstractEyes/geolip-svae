# CLAUDE.md

For Claudes entering this repo fresh. Read this before writing code that
touches PatchSVAE or the arrays infrastructure. These are the things I (a
previous Claude) wish I'd known at the start — they would have caught bugs
faster than re-deriving the architecture by grep.

## What this repo actually is

`geolip-svae` is a Patch-SVD autoencoder package. The core trick is:
image → patches → MLP encode → **sphere-normalize rows of an internal
matrix M** → SVD of M → cross-patch spectral attention on singular values
→ decode. Sphere-norm on the rows of M is what makes the latent space
behave geometrically consistent across inputs. It is load-bearing, not
cosmetic.

The repo also hosts **battery arrays** (`geolip_svae.arrays`): bundles of
many independently-trained PatchSVAE instances exposed as a single
HuggingFace `AutoModel` that emits a per-bank MSE signature. This is newer
than the core SVAE work and has different conventions — see "Arrays
infrastructure" below.

## Load-bearing things you must not casually change

If you find yourself wanting to "clean up" any of these, stop and ask.
They are there for reasons that took real experiments to pin down.

### In `geolip_svae/model.py`:

- **`nn.init.orthogonal_(self.enc_out.weight)`** — load-bearing. Applied
  unconditionally in `__init__`, and re-applied after any L-group init
  override. Do not remove from either spot.
- **`F.normalize(M, dim=-1)`** (the default row normalization) — the whole
  geometric premise. Other `row_norm` modes exist as ablations but should
  not become the default.
- **The `_svd` fp64 autocast-disable** — SVD accuracy depends on fp64.
  Changing this breaks reconstruction fidelity silently.
- **No BatchNorm, no Dropout anywhere** — these were tried and found to
  destabilize the spectral structure. Don't add them back.
- **No global average pooling** — drops accuracy from ~70% to ~29% in
  prior experiments. Flatten or use spatial statistics instead.
- **Optimizer: pure Adam, not AdamW** — weight decay fights the
  geometric structure. This is a training-side rule but applies to any
  reference training code you write in this repo.

### Ablation toggles (F/G/H/L groups) on `PatchSVAE`:

The `activation`, `row_norm`, `svd_mode`, `linear_readout`, `match_params`,
`init_scheme` kwargs are real ablation dimensions, validated by Phase 1+2
of the ablation program. They have defaults that preserve original
behavior. If you change defaults, you change every existing model's
construction path. Don't.

The `linear_readout=True, svd_mode='none', match_params=True` combination
is the **sphere-solver** variant — used by the h2-64 battery array. It
replaces SVD with a learned linear readout; downstream code treats the
readout output as U, column norms as S, identity as Vt.

## The `inference/` package (the canonical inference path)

Production inference goes through `geolip_svae.inference.InferenceEngine`,
not through `model.forward()` directly and not through the legacy flat
encode/decode/reconstruct/compute_axis_codebook functions. The flat API
still works via shims in `inference/legacy.py`, but new code uses the
engine.

The architectural rule for the inference layer is **no core limiters**.
`patch_size`, `tile_size`, calibration distribution, codebook source,
aggregation method — all overridable at engine construction or per-call.
If you find yourself hard-coding an assumption (e.g. "patch_size is
always 4"), that's a regression. The engine accepts overrides because
users may want to inference Fresnel with non-native patch sizes for
diagnostic reasons; the model will fail loudly if it doesn't tolerate
the override, which is the desired behavior.

`PatchSVAEv2` was removed entirely (scratchpad 000107). Loading a v2
checkpoint raises `UnsupportedCheckpointError` with a clear "re-train
as v1" message. Don't try to revive it.

Module ownership inside `inference/`:

```
loading.py     — load_model + VERSIONS registry + checkpoint resolution
scaling.py     — encode_at_scale / reconstruct_at_scale (direct/tile/auto)
calibration.py — calibration data generators with registry pattern
codebook.py    — Codebook artifact + extract_codebook + collapse helpers
engine.py      — InferenceEngine orchestrator
legacy.py      — back-compat shims (encode/decode/reconstruct/...)
```

When extending inference, put new code in the module that owns its
responsibility — not in `engine.py` just because the engine is the
top-level surface.

## Projective-axis codebooks

The empirical finding (scratchpad 000101, ft2 article): every trained
sphere-solver tested at D=3, 4, 16 produces an M tensor whose rows,
when antipodal pairs are merged via mutual-strongest matching, form a
near-uniformly-distributed codebook on ℝP^(D-1). Verified across
h2-64, Freckles, Johanna, plus the earlier single-bank A0–A3 probes.
The collapse method is a deterministic tensor operation, not a
learned property.

The `Codebook` artifact is how you use this:

```python
from geolip_svae.inference import (
    InferenceEngine, Codebook, extract_codebook, make_calibration,
)

calib = make_calibration('sixteen_noise', n=64, size=64)
cb = extract_codebook(
    model, calib,
    model_id='v40_freckles_noise', calibration_name='sixteen_noise',
)
cb.save('codebooks/freckles_v40__sixteen_noise')   # safetensors + JSON
cb = Codebook.load('codebooks/freckles_v40__sixteen_noise')

engine = InferenceEngine(model)
engine.attach_codebook(cb)
out = engine.encode_axes(images)   # projects M onto cb.axes
```

Things to know before writing codebook code:

- The four collapse helpers (`identify_antipodal_pairs`,
  `collapse_to_axes`, `_canonicalize_sign`, `_aggregate_M`) live in
  `inference/codebook.py`. They were relocated from `arrays/model.py`,
  which now re-imports them for back-compat. Don't add new collapse
  logic in `arrays/model.py` — go to `codebook.py`.

- `extract_codebook(...)` returns a full `Codebook` artifact with
  metadata. The legacy `compute_axis_codebook(...)` shim returns just
  the axes tensor — fine for old code, but new code uses
  `extract_codebook` so the metadata flows through.

- Compatibility check is on by default. `engine.attach_codebook(cb)`
  raises `CodebookIncompatibleError` if `cb.D != model.D`. Cross-model
  experiments (apply Codebook X to model Y) require
  `InferenceEngine(model, require_codebook_compatibility=False)` at
  construction. This is opt-in for a reason — silently mis-projecting
  M onto the wrong axes is a hard bug to debug.

- Patch-aggregation default is `'mean'` (per-patch averaging across
  all patches per image). The legacy `patch_idx=0` path was a silent
  bug for downstream classification — it carried unchallenged from
  training-time CV measurement into A0–A3 (where it didn't matter)
  into production code (where it cost ~88% of the spatial signal).
  Kept as opt-in for reproducing A0–A3 verifications, NOT for
  production use. See scratchpad 000104.

## Architecture-identity invariants (for battery arrays)

When a PatchSVAE is instantiated to load weights from a specific trained
checkpoint, **every module and every shape must match exactly or
`load_state_dict` silently partial-loads**. Symptoms: model runs but
produces garbage reconstructions; MSE much worse than training reported.

Pitfalls that have actually happened:

- **`smooth_mid`** — the `BoundarySmooth` mid-channel count is
  ps-dependent by default (`16 if ps >= 16 else 8`). The h2-64 training
  code used unconditional `mid=16`. If you build a PatchSVAE to load
  h2-64 weights, you **must** pass `smooth_mid=16` explicitly. The 440
  missing params from this mismatch manifest as 440 parameters missing
  from the state_dict load, which `strict=False` silently tolerates.
  Always verify param counts after instantiation.

- **`n_heads`** — the `SpectralCrossAttention` auto-adjusts `n_heads`
  to divide `D` evenly. At D=4 with `n_heads=4`, heads=4, head_dim=1.
  If the training code used a different auto-adjust path, shapes
  mismatch. When in doubt, verify by instantiating and printing
  `state_dict().keys()` with shapes before attempting a load.

- **Sphere-solver path** — if loading h2-64 weights, the config must
  include `svd_mode='none', linear_readout=True, match_params=True`.
  Without `linear_readout=True`, there's no `readout` module in the
  state_dict. Without `svd_mode='none'`, the encode path tries to run
  SVD on a `readout`-less model.

### The debug move that catches all of these

Before writing any loader, instantiate the class with the target kwargs
and compare state_dict keys + shapes against one real checkpoint from HF:

```python
import torch
from huggingface_hub import hf_hub_download
from geolip_svae.model import PatchSVAE

model = PatchSVAE(**target_kwargs)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
for k, v in model.state_dict().items():
    print(f"  {k:<45}  {list(v.shape)}")

# Compare against a real checkpoint
ckpt = torch.load(hf_hub_download(repo_id=REPO, filename=ONE_CKPT_PATH),
                    map_location='cpu', weights_only=False)
real_state = ckpt['model_state']
print(f"\nReal checkpoint keys:")
for k, v in real_state.items():
    print(f"  {k:<45}  {list(v.shape)}")
```

If the two don't match exactly, fix kwargs before going further. Do not
proceed to "load with strict=False and hope it works." It won't.

## Arrays infrastructure

Lives in `geolip_svae/arrays/`. The design is spec-driven:

```
arrays/
├── config.py         # BatteryArrayConfig (PretrainedConfig)
├── model.py          # BatteryArrayModel — dynamic battery-class dispatch
├── builder.py        # build_array(spec_name) — end-to-end pipeline
└── specs/
    ├── h2_64.py      # first spec, use as template
    └── __init__.py   # registry
```

`BatteryArrayModel` is **architecture-agnostic**. At init it reads
`config.battery_module` + `config.battery_class`, imports the class by
name, and instantiates `n_banks` of them with `config.battery_kwargs`.
This means adding a new array type does **not** require editing
`BatteryArrayModel`. Just write a new spec.

`BatteryArrayModel` also has projective-codebook methods:
`compute_axis_codebook(battery_idx, phase, ...)`,
`compute_axis_codebooks(targets, ...)` (batched across many banks),
`encode_axes(images, ...)`, `quantize_axes(images, ...)`. These delegate
to the helpers in `inference/codebook.py` (relocated from this module
during the inference rebuild). The `arrays/model.py` re-imports
preserve back-compat for any code still importing
`identify_antipodal_pairs` etc. from here.

### Adding a new array spec

1. Copy `specs/h2_64.py` to `specs/your_new_array.py`.
2. Update `BATTERY_CLASS`, `BATTERY_MODULE`, `BATTERY_KWARGS` for the
   architecture that was trained.
3. Update `N_BATTERIES`, `EPOCH_PHASE_NAMES`, `SOURCE_REPO`.
4. Rewrite `get_configs()` to describe how the training configs are
   organized (which battery sees which training data).
5. Rewrite `checkpoint_path()` and `report_path()` if the training repo
   uses a different directory scheme.
6. Add your spec to `SPECS` in `specs/__init__.py`.
7. Run `build_array(spec_name="your_new_array")` and verify param counts.

### Verifying a spec before running the full builder

The full builder downloads ~190 checkpoints. If the spec is wrong,
that's a lot of wasted bandwidth. Sanity-check first:

```python
from geolip_svae.arrays.specs import get_spec
from geolip_svae.arrays.config import BatteryArrayConfig
from geolip_svae.arrays.model import BatteryArrayModel

spec = get_spec("your_new_array")
configs = spec.get_configs()
assert len(configs) == spec.N_BATTERIES

# Instantiate with dummy metadata to verify architecture matches training
dummy = [{'battery_id': i, 'subgroup': 'x', 'variant': f'v{i}',
           'noise_types': [0],
           'epoch_phases': {'epoch_1': 1, 'best': 5, 'final': 10},
           'per_phase_mse': {}, 'per_phase_cv': {}} for i in range(spec.N_BATTERIES)]
config = BatteryArrayConfig(
    battery_class=spec.BATTERY_CLASS,
    battery_module=spec.BATTERY_MODULE,
    battery_kwargs=spec.BATTERY_KWARGS,
    n_batteries=spec.N_BATTERIES,
    n_epoch_phases=len(spec.EPOCH_PHASE_NAMES),
    epoch_phase_names=spec.EPOCH_PHASE_NAMES,
    batteries=dummy,
)
model = BatteryArrayModel(config)
print(f"Per-bank: {sum(p.numel() for p in model.banks[0].parameters()):,}")
# Compare against one real checkpoint from the source repo
```

## `experimental/` subpackage

Three modules: `spectral_cell`, `spectral_battery`, `experimental_codebook`
(the last is the renamed `spectral_codebook`). They're preserved earlier
sphere-solver variants and codebook tooling that pre-date the
projective-axis discovery. They are NOT the canonical inference path.

Be aware:

- `experimental_codebook` is **not** the projective `Codebook` artifact.
  It pre-dates the antipodal-collapse method and reports different
  geometric statistics. Don't conflate the two. Use `Codebook` from
  `inference/codebook.py` for new work.

- Anything in `experimental/` may change without warning. The package's
  stability promise is on `inference/` and the canonical surfaces, not
  here.

- These modules import at package-load time via `experimental/__init__.py`.
  Module-level `print()` statements there fire on every
  `import geolip_svae`. `spectral_battery.py` had a "Has conduit" /
  "Has no conduit, defaulting to invalid code" pair at the top that
  escaped this way; both were silenced. If you add new code here,
  no module-level prints — use logging or just track state in a flag.

## Common problems and where to look

### "AttributeError: module '__main__' has no attribute '__file__'"
Transformers inspects `cls.__module__.__file__` at `PreTrainedModel.__init__`
for MoE detection. If a `PreTrainedModel` subclass is defined inline
(e.g., in a Colab cell via `exec()`), `cls.__module__` resolves to
`__main__`, which has no `__file__`.

**Fix: define the class in a real importable module.** That's why
`BatteryArrayModel` lives in `geolip_svae.arrays.model` and not inline.
Do not try to work around this with `__main__.__file__ = '<colab>'` —
transformers will then try to `open()` the string and get a
FileNotFoundError.

### State dict loads but reconstructions are garbage
Almost certainly an architecture mismatch between the class you
instantiated and the class that produced the weights. Run the "debug
move" above. 99% of the time it's `smooth_mid`, `n_heads`, or a missing
ablation flag.

### Weights load with `missing=[...]` under `strict=False`
The keys in the missing list tell you exactly which modules are absent
from the state_dict vs. expected. If `readout.weight` is missing, you
forgot `linear_readout=True`. If `boundary_smooth.net.0.weight` has a
shape mismatch, you forgot `smooth_mid=16`. If cross-attention keys are
missing, `n_cross` or `n_heads` is wrong.

### AutoModel.from_pretrained() says "no such model type 'battery_array'"
The auto-registration in `geolip_svae/arrays/__init__.py` didn't run.
Either `geolip-svae` isn't pip-installed in the active environment, or
the registration silently swallowed an ImportError. Check by:

```python
import geolip_svae.arrays  # triggers registration
from transformers import AutoConfig
print("battery_array" in AutoConfig._model_mapping._model_type_to_module_name
      if hasattr(AutoConfig, '_model_mapping') else "check via AutoModel.register")
```

Or just pass `trust_remote_code=True` — the `auto_map` in `config.json`
provides a fallback path.

### Builder hangs on "downloading checkpoints"
`hf_hub_download` has no visible progress per-call in some environments.
Watch `n_downloaded` in the periodic `[16/64]` progress lines.
Typically ~2MB/file × ~190 files = ~380MB total, a few minutes on a
good connection.

### Builder crashes in `load_state_dict`
Look at what `_extract_model_state` returned. Training checkpoints use
`'model_state'` (ablation trainer) or `'model_state_dict'` (older
trainers). If the checkpoint is just a raw state_dict without a
wrapper, the fallback branch catches it. If it uses yet another key,
add it to `_extract_model_state`.

## Things that will look wrong but aren't

- **`cross_attn.0.qkv.weight: [12, 4]`** at D=4 — that's `3*D` for QKV
  packed, n_heads=4, head_dim=1. Correct.
- **Per-bank param count of 57,215** for the h2-64 sphere-solver
  variant — correct, verified against training output.
- **`svd_mode='none'` with `linear_readout=True`** creating an identity
  Vt — the decode path uses it as `bmm(U*S.unsqueeze(1), I) = U*S`,
  which is correct for the sphere-solver.
- **The research code in `ablation_trainer.py` / `johanna_F_trainer.py`
  re-implements primitives that also exist in `geolip_svae.model`** —
  this is training-side code, not imported by the package. Copy-adjacent
  rather than upstream-linked. The `PatchSVAE_F_Ablation` class there
  is the predecessor to the extended `PatchSVAE` in the package; they
  should produce identical state_dicts for H-group configs.

## When you're uncertain

This repo has extensive experimental lineage. If you're about to change
something that looks like a free choice (default kwarg, init scheme,
normalization, optimizer, architecture detail), it probably isn't — it
was validated by a specific experiment. Ask before changing.

If you're asked to extend something and the extension requires touching
load-bearing code, propose the change explicitly and wait for
confirmation. The repo maintainer tracks long research threads, and an
unexpected default change can invalidate months of training runs.