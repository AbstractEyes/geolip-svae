"""
U5 — Cross-band codebook capacity test

Tests whether the projective-axis property (scratchpad 000101) extends to:

    h2-64 gaussian battery     D=4,  V=32, 64×64       (canonical reference)
    Freckles v40_freckles_noise D=4,  V=?,  64×64       (large-param D=4)
    Johanna v18_johanna_curri.  D=16, V=?,  64×64       (LOW band)

For each model × each calibration distribution (gaussian, sixteen_noise),
extract a Codebook and report:
    - n_axes, n_pairs, n_unpaired
    - mean projective angle
    - deviation from uniform RP^(D-1) baseline
    - is_projective_clean (|deviation| < 0.05)

BINARY OUTCOME determines ft3 direction:
    All clean → batteries packageable as-is, no distillation needed.
    Any not clean → distillation training methodology required for
                    larger variants. Training procedure changes.
                    Documentation updates needed.

Outputs in ``output_dir`` (default /content/phaseU/U5_codebook_capacity/):
    U5_summary.json             Aggregate findings + verdict
    U5_summary.md               Human-readable table + verdict
    codebooks/{model}_{calib}.safetensors + .json  Persisted artifacts

Run from the geolip-svae repo root::

    python -m geolip_svae.tests.U5_codebook_capacity \\
        --output-dir /content/phaseU/U5_codebook_capacity \\
        --n-calib 64

Expects HF_TOKEN in env if pulling Johanna/Freckles checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from geolip_svae.inference import (
    Codebook,
    InferenceEngine,
    extract_codebook,
    load_model,
    make_calibration,
)


# ════════════════════════════════════════════════════════════════════
# Test specification
# ════════════════════════════════════════════════════════════════════

# Model under test: (display_name, hf_version | local_path, expected_D, native_size)
MODELS_TO_TEST: List[Tuple[str, str, int, int]] = [
    # (display_name,        hf_version,                expected_D, size)
    ('h2-64-gaussian',      'h2_64_battery_0_local',   4,          64),
    ('freckles-v40',        'v40_freckles_noise',      4,          64),
    ('johanna-v18',         'v18_johanna_curriculum',  16,         64),
]

CALIBRATIONS = ['gaussian', 'sixteen_noise']

PROJECTIVE_CLEAN_THRESHOLD = 0.05


# ════════════════════════════════════════════════════════════════════
# Custom h2-64 loader
# ════════════════════════════════════════════════════════════════════
# h2-64 is a BatteryArrayModel (HuggingFace PreTrainedModel) and lives
# in a different repo than the Fresnel/Johanna/Freckles checkpoints.
# We expose its single gaussian battery (battery_idx=0) as an
# inference target by wrapping it as a thin shim that mimics
# PatchSVAE's forward → dict[svd] contract.

def _load_h2_64_battery_0(device: str = 'cuda'):
    """Load h2-64 BatteryArrayModel and return battery_idx=0 (gaussian).

    Returns a model instance whose forward(images) produces the same
    dict[svd] structure as PatchSVAE, suitable for extract_codebook.
    """
    from geolip_svae.arrays import BatteryArrayModel
    arr = BatteryArrayModel.from_pretrained(
        'AbstractPhil/geolip-svae-h2-64-batteries',
    )
    arr.to(device).eval()
    bank = arr.bank(battery_idx=0, phase='final')
    return bank


# ════════════════════════════════════════════════════════════════════
# Single-cell run
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_one(
    display_name: str,
    hf_version: str,
    expected_D: int,
    native_size: int,
    calibration_name: str,
    n_calib: int,
    output_dir: Path,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Extract one (model, calibration) codebook and return its result row."""
    print(f"\n  ── {display_name} × {calibration_name} ──")

    # Load model
    t0 = time.time()
    if hf_version == 'h2_64_battery_0_local':
        model = _load_h2_64_battery_0(device=device)
        model_id = 'h2-64/battery_0/final'
    else:
        model, _cfg = load_model(hf_version=hf_version, device=device)
        model_id = hf_version
    load_time = time.time() - t0

    # Verify D matches expectation
    actual_D = getattr(model, 'D', None)
    if actual_D is None:
        return {
            'display_name': display_name,
            'calibration': calibration_name,
            'error': 'model has no D attribute',
        }
    if int(actual_D) != int(expected_D):
        print(f"    ⚠ Expected D={expected_D}, got D={actual_D}")

    # Build calibration tensor
    if calibration_name == 'sixteen_noise' and native_size % 2:
        return {
            'display_name': display_name,
            'calibration': calibration_name,
            'error': 'sixteen_noise requires even size',
        }
    calib = make_calibration(
        calibration_name, n=n_calib, size=native_size, seed=42,
    )

    # Extract codebook
    extract_t0 = time.time()
    cb = extract_codebook(
        model, calib, batch_size=16,
        model_id=model_id, calibration_name=calibration_name,
    )
    extract_time = time.time() - extract_t0

    # Save artifact
    codebook_dir = output_dir / 'codebooks'
    codebook_dir.mkdir(parents=True, exist_ok=True)
    safe_name = display_name.replace('-', '_').replace('/', '_')
    artifact_path = codebook_dir / f"{safe_name}__{calibration_name}"
    cb.save(artifact_path)

    # Free model memory before loading next
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    Loaded in {load_time:.1f}s, extracted in {extract_time:.1f}s")
    print(f"    {cb}")
    print(f"    artifact: {artifact_path}.{{safetensors,json}}")

    return {
        'display_name': display_name,
        'model_id': model_id,
        'hf_version': hf_version,
        'calibration': calibration_name,
        'D': cb.D,
        'V_source': cb.metadata.V,
        'n_axes': cb.n_axes,
        'n_pairs': cb.metadata.n_pairs,
        'n_unpaired': cb.metadata.n_unpaired,
        'mean_projective_angle': cb.metadata.mean_projective_angle,
        'uniform_baseline': cb.metadata.uniform_baseline,
        'deviation': cb.metadata.deviation,
        'is_projective_clean': cb.metadata.is_projective_clean,
        'n_calibration_images': n_calib,
        'calibration_size': native_size,
        'load_time_s': load_time,
        'extract_time_s': extract_time,
        'artifact_path': str(artifact_path),
    }


# ════════════════════════════════════════════════════════════════════
# Verdict synthesis
# ════════════════════════════════════════════════════════════════════

def synthesize_verdict(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate findings into a single verdict block."""
    valid = [r for r in rows if 'error' not in r]
    failed = [r for r in rows if 'error' in r]

    by_model: Dict[str, List[Dict]] = {}
    for r in valid:
        by_model.setdefault(r['display_name'], []).append(r)

    per_model_status = {}
    for name, model_rows in by_model.items():
        clean_count = sum(1 for r in model_rows if r['is_projective_clean'])
        per_model_status[name] = {
            'n_calibrations_tested': len(model_rows),
            'n_projective_clean': clean_count,
            'all_clean': clean_count == len(model_rows),
            'any_clean': clean_count > 0,
            'deviations': {
                r['calibration']: r['deviation'] for r in model_rows
            },
        }

    n_models = len(by_model)
    n_models_all_clean = sum(
        1 for s in per_model_status.values() if s['all_clean']
    )
    n_models_any_clean = sum(
        1 for s in per_model_status.values() if s['any_clean']
    )

    if n_models_all_clean == n_models:
        headline = (
            "ALL_CLEAN — every (model × calibration) tested produced a "
            "projective-clean codebook. Direct extraction works at all "
            "tested D values without distillation. Batteries can be "
            "packaged as-is for ft3 work and the public release."
        )
        ft3_direction = "as_is_packaging"
    elif n_models_any_clean > 0:
        headline = (
            f"PARTIAL — {n_models_all_clean}/{n_models} models "
            f"projective-clean across all calibrations; "
            f"{n_models_any_clean}/{n_models} clean on at least one. "
            f"Direct extraction is calibration-sensitive AT THIS D. "
            f"Ft3 should investigate (a) which calibration produces "
            f"the cleanest codebook per model, (b) whether distillation "
            f"is needed for the partial cases, and (c) whether the failures "
            f"correlate with model size, training mix, or D."
        )
        ft3_direction = "calibration_sensitivity_investigation"
    else:
        headline = (
            "NO_CLEAN — direct extraction does NOT produce projective-clean "
            "codebooks at the tested D values. Distillation training "
            "methodology is required for ft3+. This means: (a) update "
            "training procedure to include a projective-codebook objective, "
            "(b) anchor against a smaller verified bank's codebook during "
            "training, (c) update training documentation. Batteries cannot "
            "currently be packaged as-is for these D values."
        )
        ft3_direction = "distillation_required"

    return {
        'headline': headline,
        'ft3_direction': ft3_direction,
        'n_models': n_models,
        'n_models_all_clean': n_models_all_clean,
        'n_models_any_clean': n_models_any_clean,
        'per_model_status': per_model_status,
        'n_failed_runs': len(failed),
    }


# ════════════════════════════════════════════════════════════════════
# Markdown summary
# ════════════════════════════════════════════════════════════════════

def render_markdown(
    rows: List[Dict[str, Any]],
    verdict: Dict[str, Any],
    n_calib: int,
) -> str:
    lines = []
    lines.append("# U5 — Cross-Band Codebook Capacity Test")
    lines.append("")
    lines.append(f"**Calibration size**: {n_calib} images per cell")
    lines.append(f"**Projective-clean threshold**: |deviation| < {PROJECTIVE_CLEAN_THRESHOLD}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"> {verdict['headline']}")
    lines.append("")
    lines.append(f"**ft3 direction**: `{verdict['ft3_direction']}`")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| Model | Calibration | D | V | n_axes | pairs | unpaired | "
                 "mean angle | uniform | deviation | clean |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if 'error' in r:
            lines.append(
                f"| {r['display_name']} | {r['calibration']} | "
                f"— | — | — | — | — | — | — | — | "
                f"❌ ERROR: {r['error']} |"
            )
            continue
        clean_mark = '✓' if r['is_projective_clean'] else '✗'
        lines.append(
            f"| {r['display_name']} | {r['calibration']} | "
            f"{r['D']} | {r['V_source']} | {r['n_axes']} | "
            f"{r['n_pairs']} | {r['n_unpaired']} | "
            f"{r['mean_projective_angle']:.4f} | "
            f"{r['uniform_baseline']:.4f} | "
            f"{r['deviation']:+.4f} | {clean_mark} |"
        )
    lines.append("")
    lines.append("## Per-model status")
    lines.append("")
    for name, status in verdict['per_model_status'].items():
        all_clean_str = '✓' if status['all_clean'] else '✗'
        lines.append(f"### {name} {all_clean_str}")
        lines.append("")
        lines.append(f"- Calibrations tested: {status['n_calibrations_tested']}")
        lines.append(f"- Projective-clean: {status['n_projective_clean']}")
        lines.append(f"- All clean: {status['all_clean']}")
        for calib, dev in status['deviations'].items():
            lines.append(f"  - {calib}: deviation {dev:+.4f}")
        lines.append("")

    if verdict['n_failed_runs']:
        lines.append(f"## Failed runs: {verdict['n_failed_runs']}")
        lines.append("")
        for r in rows:
            if 'error' in r:
                lines.append(
                    f"- `{r['display_name']}` × `{r['calibration']}`: "
                    f"{r['error']}"
                )
        lines.append("")

    lines.append("## What ft3 does next")
    lines.append("")
    if verdict['ft3_direction'] == 'as_is_packaging':
        lines.append(
            "- Proceed to U1-U4 lens-scope tests using existing batteries.")
        lines.append(
            "- Package h2-64, Johanna, Freckles checkpoints + extracted "
            "codebooks for the public release.")
        lines.append(
            "- Document direct-extraction recipe in the user guide.")
    elif verdict['ft3_direction'] == 'calibration_sensitivity_investigation':
        lines.append(
            "- For each partial model, sweep a wider calibration grid "
            "(more distributions, larger n_calib).")
        lines.append(
            "- Investigate whether the cleanest calibration matches "
            "the model's training distribution.")
        lines.append(
            "- Decide per-model whether direct extraction is acceptable "
            "or distillation is needed.")
    else:  # distillation_required
        lines.append(
            "- Add projective-codebook objective to training procedure.")
        lines.append(
            "- Anchor distillation against verified D=4 h2-64 baseline.")
        lines.append(
            "- Update training documentation; revise public release plan "
            "to ship distilled variants.")
    lines.append("")
    lines.append(
        f"---\n\n*Generated by `tests/U5_codebook_capacity.py`. "
        f"Codebook artifacts saved to `codebooks/`.*"
    )
    return '\n'.join(lines)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='U5 — Cross-band codebook capacity test',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/content/phaseU/U5_codebook_capacity',
        help='Where to write summary + codebook artifacts',
    )
    parser.add_argument(
        '--n-calib', type=int, default=64,
        help='Number of calibration images per cell',
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument(
        '--skip-h2-64', action='store_true',
        help='Skip h2-64 (useful when local checkpoint is unavailable)',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("U5 — Cross-band codebook capacity test")
    print("=" * 72)
    print(f"Output dir:       {output_dir}")
    print(f"Calibration size: {args.n_calib} images/cell")
    print(f"Device:           {args.device}")
    print()

    rows: List[Dict[str, Any]] = []
    overall_t0 = time.time()

    for display_name, hf_version, expected_D, native_size in MODELS_TO_TEST:
        if args.skip_h2_64 and 'h2-64' in display_name:
            print(f"\n  Skipping {display_name} (--skip-h2-64)")
            continue
        for calib_name in CALIBRATIONS:
            try:
                row = run_one(
                    display_name=display_name,
                    hf_version=hf_version,
                    expected_D=expected_D,
                    native_size=native_size,
                    calibration_name=calib_name,
                    n_calib=args.n_calib,
                    output_dir=output_dir,
                    device=args.device,
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"    ✗ FAILED: {type(e).__name__}: {str(e)[:300]}")
                row = {
                    'display_name': display_name,
                    'calibration': calib_name,
                    'error': f'{type(e).__name__}: {str(e)[:200]}',
                    'traceback': tb[:2000],
                }
            rows.append(row)

    overall_elapsed = time.time() - overall_t0

    verdict = synthesize_verdict(rows)
    summary = {
        'rows': rows,
        'verdict': verdict,
        'n_calibration_images': args.n_calib,
        'projective_clean_threshold': PROJECTIVE_CLEAN_THRESHOLD,
        'elapsed_s': overall_elapsed,
    }

    json_path = output_dir / 'U5_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✓ JSON summary: {json_path}")

    md_path = output_dir / 'U5_summary.md'
    md_path.write_text(render_markdown(rows, verdict, args.n_calib))
    print(f"✓ Markdown summary: {md_path}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(verdict['headline'])
    print()
    print(f"ft3 direction: {verdict['ft3_direction']}")
    print(f"Total elapsed: {overall_elapsed/60:.1f} min")


if __name__ == '__main__':
    main()