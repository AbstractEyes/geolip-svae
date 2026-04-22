"""
geolip_svae.arrays.builder
============================
Generic builder for battery arrays. Given an array spec (which describes
the battery class, training-config layout, and checkpoint path scheme),
this downloads all checkpoints from the source HF repo, assembles them
into a BatteryArrayModel, and writes out a standard HF checkpoint
(config.json + model.safetensors).

The builder is spec-driven so a single entry point serves h2-64 today
and future arrays (h3, frequency-triad, etc.) tomorrow.

Usage (after pip install geolip-svae):
    from geolip_svae.arrays import build_array

    # Build and save locally (no upload):
    build_array(spec_name="h2_64", output_dir="./out/h2_64")

    # Build and upload to HF (uses the spec's SOURCE_REPO by default):
    build_array(spec_name="h2_64", upload=True)

    # Override target repo:
    build_array(
        spec_name="h2_64",
        target_repo="AbstractPhil/geolip-svae-h2-64",
        upload=True,
    )
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file as save_safetensors

from geolip_svae.arrays.config import BatteryArrayConfig
from geolip_svae.arrays.model import BatteryArrayModel
from geolip_svae.arrays.specs import get_spec


# ── Metadata fetch ───────────────────────────────────────────────────

def fetch_battery_metadata(
    spec,
    source_repo: str,
    hf_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """For each config in spec.get_configs(), fetch final_report.json
    from source_repo and derive per-phase epoch + MSE + CV.
    """
    configs = spec.get_configs()
    metadata = []

    print(f"Fetching metadata for {len(configs)} batteries from {source_repo}...")

    for i, cfg in enumerate(configs):
        report_path = spec.report_path(cfg)
        try:
            local = hf_hub_download(
                repo_id=source_repo,
                filename=report_path,
                token=hf_token,
            )
            with open(local) as f:
                report = json.load(f)
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] ⚠ missing {report_path}: {e}")
            continue

        # Derive epoch_1 / best / final
        per_epoch = report.get('per_epoch_metrics', [])
        if not per_epoch:
            # Fallback if trajectory isn't in the report
            n_epochs = report.get('num_epochs_run', 10)
            best_epoch = final_epoch = n_epochs
            best_mse = final_mse = first_mse = report.get('test_mse', float('nan'))
            best_cv = final_cv = first_cv = report.get('cv_ema_final', 0.0)
        else:
            finite_epochs = [
                ep for ep in per_epoch
                if ep.get('params_finite', True)
                and ep.get('test_mse') is not None
                and ep.get('test_mse') == ep.get('test_mse')  # NaN check
            ]
            if finite_epochs:
                best_ep = min(finite_epochs, key=lambda e: e['test_mse'])
                best_epoch = best_ep['epoch']
                best_mse = best_ep['test_mse']
                best_cv = best_ep.get('cv_ema', 0.0)
            else:
                best_ep = per_epoch[-1]
                best_epoch = best_ep['epoch']
                best_mse = best_ep.get('test_mse', float('nan'))
                best_cv = best_ep.get('cv_ema', 0.0)

            first_ep = per_epoch[0]
            first_mse = first_ep.get('test_mse', float('nan'))
            first_cv = first_ep.get('cv_ema', 0.0)

            final_ep = per_epoch[-1]
            final_epoch = final_ep['epoch']
            final_mse = final_ep.get('test_mse', float('nan'))
            final_cv = final_ep.get('cv_ema', 0.0)

        metadata.append({
            **cfg,  # carries battery_id, subgroup, variant, noise_types
            'hf_path': f"{cfg['subgroup']}/{cfg['variant']}",
            'epoch_phases': {
                'epoch_1': 1,
                'best': best_epoch,
                'final': final_epoch,
            },
            'per_phase_mse': {
                'epoch_1': first_mse,
                'best': best_mse,
                'final': final_mse,
            },
            'per_phase_cv': {
                'epoch_1': first_cv,
                'best': best_cv,
                'final': final_cv,
            },
            'params_finite_final': report.get('params_finite', True),
        })

        if (i + 1) % 16 == 0:
            print(f"  [{i+1}/{len(configs)}] through {cfg['subgroup']}/{cfg['variant']}")

    print(f"Fetched metadata for {len(metadata)}/{len(configs)} batteries")
    return metadata


# ── Checkpoint download ──────────────────────────────────────────────

def download_all_checkpoints(
    spec,
    metadata: List[Dict[str, Any]],
    source_repo: str,
    hf_token: Optional[str] = None,
) -> Dict[int, Dict[str, Path]]:
    """For each battery, download the unique epoch files needed for its
    three phases. Dedups when best == first or best == final.

    Returns: {battery_id: {phase: Path_to_checkpoint}}
    """
    checkpoints = {}
    n_downloaded = 0
    n_deduped = 0

    print(f"Downloading checkpoints...")

    for i, meta in enumerate(metadata):
        battery_id = meta['battery_id']
        epochs = meta['epoch_phases']
        unique_epochs = set(epochs.values())
        epoch_to_path = {}

        for ep_num in unique_epochs:
            filename = spec.checkpoint_path(meta, ep_num)
            try:
                local = hf_hub_download(
                    repo_id=source_repo,
                    filename=filename,
                    token=hf_token,
                )
                epoch_to_path[ep_num] = Path(local)
                n_downloaded += 1
            except Exception as e:
                print(f"  ⚠ battery {battery_id} epoch {ep_num}: {e}")
                epoch_to_path[ep_num] = None

        n_deduped += (len(epochs) - len(unique_epochs))

        checkpoints[battery_id] = {
            phase: epoch_to_path.get(ep_num)
            for phase, ep_num in epochs.items()
        }

        if (i + 1) % 16 == 0:
            print(f"  [{i+1}/{len(metadata)}] {n_downloaded} files "
                  f"({n_deduped} deduped)")

    print(f"Downloaded {n_downloaded} unique files ({n_deduped} dedupe savings)")
    return checkpoints


def _extract_model_state(path: Path) -> Dict[str, torch.Tensor]:
    """Extract model state_dict from a training checkpoint."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('model_state', 'model_state_dict', 'state_dict'):
        if key in ckpt:
            return ckpt[key]
    # Fallback: assume the file IS the state dict
    return ckpt


# ── Assembly ─────────────────────────────────────────────────────────

def assemble_array_model(
    spec,
    metadata: List[Dict[str, Any]],
    checkpoints: Dict[int, Dict[str, Path]],
    source_repo: str,
) -> BatteryArrayModel:
    """Instantiate BatteryArrayModel and load each bank's weights."""

    config = BatteryArrayConfig(
        battery_class=spec.BATTERY_CLASS,
        battery_module=spec.BATTERY_MODULE,
        battery_kwargs=spec.BATTERY_KWARGS,
        n_batteries=spec.N_BATTERIES,
        n_epoch_phases=len(spec.EPOCH_PHASE_NAMES),
        epoch_phase_names=spec.EPOCH_PHASE_NAMES,
        batteries=metadata,
        source_repo=source_repo,
        built_at_utc=datetime.now(timezone.utc).isoformat(),
        array_spec_name=spec.NAME,
    )

    model = BatteryArrayModel(config)
    print(f"Instantiated BatteryArrayModel with {config.n_banks} banks")
    print(f"  battery class: {config.battery_module}.{config.battery_class}")
    print(f"  params/bank: {sum(p.numel() for p in model.banks[0].parameters()):,}")
    print(f"  total params: {sum(p.numel() for p in model.parameters()):,}")

    n_loaded = 0
    n_failed = 0
    mismatches = []

    for meta in metadata:
        battery_id = meta['battery_id']
        battery_ckpts = checkpoints.get(battery_id, {})

        for phase in spec.EPOCH_PHASE_NAMES:
            ckpt_path = battery_ckpts.get(phase)
            if ckpt_path is None:
                print(f"  ⚠ battery {battery_id} {phase}: no checkpoint")
                n_failed += 1
                continue

            try:
                state = _extract_model_state(ckpt_path)
                bank = model.bank(battery_id, phase)
                missing, unexpected = bank.load_state_dict(state, strict=False)
                if missing or unexpected:
                    mismatches.append({
                        'battery_id': battery_id,
                        'phase': phase,
                        'missing': missing,
                        'unexpected': unexpected,
                    })
                n_loaded += 1
            except Exception as e:
                print(f"  ✗ battery {battery_id} {phase}: {e}")
                n_failed += 1

    print(f"Loaded {n_loaded} / {spec.N_BATTERIES * len(spec.EPOCH_PHASE_NAMES)} "
          f"bank weights (failed: {n_failed})")

    if mismatches:
        print(f"⚠ {len(mismatches)} banks had state_dict key mismatches:")
        for m in mismatches[:3]:
            print(f"    battery {m['battery_id']} {m['phase']}: "
                  f"missing={len(m['missing'])} unexpected={len(m['unexpected'])}")
            if m['missing']:
                print(f"      missing example: {m['missing'][0]}")
            if m['unexpected']:
                print(f"      unexpected example: {m['unexpected'][0]}")

    return model


# ── Save ─────────────────────────────────────────────────────────────

def save_array_model(
    model: BatteryArrayModel,
    output_dir: Path,
) -> Path:
    """Write config.json + model.safetensors + manifest.json to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json (HF PretrainedConfig format with auto_map)
    config_dict = model.config.to_dict()
    # Add auto_map for trust_remote_code=True fallback loading
    config_dict['auto_map'] = {
        "AutoConfig": "geolip_svae.arrays.config.BatteryArrayConfig",
        "AutoModel": "geolip_svae.arrays.model.BatteryArrayModel",
    }
    config_dict['architectures'] = ["BatteryArrayModel"]
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"  saved config.json")

    # 2. model.safetensors
    state = model.state_dict()
    state = {k: v.contiguous() for k, v in state.items()}
    safetensors_path = output_dir / "model.safetensors"
    save_safetensors(state, str(safetensors_path))
    sz_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"  saved model.safetensors ({sz_mb:.1f} MB)")

    # 3. manifest.json (human-readable summary)
    manifest = {
        'model_type': model.config.model_type,
        'n_batteries': model.config.n_batteries,
        'n_epoch_phases': model.config.n_epoch_phases,
        'n_banks': model.config.n_banks,
        'battery_class': model.config.battery_class,
        'battery_module': model.config.battery_module,
        'battery_kwargs': model.config.battery_kwargs,
        'params_per_bank': sum(p.numel() for p in model.banks[0].parameters()),
        'total_params': sum(p.numel() for p in model.parameters()),
        'source_repo': model.config.source_repo,
        'built_at_utc': model.config.built_at_utc,
        'array_spec_name': model.config.array_spec_name,
        'epoch_phase_names': model.config.epoch_phase_names,
        'batteries': model.config.batteries,
    }
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  saved manifest.json")

    return output_dir


def upload_to_hf(
    output_dir: Path,
    target_repo: str,
    hf_token: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> None:
    """Upload the saved array artifact to HF."""
    api = HfApi(token=hf_token)
    if commit_message is None:
        commit_message = f"Battery array build {datetime.now(timezone.utc).isoformat()}"
    print(f"Uploading {output_dir} → {target_repo}...")
    api.upload_folder(
        repo_id=target_repo,
        folder_path=str(output_dir),
        path_in_repo=".",
        token=hf_token,
        commit_message=commit_message,
    )
    print(f"✓ uploaded to {target_repo}")


# ── One-shot entry ───────────────────────────────────────────────────

def build_array(
    spec_name: str,
    source_repo: Optional[str] = None,
    target_repo: Optional[str] = None,
    output_dir: Optional[Path] = None,
    upload: bool = False,
    hf_token: Optional[str] = None,
) -> BatteryArrayModel:
    """Full assembly pipeline: metadata → checkpoints → model → save → upload.

    Args:
        spec_name: name of the spec in geolip_svae.arrays.specs (e.g., 'h2_64')
        source_repo: HF repo with training artifacts. Defaults to spec.SOURCE_REPO.
        target_repo: HF repo to upload assembled array to. Defaults to source_repo.
        output_dir: local directory for the assembled artifact.
        upload: if True, upload to target_repo.
        hf_token: HF auth token. Defaults to env HF_TOKEN.

    Returns:
        The assembled BatteryArrayModel (also saved to disk).
    """
    spec = get_spec(spec_name)

    if source_repo is None:
        source_repo = spec.SOURCE_REPO
    if target_repo is None:
        target_repo = source_repo
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    if output_dir is None:
        output_dir = Path(f"./build/{spec_name}")

    print("=" * 70)
    print(f"BATTERY ARRAY BUILDER — spec: {spec_name}")
    print(f"  source: {source_repo}")
    print(f"  target: {target_repo}")
    print(f"  output: {output_dir}")
    print(f"  upload: {upload}")
    print("=" * 70)

    metadata = fetch_battery_metadata(spec, source_repo, hf_token)
    checkpoints = download_all_checkpoints(spec, metadata, source_repo, hf_token)
    model = assemble_array_model(spec, metadata, checkpoints, source_repo)
    save_array_model(model, output_dir)

    if upload:
        upload_to_hf(output_dir, target_repo, hf_token)

    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    return model