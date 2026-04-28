"""
SVAE Streaming Continuation Trainer
====================================
Long-running fine-tuning for any trained SVAE model. Resumes from
{hf_version}/checkpoints/best.pt and trains continuously on streamed
random crops, accumulating millions of unique training samples.

This is the trainer that produced v50_fresnel_64's 140M+ image run —
the "sublens perspective" mode where 64×64 random crops of larger
source images yield essentially infinite unique geometric samples.

Key differences from the unified train.py
------------------------------------------
  * Step-based, not epoch-based — runs to --max-steps (default 500K)
  * IterableDataset with infinite random crops, not finite Dataset
  * Resume-from-best by default (not from-scratch or named pretrained)
  * Pure MSE loss with grad-clip on all params (post-spectrum-lock regime)
  * Independent held-out val set (ChocolateDave/imagenet-64)
  * Streaming JSON log accumulates per-checkpoint records, pushed to HF

Usage:
    # Continue v50_fresnel_64 training from its current best
    python -m geolip_svae.train_streaming --hf-version v50_fresnel_64

    # Continue freckles_64 training similarly
    python -m geolip_svae.train_streaming --hf-version v40_freckles_noise

    # Use LAION instead of ImageNet
    python -m geolip_svae.train_streaming \\
        --hf-version v50_fresnel_64 --source laion

    # Override step count, batch size, lr
    python -m geolip_svae.train_streaming \\
        --hf-version v50_fresnel_64 --max-steps 100000 --batch 512 --lr 1e-5

Augmentations match the original fresnel_trainer_64_256.py: random crop
and horizontal flip. Each streamed sample is yielded once with no caching.

Diagnostics logged to training_log.json (uploaded to HF every save):
  * step, train_mse, val_mse, total_images_seen, elapsed_seconds
  * S_mean, S_std, S_delta, erank, ratio
  * row_cv, cv_in_band
  * alpha_mean, alpha_std (per-cross-attn-layer averaged)
  * grad_max
"""

import os
import math
import json
import time
import argparse
from typing import Optional, Dict, Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from geolip_svae.model import PatchSVAE, cv_of


# ── HuggingFace auth ─────────────────────────────────────────────────

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
except Exception:
    pass


# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_HF_REPO = 'AbstractPhil/geolip-SVAE'
DEFAULT_TARGET_SIZE = 64  # 64×64 patches — matches Freckles/Fresnel-64 architecture

# Streaming source registry — each entry produces the per-iter random crop yield.
# 'imagenet_256' randomly crops 64×64 from 256×256 source images.
# 'laion' (placeholder) would do the same but from LAION-Aesthetics, which has
# more stylistic/photographic diversity.
STREAMING_SOURCES = {
    'imagenet_256': {
        'hf_path': 'benjamin-paine/imagenet-1k-256x256',
        'split': 'train',
        'description': 'ImageNet-256 random crops',
    },
    'laion': {
        'hf_path': 'laion/laion2B-en-aesthetic',
        'split': 'train',
        'description': 'LAION-Aesthetics (streaming)',
    },
}

VAL_SOURCES = {
    64:  {'hf_path': 'ChocolateDave/imagenet-64',  'split': 'val'}, # chocolatedave has bad labels but good images.
    128: {'hf_path': 'benjamin-paine/imagenet-1k-128x128', 'split': 'validation'},
}


# ═══════════════════════════════════════════════════════════════════
# STREAMING DATASET
# ═══════════════════════════════════════════════════════════════════

class StreamingCropDataset(torch.utils.data.IterableDataset):
    """Random-crop streaming dataset. Yields target_size × target_size tensors.

    For each item, picks a random source image, applies optional horizontal
    flip, takes a random crop. Skips images smaller than target_size.

    `hf_dataset` should be an HF Dataset (in-memory or streaming). The dataset
    must have items with key 'image' returning a PIL.Image.
    """

    def __init__(self, hf_dataset, target_size: int = DEFAULT_TARGET_SIZE,
                 horizontal_flip: bool = True, normalize_mean=None, normalize_std=None):
        super().__init__()
        import torchvision.transforms as T
        self.ds = hf_dataset
        self.target = target_size
        self.flip = T.RandomHorizontalFlip() if horizontal_flip else nn.Identity()
        self.to_tensor = T.ToTensor()
        self.normalize = None
        if normalize_mean is not None and normalize_std is not None:
            self.normalize = T.Normalize(normalize_mean, normalize_std)
        # Indexable check: streaming HF datasets don't support len(), so we
        # detect that via duck typing once at init.
        self._is_indexable = hasattr(hf_dataset, '__len__')
        try:
            self._n = len(hf_dataset) if self._is_indexable else None
        except TypeError:
            self._n = None
            self._is_indexable = False

    def __iter__(self) -> Iterator[torch.Tensor]:
        import torchvision.transforms.functional as TF
        t = self.target

        if self._is_indexable:
            # Random index access — preferred mode (matches original trainer)
            n = self._n
            while True:
                try:
                    idx = torch.randint(0, n, (1,)).item()
                    img = self.ds[idx]['image']
                    if img is None:
                        continue
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    w, h = img.size
                    if w < t or h < t:
                        continue
                    i = torch.randint(0, h - t + 1, (1,)).item()
                    j = torch.randint(0, w - t + 1, (1,)).item()
                    crop = TF.crop(img, i, j, t, t)
                    crop = self.flip(crop)
                    out = self.to_tensor(crop)
                    if self.normalize is not None:
                        out = self.normalize(out)
                    yield out
                except Exception:
                    continue
        else:
            # Streaming-iterator mode (LAION case)
            while True:
                try:
                    for record in self.ds:
                        img = record.get('image', None)
                        if img is None:
                            continue
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        w, h = img.size
                        if w < t or h < t:
                            continue
                        i = torch.randint(0, h - t + 1, (1,)).item()
                        j = torch.randint(0, w - t + 1, (1,)).item()
                        crop = TF.crop(img, i, j, t, t)
                        crop = self.flip(crop)
                        out = self.to_tensor(crop)
                        if self.normalize is not None:
                            out = self.normalize(out)
                        yield out
                except Exception:
                    continue


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════

_val_loader_cache = {}


def get_val_loader(target_size: int, batch_size: int = 256):
    """Cached val loader for the given target size."""
    if target_size in _val_loader_cache:
        return _val_loader_cache[target_size]

    import torchvision.transforms as T
    from datasets import load_dataset

    if target_size not in VAL_SOURCES:
        # Fall back to ImageNet-64 even for non-64 sizes (better than nothing)
        print(f"  [warn] no val source registered for size {target_size}, "
              f"using imagenet-64")
        spec = VAL_SOURCES[64]
    else:
        spec = VAL_SOURCES[target_size]

    print(f"  Loading val from {spec['hf_path']} (split={spec['split']})...")
    val_hf = load_dataset(spec['hf_path'], split=spec['split'])

    class ValDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, t):
            self.ds = hf_ds
            self.to_tensor = T.ToTensor()
            self.t = t

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img = self.ds[idx]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Center crop / resize to target
            w, h = img.size
            if (w, h) != (self.t, self.t):
                import torchvision.transforms.functional as TF
                if w >= self.t and h >= self.t:
                    # Center crop
                    i = (h - self.t) // 2
                    j = (w - self.t) // 2
                    img = TF.crop(img, i, j, self.t, self.t)
                else:
                    img = TF.resize(img, (self.t, self.t))
            return self.to_tensor(img)

    loader = torch.utils.data.DataLoader(
        ValDataset(val_hf, target_size), batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True)
    _val_loader_cache[target_size] = loader
    return loader


def run_val(model, target_size, device, max_batches=50):
    """Quick validation MSE on held-out images at target resolution."""
    model.eval()
    loader = get_val_loader(target_size)
    total_mse, n = 0.0, 0
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if i >= max_batches:
                break
            imgs = imgs.to(device)
            out = model(imgs)
            total_mse += F.mse_loss(out['recon'], imgs).item() * len(imgs)
            n += len(imgs)
    model.train()
    return total_mse / max(n, 1)


# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT LOAD / RESUME
# ═══════════════════════════════════════════════════════════════════

def load_model_from_hf(hf_repo: str, hf_version: str, device) -> tuple:
    """Download {hf_version}/checkpoints/best.pt and rebuild the model.

    Returns (model, model_cfg, prior_step, prior_images_seen).
    Uses the `config` dict embedded in the checkpoint to reconstruct architecture
    — this works for all checkpoints saved by the unified trainer or its
    predecessors that embed a config dict.
    """
    from huggingface_hub import hf_hub_download

    print(f"  Loading {hf_version}/checkpoints/best.pt from {hf_repo}...")
    ckpt_path = hf_hub_download(
        repo_id=hf_repo,
        filename=f'{hf_version}/checkpoints/best.pt',
        repo_type='model'
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    cfg = ckpt.get('config', {})
    if not cfg:
        raise RuntimeError(
            f"Checkpoint at {hf_version}/checkpoints/best.pt has no 'config' "
            "dict; cannot reconstruct architecture. Pass --V/--D/--patch-size "
            "etc. manually or edit this trainer to add manual override."
        )

    # Build PatchSVAE from config — handle both old-style ('n_cross_layers')
    # and new-style ('n_cross') keys
    n_cross = cfg.get('n_cross', cfg.get('n_cross_layers', 2))
    model_kwargs = dict(
        V=cfg.get('V', 256),
        D=cfg.get('D', 16),
        ps=cfg.get('patch_size', 16),
        hidden=cfg.get('hidden', 768),
        depth=cfg.get('depth', 4),
        n_cross=n_cross,
    )
    if 'n_heads' in cfg and cfg['n_heads'] is not None:
        model_kwargs['n_heads'] = cfg['n_heads']
    if 'smooth_mid' in cfg and cfg['smooth_mid'] is not None:
        model_kwargs['smooth_mid'] = cfg['smooth_mid']
    if cfg.get('linear_readout', False):
        model_kwargs['linear_readout'] = True
    if cfg.get('svd_mode', 'default') != 'default':
        model_kwargs['svd_mode'] = cfg['svd_mode']
    if 'match_params' in cfg:
        model_kwargs['match_params'] = cfg['match_params']

    model = PatchSVAE(**model_kwargs).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)

    prior_step = ckpt.get('step', 0)
    prior_images_seen = ckpt.get('total_images_seen', 0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params, V={model_kwargs['V']}, "
          f"D={model_kwargs['D']}, ps={model_kwargs['ps']}")
    print(f"  Resumed from step {prior_step}, total_images_seen={prior_images_seen:,}")
    return model, cfg, prior_step, prior_images_seen


# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

def load_log(log_path: str, hf_version: str) -> Dict[str, Any]:
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)
    return {'version': hf_version, 'entries': [], 'total_images_seen': 0}


def save_log(log_path: str, log: Dict[str, Any]):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT SAVE + PUSH
# ═══════════════════════════════════════════════════════════════════

def save_and_push(model, opt, sched, model_cfg, step, val_mse, log,
                  log_path, ckpt_dir, hf_repo, hf_version, is_best=False):
    """Save checkpoint locally and push to HuggingFace."""
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt = {
        'config': model_cfg,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'step': step,
        'val_mse': val_mse,
        'total_images_seen': log['total_images_seen'],
    }

    step_path = os.path.join(ckpt_dir, f'step_{step:08d}.pt')
    torch.save(ckpt, step_path)

    best_path = os.path.join(ckpt_dir, 'best.pt')
    if is_best:
        torch.save(ckpt, best_path)

    save_log(log_path, log)

    try:
        from huggingface_hub import HfApi
        api = HfApi()

        if is_best and os.path.exists(best_path):
            api.upload_file(
                path_or_fileobj=best_path,
                path_in_repo=f'{hf_version}/checkpoints/best.pt',
                repo_id=hf_repo, repo_type='model',
                commit_message=f'{hf_version} step {step} val_mse={val_mse:.6f} (best)')

        api.upload_file(
            path_or_fileobj=step_path,
            path_in_repo=f'{hf_version}/checkpoints/step_{step:08d}.pt',
            repo_id=hf_repo, repo_type='model',
            commit_message=f'{hf_version} step {step} val_mse={val_mse:.6f}')

        api.upload_file(
            path_or_fileobj=log_path,
            path_in_repo=f'{hf_version}/training_log.json',
            repo_id=hf_repo, repo_type='model',
            commit_message=f'{hf_version} log update step {step}')

        print(f"    ☁️  Pushed step {step}")

        # Clean up old step checkpoints locally (keep last 3 + best)
        ckpt_files = sorted([f for f in os.listdir(ckpt_dir)
                              if f.startswith('step_')])
        while len(ckpt_files) > 3:
            os.remove(os.path.join(ckpt_dir, ckpt_files.pop(0)))

    except Exception as e:
        print(f"    ⚠️  Push failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_streaming(
    hf_version: str,
    hf_repo: str = DEFAULT_HF_REPO,
    source: str = 'imagenet_256',
    target_size: int = DEFAULT_TARGET_SIZE,
    max_steps: int = 500_000,
    batch_size: int = 1024,
    lr: float = 3e-5,
    val_every: int = 2_500,
    save_every: int = 2_500,
    grad_clip: float = 1.0,
    target_cv: Optional[float] = None,
    cv_weight: float = 0.0,
    local_root: str = '/content',
    device: Optional[str] = None,
):
    """Streaming continuation of a trained SVAE model.

    Args:
        hf_version: HF repo subdirectory containing checkpoints/best.pt
        hf_repo: HF model repo ID
        source: 'imagenet_256' or 'laion'
        target_size: side length of crops fed to the model (matches arch ps × grid)
        max_steps: stop after this many training steps
        batch_size: per-step batch size
        lr: learning rate (3e-5 default — fine-tuning regime)
        val_every: run validation every N steps
        save_every: checkpoint + push every N steps (typically == val_every)
        grad_clip: max grad norm for ALL parameters (post-spectrum-lock regime)
        target_cv: if set, adds a soft CV-band penalty to the loss; otherwise
                   pure MSE only
        cv_weight: weight on (cv - target_cv)^2 penalty if target_cv is set
        local_root: where to write checkpoints + logs locally
        device: 'cuda' / 'cpu' / None (auto)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    local_dir = os.path.join(local_root, f'{hf_version}_streaming')
    ckpt_dir = os.path.join(local_dir, 'checkpoints')
    log_path = os.path.join(local_dir, 'training_log.json')

    print("\n" + "=" * 70)
    print(f"SVAE STREAMING — {hf_version}")
    print("=" * 70)

    # ── Load model from HF ──
    model, model_cfg, prior_step, prior_images_seen = load_model_from_hf(
        hf_repo, hf_version, device)
    model.train()

    # ── Optimizer ──
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)

    # ── Streaming dataloader ──
    if source not in STREAMING_SOURCES:
        raise ValueError(f"Unknown source: {source}. "
                         f"Choices: {list(STREAMING_SOURCES.keys())}")
    src = STREAMING_SOURCES[source]
    print(f"  Source: {src['description']} ({src['hf_path']}, split={src['split']})")

    from datasets import load_dataset
    if source == 'laion':
        # LAION needs streaming mode (too large for in-memory)
        train_hf = load_dataset(src['hf_path'], split=src['split'], streaming=True)
    else:
        train_hf = load_dataset(src['hf_path'], split=src['split'])

    stream_ds = StreamingCropDataset(train_hf, target_size=target_size)
    train_loader = torch.utils.data.DataLoader(
        stream_ds, batch_size=batch_size, num_workers=4,
        pin_memory=True, drop_last=True)

    # ── Log ──
    log = load_log(log_path, hf_version)
    if log['total_images_seen'] < prior_images_seen:
        log['total_images_seen'] = prior_images_seen
    best_mse = float('inf')

    # ── Initial val ──
    print(f"  Running initial validation on size {target_size}...")
    val_mse = run_val(model, target_size, device)
    best_mse = val_mse
    print(f"  Initial val_mse: {val_mse:.6f}")
    print(f"  Batch={batch_size}, lr={lr}, grad_clip={grad_clip}, "
          f"max_steps={max_steps:,}")
    if target_cv is not None:
        print(f"  CV penalty: target={target_cv}, weight={cv_weight}")
    print("=" * 70)

    # ── Helpers ──
    def per_layer_alphas():
        """Return (alpha_mean, alpha_std) averaged across cross-attn layers."""
        if len(model.cross_attn) == 0:
            return 0.0, 0.0
        alphas = [layer.alpha.detach() for layer in model.cross_attn]
        a_mean = torch.stack([a.mean() for a in alphas]).mean().item()
        a_std = torch.stack([a.std() for a in alphas]).mean().item()
        return a_mean, a_std

    # ── Training loop ──
    step = 0
    total_loss = 0.0
    count = 0
    grad_max_window = 0.0
    last_cv = target_cv if target_cv is not None else 0.0
    t0 = time.time()

    pbar = tqdm(train_loader, desc=f"Streaming {hf_version}",
                total=max_steps,
                bar_format='{l_bar}{bar:20}{r_bar}')

    for images in pbar:
        if step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        out = model(images)
        recon_loss = F.mse_loss(out['recon'], images)

        # Optional CV penalty (off by default — post-lock regime)
        if target_cv is not None and cv_weight > 0:
            with torch.no_grad():
                if step % 50 == 0:
                    current_cv = cv_of(out['svd']['M'][0, 0])
                    if current_cv > 0:
                        last_cv = current_cv
            loss = recon_loss + cv_weight * (last_cv - target_cv) ** 2
        else:
            loss = recon_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        total_grad = sum(
            p.grad.pow(2).sum().item()
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        grad_max_window = max(grad_max_window, total_grad)

        opt.step()
        sched.step()

        total_loss += recon_loss.item()
        count += 1
        step += 1
        log['total_images_seen'] += len(images)

        if step % 100 == 0:
            pbar.set_postfix_str(
                f"mse={recon_loss.item():.6f} avg={total_loss/count:.6f} "
                f"imgs={log['total_images_seen']:,}")
            pbar.n = step
            pbar.refresh()

        # ── Checkpoint + val ──
        if step % val_every == 0:
            elapsed = time.time() - t0
            avg_loss = total_loss / count

            val_mse = run_val(model, target_size, device)

            with torch.no_grad():
                S = out['svd']['S']
                S_orig = out['svd']['S_orig']
                S_mean = S.mean(dim=(0, 1))
                S_std = S.std(dim=(0, 1))
                ratio = (S_mean[0] / (S_mean[-1] + 1e-8)).item()
                erank = model.effective_rank(
                    S.reshape(-1, model.D)
                ).mean().item()
                s_delta = (S - S_orig).abs().mean().item()
                a_mean, a_std = per_layer_alphas()
                # Re-measure CV here regardless of penalty config
                cv_now = cv_of(out['svd']['M'][0, 0])
                if cv_now > 0:
                    last_cv = cv_now
                cv_in_band = 0.13 <= last_cv <= 0.30

            is_best = val_mse < best_mse
            if is_best:
                best_mse = val_mse

            print(f"\n  Step {step} | train={avg_loss:.6f} val={val_mse:.6f} "
                  f"{'★ BEST' if is_best else ''} | "
                  f"S=[{', '.join(f'{v:.3f}' for v in S_mean.tolist())}] "
                  f"er={erank:.2f} | {elapsed:.0f}s")
            print(f"    cv={last_cv:.3f} band={'Y' if cv_in_band else 'N'} "
                  f"S_d={s_delta:.5f} α_m={a_mean:.3f} α_s={a_std:.3f} "
                  f"g={grad_max_window:.1f} imgs={log['total_images_seen']:,}")

            log['entries'].append({
                'step': step,
                'train_mse': avg_loss,
                'val_mse': val_mse,
                'S_mean': S_mean.tolist(),
                'S_std': S_std.tolist(),
                'S_delta': s_delta,
                'ratio': ratio,
                'erank': erank,
                'row_cv': last_cv,
                'cv_in_band': cv_in_band,
                'alpha_mean': a_mean,
                'alpha_std': a_std,
                'grad_max': grad_max_window,
                'total_images_seen': log['total_images_seen'],
                'elapsed_seconds': elapsed,
                'lr': opt.param_groups[0]['lr'],
            })

            if step % save_every == 0:
                save_and_push(
                    model, opt, sched, model_cfg, step, val_mse, log,
                    log_path, ckpt_dir, hf_repo, hf_version, is_best=is_best
                )

            # Reset window stats
            total_loss = 0.0
            count = 0
            grad_max_window = 0.0
            t0 = time.time()

    print(f"\n{'=' * 70}")
    print(f"{hf_version} STREAMING COMPLETE")
    print(f"  Steps: {step}")
    print(f"  Best val MSE: {best_mse:.6f}")
    print(f"  Total images seen: {log['total_images_seen']:,}")
    print(f"{'=' * 70}")

    return model


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='SVAE Streaming Continuation Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--hf-version', type=str, required=True,
                        help='HF subdirectory of model to continue (e.g. v50_fresnel_64)')
    parser.add_argument('--hf-repo', type=str, default=DEFAULT_HF_REPO,
                        help=f'HF model repo (default {DEFAULT_HF_REPO})')
    parser.add_argument('--source', type=str, default='imagenet_256',
                        choices=list(STREAMING_SOURCES.keys()),
                        help='Streaming data source')
    parser.add_argument('--target-size', type=int, default=DEFAULT_TARGET_SIZE,
                        help='Crop side length (default 64)')
    parser.add_argument('--max-steps', type=int, default=500_000,
                        help='Max training steps (default 500K)')
    parser.add_argument('--batch', type=int, default=1024,
                        help='Batch size (default 1024)')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate (default 3e-5)')
    parser.add_argument('--val-every', type=int, default=2_500,
                        help='Run val every N steps (default 2500)')
    parser.add_argument('--save-every', type=int, default=2_500,
                        help='Save+push every N steps (default 2500)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Max grad norm for all params (default 1.0)')
    parser.add_argument('--target-cv', type=float, default=None,
                        help='Target CV for soft penalty (default off; '
                             'post-spectrum-lock training is pure MSE)')
    parser.add_argument('--cv-weight', type=float, default=0.0,
                        help='CV-penalty weight (default 0; only used if --target-cv is set)')
    parser.add_argument('--local-root', type=str, default='/content',
                        help='Local checkpoint+log root (default /content)')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda / cpu / None=auto')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    train_streaming(
        hf_version=args.hf_version,
        hf_repo=args.hf_repo,
        source=args.source,
        target_size=args.target_size,
        max_steps=args.max_steps,
        batch_size=args.batch,
        lr=args.lr,
        val_every=args.val_every,
        save_every=args.save_every,
        grad_clip=args.grad_clip,
        target_cv=args.target_cv,
        cv_weight=args.cv_weight,
        local_root=args.local_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()