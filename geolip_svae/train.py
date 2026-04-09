"""
SVAE Unified Trainer
=====================
Single entry point for all model variants:

    Fresnel  (images):  python -m geolip_svae.train --preset fresnel_base
    Johanna  (noise):   python -m geolip_svae.train --preset johanna_base
    Alexandria (text):  python -m geolip_svae.train --preset alexandria_small

Presets:
    fresnel_tiny      TinyImageNet 64×64,  300 ep
    fresnel_small     ImageNet-128 128×128, 50 ep
    fresnel_base      ImageNet-256 256×256, 20 ep
    johanna_tiny      Curriculum noise 64×64, 300 ep
    johanna_small     Omega noise 128×128, 200 ep (pretrained from Gaussian)
    johanna_base      Scheduled noise 256×256, 30 ep
    alexandria_small  Wikipedia text 128×128, 100 ep (pretrained from Johanna)
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from geolip_svae.model import PatchSVAE, cv_of

# ── HuggingFace auth ─────────────────────────────────────────────

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════
# PRESETS
# ═══════════════════════════════════════════════════════════════

PRESETS = {
    # ── Fresnel (images) ──
    'fresnel_tiny': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='tiny_imagenet', img_size=64, batch_size=256,
        lr=1e-4, epochs=300, target_cv=0.2915,
        hf_version='v19_fresnel_tiny', save_every=10,
    ),
    'fresnel_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='imagenet_128', img_size=128, batch_size=128,
        lr=1e-4, epochs=50, target_cv=0.2915,
        hf_version='v12_imagenet128', save_every=1,
    ),
    'fresnel_base': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='imagenet_256', img_size=256, batch_size=64,
        lr=1e-4, epochs=20, target_cv=0.2915,
        hf_version='v13_imagenet256', save_every=1,
    ),

    # ── Johanna (noise) ──
    'johanna_tiny': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='curriculum_noise', img_size=64, batch_size=512,
        lr=3e-4, epochs=300, target_cv=0.125,
        hf_version='v18_johanna_curriculum', save_every=25,
        curriculum='patience',  # patience-based tier promotion
    ),
    'johanna_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='omega_noise', img_size=128, batch_size=128,
        lr=1e-4, epochs=200, target_cv=0.125,
        hf_version='v16_johanna_omega', save_every=10,
        pretrained='v14_noise/checkpoints/epoch_0200.pt',
    ),
    'johanna_base': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='scheduled_noise', img_size=256, batch_size=64,
        lr=1e-4, epochs=30, target_cv=0.2915,
        hf_version='v20_johanna_base', save_every=5,
        curriculum='scheduled', tier_schedule={5: 1, 8: 2, 10: 3, 12: 4},
    ),

    # ── Alexandria (text) ──
    'alexandria_small': dict(
        V=256, D=16, patch_size=16, hidden=768, depth=4, n_cross=2,
        dataset='wikipedia', img_size=128, batch_size=128,
        lr=1e-4, epochs=100, target_cv=0.2915,
        hf_version='v22_alexandria_small', save_every=10,
        pretrained='v16_johanna_omega/checkpoints/best.pt',
        ds_size=200000, val_size=5000,
    ),
}


# ═══════════════════════════════════════════════════════════════
# NOISE DATASETS
# ═══════════════════════════════════════════════════════════════

NOISE_NAMES = {
    0: 'gaussian', 1: 'uniform', 2: 'uniform_scaled', 3: 'poisson',
    4: 'pink', 5: 'brown', 6: 'salt_pepper', 7: 'sparse',
    8: 'block', 9: 'gradient', 10: 'checkerboard', 11: 'mixed',
    12: 'structural', 13: 'cauchy', 14: 'exponential', 15: 'laplace',
}

TIERS = {
    0: [0],              # Gaussian
    1: [4, 5, 8, 9],    # Pink, Brown, Block, Gradient
    2: [1, 2, 10, 11],  # Uniform, Scaled, Checkerboard, Mixed
    3: [3, 14, 15, 7],  # Poisson, Exponential, Laplace, Sparse
    4: [13, 6, 12],     # Cauchy, Salt-pepper, Structural
}


def _pink_noise(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(S / torch.sqrt(fx ** 2 + fy ** 2).clamp(min=1e-8), s=(h, ww))


def _brown_noise(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(S / (fx ** 2 + fy ** 2).clamp(min=1e-8), s=(h, ww))


def _generate_noise(noise_type, s, rng):
    if noise_type == 0: return torch.randn(3, s, s)
    elif noise_type == 1: return torch.rand(3, s, s) * 2 - 1
    elif noise_type == 2: return (torch.rand(3, s, s) - 0.5) * 4
    elif noise_type == 3:
        lam = rng.uniform(0.5, 20.0)
        return torch.poisson(torch.full((3, s, s), lam)) / lam - 1.0
    elif noise_type == 4:
        img = _pink_noise((3, s, s)); return img / (img.std() + 1e-8)
    elif noise_type == 5:
        img = _brown_noise((3, s, s)); return img / (img.std() + 1e-8)
    elif noise_type == 6:
        img = torch.where(torch.rand(3, s, s) > 0.5,
                          torch.ones(3, s, s) * 2, -torch.ones(3, s, s) * 2)
        return img + torch.randn(3, s, s) * 0.1
    elif noise_type == 7:
        return torch.randn(3, s, s) * (torch.rand(3, s, s) > 0.9).float() * 3
    elif noise_type == 8:
        b = rng.randint(2, 16)
        sm = torch.randn(3, s // b + 1, s // b + 1)
        return F.interpolate(sm.unsqueeze(0), size=s, mode='nearest').squeeze(0)
    elif noise_type == 9:
        gy = torch.linspace(-2, 2, s).unsqueeze(1).expand(s, s)
        gx = torch.linspace(-2, 2, s).unsqueeze(0).expand(s, s)
        a = rng.uniform(0, 2 * math.pi)
        return (math.cos(a) * gx + math.sin(a) * gy).unsqueeze(0).expand(3, -1, -1) + torch.randn(3, s, s) * 0.5
    elif noise_type == 10:
        cs = rng.randint(2, 16)
        cy = torch.arange(s) // cs; cx = torch.arange(s) // cs
        return ((cy.unsqueeze(1) + cx.unsqueeze(0)) % 2).float().unsqueeze(0).expand(3, -1, -1) * 2 - 1 + torch.randn(3, s, s) * 0.3
    elif noise_type == 11:
        alpha = rng.uniform(0.2, 0.8)
        return alpha * torch.randn(3, s, s) + (1 - alpha) * (torch.rand(3, s, s) * 2 - 1)
    elif noise_type == 12:
        img = torch.zeros(3, s, s); h2 = s // 2
        img[:, :h2, :h2] = torch.randn(3, h2, h2)
        img[:, :h2, h2:] = torch.rand(3, h2, h2) * 2 - 1
        img[:, h2:, :h2] = _pink_noise((3, h2, h2)) / 2
        img[:, h2:, h2:] = torch.where(torch.rand(3, h2, h2) > 0.5,
                                         torch.ones(3, h2, h2), -torch.ones(3, h2, h2))
        return img
    elif noise_type == 13:
        return torch.tan(math.pi * (torch.rand(3, s, s) - 0.5)).clamp(-3, 3)
    elif noise_type == 14:
        return torch.empty(3, s, s).exponential_(1.0) - 1.0
    elif noise_type == 15:
        u = torch.rand(3, s, s) - 0.5
        return -torch.sign(u) * torch.log1p(-2 * u.abs())
    return torch.randn(3, s, s)


class CurriculumNoiseDataset(torch.utils.data.Dataset):
    """Noise with tier-based type activation for Johanna curriculum training."""

    def __init__(self, size=500000, img_size=64):
        self.size = size
        self.img_size = img_size
        self._rng = np.random.RandomState(42)
        self._call_count = 0
        self.active_types = list(TIERS[0])
        self.current_tier = 0

    def unlock_tier(self, tier):
        if tier in TIERS:
            for t in TIERS[tier]:
                if t not in self.active_types:
                    self.active_types.append(t)
            self.current_tier = tier

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self._call_count += 1
        if self._call_count % 1000 == 0:
            self._rng = np.random.RandomState(int.from_bytes(os.urandom(4), 'big'))
            torch.manual_seed(int.from_bytes(os.urandom(4), 'big'))
        noise_type = self.active_types[idx % len(self.active_types)]
        img = _generate_noise(noise_type, self.img_size, self._rng).clamp(-4, 4)
        return img.float(), noise_type


class OmegaNoiseDataset(torch.utils.data.Dataset):
    """All 16 noise types for Johanna omega training."""

    def __init__(self, size=1280000, img_size=128):
        self.size = size
        self.img_size = img_size
        self._rng = np.random.RandomState(42)
        self._call_count = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self._call_count += 1
        if self._call_count % 1000 == 0:
            self._rng = np.random.RandomState(int.from_bytes(os.urandom(4), 'big'))
            torch.manual_seed(int.from_bytes(os.urandom(4), 'big'))
        noise_type = idx % 16
        img = _generate_noise(noise_type, self.img_size, self._rng).clamp(-4, 4)
        return img.float(), noise_type


# ═══════════════════════════════════════════════════════════════
# IMAGE DATASETS
# ═══════════════════════════════════════════════════════════════

class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img), item.get('label', 0)


def get_image_loaders(dataset_name, img_size, batch_size):
    """Returns (train_loader, test_loader, mean, std)."""
    from datasets import load_dataset

    if dataset_name == 'tiny_imagenet':
        ds = load_dataset('zh-plus/tiny-imagenet')
        mean, std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds = HFImageDataset(ds['train'], transform)
        val_ds = HFImageDataset(ds['valid'], transform)

    elif dataset_name == 'imagenet_128':
        ds = load_dataset('benjamin-paine/imagenet-1k-128x128')
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds = HFImageDataset(ds['train'], transform)
        val_ds = HFImageDataset(ds['validation'], transform)

    elif dataset_name == 'imagenet_256':
        ds = load_dataset('benjamin-paine/imagenet-1k-256x256')
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds = HFImageDataset(ds['train'], transform)
        val_ds = HFImageDataset(ds['validation'], transform)

    else:
        raise ValueError(f"Unknown image dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, test_loader, mean, std


# ═══════════════════════════════════════════════════════════════
# TEXT DATASET
# ═══════════════════════════════════════════════════════════════

class WikiTextAsImage(torch.utils.data.Dataset):
    """Wikipedia text packed as (3, H, W) byte tensors. [0,255] → [-1,1]."""

    def __init__(self, size=200000, img_size=128, split='train'):
        self.size = size
        self.img_size = img_size
        self.n_bytes = 3 * img_size * img_size
        from datasets import load_dataset
        ds = load_dataset('wikimedia/wikipedia', '20231101.en',
                          split=split, streaming=True)
        target_bytes = min(size * self.n_bytes, 500_000_000)
        chunks, total = [], 0
        for article in ds:
            text = article['text']
            if text.strip():
                chunks.append(text)
                total += len(text)
            if total >= target_bytes:
                break
        self.raw_bytes = '\n'.join(chunks).encode('utf-8')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        max_start = max(0, len(self.raw_bytes) - self.n_bytes)
        start = torch.randint(0, max_start + 1, (1,)).item()
        chunk = self.raw_bytes[start:start + self.n_bytes]
        if len(chunk) < self.n_bytes:
            chunk = chunk + b'\x00' * (self.n_bytes - len(chunk))
        arr = np.frombuffer(chunk, dtype=np.uint8).copy()
        tensor = torch.from_numpy(arr).float()
        tensor = (tensor / 127.5) - 1.0
        return tensor.reshape(3, self.img_size, self.img_size), 0


# ═══════════════════════════════════════════════════════════════
# PER-TYPE EVALUATION
# ═══════════════════════════════════════════════════════════════

def eval_per_type(model, active_types, img_size, device, n_per_type=64):
    """MSE for each active noise type."""
    rng = np.random.RandomState(99)
    model.eval()
    results = {}
    with torch.no_grad():
        for t in active_types:
            imgs = torch.stack([
                _generate_noise(t, img_size, rng).clamp(-4, 4)
                for _ in range(n_per_type)
            ]).to(device)
            out = model(imgs)
            results[t] = F.mse_loss(out['recon'], imgs).item()
    return results


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(cfg):
    """Main training loop. cfg is a preset dict or custom config."""

    # ── Unpack config ──
    V           = cfg['V']
    D           = cfg['D']
    patch_size  = cfg['patch_size']
    hidden      = cfg['hidden']
    depth       = cfg['depth']
    n_cross     = cfg['n_cross']
    dataset     = cfg['dataset']
    img_size    = cfg['img_size']
    batch_size  = cfg['batch_size']
    lr          = cfg['lr']
    epochs      = cfg['epochs']
    target_cv   = cfg['target_cv']
    hf_version  = cfg['hf_version']
    save_every  = cfg.get('save_every', 10)

    cv_weight   = cfg.get('cv_weight', 0.3)
    boost       = cfg.get('boost', 0.5)
    sigma       = cfg.get('sigma', 0.15)

    pretrained  = cfg.get('pretrained', None)
    curriculum  = cfg.get('curriculum', None)
    tier_schedule = cfg.get('tier_schedule', None)

    save_dir    = cfg.get('save_dir', '/content/checkpoints')
    hf_repo     = cfg.get('hf_repo', 'AbstractPhil/geolip-SVAE')
    tb_dir      = cfg.get('tb_dir', '/content/runs')

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── TensorBoard ──
    from torch.utils.tensorboard import SummaryWriter
    run_name = f"{hf_version}_{img_size}x{img_size}_h{hidden}_d{depth}_lr{lr}"
    tb_path = os.path.join(tb_dir, run_name)
    writer = SummaryWriter(tb_path)
    print(f"  TensorBoard: {tb_path}")

    # ── HuggingFace ──
    hf_enabled = False
    api = None
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        hf_enabled = True
        hf_prefix = f"{hf_version}/checkpoints"
        print(f"  HuggingFace: {hf_repo}/{hf_prefix}")
    except Exception as e:
        print(f"  HuggingFace: disabled ({e})")

    def upload_to_hf(local_path, remote_name):
        if not hf_enabled:
            return
        try:
            api.upload_file(path_or_fileobj=local_path,
                            path_in_repo=f"{hf_prefix}/{remote_name}",
                            repo_id=hf_repo, repo_type="model")
            print(f"  ☁️  Uploaded: {hf_repo}/{hf_prefix}/{remote_name}")
        except Exception as e:
            print(f"  ⚠️  HF upload: {e}")

    # ── Model ──
    model = PatchSVAE(V=V, D=D, ps=patch_size, hidden=hidden,
                      depth=depth, n_cross=n_cross).to(device)

    # ── Pretrained weights ──
    if pretrained:
        from huggingface_hub import hf_hub_download
        print(f"\n  Loading pretrained: {pretrained}")
        try:
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=pretrained,
                                         repo_type='model')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            print(f"  Loaded ep{ckpt['epoch']}, MSE={ckpt['test_mse']:.6f}")
        except Exception as e:
            print(f"  ⚠️  Pretrained load failed: {e} — training from scratch")

    total_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ── Data ──
    mean_t, std_t = None, None

    if dataset in ('tiny_imagenet', 'imagenet_128', 'imagenet_256'):
        train_loader, test_loader, mean, std = get_image_loaders(
            dataset, img_size, batch_size)
        mean_t = torch.tensor(mean).reshape(1, 3, 1, 1).to(device)
        std_t = torch.tensor(std).reshape(1, 3, 1, 1).to(device)
        is_noise = False
        is_text = False

    elif dataset == 'curriculum_noise':
        train_ds = CurriculumNoiseDataset(size=500000, img_size=img_size)
        val_ds = CurriculumNoiseDataset(size=10000, img_size=img_size)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        is_noise = True
        is_text = False

    elif dataset in ('omega_noise', 'scheduled_noise'):
        ds_size = cfg.get('ds_size', 1280000)
        val_size = cfg.get('val_size', 10000)
        train_ds = OmegaNoiseDataset(size=ds_size, img_size=img_size)
        val_ds = OmegaNoiseDataset(size=val_size, img_size=img_size)
        if dataset == 'scheduled_noise':
            # Scheduled uses curriculum dataset with fixed unlocks
            train_ds = CurriculumNoiseDataset(size=ds_size or 128000, img_size=img_size)
            val_ds = CurriculumNoiseDataset(size=val_size or 2000, img_size=img_size)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        is_noise = True
        is_text = False

    elif dataset == 'wikipedia':
        ds_size = cfg.get('ds_size', 200000)
        val_size = cfg.get('val_size', 5000)
        print(f"\n  Loading Wikipedia corpus...")
        train_ds = WikiTextAsImage(size=ds_size, img_size=img_size, split='train')
        val_ds = WikiTextAsImage(size=val_size, img_size=img_size, split='train')
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        is_noise = False
        is_text = True

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ── Print config ──
    n_patches = (img_size // patch_size) ** 2
    print(f"\n  SVAE TRAINER — {hf_version}")
    print(f"  {img_size}×{img_size}, {n_patches} patches, ({V},{D}), {total_params:,} params")
    print(f"  Dataset: {dataset}, batch={batch_size}, lr={lr}, epochs={epochs}")
    print(f"  Target CV: {target_cv}, soft hand: boost={1+boost:.1f}x, penalty={cv_weight}")
    if curriculum:
        print(f"  Curriculum: {curriculum}")
    if tier_schedule:
        print(f"  Tier schedule: {tier_schedule}")
    print("=" * 100)

    # ── Checkpoint helper ──
    best_recon = float('inf')

    def save_checkpoint(path, epoch, test_mse, extra=None, upload=True):
        ckpt_out = {
            'epoch': epoch, 'test_mse': test_mse,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'config': {
                'V': V, 'D': D, 'patch_size': patch_size,
                'hidden': hidden, 'depth': depth, 'n_cross_layers': n_cross,
                'target_cv': target_cv, 'dataset': dataset,
                'img_size': img_size, 'lr': lr,
            },
        }
        if extra:
            ckpt_out.update(extra)
        torch.save(ckpt_out, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  💾 Saved: {path} ({size_mb:.1f}MB, ep{epoch}, MSE={test_mse:.6f})")
        if upload:
            upload_to_hf(path, os.path.basename(path))

    # ── Patience promotion state (for curriculum) ──
    tier_best_mse = float('inf')
    stale_epochs = 0

    # ── Training ──
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_recon, n = 0, 0, 0
        last_cv, last_prox = target_cv, 1.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}",
                    bar_format='{l_bar}{bar:20}{r_bar}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            opt.zero_grad()
            out = model(images)
            recon_loss = F.mse_loss(out['recon'], images)

            with torch.no_grad():
                if batch_idx % 50 == 0:
                    current_cv = cv_of(out['svd']['M'][0, 0])
                    if current_cv > 0:
                        last_cv = current_cv
                    delta = last_cv - target_cv
                    last_prox = math.exp(-delta ** 2 / (2 * sigma ** 2))

            recon_w = 1.0 + boost * last_prox
            cv_pen = cv_weight * (1.0 - last_prox)
            loss = recon_w * recon_loss + cv_pen * (last_cv - target_cv) ** 2
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.cross_attn.parameters(), max_norm=0.5)
            opt.step()

            total_loss += loss.item() * len(images)
            total_recon += recon_loss.item() * len(images)
            n += len(images)
            pbar.set_postfix_str(f"mse={recon_loss.item():.4f} cv={last_cv:.3f}")

        pbar.close()
        sched.step()
        epoch_time = time.time() - t0

        # ── Eval ──
        model.eval()
        test_mse_total, test_n = 0, 0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                test_mse_total += F.mse_loss(out['recon'], imgs).item() * len(imgs)
                test_n += len(imgs)
        test_mse = test_mse_total / test_n

        # Geometry snapshot
        with torch.no_grad():
            sample, _ = next(iter(test_loader))
            sample = sample[:min(64, len(sample))].to(device)
            out = model(sample)
            S_mean = out['svd']['S'].mean(dim=(0, 1))
            S_orig = out['svd']['S_orig'].mean(dim=(0, 1))
            ratio = (S_mean[0] / (S_mean[-1] + 1e-8)).item()
            erank = model.effective_rank(out['svd']['S'].reshape(-1, D)).mean().item()
            s_delta = (S_mean - S_orig).abs().mean().item()

        # Per-type MSE for noise variants
        type_str = ""
        if is_noise:
            active = list(range(16))
            if hasattr(train_loader.dataset, 'active_types'):
                active = train_loader.dataset.active_types
            type_mse = eval_per_type(model, active, img_size, device, n_per_type=32)
            type_str = " ".join(f"{NOISE_NAMES[t][:4]}={v:.3f}" for t, v in sorted(type_mse.items()))

        # Byte accuracy for text
        byte_str = ""
        if is_text:
            with torch.no_grad():
                sample_imgs, _ = next(iter(test_loader))
                sample_imgs = sample_imgs[:32].to(device)
                sample_out = model(sample_imgs)
                orig_b = ((sample_imgs.cpu().flatten(1) + 1.0) * 127.5).round().clamp(0, 255).long()
                recon_b = ((sample_out['recon'].cpu().flatten(1) + 1.0) * 127.5).round().clamp(0, 255).long()
                byte_acc = (orig_b == recon_b).float().mean().item()
            byte_str = f"bytes={byte_acc * 100:.1f}%"

        print(f" {epoch:3d} | {total_loss/n:.4f} {total_recon/n:.4f} {epoch_time:.0f}s | "
              f"test={test_mse:.6f} | S0={S_mean[0]:.3f} SD={S_mean[-1]:.3f} "
              f"r={ratio:.2f} er={erank:.2f} | cv={last_cv:.3f} Sd={s_delta:.5f} "
              f"{byte_str} {type_str}")

        # TB
        writer.add_scalar('train/recon', total_recon / n, epoch)
        writer.add_scalar('test/mse', test_mse, epoch)
        writer.add_scalar('geo/cv', last_cv, epoch)
        writer.add_scalar('geo/S0', S_mean[0].item(), epoch)
        writer.add_scalar('geo/ratio', ratio, epoch)
        writer.add_scalar('geo/erank', erank, epoch)
        writer.add_scalar('geo/s_delta', s_delta, epoch)

        # ── Curriculum: scheduled tier unlocks ──
        if curriculum == 'scheduled' and tier_schedule and epoch in tier_schedule:
            next_tier = tier_schedule[epoch]
            train_loader.dataset.unlock_tier(next_tier)
            test_loader.dataset.unlock_tier(next_tier)
            new_names = [NOISE_NAMES[t] for t in TIERS[next_tier]]
            print(f"\n  ★ TIER {next_tier} UNLOCKED (epoch {epoch}): +{', '.join(new_names)}")
            print(f"    Active: {[NOISE_NAMES[t] for t in train_loader.dataset.active_types]}\n")
            save_checkpoint(os.path.join(save_dir, f'tier{next_tier}_start.pt'),
                            epoch, test_mse, upload=True)

        # ── Curriculum: patience-based promotion ──
        if curriculum == 'patience' and hasattr(train_loader.dataset, 'unlock_tier'):
            improvement = (tier_best_mse - test_mse) / (tier_best_mse + 1e-8)
            if test_mse < tier_best_mse:
                tier_best_mse = test_mse
            if improvement < 0.01:
                stale_epochs += 1
            else:
                stale_epochs = 0
            if stale_epochs >= 10 and train_loader.dataset.current_tier < max(TIERS.keys()):
                next_tier = train_loader.dataset.current_tier + 1
                train_loader.dataset.unlock_tier(next_tier)
                test_loader.dataset.unlock_tier(next_tier)
                new_names = [NOISE_NAMES[t] for t in TIERS[next_tier]]
                print(f"\n  ★ PROMOTED TO TIER {next_tier}: +{', '.join(new_names)}")
                print(f"    Active: {[NOISE_NAMES[t] for t in train_loader.dataset.active_types]}\n")
                tier_best_mse = test_mse
                stale_epochs = 0
                save_checkpoint(os.path.join(save_dir, f'tier{next_tier}_start.pt'),
                                epoch, test_mse, upload=True)

        # ── Checkpoints ──
        if test_mse < best_recon:
            best_recon = test_mse
            save_checkpoint(os.path.join(save_dir, 'best.pt'),
                            epoch, test_mse, upload=False)

        if epoch % save_every == 0:
            save_checkpoint(os.path.join(save_dir, f'epoch_{epoch:04d}.pt'),
                            epoch, test_mse)
            best_path = os.path.join(save_dir, 'best.pt')
            if os.path.exists(best_path):
                upload_to_hf(best_path, 'best.pt')
            writer.flush()
            if hf_enabled:
                try:
                    api.upload_folder(folder_path=tb_path,
                                      path_in_repo=f"{hf_version}/tensorboard/{run_name}",
                                      repo_id=hf_repo, repo_type="model")
                    print(f"  ☁️  TB synced")
                except:
                    pass

    writer.close()
    print(f"\n  TRAINING COMPLETE — {hf_version}")
    print(f"  Best MSE: {best_recon:.6f}")
    print(f"  Checkpoints: {save_dir}/")
    return best_recon


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SVAE Unified Trainer')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Named preset configuration')
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets')
    args = parser.parse_args()

    if args.list_presets:
        for name, cfg in PRESETS.items():
            ds = cfg['dataset']
            sz = cfg['img_size']
            ep = cfg['epochs']
            pre = cfg.get('pretrained', 'scratch')
            print(f"  {name:<22s} {ds:<20s} {sz}×{sz}  {ep:>3d} ep  from={pre}")
        exit()

    if not args.preset:
        parser.print_help()
        print("\nPresets:")
        for name in PRESETS:
            print(f"  {name}")
        exit()

    torch.set_float32_matmul_precision('high')
    train(PRESETS[args.preset])