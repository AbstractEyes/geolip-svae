"""
Universal SVAE Diagnostic Battery
===================================
One script. Any checkpoint. Every dataset.

Usage (CLI):
    python -m geolip_svae.diagnostic --hf v13_imagenet256
    python -m geolip_svae.diagnostic --hf-file v18_johanna_curriculum/checkpoints/epoch_0300.pt
    python -m geolip_svae.diagnostic --checkpoint /path/to/best.pt

Usage (Python):
    from geolip_svae.diagnostic import run
    results = run(hf_version='v13_imagenet256')

Tests:
    - 5 image datasets (CIFAR-10, MNIST, TinyImageNet, ImageNet-128, ImageNet-256)
    - 16 noise types with byte accuracy
    - 8 text byte-encoding sentences
    - Piecemeal tiling (4× resolution)
    - Signal energy survival + SNR
    - Alpha profile analysis
    - Compression metrics
    - Visual reconstruction grid
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from geolip_svae.inference import load_model, batched_forward

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Noise Generators ─────────────────────────────────────────────

NOISE_NAMES = {
    0: 'gaussian', 1: 'uniform', 2: 'uniform_scaled', 3: 'poisson',
    4: 'pink', 5: 'brown', 6: 'salt_pepper', 7: 'sparse',
    8: 'block', 9: 'gradient', 10: 'checkerboard', 11: 'mixed',
    12: 'structural', 13: 'cauchy', 14: 'exponential', 15: 'laplace',
}


def _pink(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(S / torch.sqrt(fx ** 2 + fy ** 2).clamp(min=1e-8), s=(h, ww))


def _brown(shape):
    w = torch.randn(shape)
    S = torch.fft.rfft2(w)
    h, ww = shape[-2], shape[-1]
    fy = torch.fft.fftfreq(h).unsqueeze(-1).expand(-1, ww // 2 + 1)
    fx = torch.fft.rfftfreq(ww).unsqueeze(0).expand(h, -1)
    return torch.fft.irfft2(S / (fx ** 2 + fy ** 2).clamp(min=1e-8), s=(h, ww))


def generate_noise(noise_type, n, s):
    """Generate n samples of a specific noise type at resolution s×s."""
    rng = np.random.RandomState(42)
    imgs = []
    for _ in range(n):
        if noise_type == 0:
            img = torch.randn(3, s, s)
        elif noise_type == 1:
            img = torch.rand(3, s, s) * 2 - 1
        elif noise_type == 2:
            img = (torch.rand(3, s, s) - 0.5) * 4
        elif noise_type == 3:
            lam = rng.uniform(0.5, 20.0)
            img = torch.poisson(torch.full((3, s, s), lam)) / lam - 1.0
        elif noise_type == 4:
            img = _pink((3, s, s))
            img = img / (img.std() + 1e-8)
        elif noise_type == 5:
            img = _brown((3, s, s))
            img = img / (img.std() + 1e-8)
        elif noise_type == 6:
            img = torch.where(torch.rand(3, s, s) > 0.5,
                              torch.ones(3, s, s) * 2, -torch.ones(3, s, s) * 2)
            img = img + torch.randn(3, s, s) * 0.1
        elif noise_type == 7:
            img = torch.randn(3, s, s) * (torch.rand(3, s, s) > 0.9).float() * 3
        elif noise_type == 8:
            b = rng.randint(2, 16)
            sm = torch.randn(3, s // b + 1, s // b + 1)
            img = F.interpolate(sm.unsqueeze(0), size=s, mode='nearest').squeeze(0)
        elif noise_type == 9:
            gy = torch.linspace(-2, 2, s).unsqueeze(1).expand(s, s)
            gx = torch.linspace(-2, 2, s).unsqueeze(0).expand(s, s)
            a = rng.uniform(0, 2 * math.pi)
            img = (math.cos(a) * gx + math.sin(a) * gy).unsqueeze(0).expand(3, -1, -1)
            img = img + torch.randn(3, s, s) * 0.5
        elif noise_type == 10:
            cs = rng.randint(2, 16)
            cy = torch.arange(s) // cs
            cx = torch.arange(s) // cs
            checker = ((cy.unsqueeze(1) + cx.unsqueeze(0)) % 2).float() * 2 - 1
            img = checker.unsqueeze(0).expand(3, -1, -1) + torch.randn(3, s, s) * 0.3
        elif noise_type == 11:
            alpha = rng.uniform(0.2, 0.8)
            img = alpha * torch.randn(3, s, s) + (1 - alpha) * (torch.rand(3, s, s) * 2 - 1)
        elif noise_type == 12:
            img = torch.zeros(3, s, s)
            h2 = s // 2
            img[:, :h2, :h2] = torch.randn(3, h2, h2)
            img[:, :h2, h2:] = torch.rand(3, h2, h2) * 2 - 1
            img[:, h2:, :h2] = _pink((3, h2, h2)) / 2
            img[:, h2:, h2:] = torch.where(torch.rand(3, h2, h2) > 0.5,
                                             torch.ones(3, h2, h2), -torch.ones(3, h2, h2))
        elif noise_type == 13:
            img = torch.tan(math.pi * (torch.rand(3, s, s) - 0.5)).clamp(-3, 3)
        elif noise_type == 14:
            img = torch.empty(3, s, s).exponential_(1.0) - 1.0
        elif noise_type == 15:
            u = torch.rand(3, s, s) - 0.5
            img = -torch.sign(u) * torch.log1p(-2 * u.abs())
        else:
            img = torch.randn(3, s, s)
        imgs.append(img.clamp(-4, 4))
    return torch.stack(imgs)


# ── Dataset Loaders ──────────────────────────────────────────────

IMAGE_DATASETS = ['cifar10', 'mnist', 'tiny_imagenet', 'imagenet128', 'imagenet256']


def load_dataset_batch(name, s, n=100):
    """Load n images from a dataset, resized to s×s, normalized."""
    from datasets import load_dataset as hf_load

    if name == 'cifar10':
        transform = T.Compose([T.Resize(s), T.ToTensor(),
                                T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        imgs = [ds[i][0] for i in range(min(n, len(ds)))]
        return torch.stack(imgs), (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), f'CIFAR-10→{s}'

    elif name == 'mnist':
        transform = T.Compose([T.Resize(s), T.Grayscale(3), T.ToTensor(),
                                T.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))])
        ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        imgs = [ds[i][0] for i in range(min(n, len(ds)))]
        return torch.stack(imgs), (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081), f'MNIST→{s}'

    elif name == 'tiny_imagenet':
        ds = hf_load('zh-plus/tiny-imagenet', split='valid', streaming=True)
        transform = T.Compose([T.Resize(s), T.CenterCrop(s), T.ToTensor(),
                                T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))])
        imgs = []
        for i, sample in enumerate(ds):
            imgs.append(transform(sample['image'].convert('RGB')))
            if i >= n - 1: break
        return torch.stack(imgs), (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821), f'TinyImageNet→{s}'

    elif name == 'imagenet128':
        ds = hf_load('benjamin-paine/imagenet-1k-128x128', split='validation', streaming=True)
        transform = T.Compose([T.Resize(s), T.CenterCrop(s), T.ToTensor(),
                                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        imgs = []
        for i, sample in enumerate(ds):
            imgs.append(transform(sample['image'].convert('RGB')))
            if i >= n - 1: break
        return torch.stack(imgs), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), f'ImageNet-128→{s}'

    elif name == 'imagenet256':
        ds = hf_load('benjamin-paine/imagenet-1k-256x256', split='validation', streaming=True)
        transform = T.Compose([T.Resize(s), T.CenterCrop(s), T.ToTensor(),
                                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        imgs = []
        for i, sample in enumerate(ds):
            imgs.append(transform(sample['image'].convert('RGB')))
            if i >= n - 1: break
        return torch.stack(imgs), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), f'ImageNet-256→{s}'

    raise ValueError(f"Unknown dataset: {name}")


# ── Test Functions ───────────────────────────────────────────────

def test_image_datasets(model, cfg, n=100):
    s, D = cfg['img_size'], cfg['D']
    bs = max(4, 64 // max(1, (s // 64) ** 2))
    print(f"\n{'=' * 80}\nIMAGE DATASET BATTERY ({s}×{s}, n={n})\n{'=' * 80}")
    print(f"  {'dataset':22s} {'MSE':>10s} {'std':>10s} {'min':>10s} {'max':>10s} | "
          f"{'S0':>6s} {'SD':>6s} {'ratio':>6s} {'erank':>6s}")
    print("-" * 100)

    results = {}
    for ds_name in IMAGE_DATASETS:
        try:
            imgs, mean, std, label = load_dataset_batch(ds_name, s, n)
            out = batched_forward(model, imgs, max_batch=bs)
            mse = F.mse_loss(out['recon'], imgs, reduction='none').mean(dim=(1, 2, 3))
            S_mean = out['S'].mean(dim=(0, 1))
            ratio = (S_mean[0] / (S_mean[-1] + 1e-8)).item()
            erank = model.effective_rank(out['S'].reshape(-1, D)).mean().item()
            results[ds_name] = {
                'label': label, 'mse_mean': mse.mean().item(),
                'mse_std': mse.std().item(), 'mse_min': mse.min().item(),
                'mse_max': mse.max().item(), 'S0': S_mean[0].item(),
                'SD': S_mean[-1].item(), 'ratio': ratio, 'erank': erank,
                'fidelity': (1 - mse.mean()).item() * 100,
            }
            print(f"  {label:22s} {mse.mean():10.6f} {mse.std():10.6f} "
                  f"{mse.min():10.6f} {mse.max():10.6f} | "
                  f"{S_mean[0]:6.3f} {S_mean[-1]:6.3f} {ratio:6.2f} {erank:6.2f}")
        except Exception as e:
            print(f"  {ds_name:22s} FAILED: {e}")
    return results


def test_noise_types(model, cfg, n=64):
    s, D = cfg['img_size'], cfg['D']
    bs = max(4, 64 // max(1, (s // 64) ** 2))
    print(f"\n{'=' * 80}\nNOISE TYPE BATTERY ({s}×{s}, n={n})\n{'=' * 80}")
    print(f"  {'type':18s} {'MSE':>10s} {'std':>10s} | "
          f"{'S0':>6s} {'SD':>6s} {'ratio':>6s} {'erank':>6s} | "
          f"{'byte_acc':>8s} {'±1_acc':>8s}")
    print("-" * 100)

    results = {}
    for t in range(16):
        name = NOISE_NAMES[t]
        imgs = generate_noise(t, n, s)
        out = batched_forward(model, imgs, max_batch=bs)
        mse = F.mse_loss(out['recon'], imgs, reduction='none').mean(dim=(1, 2, 3))
        S_mean = out['S'].mean(dim=(0, 1))
        ratio = (S_mean[0] / (S_mean[-1] + 1e-8)).item()
        erank = model.effective_rank(out['S'].reshape(-1, D)).mean().item()
        orig_q = ((imgs + 4) / 8 * 255).round().clamp(0, 255).long()
        recon_q = ((out['recon'] + 4) / 8 * 255).round().clamp(0, 255).long()
        byte_acc = (orig_q == recon_q).float().mean().item()
        byte_1 = ((orig_q - recon_q).abs() <= 1).float().mean().item()
        results[name] = {
            'mse_mean': mse.mean().item(), 'mse_std': mse.std().item(),
            'S0': S_mean[0].item(), 'SD': S_mean[-1].item(),
            'ratio': ratio, 'erank': erank,
            'byte_exact': byte_acc, 'byte_within1': byte_1,
        }
        print(f"  {name:18s} {mse.mean():10.6f} {mse.std():10.6f} | "
              f"{S_mean[0]:6.3f} {S_mean[-1]:6.3f} {ratio:6.2f} {erank:6.2f} | "
              f"{byte_acc * 100:7.2f}% {byte_1 * 100:7.2f}%")
    return results


def test_text_bytes(model, cfg):
    s = cfg['img_size']
    print(f"\n{'=' * 80}\nTEXT BYTE RECONSTRUCTION ({s}×{s})\n{'=' * 80}")
    texts = [
        "Hello, world! This is a test of the geometric encoder.",
        "The quick brown fox jumps over the lazy dog. 0123456789",
        "import torch; model = PatchSVAE(); output = model(x)",
        "E = mc² — Albert Einstein, theoretical physicist, 1905",
        "To be, or not to be, that is the question. — Shakespeare",
        "∫₀^∞ e^(-x²) dx = √π/2 — Gaussian integral",
        "01101000 01100101 01101100 01101100 01101111 — binary hello",
        "SELECT * FROM models WHERE cv BETWEEN 0.20 AND 0.23;",
    ]
    n_bytes = 3 * s * s
    results = {}
    device = next(model.parameters()).device
    model.eval()
    for text in texts:
        raw = text.encode('utf-8')
        actual_len = min(len(raw), n_bytes)
        padded = (raw + b'\x00' * n_bytes)[:n_bytes]
        arr = np.frombuffer(padded, dtype=np.uint8).copy()
        tensor = torch.from_numpy(arr).float()
        tensor = (tensor / 127.5) - 1.0
        tensor = tensor.reshape(1, 3, s, s).to(device)
        with torch.no_grad():
            out = model(tensor)
            mse = F.mse_loss(out['recon'], tensor).item()
        orig_b = ((tensor.squeeze(0).cpu().flatten() + 1.0) * 127.5).round().clamp(0, 255).byte()
        recon_b = ((out['recon'].squeeze(0).cpu().flatten() + 1.0) * 127.5).round().clamp(0, 255).byte()
        exact_acc = (orig_b[:actual_len] == recon_b[:actual_len]).float().mean().item()
        recovered = recon_b[:actual_len].numpy().tobytes().decode('utf-8', errors='replace')
        results[text[:40]] = {'mse': mse, 'byte_acc': exact_acc}
        print(f"\n  In:  '{text[:60]}'")
        print(f"  Out: '{recovered[:60]}'")
        print(f"  MSE: {mse:.6f}  Byte: {exact_acc * 100:.1f}%")
    return results


def test_signal_survival(model, cfg, n=32):
    s = cfg['img_size']
    bs = max(4, 32 // max(1, (s // 64) ** 2))
    print(f"\n{'=' * 80}\nSIGNAL ENERGY SURVIVAL\n{'=' * 80}")
    print(f"  {'source':22s} {'survival':>10s} {'SNR_dB':>10s} {'orig_E':>10s} {'recon_E':>10s}")
    print("-" * 70)
    results = {}
    for ds_name in IMAGE_DATASETS:
        try:
            imgs, _, _, label = load_dataset_batch(ds_name, s, n)
            out = batched_forward(model, imgs, max_batch=bs)
            orig_E = (imgs ** 2).mean().item()
            recon_E = (out['recon'] ** 2).mean().item()
            err_E = ((imgs - out['recon']) ** 2).mean().item()
            survival = recon_E / (orig_E + 1e-8) * 100
            snr = 10 * math.log10(orig_E / (err_E + 1e-8))
            results[ds_name] = {'survival': survival, 'snr': snr}
            print(f"  {label:22s} {survival:9.1f}% {snr:9.1f}dB {orig_E:10.4f} {recon_E:10.4f}")
        except:
            pass
    for t in [0, 4, 6, 13]:
        imgs = generate_noise(t, n, s)
        out = batched_forward(model, imgs, max_batch=bs)
        orig_E = (imgs ** 2).mean().item()
        recon_E = (out['recon'] ** 2).mean().item()
        err_E = ((imgs - out['recon']) ** 2).mean().item()
        survival = recon_E / (orig_E + 1e-8) * 100
        snr = 10 * math.log10(orig_E / (err_E + 1e-8))
        results[NOISE_NAMES[t]] = {'survival': survival, 'snr': snr}
        print(f"  noise/{NOISE_NAMES[t]:17s} {survival:9.1f}% {snr:9.1f}dB {orig_E:10.4f} {recon_E:10.4f}")
    return results


def test_alpha_profile(model):
    print(f"\n{'=' * 80}\nALPHA PROFILE\n{'=' * 80}")
    results = {}
    for li, layer in enumerate(model.cross_attn):
        alpha = layer.alpha.detach().cpu()
        results[f'layer_{li}'] = {
            'mean': alpha.mean().item(), 'max': alpha.max().item(),
            'min': alpha.min().item(), 'std': alpha.std().item(),
            'values': alpha.tolist(),
        }
        print(f"\n  Layer {li}: mean={alpha.mean():.5f}  max={alpha.max():.5f}  "
              f"min={alpha.min():.5f}  std={alpha.std():.6f}")
        bar_scale = 50 / (alpha.max().item() + 1e-8)
        for d in range(len(alpha)):
            bar = "█" * int(alpha[d].item() * bar_scale)
            print(f"    α[{d:2d}]: {alpha[d]:.5f}  {bar}")
    return results


def test_compression(model, cfg):
    s, D, ps = cfg['img_size'], cfg['D'], cfg['patch_size']
    n_patches = (s // ps) ** 2
    input_vals = 3 * s * s
    latent_vals = D * n_patches
    ratio = input_vals / latent_vals
    print(f"\n{'=' * 80}\nCOMPRESSION METRICS\n{'=' * 80}")
    print(f"  Input:   {s}×{s}×3 = {input_vals:,} values")
    print(f"  Latent:  {D}×{n_patches} = {latent_vals:,} omega tokens")
    print(f"  Ratio:   {ratio:.1f}:1")
    for bits in [8, 16, 32]:
        ib = input_vals * (bits // 8)
        lb = latent_vals * (bits // 8)
        print(f"  {bits}-bit: input={ib / 1024:.1f}KB  latent={lb / 1024:.1f}KB  ratio={ib / lb:.1f}:1")
    return {'input_values': input_vals, 'latent_values': latent_vals, 'ratio': ratio}


def test_piecemeal(model, cfg):
    s = cfg['img_size']
    print(f"\n{'=' * 80}\nPIECEMAL {s * 4}→{s} TILED RECONSTRUCTION\n{'=' * 80}")
    results = {}
    for t in [0, 1, 4, 6, 13]:
        large = generate_noise(t, 1, s * 4)
        B, C, H, W = large.shape
        tiles_h, tiles_w = H // s, W // s
        all_recon = []
        device = next(model.parameters()).device
        with torch.no_grad():
            for th in range(tiles_h):
                row = []
                for tw in range(tiles_w):
                    tile = large[:, :, th * s:(th + 1) * s, tw * s:(tw + 1) * s].to(device)
                    row.append(model(tile)['recon'].cpu())
                all_recon.append(torch.cat(row, dim=3))
        recon_full = torch.cat(all_recon, dim=2)
        mse = F.mse_loss(recon_full, large).item()
        results[NOISE_NAMES[t]] = mse
        n_tiles = tiles_h * tiles_w
        print(f"  {NOISE_NAMES[t]:18s}: {n_tiles} tiles, MSE={mse:.6f}")
    return results


def test_reconstruction_grid(model, cfg):
    s = cfg['img_size']
    print(f"\n{'=' * 80}\nRECONSTRUCTION GRID\n{'=' * 80}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows, labels = [], []
    for ds_name in IMAGE_DATASETS:
        try:
            imgs, mean, std, label = load_dataset_batch(ds_name, s, 2)
            mean_t = torch.tensor(mean).reshape(1, 3, 1, 1)
            std_t = torch.tensor(std).reshape(1, 3, 1, 1)
            out = batched_forward(model, imgs[:1], max_batch=1)
            orig_vis = (imgs[:1] * std_t + mean_t).clamp(0, 1)
            recon_vis = (out['recon'][:1] * std_t + mean_t).clamp(0, 1)
            rows.append((orig_vis[0], recon_vis[0]))
            labels.append(label)
        except:
            pass

    for t in [0, 6, 13, 4]:
        imgs = generate_noise(t, 1, s)
        out = batched_forward(model, imgs, max_batch=1)
        o = imgs[0].clamp(-3, 3)
        r = out['recon'][0].clamp(-3, 3)
        o = (o - o.min()) / (o.max() - o.min() + 1e-8)
        r = (r - r.min()) / (r.max() - r.min() + 1e-8)
        rows.append((o, r))
        labels.append(f'noise/{NOISE_NAMES[t]}')

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, (orig, recon) in enumerate(rows):
        diff = (orig - recon).abs().clamp(0, 1)
        axes[i, 0].imshow(orig.permute(1, 2, 0).numpy())
        axes[i, 1].imshow(recon.permute(1, 2, 0).numpy())
        axes[i, 2].imshow((diff * 5).clamp(0, 1).permute(1, 2, 0).numpy())
        axes[i, 0].set_ylabel(labels[i], fontsize=8)
        for j in range(3):
            axes[i, j].axis('off')
    axes[0, 0].set_title('Original', fontsize=9)
    axes[0, 1].set_title('Recon', fontsize=9)
    axes[0, 2].set_title('|Err|×5', fontsize=9)
    plt.tight_layout()
    fname = 'universal_diagnostic_grid.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


# ── Main Entry Point ─────────────────────────────────────────────

def run(hf_version=None, checkpoint_path=None, hf_file=None, n_samples=64):
    """Run the full diagnostic battery.

    Args:
        hf_version: named version (e.g. 'v13_imagenet256')
        checkpoint_path: local .pt path
        hf_file: specific HF file path
        n_samples: samples per test (scaled by resolution)

    Returns:
        dict: all test results, also saved as JSON
    """
    print("=" * 80)
    print("UNIVERSAL SVAE DIAGNOSTIC BATTERY")
    print("=" * 80)

    model, cfg = load_model(hf_version=hf_version, checkpoint_path=checkpoint_path,
                            hf_file=hf_file)

    print(f"  Epoch: {cfg.get('_epoch')}, MSE: {cfg.get('_test_mse', '?')}")
    print(f"  Config: {cfg}")

    # Infer img_size if not in config
    if 'img_size' not in cfg:
        ds = cfg.get('dataset', '')
        if '256' in ds: cfg['img_size'] = 256
        elif '128' in ds: cfg['img_size'] = 128
        elif 'tiny' in ds: cfg['img_size'] = 64
        else: cfg['img_size'] = 64

    s = cfg['img_size']
    n = min(n_samples, max(16, 100 // max(1, (s // 64) ** 2)))
    print(f"  Resolution: {s}×{s}, samples_per_test: {n}")

    results = {'config': cfg}
    results['image_datasets'] = test_image_datasets(model, cfg, n=n)
    results['noise_types'] = test_noise_types(model, cfg, n=n)
    results['text'] = test_text_bytes(model, cfg)
    results['piecemeal'] = test_piecemeal(model, cfg)
    results['signal_survival'] = test_signal_survival(model, cfg, n=n)
    results['alpha'] = test_alpha_profile(model)
    results['compression'] = test_compression(model, cfg)
    test_reconstruction_grid(model, cfg)

    tag = hf_version or (hf_file or 'local').replace('/', '_').replace('.pt', '')
    out_path = f'diagnostic_{tag}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results: {out_path}")
    print(f"\n{'=' * 80}\nDIAGNOSTIC COMPLETE\n{'=' * 80}")
    return results


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SVAE Universal Diagnostic Battery')
    parser.add_argument('--hf', type=str, help='HF version (e.g. v13_imagenet256)')
    parser.add_argument('--hf-file', type=str, help='Specific HF file path')
    parser.add_argument('--checkpoint', type=str, help='Local checkpoint path')
    parser.add_argument('--samples', type=int, default=64, help='Samples per test')
    args = parser.parse_args()

    if args.hf:
        run(hf_version=args.hf, n_samples=args.samples)
    elif args.hf_file:
        run(hf_file=args.hf_file, n_samples=args.samples)
    elif args.checkpoint:
        run(checkpoint_path=args.checkpoint, n_samples=args.samples)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python -m geolip_svae.diagnostic --hf v13_imagenet256")