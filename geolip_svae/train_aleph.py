"""train_aleph.py — train AlephModel (geolip-aleph-void) on byte-trigram / tiny-imagenet.

Reuses the EXACT dataset path that produced the SVAE batteries (get_dataset_bundle),
so codebooks and MSE are directly comparable (baseline byte-trigram battery
MSE ~3.8e-7; tiny-imagenet ~8e-5). Builds AlephModel via build_aleph and saves
load_model-compatible checkpoints (save_aleph_checkpoint, model_type='aleph'),
so every run yields a loadable, hostable model.

Pure recon objective (the guarantee): MSE for byte-trigram, cosine for
tiny-imagenet (direction — what the spherical latent encodes), 'cosine_mse' if
amplitude matters. No CV penalty — AlephModel's structure is architectural; CV
is LOGGED read-only (column-norm spread) to watch self-organization.

THE QUESTION this line answered: can a single tied linear map from the spherical
matrix M carry reconstruction? Yes — M is recon-real, and its extracted codebook
addresses MORE sharply than the faux-embedding SVAE batteries.

Requires geolip_svae installed (dataset bundle + aleph_model + byte metrics).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

# byte_trigram_proto_64 preset — the config the baseline battery used.
BASE_CFG = dict(
    V=32, D=4, patch_size=4, hidden=64, depth=1, channels=3,
    dataset="byte_trigram", img_size=64, bt_corpus="wikitext-103-raw-v1",
    batch_size=256, lr=1e-3, epochs=20,
    ds_size=1_000_000, val_size=10_000,
)

# tiny-imagenet (zh-plus/tiny-imagenet, normalized 64x64x3). Baseline battery
# h2_linear_tiny_imagenet_64 reached MSE ~8e-5. The HARD discriminator: does M
# stay recon-real when the substrate needs the baseline's decoder accumulator?
TINY_IMAGENET_CFG = dict(
    V=32, D=4, patch_size=4, hidden=64, depth=1, channels=3,
    dataset="tiny_imagenet", img_size=64,
    batch_size=256, lr=1e-3, epochs=20,
)

PRESETS = {"byte_trigram": BASE_CFG, "tiny_imagenet": TINY_IMAGENET_CFG}
BASELINE_MSE = {"byte_trigram": 3.8e-7, "tiny_imagenet": 8.0e-5}


def _recon_loss(recon: torch.Tensor, images: torch.Tensor, mode: str) -> torch.Tensor:
    """mode: 'mse' | 'cosine' | 'cosine_mse'. Cosine is per-image (flatten
    C*H*W) — scores direction, the quantity a spherical latent encodes. On
    zero-centered (normalized) images it isn't swamped by a DC offset. Pure
    cosine is scale-blind, so 'cosine_mse' adds MSE back if amplitude matters."""
    if mode == "mse":
        return F.mse_loss(recon, images)
    cos = F.cosine_similarity(recon.flatten(1), images.flatten(1), dim=1)  # (B,)
    cl = (1.0 - cos).mean()
    if mode == "cosine":
        return cl
    if mode == "cosine_mse":
        return cl + F.mse_loss(recon, images)
    raise ValueError(f"unknown loss mode {mode!r}")


def _col_norm_cv(M: torch.Tensor) -> float:
    """CV of the omega-token magnitudes S = ||M||_col — cheap structural probe.
    Low, stable CV => the spherical matrix self-organized a consistent spectrum."""
    S = M.norm(dim=-2)                 # (B, N, D)
    s = S.reshape(-1)
    return float((s.std() / (s.mean() + 1e-8)).item())


def _address_stats(aleph_logits: torch.Tensor, tau: float):
    """From aleph_logits (B,N,V,2K): codebook PERPLEXITY (effective # oriented
    axes used — collapse detector, low = collapsed toward few axes) and mean
    top-1 address probability (how decisively each row addresses). Returns
    (perplexity, mean_margin, usage_vec)."""
    a = torch.softmax(aleph_logits / tau, dim=-1)        # (B,N,V,2K)
    usage = a.reshape(-1, a.shape[-1]).mean(0)           # (2K,) batch-mean usage
    ent = -(usage * usage.clamp_min(1e-12).log()).sum()
    ppl = float(ent.exp().item())                        # effective axes used
    margin = float(a.max(dim=-1).values.mean().item())   # mean top-1 prob
    return ppl, margin, usage


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20):
    model.eval()
    tot_mse, tot_cos, tot_cv, tot_ppl, tot_amg, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    has_addr = getattr(model, "address", "none") != "none"
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        out = model(images)
        tot_mse += F.mse_loss(out["recon"], images).item()
        tot_cos += F.cosine_similarity(out["recon"].flatten(1),
                                       images.flatten(1), dim=1).mean().item()
        tot_cv += _col_norm_cv(out["svd"]["M"])
        if has_addr and "aleph_logits" in out["svd"]:
            ppl, amg, _ = _address_stats(out["svd"]["aleph_logits"], model.address_tau)
            tot_ppl += ppl; tot_amg += amg
        n += 1
    model.train()
    n = max(n, 1)
    return (tot_mse / n, tot_cos / n, tot_cv / n,
            tot_ppl / n if has_addr else float("nan"),
            tot_amg / n if has_addr else float("nan"))


def train_aleph(decode_mode: str = "tied", *, dataset: str = "byte_trigram",
                address: str = "soft", K: int = 64, address_tau: float = 0.1,
                loss_mode: str | None = None, quick: bool = False,
                device: str = "cuda", cfg_overrides: dict | None = None,
                save_path: str | None = None, report_every: int = 100):
    """Train one AlephModel (geolip-aleph-void). address='soft'|'hard' uses the
    aleph-address bottleneck (learned codebook of K projective axes); 'none'
    trains the recon-real tied autoencoder (the gate). loss_mode defaults to
    'cosine' for tiny_imagenet, 'mse' for byte_trigram. Set cfg_overrides=
    dict(div_weight=0.01) to enable the anti-collapse usage-entropy term if
    codebook perplexity drops. Logs codebook perplexity (effective oriented axes
    used — collapse detector) and mean address margin. Checkpoints saved in the
    load_model-compatible format. Returns (model, history) with rows
    (step, train_loss, test_mse, test_cos, test_cv, perplexity, address_margin)."""
    from geolip_svae.dataset_presets import get_dataset_bundle
    from geolip_svae.aleph_model import build_aleph, save_aleph_checkpoint

    cfg = dict(PRESETS[dataset])
    if loss_mode is None:
        loss_mode = "cosine" if dataset == "tiny_imagenet" else "mse"
    if quick:                              # fast descent check, not a real run
        cfg.update(epochs=3)
        if dataset == "byte_trigram":
            cfg.update(ds_size=50_000, val_size=2_000)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    bundle = get_dataset_bundle(cfg, channels=cfg["channels"])
    train_loader, test_loader = bundle.train_loader, bundle.test_loader

    model = build_aleph({
        "patch_size": cfg["patch_size"], "channels": cfg["channels"],
        "V": cfg["V"], "D": cfg["D"], "hidden": cfg["hidden"],
        "depth": cfg.get("depth", 1), "decode_mode": decode_mode,
        "address": cfg.get("address", address), "K": cfg.get("K", K),
        "address_tau": cfg.get("address_tau", address_tau),
        "n_atoms": cfg.get("n_atoms", 64), "code_tau": cfg.get("code_tau", 1.0),
    }).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = max(1, cfg.get("ds_size", 50_000) // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    base = BASELINE_MSE.get(dataset, float("nan"))
    print(f"AlephModel[{decode_mode}/addr={model.address}] {dataset} loss={loss_mode} "
          f"params={model.num_params()} | {cfg['img_size']}x{cfg['img_size']} | "
          f"{cfg['epochs']}ep x ~{steps_per_epoch} steps | baseline MSE≈{base:.1e}")

    # track best by the training objective (cosine -> higher is better)
    div_weight = cfg.get("div_weight", 0.0)   # anti-collapse usage-entropy term (0=off)
    history, step = [], 0
    best_cos, best_mse = -1.0, float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        for batch in train_loader:
            images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            out = model(images)
            loss = _recon_loss(out["recon"], images, loss_mode)
            # anti-collapse: push batch-mean codebook usage toward uniform
            # (minimizing negative entropy of usage raises perplexity).
            if div_weight > 0 and "aleph_logits" in out["svd"]:
                a = torch.softmax(out["svd"]["aleph_logits"] / model.address_tau, dim=-1)
                usage = a.reshape(-1, a.shape[-1]).mean(0)
                loss = loss + div_weight * (usage * usage.clamp_min(1e-12).log()).sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1

            if step % report_every == 0:
                tmse, tcos, tcv, tppl, tamg = evaluate(model, test_loader, device)
                history.append((step, loss.item(), tmse, tcos, tcv, tppl, tamg))
                addr_str = (f" ppl={tppl:.1f}/{2*model.n_axes} amargin={tamg:.3f}"
                            if model.address != "none" else "")
                print(f"  ep{epoch} step{step:6d} train_loss={loss.item():.3e} "
                      f"test_mse={tmse:.3e} test_cos={tcos:.4f} test_cv={tcv:.3f}"
                      f"{addr_str} lr={sched.get_last_lr()[0]:.2e}")
                improved = (tcos > best_cos) if loss_mode != "mse" else (tmse < best_mse)
                best_cos, best_mse = max(best_cos, tcos), min(best_mse, tmse)
                if improved and save_path:
                    # load_model-compatible checkpoint (model_type='aleph')
                    save_aleph_checkpoint(
                        model, save_path, epoch=epoch, test_mse=tmse,
                        extra={"step": step, "loss_mode": loss_mode,
                               "test_cos": tcos, "dataset": dataset})
    print(f"AlephModel[{decode_mode}] {dataset}: best test_cos={best_cos:.4f} "
          f"best test_mse={best_mse:.3e} (baseline MSE≈{base:.1e}, "
          f"ratio {best_mse/base:.1f}x)")
    return model, history


# back-compat alias: earlier session calls used train_geosphere
train_geosphere = train_aleph


__all__ = ["train_aleph", "train_geosphere", "evaluate", "BASE_CFG",
           "TINY_IMAGENET_CFG", "PRESETS", "BASELINE_MSE"]