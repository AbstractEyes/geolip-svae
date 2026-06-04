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
import os
import json
import math
import torch
import torch.nn.functional as F

try:                                          # progress bar (Colab + console)
    from tqdm.auto import tqdm as _tqdm
except Exception:                             # minimal no-op shim if tqdm absent
    class _tqdm:
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): pass
        def set_postfix(self, *a, **k): pass
        def close(self): pass
        @staticmethod
        def write(msg): print(msg)

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
    """From aleph_logits (B,N,V,2K): SOFT-field perplexity (effective oriented
    axes by the softmax mixture), mean top-1 address probability, and HARD
    perplexity (effective oriented axes actually selected by argmax — the real
    discrete-code usage; this is the one that matters for address='hard').
    Returns (soft_ppl, mean_margin, hard_ppl, usage_vec)."""
    a = torch.softmax(aleph_logits / tau, dim=-1)        # (B,N,V,2K)
    flat = a.reshape(-1, a.shape[-1])                    # (rows, 2K)
    usage = flat.mean(0)                                 # (2K,) soft batch-mean usage
    ent = -(usage * usage.clamp_min(1e-12).log()).sum()
    soft_ppl = float(ent.exp().item())                  # soft-field effective axes
    margin = float(a.max(dim=-1).values.mean().item())  # mean top-1 prob
    # discrete code usage: which oriented axis wins the argmax per row
    K2 = flat.shape[-1]
    idx = flat.argmax(dim=-1)
    hu = torch.bincount(idx, minlength=K2).float()
    hu = hu / hu.sum().clamp_min(1.0)
    h_ent = -(hu * hu.clamp_min(1e-12).log()).sum()
    hard_ppl = float(h_ent.exp().item())                # argmax effective axes
    return soft_ppl, margin, hard_ppl, usage


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20):
    model.eval()
    tot_mse, tot_cos, tot_cv, tot_ppl, tot_amg, tot_hppl, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
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
            ppl, amg, hppl, _ = _address_stats(out["svd"]["aleph_logits"], model.address_tau)
            tot_ppl += ppl; tot_amg += amg; tot_hppl += hppl
        n += 1
    model.train()
    n = max(n, 1)
    return (tot_mse / n, tot_cos / n, tot_cv / n,
            tot_ppl / n if has_addr else float("nan"),
            tot_amg / n if has_addr else float("nan"),
            tot_hppl / n if has_addr else float("nan"))


def train_aleph(decode_mode: str = "tied", *, dataset: str = "byte_trigram",
                address: str = "soft", K: int = 64, address_tau: float = 0.1,
                loss_mode: str | None = None, quick: bool = False,
                device: str = "cuda", cfg_overrides: dict | None = None,
                save_path: str | None = None, report_every: int = 100,
                hf_version: str | None = None,
                hf_repo: str = "AbstractPhil/geolip-aleph-void",
                upload: bool = True, save_dir: str | None = None,
                tb_dir: str | None = None, hf_token: str | None = None):
    """Train one AlephModel (geolip-aleph-void). address='soft'|'hard' uses the
    aleph-address bottleneck (learned codebook of K projective axes); 'none'
    trains the recon-real tied autoencoder (the gate). loss_mode defaults to
    'cosine' for tiny_imagenet, 'mse' for byte_trigram. Set cfg_overrides=
    dict(div_weight=0.01) to enable the anti-collapse usage-entropy term if
    codebook perplexity drops.

    HuggingFace + TensorBoard are wired by default. Uploads land in `hf_repo`
    (default AbstractPhil/geolip-aleph-void) under the load_model-compatible
    layout: `{hf_version}/checkpoints/best.pt` and `{hf_version}/final_report.json`,
    with TB logs under `{hf_version}/tensorboard/{run_name}`. hf_version defaults
    to a name derived from (dataset, decode_mode, address, K). Auth via hf_token
    arg or the HF_TOKEN env var. upload is forced off for quick=True smoke runs.
    Scalars logged: train/loss, test/{mse,cos,cv}, lr, and the aleph params
    aleph/{perplexity,address_margin}.

    Returns (model, history) with rows
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
        "dec_hidden": cfg.get("dec_hidden"), "dec_depth": cfg.get("dec_depth"),
        "codebook_init": cfg.get("codebook_init", "random"),
        "freeze_codebook": cfg.get("freeze_codebook", False),
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

    # ── output paths / run identity ──
    if quick:
        upload = False                          # smoke runs never pollute the repo
    if hf_version is None:
        hf_version = f"aleph_{dataset}_{decode_mode}_{model.address}_K{model.n_axes}"
    save_dir = save_dir or "/content/aleph_checkpoints"
    tb_dir = tb_dir or "/content/aleph_runs"
    os.makedirs(save_dir, exist_ok=True)
    run_name = f"{hf_version}_tau{model.address_tau}_lr{cfg['lr']}"
    best_ckpt_path = save_path or os.path.join(save_dir, "best.pt")

    # ── TensorBoard ──
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_path = os.path.join(tb_dir, run_name)
        writer = SummaryWriter(tb_path)
        print(f"  TensorBoard: {tb_path}")
    except Exception as e:
        tb_path = None
        print(f"  TensorBoard: disabled ({type(e).__name__}: {e})")

    # ── HuggingFace (uploads to the new repo by default) ──
    if hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import login as _hf_login
            _hf_login(token=hf_token, add_to_git_credential=False)
        except Exception as e:
            print(f"  [hf-auth] login from hf_token failed: {type(e).__name__}: {e}")

    hf_enabled, api = False, None
    hf_prefix = f"{hf_version}/checkpoints"
    if upload:
        try:
            from huggingface_hub import HfApi
            api = HfApi(); api.whoami()
            hf_enabled = True
            print(f"  HuggingFace: {hf_repo}/{hf_prefix}")
        except Exception as e:
            print(f"  HuggingFace: disabled ({e})")

    def upload_to_hf(local_path, remote_name, prefix=None):
        if not hf_enabled:
            return
        prefix = prefix if prefix is not None else hf_prefix
        try:
            api.upload_file(path_or_fileobj=local_path,
                            path_in_repo=f"{prefix}/{remote_name}",
                            repo_id=hf_repo, repo_type="model")
            _tqdm.write(f"  ☁️  Uploaded: {hf_repo}/{prefix}/{remote_name}")
        except Exception as e:
            _tqdm.write(f"  ⚠️  HF upload: {e}")

    def sync_tb():
        if not (hf_enabled and tb_path):
            return
        try:
            writer.flush()
            api.upload_folder(folder_path=tb_path,
                              path_in_repo=f"{hf_version}/tensorboard/{run_name}",
                              repo_id=hf_repo, repo_type="model")
        except Exception:
            pass

    # track best by the training objective (cosine -> higher is better)
    div_weight = cfg.get("div_weight", 0.0)   # anti-collapse usage-entropy term (0=off)
    if div_weight > 0:
        model._emit_logits = True             # diversity term needs the address logits
    history, step = [], 0
    best_cos, best_mse = -1.0, float("inf")
    pbar = _tqdm(total=total_steps, desc=f"{decode_mode}/{model.address}",
                 dynamic_ncols=True)
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
            pbar.update(1)
            pbar.set_postfix(ep=epoch, loss=f"{loss.item():.2e}", refresh=False)

            if step % report_every == 0:
                tmse, tcos, tcv, tppl, tamg, thppl = evaluate(model, test_loader, device)
                history.append((step, loss.item(), tmse, tcos, tcv, tppl, tamg, thppl))
                addr_str = (f" ppl={tppl:.1f}/{2*model.n_axes} hppl={thppl:.1f} "
                            f"amargin={tamg:.3f}" if model.address != "none" else "")
                _tqdm.write(f"  ep{epoch} step{step:6d} train_loss={loss.item():.3e} "
                            f"test_mse={tmse:.3e} test_cos={tcos:.4f} test_cv={tcv:.3f}"
                            f"{addr_str} lr={sched.get_last_lr()[0]:.2e}")
                pbar.set_postfix(ep=epoch, loss=f"{loss.item():.2e}",
                                 cos=f"{tcos:.4f}",
                                 hppl=(f"{thppl:.0f}" if model.address != "none" else "—"),
                                 refresh=True)
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), step)
                    writer.add_scalar("test/mse", tmse, step)
                    writer.add_scalar("test/cos", tcos, step)
                    writer.add_scalar("test/cv", tcv, step)
                    writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                    if model.address != "none":
                        writer.add_scalar("aleph/perplexity", tppl, step)
                        writer.add_scalar("aleph/hard_perplexity", thppl, step)
                        writer.add_scalar("aleph/address_margin", tamg, step)
                improved = (tcos > best_cos) if loss_mode != "mse" else (tmse < best_mse)
                best_cos, best_mse = max(best_cos, tcos), min(best_mse, tmse)
                if improved:
                    # load_model-compatible checkpoint (model_type='aleph'),
                    # saved locally; pushed to HF at the epoch boundary below.
                    save_aleph_checkpoint(
                        model, best_ckpt_path, epoch=epoch, test_mse=tmse,
                        extra={"step": step, "loss_mode": loss_mode,
                               "test_cos": tcos, "dataset": dataset})
        # end of epoch: push current best + sync TB (coarse cadence, not per-step)
        if os.path.exists(best_ckpt_path):
            upload_to_hf(best_ckpt_path, "best.pt")
        sync_tb()
    pbar.close()
    print(f"AlephModel[{decode_mode}] {dataset}: best test_cos={best_cos:.4f} "
          f"best test_mse={best_mse:.3e} (baseline MSE≈{base:.1e}, "
          f"ratio {best_mse/base:.1f}x)")

    # ── final report (load_model backfill not needed; kept for parity + metrics) ──
    if writer is not None:
        writer.close()
    final_report = {
        "run_name": run_name,
        "config": model.get_config(),
        "n_params": model.num_params(),
        "dataset": dataset, "loss_mode": loss_mode,
        "baseline_mse": base,
        "best_test_mse": best_mse, "best_test_cos": best_cos,
        "history": history,
        "history_columns": ["step", "train_loss", "test_mse", "test_cos",
                             "test_cv", "perplexity", "address_margin",
                             "hard_perplexity"],
    }
    report_path = os.path.join(save_dir, "final_report.json")
    try:
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2)
        upload_to_hf(report_path, "final_report.json", prefix=hf_version)
    except Exception as e:
        print(f"  ⚠️  final_report: {type(e).__name__}: {e}")
    sync_tb()
    if hf_enabled:
        print(f"  saved to {hf_repo}/{hf_version}")
    return model, history


# ============================================================================
#  AlephTransformer training — a macro shell over a FROZEN aleph battery.
#  The aleph is frozen + detached at the stem; only the shell (spectral
#  transformer + decoder) trains, on the EXTERNAL recon of the same data format
#  the aleph was trained on. D_lens is the multiscale knob — sweep it for a
#  series of multiscale processing capacities over one frozen address.
# ============================================================================


@torch.no_grad()
def evaluate_transformer(model, loader, device, max_batches: int = 20):
    """External-recon eval for AlephTransformer: MSE, cosine, omega-CV (column-norm
    spread of the lensed rows), mean spectral alpha (how engaged the transformer is),
    and the READ-ONLY aleph address margin of the stem against the frozen codebook."""
    model.eval()
    tot_mse = tot_cos = tot_cv = tot_amg = tot_alpha = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
        out = model(images)
        tot_mse += F.mse_loss(out["external_recon"], images).item()
        tot_cos += F.cosine_similarity(out["external_recon"].flatten(1),
                                       images.flatten(1), dim=1).mean().item()
        tot_cv += _col_norm_cv(out["M_lens"])
        tot_alpha += float(out.get("mean_alpha", 0.0))
        if "aleph" in out:
            tot_amg += float(out["aleph"]["margin"].max(dim=-1).values.mean().item())
        n += 1
    model.train()
    n = max(n, 1)
    return (tot_mse / n, tot_cos / n, tot_cv / n, tot_amg / n, tot_alpha / n)


def train_aleph_transformer(
    aleph_hf_version: str, *,
    D_lens: int = 256, dataset: str = "byte_trigram",
    stem: str = "m_hat", lens_sign: str = "signed",
    n_heads: int = 8, n_layers: int = 6, shell_hidden: int = 512,
    loss_mode: str | None = None, quick: bool = False, device: str = "cuda",
    cfg_overrides: dict | None = None, transformer_cfg: dict | None = None,
    aleph_repo: str = "AbstractPhil/geolip-aleph-void",
    hf_version: str | None = None, hf_repo: str = "AbstractPhil/geolip-aleph-void",
    upload: bool = True, save_dir: str | None = None, tb_dir: str | None = None,
    hf_token: str | None = None, report_every: int = 100):
    """Train one AlephTransformer macro shell over a FROZEN aleph battery.

    The aleph (aleph_hf_version, loaded from aleph_repo via load_model) is frozen
    and detached at the stem boundary; only the shell (spectral transformer +
    decoder) trains, on the EXTERNAL recon of the same dataset the aleph was
    trained on (default byte_trigram, the matched format). D_lens is the multiscale
    knob — sweep it for the series. stem='m_hat' lenses the addressed direction
    (strengthen the address); stem='m' lenses the raw rows (control).

    Saves a shell checkpoint to {hf_version}/checkpoints/best.pt under hf_repo
    (shell weights only; the frozen aleph is referenced by aleph_hf_version).
    Returns (model, history); rows
    (step, train_loss, ext_mse, ext_cos, omega_cv, address_margin, mean_alpha)."""
    from geolip_svae.dataset_presets import get_dataset_bundle
    from geolip_svae.aleph_model import AlephTransformer, AlephTransformerConfig
    from geolip_svae.inference.loading import load_model

    cfg = dict(PRESETS[dataset])
    if loss_mode is None:
        loss_mode = "cosine" if dataset == "tiny_imagenet" else "mse"
    if quick:
        cfg.update(epochs=3)
        if dataset == "byte_trigram":
            cfg.update(ds_size=50_000, val_size=2_000)
        upload = False
    if cfg_overrides:
        cfg.update(cfg_overrides)

    # ── frozen aleph battery ──
    if hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token
    aleph, _ = load_model(hf_version=aleph_hf_version, repo_id=aleph_repo, device=device)
    for p in aleph.parameters():
        p.requires_grad_(False)
    # the dataset must lay bytes out with the SAME patch size the aleph was trained on,
    # else the frozen encoder sees a scrambled byte grid.
    cfg["patch_size"] = aleph.patch_size
    cfg["channels"] = aleph.channels
    print(f"frozen aleph: {aleph_repo}/{aleph_hf_version}  "
          f"(V={aleph.matrix_v} D={aleph.D} ps={aleph.patch_size} addr={aleph.address})")

    # ── shell ──
    tcfg = dict(D_lens=D_lens, stem=stem, lens_sign=lens_sign,
                n_heads=n_heads, n_layers=n_layers, shell_hidden=shell_hidden)
    if transformer_cfg:
        tcfg.update(transformer_cfg)
    model = AlephTransformer(aleph, AlephTransformerConfig(**tcfg), device=device)
    print(f"AlephTransformer D_lens={D_lens} stem={stem} lens_sign={lens_sign} "
          f"n_heads={n_heads}x{n_layers} | shell_params={model.num_params()}")

    bundle = get_dataset_bundle(cfg, channels=cfg["channels"])
    train_loader, test_loader = bundle.train_loader, bundle.test_loader

    opt = torch.optim.Adam(model.shell_parameters(), lr=cfg["lr"])   # shell only; Adam (house rule)
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = max(1, cfg.get("ds_size", 50_000) // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # ── output paths / run identity ──
    if hf_version is None:
        hf_version = f"aleph_transformer_{dataset}_d{D_lens}_{lens_sign}_{stem}"
    save_dir = save_dir or "/content/aleph_transformer_checkpoints"
    tb_dir = tb_dir or "/content/aleph_transformer_runs"
    os.makedirs(save_dir, exist_ok=True)
    run_name = f"{hf_version}_h{n_heads}x{n_layers}_lr{cfg['lr']}"
    best_ckpt_path = os.path.join(save_dir, "best.pt")

    # ── TensorBoard ──
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_path = os.path.join(tb_dir, run_name)
        writer = SummaryWriter(tb_path)
        print(f"  TensorBoard: {tb_path}")
    except Exception as e:
        tb_path = None
        print(f"  TensorBoard: disabled ({type(e).__name__}: {e})")

    # ── HuggingFace ──
    hf_enabled, api = False, None
    hf_prefix = f"{hf_version}/checkpoints"
    if upload:
        try:
            from huggingface_hub import HfApi
            api = HfApi(); api.whoami()
            hf_enabled = True
            print(f"  HuggingFace: {hf_repo}/{hf_prefix}")
        except Exception as e:
            print(f"  HuggingFace: disabled ({e})")

    def upload_to_hf(local_path, remote_name, prefix=None):
        if not hf_enabled:
            return
        prefix = prefix if prefix is not None else hf_prefix
        try:
            api.upload_file(path_or_fileobj=local_path,
                            path_in_repo=f"{prefix}/{remote_name}",
                            repo_id=hf_repo, repo_type="model")
            _tqdm.write(f"  ☁️  Uploaded: {hf_repo}/{prefix}/{remote_name}")
        except Exception as e:
            _tqdm.write(f"  ⚠️  HF upload: {e}")

    def save_shell(epoch, ext_mse, ext_cos):
        # shell weights only (lens buffer + transformer + decoder + assessor buffer);
        # the frozen aleph is referenced by aleph_hf_version and reloaded on load.
        shell_state = {k: v for k, v in model.state_dict().items()
                       if not k.startswith("aleph.")}
        torch.save({
            "model_type": "aleph_transformer",
            "config": model.get_config(),
            "aleph_hf_version": aleph_hf_version, "aleph_repo": aleph_repo,
            "shell_state_dict": shell_state,
            "epoch": epoch, "ext_mse": ext_mse, "ext_cos": ext_cos,
        }, best_ckpt_path)

    history, step = [], 0
    best_cos, best_mse = -1.0, float("inf")
    pbar = _tqdm(total=total_steps, desc=f"aleph-T d{D_lens}/{stem}", dynamic_ncols=True)
    for epoch in range(1, cfg["epochs"] + 1):
        for batch in train_loader:
            images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            out = model(images)
            loss = _recon_loss(out["external_recon"], images, loss_mode)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(ep=epoch, loss=f"{loss.item():.2e}", refresh=False)

            if step % report_every == 0:
                emse, ecos, ecv, eamg, ealpha = evaluate_transformer(model, test_loader, device)
                history.append((step, loss.item(), emse, ecos, ecv, eamg, ealpha))
                _tqdm.write(f"  ep{epoch} step{step:6d} train_loss={loss.item():.3e} "
                            f"ext_mse={emse:.3e} ext_cos={ecos:.4f} omega_cv={ecv:.3f} "
                            f"amargin={eamg:.3f} alpha={ealpha:.3f} lr={sched.get_last_lr()[0]:.2e}")
                pbar.set_postfix(ep=epoch, loss=f"{loss.item():.2e}",
                                 cos=f"{ecos:.4f}", a=f"{ealpha:.2f}", refresh=True)
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), step)
                    writer.add_scalar("ext/mse", emse, step)
                    writer.add_scalar("ext/cos", ecos, step)
                    writer.add_scalar("ext/omega_cv", ecv, step)
                    writer.add_scalar("shell/mean_alpha", ealpha, step)
                    writer.add_scalar("shell/address_margin", eamg, step)
                    writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                improved = (ecos > best_cos) if loss_mode != "mse" else (emse < best_mse)
                best_cos, best_mse = max(best_cos, ecos), min(best_mse, emse)
                if improved:
                    save_shell(epoch, emse, ecos)
        if os.path.exists(best_ckpt_path):
            upload_to_hf(best_ckpt_path, "best.pt")
    pbar.close()
    print(f"AlephTransformer d{D_lens}/{stem}: best ext_cos={best_cos:.4f} "
          f"best ext_mse={best_mse:.3e}")

    final_report = {
        "run_name": run_name, "config": model.get_config(),
        "n_params": model.num_params(), "dataset": dataset, "loss_mode": loss_mode,
        "aleph_hf_version": aleph_hf_version,
        "best_ext_mse": best_mse, "best_ext_cos": best_cos, "history": history,
        "history_columns": ["step", "train_loss", "ext_mse", "ext_cos",
                            "omega_cv", "address_margin", "mean_alpha"],
    }
    report_path = os.path.join(save_dir, "final_report.json")
    try:
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2)
        upload_to_hf(report_path, "final_report.json", prefix=hf_version)
    except Exception as e:
        print(f"  ⚠️  final_report: {type(e).__name__}: {e}")
    if writer is not None:
        writer.close()
    return model, history


def train_aleph_transformer_series(aleph_hf_version: str,
                                   D_lens_series=(64, 128, 256, 512), **kw):
    """Finetune the multiscale SERIES: one AlephTransformer shell per D_lens over
    the SAME frozen aleph + same data format — a series of multiscale processing
    capacities. n_heads (default 8) must divide every D_lens in the series.
    Returns {D_lens: (model, history)}."""
    results = {}
    for d in D_lens_series:
        print("\n" + "=" * 70 + f"\n  ALEPH-TRANSFORMER  D_lens={d}\n" + "=" * 70)
        results[d] = train_aleph_transformer(aleph_hf_version, D_lens=d, **kw)
    return results


# back-compat alias: earlier session calls used train_geosphere
train_geosphere = train_aleph


__all__ = ["train_aleph", "train_geosphere", "evaluate", "BASE_CFG",
           "TINY_IMAGENET_CFG", "PRESETS", "BASELINE_MSE",
           "train_aleph_transformer", "train_aleph_transformer_series",
           "evaluate_transformer"]