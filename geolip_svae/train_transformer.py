"""
geolip_svae.train_transformer
=============================
Trainer for the geolip-svae-transformer (the model lives in
``geolip_svae.model_transformer`` and is imported, never redefined here).

Two-recon, three-optimizer separation:
  * battery (PatchSVAE)        -> INTERNAL recon, pure MSE (the microcosm)
  * shell   (spectral+decoder) -> EXTERNAL recon from the DETACHED stem
  * growth  (parked)           -> growth_loss, idle until a stencil is set
The detach boundary keeps the three gradients disjoint.

Callable two ways:
  API:  from geolip_svae.train_transformer import train_svae, run_sign_test
        train_svae(epochs=200, hf_repo='AbstractPhil/geolip-svae-transformer',
                   lens_sign='signed', D_lens=256)
  CLI:  python -m geolip_svae.train_transformer train --epochs 200 \
            --hf-repo AbstractPhil/geolip-svae-transformer --d-lens 256
        python -m geolip_svae.train_transformer sign-test --epochs 40

Optional deps degrade gracefully: tqdm (progress bar), safetensors (extra
weight format), hf_backup.py (HuggingFace checkpoint preservation), and the
geolip_svae byte-trigram substrate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm                # notebook bar in Colab, text bar in a terminal
    _HAVE_TQDM = True
except Exception:                             # tqdm absent -> silent passthrough
    _HAVE_TQDM = False

try:
    from safetensors.torch import save_file as _safe_save
    _HAVE_SAFETENSORS = True
except Exception:
    _HAVE_SAFETENSORS = False

try:                                          # skill: scripts/hf_backup.py (copy into the working dir)
    from hf_backup import HFTrainingBackup, BackupConfig
    _HAVE_HF = True
except Exception:
    _HAVE_HF = False

try:                                          # byte-trigram substrate (Colab / installed repo)
    from geolip_svae.dataset_presets import ByteTrigramDataset
    _HAVE_REPO = True
except Exception:
    ByteTrigramDataset = None
    _HAVE_REPO = False

from geolip_svae.model_transformer import (
    GeoConfig, GeoSVAETransformer, byte_recovery, canon,
)


def train_svae(epochs: int = 40, steps_per_epoch: int = 1000, batch_size: int = 512,
               lr: float = 1e-3, growth_lr: float = 1e-3, grad_clip: float = 1.0,
               scheduler: str = 'onecycle',
               corpus_id: str = 'wikitext-2-raw-v1',
               out_dir: str = './geolip_svae_v2_results',
               ckpt_name: str = 'geolip_svae_v2.pt',
               seed: int = 0, device: Optional[str] = None,
               num_workers: int = 4, progress: bool = True,
               hf_repo: Optional[str] = None, hf_preset: Optional[str] = None,
               hf_upload_every: int = 5, hf_best_count: int = 3,
               tensorboard: bool = True, **geo_kwargs) -> GeoSVAETransformer:
    """ALIGNED training with TWO recons monitored and three separated gradients.

      * opt_battery (PatchSVAE)        -> INTERNAL recon (pure MSE). Small pot.
      * opt_shell   (spectral+decoder) -> EXTERNAL recon (pure MSE) from the
        DETACHED stem. The large-pot projection.
      * opt_growth  (growth params)    -> growth_loss on the detached stem. Idle
        while no stencil is registered.

    The detach boundary makes all three gradients disjoint — the external recon
    and the growth can never reach the battery's frame, so battery wobble can
    only come from its own MSE and shows immediately in the internal recon. Both
    recons logged every epoch. Adam, wd=0; scheduler in {'onecycle' (default),
    'cosine', 'none'} anneals the two recon optimizers."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    cfg = GeoConfig(**{k: v for k, v in {**geo_kwargs, 'seed': seed}.items()
                       if k in GeoConfig.__dataclass_fields__})
    model = GeoSVAETransformer(cfg, device=device).to(device)
    bat_p, shell_p, grow_p = (model.battery_parameters(),
                              model.shell_parameters(), model.growth_parameters())
    n_bat, n_shell, n_grow = (sum(p.numel() for p in bat_p),
                              sum(p.numel() for p in shell_p),
                              sum(p.numel() for p in grow_p))
    print(f"geolip-svae-v2 TWO-RECON | battery(PatchSVAE) {n_bat:,} + shell {n_shell:,} "
          f"(shell {n_shell/max(1,n_bat):.0f}x the microcosm) + growth {n_grow:,} "
          f"| D_base{cfg.D_base}->D_lens{cfg.D_lens} V{cfg.V} ps{cfg.ps} | {device}")
    print(f"  battery -> INTERNAL recon (pure MSE) | shell -> EXTERNAL recon (detached stem, lens_sign={cfg.lens_sign}) "
          f"| adam lr={lr} wd=0 sched={scheduler}")
    print(f"  growth : {'adam lr='+str(growth_lr)+' on detached-stem growth_loss' if n_grow else 'parked (no stencil)'}")

    if not _HAVE_REPO:
        raise RuntimeError("ByteTrigramDataset unavailable - run in the Colab env "
                           "with geolip_svae installed, or supply your own loader.")
    ds = ByteTrigramDataset(size=batch_size * steps_per_epoch, img_size=cfg.img_size,
                            patch_size=cfg.ps, channels=cfg.channels, corpus_id=corpus_id)
    pin = 'cuda' in str(device)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin,
        **(dict(persistent_workers=True, prefetch_factor=4) if num_workers > 0 else {}))
    val_imgs = [b[0].to(device, non_blocking=pin)
                for _, b in zip(range(4), torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, shuffle=False))]

    opt_battery = torch.optim.Adam(bat_p, lr=lr, weight_decay=0.0)
    opt_shell = torch.optim.Adam(shell_p, lr=lr, weight_decay=0.0)
    opt_growth = torch.optim.Adam(grow_p, lr=growth_lr, weight_decay=0.0) if grow_p else None
    total = epochs * len(loader)

    def make_sched(opt):
        if scheduler == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total)
        if scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        return None
    sched_battery, sched_shell = make_sched(opt_battery), make_sched(opt_shell)
    best_int, best_ext, history = 0.0, 0.0, []
    use_bar = progress and _HAVE_TQDM
    _log = tqdm.write if use_bar else print

    # ── artifact dirs mirroring the HF subdir layout (checkpoints / config / tensorboard) ──
    ckpt_dir, cfg_dir, tb_dir = out / 'checkpoints', out / 'config', out / 'tensorboard'
    for _d in (ckpt_dir, cfg_dir, tb_dir):
        _d.mkdir(parents=True, exist_ok=True)
    stem_base = Path(ckpt_name).stem
    preset = hf_preset or f"geosvae_d{cfg.D_base}-{cfg.D_lens}_{cfg.lens_sign}"

    tb = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(tb_dir))
        except Exception as e:
            _log(f"  [tb] disabled ({e})")

    backup = None                              # one backup obj -> fixed timestamp -> one run subdir
    if hf_repo and _HAVE_HF:
        try:
            backup = HFTrainingBackup(BackupConfig(
                repo_id=hf_repo, training_preset=preset, local_checkpoint_dir=str(ckpt_dir),
                config_dir=str(cfg_dir), tensorboard_dir=str(tb_dir),
                checkpoint_interval=1, best_checkpoint_count=hf_best_count))
            _log(f"  [hf] preserving -> {hf_repo}/{backup.run_path}  (every {hf_upload_every} ep + final)")
        except Exception as e:
            _log(f"  [hf] disabled ({e})")
    elif hf_repo:
        _log("  [hf] hf_backup.py not importable — copy scripts/hf_backup.py into the working dir")

    def _save_weights(stem: str, ep: int, ir: float, er: float):
        sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        torch.save({'model_state': sd, 'cfg': vars(cfg), 'epoch': ep, 'preset': preset,
                    'internal_recovery': ir, 'external_recovery': er}, str(ckpt_dir / f'{stem}.pt'))
        if _HAVE_SAFETENSORS:
            _safe_save(sd, str(ckpt_dir / f'{stem}.safetensors'),
                       metadata={'epoch': str(ep), 'preset': preset, 'lens_sign': cfg.lens_sign,
                                 'internal_recovery': f'{ir:.4f}', 'external_recovery': f'{er:.4f}'})

    def _write_config(ep: int, final: bool = False):
        json.dump({'preset': preset, 'repo': hf_repo, 'final': final, 'last_epoch': ep,
                   'cfg': vars(cfg), 'battery_params': n_bat, 'shell_params': n_shell,
                   'growth_params': n_grow, 'best_internal_recovery': best_int,
                   'best_external_recovery': best_ext, 'history': history},
                  open(cfg_dir / 'run.json', 'w'), indent=2)

    def _push(msg: str):
        if backup is None:
            return
        try:
            backup.backup(commit_message=msg)
        except Exception as e:
            _log(f"  [hf] upload failed ({e}) — continuing")

    for epoch in range(epochs):
        model.train(); int_mses, ext_mses, glosses = [], [], []
        step_iter = (tqdm(loader, desc=f'epoch {epoch:2d}/{epochs}', leave=False,
                          total=len(loader), unit='batch', dynamic_ncols=True)
                     if use_bar else loader)
        for img, _ in step_iter:
            img = img.to(device, non_blocking=pin)
            out_ = model(img)

            L_int = model.internal_recon_loss(out_, img)        # battery (small pot)
            L_ext = model.external_recon_loss(out_, img)        # shell  (large pot), detached stem
            L_grow = model.growth_loss(out_) if opt_growth is not None else None

            opt_battery.zero_grad(); opt_shell.zero_grad()
            if opt_growth is not None:
                opt_growth.zero_grad()
            L_int.backward()                                    # battery grads only
            L_ext.backward()                                    # shell grads only (M detached)
            if L_grow is not None:
                L_grow.backward()                               # growth grads only
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(bat_p, grad_clip)
                torch.nn.utils.clip_grad_norm_(shell_p, grad_clip)
                if L_grow is not None:
                    torch.nn.utils.clip_grad_norm_(grow_p, grad_clip)
            opt_battery.step(); opt_shell.step()
            if sched_battery is not None:
                sched_battery.step(); sched_shell.step()
            if L_grow is not None:
                opt_growth.step()

            int_mses.append(float(L_int.detach()))
            ext_mses.append(float(L_ext.detach()))
            if L_grow is not None:
                glosses.append(float(L_grow.detach()))
            if use_bar:
                step_iter.set_postfix(int=f'{int_mses[-1]:.5f}', ext=f'{ext_mses[-1]:.5f}',
                                      lr=f'{opt_battery.param_groups[0]["lr"]:.1e}')

        model.eval()
        with torch.no_grad():
            ov = model(val_imgs[0])
            int_rec = sum(byte_recovery(model(vb)['internal_recon'], vb) for vb in val_imgs) / len(val_imgs)
            ext_rec = sum(byte_recovery(model(vb)['external_recon'], vb) for vb in val_imgs) / len(val_imgs)
            env = model.in_envelope(ov)
            alpha = ov['mean_alpha']
            cc_mon = model.cross_contrast_monitor(ov)
            rig_mon = model.rigidity_monitor(ov)
        im, em = sum(int_mses) / len(int_mses), sum(ext_mses) / len(ext_mses)
        gl = (sum(glosses) / len(glosses)) if glosses else float('nan')
        history.append({'epoch': epoch, 'internal_mse': im, 'external_mse': em,
                        'internal_recovery': int_rec, 'external_recovery': ext_rec,
                        'growth_loss': gl, 'cc_monitor': cc_mon, 'rigidity_monitor': rig_mon,
                        'mean_alpha': alpha, 'in_envelope': bool(env)})
        gtxt = f"gloss={gl:+.4f} " if glosses else ""
        _log(f"  epoch {epoch:2d}: int[mse={im:.5f} rec={int_rec:6.2%}] "
             f"ext[mse={em:.5f} rec={ext_rec:6.2%}] | {gtxt}"
             f"cc={cc_mon:+.2f} rig={rig_mon:.4f} a={alpha:.3f} env={env}")
        best_int = max(best_int, int_rec)
        if ext_rec > best_ext:
            best_ext = ext_rec
            _save_weights(f'best_{stem_base}', epoch, int_rec, ext_rec)
        if tb is not None:
            tb.add_scalar('mse/internal', im, epoch); tb.add_scalar('mse/external', em, epoch)
            tb.add_scalar('recovery/internal', int_rec, epoch); tb.add_scalar('recovery/external', ext_rec, epoch)
            tb.add_scalar('monitor/cc', cc_mon, epoch); tb.add_scalar('monitor/rigidity', rig_mon, epoch)
            tb.add_scalar('monitor/mean_alpha', alpha, epoch)
        _write_config(epoch)
        if backup is not None and hf_upload_every > 0 and (epoch + 1) % hf_upload_every == 0:
            _push(f"{preset}: epoch {epoch} (best ext {best_ext:.2%})")

    _save_weights(f'final_{stem_base}', epochs - 1, int_rec, ext_rec)
    _write_config(epochs - 1, final=True)
    if tb is not None:
        tb.flush(); tb.close()
    json.dump({'history': history, 'best_internal_recovery': best_int,
               'best_external_recovery': best_ext, 'cfg': vars(cfg),
               'battery_params': n_bat, 'shell_params': n_shell, 'growth_params': n_grow},
              open(out / 'geolip_svae_v2.json', 'w'), indent=2)
    _push(f"{preset}: FINAL @ epoch {epochs - 1} (best ext {best_ext:.2%})")
    _log(f"  best recovery — internal {best_int:.2%} | external {best_ext:.2%} | "
         f"weights {ckpt_dir}/best_{stem_base}.*"
         + (f" | hf {hf_repo}/{backup.run_path}" if backup is not None else ""))
    model.best_internal, model.best_external, model.history = best_int, best_ext, history
    return model


def run_sign_test(epochs: int = 40, **kw):
    """Hypothesis test: does PRESERVING the per-row sign (the channel that
    rigidity, measured on |cos|, leaves free) lift the EXTERNAL recon toward the
    battery's 99.75%? Control = 'canon' (signs dropped, the ~90% baseline);
    test = 'signed'. omega is sign-invariant, so the ONLY difference between the
    two runs is whether the decoder sees the signed scaffold — any external-recon
    delta is the sign channel and nothing else."""
    res = {}
    for mode in ('canon', 'signed'):
        print("\n" + "=" * 90 + f"\n[sign-test] lens_sign='{mode}'\n" + "=" * 90)
        m = train_svae(epochs=epochs, lens_sign=mode,
                       out_dir=f'./sign_test_{mode}', ckpt_name=f'sign_{mode}.pt', **kw)
        res[mode] = (m.best_internal, m.best_external)
        del m
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
    print("\n" + "=" * 64 + "\n  SIGN TEST — best recovery (internal / external)\n" + "=" * 64)
    for mode, (bi, be) in res.items():
        print(f"    lens_sign={mode:7s}  internal={bi:6.2%}  external={be:6.2%}")
    ce = res.get('canon', (0, 0))[1]
    se = res.get('signed', (0, 0))[1]
    verdict = ('SIGNS CARRY CONTENT — hypothesis supported' if se - ce > 0.03
               else 'no clear sign effect — plan something else')
    print(f"  external delta (signed - canon): {se - ce:+.2%}  ->  {verdict}")
    return res



# ════════════════════════════════════════════════════════════════════════
#  CLI  (the API above is unchanged; this only wraps it for the command line)
# ════════════════════════════════════════════════════════════════════════

def _cli(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog='geolip_svae.train_transformer',
        description='Train the geolip-svae-transformer (model: geolip_svae.model_transformer).')
    sub = p.add_subparsers(dest='cmd', required=True)

    def add_common(sp):
        sp.add_argument('--epochs', type=int, default=40)
        sp.add_argument('--steps-per-epoch', type=int, default=1000)
        sp.add_argument('--batch-size', type=int, default=512)
        sp.add_argument('--lr', type=float, default=1e-3)
        sp.add_argument('--scheduler', choices=['onecycle', 'cosine', 'none'], default='onecycle')
        sp.add_argument('--corpus-id', default='wikitext-2-raw-v1')
        sp.add_argument('--seed', type=int, default=0)
        sp.add_argument('--device', default=None)
        sp.add_argument('--num-workers', type=int, default=4)
        sp.add_argument('--no-progress', action='store_false', dest='progress')
        sp.add_argument('--no-tensorboard', action='store_false', dest='tensorboard')
        # HuggingFace preservation
        sp.add_argument('--hf-repo', default=None, help='e.g. AbstractPhil/geolip-svae-transformer')
        sp.add_argument('--hf-preset', default=None)
        sp.add_argument('--hf-upload-every', type=int, default=5)
        sp.add_argument('--hf-best-count', type=int, default=3)
        # architecture overrides (None -> GeoConfig defaults from model_transformer)
        sp.add_argument('--d-lens', type=int, default=None)
        sp.add_argument('--shell-hidden', type=int, default=None)
        sp.add_argument('--n-heads', type=int, default=None)
        sp.add_argument('--n-layers', type=int, default=None)
        sp.add_argument('--lens-sign', choices=['signed', 'canon'], default=None)
        sp.add_argument('--no-real-svae', action='store_false', dest='use_real_svae', default=None)

    tr = sub.add_parser('train', help='train a single model')
    add_common(tr)
    tr.add_argument('--out-dir', default='./geolip_svae_v2_results')
    tr.add_argument('--ckpt-name', default='geolip_svae_v2.pt')

    st = sub.add_parser('sign-test', help='canon vs signed A/B (run_sign_test)')
    add_common(st)

    a = p.parse_args(argv)

    # only forward architecture overrides the user actually set
    geo = {}
    for arg_name, cfg_name in (('d_lens', 'D_lens'), ('shell_hidden', 'shell_hidden'),
                               ('n_heads', 'n_heads'), ('n_layers', 'n_layers'),
                               ('lens_sign', 'lens_sign'), ('use_real_svae', 'use_real_svae')):
        v = getattr(a, arg_name, None)
        if v is not None:
            geo[cfg_name] = v

    common = dict(
        epochs=a.epochs, steps_per_epoch=a.steps_per_epoch, batch_size=a.batch_size,
        lr=a.lr, scheduler=a.scheduler, corpus_id=a.corpus_id, seed=a.seed,
        device=a.device, num_workers=a.num_workers, progress=a.progress,
        tensorboard=a.tensorboard, hf_repo=a.hf_repo, hf_preset=a.hf_preset,
        hf_upload_every=a.hf_upload_every, hf_best_count=a.hf_best_count,
    )

    if a.cmd == 'train':
        train_svae(out_dir=a.out_dir, ckpt_name=a.ckpt_name, **common, **geo)
    elif a.cmd == 'sign-test':
        geo.pop('lens_sign', None)          # run_sign_test sets lens_sign per arm
        run_sign_test(**common, **geo)


if __name__ == '__main__':
    _cli()