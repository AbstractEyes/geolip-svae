"""
U0 — Preliminary smoke test for the inference framework

Validates the new ``geolip_svae.inference`` package end-to-end before
scaling out to Johanna / Freckles / h2-64 checkpoints. This is the
GATE for U5 and the lens-scope cells (U1-U6).

Coverage:
    Helpers
        T1: pad/crop round-trip
    Codebook artifact
        T2: extract_codebook returns Codebook with valid metadata
        T3: Codebook save/load round-trips axes + metadata + pairs
        T4: deviation() + is_projective_clean() compute correctly
    Scaling
        T5: encode_at_scale direct == legacy encode at native res
        T6: encode_at_scale tile mode at native res = direct (n_tiles=1)
        T7: encode_at_scale tile mode at 2× res returns sane shapes
        T8: reconstruct_at_scale direct == legacy reconstruct at native
        T9: reconstruct_at_scale tile at 2× preserves shape
        T10: auto mode dispatches to direct at native resolution
    Engine
        T11: InferenceEngine encode/reconstruct work without codebook
        T12: encode_axes raises CodebookMissingError without codebook
        T13: encode_axes works after attach_codebook
        T14: incompatible codebook raises CodebookIncompatibleError
        T15: extract_codebook(attach=True) populates engine.codebook

Run from the geolip-svae repo root::

    python -m geolip_svae.tests.U0_smoke_test
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Allow standalone run
sys.path.insert(0, '.')

import torch

from geolip_svae.inference import (
    Codebook,
    CodebookIncompatibleError,
    CodebookMissingError,
    InferenceEngine,
    encode_at_scale,
    extract_codebook,
    reconstruct_at_scale,
)
from geolip_svae.inference.scaling import _crop_pad, _pad_to_multiple
from geolip_svae.inference.legacy import encode as legacy_encode
from geolip_svae.inference.legacy import reconstruct as legacy_reconstruct
from geolip_svae.model import PatchSVAE


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _build_tiny_model(V: int = 8, D: int = 4, ps: int = 4,
                       hidden: int = 32, depth: int = 1) -> PatchSVAE:
    """Build a tiny untrained PatchSVAE for plumbing tests.

    Untrained means the codebook will not be PROJECTIVE-CLEAN. We test
    pipeline plumbing here, not the empirical claim.
    """
    model = PatchSVAE(
        V=V, D=D, ps=ps, hidden=hidden, depth=depth, n_cross=0,
    )
    model.eval()
    return model


def _build_mismatched_codebook(D: int = 16, n_axes: int = 8) -> Codebook:
    """Build a synthetic Codebook with a non-matching D for compat tests."""
    from geolip_svae.inference.codebook import CodebookMetadata
    axes = torch.randn(n_axes, D)
    axes = axes / axes.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return Codebook(
        axes=axes,
        metadata=CodebookMetadata(
            model_id='synthetic', model_class='Synthetic',
            D=D, V=n_axes, n_axes=n_axes, n_pairs=0, n_unpaired=n_axes,
        ),
    )


# ════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════

def test_pad_round_trip():
    """T1: _pad_to_multiple → _crop_pad reconstructs original."""
    print("T1: pad/crop round-trip")
    for shape in [(1, 3, 64, 64), (2, 3, 65, 65),
                   (1, 3, 100, 140), (3, 3, 80, 80)]:
        x = torch.randn(*shape)
        padded, pad_info = _pad_to_multiple(x, 64)
        assert padded.shape[-1] % 64 == 0
        assert padded.shape[-2] % 64 == 0
        recovered = _crop_pad(padded, pad_info)
        assert recovered.shape == x.shape
        assert torch.allclose(recovered, x)
    print("    ✓ all shapes round-trip cleanly")


def test_extract_codebook_artifact():
    """T2: extract_codebook returns a valid Codebook with metadata."""
    print("T2: extract_codebook returns Codebook artifact")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(16, 3, 64, 64)
    cb = extract_codebook(
        model, images, batch_size=8,
        model_id='tiny-test', calibration_name='gaussian',
    )
    assert isinstance(cb, Codebook)
    assert cb.D == 4
    assert 0 < cb.n_axes <= 8
    assert cb.metadata.model_id == 'tiny-test'
    assert cb.metadata.calibration == 'gaussian'
    assert cb.metadata.D == 4
    assert cb.metadata.V == 8
    assert cb.metadata.n_axes == cb.n_axes
    assert cb.metadata.n_pairs + cb.metadata.n_unpaired == cb.n_axes
    norms = cb.axes.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"    ✓ Codebook(D={cb.D}, n_axes={cb.n_axes}, "
          f"pairs={cb.metadata.n_pairs}, unpaired={cb.metadata.n_unpaired})")


def test_codebook_save_load():
    """T3: Codebook.save → Codebook.load round-trips faithfully."""
    print("T3: Codebook save/load round-trip")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(8, 3, 64, 64)
    cb = extract_codebook(
        model, images, batch_size=8,
        model_id='roundtrip-test', calibration_name='gaussian',
    )

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / 'cb_test'
        stem = cb.save(save_path)
        assert (stem.parent / (stem.name + '.safetensors')).exists()
        assert (stem.parent / (stem.name + '.json')).exists()

        cb_loaded = Codebook.load(save_path)
        assert cb_loaded.D == cb.D
        assert cb_loaded.n_axes == cb.n_axes
        assert torch.allclose(cb_loaded.axes, cb.axes, atol=1e-5)
        assert cb_loaded.metadata.model_id == 'roundtrip-test'
        assert cb_loaded.metadata.calibration == 'gaussian'
        assert cb_loaded.pairs == cb.pairs
        assert cb_loaded.unpaired == cb.unpaired
    print(f"    ✓ saved + loaded with axes/metadata/pairs intact")


def test_codebook_deviation():
    """T4: deviation() and is_projective_clean() are computable + sane."""
    print("T4: deviation + is_projective_clean")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(8, 3, 64, 64)
    cb = extract_codebook(model, images, batch_size=8)

    if cb.n_axes >= 2:
        dev = cb.deviation()
        assert isinstance(dev, float)
        assert dev == dev  # not NaN
        clean = cb.is_projective_clean(threshold=0.05)
        assert isinstance(clean, bool)
        # Untrained model: deviation will NOT be near zero. Just sanity.
        assert -2.0 < dev < 2.0, f"deviation {dev} outside sane range"
        print(f"    ✓ deviation={dev:+.4f}, clean={clean} "
              f"(untrained — projective-clean NOT expected)")
    else:
        print(f"    ✓ skipped (only {cb.n_axes} axes; "
              f"insufficient for projective-angle stat)")


def test_encode_at_scale_direct():
    """T5: encode_at_scale direct mode == legacy encode at native res."""
    print("T5: encode_at_scale direct == legacy encode")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(2, 3, 64, 64)

    legacy = legacy_encode(model, images)
    new = encode_at_scale(model, images, mode='direct')

    assert new['mode_used'] == 'direct'
    assert torch.allclose(legacy['M'], new['M'], atol=1e-6)
    assert new['gh'] == legacy['gh']
    assert new['gw'] == legacy['gw']
    print("    ✓ direct mode matches legacy encode()")


def test_encode_at_scale_tile_native():
    """T6: tile mode at native res yields n_tiles=1, no padding."""
    print("T6: tile mode at native resolution")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(2, 3, 64, 64)

    direct = encode_at_scale(model, images, mode='direct')
    tile = encode_at_scale(model, images, tile_size=64, mode='tile')

    assert tile['mode_used'] == 'tile'
    assert tile['n_tiles'] == 1
    assert tile['pad_h'] == 0
    assert tile['pad_w'] == 0
    assert tile['M'].shape == direct['M'].shape
    print(f"    ✓ tile_size=64 at 64×64 → n_tiles=1, pad=0")


def test_encode_at_scale_tile_2x():
    """T7: tile mode at 2× res produces 4 tiles, 4× patch count."""
    print("T7: tile mode at 2× resolution (128×128)")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(2, 3, 128, 128)

    out = encode_at_scale(model, images, tile_size=64, mode='tile')

    assert out['mode_used'] == 'tile'
    assert out['n_tiles'] == 4
    expected_total_patches = 4 * 256
    assert out['M'].shape[1] == expected_total_patches
    assert out['M'].shape == (2, 1024, 8, 4)
    print(f"    ✓ 128×128 → 4 tiles, M shape {tuple(out['M'].shape)}")


def test_reconstruct_at_scale_direct_native():
    """T8: reconstruct_at_scale direct == legacy reconstruct at native."""
    print("T8: reconstruct_at_scale direct == legacy reconstruct")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(2, 3, 64, 64)

    legacy = legacy_reconstruct(model, images)
    out = reconstruct_at_scale(model, images, mode='direct')

    assert out['mode_used'] == 'direct'
    assert torch.allclose(out['recon'], legacy, atol=1e-6)
    assert out['recon'].shape == images.shape
    print("    ✓ direct mode matches legacy reconstruct()")


def test_reconstruct_at_scale_tile_2x():
    """T9: tile reconstruction at 2× res preserves shape."""
    print("T9: reconstruct_at_scale tile at 2×")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(2, 3, 128, 128)

    out = reconstruct_at_scale(model, images, tile_size=64, mode='tile')

    assert out['mode_used'] == 'tile'
    assert out['recon'].shape == images.shape
    assert out['mse_per_image'].shape == (2,)
    assert out['n_tiles'] == 4
    assert torch.isfinite(out['mse_per_image']).all()
    print(f"    ✓ recon shape {tuple(out['recon'].shape)}, "
          f"MSE per image {out['mse_per_image'].tolist()}")


def test_auto_mode_dispatches_direct():
    """T10: auto mode picks direct at native resolution."""
    print("T10: auto mode dispatches direct at native")
    model = _build_tiny_model(V=8, D=4)
    images = torch.randn(1, 3, 64, 64)

    out = encode_at_scale(model, images, mode='auto')
    assert out['mode_used'] == 'direct'
    print("    ✓ auto → direct at 64×64")


def test_engine_basic():
    """T11: InferenceEngine encode/reconstruct work without a codebook."""
    print("T11: InferenceEngine basic encode/reconstruct")
    model = _build_tiny_model(V=8, D=4)
    engine = InferenceEngine(model)
    images = torch.randn(2, 3, 64, 64)

    enc = engine.encode(images)
    assert enc['mode_used'] == 'direct'
    assert 'M' in enc

    rec = engine.reconstruct(images)
    assert rec['recon'].shape == images.shape
    assert 'mse_per_image' in rec

    assert engine.codebook is None
    print("    ✓ engine encodes + reconstructs without codebook")


def test_engine_codebook_missing_error():
    """T12: encode_axes raises CodebookMissingError without codebook."""
    print("T12: encode_axes raises CodebookMissingError")
    model = _build_tiny_model(V=8, D=4)
    engine = InferenceEngine(model)
    images = torch.randn(1, 3, 64, 64)

    try:
        engine.encode_axes(images)
    except CodebookMissingError as e:
        msg = str(e)
        assert 'attach' in msg.lower() or 'codebook' in msg.lower()
        print(f"    ✓ raised CodebookMissingError with helpful message")
        return
    raise AssertionError("expected CodebookMissingError, got no exception")


def test_engine_attach_and_encode_axes():
    """T13: encode_axes works after attach_codebook."""
    print("T13: attach_codebook then encode_axes")
    model = _build_tiny_model(V=8, D=4)
    engine = InferenceEngine(model)

    calib = torch.randn(8, 3, 64, 64)
    cb = extract_codebook(
        model, calib, batch_size=8,
        model_id='tiny', calibration_name='gaussian',
    )
    engine.attach_codebook(cb)
    assert engine.codebook is cb

    images = torch.randn(2, 3, 64, 64)
    out = engine.encode_axes(images)
    assert 'activations' in out
    assert 'axes' in out
    # activations: [B, n_patches, V, n_axes]
    assert out['activations'].shape[0] == 2
    assert out['activations'].shape[2] == 8  # V
    assert out['activations'].shape[3] == cb.n_axes
    print(f"    ✓ activations shape {tuple(out['activations'].shape)}")


def test_engine_incompatible_codebook():
    """T14: incompatible codebook raises CodebookIncompatibleError."""
    print("T14: attach incompatible codebook (D mismatch)")
    model = _build_tiny_model(V=8, D=4)
    engine = InferenceEngine(model)
    cb_wrong = _build_mismatched_codebook(D=16, n_axes=8)

    try:
        engine.attach_codebook(cb_wrong)
    except CodebookIncompatibleError as e:
        msg = str(e)
        assert 'D' in msg
        print(f"    ✓ raised CodebookIncompatibleError")

        # Also verify override works: same arch, opt-in to mismatch
        engine_lax = InferenceEngine(
            model, require_codebook_compatibility=False,
        )
        engine_lax.attach_codebook(cb_wrong)  # should NOT raise
        assert engine_lax.codebook is cb_wrong
        print(f"    ✓ require_codebook_compatibility=False allows override")
        return
    raise AssertionError(
        "expected CodebookIncompatibleError, got no exception"
    )


def test_engine_extract_with_attach():
    """T15: extract_codebook(attach=True) populates engine.codebook."""
    print("T15: engine.extract_codebook(attach=True)")
    model = _build_tiny_model(V=8, D=4)
    engine = InferenceEngine(model)
    calib = torch.randn(8, 3, 64, 64)

    cb = engine.extract_codebook(
        calib, batch_size=8, attach=True,
        model_id='attached', calibration_name='gaussian',
    )
    assert engine.codebook is cb
    assert cb.metadata.model_id == 'attached'
    assert cb.metadata.model_class == 'PatchSVAE'  # auto-filled
    print(f"    ✓ extracted + attached, model_class auto-filled")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("U0 Preliminary Smoke Test — inference framework rebuild")
    print("=" * 72)
    print()

    tests = [
        test_pad_round_trip,
        test_extract_codebook_artifact,
        test_codebook_save_load,
        test_codebook_deviation,
        test_encode_at_scale_direct,
        test_encode_at_scale_tile_native,
        test_encode_at_scale_tile_2x,
        test_reconstruct_at_scale_direct_native,
        test_reconstruct_at_scale_tile_2x,
        test_auto_mode_dispatches_direct,
        test_engine_basic,
        test_engine_codebook_missing_error,
        test_engine_attach_and_encode_axes,
        test_engine_incompatible_codebook,
        test_engine_extract_with_attach,
    ]

    failed = []
    t0 = time.time()
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"    ✗ FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(fn.__name__)
        print()

    elapsed = time.time() - t0
    print("=" * 72)
    if failed:
        print(f"✗ {len(failed)}/{len(tests)} tests FAILED in {elapsed:.1f}s:")
        for name in failed:
            print(f"    - {name}")
        print()
        print("Fix before proceeding to U5 (cross-band codebook capacity).")
        sys.exit(1)
    else:
        print(f"✓ All {len(tests)} smoke tests passed in {elapsed:.1f}s")
        print()
        print("Cleared to proceed:")
        print("  - U5: cross-band codebook capacity test")
        print("        (Johanna D=16 + Freckles D=4 large-param)")
        print("  - U1-U6: lens scope tests on gaussian battery")


if __name__ == '__main__':
    main()