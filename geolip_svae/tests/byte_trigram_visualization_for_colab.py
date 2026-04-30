# ═══════════════════════════════════════════════════════════════════
# Visualize text as byte-trigram pixel image
# ═══════════════════════════════════════════════════════════════════
# Each (R, G, B) pixel = three consecutive UTF-8 bytes.
# Byte ∈ [0, 255] mapped to pixel ∈ [-1, 1] via (b - 127.5) / 127.5.
# Patches go in row-major order across the image; cells go in
# row-major order within each patch. Padding (0x20 = space) fills the
# remainder of the canvas.
# ═══════════════════════════════════════════════════════════════════
import math

import torch
import matplotlib.pyplot as plt

from geolip_svae.inference import (
    load_model, InferenceEngine, SentenceEncoder,
)


HF_VERSION = 'byte_trigram_proto_64_patch_2_v1'

# Phrases chosen to exhibit different byte-pattern signatures:
PHRASES = [
    "the cat sat on the mat",            # plain ASCII English
    "import torch.nn.functional as F",   # ASCII code
    "Café résumé naïve 中文 🎉",         # multi-byte UTF-8
    "AAAA BBBB CCCC DDDD",               # rigid repetition pattern
]


# ── Load model + encoder once ───────────────────────────────────────
model, cfg = load_model(HF_VERSION)
engine = InferenceEngine(model)
enc = SentenceEncoder(
    engine,
    img_size=64,
    patch_size=cfg['patch_size'],
    pad='space',
)


# ── Visualization ───────────────────────────────────────────────────

@torch.no_grad()
def visualize_text(text: str, enc: SentenceEncoder, show_recon: bool = True):
    """Render text-as-byte-trigram-image and (optionally) the model's recon.

    Three panels:
        full image  — whole canvas (padding visible as uniform color)
        real bytes  — zoom on the patches that contain sentence bytes
        recon       — same zoom after round-trip through the model
    """
    # Encode through the same path the dataset uses
    img = enc.text_to_image(text)                            # (3, H, W) ∈ [-1, 1]

    # Bounding box of real-byte content in the image
    n_real_bytes = min(len(text.encode('utf-8')), enc.img_size ** 2 * 3)
    cells_per_patch = enc.patch_size ** 2
    n_trigrams = math.ceil(n_real_bytes / 3)
    n_patches = max(1, math.ceil(n_trigrams / cells_per_patch))
    gw = enc.img_size // enc.patch_size
    n_rows = math.ceil(n_patches / gw)
    n_cols = min(n_patches, gw)
    pixel_h = n_rows * enc.patch_size
    pixel_w = n_cols * enc.patch_size

    # Optional round-trip
    recon = None
    if show_recon:
        device = next(enc.engine.model.parameters()).device
        out = enc.engine.reconstruct(
            img.unsqueeze(0).to(device), mode='direct',
        )
        recon = out['recon'][0].cpu().clamp(-1, 1)

    # Display: [-1, 1] → [0, 1]; (3, H, W) → (H, W, 3) for matplotlib
    def to_disp(t):
        return ((t + 1) / 2).permute(1, 2, 0).numpy()

    n_panels = 3 if show_recon else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(to_disp(img), interpolation='nearest')
    axes[0].set_title(f'full image ({enc.img_size}×{enc.img_size}, ps={enc.patch_size})')
    axes[0].axis('off')

    axes[1].imshow(to_disp(img[:, :pixel_h, :pixel_w]), interpolation='nearest')
    axes[1].set_title(f'real bytes ({pixel_h}×{pixel_w} px, {n_real_bytes} B)')
    axes[1].axis('off')

    if show_recon:
        axes[2].imshow(
            to_disp(recon[:, :pixel_h, :pixel_w]), interpolation='nearest',
        )
        axes[2].set_title('reconstruction (zoomed)')
        axes[2].axis('off')

    fig.suptitle(repr(text), y=1.05, fontsize=10)
    plt.tight_layout()
    plt.show()

    # Also print the byte stream for the first cell so you can read off
    # which bytes landed where
    raw = text.encode('utf-8')[:12]   # first patch's worth
    triples = [tuple(raw[i:i+3]) for i in range(0, min(12, len(raw)), 3)]
    print(f"  first cells (R, G, B) = {triples}")
    print(f"  → {n_trigrams} trigrams across {n_patches} patches "
          f"({pixel_h}×{pixel_w} pixel block)")


# ── Run on each phrase ──────────────────────────────────────────────
if __name__ == '__main__':
    for phrase in PHRASES:
        visualize_text(phrase, enc)