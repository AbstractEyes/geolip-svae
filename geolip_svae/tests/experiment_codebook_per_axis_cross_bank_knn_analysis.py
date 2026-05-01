# ════════════════════════════════════════════════════════════════════
# Per-axis cross-bank kNN density — h2-64 (disposable cell)
# ════════════════════════════════════════════════════════════════════
# Hypothesis correction: there is no discrete canonical frame; structure
# is continuous coverage of S^3. The right per-axis statistic is then
# k-th nearest CROSS-bank neighbor distance — measures how densely
# populated an axis's neighborhood is by other banks.
#
# Outputs supporting both routes:
#   (A) Optimization (Phil): per-axis "redundancy" score = 1 / d_kNN.
#       Common axes drag cosine assessments toward 1; rare axes are
#       where bank-distinctness lives. Subtract / weight accordingly.
#   (B) Substrate characterization (Claude): per-bank rare-axis count
#       gives diversity ranking that's more informative than cosine
#       outlier z-scores; per-region density on uniform sphere sample
#       maps where on S^3 banks concentrate vs spread.
#
# Method:
#   1. Pool all axes (~1639 unit vectors), with bank provenance
#   2. Compute full pairwise angular distance matrix
#   3. For each axis, find k-th nearest neighbor *from a different bank*
#      (k = 1, 3, 5)
#   4. Classify per-axis: common (k=1 close) vs rare (k=1 far) by quantile
#   5. Per-bank rare-axis count → diversity ranking
#   6. Uniform sphere sample → per-region density
# ════════════════════════════════════════════════════════════════════

import json
import time
from pathlib import Path

import numpy as np
import torch

from geolip_svae.arrays import BatteryArrayModel
from geolip_svae.inference import extract_codebook, make_calibration

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CALIB = 64
SEED = 42
HF_REPO = 'AbstractPhil/geolip-svae-h2-64'
PHASE = 'final'
K_NEIGHBORS = [1, 3, 5]
N_SPHERE_SAMPLES = 4096
RARE_QUANTILE = 0.85   # axes in top 15% of d_kNN1 are "rare"
COMMON_QUANTILE = 0.15  # axes in bottom 15% are "common"

# ── Reload codebooks (defensive) ─────────────────────────────────────
print(f"Loading {HF_REPO} ({PHASE} phase) ...")
arr = BatteryArrayModel.from_pretrained(HF_REPO)
arr.to(DEVICE).eval()
n_batteries = arr.config.n_batteries
calib = make_calibration('gaussian', n=N_CALIB, size=64, seed=SEED)
print(f"  {n_batteries} batteries")

print(f"\nExtracting {n_batteries} {PHASE}-phase codebooks ...")
t0 = time.time()
codebooks = []
for battery_idx in range(n_batteries):
    bank = arr.bank(battery_idx=battery_idx, phase=PHASE)
    cb = extract_codebook(
        bank, calib, batch_size=16,
        model_id=f'{HF_REPO}/battery_{battery_idx}/{PHASE}',
        calibration_name='gaussian',
    )
    codebooks.append({
        'battery_idx': battery_idx,
        'axes': cb.axes.detach().cpu().numpy(),
        'n_axes': cb.n_axes,
    })
print(f"  ✓ extracted in {time.time()-t0:.1f}s")

D = codebooks[0]['axes'].shape[1]

# ── Pool axes with bank provenance ───────────────────────────────────
all_axes = []
axis_bank = []   # parallel: bank_idx for each pooled axis
for c in codebooks:
    for k in range(c['n_axes']):
        all_axes.append(c['axes'][k])
        axis_bank.append(c['battery_idx'])
all_axes = np.array(all_axes, dtype=np.float64)
axis_bank = np.array(axis_bank)
n_total = len(all_axes)
all_axes /= np.linalg.norm(all_axes, axis=1, keepdims=True).clip(min=1e-12)
print(f"\nPooled {n_total} axes from {n_batteries} banks "
      f"(mean {n_total/n_batteries:.1f} axes/bank)")

# ── Pairwise angular distance ────────────────────────────────────────
print(f"\nComputing pairwise angular distances ...")
t0 = time.time()
dot = np.clip(all_axes @ all_axes.T, -1.0, 1.0)
ang_dist = np.arccos(dot)  # in radians
np.fill_diagonal(ang_dist, np.inf)  # exclude self
print(f"  ✓ {ang_dist.shape} matrix in {time.time()-t0:.1f}s")

# ── Cross-bank kNN (the key probe) ───────────────────────────────────
# For each axis, find k-th nearest neighbor that's from a DIFFERENT bank.
print(f"\nComputing cross-bank kNN distances (k = {K_NEIGHBORS}) ...")
t0 = time.time()

# Build same-bank mask: True where row and column are same bank
same_bank_mask = (axis_bank[:, None] == axis_bank[None, :])
# Apply: distance to same-bank axes = inf so they're excluded from kNN
ang_dist_xbank = np.where(same_bank_mask, np.inf, ang_dist)

# For each row, sort and grab the k-th nearest (1-indexed)
# np.partition is O(n) per row vs full sort
max_k = max(K_NEIGHBORS)
sorted_dists = np.sort(ang_dist_xbank, axis=1)  # ascending
# k-th nearest is at index k-1
knn = {k: sorted_dists[:, k - 1] for k in K_NEIGHBORS}
print(f"  ✓ kNN computed in {time.time()-t0:.1f}s")

# ── Per-axis distribution ────────────────────────────────────────────
print(f"\n{'═' * 78}")
print("PER-AXIS CROSS-BANK kNN DISTANCE DISTRIBUTION")
print(f"{'═' * 78}")
for k in K_NEIGHBORS:
    d = knn[k]
    finite = d[np.isfinite(d)]
    print(f"\n  k = {k}: {'(no finite values)' if len(finite) == 0 else ''}")
    if len(finite) == 0:
        continue
    print(f"    N axes with valid k-NN: {len(finite)}/{n_total}")
    print(f"    angular distance: "
          f"min={np.degrees(finite.min()):.2f}°, "
          f"mean={np.degrees(finite.mean()):.2f}°, "
          f"median={np.degrees(np.median(finite)):.2f}°, "
          f"max={np.degrees(finite.max()):.2f}°")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        p = np.percentile(finite, q)
        print(f"    p{q:>2}: {np.degrees(p):>5.2f}°")

# Histogram for k=1
d1 = knn[1]
d1_deg = np.degrees(d1[np.isfinite(d1)])
print(f"\n  k=1 angular distance histogram (bin width 1°):")
edges = np.arange(0, np.ceil(d1_deg.max()) + 2)
hist, _ = np.histogram(d1_deg, bins=edges)
max_count = hist.max() if len(hist) else 1
for b in range(len(hist)):
    if hist[b] == 0:
        continue
    bar = '█' * int(40 * hist[b] / max_count)
    print(f"    [{edges[b]:>4.0f}°, {edges[b+1]:>4.0f}°): {hist[b]:>5}  {bar}")

# ── Classify axes: common vs rare ────────────────────────────────────
print(f"\n{'═' * 78}")
print(f"AXIS CLASSIFICATION (using d_kNN1 quantiles)")
print(f"{'═' * 78}")
d1_finite = d1[np.isfinite(d1)]
common_thresh = np.percentile(d1_finite, COMMON_QUANTILE * 100)
rare_thresh = np.percentile(d1_finite, RARE_QUANTILE * 100)
print(f"  common threshold (≤ p{int(COMMON_QUANTILE*100)}): "
      f"{np.degrees(common_thresh):.3f}°")
print(f"  rare threshold (≥ p{int(RARE_QUANTILE*100)}): "
      f"{np.degrees(rare_thresh):.3f}°")

is_common = d1 <= common_thresh
is_rare = d1 >= rare_thresh
is_typical = ~(is_common | is_rare)

print(f"\n  axes classified:")
print(f"    common (close cross-bank neighbor):    {is_common.sum():>5}")
print(f"    typical:                                {is_typical.sum():>5}")
print(f"    rare (no close cross-bank neighbor):   {is_rare.sum():>5}")

# ── Per-bank rare-axis ranking ───────────────────────────────────────
print(f"\n{'═' * 78}")
print(f"PER-BANK RARE-AXIS COUNTS (diversity ranking)")
print(f"  Banks with many rare axes have unusual sphere coverage —")
print(f"  better seeds for diverse text-solver work than cosine outliers")
print(f"{'═' * 78}")

bank_stats = []
for b in range(n_batteries):
    bank_mask = (axis_bank == b)
    n_axes_b = bank_mask.sum()
    n_rare_b = (is_rare & bank_mask).sum()
    n_common_b = (is_common & bank_mask).sum()
    mean_d1 = d1[bank_mask].mean()
    bank_stats.append({
        'battery_idx': b,
        'n_axes': int(n_axes_b),
        'n_common': int(n_common_b),
        'n_rare': int(n_rare_b),
        'mean_d1_deg': float(np.degrees(mean_d1)),
        'rare_fraction': float(n_rare_b / max(n_axes_b, 1)),
    })

# Sort by rare count (descending)
bank_stats_by_rare = sorted(bank_stats, key=lambda r: -r['n_rare'])
print(f"\n  Top 15 banks with MOST rare axes (most distinctive coverage):")
print(f"  {'rank':>4}  {'batt':>4}  {'n_axes':>6}  {'#common':>7}  {'#rare':>5}  "
      f"{'rare_frac':>9}  {'mean_d1°':>9}")
for rank, r in enumerate(bank_stats_by_rare[:15], 1):
    print(f"  {rank:>4}  {r['battery_idx']:>4}  {r['n_axes']:>6}  "
          f"{r['n_common']:>7}  {r['n_rare']:>5}  "
          f"{r['rare_fraction']:>9.3f}  {r['mean_d1_deg']:>8.2f}°")

print(f"\n  Top 10 banks with FEWEST rare axes (most typical coverage):")
print(f"  {'rank':>4}  {'batt':>4}  {'n_axes':>6}  {'#common':>7}  {'#rare':>5}  "
      f"{'rare_frac':>9}  {'mean_d1°':>9}")
bank_stats_by_typical = sorted(bank_stats, key=lambda r: r['n_rare'])
for rank, r in enumerate(bank_stats_by_typical[:10], 1):
    print(f"  {rank:>4}  {r['battery_idx']:>4}  {r['n_axes']:>6}  "
          f"{r['n_common']:>7}  {r['n_rare']:>5}  "
          f"{r['rare_fraction']:>9.3f}  {r['mean_d1_deg']:>8.2f}°")

# Ranking by mean_d1 — alternative diversity metric
print(f"\n  Top 15 banks by mean d_kNN1 (alternative diversity metric):")
print(f"  Banks high here have axes whose nearest cross-bank neighbor is far")
print(f"  {'rank':>4}  {'batt':>4}  {'mean_d1°':>9}  {'n_rare':>6}  {'rare_frac':>10}")
bank_stats_by_meand = sorted(bank_stats, key=lambda r: -r['mean_d1_deg'])
for rank, r in enumerate(bank_stats_by_meand[:15], 1):
    print(f"  {rank:>4}  {r['battery_idx']:>4}  {r['mean_d1_deg']:>8.2f}°  "
          f"{r['n_rare']:>6}  {r['rare_fraction']:>10.3f}")

# ── Cross-reference with cosine outliers from morning's sweep ─────────
print(f"\n{'═' * 78}")
print(f"CROSS-REFERENCE: cosine outliers vs kNN-density outliers")
print(f"{'═' * 78}")
cosine_outliers = [42, 52, 40, 7, 58]  # from yesterday's z-score ranking
print(f"\n  Morning's cosine outliers: {cosine_outliers}")
print(f"\n  Where do they sit in the rare-axis ranking?")
print(f"  {'batt':>4}  {'cosine_z':>9}  {'rare_rank':>10}  {'#rare':>6}  "
      f"{'mean_d1°':>9}  {'rare_frac':>10}")
cosine_z = {42: -2.64, 52: -2.07, 40: -1.47, 7: -1.46, 58: -1.37}
rare_rank = {r['battery_idx']: rank for rank, r in enumerate(bank_stats_by_rare, 1)}
meand_rank = {r['battery_idx']: rank for rank, r in enumerate(bank_stats_by_meand, 1)}
for b in cosine_outliers:
    r = next(r for r in bank_stats if r['battery_idx'] == b)
    print(f"  {b:>4}  {cosine_z[b]:>+9.2f}  {rare_rank[b]:>10}  "
          f"{r['n_rare']:>6}  {r['mean_d1_deg']:>8.2f}°  "
          f"{r['rare_fraction']:>10.3f}")

# ── Uniform sphere sampling for per-region density ───────────────────
print(f"\n{'═' * 78}")
print(f"PER-REGION DENSITY (uniform sphere sampling, N = {N_SPHERE_SAMPLES})")
print(f"{'═' * 78}")
np.random.seed(SEED)
sphere_samples = np.random.randn(N_SPHERE_SAMPLES, D)
sphere_samples /= np.linalg.norm(sphere_samples, axis=1, keepdims=True)
# Sign-canonicalize to match codebook convention
def sign_canonicalize(M):
    out = M.copy()
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            if abs(M[i, k]) > 1e-6:
                if M[i, k] < 0:
                    out[i] = -M[i]
                break
    return out
sphere_samples = sign_canonicalize(sphere_samples)

# For each sample, find nearest pooled axis (any bank), nearest distance, and
# count of banks that have an axis within e.g. 8° of the sample
sample_to_axis = np.clip(sphere_samples @ all_axes.T, -1.0, 1.0)
sample_to_axis_ang = np.arccos(sample_to_axis)
nearest_dist = sample_to_axis_ang.min(axis=1)
nearest_idx = sample_to_axis_ang.argmin(axis=1)
nearest_bank = axis_bank[nearest_idx]

# How many banks have an axis within 8° of this sample?
EIGHT_DEG = np.radians(8)
within_8deg = sample_to_axis_ang < EIGHT_DEG  # [N_samples, n_total]
banks_within_8 = []
for s in range(N_SPHERE_SAMPLES):
    bank_set = set(axis_bank[within_8deg[s]].tolist())
    banks_within_8.append(len(bank_set))
banks_within_8 = np.array(banks_within_8)

print(f"\n  Sample → nearest pooled axis distance:")
print(f"    min={np.degrees(nearest_dist.min()):.2f}°, "
      f"mean={np.degrees(nearest_dist.mean()):.2f}°, "
      f"median={np.degrees(np.median(nearest_dist)):.2f}°, "
      f"max={np.degrees(nearest_dist.max()):.2f}°")

print(f"\n  Per-sample density (banks with ≥1 axis within 8°):")
print(f"    min={banks_within_8.min()}, mean={banks_within_8.mean():.1f}, "
      f"max={banks_within_8.max()}")
print(f"    samples covered by ALL 64 banks: {(banks_within_8 == 64).sum()}")
print(f"    samples covered by ≥32 banks:    {(banks_within_8 >= 32).sum()}")
print(f"    samples covered by ≥16 banks:    {(banks_within_8 >= 16).sum()}")
print(f"    samples covered by ≥1 bank:      {(banks_within_8 >= 1).sum()}")
print(f"    samples covered by 0 banks:      {(banks_within_8 == 0).sum()}")

# Histogram of coverage
print(f"\n  Coverage histogram (bin width 4 banks):")
edges = np.arange(0, 70, 4)
hist, _ = np.histogram(banks_within_8, bins=edges)
max_count = hist.max() if len(hist) else 1
for b in range(len(hist)):
    if hist[b] == 0:
        continue
    bar = '█' * int(40 * hist[b] / max_count)
    print(f"    [{edges[b]:>3}-{edges[b+1]-1:>2} banks]: {hist[b]:>4}  {bar}")

# ── Persist ──────────────────────────────────────────────────────────
out_dir = Path('/content')
out_dir.mkdir(exist_ok=True)
json_path = out_dir / 'h2_64_xbank_knn_density.json'
with open(json_path, 'w') as f:
    json.dump({
        'config': {
            'hf_repo': HF_REPO,
            'phase': PHASE,
            'n_batteries': n_batteries,
            'D': D,
            'n_calib': N_CALIB,
            'seed': SEED,
            'n_total_pooled_axes': n_total,
            'k_neighbors': K_NEIGHBORS,
            'rare_quantile': RARE_QUANTILE,
            'common_quantile': COMMON_QUANTILE,
            'common_threshold_deg': float(np.degrees(common_thresh)),
            'rare_threshold_deg': float(np.degrees(rare_thresh)),
            'n_sphere_samples': N_SPHERE_SAMPLES,
        },
        'per_axis': {
            'd_knn_rad': {str(k): knn[k].tolist() for k in K_NEIGHBORS},
            'is_common': is_common.tolist(),
            'is_rare': is_rare.tolist(),
            'axis_bank': axis_bank.tolist(),
        },
        'per_bank': bank_stats,
        'per_bank_ranked_by_rare': [r['battery_idx'] for r in bank_stats_by_rare],
        'per_bank_ranked_by_mean_d1': [r['battery_idx'] for r in bank_stats_by_meand],
        'sphere_sample_density': {
            'nearest_dist_deg': np.degrees(nearest_dist).tolist(),
            'nearest_bank': nearest_bank.tolist(),
            'banks_within_8deg': banks_within_8.tolist(),
        },
    }, f, indent=2)
print(f"\n✓ JSON: {json_path}")

# ── Closing read ─────────────────────────────────────────────────────
print(f"\n{'═' * 78}")
print(f"Reading the result")
print(f"{'═' * 78}")

print(f"""
Continuous coverage interpretation:
  - Median d_kNN1 = {np.degrees(np.median(d1_finite)):.2f}° → typical axis has a
    cross-bank neighbor within ~{np.degrees(np.median(d1_finite)):.0f}° of itself
  - {is_rare.sum()} axes ({100*is_rare.sum()/n_total:.0f}%) are rare (no close
    cross-bank partner) — these are the structurally distinctive directions
  - Median sphere coverage: {int(np.median(banks_within_8))} banks have an
    axis within 8° of any random S^3 point

Diversity ranking COMPARISON:
  Cosine z-score outliers ranked: 42, 52, 40, 7, 58
  Rare-axis count ranked top 5:   {[r['battery_idx'] for r in bank_stats_by_rare[:5]]}
  Mean d_kNN1 ranked top 5:       {[r['battery_idx'] for r in bank_stats_by_meand[:5]]}

If these lists overlap heavily, the cosine measurement and the kNN measurement
agree on which banks are distinctive. If they differ substantially, kNN is
finding a different kind of distinctiveness — banks with many rare-direction
axes that cosine missed because cosine was averaging over redundant matches.

For text-solver seeding, the rare-axis-ranked list is the principled choice:
banks at the top of that list cover regions of S^3 that few other banks reach.
""")

# Set agreement metric
top5_cosine = set([42, 52, 40, 7, 58])
top5_rare = set(r['battery_idx'] for r in bank_stats_by_rare[:5])
top5_meand = set(r['battery_idx'] for r in bank_stats_by_meand[:5])
print(f"Set agreement:")
print(f"  cosine ∩ rare-count: {sorted(top5_cosine & top5_rare)}")
print(f"  cosine ∩ mean-d1:    {sorted(top5_cosine & top5_meand)}")
print(f"  rare-count ∩ mean-d1: {sorted(top5_rare & top5_meand)}")