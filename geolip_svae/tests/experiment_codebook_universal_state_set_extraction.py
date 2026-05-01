# ════════════════════════════════════════════════════════════════════
# Universal state-set extraction — h2-64 (disposable cell)
# ════════════════════════════════════════════════════════════════════
# Hypothesis: the 64 banks share an underlying canonical structure on
# S^3. Each bank is a partial sample of that structure. Pool all axes,
# cluster on angular distance, look for the cluster-count plateau.
#
# Outputs supporting both routes:
#   (A) Optimization (Phil): inter-bank shared-canonical count matrix.
#       Use this as a baseline to subtract from cosine assessments —
#       pairs sharing more canonicals have higher "expected" cosine.
#   (B) Substrate characterization (Claude): the canonical directions
#       themselves, their distribution on S^3, and per-bank coverage
#       fingerprints showing which canonicals each bank populates.
#
# Method:
#   1. Pool all final-phase axes across 64 banks (~1664 unit vectors)
#   2. Compute full angular distance matrix
#   3. Single-linkage hierarchical clustering (scipy.cluster.hierarchy)
#   4. Sweep cutoff threshold θ ∈ [0.01, 0.50]
#   5. Identify plateau: range of θ where cluster count is stable
#   6. At plateau θ: extract clusters, compute per-bank coverage, build
#      shared-count matrix
# ════════════════════════════════════════════════════════════════════

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from geolip_svae.arrays import BatteryArrayModel
from geolip_svae.inference import extract_codebook, make_calibration

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CALIB = 64
SEED = 42
HF_REPO = 'AbstractPhil/geolip-svae-h2-64'
PHASE = 'final'

# ── Reload codebooks (defensive, re-extracts) ────────────────────────
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
assert D == 4, f"Cell assumes D=4; got D={D}"

# ── Pool all axes with battery provenance ────────────────────────────
print(f"\nPooling axes across {len(codebooks)} banks ...")
all_axes = []           # (n_total, D) unit vectors
axis_origin = []        # parallel: (battery_idx, within_bank_axis_idx) per row
for c in codebooks:
    for k in range(c['n_axes']):
        all_axes.append(c['axes'][k])
        axis_origin.append((c['battery_idx'], k))
all_axes = np.array(all_axes, dtype=np.float64)
n_total = len(all_axes)
print(f"  total pooled axes: {n_total}")
print(f"  axes/bank: min={min(c['n_axes'] for c in codebooks)}, "
      f"mean={n_total/len(codebooks):.1f}, "
      f"max={max(c['n_axes'] for c in codebooks)}")

# Normalize defensively (axes should already be unit but make sure)
all_axes /= np.linalg.norm(all_axes, axis=1, keepdims=True).clip(min=1e-12)

# ── Angular distance matrix ──────────────────────────────────────────
# arccos of |dot| (absolute value because we treat antipodal-equivalent;
# but axes are sign-canonicalized so |·| ≈ · for matched canonicals)
print(f"\nComputing pairwise angular distance matrix ({n_total}×{n_total}) ...")
t0 = time.time()
dot = np.clip(all_axes @ all_axes.T, -1.0, 1.0)
# Since axes are sign-canonicalized, two "same direction" axes have dot ≈ +1.
# We use plain arccos here; if antipodal pairs had snuck through we'd see them
# as dot ≈ -1 and they'd cluster separately (which would be a bug to surface).
ang_dist = np.arccos(dot)  # in radians, range [0, π]
# Symmetrize and zero diagonal for cleanliness
ang_dist = (ang_dist + ang_dist.T) / 2
np.fill_diagonal(ang_dist, 0.0)
print(f"  ✓ matrix shape {ang_dist.shape} in {time.time()-t0:.1f}s")
print(f"  off-diagonal min: {ang_dist[~np.eye(n_total, dtype=bool)].min():.4f} rad "
      f"({np.degrees(ang_dist[~np.eye(n_total, dtype=bool)].min()):.2f}°)")
print(f"  off-diagonal max: {ang_dist.max():.4f} rad "
      f"({np.degrees(ang_dist.max()):.2f}°)")

# Sanity check: did any antipodal-equivalent pair sneak through?
# Sign-canonicalization should mean cosine ≥ 0 between any two axes that
# represent the "same" direction — a negative dot = genuinely-different direction.
# If we see arccos > π/2 (i.e. dot < 0), that's normal — axes spanning the
# sphere will of course be > 90° apart.
neg_dot_count = (dot < 0).sum() - n_total  # exclude diagonal where dot=1
print(f"  pairs with dot < 0: {neg_dot_count} of {n_total*(n_total-1)} "
      f"(directions on opposite sides of canonical hemisphere)")

# ── Single-linkage clustering ────────────────────────────────────────
print(f"\nRunning single-linkage clustering ...")
t0 = time.time()
condensed = squareform(ang_dist, checks=False)
Z = linkage(condensed, method='single')
print(f"  ✓ linkage computed in {time.time()-t0:.1f}s")

# ── Sweep cutoff threshold ───────────────────────────────────────────
print(f"\n{'═' * 72}")
print("CLUSTER COUNT vs CUTOFF THRESHOLD")
print(f"{'═' * 72}")
print(f"\nSweeping cutoff θ from 0.01 to 0.50 rad (0.6° to 28.6°)")
print(f"{'─' * 72}")
print(f"  {'θ rad':>7}  {'θ deg':>7}  {'n_clust':>8}  {'mean_size':>10}  "
      f"{'max_size':>9}  {'singletons':>11}")

sweep = []
for theta in np.arange(0.01, 0.51, 0.01):
    labels = fcluster(Z, t=theta, criterion='distance')
    n_clust = labels.max()
    sizes = np.bincount(labels)[1:]  # cluster IDs are 1-indexed
    n_single = (sizes == 1).sum()
    sweep.append({
        'theta_rad': float(theta),
        'theta_deg': float(np.degrees(theta)),
        'n_clusters': int(n_clust),
        'mean_size': float(sizes.mean()),
        'max_size': int(sizes.max()),
        'n_singletons': int(n_single),
    })
    # Print every 0.02
    if int(round(theta * 100)) % 2 == 0:
        print(f"  {theta:>7.3f}  {np.degrees(theta):>7.2f}  "
              f"{n_clust:>8}  {sizes.mean():>10.2f}  "
              f"{sizes.max():>9}  {n_single:>11}")

# ── Plateau detection ────────────────────────────────────────────────
# A plateau is a range of θ where n_clusters is stable. Look for the
# longest run of consecutive θ values where n_clusters stays within ±5%
# (or ±2 absolute, whichever is larger).
print(f"\n{'═' * 72}")
print("PLATEAU DETECTION")
print(f"{'═' * 72}")

cluster_counts = np.array([s['n_clusters'] for s in sweep])

# Find plateaus: consecutive θ where n_clusters changes by ≤ tol
def find_plateaus(counts, abs_tol=2, rel_tol=0.05):
    plateaus = []
    i = 0
    while i < len(counts):
        j = i
        anchor = counts[i]
        tol = max(abs_tol, int(rel_tol * anchor))
        while j + 1 < len(counts) and abs(counts[j+1] - anchor) <= tol:
            j += 1
        if j > i:  # at least 2 points
            plateaus.append((i, j, anchor, j - i + 1))
        i = j + 1
    return plateaus

plateaus = find_plateaus(cluster_counts)
plateaus.sort(key=lambda p: -p[3])  # longest first

print(f"\nFound {len(plateaus)} plateau regions (≥2 consecutive θ steps).")
print(f"Top 5 by length:")
print(f"  {'θ_start':>8}  {'θ_end':>8}  {'n_clust':>8}  {'width':>6}")
for (i, j, n, width) in plateaus[:5]:
    theta_i = sweep[i]['theta_rad']
    theta_j = sweep[j]['theta_rad']
    print(f"  {theta_i:>8.3f}  {theta_j:>8.3f}  {n:>8}  {width:>6}")

# Pick the most informative plateau:
# - prefer wider plateaus
# - prefer cluster counts in the "interesting" range (not 1, not n_total)
# - typically the one in the middle of the dendrogram is most informative
plateau_choices = [
    p for p in plateaus
    if 1 < p[2] < n_total // 2 and p[3] >= 3
]
if plateau_choices:
    chosen = plateau_choices[0]
    chosen_i, chosen_j, chosen_n, chosen_width = chosen
    chosen_theta = sweep[(chosen_i + chosen_j) // 2]['theta_rad']
    print(f"\nChosen plateau (longest informative):")
    print(f"  θ range: [{sweep[chosen_i]['theta_rad']:.3f}, "
          f"{sweep[chosen_j]['theta_rad']:.3f}] rad")
    print(f"  n_clusters: {chosen_n}")
    print(f"  width: {chosen_width} steps")
    print(f"  using θ = {chosen_theta:.3f} rad ({np.degrees(chosen_theta):.2f}°)")
else:
    # Fallback: pick the θ where n_clusters is closest to typical codebook size
    typical_n_axes = int(np.median([c['n_axes'] for c in codebooks]))
    diffs = [(abs(s['n_clusters'] - typical_n_axes), idx)
             for idx, s in enumerate(sweep)]
    diffs.sort()
    chosen_idx = diffs[0][1]
    chosen_theta = sweep[chosen_idx]['theta_rad']
    chosen_n = sweep[chosen_idx]['n_clusters']
    print(f"\nNo clear plateau found. Falling back to θ where n_clusters "
          f"matches typical bank size:")
    print(f"  typical bank n_axes: {typical_n_axes}")
    print(f"  using θ = {chosen_theta:.3f} rad ({np.degrees(chosen_theta):.2f}°)")
    print(f"  n_clusters at that θ: {chosen_n}")

# ── Final cluster assignment ─────────────────────────────────────────
print(f"\n{'═' * 72}")
print("CANONICAL DIRECTIONS AT THE CHOSEN θ")
print(f"{'═' * 72}")

final_labels = fcluster(Z, t=chosen_theta, criterion='distance')
n_canonicals = int(final_labels.max())
print(f"\nIdentified {n_canonicals} canonical directions across "
      f"{n_total} pooled axes from {n_batteries} banks.")

# Cluster sizes
cluster_sizes = np.bincount(final_labels)[1:]
print(f"\nCluster size distribution:")
print(f"  min={cluster_sizes.min()}, mean={cluster_sizes.mean():.1f}, "
      f"max={cluster_sizes.max()}, median={int(np.median(cluster_sizes))}")
print(f"  size buckets:")
for low, high in [(1, 1), (2, 5), (6, 16), (17, 32), (33, 48), (49, 64)]:
    if low == high:
        n = int((cluster_sizes == low).sum())
        print(f"    size = {low}:        {n:>4} clusters")
    else:
        n = int(((cluster_sizes >= low) & (cluster_sizes <= high)).sum())
        print(f"    size {low:>2}-{high:<2}:        {n:>4} clusters")

n_per_bank_max = sum(1 for s in cluster_sizes if s == n_batteries)
print(f"\nClusters present in ALL {n_batteries} banks: {n_per_bank_max}")
n_majority = sum(1 for s in cluster_sizes if s >= n_batteries // 2)
print(f"Clusters present in ≥{n_batteries // 2} banks: {n_majority}")
n_singletons = sum(1 for s in cluster_sizes if s == 1)
print(f"Singleton clusters (only one bank's axis):        {n_singletons}")

# ── Compute canonical centroids ──────────────────────────────────────
canonicals = []  # list of dicts: id, size, centroid, member_banks, sign_quadrant
for cid in range(1, n_canonicals + 1):
    member_idx = np.where(final_labels == cid)[0]
    member_axes = all_axes[member_idx]
    # Centroid: spherical mean (normalize the sum)
    centroid = member_axes.sum(axis=0)
    centroid /= np.linalg.norm(centroid).clip(min=1e-12)
    member_banks = sorted(set(axis_origin[i][0] for i in member_idx))
    # Tightness: max angular distance from centroid
    cos_to_centroid = np.clip(member_axes @ centroid, -1.0, 1.0)
    max_ang = float(np.arccos(cos_to_centroid.min())) if len(cos_to_centroid) else 0.0
    mean_ang = float(np.arccos(cos_to_centroid.mean())) if len(cos_to_centroid) else 0.0
    # Sign quadrant of the centroid
    q = 0
    for k in range(1, 4):
        if centroid[k] > 1e-6:
            q |= (1 << (k - 1))
    canonicals.append({
        'id': cid,
        'size': int(len(member_idx)),
        'centroid': centroid.tolist(),
        'member_banks': member_banks,
        'max_angular_distance': max_ang,
        'mean_angular_distance': mean_ang,
        'sign_quadrant': q,
    })

# Sort by size (most-shared first)
canonicals.sort(key=lambda c: -c['size'])

print(f"\nTop 20 most-shared canonicals (by # of banks containing one):")
print(f"  {'id':>4}  {'size':>5}  {'banks':>6}  {'tight (mean°)':>13}  "
      f"{'max°':>7}  {'q':>2}  centroid")
for c in canonicals[:20]:
    n_b = len(c['member_banks'])
    cent_str = '[' + ', '.join(f"{v:+.3f}" for v in c['centroid']) + ']'
    print(f"  {c['id']:>4}  {c['size']:>5}  {n_b:>6}  "
          f"{np.degrees(c['mean_angular_distance']):>12.2f}°  "
          f"{np.degrees(c['max_angular_distance']):>6.2f}°  "
          f"{c['sign_quadrant']:>2}  {cent_str}")

# ── Per-bank coverage fingerprint ────────────────────────────────────
print(f"\n{'═' * 72}")
print("PER-BANK COVERAGE FINGERPRINTS")
print(f"{'═' * 72}")

bank_to_canonicals = {b: set() for b in range(n_batteries)}
for ci, c in enumerate(canonicals):
    for b in c['member_banks']:
        bank_to_canonicals[b].add(c['id'])

coverage_counts = np.array([len(bank_to_canonicals[b]) for b in range(n_batteries)])
print(f"\nN canonicals per bank: min={coverage_counts.min()}, "
      f"mean={coverage_counts.mean():.1f}, max={coverage_counts.max()}, "
      f"median={int(np.median(coverage_counts))}")

# Histogram of coverage
print(f"\nCoverage histogram (canonicals populated per bank):")
edges = np.arange(coverage_counts.min(), coverage_counts.max() + 2)
hist, _ = np.histogram(coverage_counts, bins=edges)
for c, h in zip(edges[:-1], hist):
    if h:
        print(f"  {c:>3} canonicals: "
              f"{h:>3} banks  {'█' * h}")

# ── Inter-bank shared-canonical count matrix ────────────────────────
print(f"\n{'═' * 72}")
print("INTER-BANK SHARED-CANONICAL COUNT (for omit-shared cosine baseline)")
print(f"{'═' * 72}")

shared_matrix = np.zeros((n_batteries, n_batteries), dtype=int)
for i in range(n_batteries):
    si = bank_to_canonicals[i]
    for j in range(i + 1, n_batteries):
        sj = bank_to_canonicals[j]
        n_shared = len(si & sj)
        shared_matrix[i, j] = n_shared
        shared_matrix[j, i] = n_shared

# Per-pair stats
upper = shared_matrix[np.triu_indices(n_batteries, k=1)]
print(f"\nN shared canonicals per pair: min={upper.min()}, "
      f"mean={upper.mean():.1f}, max={upper.max()}, "
      f"median={int(np.median(upper))}")

# Pairs sorted by shared count
print(f"\nDistribution of pair shared counts:")
for low, high in [(0, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]:
    n = int(((upper >= low) & (upper <= high)).sum())
    pct = 100 * n / len(upper)
    print(f"  {low:>2}-{high:<2} shared: {n:>5} pairs  ({pct:>5.1f}%)")

# ── Persist ──────────────────────────────────────────────────────────
out_dir = Path('/content')
out_dir.mkdir(exist_ok=True)
json_path = out_dir / 'h2_64_universal_state_set.json'
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
            'chosen_theta_rad': float(chosen_theta),
            'chosen_theta_deg': float(np.degrees(chosen_theta)),
            'n_canonicals': n_canonicals,
        },
        'sweep': sweep,
        'plateaus_found': len(plateaus),
        'plateau_top5': [
            {'theta_start_rad': sweep[i]['theta_rad'],
             'theta_end_rad': sweep[j]['theta_rad'],
             'n_clusters': int(n),
             'width_steps': int(width)}
            for (i, j, n, width) in plateaus[:5]
        ],
        'canonicals': canonicals,
        'per_bank_coverage': {
            int(b): sorted(list(canon_set))
            for b, canon_set in bank_to_canonicals.items()
        },
        'shared_matrix': shared_matrix.tolist(),
        'cluster_sizes': cluster_sizes.tolist(),
    }, f, indent=2)
print(f"\n✓ JSON: {json_path}")

# ── Closing read ─────────────────────────────────────────────────────
print(f"\n{'═' * 72}")
print("Reading the result")
print(f"{'═' * 72}")

n_universal = sum(1 for c in canonicals if len(c['member_banks']) == n_batteries)
n_majority = sum(1 for c in canonicals if len(c['member_banks']) >= n_batteries // 2)
typical_bank_size = int(np.median([c['n_axes'] for c in codebooks]))

print(f"""
At chosen θ = {chosen_theta:.3f} rad ({np.degrees(chosen_theta):.2f}°),
{n_canonicals} canonical directions span the {n_total} pooled axes from
{n_batteries} banks.

Universality of the state-set:
  - Canonicals present in ALL {n_batteries} banks:  {n_universal}
  - Canonicals present in ≥{n_batteries//2} banks:  {n_majority}
  - Singleton canonicals (one bank only): {sum(1 for c in canonicals if c['size']==1)}

Typical bank populates {typical_bank_size} axes; identifies
{int(np.median(coverage_counts))} canonicals on average.

Pair shared-canonical counts: median {int(np.median(upper))}, mean {upper.mean():.1f}
""")

if n_universal > 0 and n_majority > n_canonicals * 0.5:
    print("VERDICT: Strong universal state-set. Most canonicals are populated")
    print("by a majority of banks; some by ALL banks. Banks are sampling the")
    print("same underlying structure with partial coverage. The omit-shared")
    print("optimization is well-defined: any pair-cosine assessment can")
    print("subtract the contribution of canonicals both banks share.")
elif n_majority > n_canonicals * 0.3:
    print("VERDICT: Weak universal state-set. Some canonicals are common")
    print("across banks but most are partial. Banks share structure but the")
    print("'shared core' is smaller than expected.")
else:
    print("VERDICT: No clear universal state-set at this θ. Most canonicals")
    print("are bank-specific. The pairwise within-quadrant agreement was")
    print("local, not transitive — banks cluster pairwise but not globally.")
    print("Consider re-running at a coarser θ.")