# ════════════════════════════════════════════════════════════════════
# Topology mapping — h2-64 axis cloud (disposable cell)
# ════════════════════════════════════════════════════════════════════
# 1639 axes from 64 banks scattered on S^3. We've established:
#   - structure is continuous (no discrete clusters at any θ)
#   - 38% of S^3 has no bank within 8° (sparse coverage)
#   - per-axis kNN distances are heterogeneous
#
# Now: is there a geometric topology to the coverage? Three probes:
#
#   PROBE A: kNN graph connectivity
#     Build graph at varying θ. Track components, largest component
#     size, percolation point. "How many topological pieces?"
#
#   PROBE B: Local intrinsic dimension (PCA on neighborhoods)
#     Per-axis k=10 nearest neighbors. Run PCA on neighbor offsets.
#     Eigenvalue spectrum reveals local manifold dimension.
#     If banks sample a 3-manifold (just S^3): local dim ≈ 3
#     If banks sample a 2-surface in S^3:        local dim ≈ 2
#     If banks sample a 1-curve:                 local dim ≈ 1
#
#   PROBE C: Persistent homology (ripser)
#     Full H0/H1/H2 persistence diagram. Voids, loops, components,
#     all properly counted. Authoritative topology.
#
# Important: ALL axes (not cross-bank only). We're characterizing
# the union's topology, so a bank's internal axes are part of the
# coverage shape.
# ════════════════════════════════════════════════════════════════════

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Try to import ripser; install if missing (Colab usually has it via
# scikit-tda but if not, fall back gracefully)
try:
    from ripser import ripser
    HAVE_RIPSER = True
except ImportError:
    print("ripser not available — installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'ripser'])
    try:
        from ripser import ripser
        HAVE_RIPSER = True
    except ImportError:
        HAVE_RIPSER = False
        print("ripser still not available; probe C will be skipped")

from geolip_svae.arrays import BatteryArrayModel
from geolip_svae.inference import extract_codebook, make_calibration

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CALIB = 64
SEED = 42
HF_REPO = 'AbstractPhil/geolip-svae-h2-64'
PHASE = 'final'
K_LOCAL_DIM = 10  # neighborhood size for local PCA

# ── Reload codebooks ─────────────────────────────────────────────────
print(f"Loading {HF_REPO} ({PHASE} phase) ...")
arr = BatteryArrayModel.from_pretrained(HF_REPO)
arr.to(DEVICE).eval()
n_batteries = arr.config.n_batteries
calib = make_calibration('gaussian', n=N_CALIB, size=64, seed=SEED)

print(f"\nExtracting {n_batteries} codebooks ...")
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

# ── Pool ALL axes (with bank provenance for cross-references) ───────
all_axes = []
axis_bank = []
for c in codebooks:
    for k in range(c['n_axes']):
        all_axes.append(c['axes'][k])
        axis_bank.append(c['battery_idx'])
all_axes = np.array(all_axes, dtype=np.float64)
axis_bank = np.array(axis_bank)
all_axes /= np.linalg.norm(all_axes, axis=1, keepdims=True).clip(min=1e-12)
n_total = len(all_axes)
print(f"\nPooled {n_total} axes from {n_batteries} banks")

# ── Pairwise angular distance ────────────────────────────────────────
print(f"\nComputing pairwise angular distances ...")
t0 = time.time()
dot = np.clip(all_axes @ all_axes.T, -1.0, 1.0)
ang_dist = np.arccos(dot)
np.fill_diagonal(ang_dist, 0.0)
print(f"  ✓ {ang_dist.shape} matrix in {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════════════
# PROBE A — kNN graph connectivity sweep
# ════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print(f"PROBE A — kNN GRAPH CONNECTIVITY (graph includes all axes)")
print(f"{'═' * 78}")

# Strategy: at threshold θ, build adjacency where edge(i,j) exists iff
# ang_dist[i,j] ≤ θ AND i ≠ j. Count connected components, track
# largest component size.

print(f"\nSweeping θ from 0.5° to 25°:")
print(f"  {'θ_deg':>7}  {'n_comp':>7}  {'largest':>8}  {'largest%':>9}  "
      f"{'isolated':>9}  {'edge density':>13}")

theta_grid_deg = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                   5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0,
                   18.0, 21.0, 25.0]

connectivity_sweep = []
percolation_theta = None
percolation_threshold = 0.5  # giant component is "percolated" at >= 50%

for theta_deg in theta_grid_deg:
    theta_rad = np.radians(theta_deg)
    adj = (ang_dist <= theta_rad) & (ang_dist > 0)  # exclude self-loops
    sparse_adj = csr_matrix(adj.astype(np.int8))
    n_comp, labels = connected_components(sparse_adj, directed=False)
    sizes = np.bincount(labels)
    largest = sizes.max()
    isolated = int((sizes == 1).sum())
    n_edges = int(adj.sum() / 2)
    edge_density = n_edges / (n_total * (n_total - 1) / 2)

    if percolation_theta is None and largest / n_total >= percolation_threshold:
        percolation_theta = theta_deg

    print(f"  {theta_deg:>7.1f}°  {n_comp:>7}  {largest:>8}  "
          f"{100*largest/n_total:>8.1f}%  {isolated:>9}  "
          f"{edge_density:>13.4f}")

    connectivity_sweep.append({
        'theta_deg': float(theta_deg),
        'n_components': int(n_comp),
        'largest_component_size': int(largest),
        'largest_component_pct': float(100*largest/n_total),
        'n_isolated': isolated,
        'n_edges': n_edges,
        'edge_density': float(edge_density),
    })

if percolation_theta is not None:
    print(f"\n  Percolation point (largest component ≥ 50%): θ = {percolation_theta}°")

# Look for "topological plateaus" — ranges where component count is stable
# (excluding the trivial cases of all-isolated or fully-merged)
n_comps = np.array([s['n_components'] for s in connectivity_sweep])
print(f"\n  Component-count stability check:")
for i in range(len(connectivity_sweep) - 1):
    delta = abs(n_comps[i+1] - n_comps[i])
    if delta < max(2, n_comps[i] * 0.05) and 5 < n_comps[i] < n_total - 5:
        print(f"    θ={connectivity_sweep[i]['theta_deg']}° → "
              f"θ={connectivity_sweep[i+1]['theta_deg']}°: "
              f"{n_comps[i]} → {n_comps[i+1]} components "
              f"(stable, Δ={delta})")

# ════════════════════════════════════════════════════════════════════
# PROBE B — Local intrinsic dimension via PCA
# ════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print(f"PROBE B — LOCAL INTRINSIC DIMENSION (k={K_LOCAL_DIM} neighbor PCA)")
print(f"{'═' * 78}")

# For each axis, take k nearest neighbors (excluding self), compute
# offsets (neighbor - center), run PCA. The eigenvalue spectrum's
# shape tells us the local dimensionality.
#
# In ambient D=4, each neighborhood has 4 eigenvalues. For a uniform
# sample on S^3 (a 3-manifold), the smallest eigenvalue should be
# near 0 (the radial direction perpendicular to the sphere) and the
# other 3 should be similar magnitude.
#
# For a 2-sub-manifold (e.g., great 2-sphere): TWO small eigenvalues
# (radial + one tangent), TWO large.
#
# For a 1-curve: THREE small, ONE large.

t0 = time.time()
local_dim_data = []
eigenvalue_spectra = []  # store all spectra for histogram

# We'll use pre-sorted distance matrix
sorted_neighbors = np.argsort(ang_dist, axis=1)  # ascending; index 0 is self

for i in range(n_total):
    neighbor_idx = sorted_neighbors[i, 1:K_LOCAL_DIM + 1]  # skip self
    center = all_axes[i]
    offsets = all_axes[neighbor_idx] - center  # [k, D]
    # Center the offsets (PCA assumes centered data)
    offsets -= offsets.mean(axis=0)
    # SVD; eigenvalues of covariance = singular values squared / k
    _U, sing_vals, _Vt = np.linalg.svd(offsets, full_matrices=False)
    eigvals = (sing_vals ** 2) / K_LOCAL_DIM  # variance along each PC
    # Sort descending (np.linalg.svd already does, but be explicit)
    eigvals = np.sort(eigvals)[::-1]
    eigenvalue_spectra.append(eigvals)

    # Two ways to estimate local dimension:
    # (1) Count eigenvalues > threshold * largest
    threshold = 0.05  # eigenvalues > 5% of top one count as "real" dimensions
    rel_eigvals = eigvals / max(eigvals[0], 1e-12)
    local_dim_count = int((rel_eigvals > threshold).sum())

    # (2) Effective dimension via participation ratio:
    # PR = (Σλ)² / Σλ²  — gives a continuous measure of "how many
    # dimensions are effectively active"
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    pr_dim = (s1 ** 2) / max(s2, 1e-20) if s2 > 0 else 0.0

    local_dim_data.append({
        'axis_idx': i,
        'bank': int(axis_bank[i]),
        'eigvals': eigvals.tolist(),
        'local_dim_count': local_dim_count,
        'local_dim_pr': float(pr_dim),
    })

eigenvalue_spectra = np.array(eigenvalue_spectra)  # [n_total, D]
print(f"  ✓ local PCA computed in {time.time()-t0:.1f}s")

# Aggregate
all_dim_count = np.array([r['local_dim_count'] for r in local_dim_data])
all_dim_pr = np.array([r['local_dim_pr'] for r in local_dim_data])

print(f"\n  Local dimension via threshold count (relative > 5%):")
for d_int in range(D + 1):
    n_at_d = int((all_dim_count == d_int).sum())
    pct = 100 * n_at_d / n_total
    bar = '█' * int(40 * pct / 100)
    print(f"    dim = {d_int}: {n_at_d:>5} axes ({pct:>5.1f}%)  {bar}")

print(f"\n  Local dimension via participation ratio (continuous):")
print(f"    min:    {all_dim_pr.min():.3f}")
print(f"    mean:   {all_dim_pr.mean():.3f}")
print(f"    median: {np.median(all_dim_pr):.3f}")
print(f"    max:    {all_dim_pr.max():.3f}")
for q in [10, 25, 50, 75, 90]:
    print(f"    p{q:>2}: {np.percentile(all_dim_pr, q):.3f}")

# Mean eigenvalue spectrum — what's the "average shape" of a neighborhood?
mean_spectrum = eigenvalue_spectra.mean(axis=0)
mean_spectrum_normalized = mean_spectrum / mean_spectrum[0]
print(f"\n  Mean eigenvalue spectrum (normalized to top eigenvalue):")
for k in range(D):
    bar = '█' * int(40 * mean_spectrum_normalized[k])
    print(f"    PC{k}: {mean_spectrum_normalized[k]:.4f}  {bar}")

# Interpretation
print(f"\n  Interpretation:")
mode_dim = int(np.bincount(all_dim_count).argmax())
print(f"    Mode of local dimension: {mode_dim}")
if mode_dim == D - 1:
    print(f"    → Banks sample a {D-1}-manifold (S^{D-1}). Expected for")
    print(f"      uniform-on-sphere samples in ambient R^{D}.")
elif mode_dim < D - 1:
    print(f"    → Banks live on a {mode_dim}-submanifold within S^{D-1}.")
    print(f"      The coverage is structurally lower-dimensional than the")
    print(f"      ambient sphere — a real geometric finding.")
else:
    print(f"    → Local dim ≥ ambient dim. Likely numerical artifact in the")
    print(f"      threshold; check the participation ratio instead.")

# ════════════════════════════════════════════════════════════════════
# PROBE C — Persistent homology
# ════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print(f"PROBE C — PERSISTENT HOMOLOGY (ripser, H0/H1/H2)")
print(f"{'═' * 78}")

if not HAVE_RIPSER:
    print("\n  Skipped (ripser unavailable)")
    persistence_summary = None
else:
    # ripser takes a distance matrix or point cloud. Use the angular
    # distance matrix directly to respect S^3 geometry.
    print(f"\n  Running ripser with maxdim=2 (computes H0, H1, H2) ...")
    t0 = time.time()

    # ripser expects metric distance; angular distance is a metric.
    # Use thresh to limit the filtration; we don't need to compute
    # past where the giant component dominates.
    # Pick thresh > percolation but not maximal — saves compute.
    max_thresh = np.radians(20.0)  # 20° — past where percolation is well-formed

    try:
        result = ripser(
            ang_dist,
            distance_matrix=True,
            maxdim=2,
            thresh=max_thresh,
        )
        diagrams = result['dgms']  # list of arrays per dimension
        print(f"  ✓ ripser computed in {time.time()-t0:.1f}s")

        # Each diagram is an array of (birth, death) for that homology dim
        persistence_summary = {}
        for h_dim, dgm in enumerate(diagrams):
            # Persistence = death - birth
            # Filter out the "infinite" feature (the full connected component
            # at birth=0, death=inf for H0)
            finite_dgm = dgm[np.isfinite(dgm[:, 1])]
            persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
            n_features = len(dgm)
            n_finite = len(finite_dgm)

            print(f"\n  H{h_dim}: {n_features} features total ({n_finite} finite)")
            if n_finite > 0:
                print(f"    Birth times (deg):  "
                      f"min={np.degrees(finite_dgm[:, 0].min()):.2f}°, "
                      f"median={np.degrees(np.median(finite_dgm[:, 0])):.2f}°, "
                      f"max={np.degrees(finite_dgm[:, 0].max()):.2f}°")
                print(f"    Death times (deg):  "
                      f"min={np.degrees(finite_dgm[:, 1].min()):.2f}°, "
                      f"median={np.degrees(np.median(finite_dgm[:, 1])):.2f}°, "
                      f"max={np.degrees(finite_dgm[:, 1].max()):.2f}°")
                print(f"    Persistence (deg):  "
                      f"min={np.degrees(persistences.min()):.3f}°, "
                      f"median={np.degrees(np.median(persistences)):.3f}°, "
                      f"max={np.degrees(persistences.max()):.3f}°")

                # Count "significant" features — persistence > some threshold
                # Use multiple thresholds to give shape info
                for p_thresh_deg in [0.5, 1.0, 2.0, 5.0]:
                    p_thresh = np.radians(p_thresh_deg)
                    n_sig = int((persistences > p_thresh).sum())
                    print(f"    Features with persistence > {p_thresh_deg}°: {n_sig}")

                # Top 10 most persistent features
                top10 = np.argsort(persistences)[::-1][:10]
                print(f"    Top 10 most persistent H{h_dim} features:")
                for rank, idx in enumerate(top10, 1):
                    b, d = finite_dgm[idx]
                    p = persistences[idx]
                    print(f"      {rank:>2}. birth={np.degrees(b):>6.2f}°  "
                          f"death={np.degrees(d):>6.2f}°  "
                          f"persistence={np.degrees(p):>6.2f}°")

            # Infinite feature for H0 = the giant component
            n_infinite = n_features - n_finite
            if n_infinite > 0:
                print(f"    Infinite features: {n_infinite}  "
                      f"(connected component(s) that never die)")

            persistence_summary[f'H{h_dim}'] = {
                'n_features': n_features,
                'n_finite': n_finite,
                'n_infinite': n_features - n_finite,
                'finite_diagram': finite_dgm.tolist(),
            }

    except MemoryError as e:
        print(f"  ✗ ripser OOMed: {e}")
        persistence_summary = None
    except Exception as e:
        import traceback
        print(f"  ✗ ripser failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        persistence_summary = None

# ════════════════════════════════════════════════════════════════════
# Synthesis
# ════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print(f"SYNTHESIS — what topology does the bank coverage form?")
print(f"{'═' * 78}")

# A: connectivity
print(f"\n[A] Connectivity:")
if percolation_theta is not None:
    print(f"    Percolation: giant component reaches 50% at θ = {percolation_theta}°")
    print(f"    Below percolation: structure is fragmented")
    print(f"    Above: single connected blob dominates")
# Find smallest theta with n_components < 5 — basically connected
for s in connectivity_sweep:
    if s['n_components'] < 5:
        print(f"    Effectively single component (≤ 5 components) at θ = {s['theta_deg']}°")
        break

# B: dimensionality
print(f"\n[B] Dimensionality:")
print(f"    Mode of local dimension count: {mode_dim} (ambient is {D})")
print(f"    Median participation ratio: {np.median(all_dim_pr):.2f}")
if D - 1 - 0.3 < np.median(all_dim_pr) < D - 0.5:
    print(f"    → Banks sample the ambient sphere as expected ({D-1}-manifold)")
elif np.median(all_dim_pr) < D - 1.5:
    print(f"    → Banks live on a structurally lower-dimensional submanifold")
    print(f"      Coverage is NOT just 'sparse on S^{D-1}', it's confined")

# C: holes
print(f"\n[C] Higher-dim topology:")
if persistence_summary is None:
    print(f"    (skipped)")
else:
    n_h1 = persistence_summary.get('H1', {}).get('n_finite', 0)
    n_h2 = persistence_summary.get('H2', {}).get('n_finite', 0)
    print(f"    H1 (loops): {n_h1} finite features")
    print(f"    H2 (voids): {n_h2} finite features")
    if n_h1 > 5:
        print(f"    → The coverage has nontrivial 1-dim cycle structure")
    if n_h2 > 0:
        print(f"    → The coverage encloses voids in S^{D-1}")
        print(f"      The 38% 'unvisited' regions of S^3 may be structured holes")

# Persist
out_dir = Path('/content')
out_dir.mkdir(exist_ok=True)
json_path = out_dir / 'h2_64_topology.json'
with open(json_path, 'w') as f:
    json.dump({
        'config': {
            'hf_repo': HF_REPO,
            'phase': PHASE,
            'n_batteries': n_batteries,
            'n_total_axes': n_total,
            'D': D,
            'k_local_dim': K_LOCAL_DIM,
        },
        'probe_a_connectivity_sweep': connectivity_sweep,
        'percolation_theta_deg': percolation_theta,
        'probe_b_local_dim_summary': {
            'count_histogram': {
                str(d_int): int((all_dim_count == d_int).sum()) for d_int in range(D + 1)
            },
            'pr_min': float(all_dim_pr.min()),
            'pr_mean': float(all_dim_pr.mean()),
            'pr_median': float(np.median(all_dim_pr)),
            'pr_max': float(all_dim_pr.max()),
            'mean_eigenvalue_spectrum_normalized': mean_spectrum_normalized.tolist(),
            'per_axis_dim': [{'axis_idx': r['axis_idx'], 'bank': r['bank'],
                              'dim_count': r['local_dim_count'],
                              'dim_pr': r['local_dim_pr']}
                             for r in local_dim_data],
        },
        'probe_c_persistence': persistence_summary,
    }, f, indent=2)
print(f"\n✓ JSON: {json_path}")