# ════════════════════════════════════════════════════════════════════
# Topology atlas — h2-64 (disposable cell)
# ════════════════════════════════════════════════════════════════════
# We've established the topology:
#   - 48 infinite H1 features (persistent loops)
#   - 9 infinite H2 features (persistent voids)
#   - 138 components at percolation (θ=10°)
#   - PCA spectrum 1.0/0.49/0.20/0.003 → 2D-ish manifold in S^3
#
# Now: produce the structural atlas. For each persistent feature,
# identify which axes participate, which banks contribute, where on
# S^3 it sits, and how persistent it is.
#
# Method: ripser with do_cocycles=True extracts representative cocycles.
# A cocycle is a list of (simplex_indices, coefficient) pairs whose
# boundary is the cycle being represented. For H1, the cocycle dual
# gives us the EDGES whose removal would break the loop — equivalently,
# the axes that border the loop. For H2, the cocycle gives the
# triangles bounding the void.
#
# We also extract:
#   - Connected components at percolation θ (138 components at θ=10°)
#   - Per-feature centroid on S^3 (where it sits)
#   - Per-feature sign-quadrant (which region of S^3)
#   - Per-bank contribution map (which features each bank touches)
# ════════════════════════════════════════════════════════════════════

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    from ripser import ripser
    HAVE_RIPSER = True
except ImportError:
    print("Installing ripser...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'ripser'])
    from ripser import ripser
    HAVE_RIPSER = True

from geolip_svae.arrays import BatteryArrayModel
from geolip_svae.inference import extract_codebook, make_calibration

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CALIB = 64
SEED = 42
HF_REPO = 'AbstractPhil/geolip-svae-h2-64'
PHASE = 'final'
PERCOLATION_THETA_DEG = 10.0  # from previous probe — giant component appears here
RIPSER_THRESH_DEG = 20.0
SIGNIFICANT_PERSISTENCE_DEG = 1.0  # only feature persistence > this gets atlased

# ── Reload codebooks (defensive) ─────────────────────────────────────
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

# ── Pool axes ────────────────────────────────────────────────────────
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

# ── Distance matrix ──────────────────────────────────────────────────
print(f"Computing angular distance matrix...")
t0 = time.time()
dot = np.clip(all_axes @ all_axes.T, -1.0, 1.0)
ang_dist = np.arccos(dot)
np.fill_diagonal(ang_dist, 0.0)
print(f"  ✓ in {time.time()-t0:.1f}s")

# ── Sign-quadrant helper (re-used from earlier cells) ────────────────
def sign_quadrant_idx(vec, eps=1e-6):
    code = 0
    for k in range(1, 4):
        if vec[k] > eps:
            code |= (1 << (k - 1))
    return code

QUADRANT_LABELS = []
for q in range(8):
    s = '+'
    for k in range(3):
        s += '+' if (q >> k) & 1 else '-'
    QUADRANT_LABELS.append(s)

# ── PROBE A.1: Component decomposition at percolation ────────────────
print(f"\n{'═' * 78}")
print(f"COMPONENT DECOMPOSITION AT PERCOLATION (θ={PERCOLATION_THETA_DEG}°)")
print(f"{'═' * 78}")

theta_perc_rad = np.radians(PERCOLATION_THETA_DEG)
adj = (ang_dist <= theta_perc_rad) & (ang_dist > 0)
sparse_adj = csr_matrix(adj.astype(np.int8))
n_comp, comp_labels = connected_components(sparse_adj, directed=False)
print(f"\n  {n_comp} components at θ={PERCOLATION_THETA_DEG}°")

# Build component records
component_records = []
for c_id in range(n_comp):
    member_idx = np.where(comp_labels == c_id)[0]
    if len(member_idx) < 2:
        continue  # skip singletons in atlas
    member_axes = all_axes[member_idx]
    centroid = member_axes.mean(axis=0)
    centroid /= np.linalg.norm(centroid).clip(min=1e-12)
    member_banks = sorted(set(int(b) for b in axis_bank[member_idx]))
    quadrant = sign_quadrant_idx(centroid)
    # Extent: max angular distance from centroid
    cos_to_cent = np.clip(member_axes @ centroid, -1.0, 1.0)
    max_extent = float(np.degrees(np.arccos(cos_to_cent.min())))
    component_records.append({
        'comp_id': c_id,
        'size': int(len(member_idx)),
        'n_banks': len(member_banks),
        'banks': member_banks,
        'centroid': centroid.tolist(),
        'sign_quadrant': quadrant,
        'sign_label': QUADRANT_LABELS[quadrant],
        'max_extent_deg': max_extent,
        'member_axis_idx': member_idx.tolist(),
    })

# Sort by size descending
component_records.sort(key=lambda c: -c['size'])

n_singletons = int((np.bincount(comp_labels) == 1).sum())
print(f"  components with ≥2 axes: {len(component_records)}")
print(f"  singleton components:    {n_singletons}")
print(f"  largest component:       {component_records[0]['size']} axes "
      f"({100*component_records[0]['size']/n_total:.1f}%)")

print(f"\n  Top 15 components (by size):")
print(f"  {'comp':>5}  {'size':>5}  {'banks':>5}  {'extent°':>8}  "
      f"{'quad':>5}  centroid")
for r in component_records[:15]:
    cent_str = '[' + ', '.join(f"{v:+.2f}" for v in r['centroid']) + ']'
    print(f"  {r['comp_id']:>5}  {r['size']:>5}  {r['n_banks']:>5}  "
          f"{r['max_extent_deg']:>7.2f}°  {r['sign_label']:>5}  {cent_str}")

# Note: at percolation, one giant component dominates. The smaller
# components are the "islands" that haven't joined yet — these are
# the structurally isolated regions.
print(f"\n  Components with ≤4 axes (structural islands at θ={PERCOLATION_THETA_DEG}°):")
small_comps = [r for r in component_records if r['size'] <= 4]
print(f"    count: {len(small_comps)}")
for r in small_comps[:10]:
    cent_str = '[' + ', '.join(f"{v:+.2f}" for v in r['centroid']) + ']'
    bank_str = '[' + ','.join(str(b) for b in r['banks']) + ']'
    print(f"    comp {r['comp_id']:>3}: size={r['size']}  banks={bank_str}  "
          f"quad={r['sign_label']}  centroid={cent_str}")

# ── PROBE A.2: Persistent homology with cocycles ─────────────────────
print(f"\n{'═' * 78}")
print(f"PERSISTENT HOMOLOGY WITH REPRESENTATIVE COCYCLES")
print(f"{'═' * 78}")

print(f"\nRunning ripser (maxdim=2, thresh={RIPSER_THRESH_DEG}°, do_cocycles=True)...")
t0 = time.time()
result = ripser(
    ang_dist,
    distance_matrix=True,
    maxdim=2,
    thresh=np.radians(RIPSER_THRESH_DEG),
    do_cocycles=True,
)
print(f"  ✓ in {time.time()-t0:.1f}s")

diagrams = result['dgms']
cocycles = result['cocycles']
# cocycles[h_dim] is a list of arrays; each array is shape (n_simplices, h_dim+2)
# where each row is [simplex_vertex_0, ..., simplex_vertex_{h_dim}, coefficient]
# For H0: each row is [vertex_idx, ?]
# For H1: each row is [v0, v1, coef]   — an edge
# For H2: each row is [v0, v1, v2, coef] — a triangle

print(f"\n  H0: {len(diagrams[0])} features ({len(cocycles[0])} cocycles)")
print(f"  H1: {len(diagrams[1])} features ({len(cocycles[1])} cocycles)")
if len(diagrams) > 2:
    print(f"  H2: {len(diagrams[2])} features ({len(cocycles[2])} cocycles)")

# Helper: convert cocycle to set of vertex indices
def cocycle_vertices(cc, h_dim):
    """Extract the vertex set involved in a cocycle.
    cc shape: (n_simplices, h_dim+2)  — last column is coefficient.
    """
    if cc is None or len(cc) == 0:
        return set()
    verts = set()
    for row in cc:
        for v in row[:h_dim + 1]:
            verts.add(int(v))
    return verts

# ── Build atlas entries per dimension ────────────────────────────────
def build_features(h_dim, dgm, cocycles_h, persistence_thresh_rad):
    """Build atlas entries for features in dimension h_dim with persistence
    above the threshold."""
    features = []
    for f_idx, (b, d) in enumerate(dgm):
        # Persistence: for finite features, d-b. For infinite, use thresh-b.
        if np.isinf(d):
            persistence = np.radians(RIPSER_THRESH_DEG) - b
            is_infinite = True
        else:
            persistence = d - b
            is_infinite = False

        if persistence < persistence_thresh_rad and not is_infinite:
            continue

        # Get the cocycle representative
        if f_idx < len(cocycles_h):
            cc = cocycles_h[f_idx]
            vert_set = cocycle_vertices(cc, h_dim)
        else:
            cc = None
            vert_set = set()

        # If we have vertices, summarize them
        if vert_set:
            verts_idx = sorted(vert_set)
            verts_axes = all_axes[verts_idx]
            centroid = verts_axes.sum(axis=0)
            centroid /= np.linalg.norm(centroid).clip(min=1e-12)
            banks = sorted(set(int(axis_bank[v]) for v in verts_idx))
            # Extent
            cos_to_cent = np.clip(verts_axes @ centroid, -1.0, 1.0)
            max_extent = float(np.degrees(np.arccos(cos_to_cent.min())))
            quadrant = sign_quadrant_idx(centroid)
        else:
            verts_idx = []
            centroid = None
            banks = []
            max_extent = None
            quadrant = -1

        features.append({
            'h_dim': h_dim,
            'feature_idx': int(f_idx),
            'birth_deg': float(np.degrees(b)),
            'death_deg': float(np.degrees(d)) if not is_infinite else None,
            'persistence_deg': float(np.degrees(persistence)),
            'is_infinite': bool(is_infinite),
            'cocycle_n_simplices': int(len(cc)) if cc is not None else 0,
            'n_vertices': len(verts_idx),
            'vertices': verts_idx,
            'banks': banks,
            'n_banks': len(banks),
            'centroid': centroid.tolist() if centroid is not None else None,
            'sign_quadrant': quadrant,
            'sign_label': QUADRANT_LABELS[quadrant] if quadrant >= 0 else '?',
            'max_extent_deg': max_extent,
        })
    # Sort by persistence (infinite features first, then by persistence size)
    features.sort(
        key=lambda f: (-1 if f['is_infinite'] else 0, -f['persistence_deg'])
    )
    return features

persistence_thresh_rad = np.radians(SIGNIFICANT_PERSISTENCE_DEG)
print(f"\n  Building atlas (persistence > {SIGNIFICANT_PERSISTENCE_DEG}° threshold)...")

h0_features = build_features(0, diagrams[0], cocycles[0], persistence_thresh_rad)
h1_features = build_features(1, diagrams[1], cocycles[1], persistence_thresh_rad)
h2_features = build_features(2, diagrams[2], cocycles[2], persistence_thresh_rad) if len(diagrams) > 2 else []

print(f"    H0 atlas entries: {len(h0_features)}")
print(f"    H1 atlas entries: {len(h1_features)}")
print(f"    H2 atlas entries: {len(h2_features)}")

# ── Display H1 (loops) atlas — these are the most interesting ─────────
print(f"\n{'═' * 78}")
print(f"H1 ATLAS — PERSISTENT LOOPS")
print(f"  (1D holes = directions you can walk around in the coverage)")
print(f"{'═' * 78}")

print(f"\n  Top 20 H1 features by persistence:")
print(f"  {'rank':>4}  {'birth°':>7}  {'death°':>7}  {'persist°':>8}  "
      f"{'∞':>1}  {'verts':>5}  {'banks':>5}  {'quad':>5}  {'extent°':>7}")
for rank, f in enumerate(h1_features[:20], 1):
    death = '∞' if f['is_infinite'] else f"{f['death_deg']:>7.2f}"
    inf_mark = '∞' if f['is_infinite'] else ' '
    extent = f"{f['max_extent_deg']:>6.2f}°" if f['max_extent_deg'] else '   ?  '
    print(f"  {rank:>4}  {f['birth_deg']:>7.2f}  {death:>7}  "
          f"{f['persistence_deg']:>7.2f}°  {inf_mark:>1}  "
          f"{f['n_vertices']:>5}  {f['n_banks']:>5}  "
          f"{f['sign_label']:>5}  {extent:>7}")

# Per-quadrant tally of H1 features
print(f"\n  H1 features by sign-quadrant:")
quad_h1 = defaultdict(int)
for f in h1_features:
    if f['sign_quadrant'] >= 0:
        quad_h1[f['sign_quadrant']] += 1
for q in range(8):
    print(f"    q={q} ({QUADRANT_LABELS[q]}): {quad_h1[q]} loops")

# ── Display H2 (voids) atlas ─────────────────────────────────────────
print(f"\n{'═' * 78}")
print(f"H2 ATLAS — PERSISTENT VOIDS")
print(f"  (2D enclosed regions = holes in the coverage of S^3)")
print(f"{'═' * 78}")

print(f"\n  All {len(h2_features)} H2 features (with significant persistence):")
print(f"  {'rank':>4}  {'birth°':>7}  {'death°':>7}  {'persist°':>8}  "
      f"{'∞':>1}  {'verts':>5}  {'banks':>5}  {'quad':>5}  {'extent°':>7}")
for rank, f in enumerate(h2_features, 1):
    death = '∞' if f['is_infinite'] else f"{f['death_deg']:>7.2f}"
    inf_mark = '∞' if f['is_infinite'] else ' '
    extent = f"{f['max_extent_deg']:>6.2f}°" if f['max_extent_deg'] else '   ?  '
    print(f"  {rank:>4}  {f['birth_deg']:>7.2f}  {death:>7}  "
          f"{f['persistence_deg']:>7.2f}°  {inf_mark:>1}  "
          f"{f['n_vertices']:>5}  {f['n_banks']:>5}  "
          f"{f['sign_label']:>5}  {extent:>7}")

# Per-quadrant H2
print(f"\n  H2 features by sign-quadrant:")
quad_h2 = defaultdict(int)
for f in h2_features:
    if f['sign_quadrant'] >= 0:
        quad_h2[f['sign_quadrant']] += 1
for q in range(8):
    if quad_h2[q]:
        print(f"    q={q} ({QUADRANT_LABELS[q]}): {quad_h2[q]} voids")

# ── Per-bank contribution map ────────────────────────────────────────
print(f"\n{'═' * 78}")
print(f"PER-BANK FEATURE CONTRIBUTION MAP")
print(f"  Which features does each bank participate in?")
print(f"{'═' * 78}")

# For each bank, count contributions to H1 and H2
bank_contributions = {b: {'h1': 0, 'h2': 0, 'h1_inf': 0, 'h2_inf': 0,
                           'h1_features': [], 'h2_features': []}
                       for b in range(n_batteries)}
for f_idx, f in enumerate(h1_features):
    for b in f['banks']:
        bank_contributions[b]['h1'] += 1
        if f['is_infinite']:
            bank_contributions[b]['h1_inf'] += 1
        bank_contributions[b]['h1_features'].append(f_idx)
for f_idx, f in enumerate(h2_features):
    for b in f['banks']:
        bank_contributions[b]['h2'] += 1
        if f['is_infinite']:
            bank_contributions[b]['h2_inf'] += 1
        bank_contributions[b]['h2_features'].append(f_idx)

# Top 15 banks by total contribution count
sorted_banks = sorted(
    bank_contributions.items(),
    key=lambda kv: -(kv[1]['h1'] + kv[1]['h2'])
)
print(f"\n  Top 15 banks by total feature-participation count:")
print(f"  {'rank':>4}  {'bank':>4}  {'h1':>3}  {'h1∞':>4}  "
      f"{'h2':>3}  {'h2∞':>4}  {'total':>5}")
for rank, (b, contr) in enumerate(sorted_banks[:15], 1):
    total = contr['h1'] + contr['h2']
    print(f"  {rank:>4}  {b:>4}  {contr['h1']:>3}  {contr['h1_inf']:>4}  "
          f"{contr['h2']:>3}  {contr['h2_inf']:>4}  {total:>5}")

# Which banks DON'T participate in any infinite feature?
no_inf_banks = sorted([b for b, c in bank_contributions.items()
                        if c['h1_inf'] == 0 and c['h2_inf'] == 0])
print(f"\n  Banks NOT contributing to any infinite (persistent-to-end) feature:")
print(f"    count: {len(no_inf_banks)}")
print(f"    banks: {no_inf_banks}")

# Cross-reference with previous outliers
cosine_top5 = [42, 52, 40, 7, 58]
knn_top5 = [9, 10, 51, 25, 48]
print(f"\n  Reference outliers from previous probes:")
print(f"    cosine top-5: {cosine_top5}")
print(f"    kNN top-5:    {knn_top5}")
print(f"\n  Their feature-participation:")
print(f"  {'bank':>4}  {'h1':>3}  {'h1∞':>4}  {'h2':>3}  {'h2∞':>4}  {'total':>5}  source")
for b in cosine_top5:
    c = bank_contributions[b]
    print(f"  {b:>4}  {c['h1']:>3}  {c['h1_inf']:>4}  "
          f"{c['h2']:>3}  {c['h2_inf']:>4}  "
          f"{c['h1']+c['h2']:>5}  cosine top-5")
for b in knn_top5:
    c = bank_contributions[b]
    print(f"  {b:>4}  {c['h1']:>3}  {c['h1_inf']:>4}  "
          f"{c['h2']:>3}  {c['h2_inf']:>4}  "
          f"{c['h1']+c['h2']:>5}  kNN top-5")

# ── Persist atlas ────────────────────────────────────────────────────
out_dir = Path('/content')
out_dir.mkdir(exist_ok=True)
json_path = out_dir / 'h2_64_topology_atlas.json'
with open(json_path, 'w') as f_out:
    json.dump({
        'config': {
            'hf_repo': HF_REPO,
            'phase': PHASE,
            'n_batteries': n_batteries,
            'n_total_axes': n_total,
            'D': D,
            'percolation_theta_deg': PERCOLATION_THETA_DEG,
            'ripser_thresh_deg': RIPSER_THRESH_DEG,
            'significant_persistence_deg': SIGNIFICANT_PERSISTENCE_DEG,
        },
        'axis_bank': axis_bank.tolist(),
        'components_at_percolation': component_records,
        'h0_atlas': h0_features,
        'h1_atlas': h1_features,
        'h2_atlas': h2_features,
        'per_bank_contributions': {
            str(b): {
                'h1_count': c['h1'],
                'h1_infinite_count': c['h1_inf'],
                'h2_count': c['h2'],
                'h2_infinite_count': c['h2_inf'],
                'h1_feature_idx': c['h1_features'],
                'h2_feature_idx': c['h2_features'],
            }
            for b, c in bank_contributions.items()
        },
        'banks_without_infinite_features': no_inf_banks,
    }, f_out, indent=2)
print(f"\n✓ Atlas JSON: {json_path}")

# ── Synthesis ────────────────────────────────────────────────────────
print(f"\n{'═' * 78}")
print(f"ATLAS SYNTHESIS")
print(f"{'═' * 78}")

n_h1_inf = sum(1 for f in h1_features if f['is_infinite'])
n_h2_inf = sum(1 for f in h2_features if f['is_infinite'])

print(f"""
The topology of the h2-64 axis cloud has:
  - {n_comp} connected components at percolation θ={PERCOLATION_THETA_DEG}°
    ({len([c for c in component_records if c['size'] <= 4])} of them are
     small islands of ≤4 axes, structurally isolated)
  - {n_h1_inf} infinite H1 features (loops that never close in the filtration)
  - {n_h2_inf} infinite H2 features (voids that persist to the end)
  - {len(h1_features)} significant H1 features (persistence > {SIGNIFICANT_PERSISTENCE_DEG}°)
  - {len(h2_features)} significant H2 features (persistence > {SIGNIFICANT_PERSISTENCE_DEG}°)

Each feature in the atlas is identified by:
  - Its participating axes (vertices in the cocycle representative)
  - Which banks contributed those axes
  - Where it sits on S^3 (centroid + sign-quadrant)
  - Its extent and persistence

Per-bank participation map shows which banks are "structural carriers"
of the topology vs which banks live in the bulk. {len(no_inf_banks)} banks
participate in zero infinite features — these are the banks whose axes
are entirely interior to the structure, not on its boundaries.
""")