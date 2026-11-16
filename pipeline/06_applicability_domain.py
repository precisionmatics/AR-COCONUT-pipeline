"""
Step 6: Applicability Domain Analysis — P10275 (Androgen Receptor)
BindingDB May 2026 | Prostate Cancer Drug Discovery ML Study

Methods:
  1. Bounding Box (descriptor ranges)
  2. Leverage / Hat matrix (Williams plot)
  3. k-NN distance-based AD (Euclidean in PCA space)
  4. Tanimoto similarity-based AD (Morgan FP)
  5. Reliability plots (prediction confidence vs error)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, logging, os, sys, json, joblib
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE    = '/home/stalin/Downloads/BindingDB_P10275'
PROC    = f'{BASE}/data/processed'
MDIR    = f'{BASE}/models'
FIGS_AD = f'{BASE}/figures/applicability_domain'
LOG_D   = f'{BASE}/logs'
RDIR    = f'{BASE}/results'
os.makedirs(FIGS_AD, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_D}/06_applicability_domain.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("Step 6: Applicability Domain — P10275 Androgen Receptor")
log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("=" * 70)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[1] Loading data...")
df        = pd.read_csv(f'{PROC}/P10275_model_ready.csv')
y         = np.load(f'{PROC}/y.npy')
X_desc    = np.load(f'{PROC}/X_desc.npy')        # scaled 153 descriptors
X_enh_fp  = np.load(f'{PROC}/X_enhanced_fp.npy') # 4263 enhanced FP
idx_train = np.load(f'{PROC}/idx_train.npy')
idx_test  = np.load(f'{PROC}/idx_test.npy')

with open(f'{PROC}/feature_names.json') as f:
    feat_info = json.load(f)
desc_cols = feat_info['desc_cols']

# Re-align enhanced FP to valid subset
def get_fps(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return (list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)) +
                list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)) +
                list(MACCSkeys.GenMACCSKeys(mol)))
    except: return None

fps_check   = df['Canonical_SMILES'].map(get_fps)
valid       = fps_check.notna()
idx_aligned = np.where(valid.values)[0]
valid_set   = set(idx_aligned)
new_pos     = {old: new for new, old in enumerate(idx_aligned)}
idx_tr_fp   = np.array([new_pos[i] for i in idx_train if i in valid_set])
idx_te_fp   = np.array([new_pos[i] for i in idx_test  if i in valid_set])
y_fp        = y[valid.values]
ytr         = y_fp[idx_tr_fp];   yte = y_fp[idx_te_fp]
Xtr_fp      = X_enh_fp[idx_tr_fp]; Xte_fp = X_enh_fp[idx_te_fp]
Xtr_d       = X_desc[idx_train];    Xte_d  = X_desc[idx_test]

# Load best model (XGBoost tuned)
best_xgb = joblib.load(f'{MDIR}/XGBoost_tuned.pkl')
y_pred   = best_xgb.predict(Xte_fp)
residuals = y_pred - yte

log.info(f"  Train: {len(ytr)}  Test: {len(yte)}")
log.info(f"  Best model predictions loaded (XGBoost tuned)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. PCA for AD space
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[2] PCA for AD chemical space...")
pca = PCA(n_components=20, random_state=42)
Xtr_pca = pca.fit_transform(Xtr_d)
Xte_pca = pca.transform(Xte_d)
log.info(f"  20 PCs explain {pca.explained_variance_ratio_.sum()*100:.1f}% variance")

# ══════════════════════════════════════════════════════════════════════════════
# 3. METHOD 1: LEVERAGE / WILLIAMS PLOT
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[3] Leverage (Hat matrix) — Williams plot...")

# Use descriptor space (scaled)
X_tr_hat = Xtr_d
X_te_hat = Xte_d

# Hat matrix: h = x(X'X)^-1 x'
# For high-dim: use pseudo-inverse
XtX     = X_tr_hat.T @ X_tr_hat
XtX_inv = np.linalg.pinv(XtX)

h_train = np.array([x @ XtX_inv @ x for x in X_tr_hat])
h_test  = np.array([x @ XtX_inv @ x for x in X_te_hat])

# Warning leverage threshold: h* = 3(p+1)/n
p = X_tr_hat.shape[1]
n = X_tr_hat.shape[0]
h_star = 3 * (p + 1) / n

# Standardised residuals for training (needed for Williams plot)
y_pred_tr = best_xgb.predict(Xtr_fp)
res_tr    = y_pred_tr - ytr
rmse_tr   = np.sqrt(mean_squared_error(ytr, y_pred_tr))
std_res_tr = res_tr / rmse_tr

std_res_te = residuals / np.sqrt(mean_squared_error(yte, y_pred))

n_high_leverage = (h_test > h_star).sum()
n_high_residual = (np.abs(std_res_te) > 3).sum()
n_outlier       = ((h_test > h_star) & (np.abs(std_res_te) > 3)).sum()

log.info(f"  h* threshold: {h_star:.4f}")
log.info(f"  Test points with h > h*: {n_high_leverage} ({n_high_leverage/len(yte)*100:.1f}%)")
log.info(f"  Test points |std_res| > 3: {n_high_residual}")
log.info(f"  Outliers (both): {n_outlier}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. METHOD 2: k-NN DISTANCE-BASED AD (PCA space)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[4] k-NN distance-based AD (k=5, PCA space)...")
k = 5

# Mean k-NN distance in training set
dist_tr = cdist(Xtr_pca, Xtr_pca, metric='euclidean')
np.fill_diagonal(dist_tr, np.inf)
knn_dist_tr = np.sort(dist_tr, axis=1)[:, :k].mean(axis=1)

# AD threshold: mean + z*std of training kNN distances
z = 1.5
ad_threshold = knn_dist_tr.mean() + z * knn_dist_tr.std()

# Test kNN distances
dist_te = cdist(Xte_pca, Xtr_pca, metric='euclidean')
knn_dist_te = np.sort(dist_te, axis=1)[:, :k].mean(axis=1)

in_ad    = knn_dist_te <= ad_threshold
out_ad   = ~in_ad
n_in_ad  = in_ad.sum()
n_out_ad = out_ad.sum()

log.info(f"  AD threshold (mean+{z}σ): {ad_threshold:.4f}")
log.info(f"  Test in AD:  {n_in_ad} ({n_in_ad/len(yte)*100:.1f}%)")
log.info(f"  Test out AD: {n_out_ad} ({n_out_ad/len(yte)*100:.1f}%)")

# Performance inside vs outside AD
r2_in  = r2_score(yte[in_ad],  y_pred[in_ad])  if in_ad.sum() > 1 else np.nan
r2_out = r2_score(yte[out_ad], y_pred[out_ad]) if out_ad.sum() > 1 else np.nan
rmse_in  = np.sqrt(mean_squared_error(yte[in_ad],  y_pred[in_ad]))  if in_ad.sum() > 1 else np.nan
rmse_out = np.sqrt(mean_squared_error(yte[out_ad], y_pred[out_ad])) if out_ad.sum() > 1 else np.nan

log.info(f"  R² inside AD:  {r2_in:.4f}   RMSE: {rmse_in:.4f}")
log.info(f"  R² outside AD: {r2_out:.4f}   RMSE: {rmse_out:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. METHOD 3: TANIMOTO SIMILARITY-BASED AD
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[5] Tanimoto similarity-based AD...")

# Compute ECFP4 RDKit fingerprint objects for Tanimoto
df_valid = df[valid].reset_index(drop=True)

def get_ecfp4(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
    except: return None

fps_rdkit = df_valid['Canonical_SMILES'].map(get_ecfp4)
fps_tr    = [fps_rdkit.iloc[i] for i in idx_tr_fp if fps_rdkit.iloc[i] is not None]
fps_te    = [fps_rdkit.iloc[i] for i in idx_te_fp if fps_rdkit.iloc[i] is not None]
n_te_tan  = len(fps_te)

# Max Tanimoto similarity of each test compound to training set
log.info(f"  Computing Tanimoto similarities for {n_te_tan} test compounds...")
max_sim = np.array([
    max(DataStructs.BulkTanimotoSimilarity(fp, fps_tr))
    for fp in fps_te
])

# AD threshold: compounds with max Tanimoto < threshold considered outside AD
sim_threshold = 0.4
in_ad_tan  = max_sim >= sim_threshold
n_in_tan   = in_ad_tan.sum()

log.info(f"  Tanimoto threshold: {sim_threshold}")
log.info(f"  In AD (Tanimoto): {n_in_tan}/{n_te_tan} ({n_in_tan/n_te_tan*100:.1f}%)")
log.info(f"  Mean max Tanimoto: {max_sim.mean():.4f} ± {max_sim.std():.4f}")

# Performance by similarity band
bands = [(0.0,0.3,'Low (<0.3)'), (0.3,0.5,'Medium (0.3–0.5)'),
         (0.5,0.7,'High (0.5–0.7)'), (0.7,1.01,'Very High (>0.7)')]
log.info("  Performance by Tanimoto band:")
for lo, hi, label in bands:
    mask = (max_sim >= lo) & (max_sim < hi)
    if mask.sum() > 2:
        r2_b   = r2_score(yte[:n_te_tan][mask], y_pred[:n_te_tan][mask])
        rmse_b = np.sqrt(mean_squared_error(yte[:n_te_tan][mask], y_pred[:n_te_tan][mask]))
        log.info(f"    {label:<22}  n={mask.sum():3d}  R²={r2_b:.4f}  RMSE={rmse_b:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[6] Generating AD figures...")

# ── Fig 01: Williams Plot (leverage vs standardised residuals) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Williams Plot — Applicability Domain | P10275 Androgen Receptor\n'
             'XGBoost (tuned) | Descriptor space', fontsize=13, fontweight='bold')

for ax, h_vals, std_res, y_true, label, col in [
    (axes[0], h_train, std_res_tr, ytr, 'Training Set', '#1976D2'),
    (axes[1], h_test,  std_res_te, yte, 'Test Set',     '#D32F2F'),
]:
    sc = ax.scatter(h_vals, std_res, c=y_true, cmap='RdYlGn',
                    alpha=0.5, s=25, linewidths=0, vmin=4, vmax=10)
    plt.colorbar(sc, ax=ax, label='pIC50')
    ax.axhline( 3, color='crimson', ls='--', lw=1.5, label='±3σ')
    ax.axhline(-3, color='crimson', ls='--', lw=1.5)
    ax.axvline(h_star, color='orange', ls='--', lw=1.5, label=f'h*={h_star:.3f}')
    ax.set_xlabel('Leverage (h)')
    ax.set_ylabel('Standardised Residual')
    ax.set_title(f'Williams Plot — {label}')
    ax.legend(fontsize=9)

    # Count zones
    in_zone  = ((h_vals <= h_star) & (np.abs(std_res) <= 3)).sum()
    out_lev  = (h_vals > h_star).sum()
    out_res  = (np.abs(std_res) > 3).sum()
    ax.text(0.02, 0.98,
            f'In AD: {in_zone} ({in_zone/len(h_vals)*100:.1f}%)\n'
            f'High leverage: {out_lev}\nHigh residual: {out_res}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
fig.savefig(f'{FIGS_AD}/01_williams_plot.png')
plt.close()
log.info("  Saved: 01_williams_plot.png")

# ── Fig 02: kNN distance AD ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('k-NN Distance-Based Applicability Domain (k=5, PCA space)\n'
             'P10275 Androgen Receptor | XGBoost (tuned)', fontsize=13, fontweight='bold')

ax = axes[0]
ax.hist(knn_dist_tr, bins=40, color='#1976D2', alpha=0.7, edgecolor='white',
        linewidth=0.4, label='Training')
ax.hist(knn_dist_te, bins=40, color='#D32F2F', alpha=0.6, edgecolor='white',
        linewidth=0.4, label='Test')
ax.axvline(ad_threshold, color='orange', ls='--', lw=2,
           label=f'AD threshold={ad_threshold:.2f}')
ax.set_xlabel('Mean k-NN Distance (PCA space)')
ax.set_ylabel('Count')
ax.set_title('Distance Distribution')
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(knn_dist_te[in_ad],  yte[in_ad],  s=20, alpha=0.6,
           color='#1976D2', label=f'In AD (n={in_ad.sum()})')
ax.scatter(knn_dist_te[out_ad], yte[out_ad], s=20, alpha=0.6,
           color='#D32F2F', label=f'Out AD (n={out_ad.sum()})', marker='^')
ax.axvline(ad_threshold, color='orange', ls='--', lw=1.5)
ax.set_xlabel('Mean k-NN Distance')
ax.set_ylabel('Actual pIC50')
ax.set_title('Distance vs Actual pIC50')
ax.legend(fontsize=9)

ax = axes[2]
ax.scatter(knn_dist_te[in_ad],  np.abs(residuals[in_ad]),  s=20, alpha=0.6,
           color='#1976D2', label=f'In AD  R²={r2_in:.3f}')
ax.scatter(knn_dist_te[out_ad], np.abs(residuals[out_ad]), s=20, alpha=0.6,
           color='#D32F2F', label=f'Out AD R²={r2_out:.3f}', marker='^')
ax.axvline(ad_threshold, color='orange', ls='--', lw=1.5, label='AD threshold')
ax.set_xlabel('Mean k-NN Distance')
ax.set_ylabel('|Prediction Error|')
ax.set_title('Distance vs Prediction Error')
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(f'{FIGS_AD}/02_knn_distance_ad.png')
plt.close()
log.info("  Saved: 02_knn_distance_ad.png")

# ── Fig 03: Tanimoto similarity AD ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Tanimoto Similarity-Based Applicability Domain\n'
             'ECFP4 | P10275 Androgen Receptor', fontsize=13, fontweight='bold')

ax = axes[0]
ax.hist(max_sim, bins=40, color='#388E3C', alpha=0.8, edgecolor='white', linewidth=0.4)
ax.axvline(sim_threshold, color='crimson', ls='--', lw=2,
           label=f'Threshold={sim_threshold}')
ax.set_xlabel('Max Tanimoto Similarity to Training Set')
ax.set_ylabel('Count')
ax.set_title('Max Tanimoto Distribution')
ax.legend(fontsize=10)
ax.text(0.02, 0.97, f'Mean={max_sim.mean():.3f}\nStd={max_sim.std():.3f}',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

ax = axes[1]
sc = ax.scatter(max_sim, yte[:n_te_tan], c=np.abs(residuals[:n_te_tan]),
                cmap='coolwarm', alpha=0.6, s=25, linewidths=0, vmin=0, vmax=2)
plt.colorbar(sc, ax=ax, label='|Residual|')
ax.axvline(sim_threshold, color='crimson', ls='--', lw=1.5)
ax.set_xlabel('Max Tanimoto Similarity')
ax.set_ylabel('Actual pIC50')
ax.set_title('Tanimoto vs pIC50\n(colour = |error|)')

ax = axes[2]
band_labels, band_r2, band_rmse, band_n = [], [], [], []
for lo, hi, label in bands:
    mask = (max_sim >= lo) & (max_sim < hi)
    if mask.sum() > 2:
        band_labels.append(label)
        band_n.append(mask.sum())
        band_r2.append(r2_score(yte[:n_te_tan][mask], y_pred[:n_te_tan][mask]))
        band_rmse.append(np.sqrt(mean_squared_error(yte[:n_te_tan][mask],
                                                     y_pred[:n_te_tan][mask])))

x_pos = range(len(band_labels))
bars = ax.bar(x_pos, band_r2, color=['#EF5350','#FF9800','#66BB6A','#1976D2'][:len(band_labels)],
              edgecolor='white', alpha=0.85)
for bar, val, n in zip(bars, band_r2, band_n):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'R²={val:.3f}\nn={n}', ha='center', fontsize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels(band_labels, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Test R²')
ax.set_title('Prediction Performance\nby Tanimoto Similarity Band')
ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig(f'{FIGS_AD}/03_tanimoto_ad.png')
plt.close()
log.info("  Saved: 03_tanimoto_ad.png")

# ── Fig 04: Chemical space (PCA) coloured by AD membership ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Chemical Space — Applicability Domain Visualisation (PCA)\n'
             'P10275 Androgen Receptor', fontsize=13, fontweight='bold')

ax = axes[0]
ax.scatter(Xtr_pca[:, 0], Xtr_pca[:, 1], s=15, alpha=0.35,
           color='#90CAF9', label=f'Training (n={len(Xtr_pca)})', linewidths=0)
ax.scatter(Xte_pca[in_ad, 0],  Xte_pca[in_ad, 1],  s=30, alpha=0.7,
           color='#1976D2', label=f'Test — In AD (n={in_ad.sum()})', linewidths=0)
ax.scatter(Xte_pca[out_ad, 0], Xte_pca[out_ad, 1], s=35, alpha=0.8,
           color='#D32F2F', marker='^', label=f'Test — Out AD (n={out_ad.sum()})', linewidths=0)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Chemical Space — AD Membership (k-NN)')
ax.legend(fontsize=9)

ax = axes[1]
sc = ax.scatter(Xte_pca[:, 0], Xte_pca[:, 1], c=np.abs(residuals),
                cmap='coolwarm', s=30, alpha=0.7, linewidths=0, vmin=0, vmax=2)
plt.colorbar(sc, ax=ax, label='|Prediction Error|')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Test Set — Prediction Error in Chemical Space')

plt.tight_layout()
fig.savefig(f'{FIGS_AD}/04_chemical_space_ad.png')
plt.close()
log.info("  Saved: 04_chemical_space_ad.png")

# ── Fig 05: Reliability plot (prediction confidence vs accuracy) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prediction Reliability — Distance vs Error | P10275 Androgen Receptor',
             fontsize=13, fontweight='bold')

ax = axes[0]
# Bin by kNN distance and show RMSE per bin
n_bins = 8
bin_edges = np.percentile(knn_dist_te, np.linspace(0, 100, n_bins+1))
bin_rmse, bin_r2, bin_centres, bin_ns = [], [], [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (knn_dist_te >= lo) & (knn_dist_te <= hi)
    if mask.sum() > 2:
        bin_rmse.append(np.sqrt(mean_squared_error(yte[mask], y_pred[mask])))
        bin_r2.append(r2_score(yte[mask], y_pred[mask]))
        bin_centres.append((lo+hi)/2)
        bin_ns.append(mask.sum())

ax.plot(bin_centres, bin_rmse, 'o-', color='#D32F2F', lw=2, ms=8)
ax2 = ax.twinx()
ax2.plot(bin_centres, bin_r2, 's--', color='#1976D2', lw=1.5, ms=7)
ax2.set_ylabel('R²', color='#1976D2')
ax.set_xlabel('Mean k-NN Distance (binned)')
ax.set_ylabel('RMSE', color='#D32F2F')
ax.set_title('RMSE & R² vs Distance from Training\n(higher distance = lower confidence)')
for c, n in zip(bin_centres, bin_ns):
    ax.text(c, ax.get_ylim()[0], f'n={n}', ha='center', fontsize=7, color='gray')

ax = axes[1]
# Bin by Tanimoto similarity
sim_bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
sim_rmse, sim_r2, sim_labels, sim_ns = [], [], [], []
for lo, hi in zip(sim_bins[:-1], sim_bins[1:]):
    mask = (max_sim >= lo) & (max_sim < hi)
    if mask.sum() > 2:
        sim_rmse.append(np.sqrt(mean_squared_error(yte[:n_te_tan][mask],
                                                    y_pred[:n_te_tan][mask])))
        sim_r2.append(r2_score(yte[:n_te_tan][mask], y_pred[:n_te_tan][mask]))
        sim_labels.append(f'{lo:.1f}–{hi:.1f}')
        sim_ns.append(mask.sum())

x = range(len(sim_labels))
ax.bar(x, sim_rmse, color='#FF9800', alpha=0.8, edgecolor='white', width=0.4, label='RMSE')
ax2b = ax.twinx()
ax2b.plot([i+0.2 for i in x], sim_r2, 'D-', color='#1976D2', ms=8, lw=1.5, label='R²')
ax2b.set_ylabel('R²', color='#1976D2')
ax.set_xticks(x)
ax.set_xticklabels(sim_labels, fontsize=9)
ax.set_xlabel('Max Tanimoto Similarity Band')
ax.set_ylabel('RMSE', color='#FF9800')
ax.set_title('Prediction Quality vs\nTanimoto Similarity Band')
for xi, n in zip(x, sim_ns):
    ax.text(xi, 0.01, f'n={n}', ha='center', fontsize=7, color='gray')

plt.tight_layout()
fig.savefig(f'{FIGS_AD}/05_reliability_plot.png')
plt.close()
log.info("  Saved: 05_reliability_plot.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE AD RESULTS
# ══════════════════════════════════════════════════════════════════════════════
ad_results = {
    'leverage': {
        'h_star':            round(float(h_star), 6),
        'n_high_leverage':   int(n_high_leverage),
        'n_high_residual':   int(n_high_residual),
        'n_outliers':        int(n_outlier),
        'pct_in_ad':         round((len(yte)-n_high_leverage)/len(yte)*100, 2),
    },
    'knn_distance': {
        'k':                 k,
        'z_threshold':       z,
        'ad_threshold':      round(float(ad_threshold), 6),
        'n_in_ad':           int(n_in_ad),
        'n_out_ad':          int(n_out_ad),
        'pct_in_ad':         round(n_in_ad/len(yte)*100, 2),
        'R2_in_ad':          round(float(r2_in), 4),
        'R2_out_ad':         round(float(r2_out), 4),
        'RMSE_in_ad':        round(float(rmse_in), 4),
        'RMSE_out_ad':       round(float(rmse_out), 4),
    },
    'tanimoto': {
        'threshold':         sim_threshold,
        'n_in_ad':           int(n_in_tan),
        'n_test':            n_te_tan,
        'pct_in_ad':         round(n_in_tan/n_te_tan*100, 2),
        'mean_max_sim':      round(float(max_sim.mean()), 4),
        'std_max_sim':       round(float(max_sim.std()), 4),
    }
}

with open(f'{RDIR}/applicability_domain.json', 'w') as f:
    json.dump(ad_results, f, indent=2)

log.info("\n" + "=" * 70)
log.info("STEP 6 COMPLETE — APPLICABILITY DOMAIN")
log.info(f"  Leverage AD:   {(len(yte)-n_high_leverage)/len(yte)*100:.1f}% of test in AD")
log.info(f"  kNN AD:        {n_in_ad/len(yte)*100:.1f}% in AD | R²(in)={r2_in:.4f} vs R²(out)={r2_out:.4f}")
log.info(f"  Tanimoto AD:   {n_in_tan/n_te_tan*100:.1f}% in AD (threshold={sim_threshold})")
log.info(f"  Mean Tanimoto: {max_sim.mean():.4f} ± {max_sim.std():.4f}")
log.info(f"  Figures → {FIGS_AD}/")
log.info("=" * 70)
