"""
Step 2: Feature Engineering — Morgan Fingerprints + RDKit 2D Descriptors
P10275 (Androgen Receptor) | Prostate Cancer Drug Discovery ML Study
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import warnings, logging, os, sys, json, joblib
from datetime import datetime
from scipy.stats import pearsonr

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE  = '/home/stalin/Downloads/BindingDB_P10275'
PROC  = f'{BASE}/data/processed'
FIGS  = f'{BASE}/figures/descriptive'
LOG_D = f'{BASE}/logs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_D}/02_feature_engineering.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("Step 2: Feature Engineering — P10275 Androgen Receptor")
log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("=" * 70)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[1] Loading processed data...")
df = pd.read_csv(f'{PROC}/P10275_processed.csv')
log.info(f"  Compounds: {len(df):,}  |  pIC50 range: {df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}")
y = df['pIC50'].values.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# 2. MORGAN FINGERPRINTS  (ECFP4: radius=2, 2048 bits)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[2] Computing Morgan fingerprints (ECFP4, r=2, 2048 bits)...")

def morgan_fp(smi, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    except:
        return None

fps = df['Canonical_SMILES'].map(morgan_fp)
valid = fps.notna()
n_invalid = (~valid).sum()
if n_invalid:
    log.info(f"  Dropped {n_invalid} compounds with failed FP computation")
df  = df[valid].reset_index(drop=True)
fps = fps[valid].reset_index(drop=True)
y   = y[valid.values]

fp_array = np.array(fps.tolist(), dtype=np.uint8)
fp_cols  = [f'Morgan_{i}' for i in range(fp_array.shape[1])]
log.info(f"  Fingerprint matrix: {fp_array.shape}")
log.info(f"  Active bits (any compound): {(fp_array.sum(axis=0) > 0).sum()} / {fp_array.shape[1]}")
log.info(f"  Mean bit density: {fp_array.mean()*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3. RDKIT 2D DESCRIPTORS
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[3] Computing RDKit 2D descriptors...")

# All available RDKit descriptors
all_desc = [n for n, _ in Descriptors._descList]
log.info(f"  Total RDKit descriptors available: {len(all_desc)}")

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(all_desc)

def calc_desc(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [np.nan] * len(all_desc)
        return list(calculator.CalcDescriptors(mol))
    except:
        return [np.nan] * len(all_desc)

desc_raw = pd.DataFrame(
    df['Canonical_SMILES'].map(calc_desc).tolist(),
    columns=all_desc
)
log.info(f"  Raw descriptor matrix: {desc_raw.shape}")

# Remove descriptors with >10% missing
miss_frac = desc_raw.isna().mean()
keep1 = miss_frac[miss_frac < 0.10].index.tolist()
desc_raw = desc_raw[keep1]
log.info(f"  After missing filter (<10%): {len(keep1)} descriptors")

# Fill remaining NaN with column median
desc_raw = desc_raw.fillna(desc_raw.median())

# Remove infinite values
desc_raw = desc_raw.replace([np.inf, -np.inf], np.nan).fillna(desc_raw.median())

# Remove near-zero variance
vt = VarianceThreshold(threshold=0.01)
desc_arr = vt.fit_transform(desc_raw.values)
keep2 = np.array(keep1)[vt.get_support()]
desc_df = pd.DataFrame(desc_arr, columns=keep2)
log.info(f"  After variance filter (>0.01): {len(keep2)} descriptors")
joblib.dump(vt, f'{PROC}/variance_selector.pkl')

# Remove highly correlated descriptors (r > 0.95)
log.info("  Removing highly correlated descriptors (|r| > 0.95)...")
corr_matrix = pd.DataFrame(desc_arr, columns=keep2).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
desc_df = desc_df.drop(columns=to_drop)
keep3 = [c for c in keep2 if c not in to_drop]
log.info(f"  After correlation filter: {len(keep3)} descriptors  (dropped {len(to_drop)})")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCALE & SAVE FEATURE MATRICES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[4] Scaling and saving feature matrices...")

X_morgan = fp_array.astype(np.float32)

scaler_desc = StandardScaler()
X_desc = scaler_desc.fit_transform(desc_df.values).astype(np.float32)
joblib.dump(scaler_desc, f'{PROC}/scaler_desc.pkl')

X_combined = np.hstack([X_morgan, X_desc]).astype(np.float32)
scaler_comb = StandardScaler()
X_combined_scaled = scaler_comb.fit_transform(X_combined).astype(np.float32)
joblib.dump(scaler_comb, f'{PROC}/scaler_combined.pkl')

np.save(f'{PROC}/X_morgan.npy',   X_morgan)
np.save(f'{PROC}/X_desc.npy',     X_desc)
np.save(f'{PROC}/X_combined.npy', X_combined_scaled)
np.save(f'{PROC}/y.npy',          y)

with open(f'{PROC}/feature_names.json', 'w') as f:
    json.dump({
        'morgan_cols':  fp_cols,
        'desc_cols':    list(keep3),
        'n_morgan':     len(fp_cols),
        'n_desc':       len(keep3),
        'n_combined':   len(fp_cols) + len(keep3),
        'n_compounds':  int(len(y)),
    }, f, indent=2)

log.info(f"  X_morgan:          {X_morgan.shape}")
log.info(f"  X_desc (scaled):   {X_desc.shape}")
log.info(f"  X_combined:        {X_combined_scaled.shape}")
log.info(f"  y (pIC50):         {y.shape}")

# Also save the full index-aligned dataframe for later use
df['array_idx'] = range(len(df))
df.to_csv(f'{PROC}/P10275_model_ready.csv', index=False)
log.info(f"  Model-ready CSV:   {PROC}/P10275_model_ready.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 5. PCA FOR CHEMICAL SPACE VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[5] PCA on descriptors for visualisation...")
pca = PCA(n_components=min(50, X_desc.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_desc)
joblib.dump(pca, f'{PROC}/pca_model.pkl')
cum_var = np.cumsum(pca.explained_variance_ratio_)
n90 = int(np.searchsorted(cum_var, 0.90)) + 1
log.info(f"  PCs to explain 90% variance: {n90}")
log.info(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[6] Generating figures...")

# ── Fig 09: Descriptor pipeline summary (bar) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
stages  = ['All RDKit\ndescriptors', 'After missing\nfilter (<10%)',
           'After variance\nfilter', 'After correlation\nfilter\n(final)']
counts  = [len(all_desc), len(keep1), len(keep2), len(keep3)]
colors  = ['#90CAF9', '#42A5F5', '#1E88E5', '#1565C0']
bars = ax.bar(stages, counts, color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Descriptors')
ax.set_title('RDKit Descriptor Filtering Pipeline — P10275', fontweight='bold')
ax.set_ylim(0, max(counts) * 1.12)
plt.tight_layout()
fig.savefig(f'{FIGS}/09_descriptor_filtering.png')
plt.close()
log.info("  Saved: 09_descriptor_filtering.png")

# ── Fig 10: Morgan FP bit frequency ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Morgan Fingerprint (ECFP4) Analysis — P10275', fontsize=13, fontweight='bold')

bit_freq = X_morgan.mean(axis=0)
ax = axes[0]
ax.bar(range(len(bit_freq)), np.sort(bit_freq)[::-1],
       color='#7B1FA2', alpha=0.75, width=1.0)
ax.set_xlabel('Bit index (sorted by frequency)')
ax.set_ylabel('Frequency across compounds')
ax.set_title(f'Bit Frequency (sorted)\n{(bit_freq > 0).sum()} active bits / 2048')

ax = axes[1]
ax.hist(bit_freq[bit_freq > 0], bins=50, color='#7B1FA2', edgecolor='white',
        linewidth=0.4, alpha=0.8)
ax.set_xlabel('Bit frequency')
ax.set_ylabel('Number of bits')
ax.set_title('Distribution of Active Bit Frequencies')
ax.text(0.97, 0.95, f'Active bits: {(bit_freq>0).sum()}\nMean freq: {bit_freq[bit_freq>0].mean()*100:.1f}%',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
fig.savefig(f'{FIGS}/10_fingerprint_bits.png')
plt.close()
log.info("  Saved: 10_fingerprint_bits.png")

# ── Fig 11: PCA explained variance ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PCA of RDKit Descriptors — Chemical Space Analysis', fontsize=13, fontweight='bold')

ev  = pca.explained_variance_ratio_
cev = np.cumsum(ev)
n_show = min(30, len(ev))

ax = axes[0]
ax.bar(range(1, n_show+1), ev[:n_show]*100, color='#1976D2', alpha=0.75)
ax2 = ax.twinx()
ax2.plot(range(1, n_show+1), cev[:n_show]*100, 'r-o', ms=4, lw=1.8)
ax2.axhline(90, color='gray', ls='--', lw=1, label='90% threshold')
ax2.set_ylabel('Cumulative Explained Variance (%)', color='crimson')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)', color='#1976D2')
ax.set_title(f'Scree Plot  |  {n90} PCs → 90% variance')

ax = axes[1]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn',
                alpha=0.5, s=18, linewidths=0, vmin=4, vmax=10)
plt.colorbar(sc, ax=ax, label='pIC50')
ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)')
ax.set_title('Chemical Space (PC1 vs PC2)\nColored by pIC50')

plt.tight_layout()
fig.savefig(f'{FIGS}/11_pca_chemical_space.png')
plt.close()
log.info("  Saved: 11_pca_chemical_space.png")

# ── Fig 12: Top descriptors correlated with pIC50 ────────────────────────────
desc_corr = pd.Series(
    [pearsonr(X_desc[:, i], y)[0] for i in range(X_desc.shape[1])],
    index=keep3
).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
top25 = desc_corr.head(25)
bar_cols = ['#D32F2F' if v > 0 else '#1565C0' for v in top25.values]
ax.barh(range(len(top25)), top25.values, color=bar_cols, alpha=0.82, edgecolor='white')
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25.index, fontsize=9)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel("Pearson r with pIC50")
ax.set_title('Top 25 RDKit Descriptors by Correlation with pIC50 — P10275',
             fontweight='bold')
ax.invert_yaxis()
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#D32F2F', label='Positive'), Patch(color='#1565C0', label='Negative')],
          fontsize=9, loc='lower right')
plt.tight_layout()
fig.savefig(f'{FIGS}/12_top_descriptors_pIC50_corr.png')
plt.close()
log.info("  Saved: 12_top_descriptors_pIC50_corr.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
feat_stats = {
    'n_compounds':   int(len(y)),
    'n_morgan_bits': len(fp_cols),
    'n_desc_final':  len(keep3),
    'n_combined':    len(fp_cols) + len(keep3),
    'pca_n90':       int(n90),
    'pIC50_min':     round(float(y.min()), 4),
    'pIC50_max':     round(float(y.max()), 4),
}
with open(f'{LOG_D}/feature_engineering_stats.json', 'w') as f:
    json.dump(feat_stats, f, indent=2)

log.info("\n" + "=" * 70)
log.info("STEP 2 COMPLETE — FEATURE ENGINEERING")
log.info(f"  Compounds:          {feat_stats['n_compounds']:,}")
log.info(f"  Morgan bits:        {feat_stats['n_morgan_bits']:,}")
log.info(f"  RDKit descriptors:  {feat_stats['n_desc_final']:,}")
log.info(f"  Combined features:  {feat_stats['n_combined']:,}")
log.info(f"  PCs for 90% var:    {feat_stats['pca_n90']}")
log.info(f"  4 figures saved to: {FIGS}/")
log.info("=" * 70)
