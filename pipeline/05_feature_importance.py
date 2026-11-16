"""
Step 5: Feature Importance Analysis — P10275 (Androgen Receptor)
BindingDB May 2026 | Prostate Cancer Drug Discovery ML Study

Methods:
  1. Tree-based feature importance (impurity) — RF, ET, XGBoost, LightGBM, CatBoost
  2. Permutation importance on test set (model-agnostic)
  3. SHAP values — top tuned model (XGBoost)
  4. Descriptor importance ranking + top Morgan bits
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, logging, os, sys, json, joblib
from datetime import datetime
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

BASE   = '/home/stalin/Downloads/BindingDB_P10275'
PROC   = f'{BASE}/data/processed'
MDIR   = f'{BASE}/models'
FIGS_I = f'{BASE}/figures/feature_importance'
LOG_D  = f'{BASE}/logs'
RDIR   = f'{BASE}/results'
os.makedirs(FIGS_I, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_D}/05_feature_importance.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("Step 5: Feature Importance Analysis — P10275")
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
PALETTE = plt.cm.tab10.colors

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & MODELS
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[1] Loading data and models...")

y         = np.load(f'{PROC}/y.npy')
X_desc    = np.load(f'{PROC}/X_desc.npy')           # 153 scaled descriptors
idx_train = np.load(f'{PROC}/idx_train.npy')
idx_test  = np.load(f'{PROC}/idx_test.npy')

with open(f'{PROC}/feature_names.json') as f:
    feat_info = json.load(f)
desc_cols  = feat_info['desc_cols']

# Tuned models trained on enhanced FP (ECFP4+ECFP6+MACCS = 4263 bits)
X_enh_fp = np.load(f'{PROC}/X_enhanced_fp.npy')   # (n, 4263)
n_ecfp4, n_ecfp6, n_maccs = 2048, 2048, 167
ecfp4_cols = [f'ECFP4_{i}' for i in range(n_ecfp4)]
ecfp6_cols = [f'ECFP6_{i}' for i in range(n_ecfp6)]
maccs_cols = [f'MACCS_{i}' for i in range(n_maccs)]
enh_fp_cols = ecfp4_cols + ecfp6_cols + maccs_cols   # 4263 names

# Re-align enhanced FP to valid SMILES subset (same as step 4)
df_mr = pd.read_csv(f'{PROC}/P10275_model_ready.csv')
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_fps(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return (list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)) +
                list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)) +
                list(MACCSkeys.GenMACCSKeys(mol)))
    except: return None

fps_check   = df_mr['Canonical_SMILES'].map(get_fps)
valid       = fps_check.notna()
idx_aligned = np.where(valid.values)[0]
valid_set   = set(idx_aligned)
new_pos     = {old: new for new, old in enumerate(idx_aligned)}
idx_tr_fp   = np.array([new_pos[i] for i in idx_train if i in valid_set])
idx_te_fp   = np.array([new_pos[i] for i in idx_test  if i in valid_set])
y_fp        = y[valid.values]
ytr         = y_fp[idx_tr_fp]
yte         = y_fp[idx_te_fp]

Xtr_m = X_enh_fp[idx_tr_fp];  Xte_m = X_enh_fp[idx_te_fp]
Xtr_d = X_desc[idx_train];    Xte_d = X_desc[idx_test]
morgan_cols = enh_fp_cols   # 4263 feature names for tuned models

# Load tuned models (morgan-based)
log.info("  Loading tuned models...")
best_et  = joblib.load(f'{MDIR}/Extra_Trees_tuned.pkl')
best_rf  = joblib.load(f'{MDIR}/Random_Forest_tuned.pkl')
best_xgb = joblib.load(f'{MDIR}/XGBoost_tuned.pkl')
best_lgb = joblib.load(f'{MDIR}/LightGBM_tuned.pkl')
best_cat = joblib.load(f'{MDIR}/CatBoost_tuned.pkl')

log.info(f"  Morgan features: {len(morgan_cols)}")
log.info(f"  Descriptor features: {len(desc_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TREE-BASED IMPORTANCE (Morgan FP — impurity / gain)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[2] Computing tree-based feature importances (Morgan FP)...")

fi_et  = best_et.feature_importances_
fi_rf  = best_rf.feature_importances_
fi_xgb = best_xgb.feature_importances_
fi_lgb = best_lgb.feature_importances_
fi_cat = best_cat.feature_importances_

# Normalise each to sum to 1
def norm(x): return x / x.sum() if x.sum() > 0 else x

fi_et  = norm(fi_et)
fi_rf  = norm(fi_rf)
fi_xgb = norm(fi_xgb)
fi_lgb = norm(fi_lgb)
fi_cat = norm(fi_cat)

# Mean importance across all 5 tree models
fi_mean = (fi_et + fi_rf + fi_xgb + fi_lgb + fi_cat) / 5
fi_std  = np.std([fi_et, fi_rf, fi_xgb, fi_lgb, fi_cat], axis=0)

fi_df = pd.DataFrame({
    'Feature': morgan_cols,
    'ET':      fi_et,
    'RF':      fi_rf,
    'XGBoost': fi_xgb,
    'LightGBM':fi_lgb,
    'CatBoost':fi_cat,
    'Mean':    fi_mean,
    'Std':     fi_std,
}).sort_values('Mean', ascending=False).reset_index(drop=True)

fi_df.to_csv(f'{RDIR}/morgan_feature_importance.csv', index=False)
log.info(f"  Top 10 Morgan bits by mean importance:")
for _, row in fi_df.head(10).iterrows():
    log.info(f"    {row['Feature']}  mean={row['Mean']:.6f} ± {row['Std']:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DESCRIPTOR IMPORTANCE — use CatBoost (trained on descriptors)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[3] Descriptor importance via CatBoost (desc features)...")
best_cat_desc = joblib.load(f'{MDIR}/CatBoost_tuned.pkl')

# Re-train CatBoost on descriptors to get descriptor importances
import xgboost as xgb_mod

# Use XGBoost on descriptors (fast, reliable importance)
xgb_desc = xgb_mod.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                  max_depth=5, n_jobs=-1, random_state=42, verbosity=0)
xgb_desc.fit(Xtr_d, ytr)
fi_desc = norm(xgb_desc.feature_importances_)

desc_fi_df = pd.DataFrame({
    'Descriptor': desc_cols,
    'Importance': fi_desc,
}).sort_values('Importance', ascending=False).reset_index(drop=True)
desc_fi_df.to_csv(f'{RDIR}/descriptor_feature_importance.csv', index=False)

log.info(f"  Top 15 descriptors:")
for _, row in desc_fi_df.head(15).iterrows():
    log.info(f"    {row['Descriptor']:<30}  {row['Importance']:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PERMUTATION IMPORTANCE — XGBoost (tuned) on test set
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[4] Permutation importance — XGBoost (tuned) on test set...")
perm = permutation_importance(best_xgb, Xte_m, yte,
                               n_repeats=20, random_state=42,
                               scoring='r2', n_jobs=-1)

perm_df = pd.DataFrame({
    'Feature':   morgan_cols,
    'Perm_mean': perm.importances_mean,
    'Perm_std':  perm.importances_std,
}).sort_values('Perm_mean', ascending=False).reset_index(drop=True)

# Keep only positive (actually important) features
perm_pos = perm_df[perm_df['Perm_mean'] > 0].copy()
perm_df.to_csv(f'{RDIR}/permutation_importance.csv', index=False)
log.info(f"  Positive permutation importance bits: {len(perm_pos)} / {len(perm_df)}")
log.info(f"  Top 10:")
for _, row in perm_pos.head(10).iterrows():
    log.info(f"    {row['Feature']}  {row['Perm_mean']:.6f} ± {row['Perm_std']:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SHAP VALUES — XGBoost (tuned)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[5] SHAP values — XGBoost (tuned)...")
try:
    import shap
    explainer  = shap.TreeExplainer(best_xgb)
    shap_vals  = explainer.shap_values(Xte_m)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    shap_df = pd.DataFrame({
        'Feature':        morgan_cols,
        'SHAP_mean_abs':  mean_abs_shap,
    }).sort_values('SHAP_mean_abs', ascending=False).reset_index(drop=True)
    shap_df.to_csv(f'{RDIR}/shap_importance.csv', index=False)
    has_shap = True
    log.info(f"  SHAP computed. Top 10 features:")
    for _, row in shap_df.head(10).iterrows():
        log.info(f"    {row['Feature']}  {row['SHAP_mean_abs']:.6f}")
except ImportError:
    log.info("  SHAP not installed — skipping")
    has_shap = False

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[6] Generating feature importance figures...")

# ── Fig 01: Top Morgan bits — mean importance across 5 tree models ────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Morgan Fingerprint (ECFP4) Feature Importance — P10275 Androgen Receptor\n'
             '5 Tree Models: ET, RF, XGBoost, LightGBM, CatBoost',
             fontsize=13, fontweight='bold')

N = 30
top_fi = fi_df.head(N)
ax = axes[0]
ax.barh(range(N), top_fi['Mean'].values, xerr=top_fi['Std'].values,
        color='#1976D2', alpha=0.8, capsize=3, edgecolor='white', height=0.7)
ax.set_yticks(range(N))
ax.set_yticklabels(top_fi['Feature'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Mean Normalised Importance (± Std)')
ax.set_title(f'Top {N} Morgan Bits — Mean Importance')

# Per-model heatmap (top 20)
top20 = fi_df.head(20)
heat  = top20[['ET','RF','XGBoost','LightGBM','CatBoost']].values
ax = axes[1]
im = ax.imshow(heat, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(5))
ax.set_xticklabels(['Extra\nTrees','Random\nForest','XGBoost','LightGBM','CatBoost'], fontsize=9)
ax.set_yticks(range(20))
ax.set_yticklabels(top20['Feature'].values, fontsize=8)
plt.colorbar(im, ax=ax, label='Normalised Importance')
ax.set_title('Top 20 Bits — Per-Model Heatmap')

plt.tight_layout()
fig.savefig(f'{FIGS_I}/01_morgan_importance_tree_models.png')
plt.close()
log.info("  Saved: 01_morgan_importance_tree_models.png")

# ── Fig 02: Descriptor importance ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
N = 30
top_desc = desc_fi_df.head(N)
bar_colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, N))
ax.barh(range(N), top_desc['Importance'].values,
        color=bar_colors, edgecolor='white', height=0.7, alpha=0.9)
ax.set_yticks(range(N))
ax.set_yticklabels(top_desc['Descriptor'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Normalised Feature Importance (XGBoost, Gain)')
ax.set_title(f'Top {N} RDKit Descriptors — Feature Importance\nP10275 Androgen Receptor',
             fontweight='bold')
for i, (_, row) in enumerate(top_desc.iterrows()):
    ax.text(row['Importance'] + 0.0002, i,
            f"{row['Importance']:.4f}", va='center', fontsize=7.5)
plt.tight_layout()
fig.savefig(f'{FIGS_I}/02_descriptor_importance.png')
plt.close()
log.info("  Saved: 02_descriptor_importance.png")

# ── Fig 03: Permutation importance ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
N = 25
top_perm = perm_pos.head(N)
ax.barh(range(len(top_perm)), top_perm['Perm_mean'].values,
        xerr=top_perm['Perm_std'].values,
        color='#7B1FA2', alpha=0.8, capsize=3, edgecolor='white', height=0.65)
ax.set_yticks(range(len(top_perm)))
ax.set_yticklabels(top_perm['Feature'].values, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Mean R² Decrease (± Std) | 20 Repeats')
ax.set_title(f'Permutation Importance — XGBoost (tuned) | Test Set (n={len(yte)})\n'
             f'P10275 Androgen Receptor  |  Top {len(top_perm)} positive bits shown',
             fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGS_I}/03_permutation_importance.png')
plt.close()
log.info("  Saved: 03_permutation_importance.png")

# ── Fig 04: SHAP summary ─────────────────────────────────────────────────────
if has_shap:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('SHAP Feature Importance — XGBoost (tuned) | Test Set\nP10275 Androgen Receptor',
                 fontsize=13, fontweight='bold')

    N = 25
    top_shap = shap_df.head(N)
    ax = axes[0]
    ax.barh(range(N), top_shap['SHAP_mean_abs'].values,
            color='#E65100', alpha=0.8, edgecolor='white', height=0.65)
    ax.set_yticks(range(N))
    ax.set_yticklabels(top_shap['Feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(f'Top {N} Features — Mean |SHAP|')

    # SHAP scatter: top feature vs pIC50
    top1 = shap_df.iloc[0]['Feature']
    top1_idx = morgan_cols.index(top1)
    ax = axes[1]
    sc = ax.scatter(Xte_m[:, top1_idx], shap_vals[:, top1_idx],
                    c=yte, cmap='RdYlGn', alpha=0.6, s=20, linewidths=0,
                    vmin=yte.min(), vmax=yte.max())
    plt.colorbar(sc, ax=ax, label='Actual pIC50')
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel(f'{top1} (bit value: 0 or 1)')
    ax.set_ylabel('SHAP Value')
    ax.set_title(f'SHAP Dependence — {top1}\n(most important Morgan bit)')

    plt.tight_layout()
    fig.savefig(f'{FIGS_I}/04_shap_importance.png')
    plt.close()
    log.info("  Saved: 04_shap_importance.png")

# ── Fig 05: Importance concordance across methods ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

# Rank each method, compare top-50 overlap
top50_tree = set(fi_df.head(50)['Feature'])
top50_perm = set(perm_pos.head(50)['Feature'])
methods_compared = {'Tree (Mean)': top50_tree, 'Permutation': top50_perm}
if has_shap:
    top50_shap = set(shap_df.head(50)['Feature'])
    methods_compared['SHAP'] = top50_shap

# Venn-style overlap bar chart
all_feats = sorted(top50_tree | top50_perm | (top50_shap if has_shap else set()))
overlap_data = {m: [1 if f in s else 0 for f in all_feats]
                for m, s in methods_compared.items()}

# Rank correlation between methods
from scipy.stats import spearmanr
fi_df_idx  = fi_df.set_index('Feature')['Mean']
perm_df_idx = perm_df.set_index('Feature')['Perm_mean']
common = fi_df_idx.index.intersection(perm_df_idx.index)
rho_tp, _ = spearmanr(fi_df_idx[common], perm_df_idx[common])

if has_shap:
    shap_df_idx = shap_df.set_index('Feature')['SHAP_mean_abs']
    common2 = fi_df_idx.index.intersection(shap_df_idx.index)
    rho_ts, _ = spearmanr(fi_df_idx[common2], shap_df_idx[common2])
    common3 = perm_df_idx.index.intersection(shap_df_idx.index)
    rho_ps, _ = spearmanr(perm_df_idx[common3], shap_df_idx[common3])

methods = list(methods_compared.keys())
n = len(methods)
corr_matrix = np.eye(n)
corr_matrix[0,1] = corr_matrix[1,0] = rho_tp
if has_shap:
    corr_matrix[0,2] = corr_matrix[2,0] = rho_ts
    corr_matrix[1,2] = corr_matrix[2,1] = rho_ps

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=methods, yticklabels=methods,
            linewidths=1, ax=ax, vmin=0, vmax=1,
            annot_kws={'size': 12})
ax.set_title('Spearman Rank Correlation Between Feature Importance Methods\n'
             'P10275 | 2048 Morgan bits', fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGS_I}/05_importance_method_concordance.png')
plt.close()
log.info("  Saved: 05_importance_method_concordance.png")

# ── Fig 06: Cumulative importance ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
cum_tree = np.cumsum(fi_df['Mean'].values)
cum_perm = np.cumsum(np.clip(perm_df['Perm_mean'].values, 0, None))
cum_perm = cum_perm / cum_perm[-1] if cum_perm[-1] > 0 else cum_perm

ax.plot(range(1, len(cum_tree)+1), cum_tree, color='#1976D2', lw=2, label='Tree Mean Importance')
ax.plot(range(1, len(cum_perm)+1), cum_perm, color='#7B1FA2', lw=2, label='Permutation Importance')
if has_shap:
    cum_shap = np.cumsum(shap_df['SHAP_mean_abs'].values)
    cum_shap /= cum_shap[-1]
    ax.plot(range(1, len(cum_shap)+1), cum_shap, color='#E65100', lw=2, label='SHAP |mean|')

for thresh in [0.5, 0.8, 0.9]:
    n_tree = int(np.searchsorted(cum_tree, thresh)) + 1
    ax.axhline(thresh, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.text(2100, thresh+0.01, f'{thresh:.0%}', fontsize=8, color='gray')
    ax.axvline(n_tree, color='#1976D2', ls=':', lw=0.8, alpha=0.5)

ax.set_xlabel('Number of Morgan Bits (ranked by importance)')
ax.set_ylabel('Cumulative Importance')
ax.set_title('Cumulative Feature Importance — How Many Bits Capture the Signal?\nP10275 Androgen Receptor',
             fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 500)
ax.set_ylim(0, 1.05)

n50 = int(np.searchsorted(cum_tree, 0.50)) + 1
n80 = int(np.searchsorted(cum_tree, 0.80)) + 1
ax.text(0.65, 0.35, f'Top {n50} bits → 50% importance\nTop {n80} bits → 80% importance',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
fig.savefig(f'{FIGS_I}/06_cumulative_importance.png')
plt.close()
log.info("  Saved: 06_cumulative_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
n50 = int(np.searchsorted(cum_tree, 0.50)) + 1
n80 = int(np.searchsorted(cum_tree, 0.80)) + 1
n90 = int(np.searchsorted(cum_tree, 0.90)) + 1
log.info("\n" + "=" * 70)
log.info("STEP 5 COMPLETE — FEATURE IMPORTANCE")
log.info(f"  Top Morgan bit (tree):   {fi_df.iloc[0]['Feature']}  ({fi_df.iloc[0]['Mean']:.6f})")
log.info(f"  Top descriptor (XGB):    {desc_fi_df.iloc[0]['Descriptor']}  ({desc_fi_df.iloc[0]['Importance']:.6f})")
log.info(f"  Top perm. bit:           {perm_pos.iloc[0]['Feature']}  ({perm_pos.iloc[0]['Perm_mean']:.6f})")
if has_shap:
    log.info(f"  Top SHAP bit:            {shap_df.iloc[0]['Feature']}  ({shap_df.iloc[0]['SHAP_mean_abs']:.6f})")
log.info(f"  Bits for 50% importance: {n50}")
log.info(f"  Bits for 80% importance: {n80}")
log.info(f"  Bits for 90% importance: {n90}")
log.info(f"  Figures saved → {FIGS_I}/")
log.info("=" * 70)
