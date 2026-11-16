"""
Step 3: Training 15 ML Models — pIC50 Prediction for P10275 (Androgen Receptor)
BindingDB May 2026 | Prostate Cancer Drug Discovery ML Study

Models trained:
  1.  Random Forest
  2.  Extra Trees
  3.  Gradient Boosting (sklearn)
  4.  XGBoost
  5.  LightGBM
  6.  CatBoost
  7.  Support Vector Regression (RBF)
  8.  Support Vector Regression (Linear)
  9.  Ridge Regression
  10. Lasso Regression
  11. ElasticNet
  12. K-Nearest Neighbors
  13. MLP Neural Network
  14. Partial Least Squares (PLS)
  15. AdaBoost

Feature sets: Morgan FP (2048 bits) | RDKit Descriptors (153) | Combined (2201)
Split: 80% train / 20% test (stratified by pIC50 quartile, random_state=42)
CV: 5-fold cross-validation on training set
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, logging, os, sys, json, joblib, time
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor

import xgboost  as xgb
import lightgbm as lgb
from catboost  import CatBoostRegressor

warnings.filterwarnings('ignore')

BASE   = '/home/stalin/Downloads/BindingDB_P10275'
PROC   = f'{BASE}/data/processed'
MDIR   = f'{BASE}/models'
FIGS_P = f'{BASE}/figures/model_performance'
LOG_D  = f'{BASE}/logs'
RDIR   = f'{BASE}/results'
os.makedirs(MDIR,   exist_ok=True)
os.makedirs(FIGS_P, exist_ok=True)
os.makedirs(RDIR,   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_D}/03_model_training.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("Step 3: Model Training — 15 ML Models | P10275 Androgen Receptor")
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
# 1. LOAD FEATURES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[1] Loading feature matrices...")
X_morgan   = np.load(f'{PROC}/X_morgan.npy')
X_desc     = np.load(f'{PROC}/X_desc.npy')
X_combined = np.load(f'{PROC}/X_combined.npy')
y          = np.load(f'{PROC}/y.npy')

with open(f'{PROC}/feature_names.json') as f:
    feat_info = json.load(f)

log.info(f"  X_morgan:   {X_morgan.shape}")
log.info(f"  X_desc:     {X_desc.shape}")
log.info(f"  X_combined: {X_combined.shape}")
log.info(f"  y:          {y.shape}  | range {y.min():.2f}–{y.max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT (stratified by pIC50 quartile)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[2] Train/test split (80/20, stratified by pIC50 quartile)...")
quartile_labels = pd.qcut(y, q=4, labels=False)
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.20,
    stratify=quartile_labels, random_state=42
)

splits = {}
for name, X in [('morgan', X_morgan), ('desc', X_desc), ('combined', X_combined)]:
    splits[name] = {
        'X_train': X[idx_train], 'X_test': X[idx_test],
        'y_train': y[idx_train], 'y_test': y[idx_test],
    }

log.info(f"  Train: {len(idx_train):,}  |  Test: {len(idx_test):,}")
log.info(f"  Train pIC50: {y[idx_train].mean():.3f} ± {y[idx_train].std():.3f}")
log.info(f"  Test  pIC50: {y[idx_test].mean():.3f}  ± {y[idx_test].std():.3f}")

np.save(f'{PROC}/idx_train.npy', idx_train)
np.save(f'{PROC}/idx_test.npy',  idx_test)

# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
# Each model entry: (name, model_object, preferred_feature_set)
# Feature sets: 'morgan', 'desc', 'combined'
MODELS = [
    ('Random Forest',           RandomForestRegressor(n_estimators=500, max_features='sqrt',
                                 min_samples_leaf=2, n_jobs=-1, random_state=42),
     'morgan'),

    ('Extra Trees',             ExtraTreesRegressor(n_estimators=500, max_features='sqrt',
                                 min_samples_leaf=2, n_jobs=-1, random_state=42),
     'morgan'),

    ('Gradient Boosting',       GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                 max_depth=4, subsample=0.8, random_state=42),
     'desc'),

    ('XGBoost',                 xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                 max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=0.1, reg_lambda=1.0,
                                 n_jobs=-1, random_state=42, verbosity=0),
     'morgan'),

    ('LightGBM',                lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                 num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=0.1, reg_lambda=1.0,
                                 n_jobs=-1, random_state=42, verbose=-1),
     'morgan'),

    ('CatBoost',                CatBoostRegressor(iterations=500, learning_rate=0.05,
                                 depth=6, l2_leaf_reg=3,
                                 random_seed=42, verbose=0),
     'desc'),

    ('SVR (RBF)',               SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'),
     'desc'),

    ('SVR (Linear)',            SVR(kernel='linear', C=1.0, epsilon=0.1),
     'desc'),

    ('Ridge Regression',        Ridge(alpha=1.0),
     'combined'),

    ('Lasso Regression',        Lasso(alpha=0.01, max_iter=5000),
     'combined'),

    ('ElasticNet',              ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
     'combined'),

    ('K-Nearest Neighbors',     KNeighborsRegressor(n_neighbors=7, weights='distance',
                                 metric='euclidean', n_jobs=-1),
     'desc'),

    ('MLP Neural Network',      MLPRegressor(hidden_layer_sizes=(512, 256, 128),
                                 activation='relu', solver='adam',
                                 learning_rate_init=0.001, max_iter=500,
                                 early_stopping=True, validation_fraction=0.1,
                                 random_state=42, batch_size=64),
     'desc'),

    ('PLS Regression',          PLSRegression(n_components=20),
     'desc'),

    ('AdaBoost',                AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4),
                                 n_estimators=200, learning_rate=0.1, random_state=42),
     'desc'),
]

log.info(f"\n[3] {len(MODELS)} models defined")
for name, _, fset in MODELS:
    log.info(f"  {name:<28}  features={fset}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS HELPER
# ══════════════════════════════════════════════════════════════════════════════
def metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Pearson_r': r, 'Spearman_rho': rho}

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN ALL MODELS + 5-FOLD CV
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[4] Training models with 5-fold CV...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_results = []
predictions = {}   # store test predictions for each model

for i, (mname, model, fset) in enumerate(MODELS):
    log.info(f"\n  [{i+1:02d}/15] {mname}  (features: {fset})")
    sp    = splits[fset]
    Xtr   = sp['X_train'];  ytr = sp['y_train']
    Xte   = sp['X_test'];   yte = sp['y_test']

    # 5-fold CV on training set
    t0    = time.time()
    cv_r2 = cross_val_score(model, Xtr, ytr, cv=kf, scoring='r2', n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(model, Xtr, ytr, cv=kf,
                                        scoring='neg_mean_squared_error', n_jobs=-1))
    cv_time = time.time() - t0

    # Final fit on full training set
    t1 = time.time()
    model.fit(Xtr, ytr)
    fit_time = time.time() - t1

    # Test set predictions
    y_pred_test  = model.predict(Xte)
    y_pred_train = model.predict(Xtr)

    test_m  = metrics(yte,  y_pred_test)
    train_m = metrics(ytr, y_pred_train)

    predictions[mname] = {
        'y_test':       yte.tolist(),
        'y_pred_test':  y_pred_test.tolist(),
        'y_train':      ytr.tolist(),
        'y_pred_train': y_pred_train.tolist(),
        'fset':         fset,
    }

    row = {
        'Model':           mname,
        'Feature_Set':     fset,
        'CV_R2_mean':      round(cv_r2.mean(), 4),
        'CV_R2_std':       round(cv_r2.std(),  4),
        'CV_RMSE_mean':    round(cv_rmse.mean(), 4),
        'CV_RMSE_std':     round(cv_rmse.std(),  4),
        'Test_R2':         round(test_m['R2'], 4),
        'Test_RMSE':       round(test_m['RMSE'], 4),
        'Test_MAE':        round(test_m['MAE'], 4),
        'Test_Pearson_r':  round(test_m['Pearson_r'], 4),
        'Test_Spearman':   round(test_m['Spearman_rho'], 4),
        'Train_R2':        round(train_m['R2'], 4),
        'Train_RMSE':      round(train_m['RMSE'], 4),
        'Overfit_R2':      round(train_m['R2'] - test_m['R2'], 4),
        'Fit_time_s':      round(fit_time, 2),
        'CV_time_s':       round(cv_time, 2),
    }
    all_results.append(row)

    log.info(f"    CV  R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}  |  "
             f"CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    log.info(f"    Test R²: {test_m['R2']:.4f}  |  RMSE: {test_m['RMSE']:.4f}  |  "
             f"MAE: {test_m['MAE']:.4f}  |  Pearson r: {test_m['Pearson_r']:.4f}")
    log.info(f"    Overfit (Train−Test R²): {train_m['R2']-test_m['R2']:.4f}  |  "
             f"Fit: {fit_time:.1f}s")

    # Save model
    model_path = f'{MDIR}/{mname.replace(" ", "_").replace("(","").replace(")","")}.pkl'
    joblib.dump(model, model_path)

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_results).sort_values('Test_R2', ascending=False).reset_index(drop=True)
results_df.insert(0, 'Rank', range(1, len(results_df)+1))
results_df.to_csv(f'{RDIR}/model_results.csv', index=False)
log.info(f"\n[5] Results saved → {RDIR}/model_results.csv")

with open(f'{RDIR}/predictions.json', 'w') as f:
    json.dump(predictions, f)

# Print leaderboard
log.info("\n  ── MODEL LEADERBOARD (ranked by Test R²) ──")
log.info(f"  {'Rank':<5} {'Model':<28} {'Feat':<10} {'Test R²':>8} {'Test RMSE':>10} {'Test MAE':>9} {'CV R²':>8} {'Pearson r':>10}")
log.info("  " + "-"*95)
for _, row in results_df.iterrows():
    log.info(f"  {row['Rank']:<5} {row['Model']:<28} {row['Feature_Set']:<10} "
             f"{row['Test_R2']:>8.4f} {row['Test_RMSE']:>10.4f} {row['Test_MAE']:>9.4f} "
             f"{row['CV_R2_mean']:>8.4f} {row['Test_Pearson_r']:>10.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[6] Generating performance figures...")
PALETTE_15 = plt.cm.tab20.colors[:15]

# ── Fig A: Leaderboard bar chart ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Model Performance Leaderboard — P10275 Androgen Receptor\n'
             f'Test set (n={len(idx_test):,}) | 5-fold CV on train (n={len(idx_train):,})',
             fontsize=13, fontweight='bold')

mnames = results_df['Model'].tolist()
colors = [PALETTE_15[i] for i in range(len(mnames))]

for ax, metric, label in zip(
        axes,
        ['Test_R2', 'Test_RMSE', 'Test_MAE'],
        ['Test R²', 'Test RMSE', 'Test MAE']):
    vals = results_df[metric].values
    bars = ax.barh(range(len(mnames)), vals, color=colors, edgecolor='white', height=0.65)
    ax.set_yticks(range(len(mnames)))
    ax.set_yticklabels(mnames, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.set_title(label)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + (max(vals)-min(vals))*0.01,
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    if metric == 'Test_R2':
        ax.set_xlim(0, min(1.0, max(vals)*1.15))
    else:
        ax.set_xlim(0, max(vals)*1.15)

plt.tight_layout()
fig.savefig(f'{FIGS_P}/01_model_leaderboard.png')
plt.close()
log.info("  Saved: 01_model_leaderboard.png")

# ── Fig B: CV R² with error bars ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
x = range(len(results_df))
ax.bar(x, results_df['CV_R2_mean'], yerr=results_df['CV_R2_std'],
       color=colors, edgecolor='white', capsize=4, alpha=0.85)
ax.bar(x, results_df['Test_R2'], color='none',
       edgecolor='black', linewidth=1.5, linestyle='--', label='Test R²')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=40, ha='right', fontsize=9)
ax.set_ylabel('R²')
ax.set_title('5-Fold CV R² (bars ± std) vs Test R² (dashed outline) — P10275', fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig(f'{FIGS_P}/02_cv_vs_test_R2.png')
plt.close()
log.info("  Saved: 02_cv_vs_test_R2.png")

# ── Fig C: Overfitting plot (Train R² vs Test R²) ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
for i, (_, row) in enumerate(results_df.iterrows()):
    ax.scatter(row['Train_R2'], row['Test_R2'], s=100, color=PALETTE_15[i],
               zorder=3, edgecolors='white', linewidths=0.5)
    ax.annotate(row['Model'], (row['Train_R2'], row['Test_R2']),
                fontsize=7.5, ha='left', va='bottom',
                xytext=(3, 3), textcoords='offset points')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='No overfit line')
ax.set_xlabel('Train R²')
ax.set_ylabel('Test R²')
ax.set_title('Overfitting Diagnosis — Train vs Test R²\n(Points above diagonal = underfit; below = overfit)',
             fontweight='bold')
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f'{FIGS_P}/03_overfit_diagnosis.png')
plt.close()
log.info("  Saved: 03_overfit_diagnosis.png")

# ── Fig D: Predicted vs Actual — all 15 models (5×3 grid) ────────────────────
fig, axes = plt.subplots(3, 5, figsize=(22, 13))
fig.suptitle('Predicted vs Actual pIC50 — All 15 Models | Test Set — P10275',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, (_, row) in enumerate(results_df.iterrows()):
    mname = row['Model']
    pred  = predictions[mname]
    yt    = np.array(pred['y_test'])
    yp    = np.array(pred['y_pred_test'])

    ax = axes[i]
    ax.scatter(yt, yp, alpha=0.45, s=18, color=PALETTE_15[i], linewidths=0)
    lims = [min(yt.min(), yp.min())-0.2, max(yt.max(), yp.max())+0.2]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual pIC50', fontsize=9)
    ax.set_ylabel('Predicted pIC50', fontsize=9)
    ax.set_title(f'#{row["Rank"]} {mname}\nR²={row["Test_R2"]:.3f}  RMSE={row["Test_RMSE"]:.3f}',
                 fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{FIGS_P}/04_predicted_vs_actual_all.png')
plt.close()
log.info("  Saved: 04_predicted_vs_actual_all.png")

# ── Fig E: Residual plots — all 15 models ────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(22, 13))
fig.suptitle('Residuals (Predicted − Actual) — All 15 Models | Test Set — P10275',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, (_, row) in enumerate(results_df.iterrows()):
    mname = row['Model']
    pred  = predictions[mname]
    yt    = np.array(pred['y_test'])
    yp    = np.array(pred['y_pred_test'])
    resid = yp - yt

    ax = axes[i]
    ax.scatter(yp, resid, alpha=0.45, s=18, color=PALETTE_15[i], linewidths=0)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.axhline(+row['Test_RMSE'], color='gray', lw=0.8, ls=':')
    ax.axhline(-row['Test_RMSE'], color='gray', lw=0.8, ls=':')
    ax.set_xlabel('Predicted pIC50', fontsize=9)
    ax.set_ylabel('Residual', fontsize=9)
    ax.set_title(f'#{row["Rank"]} {mname}\nRMSE={row["Test_RMSE"]:.3f}  MAE={row["Test_MAE"]:.3f}',
                 fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{FIGS_P}/05_residuals_all.png')
plt.close()
log.info("  Saved: 05_residuals_all.png")

# ── Fig F: Metrics heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
heat_cols = ['Test_R2','Test_RMSE','Test_MAE','CV_R2_mean','Test_Pearson_r','Test_Spearman','Overfit_R2']
heat_df   = results_df.set_index('Model')[heat_cols]

# Normalise each column to 0–1 for display
heat_norm = heat_df.copy()
for col in heat_cols:
    mn, mx = heat_df[col].min(), heat_df[col].max()
    if mx > mn:
        heat_norm[col] = (heat_df[col] - mn) / (mx - mn)

sns.heatmap(heat_norm, annot=heat_df.round(4), fmt='.4f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalised score'},
            annot_kws={'size': 8})
ax.set_title('Model Performance Metrics — P10275 (values = raw, colour = normalised)',
             fontweight='bold', pad=12)
ax.set_xticklabels(heat_cols, rotation=30, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
fig.savefig(f'{FIGS_P}/06_metrics_heatmap.png')
plt.close()
log.info("  Saved: 06_metrics_heatmap.png")

# ── Fig G: Top-3 models detailed scatter ─────────────────────────────────────
top3 = results_df.head(3)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Top-3 Models — Predicted vs Actual pIC50 | Test Set — P10275',
             fontsize=13, fontweight='bold')

for ax, (_, row) in zip(axes, top3.iterrows()):
    mname = row['Model']
    pred  = predictions[mname]
    yt    = np.array(pred['y_test'])
    yp    = np.array(pred['y_pred_test'])
    resid = yp - yt

    lims = [min(yt.min(), yp.min())-0.3, max(yt.max(), yp.max())+0.3]
    sc = ax.scatter(yt, yp, c=np.abs(resid), cmap='coolwarm_r',
                    s=35, alpha=0.7, linewidths=0, vmin=0, vmax=1.5)
    plt.colorbar(sc, ax=ax, label='|Residual|')
    ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.7, label='y=x')
    m, b = np.polyfit(yt, yp, 1)
    xl = np.linspace(lims[0], lims[1], 100)
    ax.plot(xl, m*xl+b, 'r-', lw=1.2, alpha=0.7, label=f'Fit (slope={m:.2f})')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual pIC50', fontsize=11)
    ax.set_ylabel('Predicted pIC50', fontsize=11)
    ax.set_title(f'#{row["Rank"]} {mname}\nR²={row["Test_R2"]:.4f}  '
                 f'RMSE={row["Test_RMSE"]:.4f}  MAE={row["Test_MAE"]:.4f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.text(0.03, 0.97, f'n={len(yt)}  Pearson r={row["Test_Pearson_r"]:.4f}\n'
                        f'Spearman ρ={row["Test_Spearman"]:.4f}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout()
fig.savefig(f'{FIGS_P}/07_top3_models_detail.png')
plt.close()
log.info("  Saved: 07_top3_models_detail.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL LOG
# ══════════════════════════════════════════════════════════════════════════════
best = results_df.iloc[0]
log.info("\n" + "=" * 70)
log.info("STEP 3 COMPLETE — MODEL TRAINING")
log.info(f"  Best model:     {best['Model']}")
log.info(f"  Test R²:        {best['Test_R2']:.4f}")
log.info(f"  Test RMSE:      {best['Test_RMSE']:.4f}")
log.info(f"  Test MAE:       {best['Test_MAE']:.4f}")
log.info(f"  CV R² (mean):   {best['CV_R2_mean']:.4f} ± {best['CV_R2_std']:.4f}")
log.info(f"  Pearson r:      {best['Test_Pearson_r']:.4f}")
log.info(f"  Models saved → {MDIR}/")
log.info(f"  Figures saved → {FIGS_P}/")
log.info("=" * 70)
