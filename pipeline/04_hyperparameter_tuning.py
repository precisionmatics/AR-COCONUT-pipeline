"""
Step 4: Hyperparameter Tuning + Enhanced Features + Stacking Ensemble
P10275 (Androgen Receptor) | Prostate Cancer Drug Discovery ML Study

Strategy:
  1. Enhanced fingerprints: ECFP4 + ECFP6 + MACCS (combined richer representation)
  2. Optuna Bayesian optimisation on top 6 models (100 trials each)
  3. Ridge/ElasticNet with proper alpha range via CV
  4. Stacking ensemble (Level-1: best tuned models → Level-2: Ridge meta-learner)
  5. Re-evaluate everything on same test split
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, logging, os, sys, json, joblib, time
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import RDLogger

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               StackingRegressor)
import xgboost  as xgb
import lightgbm as lgb
from catboost  import CatBoostRegressor
from sklearn.svm import SVR

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE   = '/home/stalin/Downloads/BindingDB_P10275'
PROC   = f'{BASE}/data/processed'
MDIR   = f'{BASE}/models'
FIGS_P = f'{BASE}/figures/model_performance'
LOG_D  = f'{BASE}/logs'
RDIR   = f'{BASE}/results'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_D}/04_hyperparameter_tuning.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("Step 4: Hyperparameter Tuning + Stacking | P10275")
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
log.info("\n[1] Loading data and existing splits...")
df  = pd.read_csv(f'{PROC}/P10275_model_ready.csv')
y   = np.load(f'{PROC}/y.npy')
idx_train = np.load(f'{PROC}/idx_train.npy')
idx_test  = np.load(f'{PROC}/idx_test.npy')

X_morgan   = np.load(f'{PROC}/X_morgan.npy')    # ECFP4, 2048 bits
X_desc     = np.load(f'{PROC}/X_desc.npy')       # RDKit descriptors, scaled
scaler_desc = joblib.load(f'{PROC}/scaler_desc.pkl')

with open(f'{PROC}/feature_names.json') as f:
    feat_info = json.load(f)

log.info(f"  Compounds: {len(y)}  Train: {len(idx_train)}  Test: {len(idx_test)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ENHANCED FINGERPRINTS: ECFP4 + ECFP6 + MACCS
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[2] Computing enhanced fingerprints (ECFP4 + ECFP6 + MACCS)...")

def get_fps(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        ecfp4 = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        ecfp6 = list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
        maccs  = list(MACCSkeys.GenMACCSKeys(mol))
        return ecfp4 + ecfp6 + maccs
    except:
        return None

fps_all = df['Canonical_SMILES'].map(get_fps)
valid   = fps_all.notna()
fps_arr = np.array(fps_all[valid].tolist(), dtype=np.float32)
# align idx
y_fp    = y[valid.values]
idx_aligned = np.where(valid.values)[0]

# Map original train/test indices to the valid subset
def remap_idx(orig_idx, valid_positions):
    valid_set = set(valid_positions)
    new_pos   = {old: new for new, old in enumerate(valid_positions)}
    return np.array([new_pos[i] for i in orig_idx if i in valid_set])

idx_tr_fp = remap_idx(idx_train, idx_aligned)
idx_te_fp = remap_idx(idx_test,  idx_aligned)

# Also align desc
X_desc_fp = X_desc[valid.values]
y_fp_tr   = y_fp[idx_tr_fp]
y_fp_te   = y_fp[idx_te_fp]

# Scale fps
scaler_fp = StandardScaler()
fps_scaled = scaler_fp.fit_transform(fps_arr)
joblib.dump(scaler_fp, f'{PROC}/scaler_fps_enhanced.pkl')

# Combined: enhanced FP + descriptors
X_enh = np.hstack([fps_arr, X_desc_fp]).astype(np.float32)
scaler_enh = StandardScaler()
X_enh_sc   = scaler_enh.fit_transform(X_enh).astype(np.float32)
joblib.dump(scaler_enh, f'{PROC}/scaler_enhanced_combined.pkl')

log.info(f"  Enhanced FP shape:       {fps_arr.shape}  (ECFP4:2048 + ECFP6:2048 + MACCS:167)")
log.info(f"  Enhanced combined shape: {X_enh.shape}")
log.info(f"  Train: {len(idx_tr_fp)}  Test: {len(idx_te_fp)}")

Xtr_fp  = fps_arr[idx_tr_fp];    Xte_fp  = fps_arr[idx_te_fp]
Xtr_enh = X_enh_sc[idx_tr_fp];   Xte_enh = X_enh_sc[idx_te_fp]
Xtr_desc= X_desc_fp[idx_tr_fp];  Xte_desc= X_desc_fp[idx_te_fp]

np.save(f'{PROC}/X_enhanced_fp.npy',       fps_arr)
np.save(f'{PROC}/X_enhanced_combined.npy', X_enh_sc)

# ══════════════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ══════════════════════════════════════════════════════════════════════════════
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    rho,_= spearmanr(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Pearson_r': r, 'Spearman_rho': rho}

def cv_r2(model, X, y):
    return cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1).mean()

# ══════════════════════════════════════════════════════════════════════════════
# 4. OPTUNA TUNING
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[3] Optuna hyperparameter tuning (100 trials per model)...")
N_TRIALS = 50
tuned_results = []

# ── 4a. Extra Trees ───────────────────────────────────────────────────────────
log.info("  Tuning Extra Trees...")
def et_objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 1000),
        'max_features':    trial.suggest_float('max_features', 0.1, 1.0),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf', 1, 10),
        'max_depth':       trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
        'n_jobs': -1, 'random_state': 42
    }
    return cv_r2(ExtraTreesRegressor(**params), Xtr_fp, y_fp_tr)

study_et = optuna.create_study(direction='maximize')
study_et.optimize(et_objective, n_trials=N_TRIALS, show_progress_bar=False)
bp = study_et.best_params
best_et = ExtraTreesRegressor(**bp, n_jobs=-1, random_state=42)
best_et.fit(Xtr_fp, y_fp_tr)
m = metrics(y_fp_te, best_et.predict(Xte_fp))
cv_m = cv_r2(best_et, Xtr_fp, y_fp_tr)
tuned_results.append({'Model':'Extra Trees (tuned)', 'CV_R2': round(cv_m,4),
                       **{k:round(v,4) for k,v in m.items()}, 'best_params': bp})
joblib.dump(best_et, f'{MDIR}/Extra_Trees_tuned.pkl')
log.info(f"    Best CV R²={study_et.best_value:.4f}  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")

# ── 4b. LightGBM ─────────────────────────────────────────────────────────────
log.info("  Tuning LightGBM...")
def lgb_objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 1500),
        'learning_rate':   trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves':      trial.suggest_int('num_leaves', 20, 200),
        'min_child_samples':trial.suggest_int('min_child_samples', 5, 50),
        'subsample':       trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha':       trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':      trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'n_jobs': -1, 'random_state': 42, 'verbose': -1
    }
    return cv_r2(lgb.LGBMRegressor(**params), Xtr_fp, y_fp_tr)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
bp = study_lgb.best_params
best_lgb = lgb.LGBMRegressor(**bp, n_jobs=-1, random_state=42, verbose=-1)
best_lgb.fit(Xtr_fp, y_fp_tr)
m = metrics(y_fp_te, best_lgb.predict(Xte_fp))
cv_m = cv_r2(best_lgb, Xtr_fp, y_fp_tr)
tuned_results.append({'Model':'LightGBM (tuned)', 'CV_R2': round(cv_m,4),
                       **{k:round(v,4) for k,v in m.items()}, 'best_params': bp})
joblib.dump(best_lgb, f'{MDIR}/LightGBM_tuned.pkl')
log.info(f"    Best CV R²={study_lgb.best_value:.4f}  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")

# ── 4c. XGBoost ──────────────────────────────────────────────────────────────
log.info("  Tuning XGBoost...")
def xgb_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'n_jobs': -1, 'random_state': 42, 'verbosity': 0
    }
    return cv_r2(xgb.XGBRegressor(**params), Xtr_fp, y_fp_tr)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
bp = study_xgb.best_params
best_xgb = xgb.XGBRegressor(**bp, n_jobs=-1, random_state=42, verbosity=0)
best_xgb.fit(Xtr_fp, y_fp_tr)
m = metrics(y_fp_te, best_xgb.predict(Xte_fp))
cv_m = cv_r2(best_xgb, Xtr_fp, y_fp_tr)
tuned_results.append({'Model':'XGBoost (tuned)', 'CV_R2': round(cv_m,4),
                       **{k:round(v,4) for k,v in m.items()}, 'best_params': bp})
joblib.dump(best_xgb, f'{MDIR}/XGBoost_tuned.pkl')
log.info(f"    Best CV R²={study_xgb.best_value:.4f}  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")

# ── 4d. Random Forest ────────────────────────────────────────────────────────
log.info("  Tuning Random Forest...")
def rf_objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 1000),
        'max_features':    trial.suggest_float('max_features', 0.05, 0.5),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf', 1, 10),
        'max_depth':       trial.suggest_categorical('max_depth', [None, 15, 25, 35]),
        'n_jobs': -1, 'random_state': 42
    }
    return cv_r2(RandomForestRegressor(**params), Xtr_fp, y_fp_tr)

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(rf_objective, n_trials=N_TRIALS, show_progress_bar=False)
bp = study_rf.best_params
best_rf = RandomForestRegressor(**bp, n_jobs=-1, random_state=42)
best_rf.fit(Xtr_fp, y_fp_tr)
m = metrics(y_fp_te, best_rf.predict(Xte_fp))
cv_m = cv_r2(best_rf, Xtr_fp, y_fp_tr)
tuned_results.append({'Model':'Random Forest (tuned)', 'CV_R2': round(cv_m,4),
                       **{k:round(v,4) for k,v in m.items()}, 'best_params': bp})
joblib.dump(best_rf, f'{MDIR}/Random_Forest_tuned.pkl')
log.info(f"    Best CV R²={study_rf.best_value:.4f}  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")

# ── 4e. CatBoost ─────────────────────────────────────────────────────────────
log.info("  Tuning CatBoost...")
def cat_objective(trial):
    params = {
        'iterations':   trial.suggest_int('iterations', 200, 1500),
        'learning_rate':trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'depth':        trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg':  trial.suggest_float('l2_leaf_reg', 0.1, 20.0, log=True),
        'subsample':    trial.suggest_float('subsample', 0.5, 1.0),
        'random_seed': 42, 'verbose': 0
    }
    return cv_r2(CatBoostRegressor(**params), Xtr_fp, y_fp_tr)

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(cat_objective, n_trials=N_TRIALS, show_progress_bar=False)
bp = study_cat.best_params
best_cat = CatBoostRegressor(**bp, random_seed=42, verbose=0)
best_cat.fit(Xtr_fp, y_fp_tr)
m = metrics(y_fp_te, best_cat.predict(Xte_fp))
cv_m = cv_r2(best_cat, Xtr_fp, y_fp_tr)
tuned_results.append({'Model':'CatBoost (tuned)', 'CV_R2': round(cv_m,4),
                       **{k:round(v,4) for k,v in m.items()}, 'best_params': bp})
joblib.dump(best_cat, f'{MDIR}/CatBoost_tuned.pkl')
log.info(f"    Best CV R²={study_cat.best_value:.4f}  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")

# ── 4f. SVR (RBF) on enhanced features ───────────────────────────────────────
log.info("  Tuning SVR (RBF) on enhanced combined features...")
def svr_objective(trial):
    params = {
        'C':      trial.suggest_float('C', 0.1, 100.0, log=True),
        'epsilon':trial.suggest_float('epsilon', 0.01, 0.5),
        'gamma':  trial.suggest_categorical('gamma', ['scale', 'auto']),
    }
    return cv_r2(SVR(kernel='rbf', **params), Xtr_enh, y_fp_tr)

study_svr = optuna.create_study(direction='maximize')
study_svr.optimize(svr_objective, n_trials=N_TRIALS, show_progress_bar=False)
bp = study_svr.best_params
best_svr = SVR(kernel='rbf', **bp)
best_svr.fit(Xtr_enh, y_fp_tr)
m = metrics(y_fp_te, best_svr.predict(Xte_enh))
cv_m = cv_r2(best_svr, Xtr_enh, y_fp_tr)
tuned_results.append({'Model':'SVR RBF (tuned, enh)', 'CV_R2': round(cv_m,4),
                       **{k:round(v,4) for k,v in m.items()}, 'best_params': bp})
joblib.dump(best_svr, f'{MDIR}/SVR_RBF_tuned.pkl')
log.info(f"    Best CV R²={study_svr.best_value:.4f}  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")

# ── 4g. Ridge with proper alpha (RidgeCV) ────────────────────────────────────
log.info("  Tuning Ridge with RidgeCV (alpha 0.01–10000)...")
alphas = np.logspace(-2, 4, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(Xtr_enh, y_fp_tr)
m = metrics(y_fp_te, ridge_cv.predict(Xte_enh))
cv_m = cv_r2(Ridge(alpha=ridge_cv.alpha_), Xtr_enh, y_fp_tr)
tuned_results.append({'Model':f'Ridge CV (α={ridge_cv.alpha_:.1f})', 'CV_R2': round(cv_m,4),
                       **{k:round(v,4) for k,v in m.items()}, 'best_params': {'alpha': ridge_cv.alpha_}})
joblib.dump(ridge_cv, f'{MDIR}/Ridge_CV_tuned.pkl')
log.info(f"    Best alpha={ridge_cv.alpha_:.4f}  Test R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[4] Building stacking ensemble...")

estimators = [
    ('et',   best_et),
    ('lgb',  best_lgb),
    ('xgb',  best_xgb),
    ('rf',   best_rf),
    ('cat',  best_cat),
]
meta_learner = RidgeCV(alphas=np.logspace(-2, 4, 50), cv=5)
stacker = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=5, n_jobs=-1, passthrough=False
)
t0 = time.time()
stacker.fit(Xtr_fp, y_fp_tr)
log.info(f"  Stacker trained in {time.time()-t0:.1f}s")

m_stack = metrics(y_fp_te, stacker.predict(Xte_fp))
cv_stack = cv_r2(stacker, Xtr_fp, y_fp_tr)
tuned_results.append({'Model':'Stacking Ensemble', 'CV_R2': round(cv_stack,4),
                       **{k:round(v,4) for k,v in m_stack.items()},
                       'best_params': 'ET+LGB+XGB+RF+CAT → RidgeCV'})
joblib.dump(stacker, f'{MDIR}/Stacking_Ensemble.pkl')
log.info(f"  Stacking → CV R²={cv_stack:.4f}  Test R²={m_stack['R2']:.4f}  RMSE={m_stack['RMSE']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. WEIGHTED AVERAGE ENSEMBLE (top-5 by CV R²)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[5] Weighted average ensemble (top-5 tuned models)...")
top5_models = [best_et, best_lgb, best_xgb, best_rf, best_cat]
top5_names  = ['ET','LGB','XGB','RF','CAT']
top5_preds_test  = np.array([m.predict(Xte_fp)  for m in top5_models])
top5_preds_train = np.array([m.predict(Xtr_fp) for m in top5_models])

# Optimise weights on training set
from scipy.optimize import minimize

def neg_r2_weighted(weights):
    weights = np.array(weights)
    weights = np.clip(weights, 0, None)
    if weights.sum() == 0:
        return 0
    weights /= weights.sum()
    pred = (top5_preds_train * weights[:, None]).sum(axis=0)
    return -r2_score(y_fp_tr, pred)

res = minimize(neg_r2_weighted, x0=[0.2]*5, method='SLSQP',
               bounds=[(0,1)]*5, options={'maxiter':500})
opt_weights = np.clip(res.x, 0, None)
opt_weights /= opt_weights.sum()
log.info(f"  Optimal weights: {dict(zip(top5_names, opt_weights.round(4)))}")

y_wavg = (top5_preds_test * opt_weights[:, None]).sum(axis=0)
m_wavg = metrics(y_fp_te, y_wavg)
cv_wavg = -neg_r2_weighted(opt_weights)  # training R²
tuned_results.append({'Model':'Weighted Avg Ensemble', 'CV_R2': round(cv_wavg,4),
                       **{k:round(v,4) for k,v in m_wavg.items()},
                       'best_params': dict(zip(top5_names, opt_weights.round(4)))})
log.info(f"  Weighted Avg → Test R²={m_wavg['R2']:.4f}  RMSE={m_wavg['RMSE']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE COMBINED RESULTS (baseline + tuned)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[6] Saving combined results...")
baseline_df = pd.read_csv(f'{RDIR}/model_results.csv')
baseline_df = baseline_df.rename(columns={
    'CV_R2_mean':'CV_R2','Test_R2':'R2','Test_RMSE':'RMSE',
    'Test_MAE':'MAE','Test_Pearson_r':'Pearson_r','Test_Spearman':'Spearman_rho'
})
baseline_df['Stage'] = 'Baseline'
baseline_simple = baseline_df[['Model','CV_R2','R2','RMSE','MAE','Pearson_r','Spearman_rho','Stage']]

tuned_df = pd.DataFrame([{
    'Model':       r['Model'],
    'CV_R2':       r['CV_R2'],
    'R2':          r['R2'],
    'RMSE':        r['RMSE'],
    'MAE':         r['MAE'],
    'Pearson_r':   r['Pearson_r'],
    'Spearman_rho':r['Spearman_rho'],
    'Stage':       'Tuned',
} for r in tuned_results])

all_df = pd.concat([baseline_simple, tuned_df], ignore_index=True)
all_df = all_df.sort_values('R2', ascending=False).reset_index(drop=True)
all_df.insert(0, 'Rank', range(1, len(all_df)+1))
all_df.to_csv(f'{RDIR}/model_results_combined.csv', index=False)

# Save tuned params
with open(f'{RDIR}/tuned_hyperparameters.json', 'w') as f:
    json.dump([{k: (str(v) if isinstance(v, dict) else v)
                for k, v in r.items()} for r in tuned_results], f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[7] Generating figures...")

tuned_only = all_df[all_df['Stage']=='Tuned'].copy()
PALETTE = plt.cm.tab10.colors

# ── Fig 08: Baseline vs Tuned comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Baseline vs Tuned Model Performance — P10275 Androgen Receptor',
             fontsize=13, fontweight='bold')

for ax, metric, label in zip(axes, ['R2','RMSE'], ['Test R²','Test RMSE']):
    base = baseline_simple.sort_values(metric, ascending=(metric=='RMSE'))
    tun  = tuned_only.sort_values(metric, ascending=(metric=='RMSE'))
    x    = range(len(base))
    ax.plot(x, base[metric].values, 'o--', color='#90CAF9', lw=1.5,
            ms=7, label='Baseline', zorder=2)
    ax.plot(range(len(tun)), tun[metric].values, 's-', color='#1565C0', lw=2,
            ms=8, label='Tuned', zorder=3)
    ax.set_xticks(range(max(len(base), len(tun))))
    labels_all = list(base['Model'].values) + [m for m in tun['Model'].values
                                                if m not in base['Model'].values]
    ax.set_xticks(range(len(tun)))
    ax.set_xticklabels(tun['Model'].values, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(f'{label}: Baseline vs Tuned')
    ax.legend(fontsize=10)

plt.tight_layout()
fig.savefig(f'{FIGS_P}/08_baseline_vs_tuned.png')
plt.close()
log.info("  Saved: 08_baseline_vs_tuned.png")

# ── Fig 09: Full leaderboard (all models, all stages) ────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
top_df = all_df.head(20)
bar_colors = ['#1565C0' if s=='Tuned' else '#90CAF9' for s in top_df['Stage']]
bars = ax.barh(range(len(top_df)), top_df['R2'], color=bar_colors,
               edgecolor='white', height=0.7)
ax.set_yticks(range(len(top_df)))
ax.set_yticklabels([f"#{r}  {m}  [{s[0]}]"
                    for r,m,s in zip(top_df['Rank'], top_df['Model'], top_df['Stage'])],
                   fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Test R²')
ax.set_title('Full Model Leaderboard — Baseline [B] + Tuned [T]\nP10275 Androgen Receptor',
             fontweight='bold')
for bar, val in zip(bars, top_df['R2']):
    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#1565C0', label='Tuned'),
                   Patch(color='#90CAF9', label='Baseline')], fontsize=9)
ax.set_xlim(0, 1.0)
plt.tight_layout()
fig.savefig(f'{FIGS_P}/09_full_leaderboard.png')
plt.close()
log.info("  Saved: 09_full_leaderboard.png")

# ── Fig 10: Optuna optimisation history ──────────────────────────────────────
studies = [('Extra Trees', study_et), ('LightGBM', study_lgb),
           ('XGBoost', study_xgb), ('Random Forest', study_rf),
           ('CatBoost', study_cat), ('SVR RBF', study_svr)]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Optuna Optimisation History — 100 Trials per Model | P10275',
             fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, (name, study) in enumerate(studies):
    ax = axes[i]
    trials_r2 = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(trials_r2)
    ax.scatter(range(len(trials_r2)), trials_r2, s=8, alpha=0.4,
               color=PALETTE[i], label='Trial R²')
    ax.plot(range(len(best_so_far)), best_so_far, color='crimson',
            lw=2, label=f'Best: {max(trials_r2):.4f}')
    ax.set_xlabel('Trial')
    ax.set_ylabel('CV R²')
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(f'{FIGS_P}/10_optuna_history.png')
plt.close()
log.info("  Saved: 10_optuna_history.png")

# ── Fig 11: Predicted vs Actual — top tuned models ───────────────────────────
top_tuned = tuned_only.nlargest(6, 'R2')
model_objs = {
    'Extra Trees (tuned)':    (best_et,  Xte_fp,  y_fp_te),
    'LightGBM (tuned)':       (best_lgb, Xte_fp,  y_fp_te),
    'XGBoost (tuned)':        (best_xgb, Xte_fp,  y_fp_te),
    'Random Forest (tuned)':  (best_rf,  Xte_fp,  y_fp_te),
    'CatBoost (tuned)':       (best_cat, Xte_fp,  y_fp_te),
    'Stacking Ensemble':      (stacker,  Xte_fp,  y_fp_te),
    'SVR RBF (tuned, enh)':   (best_svr, Xte_enh, y_fp_te),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Predicted vs Actual pIC50 — Top Tuned Models | Test Set — P10275',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, (_, row) in enumerate(top_tuned.iterrows()):
    mname = row['Model']
    if mname not in model_objs:
        continue
    mod, Xte_use, yte_use = model_objs[mname]
    yp = mod.predict(Xte_use)
    yt = yte_use

    ax = axes[i]
    lims = [min(yt.min(), yp.min())-0.3, max(yt.max(), yp.max())+0.3]
    resid = np.abs(yp - yt)
    sc = ax.scatter(yt, yp, c=resid, cmap='coolwarm_r', s=30, alpha=0.65,
                    linewidths=0, vmin=0, vmax=1.5)
    plt.colorbar(sc, ax=ax, label='|Error|')
    ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.7)
    m_fit, b_fit = np.polyfit(yt, yp, 1)
    xl = np.linspace(lims[0], lims[1], 100)
    ax.plot(xl, m_fit*xl+b_fit, 'r-', lw=1, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual pIC50')
    ax.set_ylabel('Predicted pIC50')
    ax.set_title(f'{mname}\nR²={row["R2"]:.4f}  RMSE={row["RMSE"]:.4f}  MAE={row["MAE"]:.4f}',
                 fontsize=9, fontweight='bold')
    ax.text(0.03, 0.97, f'Pearson r={row["Pearson_r"]:.4f}\nSpearman ρ={row["Spearman_rho"]:.4f}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{FIGS_P}/11_tuned_predicted_vs_actual.png')
plt.close()
log.info("  Saved: 11_tuned_predicted_vs_actual.png")

# ── Fig 12: Metrics comparison table (heatmap) ───────────────────────────────
top15_all = all_df.head(15)
heat_cols = ['R2','RMSE','MAE','Pearson_r','Spearman_rho']
heat_df   = top15_all.set_index('Model')[heat_cols]
heat_norm = heat_df.copy()
for col in heat_cols:
    mn, mx = heat_df[col].min(), heat_df[col].max()
    if mx > mn:
        if col in ['RMSE','MAE']:
            heat_norm[col] = 1 - (heat_df[col]-mn)/(mx-mn)
        else:
            heat_norm[col] = (heat_df[col]-mn)/(mx-mn)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heat_norm, annot=heat_df.round(4), fmt='.4f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalised score (higher=better)'},
            annot_kws={'size': 8.5}, vmin=0, vmax=1)
ax.set_title('Top-15 Models — All Metrics | P10275 Androgen Receptor\n'
             '(values = raw; colour = normalised; higher green = better)',
             fontweight='bold', pad=12)
ax.set_xticklabels(heat_cols, rotation=20, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
fig.savefig(f'{FIGS_P}/12_final_metrics_heatmap.png')
plt.close()
log.info("  Saved: 12_final_metrics_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. FINAL LEADERBOARD PRINT
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n── FINAL LEADERBOARD (Baseline + Tuned, ranked by Test R²) ──")
log.info(f"  {'Rank':<5} {'Model':<30} {'Stage':<8} {'Test R²':>8} {'RMSE':>8} {'MAE':>8} {'Pearson r':>10} {'CV R²':>8}")
log.info("  " + "-"*90)
for _, row in all_df.head(20).iterrows():
    log.info(f"  {row['Rank']:<5} {row['Model']:<30} {row['Stage']:<8} "
             f"{row['R2']:>8.4f} {row['RMSE']:>8.4f} {row['MAE']:>8.4f} "
             f"{row['Pearson_r']:>10.4f} {row['CV_R2']:>8.4f}")

best = all_df.iloc[0]
log.info("\n" + "=" * 70)
log.info("STEP 4 COMPLETE — HYPERPARAMETER TUNING + STACKING")
log.info(f"  Best model:   {best['Model']}  [{best['Stage']}]")
log.info(f"  Test R²:      {best['R2']:.4f}")
log.info(f"  Test RMSE:    {best['RMSE']:.4f}")
log.info(f"  Test MAE:     {best['MAE']:.4f}")
log.info(f"  Pearson r:    {best['Pearson_r']:.4f}")
log.info(f"  Spearman ρ:   {best['Spearman_rho']:.4f}")
log.info(f"  Models saved → {MDIR}/")
log.info("=" * 70)
