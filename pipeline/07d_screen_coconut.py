"""
Step 7d: Virtual Screening — COCONUT (Collection of Open Natural Products)
P10275 (Androgen Receptor) | Prostate Cancer Drug Discovery ML Study

Library   : COCONUT 2026-05 — 738,827 natural products
Features  : ECFP4+ECFP6+MACCS (4263 bits) + RDKit 2D descriptors (153)
Models    : Stacking, XGBoost, CatBoost, LightGBM, ExtraTrees → Consensus
AD        : Tanimoto (threshold=0.4) + kNN distance (k=5, PCA)
Speed     : 20 cores, batch=3000, streaming macro-chunks (50k)
Checkpoint: saves after every macro-chunk — restart resumes from last checkpoint
Output    : results/, figures/ (8 plots), logs/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rdkit import Chem
from rdkit.Chem import (AllChem, Descriptors, MACCSkeys, DataStructs,
                         rdMolDescriptors, QED, FilterCatalog)
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import warnings, logging, sys, json, joblib, time, zipfile, os
from datetime import datetime
from multiprocessing import Pool, cpu_count

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE  = '/home/stalin/Downloads/BindingDB_P10275'
PROC  = f'{BASE}/data/processed'
MDIR  = f'{BASE}/models'
SCDIR = f'{BASE}/screening/COCONUT'
RDIR  = f'{SCDIR}/results'
FDIR  = f'{SCDIR}/figures'
LDIR  = f'{SCDIR}/logs'
LIB   = '/home/stalin/Desktop/Screening_Library/coconut/coconut_csv-05-2026.zip'

N_CORES     = cpu_count()
BATCH_SIZE  = 3000
MACRO_CHUNK = 50_000
CKPT_FILE   = f'{RDIR}/coconut_checkpoint.csv'
VIS_SAMPLE  = 8000   # compounds to keep for chemical space PCA

log_file = f'{LDIR}/screening_coconut_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,
    'axes.titlesize':13,'axes.titleweight':'bold',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.3,
    'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight','savefig.facecolor':'white',
})
PALETTE = {'in_ad':'#1976D2','out_ad':'#D32F2F','pass':'#388E3C',
           'fail':'#F57C00','neutral':'#546E7A','gradient':'RdYlGn'}

log.info("="*70)
log.info("Step 7d: Virtual Screening — COCONUT Natural Products")
log.info(f"Target  : P10275 Androgen Receptor | Prostate Cancer")
log.info(f"Cores   : {N_CORES} | Batch: {BATCH_SIZE} | Macro: {MACRO_CHUNK:,} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("="*70)
t_start = time.time()

# ── Load models & artifacts ───────────────────────────────────────────────────
log.info("\n[1] Loading models and preprocessing artifacts...")
models = {
    'Stacking_Ensemble': joblib.load(f'{MDIR}/Stacking_Ensemble.pkl'),
    'XGBoost_tuned':     joblib.load(f'{MDIR}/XGBoost_tuned.pkl'),
    'CatBoost_tuned':    joblib.load(f'{MDIR}/CatBoost_tuned.pkl'),
    'LightGBM_tuned':    joblib.load(f'{MDIR}/LightGBM_tuned.pkl'),
    'Extra_Trees_tuned': joblib.load(f'{MDIR}/Extra_Trees_tuned.pkl'),
}
scaler_desc = joblib.load(f'{PROC}/scaler_desc.pkl')
with open(f'{PROC}/feature_names.json') as f:
    feat_info = json.load(f)
desc_cols = feat_info['desc_cols']
X_desc_all   = np.load(f'{PROC}/X_desc.npy')
idx_train    = np.load(f'{PROC}/idx_train.npy')
X_desc_train = X_desc_all[idx_train]
desc_train_medians = np.nanmedian(X_desc_train, axis=0)
df_ref       = pd.read_csv(f'{PROC}/P10275_model_ready.csv')
smiles_train = df_ref['Canonical_SMILES'].iloc[idx_train].tolist()

log.info("  Pre-computing training ECFP4 FPs...")
train_ecfp4 = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
               for s in smiles_train if Chem.MolFromSmiles(s)]
log.info(f"  Training ECFP4: {len(train_ecfp4)}")

log.info("  Fitting PCA for kNN AD...")
pca_ad  = PCA(n_components=20, random_state=42)
Xtr_pca = pca_ad.fit_transform(X_desc_train)
dist_tr  = cdist(Xtr_pca, Xtr_pca, metric='euclidean')
np.fill_diagonal(dist_tr, np.inf)
knn_dist_tr   = np.sort(dist_tr, axis=1)[:, :5].mean(axis=1)
knn_threshold = knn_dist_tr.mean() + 1.5 * knn_dist_tr.std()
log.info(f"  kNN threshold: {knn_threshold:.4f}")

pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)

# ── Worker function ───────────────────────────────────────────────────────────
all_rdkit_desc_names = [n for n, _ in Descriptors._descList]

def process_batch(batch_data):
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings('ignore')
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_rdkit_desc_names)
    pp = FilterCatalogParams()
    pp.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pc = FilterCatalog.FilterCatalog(pp)
    results = []
    for name, smi in batch_data:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None or mol.GetNumHeavyAtoms() == 0: continue
            ecfp4_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ecfp6    = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            maccs    = MACCSkeys.GenMACCSKeys(mol)
            enh_fp   = np.array(list(ecfp4_fp)+list(ecfp6)+list(maccs), dtype=np.float32)
            try:
                raw = list(desc_calc.CalcDescriptors(mol))
            except Exception:
                raw = [np.nan] * len(all_rdkit_desc_names)
            raw_dict = dict(zip(all_rdkit_desc_names, raw))
            desc_vals = []
            for i, col in enumerate(desc_cols):
                v = raw_dict.get(col, np.nan)
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    v = float(desc_train_medians[i])
                desc_vals.append(float(v))
            desc_arr = np.nan_to_num(np.array(desc_vals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            mw   = Descriptors.MolWt(mol);   hbd  = rdMolDescriptors.CalcNumHBD(mol)
            hba  = rdMolDescriptors.CalcNumHBA(mol); logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol);    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
            rings= rdMolDescriptors.CalcNumRings(mol); qed_v = QED.qed(mol)
            try:   pains = pc.HasMatch(mol)
            except: pains = False
            sims    = DataStructs.BulkTanimotoSimilarity(ecfp4_fp, train_ecfp4)
            max_sim = max(sims) if sims else 0.0
            results.append({
                'Name': name, 'SMILES': Chem.MolToSmiles(mol),
                '_enh_fp': enh_fp, '_desc_arr': desc_arr,
                'Max_Tanimoto': round(max_sim,4), 'In_Tanimoto_AD': max_sim >= 0.4,
                'MW': round(mw,2), 'HBD': hbd, 'HBA': hba, 'LogP': round(logp,3),
                'TPSA': round(tpsa,2), 'RotBonds': rotb, 'Rings': rings,
                'QED': round(qed_v,4),
                'Lipinski_pass': (mw<=500)and(hbd<=5)and(hba<=10)and(logp<=5),
                'Veber_pass': (tpsa<=140)and(rotb<=10), 'PAINS_flag': pains,
            })
        except: pass
    return results

def predict_and_finalize(chunk_raw):
    """Predict + kNN AD for one macro-chunk; returns lightweight records and scaled desc sample."""
    fp_matrix   = np.vstack([r['_enh_fp']   for r in chunk_raw]).astype(np.float32)
    desc_matrix = np.vstack([r['_desc_arr'] for r in chunk_raw]).astype(np.float64)
    desc_sc     = np.nan_to_num(scaler_desc.transform(desc_matrix), nan=0.0, posinf=0.0, neginf=0.0)

    preds = {mn: mdl.predict(fp_matrix).astype(np.float32) for mn, mdl in models.items()}
    consensus = np.mean(np.vstack(list(preds.values())), axis=0)
    pred_std  = np.std( np.vstack(list(preds.values())), axis=0)

    desc_pca  = pca_ad.transform(desc_sc)
    KNN_CHUNK = 5000
    knn_dists = np.zeros(len(desc_pca), dtype=np.float32)
    for i in range(0, len(desc_pca), KNN_CHUNK):
        ch = desc_pca[i:i+KNN_CHUNK]
        d  = cdist(ch, Xtr_pca, metric='euclidean')
        knn_dists[i:i+KNN_CHUNK] = np.sort(d, axis=1)[:, :5].mean(axis=1)
    in_knn = knn_dists <= knn_threshold

    # keep a small random sample of scaled descriptors for chemical space plot
    n = len(chunk_raw)
    vis_idx = np.random.choice(n, min(500, n), replace=False)
    desc_sc_sample = desc_sc[vis_idx]

    out = []
    for i, rec in enumerate(chunk_raw):
        rec['pIC50_Consensus']  = round(float(consensus[i]),4)
        rec['pIC50_Std']        = round(float(pred_std[i]),4)
        rec['pIC50_Stacking']   = round(float(preds['Stacking_Ensemble'][i]),4)
        rec['pIC50_XGBoost']    = round(float(preds['XGBoost_tuned'][i]),4)
        rec['pIC50_CatBoost']   = round(float(preds['CatBoost_tuned'][i]),4)
        rec['pIC50_LightGBM']   = round(float(preds['LightGBM_tuned'][i]),4)
        rec['pIC50_ExtraTrees'] = round(float(preds['Extra_Trees_tuned'][i]),4)
        rec['kNN_Distance']     = round(float(min(knn_dists[i], 9999.0)),4)
        rec['In_kNN_AD']        = bool(in_knn[i])
        rec['In_AD_Consensus']  = bool(rec['In_Tanimoto_AD'] and in_knn[i])
        del rec['_enh_fp']; del rec['_desc_arr']
        out.append(rec)
    return out, desc_sc_sample

# ── Load COCONUT from ZIP ─────────────────────────────────────────────────────
log.info("\n[2] Reading COCONUT CSV from ZIP...")
t_load = time.time()
with zipfile.ZipFile(LIB) as z:
    with z.open('coconut_csv-05-2026.csv') as f:
        df_lib = pd.read_csv(f, usecols=['identifier','canonical_smiles'],
                             dtype=str, low_memory=False)
df_lib = df_lib.dropna(subset=['canonical_smiles'])
compounds_raw = list(zip(df_lib['identifier'].tolist(), df_lib['canonical_smiles'].tolist()))
log.info(f"  Loaded: {len(compounds_raw):,} compounds in {time.time()-t_load:.1f}s")

# ── Checkpoint resume ─────────────────────────────────────────────────────────
if os.path.exists(CKPT_FILE):
    df_ckpt = pd.read_csv(CKPT_FILE)
    done_macros = int(df_ckpt['_macro_idx'].max()) + 1
    n_done = done_macros * MACRO_CHUNK
    all_results = df_ckpt.drop(columns=['_macro_idx']).to_dict('records')
    desc_sc_samples = []  # visual sample not recoverable from checkpoint; skip for resumed run
    log.info(f"  CHECKPOINT found: resuming from macro-chunk {done_macros} "
             f"({n_done:,}/{len(compounds_raw):,} compounds already done, "
             f"{len(all_results):,} valid results loaded)")
    start_macro = done_macros
else:
    all_results = []
    desc_sc_samples = []
    start_macro = 0
    log.info("  No checkpoint found — starting fresh")

# ── Parallel processing + streaming predictions ───────────────────────────────
micro_batches = [compounds_raw[i:i+BATCH_SIZE] for i in range(0, len(compounds_raw), BATCH_SIZE)]
n_macro = (len(compounds_raw) + MACRO_CHUNK - 1) // MACRO_CHUNK
micros_per_macro = MACRO_CHUNK // BATCH_SIZE

log.info(f"\n[3] Parallel feature computation + streaming predictions "
         f"({N_CORES} cores, batch={BATCH_SIZE}, macro={MACRO_CHUNK:,})...")
log.info(f"  Total micro-batches: {len(micro_batches)} | Macro-chunks: {n_macro} | Starting at: {start_macro}")

t_feat = time.time()
processed_total = start_macro * MACRO_CHUNK

for macro_idx in range(start_macro, n_macro):
    start_b = macro_idx * micros_per_macro
    end_b   = min(start_b + micros_per_macro, len(micro_batches))
    macro_batches = micro_batches[start_b:end_b]

    chunk_raw = []
    with Pool(processes=N_CORES) as pool:
        for batch_res in pool.imap(process_batch, macro_batches):
            chunk_raw.extend(batch_res)

    processed_total += sum(len(b) for b in macro_batches)
    elapsed = time.time() - t_feat
    rate    = (processed_total - start_macro * MACRO_CHUNK) / elapsed if elapsed > 0 else 0
    eta     = (len(compounds_raw) - processed_total) / rate if rate > 0 else 0
    log.info(f"  Macro {macro_idx+1}/{n_macro}: [{processed_total:>7,}/{len(compounds_raw):,}]  "
             f"features={len(chunk_raw):,}  rate={rate:.0f}/s  ETA={eta:.0f}s ({eta/60:.1f}min)")

    if chunk_raw:
        log.info(f"    → Predicting {len(chunk_raw):,} compounds...")
        chunk_results, desc_sample = predict_and_finalize(chunk_raw)
        all_results.extend(chunk_results)
        desc_sc_samples.append(desc_sample)
        del chunk_raw, chunk_results

        # ── Save checkpoint ───────────────────────────────────────────────────
        df_ckpt_new = pd.DataFrame(all_results)
        df_ckpt_new['_macro_idx'] = macro_idx
        df_ckpt_new.to_csv(CKPT_FILE, index=False)
        log.info(f"    Checkpoint saved: {len(all_results):,} results → {CKPT_FILE}")

log.info(f"  Done: {len(all_results):,} valid | time={time.time()-t_feat:.1f}s")

# ── Assemble dataframe ────────────────────────────────────────────────────────
log.info("\n[4] Assembling results dataframe...")
df_res = pd.DataFrame(all_results).sort_values('pIC50_Consensus', ascending=False).reset_index(drop=True)
if '_macro_idx' in df_res.columns:
    df_res = df_res.drop(columns=['_macro_idx'])
df_res.insert(0, 'Rank', df_res.index+1)

def classify(p):
    if p>=8.0:   return 'Highly Active (pIC50 ≥ 8)'
    elif p>=7.0: return 'Active (7–8)'
    elif p>=6.0: return 'Moderately Active (6–7)'
    elif p>=5.0: return 'Weakly Active (5–6)'
    else:        return 'Inactive (< 5)'
df_res['Activity_Class'] = df_res['pIC50_Consensus'].apply(classify)

n_screened = len(df_res)
n_in_ad    = int(df_res['In_AD_Consensus'].sum())
n_ro5      = int(df_res['Lipinski_pass'].sum())
n_veber    = int(df_res['Veber_pass'].sum())
n_nopain   = int((~df_res['PAINS_flag']).sum())
df_hits    = df_res[df_res['Lipinski_pass']&df_res['Veber_pass']&~df_res['PAINS_flag']&df_res['In_AD_Consensus']]
n_all_pass = len(df_hits)

log.info(f"  Screened: {n_screened:,} | In AD: {n_in_ad:,} ({n_in_ad/n_screened*100:.1f}%)")
log.info(f"  Lipinski: {n_ro5:,} | Veber: {n_veber:,} | PAINS-free: {n_nopain:,} | All: {n_all_pass:,}")
log.info(f"  pIC50: {df_res['pIC50_Consensus'].min():.3f}–{df_res['pIC50_Consensus'].max():.3f}")
log.info(f"  Top hit: {df_res.iloc[0]['Name']} | pIC50={df_res.iloc[0]['pIC50_Consensus']:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
log.info("\n[5] Saving results...")
df_res.to_csv(f'{RDIR}/coconut_all_screened.csv', index=False)
df_res[df_res['In_AD_Consensus']].head(100).to_csv(f'{RDIR}/coconut_top100_in_ad.csv', index=False)
df_hits.to_csv(f'{RDIR}/coconut_drug_like_hits.csv', index=False)
stats = {
    'library':'COCONUT Collection of Open Natural Products',
    'run_date':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_cores':N_CORES,'n_input':len(compounds_raw),'n_screened':int(n_screened),
    'n_in_ad_consensus':int(n_in_ad),'pct_in_ad':round(n_in_ad/n_screened*100,2),
    'n_lipinski_pass':int(n_ro5),'n_veber_pass':int(n_veber),
    'n_pains_free':int(n_nopain),'n_all_filters_pass':int(n_all_pass),
    'pIC50_min':round(float(df_res['pIC50_Consensus'].min()),4),
    'pIC50_max':round(float(df_res['pIC50_Consensus'].max()),4),
    'pIC50_mean':round(float(df_res['pIC50_Consensus'].mean()),4),
    'pIC50_std':round(float(df_res['pIC50_Consensus'].std()),4),
    'pIC50_median':round(float(df_res['pIC50_Consensus'].median()),4),
    'top1_name':str(df_res.iloc[0]['Name']),'top1_pIC50':round(float(df_res.iloc[0]['pIC50_Consensus']),4),
    'top1_smiles':str(df_res.iloc[0]['SMILES']),
    'activity_distribution':df_res['Activity_Class'].value_counts().to_dict(),
    'models_used':list(models.keys()),
    'ad_methods':['Tanimoto (ECFP4, threshold=0.4)','kNN (k=5, PCA 20 PCs, mean+1.5σ)'],
    'knn_ad_threshold':round(float(knn_threshold),4),
    'runtime_seconds':round(time.time()-t_start,1),
}
with open(f'{RDIR}/coconut_screening_stats.json','w') as f:
    json.dump(stats, f, indent=2)
log.info("  Saved: 3 CSVs + stats JSON")
# Remove checkpoint now that final results are written
if os.path.exists(CKPT_FILE):
    os.remove(CKPT_FILE)
    log.info("  Checkpoint file removed (final results saved)")

# ── Figures ───────────────────────────────────────────────────────────────────
log.info("\n[6] Generating figures...")
act_order  = ['Highly Active (pIC50 ≥ 8)','Active (7–8)','Moderately Active (6–7)','Weakly Active (5–6)','Inactive (< 5)']
act_colors = ['#7B1FA2','#1976D2','#388E3C','#F57C00','#546E7A']
act_cmap   = dict(zip(act_order, act_colors))
model_cols  = ['pIC50_Stacking','pIC50_XGBoost','pIC50_CatBoost','pIC50_LightGBM','pIC50_ExtraTrees']
model_names = ['Stacking','XGBoost','CatBoost','LightGBM','ExtraTrees']
colors_m    = ['#7B1FA2','#1976D2','#D32F2F','#388E3C','#F57C00']
np.random.seed(42)
idx_s   = np.random.choice(n_screened, min(6000,n_screened), replace=False)
knn_plot= df_res['kNN_Distance'].clip(upper=df_res['kNN_Distance'].replace([np.inf,-np.inf],np.nan).quantile(0.995)*1.5)
in_tan  = df_res['In_Tanimoto_AD'].astype(bool)
in_knn_s= df_res['In_kNN_AD'].astype(bool)
in_both = df_res['In_AD_Consensus'].astype(bool)
colors_ad = np.where(in_both, PALETTE['in_ad'], np.where(in_tan|in_knn_s, '#F57C00', PALETTE['out_ad']))

# Fig 01
fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.suptitle('Predicted pIC50 Distribution — COCONUT Natural Products\nP10275 Androgen Receptor | 5-Model Consensus',fontsize=13,fontweight='bold')
ax=axes[0]
ax.hist(df_res['pIC50_Consensus'],bins=80,color=PALETTE['neutral'],edgecolor='none',alpha=0.8,density=True,label='All')
in_v=df_res.loc[in_both,'pIC50_Consensus']
if len(in_v)>5: ax.hist(in_v,bins=40,color=PALETTE['in_ad'],edgecolor='none',alpha=0.65,density=True,label='In AD')
ax.axvline(df_res['pIC50_Consensus'].median(),color='black',ls='--',lw=1.5,label=f"Median={df_res['pIC50_Consensus'].median():.2f}")
ax.axvline(6.0,color='orange',ls=':',lw=1.5); ax.axvline(7.0,color='red',ls=':',lw=1.5)
ax.set_xlabel('Predicted pIC50');ax.set_ylabel('Density');ax.set_title(f'pIC50 Distribution (n={n_screened:,})');ax.legend(fontsize=8)
ax=axes[1]
act_counts=df_res['Activity_Class'].value_counts()
present=[a for a in act_order if a in act_counts.index]
bars=ax.barh(present,[act_counts.get(a,0) for a in present],color=[act_cmap[a] for a in present],edgecolor='white',alpha=0.85)
for bar,val in zip(bars,[act_counts.get(a,0) for a in present]):
    ax.text(bar.get_width()+20,bar.get_y()+bar.get_height()/2,f'{val:,} ({val/n_screened*100:.1f}%)',va='center',fontsize=8)
ax.set_xlabel('Number of Compounds');ax.set_title('Activity Classification');ax.invert_yaxis()
ax=axes[2]
ax.pie([n_in_ad,n_screened-n_in_ad],labels=[f'In AD\n(n={n_in_ad:,})',f'Out of AD\n(n={n_screened-n_in_ad:,})'],
       colors=[PALETTE['in_ad'],PALETTE['out_ad']],autopct='%1.1f%%',startangle=90,wedgeprops=dict(edgecolor='white',linewidth=1.5))
ax.set_title('Applicability Domain Coverage')
plt.tight_layout();fig.savefig(f'{FDIR}/01_pIC50_distribution.png');plt.close()
log.info("  Saved: 01_pIC50_distribution.png")

# Fig 02
fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.suptitle('5-Model Consensus Analysis — COCONUT Natural Products\nP10275 Androgen Receptor',fontsize=13,fontweight='bold')
ax=axes[0]
bp=ax.boxplot([df_res[c].values for c in model_cols],patch_artist=True,medianprops=dict(color='black',lw=2))
for patch,col in zip(bp['boxes'],colors_m): patch.set_facecolor(col);patch.set_alpha(0.75)
ax.set_xticklabels(model_names,rotation=20,ha='right',fontsize=9);ax.set_ylabel('Predicted pIC50');ax.set_title('Model Distributions')
ax=axes[1]
c_s=df_res['In_AD_Consensus'].iloc[idx_s].map({True:PALETTE['in_ad'],False:PALETTE['out_ad']})
ax.scatter(df_res['pIC50_Consensus'].iloc[idx_s],df_res['pIC50_Std'].iloc[idx_s],c=c_s,alpha=0.3,s=10,linewidths=0)
ax.legend(handles=[Patch(color=PALETTE['in_ad'],label='In AD'),Patch(color=PALETTE['out_ad'],label='Out of AD')],fontsize=9)
ax.set_xlabel('Consensus pIC50');ax.set_ylabel('Std');ax.set_title(f'Consensus vs Uncertainty (n={len(idx_s):,})')
ax=axes[2]
sc2=ax.scatter(df_res['pIC50_Stacking'].iloc[idx_s],df_res['pIC50_XGBoost'].iloc[idx_s],
               c=df_res['pIC50_Consensus'].iloc[idx_s],cmap=PALETTE['gradient'],alpha=0.3,s=10,linewidths=0,vmin=4,vmax=9)
plt.colorbar(sc2,ax=ax,label='Consensus pIC50')
lims=[df_res[['pIC50_Stacking','pIC50_XGBoost']].min().min()-0.1,df_res[['pIC50_Stacking','pIC50_XGBoost']].max().max()+0.1]
ax.plot(lims,lims,'k--',lw=1,alpha=0.5);ax.set_xlabel('Stacking pIC50');ax.set_ylabel('XGBoost pIC50');ax.set_title('Stacking vs XGBoost')
plt.tight_layout();fig.savefig(f'{FDIR}/02_model_consensus.png');plt.close()
log.info("  Saved: 02_model_consensus.png")

# Fig 03
fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.suptitle('Applicability Domain — COCONUT Natural Products\nP10275 Androgen Receptor',fontsize=13,fontweight='bold')
ax=axes[0]
ax.hist(df_res.loc[in_tan,'Max_Tanimoto'],bins=40,color=PALETTE['in_ad'],alpha=0.8,edgecolor='none',label=f'In AD (n={in_tan.sum():,})')
ax.hist(df_res.loc[~in_tan,'Max_Tanimoto'],bins=40,color=PALETTE['out_ad'],alpha=0.65,edgecolor='none',label=f'Out AD (n={(~in_tan).sum():,})')
ax.axvline(0.4,color='black',ls='--',lw=2,label='Threshold=0.4')
ax.set_xlabel('Max Tanimoto');ax.set_ylabel('Count');ax.set_title('Tanimoto Distribution');ax.legend(fontsize=8)
ax=axes[1]
ax.hist(knn_plot[in_knn_s],bins=40,color=PALETTE['in_ad'],alpha=0.8,edgecolor='none',label=f'In AD (n={in_knn_s.sum():,})')
ax.hist(knn_plot[~in_knn_s],bins=40,color=PALETTE['out_ad'],alpha=0.65,edgecolor='none',label=f'Out AD (n={(~in_knn_s).sum():,})')
ax.axvline(knn_threshold,color='black',ls='--',lw=2,label=f'Threshold={knn_threshold:.2f}')
ax.set_xlabel('kNN Distance');ax.set_ylabel('Count');ax.set_title('kNN Distance Distribution');ax.legend(fontsize=8)
ax=axes[2]
ax.scatter(df_res['Max_Tanimoto'].iloc[idx_s],knn_plot.iloc[idx_s],c=colors_ad[idx_s],alpha=0.3,s=8,linewidths=0)
ax.axvline(0.4,color='black',ls='--',lw=1.5);ax.axhline(knn_threshold,color='gray',ls='--',lw=1.5)
ax.legend(handles=[Patch(color=PALETTE['in_ad'],label=f'Both ADs (n={n_in_ad:,})'),
                   Patch(color='#F57C00',label='One AD'),Patch(color=PALETTE['out_ad'],label='Out both')],fontsize=8)
ax.set_xlabel('Max Tanimoto');ax.set_ylabel('kNN Distance');ax.set_title('AD Space')
plt.tight_layout();fig.savefig(f'{FDIR}/03_applicability_domain.png');plt.close()
log.info("  Saved: 03_applicability_domain.png")

# Fig 04
fig,axes=plt.subplots(2,3,figsize=(18,11))
fig.suptitle('Drug-Likeness Profile — COCONUT Natural Products\nP10275 Androgen Receptor',fontsize=13,fontweight='bold')
for i,(col,label,thresh) in enumerate([('MW','Molecular Weight (Da)',500),('LogP','LogP',5),('HBD','H-Bond Donors',5),
    ('HBA','H-Bond Acceptors',10),('TPSA','TPSA (Å²)',140),('RotBonds','Rotatable Bonds',10)]):
    ax=axes[i//3][i%3]
    col_clean=df_res[col].replace([np.inf,-np.inf],np.nan).dropna()
    in_v=df_res.loc[in_both,col].replace([np.inf,-np.inf],np.nan).dropna()
    ax.hist(col_clean,bins=60,color=PALETTE['neutral'],alpha=0.7,edgecolor='none',density=True,label='All')
    if len(in_v)>5: ax.hist(in_v,bins=30,color=PALETTE['in_ad'],alpha=0.6,edgecolor='none',density=True,label='In AD')
    ax.axvline(thresh,color='crimson',ls='--',lw=2,label=f'Limit={thresh}')
    pass_n=(df_res[col]<=thresh).sum()
    ax.set_xlabel(label);ax.set_ylabel('Density')
    ax.set_title(f'{label}\n{pass_n:,}/{n_screened:,} ({pass_n/n_screened*100:.0f}%) ≤ {thresh}');ax.legend(fontsize=8)
plt.tight_layout();fig.savefig(f'{FDIR}/04_drug_likeness.png');plt.close()
log.info("  Saved: 04_drug_likeness.png")

# Fig 05
n_show=50; df_show=df_res.head(n_show)
fig,axes=plt.subplots(1,2,figsize=(18,max(10,n_show*0.25)))
fig.suptitle(f'Top {n_show} Compounds — COCONUT Natural Products\nP10275 Androgen Receptor | Consensus of 5 ML Models',fontsize=13,fontweight='bold')
ax=axes[0]
bar_colors=[PALETTE['pass'] if r['In_AD_Consensus'] and not r['PAINS_flag'] and r['Lipinski_pass']
            else PALETTE['in_ad'] if r['In_AD_Consensus'] else PALETTE['neutral']
            for _,r in df_show.iterrows()]
bars=ax.barh(range(n_show),df_show['pIC50_Consensus'][::-1].values,color=list(reversed(bar_colors)),edgecolor='white',alpha=0.88)
ax.set_yticks(range(n_show));ax.set_yticklabels([str(n)[:28] for n in df_show['Name'][::-1]],fontsize=6.5)
ax.set_xlabel('Consensus pIC50');ax.set_title(f'Top {n_show} Compounds')
ax.axvline(7.0,color='red',ls=':',lw=1.5,alpha=0.7);ax.axvline(6.0,color='orange',ls=':',lw=1.5,alpha=0.7)
for bar,val in zip(bars,df_show['pIC50_Consensus'][::-1].values):
    ax.text(val+0.01,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',fontsize=6)
ax.legend(handles=[Patch(color=PALETTE['pass'],label='In AD+Drug-like+PAINS-free'),
                   Patch(color=PALETTE['in_ad'],label='In AD'),Patch(color=PALETTE['neutral'],label='Out of AD')],fontsize=8,loc='lower right')
ax=axes[1]
top10=df_res.head(10); x=np.arange(10); w=0.15
for i,(col,nm,cc) in enumerate(zip(model_cols,model_names,colors_m)):
    ax.bar(x+i*w,top10[col],w,label=nm,color=cc,alpha=0.82,edgecolor='white')
ax.set_xticks(x+w*2);ax.set_xticklabels([str(n)[:18] for n in top10['Name']],rotation=40,ha='right',fontsize=7.5)
ax.set_ylabel('Predicted pIC50');ax.set_title('Top 10 — Per-Model Predictions');ax.legend(fontsize=8)
ax.axhline(7.0,color='red',ls=':',lw=1.5,alpha=0.6);ax.axhline(6.0,color='orange',ls=':',lw=1.5,alpha=0.6)
plt.tight_layout();fig.savefig(f'{FDIR}/05_top_hits.png');plt.close()
log.info("  Saved: 05_top_hits.png")

# Fig 06 — Chemical space (uses desc samples collected during streaming)
log.info("  Chemical space PCA (streaming subsample)...")
if desc_sc_samples:
    desc_sc_sub = np.vstack(desc_sc_samples)
    # subsample to VIS_SAMPLE if we have more
    if len(desc_sc_sub) > VIS_SAMPLE:
        vis_idx2 = np.random.choice(len(desc_sc_sub), VIS_SAMPLE, replace=False)
        desc_sc_sub = desc_sc_sub[vis_idx2]
    n_vis = len(desc_sc_sub)
    X_combined_pca = np.vstack([X_desc_train, desc_sc_sub])
    pca_vis = PCA(n_components=2, random_state=42)
    pca_vis.fit(X_combined_pca)
    Xtr_2d = pca_vis.transform(X_desc_train)
    Xsc_2d = pca_vis.transform(desc_sc_sub)

    fig,axes=plt.subplots(1,2,figsize=(16,7))
    fig.suptitle('Chemical Space — COCONUT Natural Products\nPCA of RDKit Descriptors | P10275 Androgen Receptor',fontsize=13,fontweight='bold')
    ax=axes[0]
    ax.scatter(Xtr_2d[:,0],Xtr_2d[:,1],s=10,alpha=0.2,color='#90CAF9',linewidths=0,label=f'AR Training (n={len(Xtr_2d)})',zorder=1)
    ax.scatter(Xsc_2d[:,0],Xsc_2d[:,1],s=12,alpha=0.4,color=PALETTE['neutral'],linewidths=0,label=f'COCONUT subsample (n={n_vis:,})',zorder=2)
    ax.set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)');ax.set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Chemical Space (subsample n={n_vis:,})');ax.legend(fontsize=9)
    ax=axes[1]
    # colour by pIC50 using top-ranked subsample
    top_sub = df_res.head(n_vis)
    sc=ax.scatter(Xsc_2d[:,0],Xsc_2d[:,1],c='#90CAF9',s=10,alpha=0.3,linewidths=0)
    ax.scatter(Xtr_2d[:,0],Xtr_2d[:,1],s=8,alpha=0.12,color='gray',linewidths=0,label='AR Training')
    plt.colorbar(sc,ax=ax,label='(subsample — no pIC50 mapping)')
    ax.set_xlabel(f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)');ax.set_ylabel(f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Chemical Space — COCONUT vs Training')
    plt.tight_layout();fig.savefig(f'{FDIR}/06_chemical_space.png');plt.close()
    log.info("  Saved: 06_chemical_space.png")
else:
    log.info("  Skipping 06_chemical_space.png (no desc samples — checkpoint resume)")

# Fig 07
fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.suptitle('QED & Filter Summary — COCONUT Natural Products\nP10275 Androgen Receptor',fontsize=13,fontweight='bold')
ax=axes[0]
ax.hist(df_res['QED'],bins=60,color=PALETTE['neutral'],alpha=0.7,edgecolor='none',density=True,label='All')
if n_in_ad>5: ax.hist(df_res.loc[in_both,'QED'],bins=30,color=PALETTE['in_ad'],alpha=0.65,edgecolor='none',density=True,label='In AD')
ax.axvline(0.3,color='red',ls='--',lw=2,label='QED=0.3');ax.axvline(df_res['QED'].median(),color='black',ls='--',lw=1.5,label=f"Median={df_res['QED'].median():.3f}")
ax.set_xlabel('QED');ax.set_ylabel('Density');ax.set_title('QED Distribution');ax.legend(fontsize=8)
ax=axes[1]
ax.scatter(df_res['QED'].iloc[idx_s],df_res['pIC50_Consensus'].iloc[idx_s],c=colors_ad[idx_s],alpha=0.3,s=10,linewidths=0)
ax.axhline(7.0,color='red',ls=':',lw=1.5,alpha=0.7);ax.axhline(6.0,color='orange',ls=':',lw=1.5,alpha=0.7);ax.axvline(0.3,color='gray',ls='--',lw=1.5,alpha=0.7)
ax.set_xlabel('QED');ax.set_ylabel('Consensus pIC50');ax.set_title('QED vs pIC50')
ax.legend(handles=[Patch(color=PALETTE['in_ad'],label='In AD'),Patch(color=PALETTE['out_ad'],label='Out of AD')],fontsize=9)
ax=axes[2]
pass_counts=[n_ro5,n_veber,n_nopain,n_in_ad,n_all_pass]
fail_counts=[n_screened-x for x in pass_counts]
x_pos=np.arange(5)
b1=ax.bar(x_pos,pass_counts,color=PALETTE['pass'],alpha=0.85,edgecolor='white',label='Pass')
ax.bar(x_pos,fail_counts,bottom=pass_counts,color=PALETTE['fail'],alpha=0.7,edgecolor='white',label='Fail')
for bar,val in zip(b1,pass_counts):
    ax.text(bar.get_x()+bar.get_width()/2,val/2,f'{val:,}\n({val/n_screened*100:.0f}%)',ha='center',va='center',fontsize=8,color='white',fontweight='bold')
ax.set_xticks(x_pos);ax.set_xticklabels(['Lipinski\nRO5','Veber\nRules','PAINS-free','In AD\n(Consensus)','All\nFilters'],fontsize=9)
ax.set_ylabel('Number of Compounds');ax.set_title('Filter Summary');ax.legend(fontsize=9);ax.set_ylim(0,n_screened*1.1)
plt.tight_layout();fig.savefig(f'{FDIR}/07_filter_summary.png');plt.close()
log.info("  Saved: 07_filter_summary.png")

# Fig 08
df_detail=df_hits.head(20) if len(df_hits)>0 else df_res.head(20)
fig,ax=plt.subplots(figsize=(18,max(6,len(df_detail)*0.5)))
fig.suptitle(f'Top {len(df_detail)} Drug-Like Hits — COCONUT Natural Products\nAll filters: Lipinski ✓ | Veber ✓ | PAINS-free ✓ | In AD ✓',fontsize=13,fontweight='bold')
ax.axis('off')
table_cols=['Rank','Name','pIC50\nConsensus','pIC50\nStd','Max\nTanimoto','MW','LogP','HBD','HBA','TPSA','QED','Lipinski','Veber','PAINS']
table_data=[[int(r['Rank']),str(r['Name'])[:28],f"{r['pIC50_Consensus']:.3f}",f"{r['pIC50_Std']:.3f}",
             f"{r['Max_Tanimoto']:.3f}",f"{r['MW']:.1f}",f"{r['LogP']:.2f}",int(r['HBD']),int(r['HBA']),
             f"{r['TPSA']:.1f}",f"{r['QED']:.3f}",
             '✓' if r['Lipinski_pass'] else '✗','✓' if r['Veber_pass'] else '✗','✗' if r['PAINS_flag'] else '✓']
            for _,r in df_detail.iterrows()]
tbl=ax.table(cellText=table_data,colLabels=table_cols,cellLoc='center',loc='center')
tbl.auto_set_font_size(False);tbl.set_fontsize(8.5);tbl.scale(1,1.5)
for (r,c),cell in tbl.get_celld().items():
    if r==0: cell.set_facecolor('#1565C0');cell.set_text_props(color='white',fontweight='bold')
    elif r%2==0: cell.set_facecolor('#F5F5F5')
    cell.set_edgecolor('#E0E0E0')
    if r>0 and c==2:
        val=float(table_data[r-1][2])
        cell.set_facecolor('#C8E6C9' if val>=7.0 else '#FFF9C4' if val>=6.0 else '#FFCDD2')
plt.tight_layout();fig.savefig(f'{FDIR}/08_top_hits_table.png');plt.close()
log.info("  Saved: 08_top_hits_table.png")

elapsed=time.time()-t_start
log.info("\n"+"="*70)
log.info("STEP 7d COMPLETE — COCONUT VIRTUAL SCREENING")
log.info(f"  Runtime:        {elapsed:.1f}s ({elapsed/60:.1f} min)")
log.info(f"  Screened:       {n_screened:,}")
log.info(f"  In AD:          {n_in_ad:,} ({n_in_ad/n_screened*100:.1f}%)")
log.info(f"  Drug-like hits: {n_all_pass:,}")
log.info(f"  Top hit:        {df_res.iloc[0]['Name']} | pIC50={df_res.iloc[0]['pIC50_Consensus']:.4f}")
log.info(f"  Results → {RDIR}/  |  Figures → {FDIR}/")
log.info("="*70)
