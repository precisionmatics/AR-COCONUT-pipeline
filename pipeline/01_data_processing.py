"""
Step 1: Data Processing Pipeline for P10275 (Androgen Receptor)
BindingDB May 2026 | Prostate Cancer Drug Discovery ML Study

Handling decisions (documented for journal methods):
  - PROTAC entries (Protein cereblon/CUL4A/AR) → EXCLUDED (ternary degrader mechanism)
  - Mutant AR entries (W742C, E709Y) → EXCLUDED (different protein)
  - Truncated AR constructs ([633-919],[646-919]) → INCLUDED (LBD binding, same endpoint)
  - Censored IC50 (>X, <X) → EXCLUDED from regression dataset
  - IC50 is primary endpoint; Ki and EC50 used to augment via separate analysis
  - Duplicate SMILES → median IC50 across measurements retained
  - "Human" unified to "Homo sapiens"
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
import warnings, logging, os, sys, json
from datetime import datetime
from scipy.stats import pearsonr

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = '/home/stalin/Downloads/BindingDB_P10275'
RAW     = f'{BASE}/data/raw/P10275_raw.tsv'
PROC    = f'{BASE}/data/processed'
FIGS    = f'{BASE}/figures/descriptive'
LOG_DIR = f'{BASE}/logs'
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Logger ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/01_data_processing.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("P10275 (Androgen Receptor) — Data Processing Pipeline")
log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("=" * 70)

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})

stats = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[1] Loading raw data...")
df = pd.read_csv(RAW, sep='\t', low_memory=False)
stats['01_raw_rows'] = len(df)
log.info(f"  Raw entries: {len(df):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ORGANISM HARMONISATION
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[2] Harmonising organism labels...")
df['Organism'] = df['Organism'].str.strip().replace({'Human': 'Homo sapiens'})
log.info(f"  Organism counts: {df['Organism'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. EXCLUDE PROTAC / TERNARY COMPLEX ENTRIES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[3] Excluding PROTAC/ternary complex entries...")
protac_mask = df['Target_Name'].str.contains('cereblon', case=False, na=False)
n_protac = protac_mask.sum()
df_protac = df[protac_mask].copy()   # save separately
df = df[~protac_mask].reset_index(drop=True)
stats['02_protac_excluded'] = int(n_protac)
log.info(f"  PROTAC entries removed: {n_protac}")
log.info(f"  Remaining: {len(df):,}")

# Save PROTACs as a separate file for future analysis
df_protac.to_csv(f'{PROC}/P10275_PROTACs_excluded.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 4. EXCLUDE MUTANT AR ENTRIES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[4] Excluding mutant AR entries...")
mutant_mask = df['Target_Name'].str.contains(r'\[W742C\]|\[E709Y\]', regex=True, na=False)
n_mutant = mutant_mask.sum()
df_mutants = df[mutant_mask].copy()
df = df[~mutant_mask].reset_index(drop=True)
stats['03_mutant_excluded'] = int(n_mutant)
log.info(f"  Mutant AR entries removed: {n_mutant}  (W742C={df_mutants['Target_Name'].str.contains('W742C').sum()}, E709Y={df_mutants['Target_Name'].str.contains('E709Y').sum()})")
log.info(f"  Remaining: {len(df):,}")
df_mutants.to_csv(f'{PROC}/P10275_mutants_excluded.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 5. LOG TARGET NAME BREAKDOWN (after exclusions)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[5] Target name breakdown (retained):")
for name, cnt in df['Target_Name'].value_counts().items():
    log.info(f"  {cnt:5d}  {name}")
stats['04_retained_rows'] = len(df)

# ══════════════════════════════════════════════════════════════════════════════
# 6. IC50 EXTRACTION AND CLEANING
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[6] IC50 extraction and cleaning...")

def parse_binding_value(val):
    """Returns (numeric_value, is_censored, is_valid)."""
    s = str(val).strip()
    if s in ('', 'nan', 'NA', 'None', 'NaN'):
        return np.nan, False, False
    censored = any(c in s for c in ['>', '<', '~'])
    clean = s.replace('>', '').replace('<', '').replace('~', '').replace(',', '').strip()
    try:
        v = float(clean)
        return v, censored, True
    except:
        return np.nan, False, False

# Parse IC50
ic50_parsed = df['IC50_nM'].map(parse_binding_value)
df['IC50_val']      = [x[0] for x in ic50_parsed]
df['IC50_censored'] = [x[1] for x in ic50_parsed]
df['IC50_valid']    = [x[2] for x in ic50_parsed]

n_ic50_total    = df['IC50_valid'].sum()
n_ic50_censored = df['IC50_censored'].sum()
n_ic50_exact    = int((df['IC50_valid'] & ~df['IC50_censored']).sum())
log.info(f"  Total IC50 entries:   {n_ic50_total:,}")
log.info(f"  Censored (>, <):      {n_ic50_censored:,}  → EXCLUDED from regression")
log.info(f"  Exact (uncensored):   {n_ic50_exact:,}")
stats['05_ic50_total']    = int(n_ic50_total)
stats['05_ic50_censored'] = int(n_ic50_censored)
stats['05_ic50_exact']    = n_ic50_exact

# Parse Ki
ki_parsed = df['Ki_nM'].map(parse_binding_value)
df['Ki_val']      = [x[0] for x in ki_parsed]
df['Ki_censored'] = [x[1] for x in ki_parsed]
df['Ki_valid']    = [x[2] for x in ki_parsed]
n_ki_exact = int((df['Ki_valid'] & ~df['Ki_censored']).sum())
log.info(f"  Exact Ki entries:     {n_ki_exact:,}")
stats['05_ki_exact'] = n_ki_exact

# Parse EC50
ec50_parsed = df['EC50_nM'].map(parse_binding_value)
df['EC50_val']      = [x[0] for x in ec50_parsed]
df['EC50_censored'] = [x[1] for x in ec50_parsed]
df['EC50_valid']    = [x[2] for x in ec50_parsed]
n_ec50_exact = int((df['EC50_valid'] & ~df['EC50_censored']).sum())
log.info(f"  Exact EC50 entries:   {n_ec50_exact:,}")
stats['05_ec50_exact'] = n_ec50_exact

# ══════════════════════════════════════════════════════════════════════════════
# 7. BUILD IC50 DATASET (primary ML endpoint — exact values only)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[7] Building IC50 regression dataset (exact values only)...")
df_ic50 = df[df['IC50_valid'] & ~df['IC50_censored'] & (df['IC50_val'] > 0)].copy()
log.info(f"  Entries with exact IC50: {len(df_ic50):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SMILES VALIDATION & CANONICALISATION
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[8] SMILES validation and canonicalisation...")

def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None

df_ic50['Canonical_SMILES'] = df_ic50['Ligand_SMILES'].map(canonicalize)
n_invalid_smi = df_ic50['Canonical_SMILES'].isna().sum()
df_ic50 = df_ic50[df_ic50['Canonical_SMILES'].notna()].copy()
log.info(f"  Invalid SMILES removed: {n_invalid_smi}")
log.info(f"  Remaining: {len(df_ic50):,}")
stats['06_invalid_smiles'] = int(n_invalid_smi)

# ══════════════════════════════════════════════════════════════════════════════
# 9. DUPLICATE HANDLING — median IC50 per canonical SMILES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[9] Aggregating duplicate SMILES (median IC50)...")
n_before_dedup = len(df_ic50)

agg = (df_ic50
       .groupby('Canonical_SMILES')
       .agg(
           IC50_nM_median  = ('IC50_val', 'median'),
           IC50_nM_mean    = ('IC50_val', 'mean'),
           IC50_nM_std     = ('IC50_val', 'std'),
           IC50_nM_min     = ('IC50_val', 'min'),
           IC50_nM_max     = ('IC50_val', 'max'),
           n_measurements  = ('IC50_val', 'count'),
           Ligand_Name     = ('Ligand_Name', 'first'),
           Target_Name     = ('Target_Name', 'first'),
           Organism        = ('Organism', 'first'),
           Ligand_SMILES   = ('Ligand_SMILES', 'first'),
       )
       .reset_index())

n_unique = len(agg)
n_multi  = int((agg['n_measurements'] > 1).sum())
log.info(f"  Before dedup: {n_before_dedup:,} entries")
log.info(f"  After dedup:  {n_unique:,} unique SMILES")
log.info(f"  Multi-measured: {n_multi:,} compounds")
log.info(f"  Measurement distribution:")
for n, cnt in agg['n_measurements'].value_counts().sort_index().items():
    if n <= 5 or cnt > 5:
        log.info(f"    {n} measurements: {cnt} compounds")
stats['07_before_dedup'] = n_before_dedup
stats['07_unique_smiles'] = n_unique
stats['07_multi_measured'] = n_multi

# Use median as primary IC50
agg['IC50_nM'] = agg['IC50_nM_median']

# ══════════════════════════════════════════════════════════════════════════════
# 10. pIC50 CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[10] Converting IC50 → pIC50...")
# pIC50 = -log10(IC50_M) = -log10(IC50_nM × 10⁻⁹) = 9 - log10(IC50_nM)
agg['pIC50'] = 9.0 - np.log10(agg['IC50_nM'])

log.info(f"  pIC50 min:    {agg['pIC50'].min():.3f}")
log.info(f"  pIC50 max:    {agg['pIC50'].max():.3f}")
log.info(f"  pIC50 mean:   {agg['pIC50'].mean():.3f}")
log.info(f"  pIC50 median: {agg['pIC50'].median():.3f}")
log.info(f"  pIC50 std:    {agg['pIC50'].std():.3f}")
stats.update({
    'pIC50_min':    round(float(agg['pIC50'].min()), 4),
    'pIC50_max':    round(float(agg['pIC50'].max()), 4),
    'pIC50_mean':   round(float(agg['pIC50'].mean()), 4),
    'pIC50_median': round(float(agg['pIC50'].median()), 4),
    'pIC50_std':    round(float(agg['pIC50'].std()), 4),
})

# Activity classification
bins   = [-np.inf, 5.0, 6.0, 7.0, np.inf]
labels = ['Inactive (<5)', 'Weakly Active (5–6)', 'Active (6–7)', 'Highly Active (>7)']
agg['Activity_Class'] = pd.cut(agg['pIC50'], bins=bins, labels=labels)
class_counts = agg['Activity_Class'].value_counts().sort_index()
log.info(f"\n  Activity class distribution:")
for cls, cnt in class_counts.items():
    log.info(f"    {cls}: {cnt:,}  ({cnt/len(agg)*100:.1f}%)")
stats['activity_classes'] = class_counts.to_dict()

# ══════════════════════════════════════════════════════════════════════════════
# 11. MOLECULAR PROPERTIES (Lipinski + drug-likeness)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[11] Computing molecular properties...")

def mol_props(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [np.nan] * 11
        return [
            Descriptors.MolWt(mol),
            Descriptors.ExactMolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            mol.GetNumHeavyAtoms(),
        ]
    except:
        return [np.nan] * 11

prop_cols = ['MW','ExactMW','LogP','HBD','HBA','TPSA','RotBonds','AromaticRings','Rings','Fsp3','HeavyAtoms']
props = pd.DataFrame(agg['Canonical_SMILES'].map(mol_props).tolist(), columns=prop_cols)
agg = pd.concat([agg.reset_index(drop=True), props], axis=1)
agg = agg.dropna(subset=['MW']).reset_index(drop=True)

# Lipinski Ro5
agg['Ro5'] = (agg['MW'] <= 500) & (agg['LogP'] <= 5) & (agg['HBD'] <= 5) & (agg['HBA'] <= 10)
n_ro5 = int(agg['Ro5'].sum())
log.info(f"  Ro5 compliant: {n_ro5:,} / {len(agg):,}  ({n_ro5/len(agg)*100:.1f}%)")
stats['Ro5_compliant'] = n_ro5
stats['Ro5_pct'] = round(n_ro5/len(agg)*100, 1)

log.info(f"  Final dataset: {len(agg):,} compounds")
stats['final_compounds'] = len(agg)

# ══════════════════════════════════════════════════════════════════════════════
# 12. ALSO SAVE COMBINED DATASET (IC50 + Ki where IC50 missing, exact only)
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[12] Building combined IC50+Ki dataset for reference...")
# Rows with exact Ki but NO IC50
df_ki_only = df[
    (df['Ki_valid'] & ~df['Ki_censored'] & (df['Ki_val'] > 0)) &
    ~(df['IC50_valid'] & ~df['IC50_censored'])
].copy()
df_ki_only['Canonical_SMILES'] = df_ki_only['Ligand_SMILES'].map(canonicalize)
df_ki_only = df_ki_only[df_ki_only['Canonical_SMILES'].notna()].copy()
ki_agg = (df_ki_only.groupby('Canonical_SMILES')
          .agg(Ki_nM=('Ki_val','median'), n_Ki=('Ki_val','count'), Ligand_Name=('Ligand_Name','first'))
          .reset_index())
ki_agg = ki_agg[~ki_agg['Canonical_SMILES'].isin(agg['Canonical_SMILES'])]
ki_agg['pKi'] = 9.0 - np.log10(ki_agg['Ki_nM'])
ki_agg.to_csv(f'{PROC}/P10275_Ki_additional.csv', index=False)
log.info(f"  Additional Ki-only compounds: {len(ki_agg):,} (saved separately)")
stats['ki_additional'] = len(ki_agg)

# ══════════════════════════════════════════════════════════════════════════════
# 13. SAVE PROCESSED DATA
# ══════════════════════════════════════════════════════════════════════════════
out_path = f'{PROC}/P10275_processed.csv'
agg.to_csv(out_path, index=False)
log.info(f"\n  Saved: {out_path}  ({len(agg):,} × {len(agg.columns)} columns)")

# ══════════════════════════════════════════════════════════════════════════════
# 14. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
log.info("\n[13] Generating figures...")
PALETTE = ['#1976D2','#388E3C','#F57C00','#D32F2F','#7B1FA2','#0097A7']

# ── Fig 1: Data funnel ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
funnel_labels = [
    f'Raw BindingDB\n(P10275 entries)\nn={stats["01_raw_rows"]:,}',
    f'After PROTAC exclusion\nn={stats["01_raw_rows"]-stats["02_protac_excluded"]:,}',
    f'After mutant exclusion\nn={stats["04_retained_rows"]:,}',
    f'Exact IC50 only\n(censored removed)\nn={stats["05_ic50_exact"]:,}',
    f'Valid SMILES\nn={stats["05_ic50_exact"]-stats["06_invalid_smiles"]:,}',
    f'Unique compounds\n(deduplication)\nn={stats["final_compounds"]:,}',
]
funnel_n = [
    stats['01_raw_rows'],
    stats['01_raw_rows'] - stats['02_protac_excluded'],
    stats['04_retained_rows'],
    stats['05_ic50_exact'],
    stats['05_ic50_exact'] - stats['06_invalid_smiles'],
    stats['final_compounds'],
]
bar_colors = ['#90CAF9','#64B5F6','#42A5F5','#2196F3','#1E88E5','#1565C0']
bars = ax.barh(range(len(funnel_n)), funnel_n, color=bar_colors, edgecolor='white', height=0.6)
ax.set_yticks(range(len(funnel_labels)))
ax.set_yticklabels(funnel_labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of Entries')
ax.set_title('Data Processing Funnel — P10275 (Androgen Receptor)\nBindingDB May 2026', fontweight='bold')
for bar, n in zip(bars, funnel_n):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{n:,}', va='center', fontsize=9, color='#333')
ax.grid(axis='x', alpha=0.3)
ax.spines['left'].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIGS}/01_data_funnel.png')
plt.close()
log.info("  Saved: 01_data_funnel.png")

# ── Fig 2: pIC50 distribution (3-panel) ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'pIC50 Distribution — P10275 Androgen Receptor\n'
             f'n = {len(agg):,} unique compounds | BindingDB May 2026',
             fontsize=13, fontweight='bold')

ax = axes[0]
n_bins = 50
ax.hist(agg['pIC50'], bins=n_bins, color='#1976D2', edgecolor='white', linewidth=0.5, alpha=0.85)
ax.axvline(agg['pIC50'].mean(),   color='crimson', ls='--', lw=2, label=f'Mean = {agg["pIC50"].mean():.2f}')
ax.axvline(agg['pIC50'].median(), color='orange',  ls='--', lw=2, label=f'Median = {agg["pIC50"].median():.2f}')
for v, c, lbl in [(5,'#616161','pIC50=5'), (6,'#9E9E9E','pIC50=6'), (7,'#757575','pIC50=7')]:
    ax.axvline(v, color=c, ls=':', lw=1, alpha=0.6)
ax.set_xlabel('pIC50', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Histogram')
ax.legend(fontsize=9)
ax.text(0.02, 0.97, f'n = {len(agg):,}\nμ = {agg["pIC50"].mean():.2f}\nσ = {agg["pIC50"].std():.2f}',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax = axes[1]
agg['pIC50'].plot.kde(ax=ax, color='#1976D2', lw=2.5)
kde_line = ax.get_lines()[0]
kx, ky = kde_line.get_data()
ax.fill_between(kx, ky, alpha=0.15, color='#1976D2')
for v, c, lbl in [(5,'#F44336','Inactive/Active\nboundary (5)'),
                   (6,'#FF9800','Active/Potent\nboundary (6)'),
                   (7,'#388E3C','Potent/Highly Potent\nboundary (7)')]:
    ax.axvline(v, color=c, ls='--', lw=1.5, alpha=0.8, label=lbl)
ax.set_xlabel('pIC50', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Kernel Density Estimate')
ax.legend(fontsize=8)

ax = axes[2]
class_order = ['Inactive (<5)', 'Weakly Active (5–6)', 'Active (6–7)', 'Highly Active (>7)']
cc = agg['Activity_Class'].value_counts().reindex(class_order).fillna(0)
colors_pie = ['#EF5350','#FF9800','#66BB6A','#1976D2']
wedges, texts, autotexts = ax.pie(
    cc.values, labels=[f'{l}\n(n={int(v):,})' for l, v in zip(class_order, cc.values)],
    autopct='%1.1f%%', colors=colors_pie,
    startangle=140, textprops={'fontsize': 8},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title('Activity Class Distribution')

plt.tight_layout()
fig.savefig(f'{FIGS}/02_pIC50_distribution.png')
plt.close()
log.info("  Saved: 02_pIC50_distribution.png")

# ── Fig 3: IC50 endpoint coverage (bar chart) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ep_labels = ['IC50\n(all)', 'IC50\n(exact)', 'IC50\n(censored)', 'Ki\n(exact)', 'EC50\n(exact)']
ep_vals   = [stats['05_ic50_total'], stats['05_ic50_exact'],
             stats['05_ic50_censored'], stats['05_ki_exact'], stats['05_ec50_exact']]
ep_cols   = ['#1976D2','#42A5F5','#EF5350','#388E3C','#FF9800']
bars = ax.bar(ep_labels, ep_vals, color=ep_cols, edgecolor='white', linewidth=0.5, width=0.6)
for bar, val in zip(bars, ep_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of Entries')
ax.set_title('Binding Data Coverage by Endpoint — P10275\n(After PROTAC & Mutant Exclusion)', fontweight='bold')
ax.set_ylim(0, max(ep_vals) * 1.15)
plt.tight_layout()
fig.savefig(f'{FIGS}/03_endpoint_coverage.png')
plt.close()
log.info("  Saved: 03_endpoint_coverage.png")

# ── Fig 4: Physicochemical properties (2×4 grid) ──────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(f'Physicochemical Properties — P10275 Dataset (n = {len(agg):,})',
             fontsize=13, fontweight='bold')
axes = axes.flatten()

prop_cfg = [
    ('MW',           'Molecular Weight (Da)',    500,  'Ro5: ≤500'),
    ('LogP',         'LogP',                     5,    'Ro5: ≤5'),
    ('HBD',          'H-Bond Donors',            5,    'Ro5: ≤5'),
    ('HBA',          'H-Bond Acceptors',         10,   'Ro5: ≤10'),
    ('TPSA',         'TPSA (Ų)',                140,  'Oral: ≤140'),
    ('RotBonds',     'Rotatable Bonds',          10,   '≤10'),
    ('AromaticRings','Aromatic Rings',           None, None),
    ('Fsp3',         'Fraction sp³ Carbon',      None, None),
]
pal = plt.cm.tab10.colors
for i, (col, xlabel, thresh, tlabel) in enumerate(prop_cfg):
    ax = axes[i]
    data = agg[col].dropna()
    ax.hist(data, bins=40, color=pal[i % len(pal)], edgecolor='white', linewidth=0.4, alpha=0.85)
    if thresh is not None:
        ax.axvline(thresh, color='crimson', ls='--', lw=1.8, label=tlabel)
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    n_pass = (data <= thresh).sum() if thresh else len(data)
    ax.set_title(f'{col}  μ={data.mean():.2f} | σ={data.std():.2f}', fontsize=10)

plt.tight_layout()
fig.savefig(f'{FIGS}/04_physicochemical_properties.png')
plt.close()
log.info("  Saved: 04_physicochemical_properties.png")

# ── Fig 5: pIC50 vs properties scatter ───────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('pIC50 vs Physicochemical Properties — P10275', fontsize=13, fontweight='bold')
axes = axes.flatten()
scatter_props = ['MW','LogP','HBD','HBA','TPSA','RotBonds']
for i, prop in enumerate(scatter_props):
    ax = axes[i]
    x = agg[prop].values
    y = agg['pIC50'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    sc = ax.scatter(x[mask], y[mask], c=y[mask], cmap='RdYlGn',
                    alpha=0.45, s=18, linewidths=0, vmin=4, vmax=10)
    plt.colorbar(sc, ax=ax, label='pIC50')
    r, p = pearsonr(x[mask], y[mask])
    m = np.polyfit(x[mask], y[mask], 1)
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xl, np.polyval(m, xl), 'k--', lw=1, alpha=0.5)
    ax.set_xlabel(prop, fontsize=11)
    ax.set_ylabel('pIC50', fontsize=11)
    ax.set_title(f'pIC50 vs {prop}\nr = {r:.3f}  p = {p:.2e}', fontsize=10)
plt.tight_layout()
fig.savefig(f'{FIGS}/05_pIC50_vs_properties.png')
plt.close()
log.info("  Saved: 05_pIC50_vs_properties.png")

# ── Fig 6: Correlation heatmap ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
corr_cols = ['pIC50','MW','LogP','HBD','HBA','TPSA','RotBonds','AromaticRings','Rings','Fsp3','HeavyAtoms']
corr_mat  = agg[corr_cols].corr(method='spearman')
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 9}, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8, 'label': 'Spearman ρ'})
ax.set_title('Spearman Correlation — pIC50 & Molecular Properties\nP10275 Androgen Receptor',
             fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig(f'{FIGS}/06_correlation_heatmap.png')
plt.close()
log.info("  Saved: 06_correlation_heatmap.png")

# ── Fig 7: IC50 raw & pIC50 boxplot by activity class ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('IC50 & pIC50 Distribution by Activity Class — P10275', fontsize=13, fontweight='bold')

ax = axes[0]
ax.hist(np.log10(agg['IC50_nM']), bins=50, color='#7B1FA2', edgecolor='white', linewidth=0.4, alpha=0.85)
ax.set_xlabel('log₁₀(IC50 [nM])')
ax.set_ylabel('Count')
ax.set_title('log₁₀(IC50) Distribution')
for v, lbl in [(-1,'0.1 nM'), (0,'1 nM'), (3,'1 µM'), (6,'1 mM')]:
    ax.axvline(v, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.text(v, ax.get_ylim()[1]*0.95, lbl, ha='center', fontsize=7, color='gray')

ax = axes[1]
class_order = ['Inactive (<5)', 'Weakly Active (5–6)', 'Active (6–7)', 'Highly Active (>7)']
bp_data = [agg.loc[agg['Activity_Class'] == c, 'pIC50'].values for c in class_order]
bp = ax.boxplot(bp_data, patch_artist=True, notch=False, widths=0.5,
                medianprops={'color': 'black', 'linewidth': 2},
                flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
box_colors = ['#EF5350','#FF9800','#66BB6A','#1976D2']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(class_order)+1))
ax.set_xticklabels([c for c in class_order], fontsize=8, rotation=10)
ax.set_ylabel('pIC50')
ax.set_title('pIC50 by Activity Class')
for i, (data, cls) in enumerate(zip(bp_data, class_order)):
    ax.text(i+1, agg['pIC50'].min()-0.2, f'n={len(data):,}', ha='center', fontsize=8, color='#555')

plt.tight_layout()
fig.savefig(f'{FIGS}/07_IC50_by_class.png')
plt.close()
log.info("  Saved: 07_IC50_by_class.png")

# ── Fig 8: Reproducibility of multi-measured compounds ────────────────────────
multi = agg[agg['n_measurements'] > 1].copy()
if len(multi) > 5:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Measurement Reproducibility — {len(multi):,} Multi-Measured Compounds',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    max_n = min(int(multi['n_measurements'].max()), 20)
    ax.hist(multi['n_measurements'], bins=range(2, max_n+2),
            color='#00ACC1', edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.set_xlabel('Number of IC50 Measurements per Compound')
    ax.set_ylabel('Count')
    ax.set_title('Measurement Frequency')

    ax = axes[1]
    std_vals = multi['IC50_nM_std'].fillna(0)
    ax.scatter(np.log10(multi['IC50_nM_mean']+1e-9), np.log10(std_vals+1e-9),
               alpha=0.4, s=20, c='#00ACC1', linewidths=0)
    ax.set_xlabel('log₁₀(Mean IC50 [nM])')
    ax.set_ylabel('log₁₀(Std IC50 [nM])')
    ax.set_title('Mean vs Std (log scale)\nInter-assay Variability')

    plt.tight_layout()
    fig.savefig(f'{FIGS}/08_reproducibility.png')
    plt.close()
    log.info("  Saved: 08_reproducibility.png")

# ══════════════════════════════════════════════════════════════════════════════
# 15. SAVE STATS JSON
# ══════════════════════════════════════════════════════════════════════════════
with open(f'{LOG_DIR}/data_processing_stats.json', 'w') as f:
    json.dump({k: (str(v) if not isinstance(v, (int, float, dict, list, bool)) else v)
               for k, v in stats.items()}, f, indent=2)

log.info("\n" + "=" * 70)
log.info("STEP 1 COMPLETE — DATA PROCESSING")
log.info(f"  Raw entries:          {stats['01_raw_rows']:,}")
log.info(f"  PROTAC excluded:      {stats['02_protac_excluded']:,}")
log.info(f"  Mutants excluded:     {stats['03_mutant_excluded']:,}")
log.info(f"  Censored IC50 excl.:  {stats['05_ic50_censored']:,}")
log.info(f"  Final compounds:      {stats['final_compounds']:,}")
log.info(f"  pIC50 range:          {stats['pIC50_min']:.2f} – {stats['pIC50_max']:.2f}")
log.info(f"  Ro5 compliant:        {stats['Ro5_compliant']:,} ({stats['Ro5_pct']:.1f}%)")
log.info(f"  Figures saved to:     {FIGS}/")
log.info("=" * 70)
