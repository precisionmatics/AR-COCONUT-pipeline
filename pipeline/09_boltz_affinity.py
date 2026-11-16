"""
Step 9: Boltz-2 Binding Affinity Prediction — AR (P10275), 4 Libraries
Target  : P10275 Androgen Receptor LBD (2AM9, 250 residues)
Input   : Top 50 per library by AutoDock-GPU binding energy (200 compounds total)
Engine  : Boltz-2 (conda env: boltz) | GPU
Pipeline:
  1. Post-process docking — extract best pose SDF + complex PDB for Mushroom/NAPRORE-CR/COCONUT
  2. Clean docking folders — remove intermediate filelist files
  3. Extract top 50 per library → top50 CSV
  4. Generate per-compound YAML inputs (protein + ligand + affinity property)
  5. Run Boltz-2 per library
  6. Parse affinity_pred_value → ranked results CSV per library + cross-library summary
Output  : docking/<Library>/boltz/  |  docking/boltz_summary_all_libraries.csv
"""

import os, sys, json, subprocess, shutil, logging, time, re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

BASE     = Path('/home/stalin/Downloads/BindingDB_P10275')
DOCK_DIR = BASE / 'docking'
REC_DIR  = DOCK_DIR / 'receptor'
REC_JSON = REC_DIR / 'ar_receptor.json'
ADGPU    = '/home/stalin/Downloads/Enamine/GIPR/autodock_gpu_ocl'
N_CORES  = cpu_count()

AR_SEQ = ("QPIFLNVLEAIEPGVVCAGHDNNQPDSFAALLSSLNELGERQLVHVVKWAKALPGFRNLHVDDQMAVIQ"
          "YSWMGLMVFAMGWRSFTNVNSRMLYFAPDLVFNEYRMHKSRMYSQCVRMRHLSQEFGWLQITPQEFLCM"
          "KALLLFSIIPVDGLKNQKFFDELRMNYIKELDRIIACKRKNPTSCSRRFYQLTKLLDSVQPIARELHQFTF"
          "DLLIKSHMVSVDFPEMMAEIISVQVPKILSGKVKPIYFHTQ")

LIBRARIES = ['Marine', 'Mushroom', 'NAPRORE-CR', 'COCONUT']

DOCK_CSV = {
    'Marine':     DOCK_DIR / 'Marine/results/marine_docking_results.csv',
    'Mushroom':   DOCK_DIR / 'Mushroom/results/mushroom_docking_results.csv',
    'NAPRORE-CR': DOCK_DIR / 'NAPRORE-CR/results/naprore_cr_docking_results.csv',
    'COCONUT':    DOCK_DIR / 'COCONUT/results/coconut_docking_results.csv',
}

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = DOCK_DIR / f'boltz_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

log.info("="*70)
log.info("Step 9: Boltz-2 Affinity Prediction — AR (P10275) | 4 Libraries")
log.info(f"Sequence : AR LBD (2AM9), {len(AR_SEQ)} residues")
log.info(f"Input    : Top 50 per library by AutoDock-GPU energy")
log.info(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("="*70)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Post-process docking: poses + complexes for Mushroom/NAPRORE-CR/COCONUT
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\n[1] Post-processing docking — extracting best pose SDF + complex PDB...")

def extract_poses(lib_name, top_n=20):
    lig_dir     = DOCK_DIR / lib_name / 'ligands'
    res_dir     = DOCK_DIR / lib_name / 'results'
    poses_dir   = res_dir / 'poses';    poses_dir.mkdir(exist_ok=True)
    complex_dir = res_dir / 'complexes'; complex_dir.mkdir(exist_ok=True)

    csv_path = DOCK_CSV[lib_name]
    if not csv_path.exists():
        log.warning(f"  {lib_name}: results CSV not found, skipping pose extraction")
        return 0

    df = pd.read_csv(csv_path).sort_values('Best_Binding_Energy_kcal_mol').head(top_n)
    saved = 0
    for _, row in df.iterrows():
        name  = str(row['Name'])
        dlg   = lig_dir / f'{name}.dlg'
        gbest = lig_dir / f'{name}_best.pdbqt'
        sdf   = poses_dir / f'{name}_best_pose.sdf'
        cplx  = complex_dir / f'{name}_complex.pdb'
        if gbest.exists() and not (poses_dir / f'{name}_best.pdbqt').exists():
            shutil.copy(gbest, poses_dir / f'{name}_best.pdbqt')
        if dlg.exists() and not sdf.exists():
            r = subprocess.run(
                ['mk_export.py', str(dlg), '-s', str(sdf), '-p', str(cplx),
                 '-j', str(REC_JSON)],
                capture_output=True, text=True, timeout=60)
            if sdf.exists(): saved += 1
    log.info(f"  {lib_name}: {saved}/{top_n} pose SDF + complex PDB saved")
    return saved

for lib in ['Mushroom', 'NAPRORE-CR', 'COCONUT']:
    extract_poses(lib, top_n=20)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Clean docking folders: remove intermediate filelist files
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\n[2] Cleaning docking folders (moving filelists to logs/)...")
for lib in LIBRARIES:
    res_dir = DOCK_DIR / lib / 'results'
    log_dir = DOCK_DIR / lib / 'logs'
    log_dir.mkdir(exist_ok=True)
    for f in res_dir.glob('filelist_batch*.txt*'):
        shutil.move(str(f), str(log_dir / f.name))
log.info("  Done — filelist files moved to logs/")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Extract top 50 per library
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\n[3] Extracting top 50 per library by docking energy...")

top50_all = []
lib_top50 = {}

for lib in LIBRARIES:
    csv_path = DOCK_CSV[lib]
    if not csv_path.exists():
        log.warning(f"  {lib}: CSV not found"); continue
    df = pd.read_csv(csv_path).sort_values('Best_Binding_Energy_kcal_mol').head(50).copy()
    df.insert(0, 'Boltz_Input_Rank', range(1, len(df)+1))
    lib_top50[lib] = df
    top50_path = DOCK_DIR / lib / 'results' / f'{lib.lower().replace("-","_")}_top50.csv'
    df.to_csv(top50_path, index=False)
    log.info(f"  {lib}: top 50 | energy range {df['Best_Binding_Energy_kcal_mol'].min():.3f}"
             f" – {df['Best_Binding_Energy_kcal_mol'].max():.3f} kcal/mol → {top50_path.name}")
    top50_all.append(df)

df_top50_combined = pd.concat(top50_all, ignore_index=True)
df_top50_combined.to_csv(DOCK_DIR / 'top50_combined_all_libraries.csv', index=False)
log.info(f"  Combined top50 saved: {len(df_top50_combined)} compounds total")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Generate Boltz YAML inputs per library
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\n[4] Generating Boltz-2 YAML inputs...")

def safe_name(s):
    return re.sub(r'[^\w\-]', '_', str(s))[:60]

boltz_base = DOCK_DIR / 'boltz'
boltz_base.mkdir(exist_ok=True)

for lib, df in lib_top50.items():
    inp_dir = boltz_base / lib / 'inputs'
    out_dir = boltz_base / lib / 'outputs'
    inp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for _, row in df.iterrows():
        name   = safe_name(row['Name'])
        smiles = str(row['SMILES']).split(' |')[0].strip()
        yaml_path = inp_dir / f'{name}.yaml'
        if yaml_path.exists(): continue
        yaml_path.write_text(
            f"version: 1\n"
            f"sequences:\n"
            f"  - protein:\n"
            f"      id: A\n"
            f"      sequence: {AR_SEQ}\n"
            f"  - ligand:\n"
            f"      id: B\n"
            f"      smiles: '{smiles}'\n"
            f"properties:\n"
            f"  - affinity:\n"
            f"      binder: B\n"
        )
        written += 1
    log.info(f"  {lib}: {written} YAML files written → boltz/{lib}/inputs/")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Run Boltz-2 per library
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\n[5] Running Boltz-2 affinity predictions (GPU)...")

all_boltz_results = []

for lib in LIBRARIES:
    if lib not in lib_top50: continue
    inp_dir = boltz_base / lib / 'inputs'
    out_dir = boltz_base / lib / 'outputs'
    n_yamls = len(list(inp_dir.glob('*.yaml')))
    log.info(f"\n  Library: {lib} | {n_yamls} compounds")

    cmd = [
        'conda', 'run', '-n', 'boltz', '--no-capture-output',
        'boltz', 'predict', str(inp_dir),
        '--out_dir', str(out_dir),
        '--accelerator', 'gpu',
        '--devices', '1',
        '--model', 'boltz2',
        '--use_msa_server',
        '--diffusion_samples', '1',
        '--diffusion_samples_affinity', '5',
        '--sampling_steps', '100',
        '--sampling_steps_affinity', '100',
        '--affinity_mw_correction',
        '--recycling_steps', '3',
        '--output_format', 'pdb',
        '--max_parallel_samples', '1',
    ]

    t0 = time.time()
    log.info(f"  CMD: {' '.join(cmd[-10:])}")
    result = subprocess.run(cmd, cwd=str(DOCK_DIR),
                            capture_output=False, text=True)
    elapsed = time.time() - t0
    log.info(f"  Boltz done: {elapsed/60:.1f} min (exit={result.returncode})")

    # ── Parse affinity results ────────────────────────────────────────────────
    # Boltz output: out_dir/boltz_results_inputs/predictions/<name>/affinity_<name>.json
    pred_root = out_dir / 'boltz_results_inputs' / 'predictions'
    if not pred_root.exists():
        # Try alternative naming
        for d in out_dir.iterdir():
            if d.is_dir() and 'predictions' in str(list(d.iterdir())):
                pred_root = d / 'predictions'
                break

    records = []
    df_lib = lib_top50[lib]
    name_lookup = {safe_name(r['Name']): r for _, r in df_lib.iterrows()}

    if pred_root.exists():
        for pred_dir in sorted(pred_root.iterdir()):
            if not pred_dir.is_dir(): continue
            aff_files = list(pred_dir.glob('affinity_*.json'))
            if not aff_files: continue
            try:
                data = json.loads(aff_files[0].read_text())
                affinity = None
                if isinstance(data, dict):
                    affinity = (data.get('affinity_pred_value') or
                                data.get('pKd') or data.get('affinity'))
                elif isinstance(data, list) and data:
                    affinity = (data[0].get('affinity_pred_value') or
                                data[0].get('pKd'))
                if affinity is None: continue
                stem = pred_dir.name
                row  = name_lookup.get(stem, {})
                rec  = {'Library': lib, 'Name': stem,
                        'Boltz2_Affinity': round(float(affinity), 4),
                        'Best_Binding_Energy_kcal_mol': row.get('Best_Binding_Energy_kcal_mol', np.nan),
                        'pIC50_Consensus': row.get('pIC50_Consensus', np.nan),
                        'SMILES': row.get('SMILES', '')}
                records.append(rec)
            except Exception as e:
                log.warning(f"  Parse error {pred_dir.name}: {e}")
    else:
        log.warning(f"  predictions dir not found: {pred_root}")

    if records:
        df_boltz = pd.DataFrame(records).sort_values('Boltz2_Affinity',
                                                      ascending=False).reset_index(drop=True)
        df_boltz.insert(0, 'Boltz_Rank', df_boltz.index + 1)
        out_csv = boltz_base / lib / f'{lib.lower().replace("-","_")}_boltz_results.csv'
        df_boltz.to_csv(out_csv, index=False)
        log.info(f"  Parsed: {len(df_boltz)} results | "
                 f"Best affinity: {df_boltz['Boltz2_Affinity'].max():.4f}")
        log.info(f"  Top 5:")
        for _, r in df_boltz.head(5).iterrows():
            log.info(f"    Rank {int(r['Boltz_Rank']):2d} | {str(r['Name'])[:40]:40s} | "
                     f"Affinity={r['Boltz2_Affinity']:.4f} | "
                     f"pIC50={r['pIC50_Consensus']:.3f} | "
                     f"Dock={r['Best_Binding_Energy_kcal_mol']:.3f} kcal/mol")
        all_boltz_results.append(df_boltz)
    else:
        log.warning(f"  No Boltz results parsed for {lib}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Cross-library Boltz summary
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\n[6] Cross-library Boltz summary...")

if all_boltz_results:
    df_all = pd.concat(all_boltz_results, ignore_index=True)
    df_all = df_all.sort_values('Boltz2_Affinity', ascending=False).reset_index(drop=True)
    df_all.insert(0, 'Global_Rank', df_all.index + 1)
    summary_path = DOCK_DIR / 'boltz_summary_all_libraries.csv'
    df_all.to_csv(summary_path, index=False)
    log.info(f"\n  Cross-library summary saved: {summary_path}")
    log.info(f"  Total compounds: {len(df_all)}")
    log.info(f"\n  Per-library best hits:")
    for lib in LIBRARIES:
        sub = df_all[df_all['Library'] == lib]
        if len(sub):
            best = sub.iloc[0]
            log.info(f"    {lib:12s} | Best: {best['Name'][:35]} | "
                     f"Affinity={best['Boltz2_Affinity']:.4f} | "
                     f"pIC50={best['pIC50_Consensus']:.3f}")

log.info(f"\n{'='*70}")
log.info("BOLTZ PIPELINE COMPLETE")
log.info(f"{'='*70}")
