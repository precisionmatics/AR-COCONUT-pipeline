"""
Step 8: AutoDock-GPU Virtual Docking — AR (P10275), 4 Libraries
Target  : P10275 Androgen Receptor (2AM9, testosterone-bound LBD, 1.64 Å)
Ligands : Marine (100), Mushroom (100), NAPRORE-CR (100), COCONUT (1792)
Engine  : AutoDock-GPU v1.6 (OCL) | RTX 5060 Ti
Pipeline: SMILES → RDKit 3D → Meeko PDBQT → batch filelist → AutoDock-GPU → parse DLG
Output  : docking/<Library>/results/, logs/, summary CSV per library
"""

import os, sys, json, logging, time, subprocess, re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

BASE      = Path('/home/stalin/Downloads/BindingDB_P10275')
DOCK_DIR  = BASE / 'docking'
REC_DIR   = DOCK_DIR / 'receptor'
SUMM_DIR  = BASE / 'screening' / 'summary'
ADGPU     = '/home/stalin/Downloads/Enamine/GIPR/autodock_gpu_ocl'
MAPS_FLD  = str(REC_DIR / 'ar_receptor.maps.fld')
N_CORES   = cpu_count()
NRUN      = 20       # GA runs per ligand (higher = better sampling, GIPR used 5)
BATCH     = 500      # ligands per AutoDock-GPU call

LIBRARIES = {
    'Marine':     SUMM_DIR / 'marine_selected_hits.csv',
    'Mushroom':   SUMM_DIR / 'mushroom_selected_hits.csv',
    'NAPRORE-CR': SUMM_DIR / 'naprore_cr_selected_hits.csv',
    'COCONUT':    SUMM_DIR / 'coconut_selected_hits.csv',
}

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = DOCK_DIR / f'docking_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

log.info("="*70)
log.info("Step 8: AutoDock-GPU Docking — AR (P10275) | 4 Libraries")
log.info(f"Receptor  : 2AM9 LBD + testosterone pocket | grid 66x66x66 @ 0.375Å")
log.info(f"Engine    : AutoDock-GPU v1.6 (OCL) | {N_CORES} cores | nrun={NRUN}")
log.info(f"Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("="*70)

# ── Verify maps exist ─────────────────────────────────────────────────────────
if not Path(MAPS_FLD).exists():
    log.error(f"Grid maps not found: {MAPS_FLD} — is autogrid4 still running?")
    sys.exit(1)
log.info(f"Grid maps verified: {MAPS_FLD}")

# ── Ligand prep worker ────────────────────────────────────────────────────────
def prep_ligand(args):
    """SMILES → 3D SDF → Meeko PDBQT. Returns (name, pdbqt_path) or (name, None)."""
    import warnings; warnings.filterwarnings('ignore')
    from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    name, smiles, out_dir = args
    safe_name = re.sub(r'[^\w\-]', '_', str(name))[:60]
    pdbqt_path = Path(out_dir) / f'{safe_name}.pdbqt'
    if pdbqt_path.exists():
        return name, str(pdbqt_path)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return name, None
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if res != 0:
            res = AllChem.EmbedMolecule(mol, randomSeed=42)
        if res != 0: return name, None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        pdbqt_str, _, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
        pdbqt_path.write_text(pdbqt_str)
        return name, str(pdbqt_path)
    except Exception as e:
        return name, None

def save_best_poses_and_complexes(lig_dir, res_dir, df_dock, lib_name, top_n=20):
    """
    For top_n hits: use mk_export.py on DLG to write best-pose SDF + receptor-ligand complex PDB.
    Also copies best-pose PDBQT (_best.pdbqt from --gbest) into results/poses/.
    """
    import shutil
    poses_dir   = res_dir / 'poses'
    complex_dir = res_dir / 'complexes'
    poses_dir.mkdir(exist_ok=True)
    complex_dir.mkdir(exist_ok=True)

    receptor_pdbqt = str(REC_DIR / 'ar_receptor.pdbqt')
    saved = 0
    for _, row in df_dock.head(top_n).iterrows():
        name    = row['Name']
        dlg     = Path(lig_dir) / f'{name}.dlg'
        gbest   = Path(lig_dir) / f'{name}_best.pdbqt'
        sdf_out = poses_dir / f'{name}_best_pose.sdf'
        cplx_out= complex_dir / f'{name}_complex.pdb'

        # Copy gbest PDBQT if available
        if gbest.exists():
            shutil.copy(gbest, poses_dir / f'{name}_best.pdbqt')

        # Use mk_export.py: DLG → SDF (best pose) + complex PDB
        if dlg.exists():
            try:
                subprocess.run(
                    ['mk_export.py', str(dlg),
                     '-s', str(sdf_out),
                     '-p', str(cplx_out),
                     '-j', str(REC_DIR / 'ar_receptor.json')],
                    capture_output=True, text=True, timeout=60
                )
                if sdf_out.exists(): saved += 1
            except Exception as e:
                log.warning(f"  mk_export failed for {name}: {e}")

    log.info(f"  Best poses saved: {saved}/{min(top_n, len(df_dock))} → "
             f"results/poses/ (PDBQT+SDF) | results/complexes/ (PDB complex)")

def parse_xml_results(lig_dir):
    """Parse AutoDock-GPU XML files (written next to PDBQTs) → DataFrame."""
    import xml.etree.ElementTree as ET
    records = []
    for xml_path in Path(lig_dir).glob('*.xml'):
        name = xml_path.stem
        try:
            root = ET.parse(xml_path).getroot()
            energies = [float(e.find('free_NRG_binding').text.strip())
                        for e in root.iter('run')
                        if e.find('free_NRG_binding') is not None]
            if energies:
                records.append({'Name': name, 'Best_Binding_Energy_kcal_mol': min(energies)})
        except Exception:
            pass
    if not records:
        return pd.DataFrame(columns=['Dock_Rank', 'Name', 'Best_Binding_Energy_kcal_mol'])
    df = pd.DataFrame(records).sort_values('Best_Binding_Energy_kcal_mol').reset_index(drop=True)
    df.insert(0, 'Dock_Rank', df.index + 1)
    return df

# ── Main loop over libraries ──────────────────────────────────────────────────
all_summary = []

for lib_name, csv_path in LIBRARIES.items():
    t_lib = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"Library: {lib_name}")
    log.info(f"{'='*60}")

    if not csv_path.exists():
        log.warning(f"  CSV not found: {csv_path} — skipping")
        continue

    df = pd.read_csv(csv_path)
    log.info(f"  Compounds to dock: {len(df)}")

    lig_dir = DOCK_DIR / lib_name / 'ligands'
    res_dir = DOCK_DIR / lib_name / 'results'
    log_dir = DOCK_DIR / lib_name / 'logs'

    # ── Ligand preparation ────────────────────────────────────────────────────
    log.info(f"  [1] Preparing ligand PDBQTs ({N_CORES} cores)...")
    id_col  = 'Name' if 'Name' in df.columns else df.columns[2]
    smi_col = 'SMILES' if 'SMILES' in df.columns else 'canonical_smiles'
    args = [(row[id_col], row[smi_col], str(lig_dir)) for _, row in df.iterrows()]

    t_prep = time.time()
    with Pool(N_CORES) as pool:
        prep_results = pool.map(prep_ligand, args)

    valid = [(n, p) for n, p in prep_results if p is not None]
    failed = len(prep_results) - len(valid)
    log.info(f"  Prepared: {len(valid)} | Failed: {failed} | Time: {time.time()-t_prep:.1f}s")

    if not valid:
        log.warning(f"  No valid ligands — skipping {lib_name}")
        continue

    # ── Write batch filelists and run AutoDock-GPU ────────────────────────────
    # Filelist format: line 1 = maps.fld path, lines 2+ = ligand PDBQT paths
    # XMLs are written by AutoDock-GPU next to each input PDBQT (in lig_dir)
    log.info(f"  [2] Running AutoDock-GPU (nrun={NRUN}, batch={BATCH})...")
    batches = [valid[i:i+BATCH] for i in range(0, len(valid), BATCH)]
    log.info(f"  Total batches: {len(batches)}")

    t_dock = time.time()
    for b_idx, batch in enumerate(batches):
        flist_path = res_dir / f'filelist_batch{b_idx:04d}.txt'
        flist_lines = [MAPS_FLD]                          # first line = receptor maps
        for name, pdbqt_path in batch:
            flist_lines.append(pdbqt_path)                # subsequent lines = ligands
        flist_path.write_text('\n'.join(flist_lines) + '\n')

        cmd = [
            ADGPU,
            '--filelist', str(flist_path),
            '--nrun', str(NRUN),
            '--xmloutput', '1',
            '--dlgoutput', '1',   # full DLG with all poses (default on, explicit)
            '--gbest', '1',       # also write best-pose PDBQT per ligand
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(lig_dir))          # run from ligands dir
        done = min((b_idx+1)*BATCH, len(valid))
        elapsed = time.time() - t_dock
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(valid) - done) / rate if rate > 0 else 0
        log.info(f"  Batch {b_idx+1}/{len(batches)} | {done}/{len(valid)} docked | "
                 f"elapsed={elapsed:.1f}s | ETA={eta/60:.1f}min")
        if result.returncode != 0:
            log.warning(f"  Batch {b_idx} stderr: {result.stderr[-300:]}")

    dock_time = time.time() - t_dock
    log.info(f"  Docking done: {dock_time:.1f}s ({dock_time/60:.1f} min)")

    # ── Parse results ─────────────────────────────────────────────────────────
    log.info(f"  [3] Parsing XML results...")
    df_dock = parse_xml_results(str(lig_dir))
    log.info(f"  Parsed: {len(df_dock)} docked | "
             f"Best: {df_dock['Best_Binding_Energy_kcal_mol'].min():.3f} kcal/mol")

    # ── Save best poses + complexes for top 20 hits ───────────────────────────
    log.info(f"  [4] Saving best poses + receptor-ligand complexes (top 20)...")
    save_best_poses_and_complexes(lig_dir, res_dir, df_dock, lib_name, top_n=20)

    # Merge with original data
    df[id_col] = df[id_col].astype(str).str[:60].apply(lambda x: re.sub(r'[^\w\-]', '_', x))
    df_merged = df_dock.merge(df.rename(columns={id_col: 'Name'}), on='Name', how='left')
    df_merged.to_csv(res_dir / f'{lib_name.lower().replace("-","_")}_docking_results.csv',
                     index=False)

    top10 = df_dock.head(10)
    log.info(f"  Top 10 hits:")
    for _, row in top10.iterrows():
        log.info(f"    Rank {int(row['Dock_Rank']):3d} | {row['Name'][:40]:40s} | "
                 f"{row['Best_Binding_Energy_kcal_mol']:.3f} kcal/mol")

    summary = {
        'Library': lib_name,
        'n_input': len(df),
        'n_prepared': len(valid),
        'n_docked': len(df_dock),
        'best_energy_kcal_mol': round(float(df_dock['Best_Binding_Energy_kcal_mol'].min()), 3),
        'mean_energy_kcal_mol': round(float(df_dock['Best_Binding_Energy_kcal_mol'].mean()), 3),
        'top1_name': str(df_dock.iloc[0]['Name']),
        'runtime_min': round((time.time() - t_lib) / 60, 2),
    }
    all_summary.append(summary)
    with open(log_dir / f'{lib_name}_docking_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"  Library done: {(time.time()-t_lib)/60:.1f} min")

# ── Cross-library summary ─────────────────────────────────────────────────────
log.info(f"\n{'='*70}")
log.info("DOCKING COMPLETE — Cross-library Summary")
log.info(f"{'='*70}")
df_sum = pd.DataFrame(all_summary)
print(df_sum.to_string(index=False))
df_sum.to_csv(DOCK_DIR / 'docking_summary_all_libraries.csv', index=False)
log.info(f"\nSummary saved: {DOCK_DIR / 'docking_summary_all_libraries.csv'}")
log.info(f"Total runtime: {sum(s['runtime_min'] for s in all_summary):.1f} min")
