"""
09b — Boltz-2 run only (steps 1-4 already done by 09_boltz_affinity.py)
Runs Boltz for all 4 libraries and parses results.
"""
import json, subprocess, logging, sys, time, re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE     = Path('/home/stalin/Downloads/BindingDB_P10275')
DOCK_DIR = BASE / 'docking'
boltz_base = DOCK_DIR / 'boltz'

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

log_path = DOCK_DIR / f'boltz_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

log.info("="*70)
log.info("Step 9b: Boltz-2 Prediction Run — AR (P10275) | 4 Libraries")
log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("="*70)

def safe_name(s):
    return re.sub(r'[^\w\-]', '_', str(s))[:60]

all_boltz_results = []

for lib in LIBRARIES:
    inp_dir = boltz_base / lib / 'inputs'
    out_dir = boltz_base / lib / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    n_yamls = len(list(inp_dir.glob('*.yaml')))
    log.info(f"\n{'='*60}")
    log.info(f"Library: {lib} | {n_yamls} compounds")
    log.info(f"{'='*60}")

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

    log.info(f"  Running Boltz-2 (--use_msa_server)...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(DOCK_DIR), text=True)
    elapsed = time.time() - t0
    log.info(f"  Boltz done: {elapsed/60:.1f} min (exit={result.returncode})")

    # ── Parse results ─────────────────────────────────────────────────────────
    pred_root = out_dir / 'boltz_results_inputs' / 'predictions'
    if not pred_root.exists():
        # scan for predictions dir
        for candidate in out_dir.rglob('predictions'):
            if candidate.is_dir():
                pred_root = candidate
                break

    df_lib = pd.read_csv(DOCK_CSV[lib]).sort_values('Best_Binding_Energy_kcal_mol').head(50)
    name_lookup = {safe_name(r['Name']): dict(r) for _, r in df_lib.iterrows()}

    records = []
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
                records.append({
                    'Library':                      lib,
                    'Name':                         row.get('Name', stem),
                    'Boltz2_Affinity':              round(float(affinity), 4),
                    'Best_Binding_Energy_kcal_mol': row.get('Best_Binding_Energy_kcal_mol', np.nan),
                    'pIC50_Consensus':              row.get('pIC50_Consensus', np.nan),
                    'MW':                           row.get('MW', np.nan),
                    'LogP':                         row.get('LogP', np.nan),
                    'QED':                          row.get('QED', np.nan),
                    'Lipinski_pass':                row.get('Lipinski_pass', np.nan),
                    'Activity_Class':               row.get('Activity_Class', ''),
                    'SMILES':                       row.get('SMILES', ''),
                })
            except Exception as e:
                log.warning(f"  Parse error {pred_dir.name}: {e}")
    else:
        log.warning(f"  predictions dir not found: {pred_root}")

    if records:
        df_boltz = (pd.DataFrame(records)
                    .sort_values('Boltz2_Affinity', ascending=False)
                    .reset_index(drop=True))
        df_boltz.insert(0, 'Boltz_Rank', df_boltz.index + 1)
        out_csv = boltz_base / lib / f'{lib.lower().replace("-","_")}_boltz_results.csv'
        df_boltz.to_csv(out_csv, index=False)
        log.info(f"  Saved: {len(df_boltz)} results → {out_csv.name}")
        log.info(f"  Best affinity: {df_boltz['Boltz2_Affinity'].max():.4f}")
        log.info(f"  Top 5:")
        for _, r in df_boltz.head(5).iterrows():
            log.info(f"    Rank {int(r['Boltz_Rank']):2d} | {str(r['Name'])[:40]:40s} | "
                     f"Affinity={r['Boltz2_Affinity']:.4f} | "
                     f"pIC50={r['pIC50_Consensus']:.3f} | "
                     f"Dock={r['Best_Binding_Energy_kcal_mol']:.3f} kcal/mol")
        all_boltz_results.append(df_boltz)
    else:
        log.warning(f"  No results parsed for {lib}")

# Cross-library summary
if all_boltz_results:
    df_all = (pd.concat(all_boltz_results, ignore_index=True)
              .sort_values('Boltz2_Affinity', ascending=False)
              .reset_index(drop=True))
    df_all.insert(0, 'Global_Rank', df_all.index + 1)
    df_all.to_csv(DOCK_DIR / 'boltz_summary_all_libraries.csv', index=False)
    log.info(f"\n{'='*70}")
    log.info("BOLTZ COMPLETE — Cross-library summary:")
    for lib in LIBRARIES:
        sub = df_all[df_all['Library'] == lib]
        if len(sub):
            b = sub.iloc[0]
            log.info(f"  {lib:12s} | Best: {str(b['Name'])[:35]:35s} | "
                     f"Affinity={b['Boltz2_Affinity']:.4f} | pIC50={b['pIC50_Consensus']:.3f}")
    log.info(f"{'='*70}")
