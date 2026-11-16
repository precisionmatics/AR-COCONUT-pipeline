"""
Script 13: Create md/ folder inside each library folder.
Contents per library: top 2 lead compound docked poses (SDF) + AR receptor (PDB).
"""

import shutil
from pathlib import Path

BASE   = Path("/home/stalin/Downloads/BindingDB_P10275")
SCREEN = BASE / "screening"
DOCK   = BASE / "docking"
RECEPTOR_PDB = DOCK / "receptor/2AM9_clean.pdb"

LIBRARIES = {
    "Marine": {
        "md_dir":   SCREEN / "Marine/md",
        "poses_dir": DOCK / "Marine/results/poses",
        "leads": [
            {
                "name":     "CMNPD17322",
                "pose_file":"CMNPD17322_best_pose.sdf",
                "boltz":    0.3722,
                "dock":     -13.38,
                "pIC50":    7.088,
            },
            {
                "name":     "CMNPD29753",
                "pose_file":"CMNPD29753_best_pose.sdf",
                "boltz":    -0.560,
                "dock":     -13.09,
                "pIC50":    7.192,
            },
        ],
    },
    "COCONUT": {
        "md_dir":   SCREEN / "COCONUT/md",
        "poses_dir": DOCK / "COCONUT/results/poses",
        "leads": [
            {
                "name":     "CNP0314682_1",
                "pose_file":"CNP0314682_1_best_pose.sdf",
                "boltz":    0.0560,
                "dock":     -13.67,
                "pIC50":    7.088,
            },
            {
                # CNP0249815_4 has no pose file — next best with pose
                "name":     "CNP0144365_2",
                "pose_file":"CNP0144365_2_best_pose.sdf",
                "boltz":    -0.4646,
                "dock":     -12.93,
                "pIC50":    7.188,
            },
        ],
    },
    "Mushroom": {
        "md_dir":   SCREEN / "Mushroom/md",
        "poses_dir": DOCK / "Mushroom/results/poses",
        "leads": [
            {
                "name":     "Ganoderic_acid_Z",
                "pose_file":"Ganoderic_acid_Z_best_pose.sdf",
                "boltz":    -0.912,
                "dock":     -13.38,
                "pIC50":    6.595,
            },
            {
                # Ganolucidic_acid_E has no pose — next best Boltz with pose
                "name":     "beta-carotene",
                "pose_file":"beta-carotene_best_pose.sdf",
                "boltz":    1.0673,
                "dock":     -12.29,
                "pIC50":    6.278,
            },
        ],
    },
    "NAPRORE-CR": {
        "md_dir":   SCREEN / "NAPRORE-CR/md",
        "poses_dir": DOCK / "NAPRORE-CR/results/poses",
        "leads": [
            {
                # cyclosativene has no SDF pose file; isolongifolene is top Boltz2 hit with pose
                "name":     "isolongifolene",
                "pose_file":"__isolongifolene___best_pose.sdf",
                "boltz":    1.4663,
                "dock":     -8.00,
                "pIC50":    6.689,
            },
            {
                "name":     "mint_sulfide",
                "pose_file":"__mint_sulfide____mint_sulfide___best_pose.sdf",
                "boltz":    0.6530,
                "dock":     -7.91,
                "pIC50":    6.784,
            },
        ],
    },
}


def create_md_folder(lib_name, cfg):
    print(f"\n{'='*55}")
    print(f"  {lib_name}")
    print(f"{'='*55}")

    md_dir = cfg["md_dir"]
    md_dir.mkdir(parents=True, exist_ok=True)

    copied = []

    # ── Copy protein PDB ─────────────────────────────────────────────────────
    receptor_dst = md_dir / "AR_receptor_2AM9.pdb"
    if RECEPTOR_PDB.exists():
        shutil.copy(RECEPTOR_PDB, receptor_dst)
        print(f"  Receptor PDB : AR_receptor_2AM9.pdb")
    else:
        print(f"  ERROR: Receptor PDB not found at {RECEPTOR_PDB}")

    # ── Copy lead compound SDF poses ─────────────────────────────────────────
    for rank, lead in enumerate(cfg["leads"], 1):
        src = cfg["poses_dir"] / lead["pose_file"]

        # destination: clean numbered name for MD clarity
        clean_name = lead["name"].replace(" ", "_").replace("/", "-")
        dst_name   = f"Lead{rank}_{clean_name}_docked_pose.sdf"
        dst        = md_dir / dst_name

        if src.exists():
            shutil.copy(src, dst)
            print(f"  Lead {rank} SDF    : {dst_name}")
            print(f"             Boltz2={lead['boltz']:.4f} | "
                  f"ΔG={lead['dock']:.2f} kcal/mol | "
                  f"pIC50={lead['pIC50']:.4f}")
            copied.append(dst_name)
        else:
            # try alternate name patterns
            alt_names = [
                f"__{lead['name']}___best_pose.sdf",
                f"{lead['name'].lower()}_best_pose.sdf",
            ]
            found = False
            for alt in alt_names:
                alt_src = cfg["poses_dir"] / alt
                if alt_src.exists():
                    shutil.copy(alt_src, dst)
                    print(f"  Lead {rank} SDF    : {dst_name}  (from alt: {alt})")
                    copied.append(dst_name)
                    found = True
                    break
            if not found:
                print(f"  Lead {rank} WARNING: pose file not found — {lead['pose_file']}")

    # ── Write README ──────────────────────────────────────────────────────────
    readme = md_dir / "README_MD_inputs.txt"
    lines  = [
        f"MD Simulation Input Files — {lib_name}",
        "="*55,
        "Target  : Androgen Receptor LBD (UniProt P10275)",
        "Crystal : PDB 2AM9 (resolution 1.90 Å)",
        "Protocol: GROMACS 2024.1 | AMBER99SB-ILDN + GAFF2 | 200 ns NPT",
        "",
        "FILES IN THIS FOLDER",
        "-"*40,
        "AR_receptor_2AM9.pdb",
        "  Clean AR LBD receptor structure (water/heteroatoms removed).",
        "  Use as input for GROMACS pdb2gmx with AMBER99SB-ILDN force field.",
        "",
    ]
    for rank, lead in enumerate(cfg["leads"], 1):
        clean = lead["name"].replace(" ", "_").replace("/", "-")
        lines += [
            f"Lead{rank}_{clean}_docked_pose.sdf",
            f"  Compound : {lead['name']}",
            f"  Source   : {lib_name}",
            f"  Boltz2   : {lead['boltz']:.4f}",
            f"  Docking  : {lead['dock']:.2f} kcal/mol (AutoDock Vina)",
            f"  pIC50    : {lead['pIC50']:.4f} (consensus ML)",
            f"  Pose     : Best docked pose (lowest ΔG) from AutoDock Vina.",
            f"  Usage    : Convert to PDB/MOL2 with Open Babel for ACPYPE parameterisation.",
            "",
        ]
    lines += [
        "SUGGESTED MD WORKFLOW",
        "-"*40,
        "1. Protein prep  : pdb2gmx -f AR_receptor_2AM9.pdb -ff amber99sb-ildn -water tip3p",
        "2. Ligand prep   : obabel Lead1_*.sdf -O lead1.mol2  →  acpype -i lead1.mol2 -c bcc",
        "3. Complex build : combine protein + ligand in one topology (GROMACS itp/top)",
        "4. Solvation     : editconf → solvate → genion (0.15 M NaCl, TIP3P)",
        "5. EM            : minim.mdp (steepest descent, 50,000 steps)",
        "6. NVT equil.    : nvt.mdp  (300 K, 100 ps, V-rescale, position restraints)",
        "7. NPT equil.    : npt.mdp  (300 K, 1 bar, 100 ps, Parrinello-Rahman)",
        "8. Production    : md.mdp   (300 K, 1 bar, 200 ns, 2 fs timestep, PME)",
        "9. Analysis      : gmx rms / rmsf / gyrate / hbond / mmpbsa",
        "",
        "Note: S6 docking pose PyMOL figures and MM-PBSA results (Figure S6, S7)",
        "      should be generated after MD completion and added to supplementary/.",
    ]

    readme.write_text("\n".join(lines))
    print(f"  README       : README_MD_inputs.txt")
    print(f"\n  Folder ready : {md_dir}")
    print(f"  Total files  : {len(list(md_dir.iterdir()))}")


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(LIBRARIES.keys())
    for lib in targets:
        if lib in LIBRARIES:
            create_md_folder(lib, LIBRARIES[lib])
        else:
            print(f"Unknown library: {lib}")
    print("\nAll MD folders prepared.")
