"""
Script 10: Compile per-library results into individual Excel files.
Each library gets its own .xlsx with sheets: Summary, ML_Screening, Docking, Boltz2
Output: placed in each library's results folder.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import (PatternFill, Font, Alignment, Border, Side,
                              GradientFill)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
import warnings
warnings.filterwarnings("ignore")

BASE = Path("/home/stalin/Downloads/BindingDB_P10275")
SCREEN = BASE / "screening"
DOCK   = BASE / "docking"

LIBRARIES = {
    "COCONUT": {
        "ml_file":    SCREEN / "COCONUT/results/coconut_all_screened.csv",
        "dock_file":  DOCK / "COCONUT/results/coconut_docking_results.csv",
        "boltz_file": DOCK / "boltz/COCONUT/coconut_boltz_results.csv",
        "out_dir":    SCREEN / "COCONUT/results",
        "out_name":   "COCONUT_all_results.xlsx",
        "color":      "2E86AB",   # blue
    },
    "Marine": {
        "ml_file":    SCREEN / "Marine/results/marine_all_screened.csv",
        "dock_file":  DOCK / "Marine/results/marine_docking_results.csv",
        "boltz_file": DOCK / "boltz/Marine/marine_boltz_results.csv",
        "out_dir":    SCREEN / "Marine/results",
        "out_name":   "Marine_all_results.xlsx",
        "color":      "1B998B",   # teal
    },
    "Mushroom": {
        "ml_file":    SCREEN / "Mushroom/results/mushroom_all_screened.csv",
        "dock_file":  DOCK / "Mushroom/results/mushroom_docking_results.csv",
        "boltz_file": DOCK / "boltz/Mushroom/mushroom_boltz_results.csv",
        "out_dir":    SCREEN / "Mushroom/results",
        "out_name":   "Mushroom_all_results.xlsx",
        "color":      "8B5E3C",   # brown
    },
    "NAPRORE-CR": {
        "ml_file":    SCREEN / "NAPRORE-CR/results/naprore_all_screened.csv",
        "dock_file":  DOCK / "NAPRORE-CR/results/naprore_cr_docking_results.csv",
        "boltz_file": DOCK / "boltz/NAPRORE-CR/naprore_cr_boltz_results.csv",
        "out_dir":    SCREEN / "NAPRORE-CR/results",
        "out_name":   "NAPRORE-CR_all_results.xlsx",
        "color":      "6A4C93",   # purple
    },
}

# ─── column display names ────────────────────────────────────────────────────
ML_RENAME = {
    "Rank": "ML Rank", "Name": "Compound ID", "SMILES": "SMILES",
    "pIC50_Consensus": "pIC50 (Consensus)", "pIC50_Std": "pIC50 Std Dev",
    "pIC50_Stacking": "pIC50 (Stacking)", "pIC50_XGBoost": "pIC50 (XGBoost)",
    "pIC50_CatBoost": "pIC50 (CatBoost)", "pIC50_LightGBM": "pIC50 (LightGBM)",
    "pIC50_ExtraTrees": "pIC50 (ExtraTrees)",
    "Activity_Class": "Activity Class",
    "MW": "MW (Da)", "LogP": "LogP", "HBD": "HBD", "HBA": "HBA",
    "TPSA": "TPSA (Å²)", "RotBonds": "Rotatable Bonds", "Rings": "Rings",
    "QED": "QED", "Lipinski_pass": "Lipinski", "Veber_pass": "Veber",
    "PAINS_flag": "PAINS Flag",
    "Max_Tanimoto": "Max Tanimoto (AD)", "In_Tanimoto_AD": "In Tanimoto AD",
    "kNN_Distance": "kNN Distance (AD)", "In_kNN_AD": "In kNN AD",
    "In_AD_Consensus": "In AD (Consensus)",
}

DOCK_RENAME = {
    "Dock_Rank": "Docking Rank", "Name": "Compound ID",
    "Best_Binding_Energy_kcal_mol": "Binding Energy (kcal/mol)",
    "Library": "Library", "pIC50_Consensus": "pIC50 (Consensus)",
    "Activity_Class": "Activity Class",
    "MW": "MW (Da)", "LogP": "LogP", "HBD": "HBD", "HBA": "HBA",
    "TPSA": "TPSA (Å²)", "RotBonds": "Rotatable Bonds", "Rings": "Rings",
    "QED": "QED", "Lipinski_pass": "Lipinski", "Veber_pass": "Veber",
    "PAINS_flag": "PAINS Flag",
    "Max_Tanimoto": "Max Tanimoto (AD)", "In_Tanimoto_AD": "In Tanimoto AD",
    "kNN_Distance": "kNN Distance (AD)", "In_kNN_AD": "In kNN AD",
    "In_AD_Consensus": "In AD (Consensus)",
    "pIC50_Stacking": "pIC50 (Stacking)", "pIC50_XGBoost": "pIC50 (XGBoost)",
    "pIC50_CatBoost": "pIC50 (CatBoost)", "pIC50_LightGBM": "pIC50 (LightGBM)",
    "pIC50_ExtraTrees": "pIC50 (ExtraTrees)", "pIC50_Std": "pIC50 Std Dev",
    "SMILES": "SMILES",
}

BOLTZ_RENAME = {
    "Boltz_Rank": "Boltz2 Rank", "Library": "Library", "Name": "Compound ID",
    "Boltz2_Affinity": "Boltz2 Affinity Score",
    "Best_Binding_Energy_kcal_mol": "Binding Energy (kcal/mol)",
    "pIC50_Consensus": "pIC50 (Consensus)",
    "Activity_Class": "Activity Class",
    "MW": "MW (Da)", "LogP": "LogP", "QED": "QED",
    "Lipinski_pass": "Lipinski", "SMILES": "SMILES",
}

# ─── helpers ─────────────────────────────────────────────────────────────────
def hex_fill(hex_color, tint=0):
    """Return PatternFill for a hex color (no tint support here, just the color)."""
    return PatternFill("solid", fgColor=hex_color)

def style_header_row(ws, row, bg_hex, font_color="FFFFFF"):
    fill = hex_fill(bg_hex)
    font = Font(bold=True, color=font_color, size=10)
    align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Side(style="thin", color="CCCCCC")
    border = Border(bottom=thin, right=thin)
    for cell in ws[row]:
        if cell.value is not None:
            cell.fill = fill
            cell.font = font
            cell.alignment = align
            cell.border = border

def style_data_rows(ws, start_row, end_row, zebra_hex="F5F5F5"):
    thin = Side(style="thin", color="E0E0E0")
    border = Border(bottom=thin, right=thin)
    for r in range(start_row, end_row + 1):
        fill = hex_fill(zebra_hex) if r % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
        for cell in ws[r]:
            if cell.column <= ws.max_column:
                cell.fill = fill
                cell.border = border
                cell.alignment = Alignment(horizontal="left", vertical="center",
                                           wrap_text=False)

def auto_col_width(ws, min_w=8, max_w=40):
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max((len(str(c.value)) if c.value else 0) for c in col)
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, min_w), max_w)

def freeze_and_filter(ws, freeze="A2"):
    ws.freeze_panes = freeze
    ws.auto_filter.ref = ws.dimensions

def write_df_to_sheet(ws, df, header_hex, zebra_hex="EFF6FF"):
    # write header
    for ci, col in enumerate(df.columns, 1):
        ws.cell(row=1, column=ci, value=col)
    # write data
    for ri, row_data in enumerate(df.itertuples(index=False), 2):
        for ci, val in enumerate(row_data, 1):
            ws.cell(row=ri, column=ci, value=val)
    style_header_row(ws, 1, header_hex)
    style_data_rows(ws, 2, len(df) + 1, zebra_hex)
    auto_col_width(ws)
    freeze_and_filter(ws)
    ws.row_dimensions[1].height = 30

def add_color_scale(ws, col_letter, start_row, end_row,
                    min_col="F8696B", mid_col="FFEB84", max_col="63BE7B"):
    rule = ColorScaleRule(
        start_type="min", start_color=min_col,
        mid_type="percentile", mid_value=50, mid_color=mid_col,
        end_type="max", end_color=max_col,
    )
    ws.conditional_formatting.add(f"{col_letter}{start_row}:{col_letter}{end_row}", rule)

# ─── summary sheet builder ────────────────────────────────────────────────────
def build_summary_sheet(ws, lib_name, ml_df, dock_df, boltz_df, color_hex):
    ws.sheet_view.showGridLines = False

    title_fill = hex_fill(color_hex)
    title_font = Font(bold=True, size=14, color="FFFFFF")
    sub_font   = Font(bold=True, size=11, color="333333")
    val_font   = Font(size=10, color="222222")
    hdr_fill   = hex_fill("D9E1F2")

    # Title
    ws.merge_cells("A1:F1")
    c = ws["A1"]
    c.value = f"Virtual Screening Summary — {lib_name} Library (Target: AR / P10275)"
    c.fill = title_fill; c.font = title_font
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 36

    ws.merge_cells("A2:F2")
    ws["A2"].value = "Androgen Receptor | Prostate Cancer | BindingDB P10275 Pipeline"
    ws["A2"].font = Font(italic=True, size=10, color="555555")
    ws["A2"].alignment = Alignment(horizontal="center")

    # ── Pipeline Overview ──
    row = 4
    ws.merge_cells(f"A{row}:F{row}")
    ws[f"A{row}"].value = "Pipeline Overview"
    ws[f"A{row}"].font = sub_font
    ws[f"A{row}"].fill = hdr_fill
    ws[f"A{row}"].alignment = Alignment(horizontal="left")
    ws.row_dimensions[row].height = 22
    row += 1

    pipeline_rows = [
        ("Step", "Method", "Input", "Output", "Tool/Model", "Key Metric"),
        ("1", "ML Virtual Screening", "Library SMILES", "pIC50 predictions",
         "Stacking Ensemble + 4 tuned models (ECFP4+6+MACCS, 4263-bit)",
         "pIC50 Consensus"),
        ("2", "Molecular Docking", "ML Top hits", "Binding energy (kcal/mol)",
         "AutoDock Vina | AR LBD (PDB: 2AM9)", "ΔG binding"),
        ("3", "Boltz2 Affinity", "Docking Top 50", "Boltz2 affinity score",
         "Boltz2 (structure-based ML)", "Boltz2 Affinity Score"),
    ]
    for pr in pipeline_rows:
        for ci, val in enumerate(pr, 1):
            cell = ws.cell(row=row, column=ci, value=val)
            if row == 5:
                cell.font = Font(bold=True, size=9, color="FFFFFF")
                cell.fill = hex_fill(color_hex)
                cell.alignment = Alignment(horizontal="center")
            else:
                cell.font = Font(size=9)
                cell.fill = PatternFill("solid", fgColor="F9F9F9") if row % 2 == 0 else PatternFill("solid", fgColor="FFFFFF")
                cell.alignment = Alignment(horizontal="left", wrap_text=True)
            ws.row_dimensions[row].height = 18
        row += 1

    row += 1

    # ── Counts Table ──
    ws.merge_cells(f"A{row}:F{row}")
    ws[f"A{row}"].value = "Compound Counts per Stage"
    ws[f"A{row}"].font = sub_font
    ws[f"A{row}"].fill = hdr_fill
    ws[f"A{row}"].alignment = Alignment(horizontal="left")
    ws.row_dimensions[row].height = 22
    row += 1

    count_headers = ("Stage", "Description", "Compounds", "", "", "")
    for ci, val in enumerate(count_headers, 1):
        cell = ws.cell(row=row, column=ci, value=val)
        cell.font = Font(bold=True, size=9, color="FFFFFF")
        cell.fill = hex_fill(color_hex)
        cell.alignment = Alignment(horizontal="center")
    ws.row_dimensions[row].height = 18
    row += 1

    counts = [
        ("ML Screening", "Top hits in Applicability Domain (AD)", len(ml_df)),
        ("Docking", "Compounds successfully docked", len(dock_df)),
        ("Boltz2", "Compounds with Boltz2 affinity prediction", len(boltz_df)),
    ]
    for stage, desc, count in counts:
        ws.cell(row=row, column=1, value=stage).font = Font(bold=True, size=9)
        ws.cell(row=row, column=2, value=desc).font = Font(size=9)
        ws.cell(row=row, column=3, value=count).font = Font(bold=True, size=11, color=color_hex)
        ws.cell(row=row, column=3).alignment = Alignment(horizontal="center")
        ws.row_dimensions[row].height = 18
        row += 1

    row += 1

    # ── Top Hits Summary ──
    ws.merge_cells(f"A{row}:F{row}")
    ws[f"A{row}"].value = "Top 10 Consensus Hits (Boltz2 → Docking → ML Ranked)"
    ws[f"A{row}"].font = sub_font
    ws[f"A{row}"].fill = hdr_fill
    ws[f"A{row}"].alignment = Alignment(horizontal="left")
    ws.row_dimensions[row].height = 22
    row += 1

    top_headers = ("Rank", "Compound ID", "Boltz2 Score", "ΔG Docking (kcal/mol)",
                   "pIC50 Consensus", "Activity Class")
    for ci, val in enumerate(top_headers, 1):
        cell = ws.cell(row=row, column=ci, value=val)
        cell.font = Font(bold=True, size=9, color="FFFFFF")
        cell.fill = hex_fill(color_hex)
        cell.alignment = Alignment(horizontal="center")
    ws.row_dimensions[row].height = 18
    row += 1

    # Merge boltz with docking/ml for top hits
    b_cols = {"Name": "Compound ID", "Boltz2_Affinity": "Boltz2 Score",
              "Best_Binding_Energy_kcal_mol": "ΔG Docking",
              "pIC50_Consensus": "pIC50", "Activity_Class": "Activity Class"}
    merged = boltz_df[list(b_cols.keys())].rename(columns=b_cols)
    merged = merged.sort_values("Boltz2 Score", ascending=False).head(10).reset_index(drop=True)

    for idx, r_data in merged.iterrows():
        rank = idx + 1
        fill_color = "FFF2CC" if rank <= 3 else ("F0F0F0" if rank % 2 == 0 else "FFFFFF")
        ws.cell(row=row, column=1, value=rank)
        ws.cell(row=row, column=2, value=r_data["Compound ID"])
        ws.cell(row=row, column=3, value=round(r_data["Boltz2 Score"], 4))
        ws.cell(row=row, column=4, value=round(r_data["ΔG Docking"], 3))
        ws.cell(row=row, column=5, value=round(r_data["pIC50"], 4))
        ws.cell(row=row, column=6, value=r_data["Activity Class"])
        for ci in range(1, 7):
            ws.cell(row=row, column=ci).fill = PatternFill("solid", fgColor=fill_color)
            ws.cell(row=row, column=ci).font = Font(size=9, bold=(rank <= 3))
            ws.cell(row=row, column=ci).alignment = Alignment(horizontal="center")
        ws.row_dimensions[row].height = 16
        row += 1

    row += 1

    # ── Key Stats ──
    ws.merge_cells(f"A{row}:F{row}")
    ws[f"A{row}"].value = "Key Statistics"
    ws[f"A{row}"].font = sub_font
    ws[f"A{row}"].fill = hdr_fill
    ws[f"A{row}"].alignment = Alignment(horizontal="left")
    ws.row_dimensions[row].height = 22
    row += 1

    stats = [
        ("Best pIC50 (ML)", f"{ml_df['pIC50_Consensus'].max():.4f}",
         "Highest predicted binding potency"),
        ("Best Docking ΔG", f"{dock_df['Best_Binding_Energy_kcal_mol'].min():.3f} kcal/mol",
         "Most negative = strongest predicted binding"),
        ("Best Boltz2 Score", f"{boltz_df['Boltz2_Affinity'].max():.4f}",
         "Highest Boltz2 affinity (positive = binding)"),
        ("Top Boltz2 Compound", boltz_df.loc[boltz_df['Boltz2_Affinity'].idxmax(), 'Name'],
         "Compound with highest Boltz2 score"),
        ("Top Docking Compound", dock_df.loc[dock_df['Best_Binding_Energy_kcal_mol'].idxmin(), 'Name'],
         "Compound with best docking energy"),
        ("Lipinski-passing (Boltz2 set)",
         f"{boltz_df['Lipinski_pass'].sum()} / {len(boltz_df)}",
         "Drug-likeness filter"),
    ]
    for stat_name, stat_val, stat_desc in stats:
        ws.cell(row=row, column=1, value=stat_name).font = Font(bold=True, size=9)
        ws.cell(row=row, column=2, value=stat_val).font = Font(size=9, color=color_hex, bold=True)
        ws.cell(row=row, column=3, value=stat_desc).font = Font(size=9, italic=True, color="666666")
        ws.merge_cells(f"C{row}:F{row}")
        ws.row_dimensions[row].height = 16
        row += 1

    # column widths
    for col, w in [(1, 22), (2, 28), (3, 22), (4, 22), (5, 18), (6, 28)]:
        ws.column_dimensions[get_column_letter(col)].width = w


# ─── main ─────────────────────────────────────────────────────────────────────
def process_library(lib_name, cfg):
    print(f"\n{'='*60}")
    print(f"  Processing: {lib_name}")
    print(f"{'='*60}")

    ml_df    = pd.read_csv(cfg["ml_file"])
    dock_df  = pd.read_csv(cfg["dock_file"])
    boltz_df = pd.read_csv(cfg["boltz_file"])
    color    = cfg["color"]

    # Build top-100 ML set: Active+Highly Active first, fill from Moderately Active if < 100
    if "Activity_Class" in ml_df.columns:
        priority    = ml_df[ml_df["Activity_Class"].isin(
                          ["Active (7–8)", "Highly Active (pIC50 ≥ 8)"]
                      )].sort_values("pIC50_Consensus", ascending=False)
        fill_pool   = ml_df[ml_df["Activity_Class"] == "Moderately Active (6–7)"]\
                          .sort_values("pIC50_Consensus", ascending=False)
        if len(priority) >= 100:
            ml_df = priority.head(100).reset_index(drop=True)
            fill_note = f"top 100 of {len(priority)} Active/Highly Active"
        else:
            needed = 100 - len(priority)
            ml_df = pd.concat([priority, fill_pool.head(needed)], ignore_index=True)
            fill_note = f"all {len(priority)} Active + top {min(needed, len(fill_pool))} Moderately Active"
        ml_df["Rank"] = range(1, len(ml_df) + 1)
    else:
        fill_note = "no Activity_Class column"

    # clean up name column consistency
    for df in [ml_df, dock_df, boltz_df]:
        if "Name" in df.columns:
            df["Name"] = df["Name"].astype(str).str.strip().str.replace("^__(.+)__$", r"\1", regex=True)

    print(f"  ML rows: {len(ml_df)}  ({fill_note})")
    print(f"  Dock rows:  {len(dock_df)}")
    print(f"  Boltz rows: {len(boltz_df)}")

    out_path = cfg["out_dir"] / cfg["out_name"]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # ── Sheet 1: Summary ──────────────────────────────────────────────────
        pd.DataFrame().to_excel(writer, sheet_name="Summary", index=False)

        # ── Sheet 2: ML Screening ─────────────────────────────────────────────
        ml_out = ml_df.rename(columns={k: v for k, v in ML_RENAME.items() if k in ml_df.columns})
        # reorder: rank + key cols first
        key_first = [c for c in ["ML Rank", "Compound ID", "pIC50 (Consensus)", "pIC50 Std Dev",
                                  "Activity Class", "MW (Da)", "LogP", "HBD", "HBA",
                                  "TPSA (Å²)", "QED", "Lipinski", "Veber", "PAINS Flag",
                                  "In AD (Consensus)", "Max Tanimoto (AD)", "In kNN AD",
                                  "pIC50 (Stacking)", "pIC50 (XGBoost)", "pIC50 (CatBoost)",
                                  "pIC50 (LightGBM)", "pIC50 (ExtraTrees)", "SMILES"]
                     if c in ml_out.columns]
        remaining = [c for c in ml_out.columns if c not in key_first]
        ml_out = ml_out[key_first + remaining]
        ml_out.to_excel(writer, sheet_name="ML_Screening", index=False)

        # ── Sheet 3: Docking ──────────────────────────────────────────────────
        dock_out = dock_df.rename(columns={k: v for k, v in DOCK_RENAME.items() if k in dock_df.columns})
        key_first_d = [c for c in ["Docking Rank", "Compound ID", "Binding Energy (kcal/mol)",
                                    "pIC50 (Consensus)", "Activity Class",
                                    "MW (Da)", "LogP", "HBD", "HBA", "TPSA (Å²)", "QED",
                                    "Lipinski", "Veber", "PAINS Flag",
                                    "In AD (Consensus)", "Max Tanimoto (AD)", "In kNN AD",
                                    "pIC50 (Stacking)", "pIC50 (XGBoost)", "pIC50 (CatBoost)",
                                    "pIC50 (LightGBM)", "pIC50 (ExtraTrees)", "SMILES"]
                       if c in dock_out.columns]
        rem_d = [c for c in dock_out.columns if c not in key_first_d]
        dock_out = dock_out[key_first_d + rem_d]
        dock_out.to_excel(writer, sheet_name="Docking", index=False)

        # ── Sheet 4: Boltz2 ───────────────────────────────────────────────────
        boltz_out = boltz_df.rename(columns={k: v for k, v in BOLTZ_RENAME.items() if k in boltz_df.columns})
        key_first_b = [c for c in ["Boltz2 Rank", "Compound ID", "Boltz2 Affinity Score",
                                    "Binding Energy (kcal/mol)", "pIC50 (Consensus)",
                                    "Activity Class", "MW (Da)", "LogP", "QED",
                                    "Lipinski", "SMILES"]
                       if c in boltz_out.columns]
        rem_b = [c for c in boltz_out.columns if c not in key_first_b]
        boltz_out = boltz_out[key_first_b + rem_b]
        boltz_out.to_excel(writer, sheet_name="Boltz2", index=False)

    # ── Post-process: apply styling ───────────────────────────────────────────
    wb = load_workbook(out_path)

    # Style ML sheet
    ws_ml = wb["ML_Screening"]
    style_header_row(ws_ml, 1, color)
    style_data_rows(ws_ml, 2, len(ml_out) + 1)
    auto_col_width(ws_ml)
    freeze_and_filter(ws_ml)
    ws_ml.row_dimensions[1].height = 30
    # color scale on pIC50 Consensus
    if "pIC50 (Consensus)" in ml_out.columns:
        col_idx = list(ml_out.columns).index("pIC50 (Consensus)") + 1
        col_l = get_column_letter(col_idx)
        add_color_scale(ws_ml, col_l, 2, len(ml_out) + 1)

    # Style Docking sheet
    ws_dock = wb["Docking"]
    style_header_row(ws_dock, 1, color)
    style_data_rows(ws_dock, 2, len(dock_out) + 1)
    auto_col_width(ws_dock)
    freeze_and_filter(ws_dock)
    ws_dock.row_dimensions[1].height = 30
    if "Binding Energy (kcal/mol)" in dock_out.columns:
        col_idx = list(dock_out.columns).index("Binding Energy (kcal/mol)") + 1
        col_l = get_column_letter(col_idx)
        # reverse scale for docking (more negative = better)
        rule = ColorScaleRule(
            start_type="min", start_color="63BE7B",
            mid_type="percentile", mid_value=50, mid_color="FFEB84",
            end_type="max", end_color="F8696B",
        )
        ws_dock.conditional_formatting.add(
            f"{col_l}2:{col_l}{len(dock_out)+1}", rule)

    # Style Boltz2 sheet
    ws_boltz = wb["Boltz2"]
    style_header_row(ws_boltz, 1, color)
    style_data_rows(ws_boltz, 2, len(boltz_out) + 1)
    auto_col_width(ws_boltz)
    freeze_and_filter(ws_boltz)
    ws_boltz.row_dimensions[1].height = 30
    if "Boltz2 Affinity Score" in boltz_out.columns:
        col_idx = list(boltz_out.columns).index("Boltz2 Affinity Score") + 1
        col_l = get_column_letter(col_idx)
        add_color_scale(ws_boltz, col_l, 2, len(boltz_out) + 1)

    # Build summary sheet
    ws_sum = wb["Summary"]
    ws_sum.sheet_view.showGridLines = False
    build_summary_sheet(ws_sum, lib_name, ml_df, dock_df, boltz_df, color)

    # Reorder sheets: Summary first
    sheet_order = ["Summary", "ML_Screening", "Docking", "Boltz2"]
    for i, name in enumerate(sheet_order):
        if name in wb.sheetnames:
            wb.move_sheet(name, offset=wb.sheetnames.index(name) - i)

    wb.save(out_path)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    for lib_name, cfg in LIBRARIES.items():
        process_library(lib_name, cfg)
    print("\nDone. All 4 Excel files created.")
