#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AIDO-D-PHY-I SI V4
Supplementary Figure S1 V3 STRICT

Purpose:
Generate a clean SI Figure S1 using only real-data activity-width summary tables.

Panels:
A. Global |mu|_B – W_B landscape across cancers and databases
B. Cancer-wise median W_B by database
C. Cancer-wise median coherence by database
D. Relationship between N_B and W_B

Key fixes from earlier versions:
1. Fixes NaN cancer/source_file problem.
2. Strictly scans only original activity-width summary files.
3. Excludes random tests, SF extracted files, Top50/ranked/figure intermediate files.
4. Keeps only clean database categories: Hallmark, GO:BP, Reactome, KEGG, WikiPathways, C2:CP, BioCarta, PID.
5. Saves an integrated clean table for inspection.

Author: ChatGPT for AIDO-D-PHY-I SI V4
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================

ROOT = Path(r"D:/AIDO-Temp")
OUTDIR = ROOT / "AIDO-D-PHY-I-SI-V4-FIGURES"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TABLE = OUTDIR / "SI_V4_FigureS1_V3_clean_activity_width_table.csv"
OUTPUT_PNG = OUTDIR / "SI_V4_FigureS1_V3_activity_width_structure.png"
OUTPUT_PDF = OUTDIR / "SI_V4_FigureS1_V3_activity_width_structure.pdf"
DIAG_FILE = OUTDIR / "SI_V4_FigureS1_V3_file_scan_diagnostics.csv"

# Use only original activity-width summary folders/files.
INCLUDE_FOLDER_PATTERNS = [
    r"AIDO-D-MeanFWHM_EXP-W0_",
    r"AIDO-D-PlanB-",
]

EXCLUDE_FOLDER_PATTERNS = [
    r"RANDOM",
    r"RandomTest",
    r"MASTER-INTEGRATED",
    r"SF-",
    r"SF_",
    r"FIGURE",
    r"QC",
    r"MANUSCRIPT",
    r"EXTRACTED",
    r"REBUILT",
    r"INTEGRATED",
]

# Strict filename rule.
# These are the likely original activity-width summary tables.
INCLUDE_FILE_PATTERNS = [
    r"ALL_gene_set_summary\.csv$",
    r"GO_BP_gene_set_summary\.csv$",
    r"Hallmark_gene_set_summary\.csv$",
    r"Reactome_gene_set_summary\.csv$",
    r"KEGG_gene_set_summary\.csv$",
    r"WikiPathways_gene_set_summary\.csv$",
    r"BioCarta_gene_set_summary\.csv$",
    r"PID_gene_set_summary\.csv$",
]

EXCLUDE_FILE_PATTERNS = [
    r"ranked",
    r"Top50",
    r"top",
    r"matrix",
    r"scores",
    r"patient",
    r"lambda",
    r"K_scale",
    r"random",
    r"database_summary",
    r"max_D",
    r"survival",
]


DATABASE_ORDER = ["Hallmark", "GO:BP", "Reactome", "KEGG", "WikiPathways", "C2:CP", "BioCarta", "PID"]
CANCER_ORDER = ["BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LAML", "LIHC",
                "LUAD", "LUSC", "OV", "PAAD", "PRAD", "STAD", "THCA", "UCEC"]


# =========================
# Helper functions
# =========================

def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def match_any(text, patterns):
    text = str(text)
    return any(re.search(p, text, flags=re.I) for p in patterns)


def should_use_file(path: Path):
    s = str(path).replace("\\", "/")
    fname = path.name

    if not match_any(s, INCLUDE_FOLDER_PATTERNS):
        return False
    if match_any(s, EXCLUDE_FOLDER_PATTERNS):
        return False
    if not match_any(fname, INCLUDE_FILE_PATTERNS):
        return False
    if match_any(fname, EXCLUDE_FILE_PATTERNS):
        return False
    return True


def infer_cancer_from_path(path: Path):
    s = str(path).replace("\\", "/").upper()
    for c in CANCER_ORDER + ["READ", "COADREAD", "LGG", "SKCM", "SARC", "CESC", "ESCA"]:
        if re.search(rf"(^|[^A-Z0-9]){c}([^A-Z0-9]|$)", s):
            return c
    return "UNKNOWN"


def infer_database_from_text(text):
    t = str(text).upper()
    if "HALLMARK" in t:
        return "Hallmark"
    if "GO_BP" in t or "GOBP" in t or "GO-BP" in t or "BIOLOGICAL_PROCESS" in t:
        return "GO:BP"
    if "REACTOME" in t:
        return "Reactome"
    if "KEGG" in t:
        return "KEGG"
    if "WIKIPATHWAY" in t:
        return "WikiPathways"
    if "BIOCARTA" in t:
        return "BioCarta"
    if "PID" in t:
        return "PID"
    if "C2CP" in t or "C2_CP" in t:
        return "C2:CP"
    return "Unknown"


def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return None


def pick_col(columns, candidates):
    norm_map = {norm_col(c): c for c in columns}
    keys = list(norm_map.keys())

    for cand in candidates:
        c = norm_col(cand)
        if c in norm_map:
            return norm_map[c]

    for cand in candidates:
        for k in keys:
            if re.search(cand, k, flags=re.I):
                return norm_map[k]

    return None


def standardize_table(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        return None

    cols = list(df.columns)

    database_col = pick_col(cols, [
        "database", "db", "source", "collection"
    ])

    observable_col = pick_col(cols, [
        "gene_set", "geneset", "gene_set_name", "pathway", "pathway_name",
        "bp", "bp_name", "term", "term_name", "observable", r".*gene.*set.*", r".*pathway.*"
    ])

    n_col = pick_col(cols, [
        "n_genes_mapped", "mapped_genes", "n_mapped", "gene_count",
        "n_genes", "num_genes", "set_size", "size", r".*mapped.*gene.*", r".*gene.*count.*"
    ])

    mu_col = pick_col(cols, [
        "mean_abs_mu_across_patients", "mean_abs_mu", "abs_mu_mean",
        "mean_abs_activity", "mu_abs", "abs_mu", "mean_mu_abs",
        "mean_mu", "mu", r".*abs.*mu.*", r".*mu.*"
    ])

    w_col = pick_col(cols, [
        "mean_W_across_patients", "mean_w", "w_mean",
        "mean_fwhm", "fwhm_mean", "w_b", "wb", "w", "fwhm",
        "width", "observable_width", r".*(fwhm|width).*", r"^w$"
    ])

    if observable_col is None or mu_col is None or w_col is None:
        return None

    n = len(df)
    out = pd.DataFrame(index=df.index)

    out["source_file"] = [str(path)] * n
    out["cancer"] = [infer_cancer_from_path(path)] * n

    if database_col is not None:
        db_series = df[database_col].astype(str)
        out["database"] = db_series.apply(lambda x: infer_database_from_text(x) if infer_database_from_text(x) != "Unknown" else x)
    else:
        out["database"] = [infer_database_from_text(path.name)] * n

    # If database remains unknown, infer from filename.
    unknown_mask = out["database"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    if unknown_mask.any():
        out.loc[unknown_mask, "database"] = infer_database_from_text(path.name)

    out["observable"] = df[observable_col].astype(str)

    if n_col is not None:
        out["N_B"] = pd.to_numeric(df[n_col], errors="coerce")
    else:
        out["N_B"] = np.nan

    out["abs_mu"] = pd.to_numeric(df[mu_col], errors="coerce").abs()
    out["W"] = pd.to_numeric(df[w_col], errors="coerce")

    # Operational coherence for cohort-level summary.
    out["coherence"] = out["abs_mu"] / (out["W"].abs() + 1e-8)

    for col in ["N_B", "abs_mu", "W", "coherence"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=["cancer", "database", "observable", "abs_mu", "W"])
    out = out[(out["cancer"] != "UNKNOWN") & (out["database"] != "Unknown")]
    out = out[(out["W"] >= 0) & (out["abs_mu"] >= 0)]

    # Keep only known database groups for cleaner SI figure.
    out = out[out["database"].isin(DATABASE_ORDER)]

    if len(out) < 3:
        return None

    return out


def collect_clean_tables():
    diagnostics = []
    tables = []

    candidates = []
    for ext in ["*.csv", "*.tsv", "*.txt"]:
        candidates.extend(ROOT.rglob(ext))

    candidates = [p for p in candidates if should_use_file(p)]

    print(f"[INFO] Strict scan found {len(candidates)} candidate activity-width files.")

    for p in candidates:
        df = safe_read_csv(p)
        if df is None or df.empty:
            diagnostics.append({
                "file": str(p), "status": "read_failed_or_empty",
                "n_rows": 0, "used_rows": 0, "columns": ""
            })
            continue

        std = standardize_table(df, p)

        if std is not None and len(std) > 0:
            tables.append(std)
            diagnostics.append({
                "file": str(p), "status": "USED",
                "n_rows": len(df), "used_rows": len(std),
                "cancer": std["cancer"].iloc[0],
                "databases": ";".join(sorted(std["database"].unique())),
                "columns": " | ".join(map(str, df.columns))
            })
            print(f"[USED] {p} | rows={len(std)} | cancer={std['cancer'].iloc[0]} | db={sorted(std['database'].unique())}")
        else:
            diagnostics.append({
                "file": str(p), "status": "not_usable_after_standardization",
                "n_rows": len(df), "used_rows": 0,
                "cancer": infer_cancer_from_path(p),
                "databases": "",
                "columns": " | ".join(map(str, df.columns))
            })

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(DIAG_FILE, index=False, encoding="utf-8-sig")

    if not tables:
        return pd.DataFrame()

    out = pd.concat(tables, ignore_index=True)

    # Deduplicate exact repeats.
    out = out.drop_duplicates(subset=["cancer", "database", "observable", "N_B", "abs_mu", "W"])

    # Use only cancers with at least one meaningful database.
    out = out[out["cancer"].isin(CANCER_ORDER)]

    return out


def plot_heatmap(ax, pivot, title, cbar_label, fig):
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    return im


def make_figure(df):
    df = df.copy()

    cancer_order = [c for c in CANCER_ORDER if c in set(df["cancer"])]
    db_order = [d for d in DATABASE_ORDER if d in set(df["database"])]

    fig = plt.figure(figsize=(15.5, 12.5))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.30)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # Panel A
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, db in enumerate(db_order):
        sub = df[df["database"] == db]
        # Downsample by database for visibility if very large
        if len(sub) > 6000:
            sub = sub.sample(6000, random_state=10 + i)
        axA.scatter(
            sub["abs_mu"], sub["W"],
            s=23, alpha=0.42, marker=markers[i % len(markers)],
            edgecolors="none", label=db
        )

    axA.set_xlabel(r"Cohort-level mean activity, $\overline{|\mu|}_B$")
    axA.set_ylabel(r"Cohort-level observable width, $\overline{W}_B$")
    axA.set_title(r"A | Global $\overline{|\mu|}_B$-$\overline{W}_B$ landscape", loc="left", fontweight="bold")
    axA.grid(True, alpha=0.25)
    axA.legend(frameon=False, fontsize=8, ncol=2)

    # Panel B
    pivot_w = (
        df.groupby(["cancer", "database"], as_index=False)["W"]
        .median()
        .pivot(index="cancer", columns="database", values="W")
        .reindex(index=cancer_order, columns=db_order)
    )
    plot_heatmap(
        axB, pivot_w,
        r"B | Cancer-wise median $\overline{W}_B$ by database",
        r"Median $\overline{W}_B$", fig
    )

    # Panel C
    pivot_c = (
        df.groupby(["cancer", "database"], as_index=False)["coherence"]
        .median()
        .pivot(index="cancer", columns="database", values="coherence")
        .reindex(index=cancer_order, columns=db_order)
    )
    plot_heatmap(
        axC, pivot_c,
        r"C | Cancer-wise coherence summary by database",
        r"Median coherence, $\overline{|\mu|}_B/(\overline{W}_B+\delta)$", fig
    )

    # Panel D
    dfd = df.dropna(subset=["N_B", "W"]).copy()
    dfd = dfd[(dfd["N_B"] > 0) & (dfd["W"] >= 0)]

    if len(dfd) > 15000:
        dfd_plot = dfd.sample(15000, random_state=22)
    else:
        dfd_plot = dfd

    axD.scatter(dfd_plot["N_B"], dfd_plot["W"], s=18, alpha=0.35, edgecolors="none")

    # Binned median trend
    try:
        q = np.linspace(0, 1, 18)
        bins = np.unique(np.quantile(dfd["N_B"], q))
        if len(bins) > 4:
            dfd["N_bin"] = pd.cut(dfd["N_B"], bins=bins, include_lowest=True, duplicates="drop")
            trend = dfd.groupby("N_bin", observed=True).agg(
                N_mid=("N_B", "median"),
                W_med=("W", "median")
            ).dropna()
            axD.plot(trend["N_mid"], trend["W_med"], linewidth=2.2, marker="o", label="Binned median")
            axD.legend(frameon=False, fontsize=8)
    except Exception as e:
        print(f"[WARN] Panel D trend failed: {e}")

    axD.set_xscale("log")
    axD.set_xlabel(r"Mapped genes within one observable, $N_B$")
    axD.set_ylabel(r"Cohort-level observable width, $\overline{W}_B$")
    axD.set_title(r"D | Relationship between $N_B$ and $\overline{W}_B$", loc="left", fontweight="bold")
    axD.grid(True, alpha=0.25)

    fig.suptitle(
        "Supplementary Figure S1. Complete real-data activity-width structure across cancers",
        fontsize=15, fontweight="bold", y=0.985
    )

    fig.text(
        0.5, 0.012,
        r"$N_B$ denotes mapped genes within one BP/pathway observable; $K$ is not used in this figure.",
        ha="center", fontsize=10
    )

    fig.savefig(OUTPUT_PNG, dpi=450, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = collect_clean_tables()

    if df.empty:
        print("[ERROR] No clean activity-width summary tables were found.")
        print(f"[INFO] Diagnostic file saved to: {DIAG_FILE}")
        return

    df.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")

    print("\n[INFO] Clean integrated table saved:")
    print(f"       {OUTPUT_TABLE}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Cancers: {sorted(df['cancer'].unique())}")
    print(f"[INFO] Databases: {sorted(df['database'].unique())}")

    make_figure(df)

    print("\n[DONE] Supplementary Figure S1 V3 generated:")
    print(f"       PNG: {OUTPUT_PNG}")
    print(f"       PDF: {OUTPUT_PDF}")
    print(f"       Diagnostics: {DIAG_FILE}")


if __name__ == "__main__":
    main()
