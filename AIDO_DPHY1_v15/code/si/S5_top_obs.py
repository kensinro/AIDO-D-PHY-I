#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AIDO-D-PHY-I SI V4
Supplementary Figure S5 (Revised V2.1)
Representative top-performing process-level observables across cancer cohorts

Panels:
A. Top 20 representative observables ranked by width-aware D_clinical
B. Maximum width-aware discriminability by cancer and database
C. Distribution of top-observable D_clinical by database
D. Number of top-ranked observables contributed by each database

Key fixes:
- Panel A uses horizontal bar chart to avoid text overflow.
- Extra left margin added for long labels.
- Panel C uses labels=labelsC instead of tick_labels=labelsC for older Matplotlib compatibility.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# User settings
# =========================================================

ROOT = Path(r"D:/AIDO-Temp")
OUTDIR = ROOT / "AIDO-D-PHY-I-SI-V4-FIGURES"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TABLE = OUTDIR / "SI_V4_FigureS5_REVISED_V21_integrated_table.csv"
OUTPUT_TOP = OUTDIR / "SI_V4_FigureS5_REVISED_V21_top_summary.csv"
OUTPUT_PNG = OUTDIR / "SI_V4_FigureS5_REVISED_V21_top_observables.png"
OUTPUT_PDF = OUTDIR / "SI_V4_FigureS5_REVISED_V21_top_observables.pdf"
DIAG_FILE = OUTDIR / "SI_V4_FigureS5_REVISED_V21_scan_diagnostics.csv"

CANCER_ORDER = [
    "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LAML", "LIHC",
    "LUAD", "LUSC", "OV", "PAAD", "PRAD", "STAD", "THCA", "UCEC"
]

DATABASE_ORDER = [
    "Hallmark", "GO:BP", "Reactome", "KEGG", "WikiPathways", "C2:CP", "BioCarta", "PID"
]

INCLUDE_PATH_KEYWORDS = [
    "PlanB",
    "MeanFWHM",
    "D-PHY",
    "REAL_DATA",
    "PHY-I",
    "Figures",
    "result",
    "summary",
    "ranked",
]

EXCLUDE_PATH_KEYWORDS = [
    "RANDOM-TEST",
    "RandomTest",
    "MASTER-INTEGRATED",
    "FIGURE4_REVISED",
    "QC-AUDIT",
    "SF-FIGURES",
    "SF-EXTRACTED",
    "lambda",
    "K_scale",
    "observation_lens",
]

PREFERRED_FILE_KEYWORDS = [
    "ranked",
    "summary",
    "top",
    "gene_set",
    "hallmark",
    "reactome",
    "go_bp",
    "kegg",
    "wikipathways",
    "all_gene_set_summary",
]

TOP_N_PANEL_A = 20


# =========================================================
# Helper functions
# =========================================================

def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def match_any(text, keywords):
    text = str(text).lower()
    return any(k.lower() in text for k in keywords)


def safe_read_table(path: Path):
    try:
        if path.suffix.lower() in [".tsv", ".txt"]:
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return None


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
    if "C2CP" in t or "C2_CP" in t or "CP_" in t:
        return "C2:CP"
    return "Unknown"


def pick_col(columns, exact_candidates=None, regex_candidates=None):
    exact_candidates = exact_candidates or []
    regex_candidates = regex_candidates or []

    norm_map = {norm_col(c): c for c in columns}
    keys = list(norm_map.keys())

    for cand in exact_candidates:
        c = norm_col(cand)
        if c in norm_map:
            return norm_map[c]

    for pat in regex_candidates:
        for k in keys:
            if re.fullmatch(pat, k, flags=re.I):
                return norm_map[k]

    for pat in regex_candidates:
        for k in keys:
            if re.search(pat, k, flags=re.I):
                return norm_map[k]

    return None


def should_scan_file(path: Path):
    s = str(path).replace("\\", "/")
    fname = path.name.lower()

    if not match_any(s, INCLUDE_PATH_KEYWORDS):
        return False
    if match_any(s, EXCLUDE_PATH_KEYWORDS):
        return False
    if path.suffix.lower() not in [".csv", ".tsv", ".txt"]:
        return False
    if not match_any(fname, PREFERRED_FILE_KEYWORDS):
        return False
    return True


def shorten(text, maxlen=36):
    text = str(text)
    if len(text) <= maxlen:
        return text
    return text[:maxlen - 1] + "…"


def standardize_observable_table(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        return None, "empty"

    cols = list(df.columns)

    cancer_col = pick_col(
        cols,
        exact_candidates=["cancer", "cohort", "dataset", "cancer_type", "tumor"],
        regex_candidates=[r"^cancer.*", r"^cohort$", r"^dataset$"]
    )

    db_col = pick_col(
        cols,
        exact_candidates=["database", "db", "source", "collection", "gene_set_database", "pathway_database"],
        regex_candidates=[r".*database.*", r"^db$", r"^source$"]
    )

    obs_col = pick_col(
        cols,
        exact_candidates=[
            "observable", "gene_set", "geneset", "gene_set_name",
            "pathway", "pathway_name", "bp", "bp_name", "term", "term_name"
        ],
        regex_candidates=[r".*gene.*set.*", r".*pathway.*", r".*observable.*", r".*term.*"]
    )

    d_width_col = pick_col(
        cols,
        exact_candidates=[
            "D_clinical", "clinical_D", "Dclinical",
            "D", "max_D", "best_D", "D_max",
            "neg_log10_p", "minus_log10_p", "logrank_D",
            "survival_D", "discriminability",
            "width_aware_D", "D_width_aware", "uncertainty_aware_D", "dw_phy"
        ],
        regex_candidates=[
            r"^d$",
            r".*clinical.*d.*",
            r".*max.*d.*",
            r".*log.*rank.*",
            r".*discrimin.*",
            r".*width.*aware.*d.*",
            r".*uncertainty.*aware.*d.*",
            r".*dw.*phy.*"
        ]
    )

    if obs_col is None or d_width_col is None:
        return None, "missing_required_observable_or_D"

    out = pd.DataFrame(index=df.index)
    out["source_file"] = [str(path)] * len(df)

    if cancer_col is not None:
        out["cancer"] = df[cancer_col].astype(str)
    else:
        out["cancer"] = infer_cancer_from_path(path)

    if db_col is not None:
        out["database"] = df[db_col].astype(str)
        out["database"] = out["database"].apply(
            lambda x: infer_database_from_text(x) if infer_database_from_text(x) != "Unknown" else str(x)
        )
    else:
        out["database"] = infer_database_from_text(path)

    out["observable"] = df[obs_col].astype(str)
    out["D_width"] = pd.to_numeric(df[d_width_col], errors="coerce")

    bad_db = out["database"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    out.loc[bad_db, "database"] = infer_database_from_text(path)

    bad_cancer = out["cancer"].astype(str).str.upper().isin(["UNKNOWN", "NAN", "NONE", ""])
    out.loc[bad_cancer, "cancer"] = infer_cancer_from_path(path)

    out["D_width"] = out["D_width"].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=["observable", "D_width"])
    out = out[out["D_width"] >= 0]
    out = out[out["cancer"].isin(CANCER_ORDER)]
    out = out[out["database"].isin(DATABASE_ORDER)]

    if out.empty:
        return None, "all_rows_filtered_out"

    return out, f"USED; observable={obs_col}; D_width={d_width_col}"


def collect_tables():
    diagnostics = []
    tables = []

    files = []
    for ext in ["*.csv", "*.tsv", "*.txt"]:
        files.extend(ROOT.rglob(ext))

    candidates = [p for p in files if should_scan_file(p)]
    print(f"[INFO] Scan found {len(candidates)} candidate files for revised Figure S5 V2.1.")

    for p in candidates:
        df = safe_read_table(p)
        if df is None or df.empty:
            diagnostics.append({
                "file": str(p),
                "status": "read_failed_or_empty",
                "n_rows": 0,
                "used_rows": 0,
                "columns": ""
            })
            continue

        std, status = standardize_observable_table(df, p)

        if std is not None and len(std) > 0:
            tables.append(std)
            diagnostics.append({
                "file": str(p),
                "status": status,
                "n_rows": len(df),
                "used_rows": len(std),
                "cancers": ";".join(sorted(std["cancer"].unique())),
                "databases": ";".join(sorted(std["database"].unique())),
                "columns": " | ".join(map(str, df.columns))
            })
            print(
                f"[USED] {p} | rows={len(std)} | "
                f"cancer={sorted(std['cancer'].unique())} | "
                f"db={sorted(std['database'].unique())}"
            )
        else:
            diagnostics.append({
                "file": str(p),
                "status": status,
                "n_rows": len(df),
                "used_rows": 0,
                "columns": " | ".join(map(str, df.columns))
            })

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(DIAG_FILE, index=False, encoding="utf-8-sig")

    if not tables:
        return pd.DataFrame()

    out = pd.concat(tables, ignore_index=True)
    out = out.drop_duplicates(subset=["cancer", "database", "observable", "D_width"])
    return out


def make_top_summary(df):
    grouped = (
        df.groupby(["cancer", "database", "observable"], as_index=False)["D_width"]
        .max()
    )

    idx = grouped.groupby(["cancer", "database"])["D_width"].idxmax()
    top = grouped.loc[idx].copy().sort_values(["cancer", "database"])

    return grouped, top


# =========================================================
# Figure drawing
# =========================================================

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


def make_figure(grouped, top):
    cancer_order = [c for c in CANCER_ORDER if c in set(top["cancer"])]
    db_order = [d for d in DATABASE_ORDER if d in set(top["database"])]

    fig = plt.figure(figsize=(16.0, 12.8))
    fig.subplots_adjust(left=0.24)
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    db_to_color = {}
    for i, db in enumerate(db_order):
        db_to_color[db] = default_colors[i % len(default_colors)]

    # -----------------------------------------------------
    # Panel A: top 20 representative observables
    # Horizontal bar chart to prevent text overflow.
    # -----------------------------------------------------
    top20 = top.sort_values("D_width", ascending=False).head(TOP_N_PANEL_A).copy()

    top20["label"] = top20.apply(
        lambda r: f"{r['cancer']} | {shorten(r['observable'], 36)}",
        axis=1
    )

    # Reverse so highest D appears at the top.
    top20 = top20.iloc[::-1].reset_index(drop=True)

    bar_colors = [db_to_color.get(db, "gray") for db in top20["database"]]

    axA.barh(
        np.arange(len(top20)),
        top20["D_width"],
        color=bar_colors,
        alpha=0.85
    )

    axA.set_yticks(np.arange(len(top20)))
    axA.set_yticklabels(top20["label"], fontsize=8)
    axA.set_xlabel(r"Width-aware clinical discriminability, $D_{\mathrm{clinical}}$")
    axA.set_ylabel("Representative top observables")
    axA.set_title(
        r"A | Top representative observables ranked by width-aware discriminability",
        loc="left",
        fontweight="bold"
    )
    axA.grid(True, axis="x", alpha=0.25)

    xmax = top20["D_width"].max()
    axA.set_xlim(0, xmax * 1.18)

    for i, v in enumerate(top20["D_width"]):
        axA.text(v + xmax * 0.015, i, f"{v:.1f}", va="center", fontsize=8)

    handles = []
    for db in db_order:
        if db in set(top20["database"]):
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="s",
                    linestyle="",
                    markersize=8,
                    markerfacecolor=db_to_color[db],
                    markeredgecolor=db_to_color[db],
                    label=db
                )
            )
    axA.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    # -----------------------------------------------------
    # Panel B: max width-aware D by cancer/database
    # -----------------------------------------------------
    pivotB = (
        top.pivot(index="cancer", columns="database", values="D_width")
        .reindex(index=cancer_order, columns=db_order)
    )
    plot_heatmap(
        axB, pivotB,
        r"B | Maximum width-aware discriminability by cancer and database",
        r"Maximum $D_{\mathrm{clinical}}$", fig
    )

    # -----------------------------------------------------
    # Panel C: distribution by database
    # Older Matplotlib compatibility: labels=labelsC, not tick_labels.
    # -----------------------------------------------------
    dataC = []
    labelsC = []
    for db in db_order:
        vals = top.loc[top["database"] == db, "D_width"].dropna().values
        if len(vals) > 0:
            dataC.append(vals)
            labelsC.append(db)

    if len(dataC) > 0:
        axC.boxplot(dataC, labels=labelsC, vert=True)
        axC.tick_params(axis="x", rotation=45)
    else:
        axC.text(
            0.5, 0.5,
            "No database-level distribution available",
            ha="center", va="center",
            transform=axC.transAxes,
            fontsize=10
        )

    axC.set_ylabel(r"Top-observable $D_{\mathrm{clinical}}$")
    axC.set_title(
        r"C | Distribution of top-observable discriminability by database",
        loc="left",
        fontweight="bold"
    )
    axC.grid(True, alpha=0.25)

    # -----------------------------------------------------
    # Panel D: count contributed by each database
    # -----------------------------------------------------
    countD = (
        top20["database"]
        .value_counts()
        .reindex(labelsC if len(labelsC) > 0 else db_order)
        .dropna()
    )

    if len(countD) > 0:
        bar_colors_D = [db_to_color.get(db, "gray") for db in countD.index]
        axD.bar(np.arange(len(countD)), countD.values, color=bar_colors_D, alpha=0.85)
        axD.set_xticks(np.arange(len(countD)))
        axD.set_xticklabels(countD.index, rotation=45, ha="right")
        for i, v in enumerate(countD.values):
            axD.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    else:
        axD.text(
            0.5, 0.5,
            "No database contribution summary available",
            ha="center", va="center",
            transform=axD.transAxes,
            fontsize=10
        )

    axD.set_ylabel("Count among displayed top observables")
    axD.set_title(
        r"D | Database contribution count among displayed top observables",
        loc="left",
        fontweight="bold"
    )
    axD.grid(True, alpha=0.25)

    fig.suptitle(
        "Supplementary Figure S5. Representative top-performing process-level observables across cancer cohorts",
        fontsize=15,
        fontweight="bold",
        y=0.985
    )

    fig.text(
        0.5,
        0.012,
        "Representative top observables were summarized by width-aware discriminability, with supplementary comparison across cancers and gene-set databases.",
        ha="center",
        fontsize=10
    )

    fig.savefig(OUTPUT_PNG, dpi=450, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = collect_tables()

    if df.empty:
        print("\n[ERROR] No usable summary tables were found for revised Figure S5 V2.1.")
        print(f"[INFO] Diagnostic file saved to: {DIAG_FILE}")
        print("\nNeed summary-like tables with columns similar to:")
        print("observable/gene_set/pathway + D_clinical/max_D/discriminability")
        return

    grouped, top = make_top_summary(df)

    grouped.to_csv(OUTPUT_TABLE, index=False, encoding="utf-8-sig")
    top.to_csv(OUTPUT_TOP, index=False, encoding="utf-8-sig")

    print("\n[INFO] Integrated observable-level table saved:")
    print(f"       {OUTPUT_TABLE}")
    print("[INFO] Top-observable summary saved:")
    print(f"       {OUTPUT_TOP}")
    print(f"[INFO] Rows: {len(grouped)}")
    print(f"[INFO] Top rows: {len(top)}")
    print(f"[INFO] Cancers: {sorted(top['cancer'].unique())}")
    print(f"[INFO] Databases: {sorted(top['database'].unique())}")

    make_figure(grouped, top)

    print("\n[DONE] Revised Supplementary Figure S5 V2.1 generated:")
    print(f"       PNG: {OUTPUT_PNG}")
    print(f"       PDF: {OUTPUT_PDF}")
    print(f"       Diagnostics: {DIAG_FILE}")


if __name__ == "__main__":
    main()
