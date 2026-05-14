# ============================================================
# AIDO-D-PHY Supplementary Figures 7–10 Builder
# Version: 2026-05-13
#
# Purpose:
#   Generate REAL-DATA supplementary figures from extracted CSV files.
#
# Input folder:
#   D:/AIDO-Temp/AIDO-D-PHY-SF-EXTRACTED
#
# Preferred input files:
#   SF7_top_KM_file_index.csv
#   SF8_highD_random_sets_all.csv
#   SF8_cross_cancer_gene_recurrence_in_highD_random.csv
#   SF8_cross_cancer_exact_random_signature_recurrence.csv
#   SF9_FWHM_W_matrix_file_index_for_robustness.csv
#   SF10_best_K_lambda_summary.csv
#   SF2_best_lambda_summary.csv
#   SF3_DK_curve_best_lambda_each_K.csv
#
# Output folder:
#   D:/AIDO-Temp/AIDO-D-PHY-SF-FIGURES
#
# Required packages:
#   pip install pandas numpy matplotlib
# ============================================================

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. Paths and settings
# ============================================================

INPUT_DIR = Path(r"D:/AIDO-Temp/AIDO-D-PHY-SF-EXTRACTED")
OUT_DIR = Path(r"D:/AIDO-Temp/AIDO-D-PHY-SF-FIGURES")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

CANCER_ORDER = [
    "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC",
    "LAML", "LIHC", "LUAD", "LUSC", "OV", "PAAD",
    "PRAD", "SARC", "SKCM", "STAD", "THCA", "UCEC"
]

DATABASE_ORDER = ["Hallmark", "GO_BP", "Reactome", "KEGG_Medicus", "WikiPathways"]


# ============================================================
# 1. Utilities
# ============================================================

def log(msg):
    print(f"[AIDO-D-PHY SF FIG7-10] {msg}", flush=True)


def read_optional(filename):
    path = INPUT_DIR / filename
    if not path.exists():
        log(f"Optional file missing: {path}")
        return None
    try:
        df = pd.read_csv(path)
        log(f"Loaded {filename}: {df.shape}")
        return df
    except Exception as e:
        log(f"Failed reading {path}: {e}")
        return None


def read_required(filename):
    path = INPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    log(f"Loaded {filename}: {df.shape}")
    return df


def savefig(name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    log(f"Saved: {png}")
    log(f"Saved: {pdf}")


def panel(ax, label):
    ax.text(
        -0.10, 1.08, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )


def order_cancers(values):
    values = set(str(v) for v in values)
    return [c for c in CANCER_ORDER if c in values] + sorted([c for c in values if c not in CANCER_ORDER])


def clean_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of candidate columns found: {candidates}. Available={list(df.columns)}")
    return None


def file_exists(path_string):
    try:
        return Path(str(path_string)).exists()
    except Exception:
        return False


# ============================================================
# 2. Supplementary Figure 7
# Top KM plot inventory / representative survival evidence index
# ============================================================

def make_sf7():
    """
    SF7 is a real-data figure index, not a montage of KM images.
    It summarizes generated KM plot availability by cancer and type.

    Input:
      SF7_top_KM_file_index.csv

    Output:
      SF7_top_KM_file_inventory.png/pdf
      SF7_available_KM_files_filtered.csv
    """

    df = read_optional("SF7_top_KM_file_index.csv")

    if df is None or df.empty:
        log("SF7 skipped: no KM index available.")
        return

    required = ["Cancer", "file_name", "file_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SF7 missing columns: {missing}")

    if "kind_guess" not in df.columns:
        df["kind_guess"] = "Unknown"

    df["exists_now"] = df["file_path"].apply(file_exists)
    df.to_csv(OUT_DIR / "SF7_available_KM_files_filtered.csv", index=False)

    # Count by cancer and kind
    count_table = (
        df.groupby(["Cancer", "kind_guess"])
        .size()
        .reset_index(name="count")
    )

    cancers = order_cancers(count_table["Cancer"])

    pivot = count_table.pivot_table(
        index="Cancer",
        columns="kind_guess",
        values="count",
        fill_value=0,
        aggfunc="sum"
    ).reindex(cancers)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    # A. stacked bar of available KM files
    ax = fig.add_subplot(gs[0, 0])
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_ylabel("Number of KM plots")
    ax.set_title("Available top-pathway KM plots by cancer")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    panel(ax, "A")

    # B. heatmap cancer x kind
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("KM plot availability matrix")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count")
    panel(ax, "B")

    # C. total available by cancer
    ax = fig.add_subplot(gs[1, 0])
    totals = pivot.sum(axis=1)
    ax.bar(np.arange(len(totals)), totals.values)
    ax.set_xticks(np.arange(len(totals)))
    ax.set_xticklabels(totals.index, rotation=60, fontsize=8)
    ax.set_ylabel("Total KM plots")
    ax.set_title("Total KM evidence files by cancer")
    panel(ax, "C")

    # D. file existence check
    ax = fig.add_subplot(gs[1, 1])
    exist_counts = df["exists_now"].value_counts().rename(index={True: "Exists", False: "Missing"})
    ax.bar(exist_counts.index.astype(str), exist_counts.values)
    ax.set_ylabel("File count")
    ax.set_title("KM file existence check")
    panel(ax, "D")

    fig.suptitle(
        "Supplementary Figure 7. Inventory of representative Kaplan–Meier survival evidence",
        fontsize=16,
        fontweight="bold"
    )

    savefig("SF7_top_KM_file_inventory")


# ============================================================
# 3. Supplementary Figure 8
# High-D random recurrence analysis
# ============================================================

def make_sf8():
    """
    SF8:
      A. Top recurrent genes in high-D random sets
      B. CancerCount distribution
      C. TotalHighDSetCount distribution
      D. Exact random signature recurrence, if available

    Inputs:
      SF8_cross_cancer_gene_recurrence_in_highD_random.csv
      SF8_cross_cancer_exact_random_signature_recurrence.csv [optional]
      SF8_highD_random_sets_all.csv [optional]
    """

    gene = read_optional("SF8_cross_cancer_gene_recurrence_in_highD_random.csv")
    exact = read_optional("SF8_cross_cancer_exact_random_signature_recurrence.csv")
    highd = read_optional("SF8_highD_random_sets_all.csv")

    if gene is None or gene.empty:
        log("SF8 skipped: no gene recurrence file available.")
        return

    required = ["Gene", "CancerCount", "TotalHighDSetCount"]
    missing = [c for c in required if c not in gene.columns]
    if missing:
        raise ValueError(f"SF8 gene recurrence missing columns: {missing}")

    gene = clean_numeric(gene, ["CancerCount", "TotalHighDSetCount"])
    gene = gene.dropna(subset=["Gene", "CancerCount", "TotalHighDSetCount"])
    gene = gene.sort_values(["CancerCount", "TotalHighDSetCount"], ascending=False)

    top_n = min(25, len(gene))
    top = gene.head(top_n).copy()

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2)

    # A. top recurrent genes
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["CancerCount"].values[::-1])
    ax.set_yticks(y)
    ax.set_yticklabels(top["Gene"].values[::-1], fontsize=8)
    ax.set_xlabel("Cancer count")
    ax.set_title("Top recurrent genes in high-D random sets")
    panel(ax, "A")

    # B. CancerCount distribution
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(gene["CancerCount"], bins=np.arange(0.5, gene["CancerCount"].max() + 1.5, 1), edgecolor="black")
    ax.set_xlabel("Cancer count")
    ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of cross-cancer recurrence")
    panel(ax, "B")

    # C. TotalHighDSetCount distribution
    ax = fig.add_subplot(gs[1, 0])
    vals = gene["TotalHighDSetCount"].dropna()
    ax.hist(vals, bins=50, edgecolor="black")
    ax.set_xlabel("Total high-D set count")
    ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of high-D random-set participation")
    panel(ax, "C")

    # D. exact signature recurrence or highD summary
    ax = fig.add_subplot(gs[1, 1])
    if exact is not None and not exact.empty and "RecurrenceCount" in exact.columns:
        exact = clean_numeric(exact, ["RecurrenceCount", "MaxD", "MeanD"])
        top_exact = exact.sort_values(["RecurrenceCount", "MaxD"], ascending=False).head(20)
        ax.bar(np.arange(len(top_exact)), top_exact["RecurrenceCount"])
        ax.set_xticks(np.arange(len(top_exact)))
        ax.set_xticklabels([f"S{i+1}" for i in range(len(top_exact))], rotation=60, fontsize=8)
        ax.set_ylabel("Recurrence count")
        ax.set_title("Exact high-D random signature recurrence")
    elif highd is not None and not highd.empty:
        d_col = get_col(highd, ["D", "D_clinical"], required=False)
        if d_col is not None:
            highd = clean_numeric(highd, [d_col])
            ax.hist(highd[d_col].dropna(), bins=50, edgecolor="black")
            ax.set_xlabel("D")
            ax.set_ylabel("Number of high-D random sets")
            ax.set_title("High-D random-set D distribution")
        else:
            ax.text(0.5, 0.5, "No D column available", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "No exact recurrence/high-D file available", ha="center", va="center")
        ax.set_axis_off()
    panel(ax, "D")

    fig.suptitle(
        "Supplementary Figure 8. High-discriminability random recurrence analysis",
        fontsize=16,
        fontweight="bold"
    )

    savefig("SF8_highD_random_recurrence_analysis")


# ============================================================
# 4. Supplementary Figure 9
# FWHM robustness / W matrix availability and distribution audit
# ============================================================

def make_sf9():
    """
    SF9:
      Uses W matrix file index and audits whether W matrices are available.
      If matrices are readable, estimates basic distribution summaries.

    Input:
      SF9_FWHM_W_matrix_file_index_for_robustness.csv

    Output:
      SF9_FWHM_W_matrix_audit_summary.csv
      SF9_FWHM_width_matrix_audit.png/pdf
    """

    idx = read_optional("SF9_FWHM_W_matrix_file_index_for_robustness.csv")
    if idx is None or idx.empty:
        log("SF9 skipped: no W matrix file index available.")
        return

    required = ["Cancer", "database", "file_path"]
    missing = [c for c in required if c not in idx.columns]
    if missing:
        raise ValueError(f"SF9 missing columns: {missing}")

    rows = []

    for _, r in idx.iterrows():
        path = Path(str(r["file_path"]))
        exists = path.exists()

        out = {
            "Cancer": r["Cancer"],
            "database": r["database"],
            "file_path": str(path),
            "exists": exists,
            "n_rows": np.nan,
            "n_cols": np.nan,
            "n_values": np.nan,
            "W_mean": np.nan,
            "W_median": np.nan,
            "W_q25": np.nan,
            "W_q75": np.nan,
            "W_min": np.nan,
            "W_max": np.nan
        }

        if exists:
            try:
                df = pd.read_csv(path, index_col=0)
                vals = pd.to_numeric(df.stack(), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                vals = vals[vals > 0]
                out.update({
                    "n_rows": df.shape[0],
                    "n_cols": df.shape[1],
                    "n_values": len(vals),
                    "W_mean": vals.mean() if len(vals) else np.nan,
                    "W_median": vals.median() if len(vals) else np.nan,
                    "W_q25": vals.quantile(0.25) if len(vals) else np.nan,
                    "W_q75": vals.quantile(0.75) if len(vals) else np.nan,
                    "W_min": vals.min() if len(vals) else np.nan,
                    "W_max": vals.max() if len(vals) else np.nan
                })
            except Exception as e:
                log(f"Failed reading W matrix {path}: {e}")

        rows.append(out)

    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_DIR / "SF9_FWHM_W_matrix_audit_summary.csv", index=False)
    log(f"Saved: {OUT_DIR / 'SF9_FWHM_W_matrix_audit_summary.csv'}")

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2)

    # A. availability by cancer/database
    ax = fig.add_subplot(gs[0, 0])
    avail = audit.pivot_table(index="Cancer", columns="database", values="exists", aggfunc="max", fill_value=False)
    cancers = order_cancers(avail.index)
    avail = avail.reindex(cancers)
    im = ax.imshow(avail.astype(int).values, aspect="auto")
    ax.set_xticks(np.arange(len(avail.columns)))
    ax.set_xticklabels(avail.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(avail.index)))
    ax.set_yticklabels(avail.index, fontsize=8)
    ax.set_title("W matrix availability")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Available")
    panel(ax, "A")

    # B. W median by cancer/database
    ax = fig.add_subplot(gs[0, 1])
    med = audit.pivot_table(index="Cancer", columns="database", values="W_median", aggfunc="median")
    med = med.reindex(cancers)
    im = ax.imshow(med.values, aspect="auto")
    ax.set_xticks(np.arange(len(med.columns)))
    ax.set_xticklabels(med.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(med.index)))
    ax.set_yticklabels(med.index, fontsize=8)
    ax.set_title("Median W across available matrices")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Median W")
    panel(ax, "B")

    # C. number of values
    ax = fig.add_subplot(gs[1, 0])
    valid = audit[audit["exists"] == True].copy()
    if not valid.empty:
        labels = valid["Cancer"].astype(str) + "-" + valid["database"].astype(str)
        ax.bar(np.arange(len(valid)), valid["n_values"])
        ax.set_xticks(np.arange(len(valid)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("Number of W values")
        ax.set_title("Readable W matrix sizes")
    else:
        ax.text(0.5, 0.5, "No readable W matrices", ha="center", va="center")
        ax.set_axis_off()
    panel(ax, "C")

    # D. W IQR range
    ax = fig.add_subplot(gs[1, 1])
    if not valid.empty:
        ax.scatter(valid["W_median"], valid["W_q75"] - valid["W_q25"], s=70)
        for _, r in valid.iterrows():
            label = f"{r['Cancer']}-{r['database']}"
            ax.annotate(label, (r["W_median"], r["W_q75"] - r["W_q25"]), fontsize=6, alpha=0.7)
        ax.set_xlabel("Median W")
        ax.set_ylabel("IQR of W")
        ax.set_title("W distribution spread across matrices")
    else:
        ax.text(0.5, 0.5, "No readable W matrices", ha="center", va="center")
        ax.set_axis_off()
    panel(ax, "D")

    fig.suptitle(
        "Supplementary Figure 9. FWHM width-matrix availability and robustness audit",
        fontsize=16,
        fontweight="bold"
    )

    savefig("SF9_FWHM_width_matrix_audit")


# ============================================================
# 5. Supplementary Figure 10
# Integrated optimal λ/K/D summary
# ============================================================

def make_sf10():
    """
    SF10:
      A. best K by cancer
      B. best lambda by cancer
      C. max D by cancer
      D. lambda-only best D vs K-scale max D if both available

    Inputs:
      SF10_best_K_lambda_summary.csv
      SF2_best_lambda_summary.csv [optional]
    """

    bestk = read_optional("SF10_best_K_lambda_summary.csv")
    bestlam = read_optional("SF2_best_lambda_summary.csv")

    if bestk is None or bestk.empty:
        log("SF10 skipped: best K/lambda summary unavailable.")
        return

    required = ["Cancer", "K", "lambda", "D_clinical"]
    missing = [c for c in required if c not in bestk.columns]
    if missing:
        raise ValueError(f"SF10 missing columns: {missing}")

    bestk = clean_numeric(bestk, ["K", "lambda", "D_clinical"])
    cancers = order_cancers(bestk["Cancer"])
    bestk = bestk.set_index("Cancer").reindex(cancers).reset_index()

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2)
    x = np.arange(len(bestk))

    # A. K*
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x, bestk["K"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bestk["Cancer"], rotation=60, fontsize=8)
    ax.set_ylabel("Optimal K*")
    ax.set_title("Optimal observation scale across cancers")
    panel(ax, "A")

    # B. lambda*
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, bestk["lambda"])
    ax.set_xticks(x)
    ax.set_xticklabels(bestk["Cancer"], rotation=60, fontsize=8)
    ax.set_ylabel("Optimal λ*")
    ax.set_title("Optimal uncertainty penalty across cancers")
    panel(ax, "B")

    # C. max D
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(x, bestk["D_clinical"])
    ax.set_xticks(x)
    ax.set_xticklabels(bestk["Cancer"], rotation=60, fontsize=8)
    ax.set_ylabel("Max D")
    ax.set_title("Maximum K-scale discriminability")
    panel(ax, "C")

    # D. comparison lambda-only vs K-scale
    ax = fig.add_subplot(gs[1, 1])
    if bestlam is not None and not bestlam.empty and {"Cancer", "D_clinical"}.issubset(bestlam.columns):
        bestlam = clean_numeric(bestlam, ["D_clinical", "lambda"])
        merged = bestk[["Cancer", "D_clinical"]].rename(columns={"D_clinical": "K_scale_max_D"}).merge(
            bestlam[["Cancer", "D_clinical"]].rename(columns={"D_clinical": "lambda_only_max_D"}),
            on="Cancer",
            how="inner"
        )
        merged = merged.set_index("Cancer").reindex([c for c in cancers if c in set(merged["Cancer"])])
        x2 = np.arange(len(merged))
        width = 0.38
        ax.bar(x2 - width/2, merged["lambda_only_max_D"], width, label="λ scan")
        ax.bar(x2 + width/2, merged["K_scale_max_D"], width, label="K-scale")
        ax.set_xticks(x2)
        ax.set_xticklabels(merged.index, rotation=60, fontsize=8)
        ax.set_ylabel("Max D")
        ax.set_title("λ-only versus K-scale discriminability")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(bestk["K"], bestk["lambda"], s=90)
        ax.set_xscale("log")
        ax.set_xlabel("Optimal K*")
        ax.set_ylabel("Optimal λ*")
        ax.set_title("Relationship between K* and λ*")
    panel(ax, "D")

    fig.suptitle(
        "Supplementary Figure 10. Integrated optimal scale, uncertainty penalty, and discriminability summary",
        fontsize=16,
        fontweight="bold"
    )

    savefig("SF10_integrated_optimal_K_lambda_D_summary")


# ============================================================
# 6. Main
# ============================================================

def main():
    log("============================================================")
    log("AIDO-D-PHY Supplementary Figures 7–10 Builder START")
    log(f"INPUT_DIR = {INPUT_DIR}")
    log(f"OUT_DIR   = {OUT_DIR}")
    log("============================================================")

    make_sf7()
    make_sf8()
    make_sf9()
    make_sf10()

    log("============================================================")
    log("DONE.")
    log(f"Check output folder: {OUT_DIR}")
    log("============================================================")


if __name__ == "__main__":
    main()
