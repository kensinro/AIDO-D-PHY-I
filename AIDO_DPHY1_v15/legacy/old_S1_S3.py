# ============================================================
# AIDO-D-PHY Supplementary Figures 1–3 Builder
# Version: 2026-05-13
#
# Purpose:
#   Generate REAL-DATA supplementary figures from extracted CSV files.
#
# Input folder:
#   D:/AIDO-Temp/AIDO-D-PHY-SF-EXTRACTED
#
# Required files:
#   SF1_all_gene_set_mu_W_summary.csv
#   SF2_lambda_scan_all_cancers.csv
#   SF3_DK_curve_best_lambda_each_K.csv
#
# Output folder:
#   D:/AIDO-Temp/AIDO-D-PHY-SF-FIGURES
#
# Required packages:
#   pip install pandas numpy matplotlib
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def log(msg):
    print(f"[AIDO-D-PHY SF FIG1-3] {msg}", flush=True)


def read_required(filename):
    path = INPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    log(f"Loaded {filename}: {df.shape}")
    return df


def order_cancers(values):
    values = set(str(v) for v in values)
    return [c for c in CANCER_ORDER if c in values] + sorted([c for c in values if c not in CANCER_ORDER])


def order_databases(values):
    values = set(str(v) for v in values)
    return [d for d in DATABASE_ORDER if d in values] + sorted([d for d in values if d not in DATABASE_ORDER])


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
    ax.text(-0.10, 1.08, label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")


def clean_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def make_sf1():
    df = read_required("SF1_all_gene_set_mu_W_summary.csv")

    required = ["Cancer", "database", "mu_mean_abs", "W_mean", "coherence_mean", "n_genes_mapped"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SF1 missing columns: {missing}")

    df = clean_numeric(df, ["mu_mean_abs", "W_mean", "W_median", "coherence_mean", "n_genes_mapped"])
    df = df.dropna(subset=["Cancer", "database", "mu_mean_abs", "W_mean"])
    df = df[df["W_mean"] > 0]

    cancers = order_cancers(df["Cancer"])
    dbs = order_databases(df["database"])

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    for db in dbs:
        sub = df[df["database"] == db]
        if not sub.empty:
            ax.scatter(sub["mu_mean_abs"], sub["W_mean"], s=18, alpha=0.45, label=db)
    ax.set_xlabel("Mean |μ|")
    ax.set_ylabel("Mean W (FWHM)")
    ax.set_title("Global μ–W uncertainty landscape")
    ax.legend(frameon=False, fontsize=8)
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    pivot_w = df.pivot_table(index="Cancer", columns="database", values="W_mean", aggfunc="median").reindex(cancers)
    present_dbs = [d for d in dbs if d in pivot_w.columns]
    pivot_w[present_dbs].plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylabel("Median W")
    ax.set_title("Cancer-wise uncertainty width by database")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    pivot_c = df.pivot_table(index="Cancer", columns="database", values="coherence_mean", aggfunc="median").reindex(cancers)
    present_dbs = [d for d in dbs if d in pivot_c.columns]
    pivot_c[present_dbs].plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylabel("Median coherence")
    ax.set_title("Cancer-wise observable coherence by database")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    panel(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    for db in dbs:
        sub = df[df["database"] == db].dropna(subset=["n_genes_mapped", "W_mean"])
        sub = sub[sub["n_genes_mapped"] > 0]
        if not sub.empty:
            ax.scatter(sub["n_genes_mapped"], sub["W_mean"], s=18, alpha=0.45, label=db)
    ax.set_xscale("log")
    ax.set_xlabel("Mapped pathway size")
    ax.set_ylabel("Mean W (FWHM)")
    ax.set_title("Pathway size and uncertainty width")
    ax.legend(frameon=False, fontsize=8)
    panel(ax, "D")

    fig.suptitle("Supplementary Figure 1. Complete real-data μ–W uncertainty structure across cancers",
                 fontsize=16, fontweight="bold")
    savefig("SF1_complete_mu_W_uncertainty_landscape")


def make_sf2():
    df = read_required("SF2_lambda_scan_all_cancers.csv")

    required = ["Cancer", "lambda", "D_clinical"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SF2 missing columns: {missing}")

    df = clean_numeric(df, ["lambda", "D_clinical"])
    df = df.dropna(subset=["Cancer", "lambda", "D_clinical"])

    cancers = order_cancers(df["Cancer"])
    n = len(cancers)
    ncol = 6
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3.0 * nrow), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, cancer in enumerate(cancers):
        ax = axes[i]
        sub = df[df["Cancer"] == cancer].groupby("lambda", as_index=False)["D_clinical"].max().sort_values("lambda")
        ax.plot(sub["lambda"], sub["D_clinical"], marker="o", linewidth=1.8)

        if not sub.empty:
            best = sub.loc[sub["D_clinical"].idxmax()]
            ax.axvline(best["lambda"], linestyle="--", linewidth=1)
            ax.scatter([best["lambda"]], [best["D_clinical"]], s=55, zorder=5)
            ax.text(0.05, 0.88, f"λ*={best['lambda']:.2g}\nD={best['D_clinical']:.2f}",
                    transform=ax.transAxes, fontsize=8, va="top")

        ax.set_title(cancer, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

        if i % ncol == 0:
            ax.set_ylabel("D(λ)")
        if i // ncol == nrow - 1:
            ax.set_xlabel("λ")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Supplementary Figure 2. Full λ-scan discriminability curves across cancers",
                 fontsize=16, fontweight="bold", y=1.02)
    savefig("SF2_full_lambda_scan_curves")


def make_sf3():
    df = read_required("SF3_DK_curve_best_lambda_each_K.csv")

    required = ["Cancer", "K", "D_clinical"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"SF3 missing columns: {missing}")

    df = clean_numeric(df, ["K", "lambda", "D_clinical"])
    df = df.dropna(subset=["Cancer", "K", "D_clinical"])

    cancers = order_cancers(df["Cancer"])
    n = len(cancers)
    ncol = 6
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3.0 * nrow), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, cancer in enumerate(cancers):
        ax = axes[i]
        sub = df[df["Cancer"] == cancer].groupby("K", as_index=False)["D_clinical"].max().sort_values("K")
        ax.plot(sub["K"], sub["D_clinical"], marker="o", linewidth=1.8)
        ax.set_xscale("log")

        if not sub.empty:
            best = sub.loc[sub["D_clinical"].idxmax()]
            ax.axvline(best["K"], linestyle="--", linewidth=1)
            ax.scatter([best["K"]], [best["D_clinical"]], s=55, zorder=5)
            ax.text(0.05, 0.88, f"K*={int(best['K'])}\nD={best['D_clinical']:.2f}",
                    transform=ax.transAxes, fontsize=8, va="top")

        ax.set_title(cancer, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

        if i % ncol == 0:
            ax.set_ylabel("D(K)")
        if i // ncol == nrow - 1:
            ax.set_xlabel("K")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Supplementary Figure 3. Complete K-scale discriminability curves across cancers",
                 fontsize=16, fontweight="bold", y=1.02)
    savefig("SF3_complete_K_scale_curves")


def main():
    log("============================================================")
    log("AIDO-D-PHY Supplementary Figures 1–3 Builder START")
    log(f"INPUT_DIR = {INPUT_DIR}")
    log(f"OUT_DIR   = {OUT_DIR}")
    log("============================================================")

    make_sf1()
    make_sf2()
    make_sf3()

    log("============================================================")
    log("DONE.")
    log(f"Check output folder: {OUT_DIR}")
    log("============================================================")


if __name__ == "__main__":
    main()
