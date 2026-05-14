# ============================================================
# AIDO-D RANDOM MASTER PIPELINE
# Version: 1.0
#
# Use case:
#   BRCA now, other cancers later.
#
# Required:
#   Random Test output folder
#   GSEA GMT files
#   STRING PPI files
#   UCSC XENA omics folder
#
# Main analyses:
#   A. High-D random enrichment
#   B. Frequency-filtered enrichment
#   C. STRING network clustering
#   D. Largest component enrichment
#   E. Centrality gene enrichment
#   F. Random-highD vs structured-highD comparison
#   G. Multi-omics overlap: GE / CN / MU / RPPA
# ============================================================

import os
import re
import gzip
import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import hypergeom

try:
    import networkx as nx
except ImportError:
    nx = None

warnings.filterwarnings("ignore")


# ============================================================
# 0. USER SETTINGS
# ============================================================

CANCER = "BRCA"
CANCER_FOLDER = "Breast Cancer (BRCA)"

DATABASE_FOR_RANDOM_TEST = "Hallmark"

BASE_RANDOM_ROOT = r"D:\AIDO-Temp\AIDO-D-RANDOM-TEST-FORMAL"
GSEA_DIR = r"D:\AIDO-Data\GSEA"
PPI_DIR = r"D:\AIDO-Data\STRING-PPI"
XENA_ROOT = r"D:\AIDO-Data\UCSC_XENA"

BASE_RANDOM = os.path.join(
    BASE_RANDOM_ROOT,
    f"AIDO-D-RandomTest-{CANCER}"
)

RANDOM_DB_DIR = os.path.join(BASE_RANDOM, DATABASE_FOR_RANDOM_TEST)

BRCA_DIR = os.path.join(XENA_ROOT, CANCER_FOLDER)

OUT_DIR = os.path.join(
    BASE_RANDOM_ROOT,
    f"AIDO-D-RandomTest-{CANCER}-MASTER-INTEGRATED"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. INPUT FILES
# ============================================================

TOP_RANDOM_FILE = os.path.join(
    RANDOM_DB_DIR,
    f"{CANCER}_{DATABASE_FOR_RANDOM_TEST}_top_highD_random_sets.csv"
)

STRUCTURED_D_FILE = os.path.join(
    RANDOM_DB_DIR,
    f"{CANCER}_{DATABASE_FOR_RANDOM_TEST}_structured_D.csv"
)

RANDOM_D_FILE = os.path.join(
    RANDOM_DB_DIR,
    f"{CANCER}_{DATABASE_FOR_RANDOM_TEST}_random_D.csv"
)

GMT_FILES = {
    "Hallmark": os.path.join(GSEA_DIR, "h.all.v2026.1.Hs.symbols.gmt"),
    "GO_BP": os.path.join(GSEA_DIR, "c5.go.bp.v2026.1.Hs.symbols.gmt"),
    "Reactome": os.path.join(GSEA_DIR, "c2.cp.reactome.v2026.1.Hs.symbols.gmt"),
    "KEGG_Medicus": os.path.join(GSEA_DIR, "c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt"),
    "WikiPathways": os.path.join(GSEA_DIR, "c2.cp.wikipathways.v2026.1.Hs.symbols.gmt"),
}

STRING_LINKS = os.path.join(PPI_DIR, "9606.protein.links.v12.0.txt.gz")
STRING_INFO = os.path.join(PPI_DIR, "9606.protein.info.v12.0.txt.gz")

OMICS_FILES = {
    "GE": os.path.join(BRCA_DIR, "GE.tsv"),
    "CN": os.path.join(BRCA_DIR, "CN.tsv"),
    "MU": os.path.join(BRCA_DIR, "MU.tsv"),
    "RPPA": os.path.join(BRCA_DIR, "RPPA.tsv"),
}


# ============================================================
# 2. BASIC FUNCTIONS
# ============================================================

def log(msg):
    print(f"[AIDO-D-RANDOM-{CANCER}] {msg}", flush=True)


def read_csv(path):
    if not os.path.exists(path):
        log(f"WARNING: missing file: {path}")
        return None
    return pd.read_csv(path)


def parse_gene_list(x):
    if pd.isna(x):
        return []

    x = str(x).strip()

    for sep in [";", ",", "|"]:
        if sep in x:
            return sorted(set(g.strip() for g in x.split(sep) if g.strip()))

    return [x] if x else []


def load_gmt(path):
    gene_sets = {}

    if not os.path.exists(path):
        log(f"WARNING: GMT not found: {path}")
        return gene_sets

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                name = parts[0]
                genes = sorted(set(parts[2:]))
                gene_sets[name] = genes

    return gene_sets


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    if n == 0:
        return np.array([])

    order = np.argsort(pvals)
    ranked = pvals[order]

    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)

    out = np.empty(n)
    out[order] = q

    return out


def enrichment_test(query_genes, gene_sets, background_genes=None, min_overlap=2):
    query_genes = set(query_genes)

    if background_genes is None:
        background_genes = set()
        for genes in gene_sets.values():
            background_genes.update(genes)

    background_genes = set(background_genes)
    query_genes = query_genes & background_genes

    M = len(background_genes)
    n = len(query_genes)

    if M == 0 or n == 0:
        return pd.DataFrame()

    records = []

    for term, genes in gene_sets.items():
        genes = set(genes) & background_genes

        K = len(genes)
        k = len(query_genes & genes)

        if k < min_overlap:
            continue

        p = hypergeom.sf(k - 1, M, K, n)

        records.append({
            "Term": term,
            "SetSize": K,
            "QuerySize": n,
            "Overlap": k,
            "OverlapGenes": ";".join(sorted(query_genes & genes)),
            "p_value": p,
            "minus_log10_p": -math.log10(p) if p > 0 else 300
        })

    df = pd.DataFrame(records)

    if len(df) > 0:
        df["FDR"] = bh_fdr(df["p_value"].values)
        df = df.sort_values(
            ["FDR", "p_value", "Overlap"],
            ascending=[True, True, False]
        )

    return df


# ============================================================
# 3. OMICS PARSING FUNCTIONS
# ============================================================

def clean_gene_symbol(x):
    x = str(x).strip()

    if "|" in x:
        x = x.split("|")[0]

    x = re.sub(r"\s+", "", x)

    return x


def read_GE_or_CN_genes(path):
    if not os.path.exists(path):
        return set()

    try:
        df = pd.read_csv(path, sep="\t", usecols=[0])
        genes = set(df.iloc[:, 0].dropna().astype(str).map(clean_gene_symbol))
        genes = set(g for g in genes if g and g.lower() != "nan")
        return genes
    except Exception as e:
        log(f"GE/CN parsing failed for {path}: {e}")
        return set()


def read_MU_genes(path):
    if not os.path.exists(path):
        return set()

    try:
        df = pd.read_csv(path, sep="\t", nrows=5)
    except Exception:
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception as e:
            log(f"MU read failed: {e}")
            return set()

    possible_cols = [
        "Hugo_Symbol",
        "HugoSymbol",
        "Gene",
        "gene",
        "GeneSymbol",
        "symbol",
        "Symbol"
    ]

    for col in possible_cols:
        if col in df.columns:
            full = pd.read_csv(path, sep="\t", usecols=[col])
            genes = set(full[col].dropna().astype(str).map(clean_gene_symbol))
            return set(g for g in genes if g and g.lower() != "nan")

    # fallback: first column
    try:
        full = pd.read_csv(path, sep="\t", usecols=[df.columns[0]])
        genes = set(full.iloc[:, 0].dropna().astype(str).map(clean_gene_symbol))
        return set(g for g in genes if g and g.lower() != "nan")
    except Exception:
        return set()


def map_rppa_feature_to_gene(feature):
    f = str(feature).strip()

    manual = {
        "ER-alpha": "ESR1",
        "ER_alpha": "ESR1",
        "ER": "ESR1",
        "PR": "PGR",
        "HER2": "ERBB2",
        "HER2_pY1248": "ERBB2",
        "EGFR": "EGFR",
        "AKT": "AKT1",
        "Akt": "AKT1",
        "Akt_pS473": "AKT1",
        "Akt_pT308": "AKT1",
        "mTOR": "MTOR",
        "p53": "TP53",
        "BRAF": "BRAF",
        "MEK1": "MAP2K1",
        "MEK": "MAP2K1",
        "MAPK": "MAPK1",
        "ERK2": "MAPK1",
        "GSK3": "GSK3B",
        "PTEN": "PTEN",
        "PI3K": "PIK3CA",
        "c-Kit": "KIT",
        "KIT": "KIT",
        "c-Met": "MET",
        "MET": "MET",
        "VEGFR2": "KDR",
        "E-Cadherin": "CDH1",
        "N-Cadherin": "CDH2",
        "Beta-Catenin": "CTNNB1",
        "Cyclin_B1": "CCNB1",
        "Cyclin_D1": "CCND1",
        "Cyclin_E1": "CCNE1",
        "Bcl-2": "BCL2",
        "Bcl2": "BCL2",
        "BAX": "BAX",
        "CASPASE-3": "CASP3",
        "Caspase-3": "CASP3",
    }

    if f in manual:
        return manual[f]

    # remove phosphorylation / antibody suffixes
    g = f
    g = re.sub(r"_p[STY][0-9]+.*", "", g)
    g = re.sub(r"-p[STY][0-9]+.*", "", g)
    g = re.sub(r"\(.+\)", "", g)
    g = g.replace("-", "_")
    g = g.split("_")[0]
    g = g.strip()

    if not g:
        return None

    return g.upper()


def read_RPPA_genes(path):
    if not os.path.exists(path):
        return set()

    try:
        df = pd.read_csv(path, sep="\t", usecols=[0])
    except Exception as e:
        log(f"RPPA read failed: {e}")
        return set()

    features = df.iloc[:, 0].dropna().astype(str).tolist()
    genes = set()

    for f in features:
        mapped = map_rppa_feature_to_gene(f)
        if mapped:
            genes.add(mapped)

    return genes


def read_omics_genes(omics, path):
    if omics in ["GE", "CN"]:
        return read_GE_or_CN_genes(path)

    if omics == "MU":
        return read_MU_genes(path)

    if omics == "RPPA":
        return read_RPPA_genes(path)

    return set()


# ============================================================
# 4. LOAD RANDOM SETS
# ============================================================

log("Loading high-D random sets...")

top_random = read_csv(TOP_RANDOM_FILE)

if top_random is None:
    raise FileNotFoundError(TOP_RANDOM_FILE)

if "Genes" not in top_random.columns:
    raise ValueError("Column 'Genes' not found.")

top_random["GeneList"] = top_random["Genes"].apply(parse_gene_list)
top_random["GeneCount"] = top_random["GeneList"].apply(len)

all_genes = []
for genes in top_random["GeneList"]:
    all_genes.extend(genes)

gene_counter = Counter(all_genes)

gene_frequency = pd.DataFrame([
    {"Gene": g, "Frequency": c}
    for g, c in gene_counter.items()
]).sort_values(["Frequency", "Gene"], ascending=[False, True])

gene_frequency.to_csv(
    os.path.join(OUT_DIR, f"{CANCER}_highD_random_gene_frequency.csv"),
    index=False
)

highD_all = set(gene_frequency["Gene"])
highD_freq2 = set(gene_frequency.loc[gene_frequency["Frequency"] >= 2, "Gene"])
highD_freq3 = set(gene_frequency.loc[gene_frequency["Frequency"] >= 3, "Gene"])

log(f"High-D random sets: {len(top_random)}")
log(f"Unique high-D genes: {len(highD_all)}")
log(f"Frequency >=2 genes: {len(highD_freq2)}")
log(f"Frequency >=3 genes: {len(highD_freq3)}")


# ============================================================
# 5. LOAD GMT DATABASES
# ============================================================

log("Loading GMT databases...")

gmt = {}
background = set()

for db, path in GMT_FILES.items():
    log(f"Loading {db}")
    gmt[db] = load_gmt(path)

    for genes in gmt[db].values():
        background.update(genes)

log(f"Background genes from GMT databases: {len(background)}")


# ============================================================
# 6. PART A/B ¡ª ENRICHMENT
# ============================================================

query_sets = {
    "all_highD_random_genes": highD_all,
    "freq_ge_2_highD_random_genes": highD_freq2,
    "freq_ge_3_highD_random_genes": highD_freq3,
}

for query_name, query_genes in query_sets.items():
    log(f"Running enrichment for {query_name}: {len(query_genes)} genes")

    for db, gene_sets in gmt.items():
        enr = enrichment_test(
            query_genes=query_genes,
            gene_sets=gene_sets,
            background_genes=background,
            min_overlap=3
        )

        enr.to_csv(
            os.path.join(
                OUT_DIR,
                f"{CANCER}_{query_name}_enrichment_{db}.csv"
            ),
            index=False
        )


# per-random-set enrichment
log("Running per-random-set enrichment...")

per_records = []

for idx, row in top_random.iterrows():
    rid = row.get("GeneSet", f"Random_{idx}")
    D = row.get("D", np.nan)
    genes = row["GeneList"]

    for db, gene_sets in gmt.items():
        enr = enrichment_test(
            query_genes=genes,
            gene_sets=gene_sets,
            background_genes=background,
            min_overlap=2
        )

        if len(enr) > 0:
            best = enr.iloc[0]
            per_records.append({
                "RandomSet": rid,
                "D": D,
                "Database": db,
                "TopTerm": best["Term"],
                "Overlap": best["Overlap"],
                "p_value": best["p_value"],
                "FDR": best["FDR"],
                "OverlapGenes": best["OverlapGenes"]
            })

pd.DataFrame(per_records).to_csv(
    os.path.join(OUT_DIR, f"{CANCER}_per_random_set_top_enrichment.csv"),
    index=False
)


# ============================================================
# 7. PART C ¡ª STRING NETWORK
# ============================================================

centrality_df = pd.DataFrame()
component_df = pd.DataFrame()
largest_component_genes = set()
centrality_top_genes = set()

if nx is None:
    log("networkx not installed. Skipping STRING network.")
else:
    if not os.path.exists(STRING_INFO) or not os.path.exists(STRING_LINKS):
        log("STRING files missing. Skipping STRING network.")
    else:
        log("Loading STRING info...")

        with gzip.open(STRING_INFO, "rt", encoding="utf-8", errors="ignore") as f:
            info = pd.read_csv(f, sep="\t")

        protein_col = info.columns[0]
        name_col = "preferred_name" if "preferred_name" in info.columns else info.columns[1]

        protein_to_gene = dict(zip(info[protein_col].astype(str), info[name_col].astype(str)))

        gene_to_protein = {}
        for protein, gene in protein_to_gene.items():
            gene_to_protein.setdefault(gene, []).append(protein)

        query_proteins = set()
        protein_to_query_gene = {}

        for gene in highD_all:
            for protein in gene_to_protein.get(gene, []):
                query_proteins.add(protein)
                protein_to_query_gene[protein] = gene

        log(f"STRING-mapped genes/proteins: {len(query_proteins)}")

        edges = []

        log("Scanning STRING links...")

        with gzip.open(STRING_LINKS, "rt", encoding="utf-8", errors="ignore") as f:
            _ = f.readline()

            for line in f:
                parts = line.strip().split()

                if len(parts) < 3:
                    continue

                p1, p2, score = parts[0], parts[1], int(parts[2])

                if p1 in query_proteins and p2 in query_proteins and score >= 700:
                    g1 = protein_to_query_gene.get(p1)
                    g2 = protein_to_query_gene.get(p2)

                    if g1 and g2 and g1 != g2:
                        edges.append({
                            "GeneA": g1,
                            "GeneB": g2,
                            "STRING_score": score
                        })

        edge_df = pd.DataFrame(edges).drop_duplicates()

        edge_df.to_csv(
            os.path.join(OUT_DIR, f"{CANCER}_highD_random_STRING_edges_score700.csv"),
            index=False
        )

        G = nx.Graph()

        for gene in highD_all:
            G.add_node(gene, frequency=gene_counter.get(gene, 0))

        for _, r in edge_df.iterrows():
            G.add_edge(r["GeneA"], r["GeneB"], weight=r["STRING_score"])

        components = list(nx.connected_components(G))

        component_records = []

        for i, comp in enumerate(sorted(components, key=len, reverse=True), 1):
            sub = G.subgraph(comp)
            component_records.append({
                "ComponentID": i,
                "N_nodes": sub.number_of_nodes(),
                "N_edges": sub.number_of_edges(),
                "Genes": ";".join(sorted(comp))
            })

        component_df = pd.DataFrame(component_records)

        component_df.to_csv(
            os.path.join(OUT_DIR, f"{CANCER}_highD_random_STRING_components.csv"),
            index=False
        )

        if len(components) > 0:
            largest_component_genes = set(max(components, key=len))

        degree = dict(G.degree())
        betweenness = nx.betweenness_centrality(G) if G.number_of_edges() > 0 else {}

        centrality_df = pd.DataFrame([
            {
                "Gene": g,
                "Frequency": gene_counter.get(g, 0),
                "Degree": degree.get(g, 0),
                "Betweenness": betweenness.get(g, 0)
            }
            for g in G.nodes()
        ]).sort_values(["Degree", "Frequency"], ascending=False)

        centrality_df.to_csv(
            os.path.join(OUT_DIR, f"{CANCER}_highD_random_STRING_centrality.csv"),
            index=False
        )

        centrality_top_genes = set(centrality_df.head(100)["Gene"])

        # largest component plot
        if len(largest_component_genes) >= 2:
            subG = G.subgraph(largest_component_genes).copy()

            if subG.number_of_nodes() > 150:
                top_nodes = centrality_df.head(150)["Gene"].tolist()
                subG = subG.subgraph(top_nodes).copy()

            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(subG, seed=42)

            sizes = [
                50 + 20 * gene_counter.get(n, 1)
                for n in subG.nodes()
            ]

            nx.draw_networkx_nodes(subG, pos, node_size=sizes, alpha=0.8)
            nx.draw_networkx_edges(subG, pos, alpha=0.25)
            nx.draw_networkx_labels(subG, pos, font_size=6)

            plt.title(f"{CANCER} High-D Random STRING Network")
            plt.axis("off")
            plt.tight_layout()

            plt.savefig(
                os.path.join(OUT_DIR, f"{CANCER}_highD_random_STRING_network.png"),
                dpi=300
            )
            plt.close()

        log("STRING network completed.")


# ============================================================
# 8. ENRICH NETWORK-DERIVED GENE SETS
# ============================================================

network_query_sets = {
    "largest_STRING_component": largest_component_genes,
    "top100_STRING_centrality": centrality_top_genes,
}

for query_name, query_genes in network_query_sets.items():
    if len(query_genes) == 0:
        continue

    log(f"Running enrichment for {query_name}: {len(query_genes)} genes")

    for db, gene_sets in gmt.items():
        enr = enrichment_test(
            query_genes=query_genes,
            gene_sets=gene_sets,
            background_genes=background,
            min_overlap=3
        )

        enr.to_csv(
            os.path.join(
                OUT_DIR,
                f"{CANCER}_{query_name}_enrichment_{db}.csv"
            ),
            index=False
        )


# ============================================================
# 9. RANDOM-HIGHD VS STRUCTURED-HIGHD
# ============================================================

log("Comparing random-highD with structured-highD...")

structured = read_csv(STRUCTURED_D_FILE)

if structured is not None:
    hallmark_sets = gmt.get("Hallmark", {})

    if "D" in structured.columns:
        structured_top = structured.sort_values("D", ascending=False).head(20)
    else:
        structured_top = structured.head(20)

    possible_name_cols = ["GeneSet", "Hallmark", "Pathway", "Term", "Name"]

    records = []

    for _, row in structured_top.iterrows():
        name = None

        for col in possible_name_cols:
            if col in structured_top.columns:
                name = row[col]
                break

        if name is None:
            continue

        hgenes = set(hallmark_sets.get(name, []))

        if len(hgenes) == 0:
            continue

        overlap = highD_all & hgenes
        union = highD_all | hgenes

        records.append({
            "StructuredGeneSet": name,
            "Structured_D": row.get("D", np.nan),
            "Structured_p": row.get("p_logrank", row.get("p_value", np.nan)),
            "StructuredSize": len(hgenes),
            "RandomHighDGenePoolSize": len(highD_all),
            "Overlap": len(overlap),
            "Jaccard": len(overlap) / len(union) if len(union) else 0,
            "OverlapGenes": ";".join(sorted(overlap))
        })

    pd.DataFrame(records).to_csv(
        os.path.join(OUT_DIR, f"{CANCER}_random_highD_vs_structured_highD_overlap.csv"),
        index=False
    )

    # D distribution plot
    random_D = read_csv(RANDOM_D_FILE)

    if random_D is not None and "D" in random_D.columns and "D" in structured.columns:
        plt.figure(figsize=(8, 6))
        plt.boxplot(
            [structured["D"].dropna(), random_D["D"].dropna()],
            labels=["Structured", "Random"]
        )
        plt.ylabel("D = -log10(p)")
        plt.title(f"{CANCER}: Structured vs Random D")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"{CANCER}_structured_vs_random_D_boxplot.png"),
            dpi=300
        )
        plt.close()


# ============================================================
# 10. MULTI-OMICS OVERLAP
# ============================================================

log("Running multi-omics overlap analysis...")

omics_gene_sets = {}

for omics, path in OMICS_FILES.items():
    genes = read_omics_genes(omics, path)
    omics_gene_sets[omics] = genes
    log(f"{omics}: {len(genes)} parsed features/genes")

omics_records = []

for omics, genes in omics_gene_sets.items():
    overlap = highD_all & genes

    omics_records.append({
        "Omics": omics,
        "TotalFeaturesOrGenes": len(genes),
        "HighDRandomGenes": len(highD_all),
        "Overlap": len(overlap),
        "Coverage": len(overlap) / len(highD_all) if len(highD_all) else 0,
        "OverlapGenes": ";".join(sorted(overlap))
    })

omics_summary = pd.DataFrame(omics_records)

omics_summary.to_csv(
    os.path.join(OUT_DIR, f"{CANCER}_highD_random_multiomics_overlap_summary.csv"),
    index=False
)

presence_records = []

for gene in sorted(highD_all):
    presence_records.append({
        "Gene": gene,
        "RandomHighD_Frequency": gene_counter.get(gene, 0),
        "In_GE": gene in omics_gene_sets.get("GE", set()),
        "In_CN": gene in omics_gene_sets.get("CN", set()),
        "In_MU": gene in omics_gene_sets.get("MU", set()),
        "In_RPPA": gene in omics_gene_sets.get("RPPA", set()),
        "OmicsCount": sum([
            gene in omics_gene_sets.get("GE", set()),
            gene in omics_gene_sets.get("CN", set()),
            gene in omics_gene_sets.get("MU", set()),
            gene in omics_gene_sets.get("RPPA", set()),
        ])
    })

presence_df = pd.DataFrame(presence_records).sort_values(
    ["OmicsCount", "RandomHighD_Frequency"],
    ascending=False
)

presence_df.to_csv(
    os.path.join(OUT_DIR, f"{CANCER}_highD_random_gene_multiomics_presence.csv"),
    index=False
)

plt.figure(figsize=(7, 5))
plt.bar(omics_summary["Omics"], omics_summary["Coverage"])
plt.ylabel("Coverage of high-D random genes")
plt.title(f"{CANCER}: High-D Random Gene Multi-omics Coverage")
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, f"{CANCER}_highD_random_multiomics_coverage.png"),
    dpi=300
)
plt.close()


# ============================================================
# 11. WRITE SUMMARY
# ============================================================

summary = []

summary.append(f"# {CANCER} AIDO-D Random Master Integrated Summary\n")

summary.append("## Input summary\n")
summary.append(f"- Cancer: {CANCER}")
summary.append(f"- Random database: {DATABASE_FOR_RANDOM_TEST}")
summary.append(f"- High-D random sets: {len(top_random)}")
summary.append(f"- Unique high-D random genes: {len(highD_all)}")
summary.append(f"- Frequency >= 2 genes: {len(highD_freq2)}")
summary.append(f"- Frequency >= 3 genes: {len(highD_freq3)}\n")

summary.append("## Top recurrent high-D random genes\n")
for _, row in gene_frequency.head(20).iterrows():
    summary.append(f"- {row['Gene']}: {row['Frequency']}")

summary.append("\n## Multi-omics coverage\n")
for _, row in omics_summary.iterrows():
    summary.append(
        f"- {row['Omics']}: {row['Overlap']} / {row['HighDRandomGenes']} "
        f"({row['Coverage']:.3f})"
    )

summary.append("\n## STRING network summary\n")
if len(component_df) > 0:
    summary.append(f"- STRING components: {len(component_df)}")
    summary.append(f"- Largest component nodes: {component_df.iloc[0]['N_nodes']}")
    summary.append(f"- Largest component edges: {component_df.iloc[0]['N_edges']}")
else:
    summary.append("- STRING network not available or no components detected.")

summary.append("\n## Interpretation\n")
summary.append(
    "High-D random observables should not be treated only as statistical noise. "
    "They may reveal hidden or distributed observable coordinates that are not "
    "fully captured by curated biological annotation systems."
)

summary.append("\n## AIDO-D implication\n")
summary.append(
    "This analysis supports the distinction between biological annotation space "
    "and discriminability space. Random high-D gene sets can act as latent-space "
    "probes under partial observability."
)

with open(
    os.path.join(OUT_DIR, f"{CANCER}_AIDO_D_RANDOM_MASTER_SUMMARY.md"),
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(summary))


# ============================================================
# 12. DONE
# ============================================================

log("MASTER PIPELINE COMPLETED.")
log(f"Output directory: {OUT_DIR}")