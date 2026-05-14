# AIDO-D-PHY I Code Package v1.0

This package contains the frozen Python scripts used for the manuscript:

**Physics-guided discriminability of process-level observables in partially observable cancer systems**

The scripts support the analysis of gene-expression-derived biological-process observables using mean activity (mu), FWHM-based cloud thickness (W), uncertainty-aware discriminability, structured-versus-random validation, and supplementary figure generation.

## Package status

This is a **Code Freeze v1.0** package prepared for manuscript submission / reproducibility review. The analysis logic should not be modified unless the manuscript figures, Supplementary Information, and Supplementary Methods are updated accordingly.

## Folder structure

```text
AIDO-D-PHY-I_code_package_v1_0/
├── scripts/
│   ├── core/
│   │   └── 01_planB_muW_DWPHY_pipeline.py
│   ├── random_tests/
│   │   ├── 02_random_test_master_v1_3.py
│   │   └── 03_highD_random_dissection_v1_1.py
│   └── supplementary_figures/
│       ├── 04_build_supplementary_figures_1_3.py
│       ├── 05_build_supplementary_figures_4_6.py
│       └── 06_build_supplementary_figures_7_10.py
├── original_uploaded_scripts/
├── manuscript_context/
├── requirements.txt
├── FIGURE_CODE_MAP.md
├── SUPPLEMENTARY_METHODS_CODE_MAP.md
└── README.md
```

## Required Python packages

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Main analysis modules

### 1. Core Plan-B / D-PHY analysis

```bash
python scripts/core/01_planB_muW_DWPHY_pipeline.py
```

This script performs cohort-level gene-expression preprocessing, biological-process scoring, mu/W construction, FWHM-based uncertainty-width estimation, coherence calculation, uncertainty-aware discriminability, lambda scans, K-scale analysis, and core figure-related outputs.

### 2. Structured-versus-random validation

```bash
python scripts/random_tests/02_random_test_master_v1_3.py
```

This script implements the formal three-layer random validation framework:

1. structured-versus-random comparison;
2. high-D random dissection;
3. cross-cancer recurrence.

### 3. High-D random dissection

```bash
python scripts/random_tests/03_highD_random_dissection_v1_1.py
```

This script supports downstream analysis of high-discriminability random gene sets, including enrichment-style interpretation and recurrence analysis.

### 4. Supplementary figure builders

```bash
python scripts/supplementary_figures/04_build_supplementary_figures_1_3.py
python scripts/supplementary_figures/05_build_supplementary_figures_4_6.py
python scripts/supplementary_figures/06_build_supplementary_figures_7_10.py
```

These scripts build Supplementary Figures from extracted CSV summary files.

## Important path note

The original scripts were developed on a Windows local environment using paths such as:

```text
D:/AIDO-Data/
D:/AIDO-Temp/
```

Before running on another machine, update the input and output paths at the top of each script. The analysis logic is preserved; only local path settings should be changed.

## Data inputs

The pipeline expects TCGA gene-expression and survival data downloaded from UCSC Xena, plus GMT gene-set files, including Hallmark, GO Biological Process, and Reactome collections.

Typical input resources:

- TCGA gene-expression matrices from UCSC Xena
- matched survival or phenotype tables
- `h.all.v2026.1.Hs.symbols.gmt`
- `c5.go.bp.v2026.1.Hs.symbols.gmt`
- `c2.cp.reactome.v2026.1.Hs.symbols.gmt`

## Main outputs

The scripts generate:

- pathway-level mu/W summary tables;
- FWHM-based uncertainty-width matrices;
- coherence summaries;
- lambda-scan summaries;
- K-scale summaries;
- structured-versus-random validation tables;
- high-D random gene-set summaries;
- manuscript and supplementary figures.

## Reproducibility caution

Random gene-set analysis should use fixed seeds when reproducing exact random-baseline results. If rerun with different random seeds, summary trends should remain comparable, but individual random gene sets may differ.

## Citation / manuscript use

This package supports the reproducibility of the AIDO-D-PHY I manuscript and should be cited or linked in the Code Availability section if uploaded to a public repository.
