# Suggested run order

1. Prepare TCGA expression and survival tables and GMT gene-set files.
2. Run the core D-PHY pipeline:

```bash
python scripts/core/01_planB_muW_DWPHY_pipeline.py
```

3. Run structured-versus-random validation:

```bash
python scripts/random_tests/02_random_test_master_v1_3.py
```

4. Run high-D random dissection, if random outputs are available:

```bash
python scripts/random_tests/03_highD_random_dissection_v1_1.py
```

5. Generate supplementary figures from extracted summary CSV files:

```bash
python scripts/supplementary_figures/04_build_supplementary_figures_1_3.py
python scripts/supplementary_figures/05_build_supplementary_figures_4_6.py
python scripts/supplementary_figures/06_build_supplementary_figures_7_10.py
```

Note: update local path variables inside each script before execution.
