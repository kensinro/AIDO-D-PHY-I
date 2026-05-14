# Supplementary Methods-to-code map

| Supplementary Methods section | Related script(s) | Purpose |
|---|---|---|
| SM1 Overall analytical framework | All scripts | Defines the full D-PHY workflow. |
| SM2 TCGA transcriptomic cohort preparation and preprocessing | `01_planB_muW_DWPHY_pipeline.py` | Gene-expression loading, patient matching, survival parsing, z-score normalization. |
| SM3 Pathway database construction and observable generation | `01_planB_muW_DWPHY_pipeline.py` | GMT parsing, Hallmark/GO:BP/Reactome mapping, pathway-level activity and W construction. |
| SM4 Observable coherence, D_W-PHY, and lambda scan | `01_planB_muW_DWPHY_pipeline.py` | Coherence, patient-level score, lambda scan, survival discriminability. |
| SM5 K-scale observation analysis | `01_planB_muW_DWPHY_pipeline.py` | K-scale selection and scale-dependent discriminability. |
| SM6 Structured-versus-random validation | `02_random_test_master_v1_3.py`, `03_highD_random_dissection_v1_1.py` | Size-matched random sets, high-D random analysis, cross-cancer recurrence. |
| SM7 Statistical analysis and visualization | All scripts | Numerical safeguards, survival analysis, figure generation. |
| SM8 Figure generation | Supplementary figure builders | Rendering manuscript-specific and supplementary figures. |
| SM9 Biological interpretation framework | Main manuscript + `01_planB_muW_DWPHY_pipeline.py` | Defines partial observability, mu-W state cloud, and D-PHY interpretation. |
| SM10 Result interpretation and limitations | All analysis outputs | Evidence hierarchy and cautious interpretation. |
| SM11 Computational reproducibility | This code package | Execution protocol, code organization, and output structure. |
