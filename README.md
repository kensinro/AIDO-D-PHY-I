# AIDO-D-PHY-I code package v1.5 short-path edition

This is a short-path GitHub replacement package for the manuscript:
**Physics-guided discriminability of process-level observables in partially observable cancer systems**.

This edition is designed for Windows systems that may trigger `Error 0x80010135: Path too long` when extracting nested folders.

## Core definition

The revised D-PHY-I workflow defines the biological-process observable for process `B` and patient `p` as:

```text
O_B,p = {Z_g,p : g in B_mapped}
mu_B,p = mean_g(Z_g,p)
W_B,p = FWHM(O_B,p)
```

`W_B,p` is the patient-level within-process gene-expression width. It is **not** the FWHM of pathway activity scores across patients.

## Folder structure

```text
code/core/    Main Plan-B / mu-W / D_W-PHY workflow
code/rand/    Random baseline and high-D random dissection scripts
code/si/      Current SI V4 figure scripts S1-S5
legacy/       Older supplementary figure scripts kept only for traceability
ms/           Manuscript context PDFs
GA/           Graphical abstract
out/          Output placeholder
```

## Recommended run order

1. `code/core/01_planB.py`
2. `code/rand/02_random_master.py`
3. `code/rand/03_highD_dissect.py`
4. `code/si/S1_activity_width.py`
5. `code/si/S2_lambda.py`
6. `code/si/S3_K_scale.py`
7. `code/si/S4_random_diag.py`
8. `code/si/S5_top_obs.py`

Each script currently assumes the Windows data root `D:/AIDO-Temp`. Edit the `ROOT` variable near the top of each script if your local data directory is different.

## Notes

- The SI scripts are the current replacement versions used for SI V4.
- S3 uses strict K-scale filtering to avoid the earlier `K* = 0` issue.
- S4 aligns with the random-background diagnostic logic used in the manuscript.
- S5 is the revised top-observables figure script with safer labels.
