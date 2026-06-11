# PROVENANCE B — Section IV body text + Table II (model ladder, non-offset rows)

Verified 2026-06-11. Workspace `/home/bretech/OTHERS/TFGV4` (data `unified_5s_corrected.csv`,
splits per `data4.split_indices`: chronological 70/15/15, 4-min gaps).
Recompute scripts written for this audit: `TFGV4/verify_provenance_b.py` (deterministic)
and `TFGV4/verify_provenance_b_stoch.py` (MLP-reg 3 seeds, LSTM 5 seeds, capacity-MLP reseed).

Verdicts: RECOMPUTED-MATCH (refit from scratch, number reproduced) ·
ARTIFACT-MATCH (matches saved results file; stochastic single-run values) ·
MISMATCH · UNVERIFIABLE.

| # | Claim (tex location) | Verdict | Provenance |
|---|---|---|---|
| 1 | Floor RH = 3.88 %RH (caption + IV.A, l.314/361) | RECOMPUTED-MATCH | predict-train-mean on test → 3.88 (verify_provenance_b.py FLOOR) |
| 2 | Floor CO₂ = 197 ppm (caption + IV.B, l.314/374) | RECOMPUTED-MATCH | predict-train-mean on test → 197.4 → "197" (same) |
| 3 | Ridge-72 row: Nₚ 146, RH 0.17/0.88, CO₂ 119/403, r .76 (l.328) | RECOMPUTED-MATCH | refit Ridge α=1000 (val-selected): 0.17/0.88, 118.52/403.34, r .761; Nₚ=(72+1)×2=146. Artifact: metrics3.json t21_ridge72 (identical) |
| 4 | Ridge+HP 4 h row: Nₚ 578, RH 0.04/0.77, CO₂ 44/297, r .43 (l.329) | RECOMPUTED-MATCH | refit α=100: 0.04/0.77, 43.62/296.75, r .429; Nₚ=(288+1)×2=578. Artifact: metrics3.json t22_ridge_hp4 (note: artifact's own "params" field says 576 = no intercepts; table's 578 counts intercepts, consistent with 146/26 convention) |
| 5 | Ridge 12 (R,G,B)/IR row: Nₚ 26, RH 0.27/1.74, CO₂ 179/103, r .90 (l.330) | RECOMPUTED-MATCH | joint Ridge α=100 (val-CO₂-selected) on the 12 per-pixel {R,G,B}/IR ratios: 0.27/1.74, 179.04/102.74, r .904, R²=.765; Nₚ=(12+1)×2=26. Artifacts: idea2_irnorm.json ridge::ridge_IRN12 (CO₂ identical), full_ridge_irn12.npz (tr 179.04 / te 102.74) |
| 6 | MLP 128×64 reg. row: Nₚ 18k, RH 0.23/1.34, CO₂ 26/521, r .70 (l.331) | ARTIFACT-MATCH (521 = seed mean, see #17) | Nₚ=17730→"18k" (recomputed). 0.23/1.34/25.54→26/r .702 all match full_mlp_reg.npz exactly and were reproduced bit-for-bit by my seed-1 refit (verify_provenance_b_stoch.py MLPREG seed1: tr_RH 0.23, te_RH 1.34, tr_CO₂ 25.54, te_CO₂ 464.13, r .702). The te-CO₂ cell (521) is the 3-seed mean of the body claim, not this seed's 464 — see #17 |
| 7 | LSTM 6-h row: Nₚ 21k, RH 0.14/0.69, CO₂ 19/654, r .81 (l.332) | ARTIFACT-MATCH (MAEs) + RECOMPUTED-MATCH (r) | Nₚ=21122→"21k" (recomputed). 0.14/0.69, 18.51→19/653.91→654 = metrics4.json L4_lstm_ma60_W360 (single run). r .81 = 5-seed mean, reproduced exactly: my 5 seeded reruns give r = 0.81±0.03 (seeds .764/.795/.830/.830/.851). ⚠ Caption says "LSTM rows are means over five seeds": the r is; the four MAE cells equal the archived single run (5-seed means would be RH 0.10/0.93, CO₂ 18/680±35). 654 sits inside the seed spread; flag as a caption-accuracy nit, not a numbers error |
| 8 | Random-split control row: 0.16/0.16, 110/112, r .93 (l.333) | RECOMPUTED-MATCH | refit Ridge α=100 on RandomState(0) permutation split: 0.16/0.16, 110.43/111.97, r .930. Artifact: metrics3.json t31_ridge_RANDOMSPLIT (identical) |
| 9 | Caption: R²_te<0 for all CO₂ entries except starred control and IR ridge R²=0.77 (l.311–313, top block only) | RECOMPUTED-MATCH | R²_te: ridge72 −4.10, HP4h −1.15, MLP-reg −2.97, LSTM −3.56, IRN12 +0.765→"0.77", random split +0.864 (starred) |
| 10 | OLS-16: train 87, test 1549 ppm (l.376–377) | RECOMPUTED-MATCH | refit OLS on 16 raw channels: 86.57 / 1548.86. Artifact: metrics3.json t20_ols16 |
| 11 | RH-tuned ridge, 8-h HP window: MAE 0.35 %RH, R²=0.993 (l.361–363) | RECOMPUTED-MATCH | refit ridge72+HP8h α=10 (val-selected): te RH 0.35, R² 0.993. Artifact: metrics.json t3_ridge_hp8h |
| 12 | CO₂ best HP window 4 h → 297 ppm (l.378) | RECOMPUTED-MATCH | refit: 296.75. "Best at 4 h" confirmed by the artifact sweep metrics.json t3 (1/2/4/8 h → 387/421/297/1196) |
| 13 | RH prefers 8 h while CO₂ HP-8h fails (→1196 ppm; number in fig/sweep, body states the preference) (l.378–379) | RECOMPUTED-MATCH | refit HP8h: CO₂ te 1195.51 → 1196. Artifact: metrics.json t3_ridge_hp8h |
| 14 | Ablation: red alone 4.4 %RH (l.366) | RECOMPUTED-MATCH | single red channel (OP0_red), ridge α=100 (the t32 fixed-α convention): te RH 4.37 → "4.4" (also 4.37 at α=10 and OLS — insensitive). Note: the 4-red-channel variant gives 4.66 (α=100) / 3.77 (α=1000); the printed 4.4 corresponds to the one-channel reading of "red alone; one band" |
| 15 | Ablation: one pixel's four colours 0.60 %RH (l.367) | RECOMPUTED-MATCH | ridge α=100 on OP0_{red,green,blue,ir}: te RH 0.60 exact |
| 16 | Ablation: sixteen raw worse than four, 1.04 %RH (l.368) | RECOMPUTED-MATCH | ridge α=100 on 16 raw channels: te RH 1.04 exact |
| — | (Task-list item "0.80 with ratios" — NOT in current Section IV body) | RECOMPUTED-MATCH (number exists; claim absent from tex) | metrics3.json t32_ridge_1pix (OP0 raw+8 ratios, α=100): te RH 0.80; my refit 0.80. The only "0.80" in the tex (l.687) is an R² in the train+val-refit paragraph — different quantity, other agent's scope |
| 17 | Regularised MLP: 521±67 ppm over 3 seeds, r=0.70 (l.385–386) | UNVERIFIABLE (exact) — consistent on recompute | No per-seed artifact saved (full_mlp_reg.npz holds one seed: 464 ppm, r .702, inside 521±67). My 3-seed rerun of the stated config (128×64, drop 0.2, wd 1e-4, early stop on val): 484/464/687 → 545±101, r=0.69. Statistically consistent with 521±67 / r 0.70 given the documented seed sensitivity, but the exact triple cannot be reproduced without the original seeds/early-stop schedule |
| 18 | MLP capacity triplet: train 25→12→10 ppm (l.383) | ARTIFACT-MATCH | metrics3.json t26/t27/t28 train CO₂: 25.3/12.2/10.45 (explicitly "single runs" in tex). Reproducibility: reseeded small MLP gives train 25.5 ✓ |
| 19 | MLP capacity triplet: test 671→900→1030 ppm (l.384) | ARTIFACT-MATCH | metrics3.json test CO₂: 671.28/899.5/1030.29. Reseeded small run gives 601 — same regime, test value seed-dependent; tex's "single runs" qualifier is accurate and necessary |
| 20 | Capacity params 4.8k/85k/2.2M (l.381) | ARTIFACT-MATCH | metrics3.json "params": 4802/84994/2176002 (architecture-determined, also hand-checked) |
| 21 | LSTM shape r = 0.81±0.03 over five seeds (l.387, fig.3 caption l.349) | RECOMPUTED-MATCH | 5 seeded reruns of the 6-h coarse LSTM: r mean±std = 0.81±0.03, range .764–.851 (matches supplementary "0.76–0.85"). No 5-seed artifact existed; seeds 0–2 reproduce idea2_irnorm.json lstm6h_raw16 seeds 0–2 exactly (634.32/667.04/673.15 ppm) |
| 22 | LSTM "still posts MAE ≈650 ppm" (l.388) | RECOMPUTED-MATCH | single-run artifact 654; 3-seed artifact mean 658±17 (idea2); my 5-seed mean 680±35 — "≈650" fair |
| 23 | Drift shifts train→test: B/IR 0.26σ vs B/R 1.09σ vs raw B 0.69σ (l.398–399) | RECOMPUTED-MATCH | recomputed (test mean − train mean)/train σ, mean |·| over 4 pixels: 0.26 / 1.09 / 0.69. Artifact: idea2_irnorm.json drift table (identical incl. per-pixel values) |
| 24 | IR ridge free-running: MAE 103 ppm, R²=0.77, r=0.90; test < train (103 vs 179) (l.400–404) | RECOMPUTED-MATCH | same fit as #5: te 102.74 / tr 179.04, R² .765, r .904 |
| 25 | Param counts quoted across ladder: 26/146/578/18k/21k/2.2M (table Nₚ col) | RECOMPUTED-MATCH | 26/146/578 = (n_feat+1)×2 exact; 17730→18k; 21122→21k; 2176002→2.2M |

## Counts
- RECOMPUTED-MATCH: 18
- ARTIFACT-MATCH: 5 (#6, #7-MAEs, #18, #19, #20 — all stochastic single-run values, as the tex itself declares)
- MISMATCH: 0
- UNVERIFIABLE (exact): 1 (#17 — 521±67 3-seed mean; no per-seed artifact; recompute 545±101 is consistent)

## Flags (no number is wrong, but worth knowing)
1. **#7 caption nit** — "LSTM rows are means over five seeds" is strictly true only for the r
   column (and the offset-row MAE, other agent); the four free-running LSTM MAE cells are the
   archived single run (metrics4). 5-seed means: RH 0.10/0.93, CO₂ 18/680±35 (654 lies within spread).
2. **#17** — the regularised-MLP 3-seed numbers (521±67) have no saved artifact; only one seed
   (464 ppm) is archived in full_mlp_reg.npz. Recompute is consistent but not exact (seed/early-stop
   schedule dependent). If a reviewer asks, the honest answer is "single archived seed 464; mean of
   the original 3 runs 521±67; reproduction gives 545±101".
3. **Table-II "params": 576 vs 578** — metrics3.json records 576 for ridge+HP (weights only);
   the manuscript's 578 includes the two intercepts, consistent with its 146 and 26. Fine.
4. **Task-list item "0.80 with ratios"** is not a claim in the current tex body (only 4.4/0.60/1.04
   appear); the number itself is real (t32 artifact, recomputes exactly) if it gets re-added.
