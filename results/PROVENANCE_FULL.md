# PROVENANCE A — Abstract / Introduction / Sec II / Sec III (+ Catalan abstract)

Verified 2026-06-11. Every claim recomputed from
`/home/bretech/OTHERS/TFGV4/unified_5s_corrected.csv` via `data4.py` conventions
(load → dropna targets → dropna ink → chronological 70/15/15 with 4-min gaps),
unless marked otherwise. Recompute scripts: `/tmp/verify_a.py` (dataset/validation),
`/tmp/verify_b.py` (Table I, RH term, ΔB/R, lags), `/tmp/verify_c.py` (leakage +
abstract model numbers), `/tmp/verify_d.py` (64 %, RH-term variants, lamp ablation).
Methodology for Table I taken from `build_figs.py` (lamp-stable segments ≥ 2 h from
rounded `lamp_pct`, first hour skipped, pixel-averaged B/R, joint lstsq
B/R = αCO₂ + βRH + γ; partial correlation = standard 3-variable formula).

| # | CLAIM | VERDICT | PROVENANCE |
|---|-------|---------|------------|
| 1 | 77-hour protocol (abstract, II, Catalan) | RECOMPUTED-MATCH | CSV `t_h` max = 76.75 h → rounds to 77 |
| 2 | CO₂ sweep ~100–1700 ppm (abstract, II, III, Catalan) | RECOMPUTED-MATCH | measured `SCD_CO2` 93.7–1684 ppm; within the stated "~" |
| 3 | Four RGB-IR pixels = 16 ink channels (abstract, II) | RECOMPUTED-MATCH | 16 `OP{0-3}_{red,green,blue,ir}` columns in CSV |
| 4 | Sensitivity ≈ −5×10⁻⁶ per ppm (abstract, conclusions echo, Catalan) | RECOMPUTED-MATCH | mean α over the 5 included Table I segments = −5.4×10⁻⁶, range −4.3…−6.2 (verify_b.py) |
| 5 | Humidity cross-sensitivity order 10³ ppm per %RH, "loosely constrained" (abstract, III, Catalan) | MATCH-WITH-CAVEAT | recomputed global β/\|α\|: 432 (lamp-stable segs excl. seg0), 542 (all segs), 566 (whole record), 967 (train-only) ppm/%RH; order-10³ holds only loosely (0.4–1.0×10³ depending on fit window); per-segment values span +62…−1790 incl. a sign flip, exactly as the text concedes |
| 6 | Lag up to tens of minutes (abstract, Catalan) | RECOMPUTED-MATCH | max best-lag = 44.5 min (test-block lamp-stable segment, verify_b.py) |
| 7 | RH from ink alone: MAE 0.35 %, R² = 0.993 (abstract, Catalan) | RECOMPUTED-MATCH | refit ridge on 72 feats + 8-h high-pass, α-grid on val → test MAE 0.3499→0.35, R² 0.9931 (verify_c.py); matches `results/metrics.json` t3_ridge_hp8h |
| 8 | 12-feature (R,G,B)/IR linear model: MAE 103 ppm, R² = 0.77 free-running (abstract, Catalan) | RECOMPUTED-MATCH | refit IRN12 ridge (α=100 on val): test MAE 102.74, R² 0.765, r 0.904 (verify_c.py) |
| 9 | 50 ppm with one offset per illumination change (abstract, Catalan) | RECOMPUTED-MATCH | champion protocol (per-lamp-segment 30-min offset, cal samples excluded): MAE 49.71, R² 0.707, r 0.909 (verify_c.py) |
| 10 | 54-ppm hold-the-calibration baseline (abstract) | RECOMPUTED-MATCH | hold cal-window mean per segment: MAE 53.81 (verify_c.py) |
| 11 | R² 0.71 vs 0.57 (offset ridge vs hold baseline) (abstract) | RECOMPUTED-MATCH | recomputed 0.707 vs 0.566 (verify_c.py) |
| 12 | Lamp-stable 34 % of test (abstract, IV, Catalan) | RECOMPUTED-MATCH | used_frac = 0.344; evaluated-window range 283–623 ppm also reproduced (verify_c.py) |
| 13 | ~420-ppm outdoor background (intro) | UNVERIFIABLE | literature/common knowledge; not derivable from this CSV |
| 14 | >1500 ppm has measurable cognitive effects (intro) | UNVERIFIABLE | literature claim, cited (Satish 2012); not derivable from CSV |
| 15 | ΔB/R < 1 % over the full range (intro) | RECOMPUTED-MATCH | fitted CO₂-driven swing α×1600 ppm relative to mean B/R (1.176): 0.74–0.83 % (verify_b.py); raw within-segment swing incl. drift/noise is 1.12 % — claim refers to the CO₂ modulation, which holds |
| 16 | Effective time constants minutes to tens of minutes (intro) | RECOMPUTED-MATCH | best-lag cross-correlation: 0.5–19.5 min in slow segments, 44.5 min in fast block (verify_b.py) |
| 17 | CO₂ hydration step adds only seconds (intro) | UNVERIFIABLE | textbook kinetics; not testable from this dataset |
| 18 | Five slow triangular sweeps, ~11 h each (intro: "~11 h per cycle"; II) | RECOMPUTED-MATCH | 5 setpoint peaks at t = 6.8/18.3/29.8/40.8/50.3 h; cycle spacings 11.5/11.5/11.0/9.5 h (verify_a.py) — "~11 h" fair |
| 19 | Deliberately fast final ~12 h reserved for evaluation (intro) | MATCH-WITH-CAVEAT | test block spans 65.30–76.75 h = 11.44 h; appendix itself says 11.5 h; "~12" is a generous rounding |
| 20 | "its own sixteen numbers" (intro) | RECOMPUTED-MATCH | 16 ink channels (row 3) |
| 21 | 0.2 Hz acquisition, common 5-s grid (II) | RECOMPUTED-MATCH | `diff(t_s)` = 5.000 s for all 55,259 intervals → exactly 0.2 Hz |
| 22 | RH staircase between 17 and 44 %RH (II) | RECOMPUTED-MATCH | measured `BME_RH` 16.51–43.78 % → 17–44 at integer rounding |
| 23 | Lamp-duty steps between 20 and 80 % (II) | RECOMPUTED-MATCH | rounded `lamp_pct` spans 20–80 (plateaus at 20/30/40/50/60/70/80 + ramp values) |
| 24 | 55,258 samples (II) | MATCH-WITH-CAVEAT | the 5-s grid has 55,260 rows; 55,258 is after dropping 2 rows with missing ink channels (the modelling table, `data4.load_light_fe`). Off-by-two on the literal "resampled onto a common 5-s grid (55 258 samples)" reading |
| 25 | Mislabelled humidity-dosing channel tracks NDIR at r = 0.97 (II) | RECOMPUTED-MATCH | corr(`co2_set_ppm` ≡ ex-"H2O" f16, `SCD_CO2`) = 0.975; mapping documented in notes/data_truth.md |
| 26 | Channel labelled CO₂ is the dry-air dilution line (II) | RECOMPUTED-MATCH | `dry_dilution` (ex-"CO2" f18): r = −0.49 vs SCD, −0.76 vs RH — dilution-like, not CO₂; matches data_truth.md evidence |
| 27 | SCD30 reaches ~75 % of nominal setpoint (II) | RECOMPUTED-MATCH | mean(SCD/set, set>300) = 0.747; through-origin slope 0.785 (verify_a.py) |
| 28 | ~64 % when the humid line runs high (II) | MATCH-WITH-CAVEAT | statistic-dependent: validation-block median ratio 0.642 and humid-line-high (val, h2o_set ≥ median) mean 0.645 reproduce ~64 %; plain val-block mean is 0.667, wet-line-high global mean 0.70–0.71 (verify_a/d.py). The ~64 % is defensible but corresponds to a specific slice/statistic |
| 29 | 16 raw + 56 derived = 72 features (II) | RECOMPUTED-MATCH | `data4.load_light_fe()` → X.shape[1] = 72 = 16 raw + 32 per-pixel ratios + 24 cross-pixel stats |
| 30 | Adding lamp duty changes instantaneous models by ~3 % (II) | MATCH-WITH-CAVEAT | reproduced for ridge-72: free-running CO₂ MAE 403.3→390.3 = −3.2 % (verify_d.py); but the IR-ridge degrades 103→286 ppm when lamp is appended (α re-selected on val), so the "~3 %" holds only for the 72-feature model |
| 31 | …and the offset-corrected error by <1 ppm in a single-seed check (II) | UNVERIFIABLE | exact configuration unknown (no artifact found in TFGV4/results); closest reproductions change by +1.8 ppm (ridge-72+offset 122.0→123.9) and +5.3 ppm (IR-ridge+offset 49.7→55.0) (verify_d.py). Flag to author: as stated, not reproduced |
| 32 | Chronological 70/15/15 split with 4-min guard gaps (II, data4) | RECOMPUTED-MATCH | actual fractions 70.0/14.91/14.91 % (gaps shave ~0.1 %); gaps 4.08 and 4.00 min (verify_a.py) |
| 33 | Same ridge: CO₂ R² = +0.86 random split vs −4.1 chronological (II) | RECOMPUTED-MATCH | refit from CSV: chrono ridge-72 (α=1000 on val) test R² = −4.104; random split (seed 0, α=100, as `run_ladder3.t31`) test R² = +0.864 (verify_c.py); matches metrics3.json |
| 34 | Test block modulates CO₂ on 15-min steps (II) | RECOMPUTED-MATCH | median interval between setpoint changes in test = 14.9 min (verify_a.py) |
| 35 | Test range narrows to 283–1609 ppm (II) | RECOMPUTED-MATCH | SCD over test rows: 282.7–1609.1 ppm |
| 36 | Film has aged three days at test time (II) | MATCH-WITH-CAVEAT | test starts at t = 65.3 h = 2.72 days; "three days" is a rounding up |
| 37 | Floors: train-mean predictor gives test MAE 3.88 %RH and 197 ppm (II) | RECOMPUTED-MATCH | recomputed 3.877 %RH and 197.35 ppm (verify_a.py) |
| 38 | Fig. 1 caption: chamber tracks setpoint at r = 0.97 | RECOMPUTED-MATCH | same correlation as row 25: 0.975 |
| 39 | Table I row 60 %: t 14–24.5 h, 108–1680 ppm, α = −4.3×10⁻⁶, r_p = −0.77 | RECOMPUTED-MATCH | segment refit (build_figs.py method): α = −4.34, r_p = −0.773, CO₂ 108–1680, t 14.0–24.5 (verify_b.py) |
| 40 | Table I row 50 %: 25.5–36 h, 102–1668, −5.0, −0.95 | RECOMPUTED-MATCH | α = −5.00, r_p = −0.954, CO₂ 102–1668.6 (table truncates), t 25.5–36.0 |
| 41 | Table I row 40 %: 37–46.5 h, 99–1486, −5.7, −0.99 | RECOMPUTED-MATCH | α = −5.66, r_p = −0.988, CO₂ 99.6–1486 (truncated to 99), t 37.0–46.5 |
| 42 | Table I row 30 %: 47.5–55 h, 93–1070, −5.9, −0.91 | RECOMPUTED-MATCH | α = −5.90, r_p = −0.905, CO₂ 93.7–1070.6 (truncated), t 47.5–55.0 |
| 43 | Table I row 50 % late: 75–76.7 h, 282–556, −6.2, −0.83 | RECOMPUTED-MATCH | α = −6.22, r_p = −0.833, CO₂ 282.7–556 (truncated), t 75.0–76.7 |
| 44 | Excluded first segment (70 % lamp, 1–13 h): α = −7.5×10⁻⁶, r_p = −0.68 | RECOMPUTED-MATCH | α = −7.48, r_p = −0.679, t 1.0–13.0 h (verify_b.py) |
| 45 | α ≈ −4 to −6×10⁻⁶ ppm⁻¹, partial correlations reaching −0.99 (III) | RECOMPUTED-MATCH | included-segment α range −4.34…−6.22; max \|r_p\| = 0.988 → −0.99 |
| 46 | No detectable curvature over 100–1700 ppm (III) | MATCH-WITH-CAVEAT | qualitative; consistent with linear-fit r_p to −0.99 within scatter, but no explicit curvature test was recomputed |
| 47 | Slope persists: −6.2 vs −5.0×10⁻⁶ (~25 %), 50 % lamp re-measured 40 h later (III + fig2 caption + conclusions) | RECOMPUTED-MATCH | both slopes reproduced exactly; Δ = 24.4 %; gap between 50 %-lamp segments: end 36 h → start 75 h = 39 h ≈ 40 h |
| 48 | β only loosely identified, per-segment value varies in magnitude and even sign (III) | RECOMPUTED-MATCH | per-segment β/\|α\| (ppm/%RH): −1790 (70 %), +62 (60 %), −108 (50 %), −465 (40 %), −396 (30 %), −1153 (50 % late) — sign flip and 30× magnitude spread confirmed |
| 49 | ~3-min apparent lag in slow sweeps (III) | MATCH-WITH-CAVEAT | best-lag of RH-corrected B/R vs SCD: 3.0 min in the 50 %-lamp slow segment (the figure-2 segment), but 0.0/0.5/19.5 min in the 60/40/30 % segments — "~3 min" is the representative mid-protocol value, not universal |
| 50 | Up to ~44 min lag in the fast block (III) | RECOMPUTED-MATCH | best-lag 44.5 min in the test-block lamp-stable 50 % segment (verify_b.py) |
| 51 | Chamber temperature 27.3–31.2 °C (Sec IV caveat; in check list) | RECOMPUTED-MATCH | `BME_T` 27.34–31.16 °C |
| 52 | Catalan abstract echoes (77 h, ~100–1700 ppm, −5×10⁻⁶, 10³ ppm/%HR, MAE 0,35 %, R² 0,993, 103 ppm/0,77, 50 ppm, 34 %) | RECOMPUTED-MATCH | identical to rows 1–12; no numeric divergence between English and Catalan abstracts |

## Counts
- RECOMPUTED-MATCH: 39
- MATCH-WITH-CAVEAT: 9 (rows 5, 19, 24, 28, 30, 36, 46, 49 + none severe)
- MISMATCH: 0
- UNVERIFIABLE: 4 (rows 13, 14, 17 — literature; row 31 — lamp-input <1 ppm check not reproducible)

## Items worth an author pass
1. **Row 31** ("offset-corrected error changes by <1 ppm with lamp as input"): no artifact
   exists and the two closest reproductions give +1.8 and +5.3 ppm. Either soften to
   "a few ppm" or document the exact configuration that gave <1 ppm.
2. **Row 5/abstract "order-10³ ppm per %RH"**: defensible only as an order of magnitude;
   recomputed global fits give 0.4–1.0×10³ depending on the fit window (train-only fit
   gives 967). The "loosely constrained" qualifier already covers this.
3. **Row 24**: "55 258 samples" is the after-ink-dropna modelling table; the literal 5-s
   grid has 55 260 rows. Harmless, but "55 258 complete samples" would be exact.
4. **Row 28**: "~64 %" matches the validation-block median (0.642) / humid-high-val mean
   (0.645); the plain val-block mean is 0.667. Consider "~64–67 %" or state the slice.
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
# PROVENANCE_C — Offset correction, train+val refit, indicator mode, appendix

Verified 2026-06-11 against `/home/bretech/OTHERS/TFGV4` (data `unified_5s_corrected.csv`,
55 258 rows; splits `data4.split_indices`, chronological 70/15/15, 4-min gaps; test = 8 241
samples). Recomputation script: `/home/bretech/OTHERS/TFGV4/verify_provC.py` (output snapshot
`results/verify_provC.json`). Protocol = `run_ladder2.py::r1/r2` / `run_idea2.py::protocol_eval`
(per-lamp-segment 30-min offset, calibration samples excluded), reproduced independently.

Verdicts: **RECOMPUTED-MATCH** (re-derived from raw data/deterministic refit, or pure
arithmetic on stored predictions), **ARTIFACT-MATCH** (matches stored predictions/metrics of a
stochastic model; training not re-run), **MISMATCH**, **UNVERIFIABLE**.

---

## A. Offset-correction block (Sec. IV.D + Table I bottom + abstract + recal-policy appendix paragraph)

| # | Claim (manuscript) | Verdict | Found / provenance |
|---|---|---|---|
| A1 | Hold-calibration baseline 54 ppm, R²=0.57, r 0.79 | **RECOMPUTED-MATCH** | 53.81 / 0.566 / 0.788 — recomputed from raw data (hold cal-window mean per segment). No prior artifact stored this; it reproduces exactly. |
| A2 | Corrected ridge-72 = 122 ppm | **RECOMPUTED-MATCH** | 122.03 — deterministic refit + protocol; also `metrics2.json::r1_ridge72_segrecal` and `idea2_irnorm.json::ridge::ridge72`. |
| A3 | Corrected MLP = 97 ppm | **UNVERIFIABLE** | The only stored MLP prediction file (`results/full_mlp_reg.npz`, single seed) gives **78.9** under the standard protocol (82.2 if cal samples included). No artifact anywhere contains 97 for an MLP recal. Presumably the 3-seed mean (manuscript MLP is 3-seed, 521±67 free-running; stored seed free-runs at 464); the other two seeds' predictions were not retained. Claim direction ("falls short" of the 54 bar) holds for any of these values. |
| A4 | B/R+RH + recal = 54 ppm, R²=0.65 (table: r 0.82) | **RECOMPUTED-MATCH** | 53.65 / 0.645 / 0.822 — exact refit of `run_ladder2.py::r2`; also `metrics2.json::r2_BR_RH_linear_segrecal`. |
| A5 | "with measured or ink-derived RH alike" | **RECOMPUTED-MATCH** | Replacing measured RH with the manuscript's own ink-RH model (ridge72+HP8h, α=10, test MAE 0.35 %RH — itself reproduced): recal MAE **53.79** (apply-only) / **53.80** (fit+apply), R² 0.654 vs 53.65 / 0.645 measured. Also consistent with stored `preds_r3_physics_heldout.npz` test half (52.8, slightly different mask, used_frac 0.365). |
| A6 | IR-ridge + offset: **50 ppm, R²=0.71, r=0.91** | **RECOMPUTED-MATCH** | 49.71 / 0.707 / 0.909 — full deterministic re-derivation (IRN12 features from raw CSV, α=100 selected on val, protocol re-implemented). Stored `preds_irn12_recal.npz` and `full_ridge_irn12.npz` agree with the recomputation to 3.4e-11 ppm. |
| A7 | LSTM 6-h + offset 47.7±1.2 ppm over 5 seeds, R²=0.68 (table: 48, r 0.82) | **ARTIFACT-MATCH** (mean), per-seed spread unverifiable | Best-seed file `preds6_lstm_recal_best.npz` recomputed: 46.69 / 0.687 / 0.830 on the identical n=2838 mask. The 47.7±1.2 and R² 0.68 appear as frozen references in two contemporaneous records (`idea2_irnorm.json` "47.7±1.2 (best file 46.7, n_used 2838)"; `idea3_transformer.json::champion_reference` mae [47.7, 1.2], r2 0.68). The per-seed predictions (5 seeds) were not retained; the producing script of `preds6_*` is gone. |
| A8 | Without the offset Eq.(2) fails like everything else: **171 ppm** | **RECOMPUTED-MATCH** | 171.24 — free-running B/R+RH model (train fit) on the **full** test block. (On the evaluated windows only it would be 71.9 — the manuscript number is the full-test reading, consistent with "fails like everything else".) |
| A9 | Coverage: 34 % of test, 283–623 ppm, 3.9 h (n given as 2838 in notes) | **RECOMPUTED-MATCH** | used_frac 0.3444, n_used 2838 = 3.94 h, CO₂ range [283, 623] ppm. |
| A10a | Reducing cal window to 10 min degrades the physical model to ~100 ppm | **RECOMPUTED-MATCH** | 100.69 ppm (offset from first 10 min, evaluated after it — same rule as the 30-min protocol; evaluated set grows to 63 % incl. dynamic windows). |
| A10b | Extending cal window to 60 min brings no further gain | **MISMATCH (protocol-ambiguous; found 76.8 or 51.2)** | Under the protocol used everywhere else (offset from first 60 min, evaluate after it) the result is **76.75 ppm** on the shrunken evaluated set (15 % of test) — worse, not "no gain", though the evaluated windows are no longer comparable. Under the alternative reading (60-min offset, evaluation kept on the standard after-30-min mask) it is **51.16** vs 53.65 — a small 2.5-ppm *gain*, ≈"no further gain". Neither reading is stored as an artifact; the sentence is defensible only under the second reading and should state the convention. |

Abstract restatement ("50 ppm against a 54-ppm hold-the-calibration baseline (R² 0.71 vs 0.57) on the lamp-stable 34 % of test") — all four numbers covered by A1/A6/A9: RECOMPUTED-MATCH.
Fig. 6 caption "tie at 50/48 ppm" — covered by A6/A7.

## B. Train+validation refit (Sec. IV.E + appendix "Train+validation refit" paragraph)

| # | Claim | Verdict | Found / provenance |
|---|---|---|---|
| B1 | ridge-72: 403→250, R² still <0 | **RECOMPUTED-MATCH** | 403.34→249.57, R² −1.07. Deterministic refit on train∪val (α=1000); identical to stored `tvfull_ridge72.npz`. |
| B2 | ridge+HP: 297→137, R²=0.48 | **RECOMPUTED-MATCH** | 296.75→136.96, R² 0.484 (α=100, HP 4 h); identical to `tvfull_ridge_hp4.npz`. |
| B3 | MLP: 521→155, R²=0.33 (3 seeds) | **UNVERIFIABLE (single stored seed: 464→147.4, R² 0.358)** | Stored artifacts are single-seed: `full_mlp_reg.npz` free-running test 464.1 (consistent with 521±67 over 3 seeds), `tvfull_mlp_reg.npz` 147.4 / R² 0.358 (consistent with a 155 / 0.33 three-seed mean). Per-seed predictions and any metrics record of the 3-seed means are absent. Direction and magnitude of the improvement are solid. |
| B4 | IR ridge free-running: 103→86 (R²=0.80, r=0.92) | **RECOMPUTED-MATCH** | 102.74→86.41, R² 0.801, r 0.920 (α=100); identical to `tvfull_ridge_irn12.npz`. |
| B5 | IR ridge + offset: 50→45 (R²=0.76) | **RECOMPUTED-MATCH** | 49.71→45.27, R² 0.763; identical to `tvrecal_ridge_irn12.npz`. |
| B6 | LSTM shape r 0.81±0.03 (5 seeds) → 0.05 (one seed) | **ARTIFACT-MATCH** | r recomputed from stored `tvfull_lstm6h.npz` test slice = **0.052**. The 0.81±0.03 pre-refit baseline is the recorded 5-seed mean (see A7). Refit is explicitly single-seed in the manuscript — matches the single stored artifact. |
| B7 | LSTM + offset: 48→93 | **ARTIFACT-MATCH** | `tvrecal_lstm6h.npz` recomputed on its used mask: **92.5** (r 0.498, R² −0.354). Baseline 48 = 47.7 rounded (A7). |
| B8 | B/R+RH + offset: 54→53 | **RECOMPUTED-MATCH** | 53.65→53.35 (deterministic train∪val lstsq refit); identical to `tvrecal_physics.npz`. |

## C. Indicator mode (Sec. IV.D last paragraph + conclusions)

Pure arithmetic on `results/full_ridge_irn12.npz` (free-running IR-ridge, test slice, no
offset), labels from measured SCD30; independently cross-checked against a from-scratch
refit of the same model (predictions agree to 3e-11).

| # | Claim | Verdict | Found |
|---|---|---|---|
| C1 | AUC 0.98 at 800 ppm | **RECOMPUTED-MATCH** | 0.982 |
| C2 | AUC 0.99 at 1000 ppm | **RECOMPUTED-MATCH** | 0.988 → rounds to 0.99 |
| C3 | Accuracies 92 % / 96 % | **RECOMPUTED-MATCH** | 91.8 % / 96.2 % (threshold = the same 800/1000 ppm on the prediction) |
| C4 | Majority-class baselines 79 % / 88 % | **RECOMPUTED-MATCH** | 79.4 % / 88.5 % (positive fractions 0.206 / 0.115) |

## D. Appendix (supplementary material)

| # | Claim | Verdict | Found / provenance |
|---|---|---|---|
| D1 | Raw-window LSTMs (5–30 min) reach only 574–582 ppm | **ARTIFACT-MATCH** | `metrics.json`: t8_lstm_W360 = 574.27, t7_lstm_W60 = 582.17. Predictions stored (`preds_t7/t8`). Not re-trained. |
| D2 | Window-normalised LSTMs: 386/299 ppm with r=0.12/0.10 | **ARTIFACT-MATCH** | `metrics.json`: t9 = 386.47, r 0.118; t10 = 299.2, r 0.103. |
| D3 | Dead reckoning lost on test: 61 ppm | **ARTIFACT-MATCH (with caveat)** | Recomputed from stored `preds5_rnn_champion.npz` used mask: **60.78**. Caveat: 60.78 is the *bounded-integration* (leak λ=4e-3 toward the physics baseline) final shot; the pure integrate-from-anchor variant scored 97.1 (`RNN_CHAMPION_NOTES.md` §4). The manuscript sentence describes anchor integration but quotes the final number. Also "10-seed ensemble" = trained 10 seeds, averaged the val-best 5. |
| D4 | Derivative bias of only 0.025 ppm s⁻¹ … fatal in flat windows | **ARTIFACT-MATCH** | `RNN_CHAMPION_NOTES.md` §4: 88 ppm accumulated over a 1-h flat segment ⇒ 88/3600 ≈ 0.024 ppm/s. Arithmetic checks; underlying per-segment error analysis not independently re-run. |
| D5 | PatchTST transformer lost: 85 vs 48 ppm corrected; shape r ≈ −0.4 | **ARTIFACT-MATCH** | Recomputed from stored `preds_idea3_patchtst_final.npz`: corrected 84.7 on the standard mask; shape r (raw, free-running) = −0.377. Champion 47.7 (A7). 3-seed per-seed values in `idea3_transformer.json` (recal 74.1/67.6/170.0 — the 84.7 is the 3-seed prediction ensemble, which is what the manuscript quotes). |
| D5b | "(107k params, val-selected, 3 seeds)" | **MISMATCH (found 105 154)** | `idea3_transformer.json` winner P4_p24_d64_L3_dr01 `n_params` = **105 154** (~105k). The "~107 k" originated in `IDEA3_NOTES.md` and was copied. Trivial fix: 107k → 105k. |
| D6 | Adversarial RH probe still recovers RH at R²≈0.93; forcing invariance collapses CO₂ | **ARTIFACT-MATCH** | `idea4_adversarial.json` / `IDEA4_NOTES.md`: fresh-probe RH R² 0.92–0.93 at every λ (LSTM), 0.86–0.95 (MLP); at λ=5 probe dip 0.89→0.67 comes with co2_probe_r 0.30→0.18. Not re-trained. |
| D7 | "tiny MLP 143 vs 86" | **UNVERIFIABLE / NOT IN MANUSCRIPT** | The manuscript (Sec. IV.C) says only "tiny MLPs only lose", with no numbers; no "143 vs 86" appears in `TFG.tex`. No artifact in `TFGV4/results` records a tiny-MLP-on-IRN12 = 143 (the only 143s are unrelated `idea4` entries; 86 is the refit IR ridge, B4). If a draft carried these numbers, their provenance is not on disk. |
| D8 | Five seeds: shape r 0.76–0.85 | **UNVERIFIABLE as a range (endpoints consistent)** | Best seed recomputed from `preds6_lstm_recal_best.npz` raw channel (NaN warm-up masked): r = **0.851** — matches the 0.85 top end. Mean 0.81±0.03 recorded in `idea3_transformer.json::champion_reference`. The 5 per-seed values were not retained, so the 0.76 bottom end cannot be checked. |
| D9 | Five seeds: corrected MAE 46.7–49.9 | **UNVERIFIABLE as a range (endpoints consistent)** | Best seed recomputed: 46.69 — matches 46.7. Mean 47.7±1.2 recorded twice (A7). 49.9 top end consistent with mean+σ but per-seed values absent. |
| D10 | B/R noise after 60-s averaging ~1.1×10⁻⁴ ≈ 22 ppm at the measured sensitivity | **RECOMPUTED-MATCH (approx.)** | Successive-difference noise estimator on pixel-mean B/R: per-5-s-sample σ 4.17e-4 ⇒ after 60-s averaging 4.17e-4/√12 = **1.20e-4 ⇒ 24 ppm** at 5e-6 ppm⁻¹. Within 10 % of the quoted 1.1e-4/22 ppm (exact estimator undocumented; the stated arithmetic 1.1e-4 ÷ 5e-6 = 22 is exact). |
| D11a | Adding lamp duty changes the instantaneous models by ~3 % (single-seed check) | **RECOMPUTED-MATCH for ridge-72 (with caveat)** | ridge-72 + lamp_pct, val-selected α: test MAE 390.3 vs 403.3 = **−3.2 %**. Caveat: the same operation on the IR-ridge (IRN12 + lamp) degrades free-running MAE 103→286 ppm — "the instantaneous models" generalises further than the artifacts support. |
| D11b | …and the offset-corrected error by <1 ppm | **UNVERIFIABLE (deterministic analogues contradict)** | No artifact documents this check. Deterministic recomputations: ridge-72+lamp corrected 123.9 vs 122.0 (**+1.8 ppm**); IRN12+lamp corrected 55.0 vs 49.7 (**+5.3 ppm**); B/R+RH+lamp corrected 156.9 vs 53.7 (+103 ppm, lamp coefficient extrapolates badly). If the original check was the single-seed LSTM (whose "devreal" variant already consumed lamp_pct), its predictions were not retained. The "<1 ppm" figure has no surviving support and is contradicted by every deterministic analogue. |

---

## Summary

35 checkable items (counting compound claims by their numeric parts):

- **RECOMPUTED-MATCH: 22** — hold baseline (3 numbers), ridge-72 recal, B/R+RH recal (MAE+R²+r), ink-RH equivalence, IR-ridge+offset (MAE+R²+r), Eq.(2) 171, coverage (3 numbers), 10-min ~100, refit ridge-72/HP/IR-free/IR-recal/physics (with R²/r side-claims), indicator (4 lines), noise scale (approx.), lamp ~3 % (ridge-72).
- **ARTIFACT-MATCH: 9** — LSTM 47.7±1.2 & R² 0.68, refit LSTM r 0.05 and 93 ppm, raw-LSTM 574–582, winnorm 386/299, dead-reckoning 61 & 0.025 ppm/s, transformer 85/−0.4, adversarial 0.93.
- **MISMATCH: 2** — 60-min cal window "no further gain" (A10b: 76.8 under the consistent protocol; only ≈holds under a fixed-eval-mask reading); transformer "107k params" (actual 105 154).
- **UNVERIFIABLE: 6** — corrected MLP 97 (stored seed gives 78.9), MLP refit means 521→155/R² 0.33 (stored seed 464→147.4/0.358), tiny-MLP 143-vs-86 (not in manuscript, no artifact), seed ranges 0.76–0.85 and 46.7–49.9 (best endpoints verified, per-seed lists absent), lamp <1 ppm (contradicted by deterministic analogues).

Recommended edits: (1) fix 107k→105k; (2) state the 60-min-calibration evaluation convention
or soften to "60 min changes the result by a few ppm at best"; (3) either re-derive/soften the
MLP "97" (the stored seed gives 79 — the qualitative point survives) or drop the number;
(4) soften or qualify the "<1 ppm" lamp-ablation figure (which model, single seed,
not reproducible from stored artifacts; deterministic analogues give +1.8 to +5.3 ppm).
