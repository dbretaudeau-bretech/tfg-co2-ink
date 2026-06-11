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
