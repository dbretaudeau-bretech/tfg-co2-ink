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
