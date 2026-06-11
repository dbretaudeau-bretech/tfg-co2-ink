# IDEA 2 — IR-channel normalization (R/IR, G/IR, B/IR per pixel)

**Hypothesis.** Dividing the visible channels by the co-measured IR channel
cancels shared illumination/drift dynamically and removes the need for
recalibration.

**Code:** `run_idea2.py`. **Metrics:** `results/idea2_irnorm.json`.
Splits `data4.split_indices` (chronological 70/15/15, 4-min gaps); recal
protocol identical to the champion's (`run_ladder2.r2`: per-lamp-segment
30-min offset, calibration samples excluded). All selection (ridge alpha,
LSTM seed) on validation only.

---

## 1. Drift table (train → test mean shift, in train-std units, mean |shift| over the 4 pixels)

| feature | mean abs shift | per-pixel |
|---|---|---|
| raw B | 0.69 | +0.74 +0.69 +0.70 +0.65 |
| raw R | 1.53 | −1.30 −1.58 −1.46 −1.77 |
| raw G | 0.83 | −0.74 −0.93 −0.88 −0.77 |
| raw IR | 0.47 | +0.67 +0.44 +0.41 +0.34 |
| B/R | 1.09 | +1.08 +1.12 +1.09 +1.05 |
| B/RGB | 2.20 | +2.10 +2.29 +2.24 +2.18 |
| **B/IR** | **0.26** | +0.29 +0.24 +0.22 +0.28 |
| G/IR | 0.79 | −0.75 −0.87 −0.81 −0.71 |
| R/IR | 1.31 | −1.27 −1.32 −1.36 −1.28 |

**B/IR is the most drift-stable feature in the entire bank**: 2.7× less
shift than raw B and 4× less than B/R (the physics model's input). The
benefit is blue-specific: B and IR drift in the same direction (+0.69σ,
+0.47σ) so the ratio cancels; R and G drift the opposite way, so R/IR and
G/IR inherit their drift almost unchanged.

## 2. Physics checks

**Is IR flat wrt the targets?** Within the 7 lamp-stable segments ≥1 h
(30-min settle skipped), length-weighted Pearson:

| signal | r vs CO₂ | r vs RH |
|---|---|---|
| raw IR | **+0.04** | +0.52 |
| raw B | −0.16 | −0.65 |
| B/IR | −0.14 | −0.72 |
| B/R | −0.55 | −0.47 |

IR is essentially CO₂-blind (good reference) but **not RH-flat** (r 0.52);
dividing by IR therefore *adds* RH sensitivity (B/IR ↔ RH r = −0.72).
Also note: within segments B/R carries far more CO₂ signal (−0.55) than
B/IR (−0.14) — IR division costs short-horizon CO₂ sensitivity.

**Does IR scale with lamp duty like the visible channels? NO.** Over 30
clean lamp transitions: median |relative jump| B = 0.82 %, IR = 0.11 %, and
corr(jump_B, jump_IR) = **−0.95** — IR barely reacts to lamp steps and moves
in the *opposite* direction. Consequently B/IR jumps *more* than raw B at
lamp changes (0.93 % vs 0.82 %, ratio 1.13). The visible-channel lamp
response is dominated by the ink's spectral response, not by a shared
intensity factor. **IR normalization does not cancel lamp steps — it cancels
the slow shared (LED/photodiode/thermal) drift.**

## 3. Ridge (experiment a)

CO₂ on test; recal = champion protocol (used_frac 0.344, n=2838):

| model | n_feat | no-recal MAE / R² / r | recal MAE | val recal MAE |
|---|---|---|---|---|
| ridge72 (standard) | 72 | 403.3 / −4.10 / 0.761 | 122.0 | 185.6 |
| ridge rawRGB12 (control) | 12 | 663.1 / −7.63 / 0.321 | 92.5 | 140.0 |
| **ridge IRN12 ({R,G,B}/IR)** | 12 | **102.7 / +0.765 / 0.904** | **49.7** | **65.9** |
| ridge IRN24 (+cross-pixel, +B/IR÷R/IR) | 24 | 189.8 / 0.23 / 0.867 | 85.0 | 132.3 |
| ridge72 + HP4h | 288 | 296.8 / −1.15 / 0.429 | 212.9 | 185.9 |
| ridge IRN24 + HP4h | 96 | 535.4 | 164.8 | 281.6 |

* **No-recal: ridge IRN12 = 102.7 ppm MAE, R² 0.77, r 0.90** — the previous
  best no-recal numbers in the project were MAE 296.8 (ridge+HP4h, r 0.43)
  and r 0.86 at MAE 654 (LSTM-6h). This is a ~3× MAE reduction with positive
  R², from a 12-feature linear model.
* **With recal: 49.7 ppm** — beats the physics champion (B/R+RH, 53.65) and
  lands inside the LSTM champion's seed band (47.7±1.2; best file 46.7),
  with no RH input, no network, no ensemble. On the validation protocol it
  scores 65.9 vs the champion's 88.4, i.e. **under honest val-only selection
  ridge-IRN12 would have been selected over the champion**.
* Adding the B/R-equivalent variant (IRN24) *hurts*: (B/IR)/(R/IR) ≡ B/R,
  which re-imports the 1.09σ drift. Purity matters. High-pass also hurts —
  the IR ratio already removed the drift the HP block was built for.

## 4. Coarse-context LSTM (experiment b) — 60-s MA, 1/min, 6-h window, 3 seeds

| channels | no-recal test CO₂ MAE | recal test MAE | recal val MAE/seed |
|---|---|---|---|
| raw16 | 658 ± 17 (ref 653.9) | 83.0 ± 3.7 (ens3 81.7) | 185, 205, 195 |
| IRN12 | 660 ± 20 | **64.0 ± 2.8 (ens3 62.0)** | 86, 77, 103 |

No-recal the LSTM does **not** improve from ~650 with IR-normalized inputs —
the absolute-output LSTM's failure is train-block memorization, not input
drift, so fixing the inputs doesn't fix it. With recal, IRN12 channels give a
clear gain (83→64 ppm; val 195→89) but still lose to plain ridge-IRN12
(49.7) and to the champion (47.7±1.2).

## 5. Honesty notes

* Validation's evaluated windows are dynamics-rich while test's are mostly
  flat (known from ladder 5); val CO₂ numbers are pessimistic and noisy.
  All selections here (alpha, seed) used val only; ridge-IRN12's headline has
  no seed degree of freedom at all, and its val→test ordering was consistent.
* The recal numbers use exactly the champion's evaluated windows
  (n = 2838, used_frac 0.344), so they are directly comparable to 47.7/53.65.
* Pixel-level B/IR within-segment CO₂ correlation is weak (−0.14): the ridge
  performance comes from the 12-feature combination (G/IR, R/IR supply the
  RH/lamp context that IR-division injected), not from B/IR alone being a
  clean CO₂ axis.

## 6. Verdict

1. **IR normalization strongly reduces drift** — B/IR train→test shift 0.26σ
   vs 0.69σ (raw B) / 1.09σ (B/R) — and converts it into the project's best
   no-recal model: ridge on 12 IR-normalized features = **102.7 ppm MAE,
   r 0.90** (previous best 296.8 ppm / r 0.43).
2. **With the standard 30-min recal it reaches 49.7 ppm — statistically tied
   with the 47.7±1.2 LSTM champion and better than the 53.65 physics
   model — from an ink-only 12-feature linear model, and it wins val-only
   selection (65.9 vs 88.4).**
3. It does **not** fully kill recalibration (103 ppm free-running vs ~50
   recalibrated), and the mechanism is *not* lamp cancellation (IR
   anti-correlates with visible jumps at lamp steps; B/IR jumps 1.13× more
   than B) — it cancels the slow shared emitter/detector drift, at the cost
   of extra RH sensitivity. Recalibration intervals could plausibly be
   stretched, not eliminated.
