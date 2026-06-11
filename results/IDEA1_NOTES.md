# IDEA 1 — Chemical-kinetics (derivative) features: full record

**Hypothesis.** The film responds to CO₂ with a 3–44 min first-order lag.
Early rate-of-change of the ink signals should anticipate the asymptote:
either implicitly (d/dt, d²/dt² features fed to ridge/MLP/LSTM) or
explicitly (lead compensation `p + τ·dp/dt`, the textbook inversion of a
first-order lag).

**Verdict up front: kinetics did NOT help — every kinetic variant is worse
than its static counterpart, with and without recalibration.** The ink's
derivatives are essentially pure RH kinetics; their relationship to dCO₂/dt
flips sign between chronological blocks exactly like the static features do.

## Setup

- Code: `run_idea1.py`; all metrics in `results/idea1_kinetics.json`;
  test predictions in `results/preds_idea1_*.npz`.
- Features: causal derivatives of 19 ink signals (16 raw OP channels +
  pixel-mean B/R, B/RGB, IR/R). Per scale w ∈ {1, 5, 15, 60} min:
  rolling-mean smooth over w, then backward difference over w (d1, unit/s),
  and the same operator applied again (d2). 76 d1 + 76 d2 = 152 features.
  Centered Savitzky-Golay was rejected: at the 60-min scale it would leak
  30 min of future samples (deployment-dishonest).
- Splits: `data4.split_indices` (chronological 70/15/15, 4-min gaps).
  Derivatives computed on the continuous record (inputs only — same
  documented liberty as ladders 4/5).
- Evaluation: no-recal = `data4.metrics` on the full 8 241-sample 5-s test
  grid; with-recal = EXACT champion protocol (`run_ladder2.r2`/
  `run_ladder5.protocol_eval`): per-lamp-segment 30-min offset, calibration
  samples excluded (34.4 % of test used, 5 lamp-stable windows).
- All selection on validation; 3 seeds for every NN.
- **Honesty caveat (as instructed):** the validation block spans CO₂
  94–620 ppm (std 168) vs test 283–1609 ppm (std 258), and val CO₂ ranking
  was unstable across model families in every previous ladder. Val CO₂
  selection here is weak evidence; it chose alphas/τ/early-stopping only.

## Baselines (fixed reference points)

| model | test no-recal | test recal (used windows) |
|---|---|---|
| Champion: 6h coarse-LSTM + 30-min/segment offset (`preds6_lstm_recal_best.npz`) | — | **MAE 46.7, R² 0.687, r 0.830** |
| Physics B/R+RH + recal (`r2`, ladder 2) | — | MAE 53.7, R² 0.645 |
| Best no-recal MAE: ridge+HP4h (`t3`) | MAE 297, R² −1.15 | — |
| Best no-recal r: LSTM (prior ladders) | r 0.81 | — |

## Results (test CO₂; NN = mean ± std over 3 seeds)

| config | val no-recal MAE | test no-recal MAE / r | test recal MAE / R² |
|---|---|---|---|
| k1a ridge static72 (control) | 192 | 490 / 0.707 | 88.8 / −0.04 |
| k1b ridge d1-only (76) | 1342 | 1087 / −0.16 | 734 / −76.8 |
| k1c ridge d1+d2 (152) | 1330 | 844 / −0.02 | 379 / −20.5 |
| k1d ridge static72+d1+d2 (224) | 375 | 617 / 0.51 | 237 / −4.1 |
| k1f ridge static72+d1 (148) | 258 | 437 / 0.62 | 234 / −4.4 |
| k1e ridge d1@1m only | 458 | 405 / −0.02 | 193 / −7.2 |
| k1e ridge d1@5m / 15m / 60m | 548–1365 | 537–800 | 253–421 |
| k2 lead comp. τ∈{3..90 min} | τ=0 wins val (219 vs 236+ at τ=3) | τ=0 ⇒ identical to static | — |
| k3a MLP d1+d2 | 978 | 715±392 / −0.04 | 345±163 |
| k3b MLP static72+d1+d2 | 527 | 831±100 / 0.55 | 166±31 |
| k3c MLP static72 (control) | 380 | 741±57 / 0.68 | 93±12 |
| k4 LSTM on d1@1m+d1@15m windows (38ch, 2h ctx) | 248 (coarse) | 351±61 / −0.23 | 314±157 |

Every comparison is monotone in the same direction:
- deriv-only ≪ static (recal 193–734 vs 88.8);
- static+deriv < static (ridge 234 vs 88.8; MLP 166 vs 93);
- windows-of-derivatives LSTM ≪ everything (negative test r);
- explicit lead compensation: validation rejected every τ > 0 — even τ = 3 min
  (the bottom of the film's quoted lag range) made val worse, so the frozen
  test model degenerates to the uncompensated static ridge.

No kinetic config approaches the champion's 46.7 ppm recal MAE, and none
beats the 297-ppm no-recal reference either (best kinetic-containing
no-recal MAE: 351, with *negative* correlation).

## Why kinetics failed (diagnostic, in `run_idea1` post-hoc check)

Correlations of d1(ink) at the 15-min scale, per split:

| signal | corr with dRH/dt (train/val/test) | corr with dCO₂/dt (train/val/test) |
|---|---|---|
| d1 px_BR_mean | −0.86 / −0.98 / −0.97 | **−0.36 / +0.29 / −0.08** |
| d1 px_B_RGB_mean | −0.69 / −0.89 / −0.66 | **−0.55 / +0.16 / −0.61** |
| d1 OP0_blue | −0.96 / −1.00 / −1.00 | −0.21 / +0.30 / +0.09 |
| d1 OP0_red | +0.38 / −0.26 / −0.45 | +0.54 / +0.09 / +0.66 |

1. **The ink's kinetics are RH kinetics.** d1 of the ink tracks dRH/dt at
   |r| up to 1.00 in *every* block. Whatever the model learns from
   derivatives is humidity dynamics, which is then linearly mapped onto the
   train block's incidental RH↔CO₂ co-movement.
2. **The ink↔dCO₂ derivative relationship flips sign between train and val**
   (e.g. px_B_RGB: −0.55 train, +0.16 val, −0.61 test) — the *same*
   anti-generalization pathology documented for the static features in
   `L4_TUNING_NOTES.md`. Differentiation does not remove the confounder; it
   differentiates it.
3. **Recalibration can't rescue derivative models**: a per-segment offset
   fixes a constant bias, but derivative features make the *shape* of the
   prediction wrong (spiky, RH-driven), so post-offset error grows rather
   than shrinks (recal R² as low as −77).
4. The lead-compensation result is the cleanest falsification: if the
   3–44 min lag were the binding constraint, some τ > 0 had to help. None
   did, at either evaluation, on a metric (val) that contains real CO₂
   ramps. The bottleneck is not response lag — it is that the equilibrium
   ink signal itself carries almost no block-stable CO₂ information beyond
   what RH explains (cf. tiny train→test transfer in OP0_red d1 that val
   selection cannot see).

## Conclusion for the thesis

Kinetic/derivative features of the ink are a falsified branch on this
dataset: worse than static features without recalibration, worse with the
standard recalibration, and the explicit first-order-lag inversion is
rejected by validation at every time constant. This is consistent with — and
explains — ladder 5's dead-reckoning failure: the only usable dynamics in
the ink are humidity dynamics.
