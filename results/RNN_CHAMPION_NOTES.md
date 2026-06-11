# Recurrent models under the recalibration protocol — full record (ladder 5)

**Goal.** Beat the current champion — the 3-parameter physical model
(pixel-mean B/R + measured RH → CO₂, ordinary least squares on the train
block) with a 30-min single-offset recalibration at each lamp change —
**honestly**, with a recurrent network.

**Champion reference (test, evaluated windows): MAE 53.65 ppm, R² 0.645, r 0.822.**

**Outcome up front: the champion stands.** The best recurrent model reaches
**MAE 60.78 ppm, R² 0.546, r 0.757** on the identical evaluated windows
(full-test r = 0.875, above the previous RNN best of 0.86 and above the
champion's 0.858). Two test evaluations were performed in total; both are
reported below. Everything else was decided on validation.

---

## 1. Evaluation protocol (identical to the champion's)

Reference implementation: `run_ladder2.py::r2()`. Reproduced verbatim in
`run_ladder5.py::protocol_eval()`:

1. Lamp segments = maximal runs of constant `round(lamp_pct)` over the full
   record; segment ids taken on the test (or validation) block.
2. In each segment, the first `min(len, 360)` samples (30 min at 5 s) are the
   calibration window: a single offset `mean(true − pred)` is added to the
   whole segment's prediction.
3. Metrics are computed only on post-calibration samples (`used` mask).
   Segments shorter than 30 min are never evaluated. On test this leaves
   34.4 % of samples (n = 2838, CO₂ 283–623 ppm) in 5 lamp-stable windows
   (0.7 h, 3×1 h, 2.75 h). On validation it leaves 54.1 %.
4. Splits: `data4.split_indices` (chronological 70/15/15, 4-min gaps).
   Nothing is ever fitted on test except the per-segment offset above.

One documented liberty (same as `run_tune4.py`): input windows for val/test
endpoints may extend left across the split boundary. Inputs are reflectance +
the device's own RH/T/lamp sensors only — no targets — and a deployed device
has its own continuous history, so this is not leakage. Measured `BME_RH` is a
legitimate input because the champion itself consumes it.

Champion's validation-protocol score (selection reference): **MAE 88.4,
r 0.581**. (Validation is much harder than test for every model: its
evaluated windows contain large CO₂ ramps, while the test windows are mostly
slow/flat. This asymmetry decided the outcome — see §5.)

## 2. Shared RNN pipeline

* Channel bank: 16 raw `OP{0–3}_{red,green,blue,ir}`; "devreal" adds
  `lamp_pct, BME_RH, BME_T` (19 ch); optional engineered ratios
  (`px_BR_mean, px_IR_R_mean, px_B_RGB_mean, px_IR_RGB_mean`) and the
  physics prediction as a 24th channel.
* 60-s moving average → windows sampled every 60 s (`stride 12` on the 5-s
  grid), `ctx_h × 60` steps per window → LSTM → linear head on last state.
* StandardScaler (train block) on inputs and targets. Adam lr 1e-3,
  batch 512, MSE (variants: Huber). Train endpoints every 30 s
  ("dense" = every 15 s). Early stopping (patience 8→15, max 80 epochs) on
  **validation protocol MAE** — the same number used for selection.
* Predictions are produced at every 5-s test/val sample so the recalibration
  protocol operates on exactly the same grid as the champion's.

## 3. Approaches tried (all selected on validation protocol MAE)

Stage 1 — modes, 6 h context, 1 seed (`metrics5_rnn.json`, prefixes A–E):

| candidate | val MAE | verdict |
|---|---|---|
| champion physics (reference) | 88.4 | — |
| abs CO₂+RH joint, raw16 | 86.9 | marginal |
| abs joint, devreal (19 ch) | 79.6 | kept as family |
| abs CO₂-only, devreal | 112.5 | discarded (joint head regularises) |
| abs joint, devreal+ratios | 95.2 | discarded |
| abs CO₂, ratios+RH only | 70.1 | decent, not pursued (seed-lucky) |
| residual of recal'd physics (3 variants) | 132–140 | **discarded** — residual target is noisier than CO₂ itself |
| **derivative + integration (dead reckoning), devreal** | **61.4** | **winner family** |
| derivative, raw16 (no RH) | 97.4 | RH channel essential |
| anchor-as-input (t-since-recal, CO₂-at-recal), devreal | 113.1 | discarded |
| GRU / Huber variants of abs joint | 121.1 / 80.1 | discarded |

Stage 2/3 — derivative-mode sweep (prefixes C, F): context 2 h ≫ 6 h
(47.1 vs 61.4); derivative target = gradient of 2-min-smoothed CO₂ (dsm120)
and denser train windows help; hidden 64 > 32/96/128; 2-layer, GRU hurt.
Best single config `F10`: **deriv, devreal 19 ch, ctx 2 h, LSTM(64), dsm120,
train stride 15 s**.

Stage 4/5 — seeds. Seed variance is brutal (val MAE 47–117 across 10 seeds of
the same config!). Prediction-averaging ensembles fix most of it:
first-5-seed ensemble 55.4; **top-5-seeds-by-val ensemble 47.7 / r 0.935**
(seed membership selected on validation — disclosed). Physics blending
(w·phys + (1−w)·RNN) only hurt on validation for every w.

Also tried and discarded on val: displacement-from-anchor mode (`danchor`,
val 109–158), abs-mode 5-seed ensembles (97), `+phys` blends of everything.

## 4. The two test evaluations (both disclosed)

**Shot 1 — frozen val winner.** Deriv ensemble (top-5 seeds), pure
integration: cumulative sum of predicted dCO₂/dt, constant absorbed by the
per-segment offset. Val 47.7/0.869/0.935 → **test 97.1 / −0.154 / 0.714.
Lost.** Post-mortem (error analysis only, no further selection on test):
the integrator *invents motion in flat segments* — e.g. the 1-h segment where
CO₂ moves 16 ppm got 88 ppm of accumulated drift (a ~0.025 ppm/s derivative
bias × 3600 s). Validation never punished this because its evaluated windows
are dynamics-rich, where dead reckoning shines (RNN val error at 90–120 min
after recal: 78 ppm vs physics' 409 ppm).

**Shot 2 — bounded integration (final).** Amendment motivated by the above,
with λ chosen on validation by a rule stated before evaluation: leak the
integral toward the physics baseline,

  p(t) = phys(t) + I(t),  I(t) = (1−λ)·I(t−1) + [Δp̂_RNN(t) − Δphys(t)],

λ = largest value whose val MAE still beats the champion's val score by ≥20 %
(≤ 70.7) → λ = 4·10⁻³ per 5-s step (time constant ≈ 21 min), val 70.4.
As λ→∞ this reduces exactly to the champion; as λ→0 to shot 1.

**Final test (same evaluated windows): MAE 60.78, R² 0.546, r 0.757;
full-test r 0.875.** Champion: 53.65 / 0.645 / 0.822. **Champion stands.**

Per-segment (RNN vs physics, used samples): 0.7 h flat 15.5 vs 23.4 (RNN
wins); 1 h flat 7.6 vs 6.2 and 29.2 vs 21.8 (ties/slight loss); 1 h dynamic
71.0 vs 39.2 (loss); 2.75 h 81.3 vs 77.1 (slight loss). The margin lives in
the dynamic 1-h segment, where the leak (needed for the flat segments)
removed exactly the low-frequency tracking the RNN was good at.

## 5. Final recipe (reproducible)

1. Inputs: 19 channels = 16 raw OnePixel reflectances + lamp_pct + BME_RH +
   BME_T; 60-s moving average; standardised on train.
2. Windows: 120 steps × 60 s (2 h context), endpoint every 15 s for training,
   every 5 s for evaluation; target = d/dt of 2-min-smoothed SCD_CO2
   (np.gradient, ppm/s), standardised.
3. Model: LSTM(19→64, 1 layer) + linear head on last state; Adam 1e-3,
   batch 512, MSE; early stop on validation-protocol MAE (patience 15).
4. Train 10 seeds; keep the 5 best by validation-protocol MAE; average their
   predicted derivative series.
5. Integrate with leak λ=4e-3 per 5-s step toward the train-fitted B/R+RH
   physics baseline (formula in §4); apply the standard 30-min per-segment
   offset recalibration; evaluate on post-calibration samples.

Code: `run_ladder5.py` (stages s1–s6) + `analyze5.py`; all runs in
`results/metrics5_rnn.json`; raw per-seed predictions in `results/raw5/`;
final test predictions in `results/preds5_rnn_champion.npz`
(pred = recalibrated, true, t_h, used_mask).

## 6. Findings

1. **Dead reckoning is the only RNN formulation that decisively beats the
   physical model on validation** (47.7 vs 88.4 ppm MAE): predicting dCO₂/dt
   and integrating from the recalibration anchor converts baseline drift —
   the failure mode of every absolute-output network — into a slowly
   accumulating integration error. At 90–120 min after recalibration its
   validation error is 5× lower than the recalibrated physical model's.
2. **But integration bias is regime-dependent and validation could not see
   it**: a ~0.025 ppm/s derivative bias is invisible in dynamics-rich windows
   and fatal in flat ones. The chronological val block (large CO₂ ramps)
   rewarded tracking; the test block (mostly lamp-stable, 283–623 ppm,
   slow) rewarded staying put. Honest val-only selection therefore picked the
   wrong model for the test regime — the central methodological lesson of
   this round, and an argument for protocol-faithful validation design
   (validation windows should match deployment dynamics, not just deployment
   metrics).
3. **Seed variance dwarfs architecture variance** for small LSTMs on this
   data (val MAE 47–117 across seeds of one config, vs ±10 between
   reasonable configs). Nothing single-seed should ever be believed here;
   5-seed prediction averaging was worth ~20 ppm.
4. The final bounded-integration hybrid is a clean one-parameter family
   spanning champion (λ→∞) to dead reckoning (λ→0). It beats the champion in
   flat segments shortly after recalibration and achieves the best full-test
   correlation of any model to date (r 0.875), but loses the headline MAE
   because the leak sacrifices the low-frequency tracking in the one dynamic
   evaluated test segment. With longer recalibration-free stretches in the
   evaluated windows (where physics degrades to ~400 ppm), the ranking would
   invert — the champion wins this dataset's test, not the deployment
   scenario in general.
