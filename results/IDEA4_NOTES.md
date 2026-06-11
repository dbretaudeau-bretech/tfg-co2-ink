# Adversarial humidity decoupling — can the ink's CO2 signal be made RH-invariant?

**Question.** The ink's reflectance is dominated by humidity (corr(reflectance,
RH) ≈ −0.97 in every chronological block; the CO2 signal is weak and
regime-specific). If a CO2 model leans on the RH-correlated component, it
should anti-generalize. So: can we force a *humidity-invariant* representation
that still predicts CO2 — i.e. surgically remove RH from the features while
keeping CO2?

**Physics caution stated up front.** In this ink CO2 and RH act through the
*same* protonation pathway, so the reflectance directions that respond to CO2
are largely the directions that respond to RH. The risk is that any procedure
strong enough to remove RH also removes CO2. This note tests exactly that.

## Setup

* Inputs are **reflectance only** by construction (no measured RH, no lamp) —
  decoupling from RH only makes sense if RH is not also an input. Two
  extractors: an **MLP** on the 72 static light-FE features, and an **LSTM(64)**
  on coarse 6-h windows (60-s moving average, 60-s stride, 6 h context — the
  `run_ladder4.py` pipeline).
* Two heads on the shared representation: a **CO2 head** (SCD_CO2, MSE,
  trained normally) and an **RH head** (BME_RH, MSE) behind a **Gradient
  Reversal Layer**. Forward is identity; backward returns `−λ·grad`, so the
  extractor is pushed to make features the RH head *cannot* read. Total loss
  `MSE(CO2) + MSE(RH)`. Sweep `λ ∈ {0, 0.01, 0.1, 0.5, 1, 2, 5}` (0 = plain
  multitask, no reversal; the high end added to push toward full invariance),
  3 seeds.
* **Targets standardised on train; chronological 70/15/15 splits
  (`data4.split_indices`, 4-min gaps).** All model selection on the
  **validation** CO2 score under the champion recal protocol (`run_ladder2.r2`
  / per-lamp-segment 30-min single-offset recal, calibration samples excluded).
  Val's CO2 ranking is weak on this dataset (large ramps in val vs slow test
  windows) — noted and respected: nothing is chosen on test.
* Reported per run: CO2 **no-recal** (full test) and **with-recal** (used
  windows), plus two RH-invariance measures —
  1. **online RH-head R²** (what the adversary itself achieves), and
  2. **fresh-probe RH R²**: a brand-new ridge probe trained on the *frozen*
     representation. This is the honest invariance measure; the online head can
     be defeated without the representation actually losing RH.

## Reference points (test)

| model | uses RH input | recal MAE | recal R² | recal r | no-recal MAE | no-recal r |
|---|---|---|---|---|---|---|
| champion physics (B/R + measured RH, OLS) | **yes** | **53.65** | **0.645** | **0.822** | 171 | 0.238 |
| reflectance ridge-72 (no RH) | no | 122.0 | −0.84 | 0.845 | 403 | 0.761 |

## Result 1 — the GRL defeats the online head but achieves no real invariance

Honest val-selected runs, mean over 3 seeds, **LSTM 6-h, reflectance-only**:

| λ | online RH-head R² | **fresh-probe RH R²** | CO2 recal r | CO2 recal MAE |
|---|---|---|---|---|
| 0 | 0.22 | **0.93** | 0.83 | 53.8 |
| 0.1 | 0.14 | **0.92** | 0.83 | 53.8 |
| 0.5 | 0.05 | **0.93** | 0.83 | 53.4 |
| 1 | 0.02 | **0.93** | 0.83 | 53.3 |
| 2 | 0.00 | **0.92** | 0.82 | 54.0 |
| 5 | −0.00 | **0.93** | 0.80 | 54.9 |

The online RH head collapses to R² ≈ 0 as λ grows — *looks* like RH was
removed. But a fresh probe recovers RH at **R² ≈ 0.93 at every λ**: the network
only rotates RH into a subspace the moving adversary stops tracking; the
information is still fully there. Consequently CO2 is essentially unchanged.
The MLP behaves the same on the probe axis (RH probe R² 0.86–0.95 across λ),
while its online head is driven to large *negative* R² (the adversary is
pushed to anti-predict) and its absolute CO2 scale destabilises (recal MAE
89 → 143 as λ → 2).

Under honest validation selection the models early-stop at ≈ epoch 0 — i.e.
**adversarial training only *hurts* the val CO2 score**, so selection keeps the
near-pre-adversarial checkpoint. The best LSTM is λ = 0; the adversary never
helps.

## Result 2 — forcing real invariance kills CO2 (physics caution confirmed)

To make the adversary actually reshape the representation I trained a fixed
number of epochs (no early-stop restore) over an extended grid `λ ∈ {0, 0.5, 2,
5, 10, 20, 50}` and read the fresh-probe RH R² and CO2 recoverability
(`co2_probe_r`, a fresh ridge CO2 probe on the frozen features), 2 seeds:

LSTM 6-h: at the λ where the probe finally dips (λ = 5, RH probe R² 0.89 → 0.67)
the CO2 recoverability **collapses with it** (co2_probe_r 0.30 → 0.18). MLP: the
λ = 10 dip (RH probe 0.84 → 0.60) drops co2_probe_r 0.50 → 0.46. RH probe R²
otherwise springs back to ~0.9 — the dominant, redundant RH variance is almost
unkillable, and the only times it moves, CO2 moves the same way. There is **no
λ where RH-invariance increases and CO2 holds.**

## Result 3 — orthogonal projection (the linear, mechanistic version)

Remove the top-k RH-predictive directions from the 72-dim reflectance space by
OLS deflation, refit a CO2 ridge, sweep k = 0…12:

| k removed | fresh RH residual R² (test) | CO2 recal r | CO2 recal MAE |
|---|---|---|---|
| 0 | 0.973 | 0.873 | 114.5 |
| 4 | 0.983 | 0.873 | 114.5 |
| 8 | 0.951 | 0.873 | 114.5 |
| 12 | 0.922 | 0.873 | 114.5 |

RH stays recoverable at **R² ≈ 0.92–0.98 no matter how many directions are
stripped** — RH is encoded *redundantly* across the collinear reflectance
channels, so a fresh OLS immediately re-finds it in whatever subspace remains —
and the CO2 prediction is unchanged. Linear RH-subspace removal neither
achieves RH-invariance nor changes CO2, for the same reason the GRL fails: the
two signals are not linearly separable in this ink.

## Best CO2 numbers from this idea (test)

* **With-recal (used windows):** LSTM 6-h, reflectance-only, λ = 0 →
  **MAE 53.8 ± (seed) / R² 0.65 / r 0.83**, which *ties the champion*
  (53.65 / 0.645 / 0.822) **without using measured RH as an input** — because
  the LSTM reconstructs the RH proxy internally from the reflectance windows.
  The adversary moves this monotonically the wrong way (→ 54.9 / 0.60 / 0.80 at
  λ = 5).
* **No-recal (full test):** the adversarial nets are poor (LSTM MAE ≈ 170,
  full-test r ≈ 0.37 — they compress the dynamic range and cannot track the big
  ramps); the better reflectance-only no-recal correlation is the linear
  ridge-72 (r 0.76, MAE 403), still far short of needing recal.

## Verdict (3 lines)

1. **No — RH cannot be adversarially removed from this ink's reflectance
   without killing CO2.** The two share the protonation pathway and the same
   reflectance directions.
2. The GRL only *hides* RH from the online adversary (head R² → 0) while a
   fresh probe keeps recovering it at R² ≈ 0.9; orthogonal subspace removal
   fails identically (RH redundant across collinear channels). True invariance
   is never reached.
3. Whenever invariance *is* forced, CO2 recoverability collapses in lockstep;
   under honest val-selection the adversary never helps, and a plain
   reflectance-only LSTM already ties the RH-using champion via recal — so
   adversarial decoupling here is both impossible and counterproductive.

---
*Code: `run_idea4.py` (stages `refs`, `mlp`, `lstm`, `curve`, `ortho`,
`ortho_raw`). Metrics + SUMMARY in `results/idea4_adversarial.json`.*
