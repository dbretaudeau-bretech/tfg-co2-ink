# IDEA 3 — Modern sequence architectures (PatchTST) vs the champion LSTM

**Question.** Does a PatchTST-style time-series transformer beat the champion
coarse-context LSTM (16 raw ink channels → 60-s MA → 1/min resample → 360-step
6-h window → LSTM(64) → linear (RH, CO₂)) under the standard per-lamp-segment
30-min offset recalibration (`run_ladder2.py::r2()`)?

**Champion reference (3 seeds): test CO₂ MAE 47.7 ± 1.2 ppm, R² 0.68
(recal, lamp-stable windows); no-recal shape r 0.81 ± 0.03.**

**Answer up front: no. PatchTST final (3 seeds): recal MAE 103.9 ± 46.8,
R² −0.67 ± 1.54, r 0.36 ± 0.48; 3-seed prediction ensemble 84.7 / 0.06 /
0.65. No-recal full-test r is −0.38 (ensemble). Attention loses to recurrence
on every metric, by a wide margin.** Mamba was **skipped**: `mamba-ssm` is not
installed in the TFGV2 venv (torch 2.12 nightly cu128 would require a source
build); only the transformer arm was run.

## 1. What was held identical to the champion

Everything except the sequence encoder: same 16 raw channels, same 60-s moving
average, same 1/min coarse grid, same 360×16 windows (endpoints every 30 s for
training, every 5 s for val/test; windows may extend left across split
boundaries — inputs only, the documented liberty of ladder 4/5), same
standardized joint (RH, CO₂) targets, Adam 1e-3, MSE, and the *verbatim*
recalibration protocol (my `protocol_eval` reproduces the stored champion
used-mask bit-for-bit; champion best seed re-checked from
`preds6_lstm_recal_best.npz`: 46.69 / 0.687 / 0.830).

## 2. Model

Channel-independent PatchTST (`run_idea3.py::PatchTST`): each of the 16
channels is split into non-overlapping patches, linearly embedded
(patch_len → d_model, weights shared across channels), given a learnable
positional encoding, passed through pre-norm transformer encoder layers
(GELU, ff = 2·d_model), LayerNorm + mean-pooled over patches; the 16 channel
embeddings are concatenated and a single linear head outputs (RH, CO₂).

## 3. Selection (validation only; test untouched until frozen)

* **Grid** (seed 0): patch {12, 24, 30} × depth {2, 3} × d_model {64, 128} ×
  dropout {0.1, 0.2} — 8 star configs. Early stop on val loss (standardized
  joint MSE), patience 8.
* **Honesty note, as agreed in the brief:** val CO₂ ranking is weak on this
  dataset (the ink→CO₂ correlation flips sign between chronological blocks —
  see `L4_TUNING_NOTES.md`), so ranking used val loss plus the val
  recalibration-protocol score. A 40-epoch instrumented run confirmed val
  loss/val RH genuinely peak at epoch ≈ 2 and then plateau — early stopping
  was not truncating a still-improving model.
* Two finalists re-run at 3 seeds: P1 (p24/d128/L3) val loss 1.349 ± 0.166,
  val recal MAE 109.6 ± 19.8, recal r 0.586; **P4 (p24/d64/L3)** val loss
  **1.278 ± 0.015**, val recal MAE 110.9 ± 20.9, recal r **0.654** → frozen.
  Test was then read once, from the already-stored predictions of exactly
  those three runs (no retrain).
* Final config: patch 24 (15 patches), d_model 64, 4 heads, 3 layers,
  dropout 0.1, ~107 k params.

## 4. Test results (frozen winner, 3 seeds)

| metric | PatchTST (mean ± std) | ensemble(3) | champion LSTM |
|---|---|---|---|
| CO₂ recal MAE (ppm) | **103.9 ± 46.8** | 84.7 | **47.7 ± 1.2** |
| CO₂ recal R² | −0.67 ± 1.54 | 0.06 | 0.68 |
| CO₂ recal r | 0.36 ± 0.48 | 0.65 | ~0.83 |
| CO₂ no-recal full-test r (shape) | −0.24 ± 0.32 | −0.38 | 0.81 ± 0.03 |
| RH full-test r | −0.36 ± 0.06 | −0.39 | ~0.88 (LSTM family) |

Per-seed recal MAE: 74.0 / 67.6 / 169.9 — the familiar seed lottery, but even
the luckiest seed (67.6) is 20 ppm above the champion's worst.

## 5. Why it loses (post-hoc reading, no further selection)

1. **The transformer doesn't even learn RH on test** (r ≈ −0.36 vs the LSTM
   family's robust +0.88). With channel-independent patching and mean-pooled
   attention, the model appears to latch onto train-block drift statistics
   that anti-generalize across the chronological split; the LSTM's recurrent
   integration over the smoothed window is evidently a better inductive bias
   for this slow, drifting, 16-channel reflectance signal.
2. **Validation cannot rescue it**: the best achievable val recal MAE across
   the whole grid was ~83–130 vs ~47–90 for the LSTM family in ladder 5 —
   the gap is visible on val too, so this is not a selection accident.
3. Capacity is not the issue: d_model 64 beat 128, depth 3 ≈ 2, and the
   ~107 k-param winner still overfits within 1–2 epochs.

## 6. Verdict (3 lines)

Attention does **not** beat recurrence here: PatchTST ends at 103.9 ± 46.8 ppm
recal MAE (R² −0.67) vs the LSTM's 47.7 ± 1.2 (R² 0.68), and even its 3-seed
ensemble (84.7) and luckiest seed (67.6) stay far behind; the no-recal shape
correlation collapses from 0.81 to ≈ −0.4. The champion stands; Mamba untested
(library not installed).

Code: `run_idea3.py` (stages `grid`, `seeds`, `final`); all runs in
`results/idea3_transformer.json` (SUMMARY key holds the headline numbers);
raw per-seed predictions in `results/raw_idea3/`; final recalibrated ensemble
test predictions in `results/preds_idea3_patchtst_final.npz`
(pred = recalibrated, true, t_h, used, raw).
