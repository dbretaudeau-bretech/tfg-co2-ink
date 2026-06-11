# L4 coarse-context LSTM — tuning notes (2026-06-10)

Goal: squeeze the honest best out of the L4 pipeline (ink MA -> resample ->
LSTM -> (RH, CO2) at window end). Primary selection metric: **validation CO2
Pearson r**. Test was computed for every run but only looked at once, for the
frozen final config. Code: `run_tune4.py`; all runs in
`metrics4_tuning.json`; staged summaries in `tuning_summary.json` (stages 1-4
+ first final) and `tuning_summary_stage5.json` (3-seed stability round =
the reported result).

## Protocol changes vs run_ladder4.py (both honest, documented)
- Val/test input windows may extend LEFT across the split boundary
  (inputs = ink reflectance only; the predicted target always stays inside
  its split). This puts every context length on the IDENTICAL val/test
  endpoint grid (every 60 s, 687 points each) so contexts compete fairly and
  the full 11.44 h val/test blocks are used.
- Early stopping on val CO2 Pearson (patience 6, max 60 epochs, best-epoch
  checkpoint restore). Validation-only, like all other selection decisions.

## Search path (mean val CO2 r over 2 seeds unless noted)
1. **Context** (ma60/st60, h64, 1 layer): 2h −0.46, 4h −0.47, 6h −0.43,
   8h −0.47. All NEGATIVE. Picked 6h.
2. **MA × stride**: ma{30,60,120}s × st{30,60}s all within −0.43…−0.45.
   Irrelevant knob. Kept ma60/st60.
3. **Hidden**: 32 −0.35, 64 −0.43, 128 −0.45. Smaller is better.
   **Layers × dropout**: 1-layer all ≈ −0.35; **2-layer flips to +0.47**.
   Weight decay 0 vs 1e-4: no effect.
4. **Head/channels**: joint(RH+CO2) 16ch +0.47; CO2-only −0.28;
   +B/R 17th channel −0.13 (worse). Joint 16ch wins clearly.
5. **Stability round, 3 seeds each** (the honest re-check): the +0.47 of the
   2-layer h32 model is seed-dependent — seeds give [0.52, 0.42, −0.31].
   h64-L2 and h128-L2 are negative for ALL seeds. Context 4/6/8 h
   indistinguishable for h32-L2. Dropout 0/0.1/0.2 indistinguishable.

## Final config (best mean val r = 0.215 ± 0.37, 3 seeds)
ctx 6 h, MA 60 s, stride 60 s, LSTM(32)×2 layers, dropout 0.2, wd 0,
joint (RH, CO2) head, 16 raw ink channels, Adam 1e-3, batch 512, ES on val r.

### Test metrics (mean ± std over seeds 0/1/2) — looked at once, after freezing
| target | MAE | R² | Pearson r |
|---|---|---|---|
| CO2 | 263 ± 114 ppm | −0.87 ± 1.25 | **0.19 ± 0.05** |
| RH | 2.79 ± 0.46 % | 0.45 ± 0.21 | 0.88 ± 0.02 |

Val (same seeds): CO2 r 0.215 ± 0.367, MAE 276 ± 49, R² −2.9.
Best-val seed (0) predictions saved to `preds4_L4_best.npz` (687×2, RH|CO2).

## Findings (the story for the thesis)
1. **The old "test CO2 r = 0.86" does not survive an honest protocol.** That
   number came from a no-validation run reported directly on test (with
   R² = −3.6 and train r = 0.998, i.e. memorization + co-drifting bias).
   Under val-only selection the same pipeline lands at test r ≈ 0.19.
2. **The ink → CO2 relationship flips sign between chronological blocks**:
   corr(OP0_red, CO2) = +0.21 train / −0.76 val / +0.23 test;
   corr(px_BR_mean, CO2) = +0.04 / +0.54 / +0.04. Meanwhile ink↔RH is
   r ≈ −0.97 in EVERY block. The ink genuinely encodes RH; whatever CO2
   signal a model finds in train is regime-specific and anti-generalizes to
   val (hence the wall of negative val r in stages 1-3a). This matches the
   earlier ladders: ridge/kNN/RF/MLP all show val CO2 r between −0.65 and
   +0.69 with no model achieving positive R².
3. **Seed lottery dominates hyperparameters.** Every architecture knob
   (context 2-8 h, MA/stride, dropout, wd) moves val r by <0.05, but the
   random seed moves it by 0.8 (−0.31 ↔ +0.52) for the 2-layer h32 model.
   The "2-layer jump" found in stage 3b is a lucky-basin effect, not a
   capacity effect — h64/h128 2-layer models are negative for all seeds.
4. RH is the real signal: test RH r ≈ 0.88 robustly, every config, every
   seed. Honest conclusion for the chapter: the coarse-context LSTM (and
   every other model family tried in this dataset) reads humidity from the
   ink; it cannot read CO2 at better than r ≈ 0.2 once selection is honest.
