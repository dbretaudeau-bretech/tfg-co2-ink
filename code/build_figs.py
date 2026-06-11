"""Memoria figures — revtex two-column sized (3.37 in column width).

fig1_protocol   — 77 h run overview: CO2 setpoint vs SCD30, RH, lamp; split shading
fig2_calibration — B/R within a lamp-stable segment + per-segment calibration scatter
fig3_predictions — test-set predictions: RH (ridge hp8h) and CO2 (physics + recal vs static ridge)
Outputs PDF (for LaTeX) + PNG (for review) into figs/.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import data4 as D

FIGS = Path("/home/bretech/OTHERS/TFG_MEMORIA/figs")
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 0.8, "axes.linewidth": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
COL_W = 3.37  # inches

df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
idx = D.split_indices(df)
t = df["t_h"].values
t_test0 = t[idx["test"]][0]
t_val0 = t[idx["val"]][0]

lamp = df["lamp_pct"].ffill().bfill()
lampr = lamp.round(0).values
bounds = [0] + list(np.where(np.diff(lampr) != 0)[0] + 1) + [len(lampr)]
seg_id = np.zeros(len(lampr), int)
for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
    seg_id[a:b] = s

# ------------------------------------------------------------------ FIG 1
fig, axes = plt.subplots(3, 1, figsize=(COL_W, 3.2), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2, 1.2], "hspace": 0.12})
ax = axes[0]
ax.plot(t, df["co2_set_ppm"], color="0.65", lw=0.6, label="setpoint")
ax.plot(t, df["SCD_CO2"], color="C3", lw=0.7, label="measured (SCD30)")
ax.set_ylabel("CO$_2$ (ppm)")
ax.legend(loc="upper left", frameon=False, ncol=1, handlelength=1.2,
          fontsize=5.8, borderaxespad=0.1)
ax = axes[1]
ax.plot(t, df["BME_RH"], color="C0", lw=0.7)
ax.set_ylabel("RH (%)")
ax = axes[2]
ax.step(t, lamp, where="post", color="C5", lw=0.8)
ax.set_ylabel("lamp (%)")
ax.set_xlabel("time (h)")
ax.set_yticks([20, 50, 80])
for ax in axes:
    ax.axvspan(t_val0, t_test0, color="0.9", zorder=0)
    ax.axvspan(t_test0, t[-1], color="gold", alpha=0.15, zorder=0)
axes[0].text(t_val0 + 0.2, 1850, "val", fontsize=6, color="0.4")
axes[0].text(t_test0 + 0.2, 1850, "test", fontsize=6, color="0.4")
fig.savefig(FIGS / "fig1_protocol.pdf"); fig.savefig(FIGS / "fig1_protocol.png")
plt.close(fig)

# ------------------------------------------------------------------ FIG 2
br = pd.concat([df[f"OP{i}_blue"] / df[f"OP{i}_red"] for i in range(4)], axis=1).mean(axis=1)
scd = df["SCD_CO2"].values
rh = df["BME_RH"].values

# stable segments (>2h), skip first hour
segs = []
for s in np.unique(seg_id):
    rows = np.where(seg_id == s)[0]
    if len(rows) < 2 * 720:
        continue
    rows = rows[720:]  # skip 1 h settling
    segs.append((rows, lampr[rows[0]]))

fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75), gridspec_kw={"wspace": 1.0})
# (a) time series inside the 50% lamp segment
rows50 = next(r for r, L in segs if L == 50)
ax = axes[0]
ax.plot(t[rows50], br.iloc[rows50], color="C0", lw=0.7)
ax.set_xlabel("time (h)"); ax.set_ylabel("B/R", color="C0", labelpad=1)
ax.tick_params(axis="y", colors="C0")
ax.ticklabel_format(axis="y", useOffset=False)
ax2 = ax.twinx()
ax2.plot(t[rows50], scd[rows50], color="C3", lw=0.7)
ax2.set_ylabel("CO$_2$ (ppm)", color="C3", labelpad=2)
ax2.tick_params(axis="y", colors="C3")
ax.set_title("(a) lamp 50 %", loc="left")
# (b) calibration: RH-corrected B/R vs CO2, joint-fit slopes per segment
ax = axes[1]
styles = {(60, 0): ("C2", "60 %"), (50, 0): ("C1", "50 %"),
          (40, 0): ("C0", "40 %"), (50, 1): ("C3", "50 % (40 h later)")}
seen = {}
for rows, L in segs:
    k = (L, seen.get(L, 0))
    if k not in styles:
        seen[L] = seen.get(L, 0) + 1
        continue
    seen[L] = seen.get(L, 0) + 1
    c, lab = styles[k]
    x, yv, z = scd[rows], br.values[rows], rh[rows]
    A = np.column_stack([x, z, np.ones(len(rows))])
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    yc = yv - coef[1] * z          # remove fitted RH contribution
    yc = yc - yc.mean()
    ax.plot(x[::25], yc[::25] * 1e3, ".", ms=0.8, alpha=0.18, color=c)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, coef[0] * (xx - x.mean()) * 1e3, color=c, lw=1.1,
            label=f"{lab}: {coef[0]*1e6:+.1f}")
ax.set_xlabel("CO$_2$ (ppm)")
ax.set_ylabel("B/R, RH-corr. ($\\times$10$^{-3}$)", labelpad=1)
ax.legend(frameon=False, loc="upper right", handlelength=1.0, fontsize=5.0,
          title="slope ($\\times$10$^{-6}$/ppm)", title_fontsize=5.0)
ax.set_title("(b)", loc="left")
fig.savefig(FIGS / "fig2_calibration.pdf"); fig.savefig(FIGS / "fig2_calibration.png")
plt.close(fig)

# ------------------------------------------------------------------ FIG 3
res = Path("/home/bretech/OTHERS/TFGV4/results")
rh_npz = np.load(res / "preds_t3_ridge_hp8h.npz")
co2_phys = np.load(res / "preds_r2_BR_RH_linear_segrecal.npz")
co2_static = np.load(res / "preds_t2_ridge72_static.npz")

fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.9), gridspec_kw={"hspace": 0.45})
ax = axes[0]
ax.plot(rh_npz["t_h"], rh_npz["true"][:, 0], color="0.3", lw=0.8, label="measured")
ax.plot(rh_npz["t_h"], rh_npz["pred"][:, 0], color="C0", lw=0.7, alpha=0.85, label="predicted")
ax.set_ylabel("RH (%)"); ax.set_title("(a) RH, ink only — MAE 0.35 %, $R^2$ 0.993", loc="left")
ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2)
ax = axes[1]
ax.plot(co2_phys["t_h"], co2_phys["true"], color="0.3", lw=0.8, label="measured")
ax.plot(co2_static["t_h"], co2_static["pred"][:, 1], color="0.6", lw=0.5, alpha=0.6,
        label="Ridge, no recal.")
# physics model: only draw evaluated (post-recalibration) windows
pm = np.load(res / "preds_r2_BR_RH_linear_segrecal.npz")
tt, pp, true = pm["t_h"], pm["pred"], pm["true"]
# reconstruct evaluated mask: lamp segments on the test rows, skip 30-min cal
lamp_te = lamp.values[idx["test"]]
lr = np.round(lamp_te)
bd = [0] + list(np.where(np.diff(lr) != 0)[0] + 1) + [len(lr)]
CAL = int(30 * 60 / 5)
shown = np.zeros(len(tt), bool)
for a, b in zip(bd[:-1], bd[1:]):
    if b - a > CAL:
        shown[a + CAL:b] = True
pp_m = np.where(shown, pp, np.nan)
ax.plot(tt, pp_m, color="C3", lw=0.9, label="B/R+RH + recal.")
ax.set_ylabel("CO$_2$ (ppm)"); ax.set_xlabel("time (h)")
ax.set_title("(b) CO$_2$ — physical model MAE 54 ppm, $R^2$ 0.65", loc="left")
ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.2, fontsize=5.5)
ax.set_ylim(-100, 1900)
fig.savefig(FIGS / "fig3_predictions.pdf"); fig.savefig(FIGS / "fig3_predictions.png")
plt.close(fig)
print("figs written to", FIGS)
