"""Provenance verification C — offset correction, train+val refit, indicator
mode, appendix numbers. Recomputes everything deterministic; checks artifacts
for stochastic models. Read-only w.r.t. existing results (writes nothing).
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import data4 as D

RES = D.RESULTS
OUT = {}


def log(k, v):
    OUT[k] = v
    print(f"== {k}: {json.dumps(v)}", flush=True)


# ---------------------------------------------------------------- data
df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
X72, y, t_h, idx = D.load_light_fe()
aux = df[["lamp_pct", "BME_RH", "BME_T"]].reset_index(drop=True).ffill().bfill()
yarr = np.asarray(y, dtype=np.float64)
scd = yarr[:, 1]
yd = {k: yarr[v] for k, v in idx.items()}


def _segments(lamp):
    lampr = lamp.round(0).values
    bounds = [0] + list(np.where(np.diff(lampr) != 0)[0] + 1) + [len(lampr)]
    seg_id = np.zeros(len(lampr), dtype=int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg_id[a:b] = s
    return seg_id


seg_all = _segments(aux["lamp_pct"])
CAL30 = int(30 * 60 / 5)


def protocol(pred_split, split, cal=CAL30, with_extras=False):
    sel = idx[split]
    true = scd[sel]
    seg_s = seg_all[sel]
    pc = pred_split.copy()
    used = np.zeros(len(sel), bool)
    for s in np.unique(seg_s):
        rows = np.where(seg_s == s)[0]
        c = rows[:min(len(rows), cal)]
        off = (true[c] - pred_split[c]).mean()
        pc[rows] = pred_split[rows] + off
        used[rows[len(c):]] = True
    m = D.metrics(pc[used], true[used])
    if with_extras:
        m["n_used"] = int(used.sum())
        m["used_frac"] = round(float(used.mean()), 4)
        m["hours"] = round(float(used.sum() * 5 / 3600), 2)
        m["co2_range"] = [round(float(true[used].min())), round(float(true[used].max()))]
    return m, used


def ridge_grid(X, col):
    Xd = {k: np.asarray(X)[v] for k, v in idx.items()}
    sx = StandardScaler().fit(Xd["train"])
    best = None
    for a in (10.0, 100.0, 1000.0):
        m = Ridge(alpha=a).fit(sx.transform(Xd["train"]), yd["train"][:, col])
        mae = np.abs(m.predict(sx.transform(Xd["val"])) - yd["val"][:, col]).mean()
        if best is None or mae < best[0]:
            best = (mae, a, m)
    _, a, m = best
    return m, sx, a, Xd


# ================================================== 1. coverage + hold baseline
te = idx["test"]
true_te = scd[te]
seg_te = seg_all[te]

hold = np.full(len(te), np.nan)
used_h = np.zeros(len(te), bool)
for s in np.unique(seg_te):
    rows = np.where(seg_te == s)[0]
    c = rows[:min(len(rows), CAL30)]
    hold[rows] = true_te[c].mean()
    used_h[rows[len(c):]] = True
mh = D.metrics(hold[used_h], true_te[used_h])
mh.update(n_used=int(used_h.sum()), used_frac=round(float(used_h.mean()), 4),
          hours=round(float(used_h.sum() * 5 / 3600), 2),
          co2_range=[round(float(true_te[used_h].min())), round(float(true_te[used_h].max()))])
log("hold_baseline_30min", mh)

# ================================================== 2. ridge-72 + offset
m72, sx72, a72, Xd72 = ridge_grid(X72, 1)
p72 = m72.predict(sx72.transform(Xd72["test"]))
log("ridge72_free", dict(D.metrics(p72, true_te), alpha=a72))
mr, _ = protocol(p72, "test", with_extras=True)
log("ridge72_recal30", mr)

# ================================================== 3. physics B/R+RH
br = pd.concat([df[f"OP{i}_blue"] / df[f"OP{i}_red"] for i in range(4)],
               axis=1).mean(axis=1).values
rh_meas = aux["BME_RH"].values
tr = idx["train"]
A = np.column_stack([br[tr], rh_meas[tr], np.ones(len(tr))])
coef, *_ = np.linalg.lstsq(A, scd[tr], rcond=None)
phys = np.column_stack([br, rh_meas, np.ones(len(br))]) @ coef
log("physics_free_full_test", D.metrics(phys[te], true_te))
mp, used_p = protocol(phys[te], "test", with_extras=True)
log("physics_recal30", mp)
# no-offset on the SAME evaluated windows
log("physics_noOffset_on_evaluated_windows", D.metrics(phys[te][used_p], true_te[used_p]))
# cal-window sensitivity
for mins in (10, 60):
    mm, _ = protocol(phys[te], "test", cal=int(mins * 60 / 5), with_extras=True)
    log(f"physics_recal{mins}min", mm)

# ---- ink-derived RH variant: RH from ridge72+HP8h (the manuscript RH model)
Xhp8 = pd.concat([X72, D.add_highpass(X72, 8.0)], axis=1)
mrh, sxrh, arh, Xdrh = ridge_grid(Xhp8, 0)
rh_hat = mrh.predict(sxrh.transform(np.asarray(Xhp8)))
log("rh_model_test", dict(D.metrics(rh_hat[te], yarr[te, 0]), alpha=arh))
# (a) coefficients fit with measured RH, applied with ink-RH
phys_a = np.column_stack([br, rh_hat, np.ones(len(br))]) @ coef
ma_, _ = protocol(phys_a[te], "test")
log("physics_inkRH_applyOnly_recal30", ma_)
# (b) coefficients fit AND applied with ink-RH
A2 = np.column_stack([br[tr], rh_hat[tr], np.ones(len(tr))])
coef2, *_ = np.linalg.lstsq(A2, scd[tr], rcond=None)
phys_b = np.column_stack([br, rh_hat, np.ones(len(br))]) @ coef2
mb_, _ = protocol(phys_b[te], "test")
log("physics_inkRH_fitApply_recal30", mb_)

# ================================================== 4. IRN12 ridge
def irn12(dframe):
    out = {}
    for i in range(4):
        r, g, b, ir = (dframe[f"OP{i}_{c}"] for c in ("red", "green", "blue", "ir"))
        out[f"OP{i}_R_IR"] = r / ir
        out[f"OP{i}_G_IR"] = g / ir
        out[f"OP{i}_B_IR"] = b / ir
    X = pd.DataFrame(out, index=dframe.index)
    return X.replace([np.inf, -np.inf], np.nan).ffill().bfill()


XI = irn12(df)
mi, sxi, ai, Xdi = ridge_grid(XI, 1)
pi_full = mi.predict(sxi.transform(np.asarray(XI)))
pi = pi_full[te]
log("irn12_free", dict(D.metrics(pi, true_te), alpha=ai,
                       train_mae=D.metrics(mi.predict(sxi.transform(Xdi["train"])), yd["train"][:, 1])["mae"]))
mi_r, _ = protocol(pi, "test", with_extras=True)
log("irn12_recal30", mi_r)
# vs saved artifact
art = np.load(RES / "preds_irn12_recal.npz")
log("irn12_recal_artifact_check",
    dict(D.metrics(art["pred"][art["used"]], art["true"][art["used"]]),
         max_abs_diff_pred_freerun=float(np.abs(np.load(RES / 'full_ridge_irn12.npz')['pred'][te] - pi).max())))

# ================================================== 5. indicator mode
pfull = np.load(RES / "full_ridge_irn12.npz")["pred"]
pte = pfull[te]
ind = {}
for thr in (800, 1000):
    yb = (true_te > thr).astype(int)
    auc = roc_auc_score(yb, pte)
    acc = float(((pte > thr).astype(int) == yb).mean())
    maj = float(max(yb.mean(), 1 - yb.mean()))
    ind[f"thr{thr}"] = {"auc": round(float(auc), 3), "acc": round(acc, 3),
                        "majority": round(maj, 3), "pos_frac": round(float(yb.mean()), 3)}
# same with the freshly recomputed predictions
for thr in (800, 1000):
    yb = (true_te > thr).astype(int)
    ind[f"thr{thr}_recomputed_preds"] = {
        "auc": round(float(roc_auc_score(yb, pi)), 3),
        "acc": round(float(((pi > thr).astype(int) == yb).mean()), 3)}
log("indicator", ind)

# ================================================== 6. lamp ablation
Xl = pd.concat([X72, aux[["lamp_pct"]]], axis=1)
ml, sxl, al, Xdl = ridge_grid(Xl, 1)
pl = ml.predict(sxl.transform(Xdl["test"]))
log("ridge72_plus_lamp_free", dict(D.metrics(pl, true_te), alpha=al,
    pct_change_vs_403=round(100 * (D.metrics(pl, true_te)["mae"] - 403.34) / 403.34, 1)))
XIl = pd.concat([XI, aux[["lamp_pct"]]], axis=1)
mil, sxil, ail, Xdil = ridge_grid(XIl, 1)
pil = mil.predict(sxil.transform(Xdil["test"]))
log("irn12_plus_lamp", dict(free=D.metrics(pil, true_te),
                            recal=protocol(pil, "test")[0],
                            delta_recal_ppm=round(protocol(pil, "test")[0]["mae"] - mi_r["mae"], 2),
                            pct_change_free=round(100 * (D.metrics(pil, true_te)["mae"] - D.metrics(pi, true_te)["mae"]) / D.metrics(pi, true_te)["mae"], 1)))

# ================================================== 7. train+val refit
tv = np.concatenate([idx["train"], idx["val"]])


def refit(X, alpha, col=1):
    Xa = np.asarray(X)
    sx = StandardScaler().fit(Xa[tv])
    m = Ridge(alpha=alpha).fit(sx.transform(Xa[tv]), yarr[tv, col])
    return m.predict(sx.transform(Xa))


# ridge72 (alpha 1000), ridge+HP4 (alpha 100), irn12 (alpha 100), physics
ptv72 = refit(X72, 1000.0)
log("tv_ridge72_free", D.metrics(ptv72[te], true_te))
Xhp4 = pd.concat([X72, D.add_highpass(X72, 4.0)], axis=1)
ptvhp = refit(Xhp4, 100.0)
log("tv_ridge_hp4_free", D.metrics(ptvhp[te], true_te))
ptvi = refit(XI, 100.0)
log("tv_irn12_free", D.metrics(ptvi[te], true_te))
mtvi, _ = protocol(ptvi[te], "test")
log("tv_irn12_recal30", mtvi)
Atv = np.column_stack([br[tv], rh_meas[tv], np.ones(len(tv))])
ctv, *_ = np.linalg.lstsq(Atv, scd[tv], rcond=None)
ptvp = np.column_stack([br, rh_meas, np.ones(len(br))]) @ ctv
mtvp, _ = protocol(ptvp[te], "test")
log("tv_physics_recal30", mtvp)

# artifact cross-checks
for nm in ("tvfull_ridge72", "tvfull_ridge_hp4", "tvfull_ridge_irn12", "tvfull_physics",
           "tvfull_mlp_reg", "tvfull_lstm6h", "tvfull_mlp1024"):
    d = np.load(RES / f"{nm}.npz")
    p = d["pred"][te]
    rec = {"free_test": D.metrics(p[~np.isnan(p)], true_te[~np.isnan(p)])}
    rec["recal30"] = protocol(np.nan_to_num(p, nan=float(np.nanmean(p))), "test")[0] if not np.isnan(p).any() else protocol(pd.Series(p).ffill().bfill().values, "test")[0]
    log(f"artifact_{nm}", rec)
for nm in ("tvrecal_ridge_irn12", "tvrecal_physics", "tvrecal_lstm6h"):
    d = np.load(RES / f"{nm}.npz")
    log(f"artifact_{nm}", D.metrics(d["pred"][d["used"]], d["true"][d["used"]]))

# single-seed MLP artifacts (train-only fit)
for nm in ("full_mlp_reg",):
    d = np.load(RES / f"{nm}.npz")
    p = d["pred"][te].astype(np.float64)
    log(f"artifact_{nm}", {"free_test": D.metrics(p, true_te),
                           "recal30": protocol(p, "test")[0]})

# LSTM artifacts (train-only champion + tv refit)
d = np.load(RES / "preds6_lstm_recal_best.npz")
log("artifact_preds6_lstm_recal_best",
    dict(D.metrics(d["pred"][d["used"]], d["true"][d["used"]]),
         shape_r_raw=round(float(np.corrcoef(d["raw"], d["true"])[0, 1]), 3),
         n_used=int(d["used"].sum())))
d = np.load(RES / "tvfull_lstm6h.npz")
p = pd.Series(d["pred"][te]).ffill().bfill().values
log("artifact_tv_lstm6h_shape_r", {"r": round(float(np.corrcoef(p, true_te)[0, 1]), 3),
                                   "recal30": protocol(p, "test")[0]})

# ================================================== 8. appendix artifacts
d = np.load(RES / "preds5_rnn_champion.npz")
log("artifact_rnn_champion",
    dict(D.metrics(d["pred"][d["used_mask"]], d["true"][d["used_mask"]])))
d = np.load(RES / "preds_idea3_patchtst_final.npz")
log("artifact_patchtst",
    dict(D.metrics(d["pred"][d["used"]], d["true"][d["used"]]),
         shape_r_raw=round(float(np.corrcoef(d["raw"].astype(np.float64), d["true"])[0, 1]), 3)))

# r3 physics heldout npz (val+test halves) — identify
d = np.load(RES / "preds_r3_physics_heldout.npz")
n = len(d["t_h"]) // 2
for half, sl in (("val", slice(0, n)), ("test", slice(n, None))):
    u = d["used"][sl]
    log(f"artifact_r3_physics_heldout_{half}",
        dict(D.metrics(d["pred"][sl][u], d["true"][sl][u]), used_frac=round(float(u.mean()), 3)))

# ================================================== 9. B/R noise scale
x = pd.Series(br)
ma60 = x.rolling(12, min_periods=1).mean()
hf60 = x - ma60
est = {}
est["raw_minus_ma60_std"] = float(hf60.std())
# noise remaining IN the 60-s averaged signal: detrend vs 10-min rolling mean,
# inside lamp-stable flat segments only (settled)
resid = ma60 - ma60.rolling(120, min_periods=1).mean()
stable = np.zeros(len(x), bool)
for s in np.unique(seg_all):
    rows = np.where(seg_all == s)[0]
    if len(rows) > int(2 * 3600 / 5):
        stable[rows[int(3600 / 5):]] = True
est["ma60_detrended10min_std_stable"] = float(resid[stable].std())
est["ppm_at_5e-6"] = round(est["ma60_detrended10min_std_stable"] / 5e-6, 1)
est["claim_arithmetic_1.1e-4_over_5e-6"] = round(1.1e-4 / 5e-6, 1)
log("br_noise", est)

(Path := RES / "verify_provC.json").write_text(json.dumps(OUT, indent=1))
print("DONE")
