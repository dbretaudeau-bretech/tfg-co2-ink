"""TFGV4 ladder round 2 — device-realistic scenario + recalibration variants.

Scenario A (ink-only) was round 1. Here:
  B) device-realistic inputs: ink 72-feat + lamp_pct (device's own actuator)
     + BME_RH/BME_T (co-located cheap sensor) — the practical deployment case.
  C) ink-only but with per-lamp-segment standardization of features
     (device can re-zero at lamp changes).
  D) B + 2h high-pass block.
Targets unchanged: (BME_RH, SCD_CO2); when RH is an input, target is SCD_CO2 only.
"""
import json
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

import data4 as D

RES = D.RESULTS
METRICS_PATH = RES / "metrics2.json"


def save_trial(name, info, pred=None, true=None, t=None):
    all_m = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    all_m[name] = info
    METRICS_PATH.write_text(json.dumps(all_m, indent=2))
    if pred is not None:
        np.savez_compressed(RES / f"preds_{name}.npz", pred=pred, true=true, t_h=t)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps(info)}", flush=True)


# ---------------------------------------------------------------- load
df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
X72, y, t_h, idx = D.load_light_fe()
aux = df[["lamp_pct", "BME_RH", "BME_T"]].reset_index(drop=True).ffill().bfill()
scd = y["SCD_CO2"].values
yd = {k: np.asarray(y)[v] for k, v in idx.items()}
th = {k: t_h[v] for k, v in idx.items()}


def ridge_grid(Xd, ytr_col, yval_col):
    sx = StandardScaler().fit(Xd["train"])
    best = None
    for a in (10.0, 100.0, 1000.0):
        m = Ridge(alpha=a).fit(sx.transform(Xd["train"]), ytr_col)
        pv = m.predict(sx.transform(Xd["val"]))
        mae = np.abs(pv - yval_col).mean()
        if best is None or mae < best[0]:
            best = (mae, a, m)
    _, a, m = best
    return m, sx, a


def run(name, fn):
    try:
        fn()
    except Exception:
        print(f"!! {name} FAILED:\n{traceback.format_exc()}", flush=True)
        save_trial(name, {"error": "failed"})


# ------------------------------------------------ B1: ridge, ink+lamp+RH+T → CO2
def b1():
    Xb = pd.concat([X72, aux], axis=1)
    Xd, _, _ = D.split_xy(Xb, y, t_h, idx)
    m, sx, a = ridge_grid(Xd, yd["train"][:, 1], yd["val"][:, 1])
    pred = m.predict(sx.transform(Xd["test"]))
    info = {"CO2": D.metrics(pred, yd["test"][:, 1]), "alpha": a}
    save_trial("b1_ridge_devreal", info, pred, yd["test"][:, 1], th["test"])


# ------------------------------------------------ B2: ridge, devreal + HP2h
def b2():
    Xb = pd.concat([X72, D.add_highpass(X72, 2.0), aux], axis=1)
    Xd, _, _ = D.split_xy(Xb, y, t_h, idx)
    m, sx, a = ridge_grid(Xd, yd["train"][:, 1], yd["val"][:, 1])
    pred = m.predict(sx.transform(Xd["test"]))
    info = {"CO2": D.metrics(pred, yd["test"][:, 1]), "alpha": a}
    save_trial("b2_ridge_devreal_hp2h", info, pred, yd["test"][:, 1], th["test"])


# ------------------------------------------------ B3: HistGB devreal
def b3():
    Xb = pd.concat([X72, D.add_highpass(X72, 2.0), aux], axis=1)
    Xd, _, _ = D.split_xy(Xb, y, t_h, idx)
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
                                      random_state=0)
    m.fit(Xd["train"], yd["train"][:, 1])
    pred = m.predict(Xd["test"])
    save_trial("b3_histgb_devreal", {"CO2": D.metrics(pred, yd["test"][:, 1])},
               pred, yd["test"][:, 1], th["test"])


# ------------------------------------------------ C1: per-lamp-segment z-norm, ink-only
def _segments(lamp):
    lampr = lamp.round(0).values
    bounds = [0] + list(np.where(np.diff(lampr) != 0)[0] + 1) + [len(lampr)]
    seg_id = np.zeros(len(lampr), dtype=int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg_id[a:b] = s
    return seg_id


def c1():
    seg = _segments(aux["lamp_pct"])
    Xn = X72.copy()
    SETTLE = int(3600 / 5)  # drop first hour of each segment from train metric calc
    for s in np.unique(seg):
        rows = np.where(seg == s)[0]
        ref = rows[:min(len(rows), SETTLE)]  # z-norm against segment's first hour
        mu = X72.iloc[ref].mean()
        sd = X72.iloc[ref].std() + 1e-9
        Xn.iloc[rows] = (X72.iloc[rows] - mu) / sd
    Xd, _, _ = D.split_xy(Xn, y, t_h, idx)
    m, sx, a = ridge_grid(Xd, yd["train"][:, 1], yd["val"][:, 1])
    pred = m.predict(sx.transform(Xd["test"]))
    info = {"CO2": D.metrics(pred, yd["test"][:, 1]), "alpha": a,
            "note": "features re-zeroed at lamp changes (first-hour ref)"}
    save_trial("c1_ridge_segnorm", info, pred, yd["test"][:, 1], th["test"])


# ------------------------------------------------ C2: segnorm + devreal aux
def c2():
    seg = _segments(aux["lamp_pct"])
    Xn = X72.copy()
    SETTLE = int(3600 / 5)
    for s in np.unique(seg):
        rows = np.where(seg == s)[0]
        ref = rows[:min(len(rows), SETTLE)]
        mu = X72.iloc[ref].mean()
        sd = X72.iloc[ref].std() + 1e-9
        Xn.iloc[rows] = (X72.iloc[rows] - mu) / sd
    Xb = pd.concat([Xn, aux], axis=1)
    Xd, _, _ = D.split_xy(Xb, y, t_h, idx)
    m, sx, a = ridge_grid(Xd, yd["train"][:, 1], yd["val"][:, 1])
    pred = m.predict(sx.transform(Xd["test"]))
    info = {"CO2": D.metrics(pred, yd["test"][:, 1]), "alpha": a}
    save_trial("c2_ridge_segnorm_devreal", info, pred, yd["test"][:, 1], th["test"])


# ------------------------------------------------ B4: HistGB devreal RH check (RH target, ink+lamp+T, no RH input)
def b4():
    Xb = pd.concat([X72, aux[["lamp_pct", "BME_T"]]], axis=1)
    Xd, _, _ = D.split_xy(Xb, y, t_h, idx)
    m, sx, a = ridge_grid(Xd, yd["train"][:, 0], yd["val"][:, 0])
    pred = m.predict(sx.transform(Xd["test"]))
    info = {"RH": D.metrics(pred, yd["test"][:, 0]), "alpha": a}
    save_trial("b4_ridge_rh_devreal", info, pred, yd["test"][:, 0], th["test"])


# ------------------------------------------------ R1: ridge72 static + per-segment
# one-point recalibration on test (offset from first 30 min of each lamp segment —
# standard gas-sensor field practice)
def r1():
    Xd, _, _ = D.split_xy(X72, y, t_h, idx)
    m, sx, a = ridge_grid(Xd, yd["train"][:, 1], yd["val"][:, 1])
    pred = m.predict(sx.transform(Xd["test"]))
    true = yd["test"][:, 1]
    seg_all = _segments(aux["lamp_pct"])
    seg_te = seg_all[idx["test"]]
    CAL = int(30 * 60 / 5)
    pred_cal = pred.copy()
    used = np.zeros(len(pred), bool)
    for s in np.unique(seg_te):
        rows = np.where(seg_te == s)[0]
        cal = rows[:min(len(rows), CAL)]
        off = (true[cal] - pred[cal]).mean()
        pred_cal[rows] = pred[rows] + off
        used[rows[len(cal):]] = True  # exclude calibration samples from metrics
    info = {"CO2": D.metrics(pred_cal[used], true[used]), "alpha": a,
            "note": "per-lamp-segment offset recal (30min), cal samples excluded"}
    save_trial("r1_ridge72_segrecal", info, pred_cal, true, th["test"])


# ------------------------------------------------ R2: physics-based B/R calibration
# transfer: RH-corrected B/R linear model fitted on train segments, applied to test
# with per-segment offset (30 min). Connects Sec III characterisation to deployment.
def r2():
    br = pd.concat([df[f"OP{i}_blue"] / df[f"OP{i}_red"] for i in range(4)],
                   axis=1).mean(axis=1).values
    rh = aux["BME_RH"].values
    tr = idx["train"]
    A = np.column_stack([br[tr], rh[tr], np.ones(len(tr))])
    coef, *_ = np.linalg.lstsq(A, scd[tr], rcond=None)
    te = idx["test"]
    pred = np.column_stack([br[te], rh[te], np.ones(len(te))]) @ coef
    true = scd[te]
    seg_te = _segments(aux["lamp_pct"])[te]
    CAL = int(30 * 60 / 5)
    pred_cal = pred.copy()
    used = np.zeros(len(pred), bool)
    for s in np.unique(seg_te):
        rows = np.where(seg_te == s)[0]
        cal = rows[:min(len(rows), CAL)]
        off = (true[cal] - pred[cal]).mean()
        pred_cal[rows] = pred[rows] + off
        used[rows[len(cal):]] = True
    info = {"CO2": D.metrics(pred_cal[used], true[used]),
            "note": "B/R+RH linear, per-segment offset recal (30min)"}
    save_trial("r2_BR_RH_linear_segrecal", info, pred_cal, true, th["test"])


if __name__ == "__main__":
    for nm, fn in [("b1", b1), ("b2", b2), ("b3", b3),
                   ("c1", c1), ("c2", c2), ("b4", b4),
                   ("r1", r1), ("r2", r2)]:
        run(nm, fn)
    print("LADDER2 DONE", flush=True)
