"""IDEA 2 — IR-channel normalization: divide visible channels by the co-measured
IR channel (R/IR, G/IR, B/IR per pixel) to dynamically cancel illumination/drift.

Experiments:
  (0) drift quantification: train-vs-test per-feature mean shift in train-std
      units for raw B / B-over-R / B-over-IR (and friends).
  (P) physics: is IR flat wrt CO2/RH inside lamp-stable segments? Does IR jump
      at lamp transitions with the same relative magnitude as the visible
      channels (i.e. does division actually cancel the lamp)?
  (a) ridge on pure IR-normalized features vs ridge on the standard 72
      (plus HP4h variants), no-recal test metrics + champion recal protocol.
  (b) coarse-context LSTM (60-s MA, 1/min, 6-h window) fed IR-normalized
      channels vs the 16 raw channels, 3 seeds, no-recal + recal protocol.

Outputs: results/idea2_irnorm.json
Champion protocol identical to run_ladder2.r2 / run_ladder5.protocol_eval.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import data4 as D

RES = D.RESULTS
OUT_PATH = RES / "idea2_irnorm.json"
OUT = {}


def log(section, obj):
    OUT[section] = obj
    OUT_PATH.write_text(json.dumps(OUT, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] {section}: {json.dumps(obj)[:400]}",
          flush=True)


# ---------------------------------------------------------------- data
df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
X72, y, t_h, idx = D.load_light_fe()
assert len(df) == len(X72)
aux = df[["lamp_pct", "BME_RH", "BME_T"]].ffill().bfill()
yarr = np.asarray(y)
scd = yarr[:, 1]
rh_t = yarr[:, 0]
yd = {k: yarr[v] for k, v in idx.items()}
th = {k: t_h[v] for k, v in idx.items()}


def _segments(lamp):
    lampr = lamp.round(0).values
    bounds = [0] + list(np.where(np.diff(lampr) != 0)[0] + 1) + [len(lampr)]
    seg_id = np.zeros(len(lampr), dtype=int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg_id[a:b] = s
    return seg_id


seg_all = _segments(aux["lamp_pct"])
CAL = int(30 * 60 / 5)  # 30 min @ 5 s


def protocol_eval(pred_split, split):
    """EXACT champion protocol: per-lamp-segment 30-min offset recal on the
    split, calibration samples excluded from metrics."""
    sel = idx[split]
    true = scd[sel]
    seg_s = seg_all[sel]
    pred_cal = pred_split.copy()
    used = np.zeros(len(sel), bool)
    for s in np.unique(seg_s):
        rows = np.where(seg_s == s)[0]
        cal = rows[:min(len(rows), CAL)]
        off = (true[cal] - pred_split[cal]).mean()
        pred_cal[rows] = pred_split[rows] + off
        used[rows[len(cal):]] = True
    m = D.metrics(pred_cal[used], true[used])
    m["full_r"] = round(float(np.corrcoef(pred_cal, true)[0, 1]), 3)
    m["used_frac"] = round(float(used.mean()), 3)
    return m


# ---------------------------------------------------------------- IRN features
def irn12(dframe):
    out = {}
    for i in range(4):
        r, g, b, ir = (dframe[f"OP{i}_{c}"] for c in ("red", "green", "blue", "ir"))
        out[f"OP{i}_R_IR"] = r / ir
        out[f"OP{i}_G_IR"] = g / ir
        out[f"OP{i}_B_IR"] = b / ir
    X = pd.DataFrame(out, index=dframe.index)
    return X.replace([np.inf, -np.inf], np.nan).ffill().bfill()


XI12 = irn12(df)
ext = {}
for nm in ("R_IR", "G_IR", "B_IR"):
    blk = XI12[[f"OP{i}_{nm}" for i in range(4)]]
    ext[f"px_{nm}_mean"] = blk.mean(axis=1)
    ext[f"px_{nm}_std"] = blk.std(axis=1)
for i in range(4):
    # (B/IR)/(R/IR) — algebraically identical to B/R; kept because the idea
    # spec names it as the "B/R of IR-normalized" variant.
    ext[f"OP{i}_BIR_over_RIR"] = XI12[f"OP{i}_B_IR"] / XI12[f"OP{i}_R_IR"]
brn = pd.DataFrame({k: v for k, v in ext.items() if "BIR_over_RIR" in k})
ext["px_BIR_over_RIR_mean"] = brn.mean(axis=1)
ext["px_BIR_over_RIR_std"] = brn.std(axis=1)
XI24 = pd.concat([XI12, pd.DataFrame(ext, index=df.index)], axis=1)  # 24 feats


# ================================================================ (0) drift
def drift_table():
    fams = {
        "raw_B": [f"OP{i}_blue" for i in range(4)],
        "raw_R": [f"OP{i}_red" for i in range(4)],
        "raw_G": [f"OP{i}_green" for i in range(4)],
        "raw_IR": [f"OP{i}_ir" for i in range(4)],
        "B_over_R": [f"OP{i}_BR" for i in range(4)],
        "B_over_RGB": [f"OP{i}_B_RGB" for i in range(4)],
        "B_over_IR": [f"OP{i}_B_IR" for i in range(4)],
        "G_over_IR": [f"OP{i}_G_IR" for i in range(4)],
        "R_over_IR": [f"OP{i}_R_IR" for i in range(4)],
    }
    src = pd.concat([df[D.INK_RAW], X72[[c for c in X72.columns
                                         if c not in D.INK_RAW]], XI12], axis=1)
    tab = {}
    for fam, cols in fams.items():
        shifts = []
        for c in cols:
            v = src[c].values
            mu_tr, sd_tr = v[idx["train"]].mean(), v[idx["train"]].std()
            shifts.append((v[idx["test"]].mean() - mu_tr) / (sd_tr + 1e-12))
        tab[fam] = {"per_pixel": [round(s, 2) for s in shifts],
                    "mean_abs": round(float(np.mean(np.abs(shifts))), 2)}
    log("drift_train_to_test_shift_in_train_std", tab)


# ================================================================ (P) physics
def physics_checks():
    pxm = {c: df[[f"OP{i}_{c}" for i in range(4)]].mean(axis=1).values
           for c in ("red", "green", "blue", "ir")}
    b_ir = pxm["blue"] / pxm["ir"]
    b_r = pxm["blue"] / pxm["red"]

    # --- within lamp-stable segments: is IR flat wrt CO2 / RH?
    sigs = {"raw_IR": pxm["ir"], "raw_B": pxm["blue"],
            "B_over_IR": b_ir, "B_over_R": b_r}
    SETTLE = int(30 * 60 / 5)
    rows = []
    for s in np.unique(seg_all):
        r_ = np.where(seg_all == s)[0]
        if len(r_) < int(3600 / 5) + SETTLE:  # need >=1h after 30-min settle
            continue
        r_ = r_[SETTLE:]
        co2_std = scd[r_].std()
        rec = {"seg": int(s), "n": len(r_), "lamp": float(aux["lamp_pct"].values[r_[0]]),
               "co2_std": round(float(co2_std), 1),
               "rh_std": round(float(rh_t[r_].std()), 2)}
        for nm, v in sigs.items():
            rec[f"r_{nm}_CO2"] = round(float(np.corrcoef(v[r_], scd[r_])[0, 1]), 3)
            rec[f"r_{nm}_RH"] = round(float(np.corrcoef(v[r_], rh_t[r_])[0, 1]), 3)
        rows.append(rec)
    # length-weighted mean r, restricted to CO2-active segments for the CO2 column
    def wmean(key, active_key=None, thr=0.0):
        sel = [r for r in rows if (r[active_key] >= thr if active_key else True)]
        w = np.array([r["n"] for r in sel], float)
        v = np.array([r[key] for r in sel])
        return round(float((w * v).sum() / w.sum()), 3) if len(sel) else None
    summary = {}
    for nm in sigs:
        summary[f"r_{nm}_CO2_wmean_active"] = wmean(f"r_{nm}_CO2", "co2_std", 20.0)
        summary[f"r_{nm}_RH_wmean"] = wmean(f"r_{nm}_RH")
    summary["n_segments"] = len(rows)
    summary["n_co2_active_segments"] = sum(1 for r in rows if r["co2_std"] >= 20)
    log("physics_within_segment_correlations", {"summary": summary, "segments": rows})

    # --- lamp transitions: relative jumps per channel
    bounds = np.where(np.diff(aux["lamp_pct"].round(0).values) != 0)[0] + 1
    PRE = (int(15 * 60 / 5), int(60 / 5))    # minutes -15..-1 before
    POST = (int(2 * 60 / 5), int(16 * 60 / 5))  # minutes +2..+16 after
    trans = []
    for b in bounds:
        if b - PRE[0] < 0 or b + POST[1] >= len(df):
            continue
        pre = slice(b - PRE[0], b - PRE[1])
        post = slice(b + POST[0], b + POST[1])
        if seg_all[pre].max() != seg_all[pre].min() or \
           seg_all[post].max() != seg_all[post].min():
            continue  # another transition inside the window
        rec = {"t_h": round(float(t_h[b]), 2),
               "lamp": f'{aux["lamp_pct"].values[b-1]:.0f}->{aux["lamp_pct"].values[b]:.0f}'}
        for nm, v in (("R", pxm["red"]), ("G", pxm["green"]), ("B", pxm["blue"]),
                      ("IR", pxm["ir"]), ("B_over_IR", b_ir), ("B_over_R", b_r)):
            m0, m1 = np.median(v[pre]), np.median(v[post])
            rec[f"jump_{nm}_pct"] = round(float(100 * (m1 - m0) / m0), 2)
        trans.append(rec)
    jb = np.array([t["jump_B_pct"] for t in trans])
    jir = np.array([t["jump_IR_pct"] for t in trans])
    jbir = np.array([t["jump_B_over_IR_pct"] for t in trans])
    jbr = np.array([t["jump_B_over_R_pct"] for t in trans])
    summ = {
        "n_transitions": len(trans),
        "median_abs_jump_pct": {"B": round(float(np.median(np.abs(jb))), 2),
                                "IR": round(float(np.median(np.abs(jir))), 2),
                                "B_over_IR": round(float(np.median(np.abs(jbir))), 2),
                                "B_over_R": round(float(np.median(np.abs(jbr))), 2)},
        "jump_cancellation_ratio_BoverIR_vs_B": round(
            float(np.median(np.abs(jbir)) / max(np.median(np.abs(jb)), 1e-9)), 3),
        "corr_jumpB_jumpIR": round(float(np.corrcoef(jb, jir)[0, 1]), 3),
    }
    log("physics_lamp_transition_jumps", {"summary": summ, "transitions": trans})


# ================================================================ (a) ridge
def ridge_grid(Xd, col):
    sx = StandardScaler().fit(Xd["train"])
    best = None
    for a in (10.0, 100.0, 1000.0):
        m = Ridge(alpha=a).fit(sx.transform(Xd["train"]), yd["train"][:, col])
        pv = m.predict(sx.transform(Xd["val"]))
        mae = np.abs(pv - yd["val"][:, col]).mean()
        if best is None or mae < best[0]:
            best = (mae, a, m)
    val_mae, a, m = best
    return m, sx, a, val_mae


def ridge_block():
    sets = {
        "ridge72": X72,
        "ridge_rawRGB12": df[[f"OP{i}_{c}" for i in range(4)
                              for c in ("red", "green", "blue")]],
        "ridge_IRN12": XI12,
        "ridge_IRN24": XI24,
        "ridge72_hp4h": pd.concat([X72, D.add_highpass(X72, 4.0)], axis=1),
        "ridge_IRN24_hp4h": pd.concat([XI24, D.add_highpass(XI24, 4.0)], axis=1),
    }
    res = {}
    for nm, X in sets.items():
        Xd = {k: np.asarray(X)[v] for k, v in idx.items()}
        rec = {"n_features": X.shape[1]}
        for col, tgt in ((0, "RH"), (1, "CO2")):
            m, sx, a, vmae = ridge_grid(Xd, col)
            pred = m.predict(sx.transform(Xd["test"]))
            rec[tgt] = D.metrics(pred, yd["test"][:, col])
            rec[tgt]["alpha"] = a
            rec[tgt]["val_mae"] = round(float(vmae), 2)
            if tgt == "CO2":
                rec["CO2_recal30min"] = protocol_eval(pred, "test")
                pv = m.predict(sx.transform(Xd["val"]))
                rec["CO2_recal30min_val"] = protocol_eval(pv, "val")
        res[nm] = rec
        log(f"ridge::{nm}", rec)
    return res


# ================================================================ (b) LSTM
import torch
import torch.nn as nn

MA, STRIDE, W = 12, 12, 360  # 60-s MA, 1/min sampling, 6-h window
NEED = W * STRIDE


def lstm_coarse(feat_df, seed, hidden=64, epochs=40):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xs = pd.DataFrame(np.asarray(feat_df)).rolling(MA, min_periods=1).mean().values
    sx = StandardScaler().fit(Xs[idx["train"]])
    sy = StandardScaler().fit(yarr[idx["train"]])
    Xn = sx.transform(Xs)
    yn = sy.transform(yarr)
    nch = Xn.shape[1]

    def windows_inner(split, stride_out):
        # ladder4-exact: windows fully inside the split (skips first 6 h)
        sel = idx[split]
        a, b = sel[0], sel[-1]
        ends = np.arange(a + NEED, b + 1, stride_out)
        xs = np.stack([Xn[e - NEED:e:STRIDE] for e in ends])
        return xs, yn[ends], t_h[ends], ends

    def windows_full(split):
        # every 5-s sample of the split; windows may extend left across the
        # boundary (inputs only — documented liberty, same as run_tune4/ladder5)
        ends = idx[split]
        ends = ends[ends - NEED >= 0]
        xs = np.stack([Xn[e - NEED:e:STRIDE] for e in ends])
        return xs, ends

    Xtr, ytr, _, _ = windows_inner("train", 6)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(nch, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1])

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xtr_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for b in range(0, n, 512):
            s = perm[b:b + 512]
            xb, yb = Xtr_t[s].to(dev), ytr_t[s].to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
    net.eval()

    def pred_arr(X):
        with torch.no_grad():
            ps = []
            for b in range(0, len(X), 4096):
                xb = torch.tensor(X[b:b + 4096], dtype=torch.float32).to(dev)
                ps.append(net(xb).cpu().numpy())
        return sy.inverse_transform(np.concatenate(ps))

    out = {}
    # no-recal, ladder4-comparable (inner windows, every 5 s)
    Xte, yte, _, _ = windows_inner("test", 1)
    pte = pred_arr(Xte)
    yte_o = sy.inverse_transform(yte)
    out["norecal_test"] = {"RH": D.metrics(pte[:, 0], yte_o[:, 0]),
                           "CO2": D.metrics(pte[:, 1], yte_o[:, 1])}
    # full-grid preds for the recal protocol (val for selection, test for report)
    full = {}
    for split in ("val", "test"):
        Xf, ends = windows_full(split)
        pf = pred_arr(Xf)
        sel = idx[split]
        co2 = np.full(len(sel), np.nan)
        pos = np.searchsorted(sel, ends)
        co2[pos] = pf[:, 1]
        co2 = pd.Series(co2).ffill().bfill().values  # only leading 6 h on val
        full[split] = co2
        out[f"recal30min_{split}"] = protocol_eval(co2, split)
        out[f"norecal_{split}_fullgrid_CO2"] = D.metrics(co2, scd[sel])
    return out, full


def lstm_block():
    res = {}
    preds = {}
    for nm, feats in (("lstm6h_raw16", df[D.INK_RAW]), ("lstm6h_IRN12", XI12)):
        per_seed = {}
        full_test, full_val = [], []
        for seed in (0, 1, 2):
            o, full = lstm_coarse(feats, seed)
            per_seed[f"seed{seed}"] = o
            full_test.append(full["test"])
            full_val.append(full["val"])
            log(f"lstm::{nm}::seed{seed}", o)
        # aggregates
        nr = [per_seed[f"seed{s}"]["norecal_test"]["CO2"]["mae"] for s in (0, 1, 2)]
        rc = [per_seed[f"seed{s}"]["recal30min_test"]["mae"] for s in (0, 1, 2)]
        rv = [per_seed[f"seed{s}"]["recal30min_val"]["mae"] for s in (0, 1, 2)]
        best_val_seed = int(np.argmin(rv))
        ens_te = protocol_eval(np.mean(full_test, axis=0), "test")
        ens_te_nr = D.metrics(np.mean(full_test, axis=0), scd[idx["test"]])
        agg = {
            "norecal_test_CO2_mae_mean_std": [round(float(np.mean(nr)), 1),
                                              round(float(np.std(nr)), 1)],
            "recal_test_CO2_mae_mean_std": [round(float(np.mean(rc)), 1),
                                            round(float(np.std(rc)), 1)],
            "recal_val_CO2_mae_per_seed": [round(v, 1) for v in rv],
            "val_best_seed": best_val_seed,
            "val_best_seed_test_recal": per_seed[f"seed{best_val_seed}"]["recal30min_test"],
            "ensemble3_test_recal": ens_te,
            "ensemble3_test_norecal_CO2": ens_te_nr,
        }
        res[nm] = {"per_seed": per_seed, "aggregate": agg}
        log(f"lstm::{nm}::aggregate", agg)
        preds[nm] = np.mean(full_test, axis=0)
    np.savez_compressed(RES / "preds_idea2_lstm_fullgrid_test.npz",
                        raw16=preds["lstm6h_raw16"], irn12=preds["lstm6h_IRN12"],
                        true=scd[idx["test"]], t_h=th["test"])
    return res


# ================================================================ main
if __name__ == "__main__":
    log("baselines_for_reference", {
        "norecal": {"lstm6h_raw16_1seed_metrics4": {"CO2_mae": 653.91, "r": 0.863},
                    "ridge72_hp4h_metrics1": {"CO2_mae": 296.75},
                    "ridge72_static_metrics1": {"CO2_mae": 403.34}},
        "recal30min": {"champion_lstm6h_recal": "47.7±1.2 (best file 46.7, n_used 2838)",
                       "physics_BR_RH_linear": 53.65,
                       "ridge72": 122.03},
    })
    drift_table()
    physics_checks()
    ridge_block()
    lstm_block()
    print("IDEA2 DONE", flush=True)
