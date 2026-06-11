"""TFGV4 IDEA 1 — chemical-kinetics features: causal derivatives of the ink
signals (d/dt, d2/dt2 at 1/5/15/60-min smoothing scales) to anticipate the
asymptote and neutralize the film's 3-44 min lag.

Trials:
  K-ridge : ridge on deriv-only / static72+deriv / per-scale ablations
  K-lead  : explicit lead compensation p + tau * dp/dt of the static ridge
  K-mlp   : small MLP on deriv sets, 3 seeds
  K-lstm  : small LSTM over windows of the derivative series, 3 seeds

Evaluation (both, on the full 5-s test grid, n=8241):
  no-recal : data4.metrics on the whole test block
  recal    : EXACT champion protocol (run_ladder2.r2 / run_ladder5.protocol_eval)
             = per-lamp-segment 30-min offset, cal samples excluded.

Honesty: all selection on validation. Val CO2 spans only 94-620 ppm
(std 168 vs test 258) so val CO2 ranking is weak — disclosed in notes.
Derivatives are CAUSAL (rolling-mean smoothing + backward difference);
no future samples are used (a centered SG filter at the 60-min scale would
leak 30 min of future — rejected).
Splits: data4.split_indices; derivative features computed on the continuous
full record (inputs only — same documented liberty as ladders 4/5: a deployed
device has its own continuous history).
"""
import json
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

import data4 as D

RES = D.RESULTS
MPATH = RES / "idea1_kinetics.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CAL = int(30 * 60 / 5)          # 360 samples = 30 min
SCALES_MIN = [1, 5, 15, 60]

# ---------------------------------------------------------------- data (once)
df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
X72, y, t_h, idx = D.load_light_fe()
aux = df[["lamp_pct"]].reset_index(drop=True).ffill().bfill()  # recal protocol only
scd = df["SCD_CO2"].values.astype(np.float64)
yarr = np.asarray(y, dtype=np.float64)
yd = {k: yarr[v] for k, v in idx.items()}


def _segments(lamp):
    lampr = lamp.round(0).values
    bounds = [0] + list(np.where(np.diff(lampr) != 0)[0] + 1) + [len(lampr)]
    seg_id = np.zeros(len(lampr), dtype=int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg_id[a:b] = s
    return seg_id


seg_all = _segments(aux["lamp_pct"])


def protocol_eval(pred_split: np.ndarray, split: str):
    """EXACT champion protocol (run_ladder2.r2): per-lamp-segment 30-min
    offset recal, cal samples excluded from metrics."""
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
    return m, pred_cal, used


def log(name, info):
    allm = json.loads(MPATH.read_text()) if MPATH.exists() else {}
    allm[name] = info
    MPATH.write_text(json.dumps(allm, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps(info)[:500]}",
          flush=True)


# ---------------------------------------------------------------- kinetic features
SIGNALS = pd.concat([X72[D.INK_RAW],
                     X72[["px_BR_mean", "px_B_RGB_mean", "px_IR_R_mean"]]],
                    axis=1)  # 19 base ink signals (pixel-mean ratios per spec)


def kinetic_features() -> dict[str, pd.DataFrame]:
    """Causal d/dt and d2/dt2 of the 19 ink signals at each smoothing scale.
    sm_w = rolling mean over w; d1 = (sm - sm.shift(w)) / (w*5s) [unit/s];
    d2 = (d1 - d1.shift(w)) / (w*5s). NaN head -> 0 (derivative unknown)."""
    out = {}
    for mins in SCALES_MIN:
        w = int(mins * 60 / 5)
        sm = SIGNALS.rolling(w, min_periods=1).mean()
        d1 = (sm - sm.shift(w)) / (w * 5.0)
        d2 = (d1 - d1.shift(w)) / (w * 5.0)
        out[f"d1_{mins}m"] = d1.add_suffix(f"_d1_{mins}m").fillna(0.0)
        out[f"d2_{mins}m"] = d2.add_suffix(f"_d2_{mins}m").fillna(0.0)
    return out


KIN = kinetic_features()
D1_ALL = pd.concat([KIN[f"d1_{m}m"] for m in SCALES_MIN], axis=1)      # 76
D12_ALL = pd.concat(list(KIN.values()), axis=1)                        # 152
for _X in (D1_ALL, D12_ALL):
    D._audit(_X.columns)
print(f"kinetic features: d1 {D1_ALL.shape[1]}, d1+d2 {D12_ALL.shape[1]}",
      flush=True)


# ---------------------------------------------------------------- ridge helper
def ridge_trial(name, X, note=""):
    """Joint (RH, CO2) ridge; alpha picked on val CO2 MAE (no-recal).
    Reports val (no-recal + protocol) and test (no-recal + protocol)."""
    Xd = {k: np.asarray(X)[v] for k, v in idx.items()}
    sx = StandardScaler().fit(Xd["train"])
    best = None
    for a in (10.0, 100.0, 1000.0, 10000.0):
        m = Ridge(alpha=a).fit(sx.transform(Xd["train"]), yd["train"])
        pv = m.predict(sx.transform(Xd["val"]))
        mae = np.abs(pv[:, 1] - yd["val"][:, 1]).mean()
        if best is None or mae < best[0]:
            best = (mae, a, m, pv)
    _, a, m, pv = best
    pt = m.predict(sx.transform(Xd["test"]))
    mv_rec, _, _ = protocol_eval(pv[:, 1], "val")
    mt_rec, pred_cal, used = protocol_eval(pt[:, 1], "test")
    info = {
        "alpha": a, "n_feat": X.shape[1], "note": note,
        "val_norecal_CO2": D.metrics(pv[:, 1], yd["val"][:, 1]),
        "val_recal_CO2": mv_rec,
        "test_norecal_CO2": D.metrics(pt[:, 1], yd["test"][:, 1]),
        "test_norecal_RH": D.metrics(pt[:, 0], yd["test"][:, 0]),
        "test_recal_CO2": mt_rec,
    }
    log(name, info)
    np.savez_compressed(RES / f"preds_idea1_{name}.npz", pred=pt[:, 1],
                        pred_cal=pred_cal, true=yd["test"][:, 1],
                        t_h=t_h[idx["test"]], used=used)
    return info


# ---------------------------------------------------------------- K1 ridges
def k1():
    ridge_trial("k1a_ridge_static72", X72, "static baseline reproduced")
    ridge_trial("k1b_ridge_d1", D1_ALL, "d1 only, all scales")
    ridge_trial("k1c_ridge_d1d2", D12_ALL, "d1+d2, all scales")
    ridge_trial("k1d_ridge_s72_d1d2", pd.concat([X72, D12_ALL], axis=1),
                "static72 + kinetics")
    for mins in SCALES_MIN:  # scale ablation, d1 only
        ridge_trial(f"k1e_ridge_d1_{mins}m", KIN[f"d1_{mins}m"],
                    f"d1 at {mins}-min scale only")
    ridge_trial("k1f_ridge_s72_d1", pd.concat([X72, D1_ALL], axis=1),
                "static72 + d1 only")


# ---------------------------------------------------------------- K2 lead compensation
def k2():
    """Classic first-order-lag inversion: if ink responds with time constant
    tau, then p(t) + tau * dp/dt(t) recovers the step input. Applied to the
    static ridge72 CO2 prediction; tau picked on validation."""
    Xd = {k: np.asarray(X72)[v] for k, v in idx.items()}
    sx = StandardScaler().fit(Xd["train"])
    m = Ridge(alpha=1000.0).fit(sx.transform(Xd["train"]), yd["train"])
    # full-record prediction so the derivative is causal & continuous
    p_full = m.predict(sx.transform(np.asarray(X72)))[:, 1]
    w = int(5 * 60 / 5)  # 5-min smoothing for dp/dt
    sm = pd.Series(p_full).rolling(w, min_periods=1).mean()
    dp = ((sm - sm.shift(w)) / (w * 5.0)).fillna(0.0).values
    rows_v, rows_t = idx["val"], idx["test"]
    best = None
    grid_s = [0, 3 * 60, 5 * 60, 10 * 60, 20 * 60, 44 * 60, 90 * 60]
    for tau in grid_s:
        pv = p_full[rows_v] + tau * dp[rows_v]
        mae = np.abs(pv - yd["val"][:, 1]).mean()
        mv_rec, _, _ = protocol_eval(pv, "val")
        if best is None or mae < best[0]:
            best = (mae, tau)
        log(f"k2_lead_tau{tau//60}m_VAL",
            {"tau_min": tau // 60, "val_norecal_mae": round(float(mae), 1),
             "val_recal_CO2": mv_rec})
    _, tau = best
    pt = p_full[rows_t] + tau * dp[rows_t]
    mt_rec, pred_cal, used = protocol_eval(pt, "test")
    info = {"tau_min_selected_on_val": tau // 60,
            "test_norecal_CO2": D.metrics(pt, yd["test"][:, 1]),
            "test_recal_CO2": mt_rec}
    log("k2_lead_final", info)
    np.savez_compressed(RES / "preds_idea1_k2_lead.npz", pred=pt,
                        pred_cal=pred_cal, true=yd["test"][:, 1],
                        t_h=t_h[rows_t], used=used)


# ---------------------------------------------------------------- K3 MLP
class MLP(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


def mlp_trial(name, X, seeds=(0, 1, 2)):
    Xd = {k: np.asarray(X, dtype=np.float32)[v] for k, v in idx.items()}
    sx = StandardScaler().fit(Xd["train"])
    sy = StandardScaler().fit(yd["train"])
    Xtr = torch.tensor(sx.transform(Xd["train"]), dtype=torch.float32)
    ytr = torch.tensor(sy.transform(yd["train"]), dtype=torch.float32)
    Xva = torch.tensor(sx.transform(Xd["val"]), dtype=torch.float32).to(DEV)
    Xte = torch.tensor(sx.transform(Xd["test"]), dtype=torch.float32).to(DEV)
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = MLP(X.shape[1]).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        best_val, best_state, patience = np.inf, None, 0
        for ep in range(200):
            perm = torch.randperm(len(Xtr))
            net.train()
            for b in range(0, len(Xtr), 1024):
                s = perm[b:b + 1024]
                xb, yb = Xtr[s].to(DEV), ytr[s].to(DEV)
                opt.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = sy.inverse_transform(net(Xva).cpu().numpy())
            vmae = np.abs(pv[:, 1] - yd["val"][:, 1]).mean()
            if vmae < best_val - 1e-3:
                best_val, patience = vmae, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 12:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pv = sy.inverse_transform(net(Xva).cpu().numpy())
            pt = sy.inverse_transform(net(Xte).cpu().numpy())
        mv_rec, _, _ = protocol_eval(pv[:, 1], "val")
        mt_rec, pred_cal, used = protocol_eval(pt[:, 1], "test")
        per_seed.append({
            "seed": seed, "val_norecal_mae": round(float(best_val), 1),
            "val_recal_mae": mv_rec["mae"],
            "test_norecal_CO2": D.metrics(pt[:, 1], yd["test"][:, 1]),
            "test_recal_CO2": mt_rec, "pt": pt[:, 1]})
    agg = {
        "n_feat": X.shape[1],
        "val_norecal_mae_mean": round(float(np.mean(
            [s["val_norecal_mae"] for s in per_seed])), 1),
        "test_norecal_mae": f'{np.mean([s["test_norecal_CO2"]["mae"] for s in per_seed]):.1f} ± {np.std([s["test_norecal_CO2"]["mae"] for s in per_seed]):.1f}',
        "test_norecal_r": f'{np.mean([s["test_norecal_CO2"]["pearson"] for s in per_seed]):.3f} ± {np.std([s["test_norecal_CO2"]["pearson"] for s in per_seed]):.3f}',
        "test_recal_mae": f'{np.mean([s["test_recal_CO2"]["mae"] for s in per_seed]):.1f} ± {np.std([s["test_recal_CO2"]["mae"] for s in per_seed]):.1f}',
        "test_recal_r2": f'{np.mean([s["test_recal_CO2"]["r2"] for s in per_seed]):.3f} ± {np.std([s["test_recal_CO2"]["r2"] for s in per_seed]):.3f}',
        "per_seed": [{k: v for k, v in s.items() if k != "pt"}
                     for s in per_seed]}
    # 3-seed prediction-average ensemble
    pens = np.mean([s["pt"] for s in per_seed], axis=0)
    mt_rec, pred_cal, used = protocol_eval(pens, "test")
    agg["ensemble3_test_norecal_CO2"] = D.metrics(pens, yd["test"][:, 1])
    agg["ensemble3_test_recal_CO2"] = mt_rec
    log(name, agg)
    np.savez_compressed(RES / f"preds_idea1_{name}_ens.npz", pred=pens,
                        pred_cal=pred_cal, true=yd["test"][:, 1],
                        t_h=t_h[idx["test"]], used=used)


def k3():
    mlp_trial("k3a_mlp_d1d2", D12_ALL)
    mlp_trial("k3b_mlp_s72_d1d2", pd.concat([X72, D12_ALL], axis=1))
    mlp_trial("k3c_mlp_static72", X72)  # MLP control without kinetics


# ---------------------------------------------------------------- K4 LSTM on deriv series
class LSTMNet(nn.Module):
    def __init__(self, nin, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(nin, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


def k4():
    """Windows of the derivative series: d1@1m + d1@15m of the 19 ink signals
    (38 ch), 2 h context at 60-s stride. Selection: val CO2 MAE (no recal),
    coarse val grid. Final predictions on the full 5-s test grid."""
    chans = pd.concat([KIN["d1_1m"], KIN["d1_15m"]], axis=1).values
    sx = StandardScaler().fit(chans[idx["train"]])
    Xn = sx.transform(chans).astype(np.float32)
    sy = StandardScaler().fit(yd["train"])
    STRIDE, CTX = 12, 120  # 60-s steps, 2 h
    NEED = STRIDE * (CTX - 1)

    def windows(ends):
        ends = ends[ends >= NEED]
        steps = np.arange(CTX) * STRIDE
        ix = ends[:, None] - NEED + steps[None, :]
        return Xn[ix], ends

    tr_ends = idx["train"][idx["train"] % 6 == 0]            # every 30 s
    va_ends_c = idx["val"][idx["val"] % 12 == 0]             # coarse val (60 s)
    Xtr, tr_ends = windows(tr_ends)
    Xva, va_ends_c = windows(va_ends_c)
    ytr = torch.tensor(sy.transform(yarr[tr_ends]), dtype=torch.float32)
    Xtr_t = torch.tensor(Xtr)
    Xva_t = torch.tensor(Xva).to(DEV)
    yva_co2 = yarr[va_ends_c][:, 1]
    per_seed = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = LSTMNet(Xn.shape[1]).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        best_val, best_state, patience = np.inf, None, 0
        for ep in range(60):
            perm = torch.randperm(len(Xtr_t))
            net.train()
            for b in range(0, len(Xtr_t), 512):
                s = perm[b:b + 512]
                xb, yb = Xtr_t[s].to(DEV), ytr[s].to(DEV)
                opt.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = sy.inverse_transform(net(Xva_t).cpu().numpy())
            vmae = np.abs(pv[:, 1] - yva_co2).mean()
            if vmae < best_val - 1e-3:
                best_val, patience = vmae, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 8:
                    break
        net.load_state_dict(best_state)
        net.eval()
        # full 5-s grid predictions for val + test (protocol comparability)
        full = {}
        for split in ("val", "test"):
            ends = idx[split]
            preds = []
            for b in range(0, len(ends), 1024):
                Xb, _ = windows(ends[b:b + 1024])
                with torch.no_grad():
                    preds.append(net(torch.tensor(Xb).to(DEV)).cpu().numpy())
            full[split] = sy.inverse_transform(np.concatenate(preds))[:, 1]
        mv_rec, _, _ = protocol_eval(full["val"], "val")
        mt_rec, pred_cal, used = protocol_eval(full["test"], "test")
        per_seed.append({
            "seed": seed, "val_norecal_mae_coarse": round(float(best_val), 1),
            "val_recal_mae": mv_rec["mae"],
            "test_norecal_CO2": D.metrics(full["test"], yd["test"][:, 1]),
            "test_recal_CO2": mt_rec, "pt": full["test"]})
        print(f"  k4 seed {seed} done", flush=True)
    agg = {
        "channels": "d1@1m + d1@15m of 19 ink signals (38ch), ctx 2h @60s",
        "test_norecal_mae": f'{np.mean([s["test_norecal_CO2"]["mae"] for s in per_seed]):.1f} ± {np.std([s["test_norecal_CO2"]["mae"] for s in per_seed]):.1f}',
        "test_norecal_r": f'{np.mean([s["test_norecal_CO2"]["pearson"] for s in per_seed]):.3f} ± {np.std([s["test_norecal_CO2"]["pearson"] for s in per_seed]):.3f}',
        "test_recal_mae": f'{np.mean([s["test_recal_CO2"]["mae"] for s in per_seed]):.1f} ± {np.std([s["test_recal_CO2"]["mae"] for s in per_seed]):.1f}',
        "test_recal_r2": f'{np.mean([s["test_recal_CO2"]["r2"] for s in per_seed]):.3f} ± {np.std([s["test_recal_CO2"]["r2"] for s in per_seed]):.3f}',
        "per_seed": [{k: v for k, v in s.items() if k != "pt"}
                     for s in per_seed]}
    pens = np.mean([s["pt"] for s in per_seed], axis=0)
    mt_rec, pred_cal, used = protocol_eval(pens, "test")
    agg["ensemble3_test_norecal_CO2"] = D.metrics(pens, yd["test"][:, 1])
    agg["ensemble3_test_recal_CO2"] = mt_rec
    log("k4_lstm_deriv_windows", agg)
    np.savez_compressed(RES / "preds_idea1_k4_lstm_ens.npz", pred=pens,
                        pred_cal=pred_cal, true=yd["test"][:, 1],
                        t_h=t_h[idx["test"]], used=used)


if __name__ == "__main__":
    for nm, fn in [("k1", k1), ("k2", k2), ("k3", k3), ("k4", k4)]:
        try:
            fn()
        except Exception:
            print(f"!! {nm} FAILED:\n{traceback.format_exc()}", flush=True)
            log(nm, {"error": "failed"})
    print("IDEA1 DONE", flush=True)
