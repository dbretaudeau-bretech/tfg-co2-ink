"""TFGV4 night ladder — ink-only prediction of MEASURED targets (BME_RH, SCD_CO2).

Each trial appends to results/metrics.json and saves test predictions to
results/preds_<trial>.npz so figures can be built later without re-fitting.
Designed to run unattended; failures in one trial don't kill the rest.
"""
import json
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

import data4 as D

RES = D.RESULTS
METRICS_PATH = RES / "metrics.json"


def save_trial(name, info, preds_test=None, true_test=None, t_test=None):
    all_m = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    all_m[name] = info
    METRICS_PATH.write_text(json.dumps(all_m, indent=2))
    if preds_test is not None:
        np.savez_compressed(RES / f"preds_{name}.npz",
                            pred=preds_test, true=true_test, t_h=t_test)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps(info)}", flush=True)


def eval_joint(pred, yte):
    return {"RH": D.metrics(pred[:, 0], yte[:, 0]),
            "CO2": D.metrics(pred[:, 1], yte[:, 1])}


# ---------------------------------------------------------------- load once
X72, y, t_h, idx = D.load_light_fe()
Xd72, yd, th = D.split_xy(X72, y, t_h, idx)
yte = yd["test"]
print(f"rows train/val/test: {[len(v) for v in idx.values()]}", flush=True)


def run(name, fn):
    try:
        fn()
    except Exception:
        print(f"!! {name} FAILED:\n{traceback.format_exc()}", flush=True)
        save_trial(name, {"error": "failed, see log"})


# ---------------------------------------------------------------- T1 OLS raw
def t1():
    Xraw = X72[D.INK_RAW]
    Xd, _, _ = D.split_xy(Xraw, y, t_h, idx)
    sx = StandardScaler().fit(Xd["train"])
    m = LinearRegression().fit(sx.transform(Xd["train"]), yd["train"])
    pred = m.predict(sx.transform(Xd["test"]))
    save_trial("t1_ols_raw16", eval_joint(pred, yte), pred, yte, th["test"])


# ---------------------------------------------------------------- T2 ridge72 static
def t2():
    best = None
    sx = StandardScaler().fit(Xd72["train"])
    for a in (10.0, 100.0, 1000.0):
        m = Ridge(alpha=a).fit(sx.transform(Xd72["train"]), yd["train"])
        pv = m.predict(sx.transform(Xd72["val"]))
        score = D.metrics(pv[:, 1], yd["val"][:, 1])["mae"]
        if best is None or score < best[0]:
            best = (score, a, m)
    _, a, m = best
    pred = m.predict(sx.transform(Xd72["test"]))
    info = eval_joint(pred, yte); info["alpha"] = a
    save_trial("t2_ridge72_static", info, pred, yte, th["test"])


# ---------------------------------------------------------------- T3 ridge HP
def t3():
    for hours in (1.0, 2.0, 4.0, 8.0):
        Xhp = pd.concat([X72, D.add_highpass(X72, hours)], axis=1)
        Xd, _, _ = D.split_xy(Xhp, y, t_h, idx)
        sx = StandardScaler().fit(Xd["train"])
        best = None
        for a in (10.0, 100.0, 1000.0):
            m = Ridge(alpha=a).fit(sx.transform(Xd["train"]), yd["train"])
            pv = m.predict(sx.transform(Xd["val"]))
            score = D.metrics(pv[:, 1], yd["val"][:, 1])["mae"]
            if best is None or score < best[0]:
                best = (score, a, m)
        _, a, m = best
        pred = m.predict(sx.transform(Xd["test"]))
        info = eval_joint(pred, yte); info["alpha"] = a
        save_trial(f"t3_ridge_hp{hours:g}h", info, pred, yte, th["test"])


# ---------------------------------------------------------------- T4 ratio-only + HP (lamp/drift robust by construction)
def t4():
    ratio_cols = [c for c in X72.columns if any(r in c for r in D.RATIOS)]
    Xr = X72[ratio_cols]
    Xhp = pd.concat([Xr, D.add_highpass(Xr, 2.0)], axis=1)
    Xd, _, _ = D.split_xy(Xhp, y, t_h, idx)
    sx = StandardScaler().fit(Xd["train"])
    best = None
    for a in (10.0, 100.0, 1000.0):
        m = Ridge(alpha=a).fit(sx.transform(Xd["train"]), yd["train"])
        pv = m.predict(sx.transform(Xd["val"]))
        score = D.metrics(pv[:, 1], yd["val"][:, 1])["mae"]
        if best is None or score < best[0]:
            best = (score, a, m)
    _, a, m = best
    pred = m.predict(sx.transform(Xd["test"]))
    info = eval_joint(pred, yte); info["alpha"] = a
    save_trial("t4_ridge_ratios_hp2h", info, pred, yte, th["test"])


# ---------------------------------------------------------------- T5 HistGB
def t5():
    Xhp = pd.concat([X72, D.add_highpass(X72, 2.0)], axis=1)
    Xd, _, _ = D.split_xy(Xhp, y, t_h, idx)
    preds = []
    for j, tgt in enumerate(["RH", "CO2"]):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          early_stopping=False, random_state=0)
        m.fit(Xd["train"], yd["train"][:, j])
        preds.append(m.predict(Xd["test"]))
    pred = np.column_stack(preds)
    save_trial("t5_histgb_hp2h", eval_joint(pred, yte), pred, yte, th["test"])


# ---------------------------------------------------------------- T6 cascade
def t6():
    Xhp = pd.concat([X72, D.add_highpass(X72, 2.0)], axis=1)
    Xd, _, _ = D.split_xy(Xhp, y, t_h, idx)
    sx = StandardScaler().fit(Xd["train"])
    Xtr, Xva, Xte = (sx.transform(Xd[k]) for k in ("train", "val", "test"))
    m1 = Ridge(alpha=100.0).fit(Xtr, yd["train"][:, 0])
    rh_tr, rh_te = m1.predict(Xtr), m1.predict(Xte)
    m2 = Ridge(alpha=100.0).fit(np.column_stack([Xtr, rh_tr]), yd["train"][:, 1])
    co2 = m2.predict(np.column_stack([Xte, rh_te]))
    pred = np.column_stack([rh_te, co2])
    save_trial("t6_cascade_hp2h", eval_joint(pred, yte), pred, yte, th["test"])


# ---------------------------------------------------------------- LSTM trials
def _make_windows(Xs, ys, W, stride, winnorm=False):
    xs, yy = [], []
    for i in range(W, len(Xs), stride):
        w = Xs[i - W:i]
        if winnorm:
            mu, sd = w.mean(0, keepdims=True), w.std(0, keepdims=True) + 1e-9
            w = (w - mu) / sd
        xs.append(w); yy.append(ys[i])
    return np.stack(xs), np.stack(yy)


def _lstm_trial(name, W, winnorm, hidden=64, epochs=25):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xraw = X72[D.INK_RAW].values
    sx = StandardScaler().fit(Xraw[idx["train"]])
    Xs = sx.transform(Xraw)
    sy = StandardScaler().fit(yd["train"])

    def block(split, stride):
        sel = idx[split]
        Xb, yb = Xs[sel], sy.transform(np.asarray(y)[sel])
        return _make_windows(Xb, yb, W, stride, winnorm)

    Xtr, ytr = block("train", 6)
    Xte, yte_w = block("test", 1)
    tte = th["test"][W::1][:len(yte_w)]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(Xtr.shape[2], hidden, batch_first=True)
            self.head = nn.Linear(hidden, 2)
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1])

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    nb = len(Xtr_t)
    for ep in range(epochs):
        perm = torch.randperm(nb)
        tot = 0.0
        net.train()
        for b in range(0, nb, 512):
            sel = perm[b:b + 512]
            xb, yb = Xtr_t[sel].to(dev), ytr_t[sel].to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(sel)
        if ep % 5 == 0:
            print(f"  {name} ep{ep} train MSE {tot/nb:.4f}", flush=True)
    net.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(Xte), 2048):
            xb = torch.tensor(Xte[b:b + 2048], dtype=torch.float32).to(dev)
            preds.append(net(xb).cpu().numpy())
    pred = sy.inverse_transform(np.concatenate(preds))
    true = sy.inverse_transform(yte_w)
    save_trial(name, eval_joint(pred, true), pred, true, tte)


def t7():
    _lstm_trial("t7_lstm_W60", W=60, winnorm=False)

def t8():
    _lstm_trial("t8_lstm_W360", W=360, winnorm=False)

def t9():
    _lstm_trial("t9_lstm_winnorm_W360", W=360, winnorm=True)

def t10():
    _lstm_trial("t10_lstm_winnorm_W720", W=720, winnorm=True, epochs=30)


if __name__ == "__main__":
    for nm, fn in [("t1", t1), ("t2", t2), ("t3", t3), ("t4", t4), ("t5", t5),
                   ("t6", t6), ("t7", t7), ("t8", t8), ("t9", t9), ("t10", t10)]:
        run(nm, fn)
    print("LADDER DONE", flush=True)
