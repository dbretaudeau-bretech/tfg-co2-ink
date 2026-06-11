"""TFGV4 ladder round 3 — model zoo with train/val/test triples.

Goal: a dense overfitting/generalization story. Every trial records metrics on
ALL THREE splits so the train-test gap is quotable. Includes the random-split
leakage demo and a capacity ladder of MLPs (small/medium/big, big deliberately
unregularised).
Results → results/metrics3.json, preds → results/preds3_<name>.npz
"""
import json
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import data4 as D

RES = D.RESULTS
MPATH = RES / "metrics3.json"


def save(name, info, pred_test=None, true_test=None, t_test=None):
    all_m = json.loads(MPATH.read_text()) if MPATH.exists() else {}
    all_m[name] = info
    MPATH.write_text(json.dumps(all_m, indent=1))
    if pred_test is not None:
        np.savez_compressed(RES / f"preds3_{name}.npz",
                            pred=pred_test, true=true_test, t_h=t_test)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps(info)[:300]}", flush=True)


def triple(model_predict, Xd, yd):
    out = {}
    for sp in ("train", "val", "test"):
        p = model_predict(Xd[sp])
        out[sp] = {"RH": D.metrics(p[:, 0], yd[sp][:, 0]),
                   "CO2": D.metrics(p[:, 1], yd[sp][:, 1])}
    return out


X72, y, t_h, idx = D.load_light_fe()
Xd72, yd, th = D.split_xy(X72, y, t_h, idx)

Xhp4 = pd.concat([X72, D.add_highpass(X72, 4.0)], axis=1)
Xd_hp4, _, _ = D.split_xy(Xhp4, y, t_h, idx)


def run(name, fn):
    try:
        fn()
    except Exception:
        print(f"!! {name} FAILED:\n{traceback.format_exc()}", flush=True)
        save(name, {"error": "failed"})


def fit_scaled(Xd, model):
    sx = StandardScaler().fit(Xd["train"])
    model.fit(sx.transform(Xd["train"]), yd["train"])
    return lambda Xs: model.predict(sx.transform(Xs))


# ---------------- linear family ----------------
def t20():
    Xraw = X72[D.INK_RAW]
    Xd, _, _ = D.split_xy(Xraw, y, t_h, idx)
    pred = fit_scaled(Xd, LinearRegression())
    info = triple(pred, Xd, yd); info["params"] = 17 * 2
    save("t20_ols16", info, pred(Xd["test"]), yd["test"], th["test"])


def t21():
    best = None
    sx = StandardScaler().fit(Xd72["train"])
    for a in (10.0, 100.0, 1000.0):
        m = Ridge(alpha=a).fit(sx.transform(Xd72["train"]), yd["train"])
        mv = np.abs(m.predict(sx.transform(Xd72["val"]))[:, 1] - yd["val"][:, 1]).mean()
        if best is None or mv < best[0]:
            best = (mv, a, m)
    _, a, m = best
    pred = lambda Xs: m.predict(sx.transform(Xs))
    info = triple(pred, Xd72, yd); info["alpha"] = a; info["params"] = 73 * 2
    save("t21_ridge72", info, pred(Xd72["test"]), yd["test"], th["test"])


def t22():
    best = None
    sx = StandardScaler().fit(Xd_hp4["train"])
    for a in (10.0, 100.0, 1000.0):
        m = Ridge(alpha=a).fit(sx.transform(Xd_hp4["train"]), yd["train"])
        mv = np.abs(m.predict(sx.transform(Xd_hp4["val"]))[:, 1] - yd["val"][:, 1]).mean()
        if best is None or mv < best[0]:
            best = (mv, a, m)
    _, a, m = best
    pred = lambda Xs: m.predict(sx.transform(Xs))
    info = triple(pred, Xd_hp4, yd); info["alpha"] = a; info["params"] = Xd_hp4["train"].shape[1] * 2
    save("t22_ridge_hp4", info, pred(Xd_hp4["test"]), yd["test"], th["test"])


def t32():
    cols = [c for c in X72.columns if c.startswith("OP0_")]
    X1 = X72[cols]
    Xd, _, _ = D.split_xy(X1, y, t_h, idx)
    sx = StandardScaler().fit(Xd["train"])
    m = Ridge(alpha=100.0).fit(sx.transform(Xd["train"]), yd["train"])
    pred = lambda Xs: m.predict(sx.transform(Xs))
    info = triple(pred, Xd, yd); info["note"] = "single pixel (OP0 raw+ratios)"
    save("t32_ridge_1pix", info, pred(Xd["test"]), yd["test"], th["test"])


# ---------------- memorizers / trees ----------------
def t23():
    for k in (5, 50):
        sx = StandardScaler().fit(Xd72["train"])
        m = KNeighborsRegressor(n_neighbors=k, n_jobs=16).fit(
            sx.transform(Xd72["train"]), yd["train"])
        pred = lambda Xs: m.predict(sx.transform(Xs))
        info = triple(pred, Xd72, yd); info["k"] = k
        save(f"t23_knn{k}", info, pred(Xd72["test"]), yd["test"], th["test"])


def t24():
    m = RandomForestRegressor(n_estimators=200, n_jobs=24, random_state=0,
                              max_samples=0.5)
    m.fit(Xd72["train"], yd["train"])
    pred = lambda Xs: m.predict(Xs)
    info = triple(pred, Xd72, yd)
    save("t24_rf200", info, pred(Xd72["test"]), yd["test"], th["test"])


def t25():
    ms = []
    for j in range(2):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          random_state=0)
        m.fit(Xd_hp4["train"], yd["train"][:, j])
        ms.append(m)
    pred = lambda Xs: np.column_stack([m.predict(Xs) for m in ms])
    info = triple(pred, Xd_hp4, yd)
    save("t25_hgb_hp4", info, pred(Xd_hp4["test"]), yd["test"], th["test"])


# ---------------- MLP capacity ladder (torch) ----------------
def _mlp(name, hidden, epochs, dropout=0.0, wd=0.0):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sx = StandardScaler().fit(Xd72["train"])
    sy = StandardScaler().fit(yd["train"])
    Xs = {k: torch.tensor(sx.transform(v), dtype=torch.float32) for k, v in Xd72.items()}
    ys = {k: torch.tensor(sy.transform(v), dtype=torch.float32) for k, v in yd.items()}

    layers, last = [], 72
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        if dropout: layers += [nn.Dropout(dropout)]
        last = h
    layers += [nn.Linear(last, 2)]
    net = nn.Sequential(*layers).to(dev)
    nparams = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.MSELoss()
    n = len(Xs["train"])
    for ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for b in range(0, n, 1024):
            s = perm[b:b + 1024]
            xb, yb = Xs["train"][s].to(dev), ys["train"][s].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
    net.eval()

    def pred(label):
        with torch.no_grad():
            ps = []
            for b in range(0, len(Xs[label]), 4096):
                ps.append(net(Xs[label][b:b + 4096].to(dev)).cpu().numpy())
        return sy.inverse_transform(np.concatenate(ps))

    info = {}
    for sp in ("train", "val", "test"):
        p = pred(sp)
        info[sp] = {"RH": D.metrics(p[:, 0], yd[sp][:, 0]),
                    "CO2": D.metrics(p[:, 1], yd[sp][:, 1])}
    info["params"] = nparams
    save(name, info, pred("test"), yd["test"], th["test"])


def t26(): _mlp("t26_mlp_small", [64], epochs=40)
def t27(): _mlp("t27_mlp_med", [256, 256], epochs=60)
def t28(): _mlp("t28_mlp_big_overfit", [1024, 1024, 1024], epochs=150)


# ---------------- random-split leakage demo ----------------
def t31():
    rng = np.random.RandomState(0)
    n = len(X72)
    perm = rng.permutation(n)
    ridx = {"train": np.sort(perm[:int(0.7 * n)]),
            "val": np.sort(perm[int(0.7 * n):int(0.85 * n)]),
            "test": np.sort(perm[int(0.85 * n):])}
    Xd, yd_r, th_r = D.split_xy(X72, y, t_h, ridx)
    sx = StandardScaler().fit(Xd["train"])
    m = Ridge(alpha=100.0).fit(sx.transform(Xd["train"]), yd_r["train"])
    info = {}
    for sp in ("train", "test"):
        p = m.predict(sx.transform(Xd[sp]))
        info[sp] = {"RH": D.metrics(p[:, 0], yd_r[sp][:, 0]),
                    "CO2": D.metrics(p[:, 1], yd_r[sp][:, 1])}
    info["note"] = "RANDOM split — leakage demo, numbers are fake-good"
    save("t31_ridge_RANDOMSPLIT", info)


if __name__ == "__main__":
    for nm, fn in [("t20", t20), ("t21", t21), ("t22", t22), ("t32", t32),
                   ("t23", t23), ("t24", t24), ("t25", t25),
                   ("t26", t26), ("t27", t27), ("t28", t28), ("t31", t31)]:
        run(nm, fn)
    print("LADDER3 DONE", flush=True)
