"""Full-timeline predictions (train+val+test, whole 77 h grid) for every
Table-II rung, for the REPORT-style all-models figure.
Saves results/full_<name>.npz with pred (CO2), t_h; targets/splits implicit.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

import data4 as D

X72, y, t_h, idx = D.load_light_fe()
yarr = np.asarray(y)
RES = D.RESULTS


def save(name, pred_full):
    np.savez_compressed(RES / f"full_{name}.npz", pred=pred_full, t_h=t_h)
    m_tr = D.metrics(pred_full[idx["train"]], yarr[idx["train"], 1])
    m_te = D.metrics(pred_full[idx["test"]], yarr[idx["test"], 1])
    print(f"{name}: train MAE {m_tr['mae']:.0f}  test MAE {m_te['mae']:.0f}", flush=True)


def fit_linear(X, model, name):
    sx = StandardScaler().fit(np.asarray(X)[idx["train"]])
    model.fit(sx.transform(np.asarray(X)[idx["train"]]), yarr[idx["train"]])
    save(name, model.predict(sx.transform(np.asarray(X)))[:, 1])


fit_linear(X72[D.INK_RAW], LinearRegression(), "ols16")
fit_linear(X72[[c for c in X72.columns if c.startswith("OP0_")]], Ridge(alpha=100.0), "ridge1pix")
fit_linear(X72, Ridge(alpha=1000.0), "ridge72")
Xhp4 = pd.concat([X72, D.add_highpass(X72, 4.0)], axis=1)
fit_linear(Xhp4, Ridge(alpha=100.0), "ridge_hp4")
Xhp8 = pd.concat([X72, D.add_highpass(X72, 8.0)], axis=1)
fit_linear(Xhp8, Ridge(alpha=10.0), "ridge_hp8")
fit_linear(X72, KNeighborsRegressor(n_neighbors=5, n_jobs=16), "knn5")

# physics model
df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
br = pd.concat([df[f"OP{i}_blue"]/df[f"OP{i}_red"] for i in range(4)], axis=1).mean(axis=1).values
rh = df["BME_RH"].ffill().bfill().values
F = np.column_stack([br, rh, np.ones(len(br))])
coef, *_ = np.linalg.lstsq(F[idx["train"]], yarr[idx["train"], 1], rcond=None)
save("physics", F @ coef)

# ---- torch models ----
import torch
import torch.nn as nn
dev = "cuda"
sx = StandardScaler().fit(X72.values[idx["train"]])
sy = StandardScaler().fit(yarr[idx["train"]])
Xs = sx.transform(X72.values)


def mlp(hidden, epochs, name, seed=0):
    torch.manual_seed(seed)
    layers, last = [], 72
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]; last = h
    layers += [nn.Linear(last, 2)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.MSELoss()
    Xt = torch.tensor(Xs[idx["train"]], dtype=torch.float32)
    yt = torch.tensor(sy.transform(yarr[idx["train"]]), dtype=torch.float32)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n); net.train()
        for b in range(0, n, 1024):
            s = perm[b:b+1024]
            xb, yb = Xt[s].to(dev), yt[s].to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
    net.eval(); ps = []
    with torch.no_grad():
        for b in range(0, len(Xs), 8192):
            ps.append(net(torch.tensor(Xs[b:b+8192], dtype=torch.float32).to(dev)).cpu().numpy())
    save(name, sy.inverse_transform(np.concatenate(ps))[:, 1])


mlp([64], 40, "mlp64")
mlp([1024, 1024, 1024], 150, "mlp1024")

# LSTMs: raw 30-min and coarse 6-h, predictions over the FULL grid
Xraw = X72[D.INK_RAW].values
sxr = StandardScaler().fit(Xraw[idx["train"]])


def lstm_full(name, coarse, W, epochs=40, seed=2):
    torch.manual_seed(seed); np.random.seed(seed)
    if coarse:
        Xn = sxr.transform(pd.DataFrame(Xraw).rolling(12, min_periods=1).mean().values)
        STRIDE = 12
    else:
        Xn = sxr.transform(Xraw); STRIDE = 1
    need = W * STRIDE
    yn = sy.transform(yarr)

    def windows(sel, so):
        a, b = sel[0], sel[-1]
        xs, ys, ends = [], [], []
        for end in range(a + need, b + 1, so):
            xs.append(Xn[end-need:end:STRIDE]); ys.append(yn[end]); ends.append(end)
        return np.stack(xs), np.stack(ys), np.array(ends)

    Xtr, ytr, _ = windows(idx["train"], 6)

    class Net(nn.Module):
        def __init__(s2):
            super().__init__(); s2.l = nn.LSTM(16, 64, batch_first=True); s2.h = nn.Linear(64, 2)
        def forward(s2, x):
            o, _ = s2.l(x); return s2.h(o[:, -1])

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n); net.train()
        for b in range(0, n, 512):
            s = perm[b:b+512]
            xb, yb = Xt[s].to(dev), yt[s].to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
    net.eval()
    # predict over full grid (stride 12 for speed, then interp)
    pred_full = np.full(len(Xn), np.nan)
    ends_all = np.arange(need, len(Xn), 12)
    xs = np.stack([Xn[e-need:e:STRIDE] for e in ends_all])
    ps = []
    with torch.no_grad():
        for b in range(0, len(xs), 2048):
            ps.append(net(torch.tensor(xs[b:b+2048], dtype=torch.float32).to(dev)).cpu().numpy())
    p = sy.inverse_transform(np.concatenate(ps))[:, 1]
    pred_full[ends_all] = p
    pred_full = pd.Series(pred_full).interpolate(limit_area="inside").values
    save(name, pred_full)


lstm_full("lstm30min", coarse=False, W=360, epochs=25)
lstm_full("lstm6h", coarse=True, W=360, epochs=40)
print("ALL FULL PREDS DONE", flush=True)
