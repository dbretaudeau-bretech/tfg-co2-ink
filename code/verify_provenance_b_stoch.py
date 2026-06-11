"""Reproducibility checks for the stochastic Table-II rows (MLP reg, LSTM 6-h,
capacity MLP small). Seeds 0..k, GPU."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import data4 as D

dev = "cuda" if torch.cuda.is_available() else "cpu"
X72, y, t_h, idx = D.load_light_fe()
Xd72, yd, th = D.split_xy(X72, y, t_h, idx)
yarr = np.asarray(y)
M = D.metrics

sx = StandardScaler().fit(Xd72["train"])
sy = StandardScaler().fit(yd["train"])
Xs = {k: torch.tensor(sx.transform(v), dtype=torch.float32) for k, v in Xd72.items()}
ys = {k: torch.tensor(sy.transform(v), dtype=torch.float32) for k, v in yd.items()}


def mlp_train(hidden, epochs, seed, dropout=0.0, wd=0.0, early_stop=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    layers, last = [], 72
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        if dropout:
            layers += [nn.Dropout(dropout)]
        last = h
    layers += [nn.Linear(last, 2)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.MSELoss()
    n = len(Xs["train"])
    best = (np.inf, None, -1)
    patience, bad = 10, 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for b in range(0, n, 1024):
            s = perm[b:b + 1024]
            xb, yb = Xs["train"][s].to(dev), ys["train"][s].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        if early_stop:
            net.eval()
            with torch.no_grad():
                pv = sy.inverse_transform(net(Xs["val"].to(dev)).cpu().numpy())
            vmae = np.abs(pv[:, 1] - yd["val"][:, 1]).mean()
            if vmae < best[0]:
                best = (vmae, {k: v.clone() for k, v in net.state_dict().items()}, ep)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if early_stop and best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()
    out = {}
    for sp in ("train", "test"):
        with torch.no_grad():
            p = sy.inverse_transform(net(Xs[sp].to(dev)).cpu().numpy())
        out[sp] = {"RH": M(p[:, 0], yd[sp][:, 0]), "CO2": M(p[:, 1], yd[sp][:, 1])}
    out["best_ep"] = best[2]
    return out


# ---- regularized MLP 128x64, 3 seeds ----------------------------------------
maes, rs = [], []
for s in (0, 1, 2):
    o = mlp_train([128, 64], 150, s, dropout=0.2, wd=1e-4, early_stop=True)
    print(f"MLPREG seed{s} ep{o['best_ep']} tr_RH={o['train']['RH']['mae']} "
          f"tr_CO2={o['train']['CO2']['mae']} te_RH={o['test']['RH']['mae']} "
          f"te_CO2={o['test']['CO2']['mae']} r={o['test']['CO2']['pearson']}", flush=True)
    maes.append(o["test"]["CO2"]["mae"]); rs.append(o["test"]["CO2"]["pearson"])
print(f"MLPREG agg CO2 te {np.mean(maes):.0f}±{np.std(maes):.0f} r={np.mean(rs):.2f}", flush=True)

# ---- capacity MLP small [64], 1 seed -----------------------------------------
o = mlp_train([64], 40, 0)
print(f"MLPSMALL seed0 tr_CO2={o['train']['CO2']['mae']} te_CO2={o['test']['CO2']['mae']}", flush=True)

# ---- coarse 6-h LSTM, 5 seeds -------------------------------------------------
Xraw = X72[D.INK_RAW].values
Xs_full = pd.DataFrame(Xraw).rolling(12, min_periods=1).mean().values
STRIDE = 12
sxr = StandardScaler().fit(Xs_full[idx["train"]])
Xn = sxr.transform(Xs_full)
yn = sy.transform(yarr)


def windows(split, stride_out, W_min=360):
    sel = idx[split]
    a, b = sel[0], sel[-1]
    need = W_min * STRIDE
    xs, ysl, ts = [], [], []
    for end in range(a + need, b + 1, stride_out):
        xs.append(Xn[end - need:end:STRIDE]); ysl.append(yn[end]); ts.append(t_h[end])
    return np.stack(xs), np.stack(ysl), np.array(ts)


Xtr, ytr, _ = windows("train", 6)
Xte, yte, _ = windows("test", 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(16, 64, batch_first=True)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])


res = {"tr_RH": [], "tr_CO2": [], "te_RH": [], "te_CO2": [], "r": []}
for s in range(5):
    torch.manual_seed(s); np.random.seed(s)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xtr_t)
    for ep in range(40):
        perm = torch.randperm(n); net.train()
        for b in range(0, n, 512):
            sl = perm[b:b + 512]
            xb, yb = Xtr_t[sl].to(dev), ytr_t[sl].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
    net.eval()

    def pred_arr(X):
        with torch.no_grad():
            ps = []
            for b in range(0, len(X), 4096):
                xb = torch.tensor(X[b:b + 4096], dtype=torch.float32).to(dev)
                ps.append(net(xb).cpu().numpy())
        return sy.inverse_transform(np.concatenate(ps))

    ptr = pred_arr(Xtr); ytr_o = sy.inverse_transform(ytr)
    pte = pred_arr(Xte); yte_o = sy.inverse_transform(yte)
    mtr_rh = M(ptr[:, 0], ytr_o[:, 0])["mae"]; mtr_co2 = M(ptr[:, 1], ytr_o[:, 1])["mae"]
    mte_rh = M(pte[:, 0], yte_o[:, 0])["mae"]; mte = M(pte[:, 1], yte_o[:, 1])
    print(f"LSTM6h seed{s} tr_RH={mtr_rh} tr_CO2={mtr_co2} te_RH={mte_rh} "
          f"te_CO2={mte['mae']} r={mte['pearson']}", flush=True)
    res["tr_RH"].append(mtr_rh); res["tr_CO2"].append(mtr_co2)
    res["te_RH"].append(mte_rh); res["te_CO2"].append(mte["mae"]); res["r"].append(mte["pearson"])

print(f"LSTM6h agg tr_RH={np.mean(res['tr_RH']):.2f} tr_CO2={np.mean(res['tr_CO2']):.0f} "
      f"te_RH={np.mean(res['te_RH']):.2f} te_CO2={np.mean(res['te_CO2']):.0f}"
      f"±{np.std(res['te_CO2']):.0f} r={np.mean(res['r']):.2f}±{np.std(res['r']):.2f}", flush=True)
print("DONE")
