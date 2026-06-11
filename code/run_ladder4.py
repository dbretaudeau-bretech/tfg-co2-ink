"""TFGV4 ladder round 4 — Daniel's idea: smooth + downsample so the LSTM sees
HOURS of context per window.

Pipeline: 60-s moving average of the 16 raw channels -> sample every 60 s ->
LSTM windows of W steps (= W minutes of context). Predict the target at the
window's last instant. Chronological splits preserved.
"""
import json
import time
import traceback
import numpy as np
from sklearn.preprocessing import StandardScaler

import data4 as D

RES = D.RESULTS
MPATH = RES / "metrics4.json"


def save(name, info, pred=None, true=None, t=None):
    all_m = json.loads(MPATH.read_text()) if MPATH.exists() else {}
    all_m[name] = info
    MPATH.write_text(json.dumps(all_m, indent=1))
    if pred is not None:
        np.savez_compressed(RES / f"preds4_{name}.npz", pred=pred, true=true, t_h=t)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps(info)[:300]}", flush=True)


X72, y, t_h, idx = D.load_light_fe()
Xraw = X72[D.INK_RAW].values
yarr = np.asarray(y)

# 60-s moving average then 60-s sampling (stride 12 on the 5-s grid)
import pandas as pd
MA = 12
Xs_full = pd.DataFrame(Xraw).rolling(MA, min_periods=1).mean().values
STRIDE = 12


def lstm_coarse(name, W_min, hidden=64, epochs=40):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sx = StandardScaler().fit(Xs_full[idx["train"]])
    sy = StandardScaler().fit(yarr[idx["train"]])
    Xn = sx.transform(Xs_full)
    yn = sy.transform(yarr)

    def windows(split, stride_out):
        sel = idx[split]
        a, b = sel[0], sel[-1]
        xs, ys, ts = [], [], []
        # coarse grid inside the split: every STRIDE samples
        need = W_min * STRIDE  # span in 5-s samples
        for end in range(a + need, b + 1, stride_out):
            w = Xn[end - need:end:STRIDE]  # W_min coarse steps
            xs.append(w); ys.append(yn[end]); ts.append(t_h[end])
        return np.stack(xs), np.stack(ys), np.array(ts)

    Xtr, ytr, _ = windows("train", 6)
    Xte, yte, tte = windows("test", 1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(16, hidden, batch_first=True)
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
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
    net.eval()

    def pred_arr(X):
        with torch.no_grad():
            ps = []
            for b in range(0, len(X), 4096):
                xb = torch.tensor(X[b:b + 4096], dtype=torch.float32).to(dev)
                ps.append(net(xb).cpu().numpy())
        return sy.inverse_transform(np.concatenate(ps))

    info = {}
    ptr = pred_arr(Xtr); ytr_o = sy.inverse_transform(ytr)
    info["train"] = {"RH": D.metrics(ptr[:, 0], ytr_o[:, 0]),
                     "CO2": D.metrics(ptr[:, 1], ytr_o[:, 1])}
    pte = pred_arr(Xte); yte_o = sy.inverse_transform(yte)
    info["test"] = {"RH": D.metrics(pte[:, 0], yte_o[:, 0]),
                    "CO2": D.metrics(pte[:, 1], yte_o[:, 1])}
    info["context_h"] = W_min / 60
    save(name, info, pte, yte_o, tte)


if __name__ == "__main__":
    for nm, W in [("L4_lstm_ma60_W120", 120),   # 2 h context
                  ("L4_lstm_ma60_W360", 360),   # 6 h context
                  ("L4_lstm_ma60_W720", 720)]:  # 12 h context
        try:
            lstm_coarse(nm, W)
        except Exception:
            print(f"!! {nm} FAILED:\n{traceback.format_exc()}", flush=True)
            save(nm, {"error": "failed"})
    print("LADDER4 DONE", flush=True)
