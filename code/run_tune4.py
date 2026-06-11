"""TFGV4 L4 tuning — coarse-context LSTM, staged search on VALIDATION only.

Pipeline per config: MA(ma_s) smoothing of ink channels -> resample every
stride_s -> windows of (context_h*3600/stride_s) coarse steps -> LSTM ->
linear head -> target(s) at window end.

Differences vs run_ladder4.py (documented, honest):
- Input windows for val/test may extend LEFT across the split boundary
  (inputs only; the predicted target is always inside its split). This makes
  every config share the IDENTICAL val/test endpoint grid (every 60 s), so
  context lengths compete fairly and val/test cover the full 11.44 h blocks.
- Early stopping on val CO2 Pearson r (the primary selection metric),
  patience 6, max 60 epochs, best-checkpoint restore. Validation-only.
- Test metrics are computed for every run but NEVER looked at for selection;
  the driver only prints/uses val numbers until the final config is frozen.
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

import data4 as D

RES = D.RESULTS
TUNE_PATH = RES / "metrics4_tuning.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- data (once)
X72, y, t_h, idx = D.load_light_fe()
Xraw16 = X72[D.INK_RAW].values.astype(np.float64)
br17 = X72[["px_BR_mean"]].values.astype(np.float64)  # engineered B/R mean
yarr = np.asarray(y, dtype=np.float64)

VAL_ENDS = idx["val"][::12]    # every 60 s on the 5-s grid
TEST_ENDS = idx["test"][::12]
TRAIN_STRIDE = 6               # train endpoints every 30 s

_smooth_cache = {}


def get_smoothed(ma_s: int, use_br: bool) -> np.ndarray:
    key = (ma_s, use_br)
    if key not in _smooth_cache:
        base = np.concatenate([Xraw16, br17], axis=1) if use_br else Xraw16
        w = max(1, ma_s // 5)
        sm = pd.DataFrame(base).rolling(w, min_periods=1).mean().values
        _smooth_cache[key] = sm
    return _smooth_cache[key]


def build_windows(Xn, yn, ends, need, S):
    ends = ends[ends - need >= 0]
    xs = np.stack([Xn[e - need:e:S] for e in ends]).astype(np.float32)
    return xs, yn[ends].astype(np.float32), t_h[ends]


class Net(nn.Module):
    def __init__(self, nin, hidden, layers, dropout, nout):
        super().__init__()
        self.lstm = nn.LSTM(nin, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, nout)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(self.drop(o[:, -1]))


def run_config(cfg: dict, seed: int) -> dict:
    """Train one config; return val + test metrics (test stored, not used)."""
    t0 = time.time()
    ma_s, stride_s = cfg["ma_s"], cfg["stride_s"]
    ctx_h, hidden = cfg["context_h"], cfg["hidden"]
    layers, dropout, wd = cfg["layers"], cfg["dropout"], cfg["wd"]
    co2_only, use_br = cfg["co2_only"], cfg["use_br"]

    S = stride_s // 5
    steps = int(ctx_h * 3600 / stride_s)
    need = steps * S

    Xs = get_smoothed(ma_s, use_br)
    sx = StandardScaler().fit(Xs[idx["train"]])
    ycols = [1] if co2_only else [0, 1]
    sy = StandardScaler().fit(yarr[idx["train"]][:, ycols])
    Xn = sx.transform(Xs)
    yn = sy.transform(yarr[:, ycols])

    tr_ends = idx["train"][::TRAIN_STRIDE]
    Xtr, ytr, _ = build_windows(Xn, yn, tr_ends, need, S)
    Xva, yva, _ = build_windows(Xn, yn, VAL_ENDS, need, S)
    Xte, yte, tte = build_windows(Xn, yn, TEST_ENDS, need, S)

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(Xs.shape[1], hidden, layers, dropout, len(ycols)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva).to(DEV)
    yva_t = torch.tensor(yva).to(DEV)
    co2_i = len(ycols) - 1  # CO2 column index in the head output

    n = len(Xtr_t)
    yv_c = yva_t[:, co2_i]
    yv_c = (yv_c - yv_c.mean()) / yv_c.std()
    best_val, best_state, bad, best_ep = -np.inf, None, 0, 0
    for ep in range(60):
        perm = torch.randperm(n)
        net.train()
        for b in range(0, n, 512):
            s = perm[b:b + 512]
            xb, yb = Xtr_t[s].to(DEV), ytr_t[s].to(DEV)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(Xva_t[b:b + 4096]) for b in range(0, len(Xva_t), 4096)])
            pc = pv[:, co2_i]
            vr_ep = float(((pc - pc.mean()) / (pc.std() + 1e-8) * yv_c).mean())
        if vr_ep > best_val + 1e-5:
            best_val, bad, best_ep = vr_ep, 0, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state)
    net.eval()

    def pred(Xw):
        with torch.no_grad():
            ps = [net(torch.tensor(Xw[b:b + 4096]).to(DEV)).cpu().numpy()
                  for b in range(0, len(Xw), 4096)]
        return sy.inverse_transform(np.concatenate(ps))

    out = {"cfg": cfg, "seed": seed, "best_epoch": best_ep,
           "n_train": int(n), "n_val": len(Xva), "n_test": len(Xte),
           "sec": round(time.time() - t0, 1)}
    pv = pred(Xva)
    yv = sy.inverse_transform(yva)
    pt = pred(Xte)
    yt = sy.inverse_transform(yte)
    out["val"] = {"CO2": D.metrics(pv[:, co2_i], yv[:, co2_i])}
    out["test"] = {"CO2": D.metrics(pt[:, co2_i], yt[:, co2_i])}
    if not co2_only:
        out["val"]["RH"] = D.metrics(pv[:, 0], yv[:, 0])
        out["test"]["RH"] = D.metrics(pt[:, 0], yt[:, 0])
    out["_arrays"] = (pt, yt, tte)  # stripped before json
    return out


def log_run(name, res):
    allr = json.loads(TUNE_PATH.read_text()) if TUNE_PATH.exists() else {}
    slim = {k: v for k, v in res.items() if k != "_arrays"}
    allr[name] = slim
    TUNE_PATH.write_text(json.dumps(allr, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] {name} val_CO2_r={res['val']['CO2']['pearson']}"
          f" val_CO2_mae={res['val']['CO2']['mae']} ep={res['best_epoch']}"
          f" ({res['sec']}s)", flush=True)


BASE = dict(context_h=6, ma_s=60, stride_s=60, hidden=64, layers=1,
            dropout=0.0, wd=0.0, co2_only=False, use_br=False)


def vr(res):
    return res["val"]["CO2"]["pearson"]


def eval_cfg(name, cfg, seeds=(0, 1)):
    """Run cfg over seeds, log each, return mean val CO2 pearson."""
    rs = []
    for sd in seeds:
        res = run_config(cfg, sd)
        log_run(f"{name}_s{sd}", res)
        rs.append(vr(res))
    return float(np.mean(rs))


def main():
    summary = {}

    # -------- stage 1: context length
    best = dict(BASE)
    s1 = {}
    for ch in [2, 4, 6, 8]:
        c = dict(best, context_h=ch)
        s1[ch] = eval_cfg(f"s1_ctx{ch}h", c)
    best["context_h"] = max(s1, key=s1.get)
    summary["stage1_context"] = s1
    print(f"== stage1 best context: {best['context_h']}h  {s1}", flush=True)

    # -------- stage 2: MA window x resample stride
    s2 = {}
    for ma in [30, 60, 120]:
        for st in [30, 60]:
            c = dict(best, ma_s=ma, stride_s=st)
            s2[f"ma{ma}_st{st}"] = eval_cfg(f"s2_ma{ma}_st{st}", c)
    bk = max(s2, key=s2.get)
    best["ma_s"], best["stride_s"] = int(bk.split("_")[0][2:]), int(bk.split("_st")[1])
    summary["stage2_smooth"] = s2
    print(f"== stage2 best: {bk}  {s2}", flush=True)

    # -------- stage 3a: hidden size
    s3a = {}
    for h in [32, 64, 128]:
        s3a[h] = eval_cfg(f"s3a_h{h}", dict(best, hidden=h))
    best["hidden"] = max(s3a, key=s3a.get)
    summary["stage3a_hidden"] = s3a
    print(f"== stage3a best hidden: {best['hidden']}  {s3a}", flush=True)

    # -------- stage 3b: layers x dropout
    s3b = {}
    for ly in [1, 2]:
        for dr in [0.0, 0.1, 0.2]:
            if ly == best["layers"] and dr == best["dropout"]:
                key0 = f"L{ly}_d{dr}"
                s3b[key0] = s3a[best["hidden"]]  # reuse
                continue
            s3b[f"L{ly}_d{dr}"] = eval_cfg(f"s3b_L{ly}_d{dr}",
                                           dict(best, layers=ly, dropout=dr))
    bk = max(s3b, key=s3b.get)
    best["layers"] = int(bk[1])
    best["dropout"] = float(bk.split("_d")[1])
    summary["stage3b_layers_dropout"] = s3b
    print(f"== stage3b best: {bk}  {s3b}", flush=True)

    # -------- stage 3c: weight decay
    s3c = {0.0: s3b[bk]}
    s3c[1e-4] = eval_cfg("s3c_wd1e-4", dict(best, wd=1e-4))
    best["wd"] = max(s3c, key=s3c.get)
    summary["stage3c_wd"] = {str(k): v for k, v in s3c.items()}
    print(f"== stage3c best wd: {best['wd']}  {s3c}", flush=True)

    # -------- stage 4: head + 17th channel
    s4 = {"joint_16ch": s3c[best["wd"]]}
    s4["co2only_16ch"] = eval_cfg("s4_co2only", dict(best, co2_only=True))
    s4["joint_17ch"] = eval_cfg("s4_joint_br17", dict(best, use_br=True))
    s4["co2only_17ch"] = eval_cfg("s4_co2only_br17",
                                  dict(best, co2_only=True, use_br=True))
    bk = max(s4, key=s4.get)
    best["co2_only"] = bk.startswith("co2only")
    best["use_br"] = bk.endswith("17ch")
    summary["stage4_head_channels"] = s4
    print(f"== stage4 best: {bk}  {s4}", flush=True)

    # -------- final: 3 seeds, freeze, NOW look at test
    print(f"== FINAL CONFIG: {best}", flush=True)
    finals = []
    for sd in [0, 1, 2]:
        res = run_config(best, sd)
        log_run(f"final_s{sd}", res)
        finals.append(res)
    # best seed by VAL r -> save its test predictions
    bi = int(np.argmax([vr(r) for r in finals]))
    pt, yt, tte = finals[bi]["_arrays"]
    np.savez_compressed(RES / "preds4_L4_best.npz", pred=pt, true=yt, t_h=tte)

    def agg(split, tgt, key):
        v = [r[split][tgt][key] for r in finals]
        return f"{np.mean(v):.3f} ± {np.std(v):.3f} (vals {v})"

    summary["final_config"] = best
    summary["final_best_seed"] = bi
    summary["final_val_CO2_r"] = agg("val", "CO2", "pearson")
    summary["final_test"] = {
        "CO2": {k: agg("test", "CO2", k) for k in ("mae", "r2", "pearson")}}
    summary["final_val"] = {
        "CO2": {k: agg("val", "CO2", k) for k in ("mae", "r2", "pearson")}}
    if not best["co2_only"]:
        summary["final_test"]["RH"] = {k: agg("test", "RH", k)
                                       for k in ("mae", "r2", "pearson")}
        summary["final_val"]["RH"] = {k: agg("val", "RH", k)
                                      for k in ("mae", "r2", "pearson")}
    (RES / "tuning_summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY:", json.dumps(summary, indent=1), flush=True)
    print("TUNE4 DONE", flush=True)


if __name__ == "__main__":
    main()
