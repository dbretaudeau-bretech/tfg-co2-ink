"""IDEA 3 — modern sequence architectures: PatchTST-style transformer on the
champion LSTM's exact coarse grid.

Champion (idea-0 baseline, 3 seeds): 16 raw ink channels -> 60-s MA ->
resample 1/min -> window 360 steps (6 h) -> LSTM(64) -> linear (RH, CO2);
+ per-lamp-segment 30-min offset recal (run_ladder2.r2 protocol):
test CO2 MAE 47.7±1.2 ppm, R2 0.68. No-recal shape reference r 0.81±0.03.

This script keeps the ENTIRE pipeline identical (same smoothing, same coarse
grid, same 360x16 windows, same joint standardized targets, same Adam 1e-3,
same recal protocol) and swaps ONLY the sequence encoder for a channel-
independent PatchTST-style transformer:
  360x16 window -> non-overlapping patches of length P per channel ->
  linear embed P->d_model (shared across channels) + learnable pos enc ->
  N transformer encoder layers (pre-norm, GELU, ff=2*d_model) ->
  mean-pool over patches -> concat 16 channel embeddings -> linear -> (RH,CO2).

Honesty rules (same as ladder5):
- All selection on VALIDATION (early stop + config ranking on val loss =
  standardized joint MSE; val CO2 ranking is known-weak, noted in results).
- Test predictions stored per run; only the frozen final config's test is read.
- Input windows may extend LEFT across split boundaries (inputs only).
- Only test-side fitting: the standard 30-min per-segment offset recal.

Mamba: mamba-ssm NOT installed in this venv -> skipped (noted in results).
"""
import json
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

import data4 as D

RES = D.RESULTS
RAW = RES / "raw_idea3"
RAW.mkdir(exist_ok=True)
MPATH = RES / "idea3_transformer.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

STRIDE = 12            # 60-s coarse step on the 5-s grid
MA = 12                # 60-s moving average
W = 360                # window steps (6 h)
NEED = W * STRIDE      # window span in 5-s samples
CAL = int(30 * 60 / 5)  # 360 samples = 30-min recal window
TRAIN_STRIDE = 6       # train endpoints every 30 s

# ---------------------------------------------------------------- data (once)
df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
X72, y, t_h, idx = D.load_light_fe()
lamp = df["lamp_pct"].ffill().bfill()
scd = df["SCD_CO2"].values.astype(np.float64)
rh_true = df["BME_RH"].values.astype(np.float64)
yarr = np.column_stack([rh_true, scd])

Xraw = X72[D.INK_RAW].values
Xs_full = pd.DataFrame(Xraw).rolling(MA, min_periods=1).mean().values
sx = StandardScaler().fit(Xs_full[idx["train"]])
Xn = sx.transform(Xs_full).astype(np.float32)
sy = StandardScaler().fit(yarr[idx["train"]])
yn = sy.transform(yarr).astype(np.float32)


def _segments(lampcol):
    lampr = lampcol.round(0).values
    bounds = [0] + list(np.where(np.diff(lampr) != 0)[0] + 1) + [len(lampr)]
    seg_id = np.zeros(len(lampr), dtype=int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg_id[a:b] = s
    return seg_id


seg_all = _segments(lamp)


def protocol_eval(pred_split, split):
    """EXACT champion protocol (run_ladder2.r2): per-lamp-segment 30-min
    offset recal, cal samples excluded from metrics. Verified to reproduce
    the stored champion used-mask bit-for-bit."""
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
    m["used_frac"] = round(float(used.mean()), 3)
    return m, pred_cal, used


def log(name, info):
    allm = json.loads(MPATH.read_text()) if MPATH.exists() else {}
    allm[name] = info
    MPATH.write_text(json.dumps(allm, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps(info)[:400]}",
          flush=True)


def build_windows(ends):
    ends = ends[ends - NEED >= 0]
    xs = np.stack([Xn[e - NEED:e:STRIDE] for e in ends])
    return xs, ends


# ---------------------------------------------------------------- model
class PatchTST(nn.Module):
    """Channel-independent PatchTST-style encoder for window regression."""

    def __init__(self, n_ch=16, seq_len=W, patch=24, d_model=128, depth=3,
                 heads=8, dropout=0.1, n_out=2):
        super().__init__()
        assert seq_len % patch == 0
        self.patch = patch
        self.n_patches = seq_len // patch
        self.n_ch = n_ch
        self.d_model = d_model
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_ch * d_model, n_out)

    def forward(self, x):                      # x: B, L, C
        B, L, C = x.shape
        z = x.permute(0, 2, 1).reshape(B * C, self.n_patches, self.patch)
        z = self.embed(z) + self.pos           # B*C, N, d
        z = self.encoder(z)
        z = self.norm(z).mean(dim=1)           # B*C, d
        z = z.reshape(B, C * self.d_model)
        return self.head(self.drop(z))


# ---------------------------------------------------------------- train one
def run_cfg(cfg, seed=0, eval_test=False):
    t0 = time.time()
    name = cfg["name"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ends_all = idx["train"][::cfg.get("train_stride", TRAIN_STRIDE)]
    Xtr, tr_ends = build_windows(tr_ends_all)
    Xva, va_ends = build_windows(idx["val"])    # every 5 s, full block
    Xte, te_ends = build_windows(idx["test"])
    assert len(va_ends) == len(idx["val"]) and len(te_ends) == len(idx["test"])
    ytr = yn[tr_ends]
    yva = yn[va_ends]

    net = PatchTST(patch=cfg.get("patch", 24), d_model=cfg.get("d_model", 128),
                   depth=cfg.get("depth", 3), heads=cfg.get("heads", 8),
                   dropout=cfg.get("dropout", 0.1)).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 1e-3))
    lossf = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva)
    yva_t = torch.tensor(yva).to(DEV)

    def predict(Xw_t, bs=1024):
        net.eval()
        with torch.no_grad():
            ps = [net(Xw_t[b:b + bs].to(DEV)).cpu().numpy()
                  for b in range(0, len(Xw_t), bs)]
        return np.concatenate(ps)

    def val_loss():
        net.eval()
        with torch.no_grad():
            tot, cnt = 0.0, 0
            for b in range(0, len(Xva_t), 1024):
                xb = Xva_t[b:b + 1024].to(DEV)
                yb = yva_t[b:b + 1024]
                tot += nn.functional.mse_loss(
                    net(xb), yb, reduction="sum").item()
                cnt += yb.numel()
        return tot / cnt

    n = len(Xtr_t)
    BS = cfg.get("bs", 256)
    best = (np.inf, None, 0)
    bad = 0
    for ep in range(cfg.get("epochs", 60)):
        perm = torch.randperm(n)
        net.train()
        for b in range(0, n, BS):
            s = perm[b:b + BS]
            xb, yb = Xtr_t[s].to(DEV), ytr_t[s].to(DEV)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
        vl = val_loss()
        if vl < best[0] - 1e-5:
            best = (vl, {k: v.detach().clone()
                         for k, v in net.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
            if bad >= cfg.get("patience", 8):
                break
    net.load_state_dict(best[1])

    pv = sy.inverse_transform(predict(Xva_t))
    yva_o = yarr[va_ends]
    mval = {"loss": round(best[0], 5),
            "RH": D.metrics(pv[:, 0], yva_o[:, 0]),
            "CO2_norecal": D.metrics(pv[:, 1], yva_o[:, 1])}
    mvp, _, _ = protocol_eval(pv[:, 1].astype(np.float64), "val")
    mval["CO2_recal"] = mvp

    pt = sy.inverse_transform(predict(torch.tensor(Xte)))
    np.savez_compressed(RAW / f"{name}_s{seed}.npz", val=pv, test=pt)

    info = {"cfg": cfg, "seed": seed, "n_params": n_par,
            "best_epoch": best[2], "sec": round(time.time() - t0, 1),
            "val": mval}
    if eval_test:                              # only for the frozen final
        yte_o = yarr[te_ends]
        mtest = {"RH": D.metrics(pt[:, 0], yte_o[:, 0]),
                 "CO2_norecal": D.metrics(pt[:, 1], yte_o[:, 1])}
        mtp, pred_cal, used = protocol_eval(pt[:, 1].astype(np.float64), "test")
        mtest["CO2_recal"] = mtp
        info["test"] = mtest
        np.savez_compressed(RAW / f"{name}_s{seed}_testcal.npz",
                            pred=pred_cal, true=scd[idx["test"]],
                            t_h=t_h[idx["test"]], used=used,
                            raw=pt[:, 1])
    log(f"{name}_s{seed}", info)
    return info


# ---------------------------------------------------------------- stages
GRID = [
    dict(name="P1_p24_d128_L3_dr01"),                              # base
    dict(name="P2_p12_d128_L3_dr01", patch=12),
    dict(name="P3_p30_d128_L3_dr01", patch=30),
    dict(name="P4_p24_d64_L3_dr01", d_model=64, heads=4),
    dict(name="P5_p24_d128_L2_dr01", depth=2),
    dict(name="P6_p24_d128_L3_dr02", dropout=0.2),
    dict(name="P7_p24_d64_L2_dr02", d_model=64, heads=4, depth=2, dropout=0.2),
    dict(name="P8_p12_d64_L2_dr01", patch=12, d_model=64, heads=4, depth=2),
]


def main(stage):
    if stage == "grid":
        for cfg in GRID:
            try:
                run_cfg(cfg, seed=0)
            except Exception:
                print(f"!! {cfg['name']} FAILED:\n{traceback.format_exc()}",
                      flush=True)
                log(cfg["name"], {"error": "failed"})
        print("IDEA3 GRID DONE", flush=True)
    elif stage == "seeds":
        # multi-seed val for candidate configs; test preds stored, NOT evaluated
        for name in sys.argv[2:]:
            cfg = {c["name"]: c for c in GRID}[name]
            cfg = dict(cfg, name=f"S_{name}")
            for sd in (0, 1, 2):
                try:
                    run_cfg(cfg, seed=sd, eval_test=False)
                except Exception:
                    print(f"!! {name} seed {sd} FAILED:\n"
                          f"{traceback.format_exc()}", flush=True)
        print("IDEA3 SEEDS DONE", flush=True)
    elif stage == "final":
        name = sys.argv[2]
        cfg = {c["name"]: c for c in GRID}[name]
        cfg = dict(cfg, name=f"FINAL_{name}")
        for sd in (0, 1, 2):
            try:
                run_cfg(cfg, seed=sd, eval_test=True)
            except Exception:
                print(f"!! seed {sd} FAILED:\n{traceback.format_exc()}",
                      flush=True)
        print("IDEA3 FINAL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid")
