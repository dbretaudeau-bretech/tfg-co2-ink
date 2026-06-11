"""TFGV4 ladder round 5 — recurrent models under the CHAMPION's evaluation
protocol (per-lamp-segment 30-min offset recalibration, calibration samples
excluded; reference implementation: run_ladder2.py r2()).

Champion to beat (test, used windows): MAE 53.65 / R2 0.645 / r 0.822.

Honesty rules:
- All selection on VALIDATION (same recal protocol applied to val segments).
- Test raw predictions are computed and stored per run but never read until
  the single final winner is frozen.
- Input windows may extend LEFT across split boundaries (input channels only;
  the device has its own continuous sensor history). Targets always inside
  their split. Measured BME_RH is a legitimate input: the champion itself
  uses it.
- The only test-side fitting is the champion's own 30-min per-segment offset.
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
RAW = RES / "raw5"
RAW.mkdir(exist_ok=True)
MPATH = RES / "metrics5_rnn.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STRIDE = 12          # coarse step = 60 s on the 5-s grid
MA = 12              # 60-s moving average
CAL = int(30 * 60 / 5)  # 360 samples = 30 min calibration window
TRAIN_STRIDE = 6     # train endpoints every 30 s

# ---------------------------------------------------------------- data (once)
df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
X72, y, t_h, idx = D.load_light_fe()
aux = df[["lamp_pct", "BME_RH", "BME_T"]].reset_index(drop=True).ffill().bfill()
scd = df["SCD_CO2"].values.astype(np.float64)
rh_true = df["BME_RH"].values.astype(np.float64)
yarr = np.asarray(y, dtype=np.float64)


def _segments(lamp):
    lampr = lamp.round(0).values
    bounds = [0] + list(np.where(np.diff(lampr) != 0)[0] + 1) + [len(lampr)]
    seg_id = np.zeros(len(lampr), dtype=int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg_id[a:b] = s
    return seg_id


seg_all = _segments(aux["lamp_pct"])

# physics channel: B/R + RH linear, coefficients from TRAIN only (= champion)
_br = pd.concat([df[f"OP{i}_blue"] / df[f"OP{i}_red"] for i in range(4)],
                axis=1).mean(axis=1).values
_tr = idx["train"]
_A = np.column_stack([_br, aux["BME_RH"].values, np.ones(len(df))])
_coef, *_ = np.linalg.lstsq(_A[_tr], scd[_tr], rcond=None)
phys = (_A @ _coef).astype(np.float64)

# channel bank (smoothed once)
BANK = np.column_stack([
    X72[D.INK_RAW].values,                                   # 0-15
    aux.values,                                              # 16 lamp 17 RH 18 T
    X72[["px_BR_mean", "px_IR_R_mean",
         "px_B_RGB_mean", "px_IR_RGB_mean"]].values,         # 19-22
    phys,                                                    # 23
]).astype(np.float64)
BANK_S = pd.DataFrame(BANK).rolling(MA, min_periods=1).mean().values

CH = {
    "raw16": list(range(16)),
    "devreal": list(range(19)),
    "devreal_eng": list(range(23)),
    "devreal_eng_phys": list(range(24)),
    "eng_rh": [19, 20, 21, 22, 17],
    "raw16_rh": list(range(16)) + [17],
}

# anchor features (mode 'anchor'): time-since-segment-start (h) and mean true
# CO2 over the segment's first 30 min — exactly the info the champion's
# recalibration consumes.
anch_t = np.zeros(len(df))
anch_c = np.zeros(len(df))
for s in np.unique(seg_all):
    rows = np.where(seg_all == s)[0]
    cal = rows[:min(len(rows), CAL)]
    anch_t[rows] = (rows - rows[0]) * 5.0 / 3600.0
    anch_c[rows] = scd[cal].mean()
ANCH = np.column_stack([anch_t, anch_c])

# smoothed-derivative target (ppm/s) for 'deriv' mode (window configurable)
_dscd_cache = {}


def get_dscd(smooth_s: int) -> np.ndarray:
    if smooth_s not in _dscd_cache:
        w = max(1, smooth_s // 5)
        sm = pd.Series(scd).rolling(w, center=True, min_periods=1).mean().values
        _dscd_cache[smooth_s] = np.gradient(sm, 5.0)
    return _dscd_cache[smooth_s]


dscd = get_dscd(300)


# ---------------------------------------------------------------- protocol
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
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps(info)[:400]}",
          flush=True)


# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, nin, hidden, layers, nout, arch="lstm", n_extra=0,
                 dropout=0.0):
        super().__init__()
        rnn = nn.LSTM if arch == "lstm" else nn.GRU
        self.rnn = rnn(nin, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden + n_extra, nout)

    def forward(self, x, extra=None):
        o, _ = self.rnn(x)
        h = o[:, -1]
        if extra is not None:
            h = torch.cat([h, extra], dim=1)
        return self.head(h)


def build_windows(Xn, ends, need):
    ends = ends[ends - need >= 0]
    xs = np.stack([Xn[e - need:e:STRIDE] for e in ends]).astype(np.float32)
    return xs, ends


def run_cfg(cfg, seed=0):
    """Train one config. Returns dict with val protocol metrics; raw val+test
    predictions go to results/raw5/<name>_s<seed>.npz (test never read here)."""
    t0 = time.time()
    name = cfg["name"]
    chans = CH[cfg["channels"]]
    mode = cfg["mode"]                      # abs | abs_joint | resid | deriv | anchor
    ctx_h = cfg.get("ctx_h", 6)
    hidden = cfg.get("hidden", 64)
    layers = cfg.get("layers", 1)
    arch = cfg.get("arch", "lstm")
    loss_name = cfg.get("loss", "mse")
    lr = cfg.get("lr", 1e-3)
    dropout = cfg.get("dropout", 0.0)

    steps = int(ctx_h * 3600 / (STRIDE * 5))
    need = steps * STRIDE

    Xs = BANK_S[:, chans]
    sx = StandardScaler().fit(Xs[idx["train"]])
    Xn = sx.transform(Xs)

    # targets
    if mode == "abs_joint":
        ycols = np.column_stack([rh_true, scd])
        co2_i = 1
    elif mode in ("abs", "anchor"):
        ycols = scd[:, None]
        co2_i = 0
    elif mode == "resid":
        ycols = (scd - phys)[:, None]
        co2_i = 0
    elif mode == "deriv":
        ycols = get_dscd(cfg.get("dsm", 300))[:, None]
        co2_i = 0
    elif mode == "danchor":
        # displacement from the segment's 30-min anchor — bounded error,
        # no integration drift; anchor features fed to the head
        ycols = (scd - anch_c)[:, None]
        co2_i = 0
    sy = StandardScaler().fit(ycols[idx["train"]])
    yn = sy.transform(ycols)

    n_extra = 2 if mode in ("anchor", "danchor") else 0
    if n_extra:
        sa = StandardScaler().fit(ANCH[idx["train"]])
        An = sa.transform(ANCH).astype(np.float32)

    tr_ends_all = idx["train"][::cfg.get("train_stride", TRAIN_STRIDE)]
    Xtr, tr_ends = build_windows(Xn, tr_ends_all, need)
    Xva, va_ends = build_windows(Xn, idx["val"], need)   # every 5 s, full block
    Xte, te_ends = build_windows(Xn, idx["test"], need)
    assert len(va_ends) == len(idx["val"]) and len(te_ends) == len(idx["test"])

    ytr = yn[tr_ends].astype(np.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(len(chans), hidden, layers, ytr.shape[1], arch, n_extra,
              dropout).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr,
                           weight_decay=cfg.get("wd", 0.0))
    lossf = nn.HuberLoss(delta=1.0) if loss_name == "huber" else nn.MSELoss()

    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva).to(DEV)
    Etr = torch.tensor(An[tr_ends]) if n_extra else None
    Eva = torch.tensor(An[va_ends]).to(DEV) if n_extra else None
    Ete = torch.tensor(An[te_ends]) if n_extra else None

    def predict(Xw_t, Ew, bs=1024):
        with torch.no_grad():
            ps = []
            for b in range(0, len(Xw_t), bs):
                xb = Xw_t[b:b + bs]
                xb = xb.to(DEV) if xb.device.type == "cpu" else xb
                eb = None if Ew is None else Ew[b:b + bs].to(DEV)
                ps.append(net(xb, eb).cpu().numpy())
        return sy.inverse_transform(np.concatenate(ps))

    def to_co2(pred_col, ends, split):
        """convert model output to absolute CO2 on the split grid"""
        if mode == "deriv":
            return np.cumsum(pred_col) * 5.0   # ppm/s * 5 s; constant per
            # segment absorbed by the protocol's offset recal
        if mode == "resid":
            return phys[ends] + pred_col
        if mode == "danchor":
            return anch_c[ends] + pred_col
        return pred_col

    n = len(Xtr_t)
    best = (np.inf, None, 0)
    bad = 0
    for ep in range(cfg.get("epochs", 50)):
        perm = torch.randperm(n)
        net.train()
        for b in range(0, n, 512):
            s = perm[b:b + 512]
            xb, yb = Xtr_t[s].to(DEV), ytr_t[s].to(DEV)
            eb = Etr[s].to(DEV) if n_extra else None
            opt.zero_grad()
            loss = lossf(net(xb, eb), yb)
            loss.backward()
            opt.step()
        net.eval()
        pv = predict(Xva_t, Eva)[:, co2_i]
        mv, _, _ = protocol_eval(to_co2(pv, va_ends, "val"), "val")
        if mv["mae"] < best[0] - 1e-3:
            best = (mv["mae"], {k: v.detach().clone()
                                for k, v in net.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
            if bad >= cfg.get("patience", 8):
                break
    net.load_state_dict(best[1])
    net.eval()

    pv = predict(Xva_t, Eva)[:, co2_i]
    pred_val = to_co2(pv, va_ends, "val")
    mval, _, _ = protocol_eval(pred_val, "val")
    pt = predict(torch.tensor(Xte), Ete)[:, co2_i]
    pred_test = to_co2(pt, te_ends, "test")     # stored, NOT evaluated here
    np.savez_compressed(RAW / f"{name}_s{seed}.npz",
                        val=pred_val, test=pred_test)
    info = {"cfg": cfg, "seed": seed, "best_epoch": best[2],
            "sec": round(time.time() - t0, 1), "val_CO2": mval}
    log(f"{name}_s{seed}", info)
    return mval["mae"]


# ---------------------------------------------------------------- stages
def champion_refs():
    mval, _, _ = protocol_eval(phys[idx["val"]], "val")
    log("champion_physics_VAL", {"val_CO2": mval,
        "note": "B/R+RH linear (train fit) + 30min seg recal — selection ref"})


S1 = [
    dict(name="A1_abs_joint_raw16", channels="raw16", mode="abs_joint"),
    dict(name="A2_abs_joint_devreal", channels="devreal", mode="abs_joint"),
    dict(name="A3_abs_co2_devreal", channels="devreal", mode="abs"),
    dict(name="A4_abs_joint_devreal_eng", channels="devreal_eng", mode="abs_joint"),
    dict(name="A5_abs_co2_eng_rh", channels="eng_rh", mode="abs"),
    dict(name="B1_resid_devreal", channels="devreal", mode="resid"),
    dict(name="B2_resid_devreal_eng_phys", channels="devreal_eng_phys", mode="resid"),
    dict(name="B3_resid_raw16_rh", channels="raw16_rh", mode="resid"),
    dict(name="C1_deriv_raw16", channels="raw16", mode="deriv"),
    dict(name="C2_deriv_devreal", channels="devreal", mode="deriv"),
    dict(name="D1_anchor_devreal", channels="devreal", mode="anchor"),
    dict(name="E1_abs_joint_devreal_gru", channels="devreal", mode="abs_joint", arch="gru"),
    dict(name="E2_abs_joint_devreal_huber", channels="devreal", mode="abs_joint", loss="huber"),
]


DB = dict(channels="devreal", mode="deriv", patience=15, epochs=80)
S2 = [
    dict(DB, name="C2L_deriv_devreal"),                       # longer patience
    dict(DB, name="C3_deriv_devreal_eng", channels="devreal_eng"),
    dict(DB, name="C4_deriv_eng_rh", channels="eng_rh"),
    dict(DB, name="C5_deriv_ctx2", ctx_h=2),
    dict(DB, name="C6_deriv_ctx4", ctx_h=4),
    dict(DB, name="C7_deriv_ctx8", ctx_h=8),
    dict(DB, name="C8_deriv_dsm120", dsm=120),
    dict(DB, name="C9_deriv_dsm600", dsm=600),
    dict(DB, name="C10_deriv_h128", hidden=128),
    dict(DB, name="C11_deriv_h32", hidden=32),
    dict(DB, name="C12_deriv_2layer", layers=2, dropout=0.1),
    dict(DB, name="C13_deriv_gru", arch="gru"),
    dict(DB, name="C14_deriv_huber", loss="huber"),
    dict(DB, name="C15_deriv_dense", train_stride=3),
]


def run_list(cfgs, seeds=(0,)):
    for cfg in cfgs:
        for sd in seeds:
            try:
                run_cfg(cfg, seed=sd)
            except Exception:
                print(f"!! {cfg['name']} FAILED:\n{traceback.format_exc()}",
                      flush=True)
                log(cfg["name"], {"error": "failed"})


DB2 = dict(channels="devreal", mode="deriv", ctx_h=2, patience=15, epochs=80)
S3 = [
    dict(DB2, name="F1_ctx1", ctx_h=1),
    dict(DB2, name="F2_ctx1p5", ctx_h=1.5),
    dict(DB2, name="F3_ctx3", ctx_h=3),
    dict(DB2, name="F4_ctx2_dsm120", dsm=120),
    dict(DB2, name="F5_ctx2_huber", loss="huber"),
    dict(DB2, name="F6_ctx2_dsm120_huber", dsm=120, loss="huber"),
    dict(DB2, name="F7_ctx2_h128", hidden=128),
    dict(DB2, name="F8_ctx2_h96", hidden=96),
    dict(DB2, name="F9_ctx2_dense", train_stride=3),
    dict(DB2, name="F10_ctx2_dsm120_dense", dsm=120, train_stride=3),
    dict(DB2, name="F11_ctx2_raw16rh", channels="raw16_rh"),
]


def main(stage):
    if stage == "s1":
        champion_refs()
        run_list(S1)
        print("L5 S1 DONE", flush=True)
    elif stage == "s2":
        run_list(S2)
        print("L5 S2 DONE", flush=True)
    elif stage == "s3":
        run_list(S3)
        print("L5 S3 DONE", flush=True)
    elif stage == "s6":
        # integration-free family (post-mortem of dead-reckoning test failure;
        # all selection on val, one disclosed final test shot for the winner)
        HB = dict(channels="devreal", mode="danchor", patience=15, epochs=80)
        run_list([
            dict(HB, name="H1_danchor_ctx2", ctx_h=2),
            dict(HB, name="H2_danchor_ctx6", ctx_h=6),
            dict(HB, name="H3_danchor_ctx2_dense", ctx_h=2, train_stride=3),
            dict(HB, name="H4_abs_joint_ctx2",
                 channels="devreal", mode="abs_joint", ctx_h=2),
            dict(HB, name="H5_abs_joint_ctx6",
                 channels="devreal", mode="abs_joint", ctx_h=6),
        ], seeds=(0, 1, 2, 3, 4))
        print("L5 S6 DONE", flush=True)
    elif stage == "s5":
        F10 = dict(channels="devreal", mode="deriv", ctx_h=2, patience=15,
                   epochs=80, dsm=120, train_stride=3,
                   name="F10_ctx2_dsm120_dense")
        run_list([F10], seeds=(5, 6, 7, 8, 9))
        G1 = dict(F10, name="G1_ctx1p5_dsm120_dense", ctx_h=1.5)
        G2 = dict(F10, name="G2_ctx2_dsm120_dense_lr5e4", lr=5e-4)
        run_list([G1, G2], seeds=(0, 1, 2, 3, 4))
        print("L5 S5 DONE", flush=True)
    elif stage == "s4":
        # multi-seed for top configs (names passed after stage arg)
        names = sys.argv[2:]
        pool = {c["name"]: c for c in S1 + S2 + S3}
        run_list([pool[n] for n in names], seeds=(1, 2, 3, 4))
        print("L5 S4 DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "s1")
