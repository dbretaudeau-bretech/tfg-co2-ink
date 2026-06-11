"""TFGV4 IDEA 4 — ADVERSARIAL HUMIDITY DECOUPLING.

A shared feature extractor (MLP on the static 72 light-FE features, or an
LSTM(64) on coarse 6-h reflectance windows) with TWO heads:
  - a CO2 head (SCD_CO2, primary), trained normally;
  - an RH head (BME_RH, adversary) behind a Gradient Reversal Layer (GRL).

The GRL flips the RH-head gradient (scaled by lambda) into the extractor, so
the extractor is pushed to produce features that PREDICT CO2 but are INVARIANT
to RH. Sweep lambda in {0, 0.01, 0.1, 0.5, 1} (0 = plain multi-task baseline,
no reversal). 3 seeds each.

PHYSICS CAUTION under test: CO2 and RH share the same protonation pathway in
this ink, so full RH-invariance may destroy the CO2 signal too. We measure how
CO2 test performance moves as RH-invariance increases (RH-head R2 falls) ->
the CO2-vs-RH-invariance tradeoff curve.

Inputs are REFLECTANCE ONLY by construction (no measured RH/lamp). The champion
(B/R + measured RH, OLS) USES RH; it is a different input regime and is shown
only as a yardstick. The honest question here is whether the ink's reflectance
carries a CO2 signal separable from its (dominant) RH signal.

Evaluation: champion protocol = per-lamp-segment 30-min single-offset recal,
calibration samples excluded (run_ladder2.py::r2 / run_ladder5::protocol_eval).
We report BOTH no-recal (full test) and with-recal (used windows). All model
selection is on VALIDATION (val's CO2 ranking is known to be weak/misleading on
this dataset -- noted honestly).

Alternative variant (orthogonal projection): fit an RH linear probe on the
extractor's features, remove the top-k RH-predictive directions, refit the CO2
head on the projected features, sweep k. Same tradeoff via a linear mechanism.
"""
import json
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
import torch
import torch.nn as nn

import data4 as D

RES = D.RESULTS
MPATH = RES / "idea4_adversarial.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STRIDE = 12              # coarse step = 60 s on the 5-s grid
MA = 12                 # 60-s moving average
CAL = int(30 * 60 / 5)  # 360 samples = 30 min calibration window
# requested set {0.01,0.1,0.5,1} + 0 (plain multitask baseline) + 2,5 to push
# the representation toward FULL RH-invariance and map the whole tradeoff curve
LAMBDAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
SEEDS = (0, 1, 2)

# ---------------------------------------------------------------- data (once)
df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
X72, y, t_h, idx = D.load_light_fe()
lamp = df["lamp_pct"].reset_index(drop=True).ffill().bfill()
scd = df["SCD_CO2"].values.astype(np.float64)
rh_true = df["BME_RH"].values.astype(np.float64)
RAW16 = X72[D.INK_RAW].values.astype(np.float64)


def _segments(lamp_s):
    lampr = lamp_s.round(0).values
    bounds = [0] + list(np.where(np.diff(lampr) != 0)[0] + 1) + [len(lampr)]
    seg_id = np.zeros(len(lampr), dtype=int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg_id[a:b] = s
    return seg_id


seg_all = _segments(lamp)


def protocol_eval(pred_split, split):
    """EXACT champion protocol: per-lamp-segment 30-min offset recal, cal
    samples excluded from metrics. Returns metrics on used (post-cal) samples."""
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


def r2_score(pred, true):
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def log(name, info):
    allm = json.loads(MPATH.read_text()) if MPATH.exists() else {}
    allm[name] = info
    MPATH.write_text(json.dumps(allm, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps(info)[:380]}", flush=True)


# ---------------------------------------------------------------- GRL
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grl(x, lambd):
    return _GRL.apply(x, lambd)


# ---------------------------------------------------------------- models
class AdvMLP(nn.Module):
    def __init__(self, nin, hid=64):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(nin, 128), nn.ReLU(),
                                  nn.Linear(128, hid), nn.ReLU())
        self.co2 = nn.Linear(hid, 1)
        self.rh = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, lambd):
        f = self.feat(x)
        return self.co2(f), self.rh(grl(f, lambd)), f


class AdvLSTM(nn.Module):
    def __init__(self, nin, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(nin, hid, batch_first=True)
        self.co2 = nn.Linear(hid, 1)
        self.rh = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, lambd):
        o, _ = self.lstm(x)
        h = o[:, -1]
        return self.co2(h), self.rh(grl(h, lambd)), h


# ---------------------------------------------------------------- MLP track
def prep_mlp():
    Xs = X72.values.astype(np.float64)
    sx = StandardScaler().fit(Xs[idx["train"]])
    Xn = sx.transform(Xs).astype(np.float32)
    sc = StandardScaler().fit(scd[idx["train"], None])
    sr = StandardScaler().fit(rh_true[idx["train"], None])
    return Xn, sc, sr


def run_mlp(lambd, seed, Xn, sc, sr, epochs=120, patience=15):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    Xtr = torch.tensor(Xn[tr]).to(DEV)
    ytr_c = torch.tensor(sc.transform(scd[tr, None]).astype(np.float32)).to(DEV)
    ytr_r = torch.tensor(sr.transform(rh_true[tr, None]).astype(np.float32)).to(DEV)
    Xva = torch.tensor(Xn[va]).to(DEV)
    Xte = torch.tensor(Xn[te]).to(DEV)

    net = AdvMLP(Xn.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    n = len(Xtr)
    best = (np.inf, None, 0)
    bad = 0

    def predict(X):
        net.eval()
        with torch.no_grad():
            c, r, f = net(X, lambd)
        c = sc.inverse_transform(c.cpu().numpy()).ravel()
        r = sr.inverse_transform(r.cpu().numpy()).ravel()
        return c, r, f.cpu().numpy()

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for b in range(0, n, 1024):
            s = perm[b:b + 1024]
            c, r, _ = net(Xtr[s], lambd)
            loss = mse(c, ytr_c[s]) + mse(r, ytr_r[s])
            opt.zero_grad(); loss.backward(); opt.step()
        pc, _, _ = predict(Xva)
        mv, _, _ = protocol_eval(pc, "val")
        if mv["mae"] < best[0] - 1e-3:
            best = (mv["mae"], {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best[1])
    _, _, Ftr = predict(Xtr)
    pc_va, pr_va, Fva = predict(Xva)
    pc_te, pr_te, Fte = predict(Xte)
    return finalize((pc_va, pr_va), (pc_te, pr_te), lambd, seed, best[2],
                    Ftr, Fva, Fte, rh_true[tr], scd[tr])


# ---------------------------------------------------------------- LSTM track
def prep_lstm(ctx_h=6):
    Xs = pd.DataFrame(RAW16).rolling(MA, min_periods=1).mean().values
    sx = StandardScaler().fit(Xs[idx["train"]])
    Xn = sx.transform(Xs).astype(np.float32)
    sc = StandardScaler().fit(scd[idx["train"], None])
    sr = StandardScaler().fit(rh_true[idx["train"], None])
    steps = int(ctx_h * 3600 / (STRIDE * 5))
    need = steps * STRIDE

    def windows(ends):
        ends = ends[ends - need >= 0]
        xs = np.stack([Xn[e - need:e:STRIDE] for e in ends]).astype(np.float32)
        return xs, ends

    Xtr, tr_e = windows(idx["train"][::6])      # train endpoint every 30 s
    Xva, va_e = windows(idx["val"])             # every 5 s
    Xte, te_e = windows(idx["test"])
    return (Xn, sc, sr, Xtr, tr_e, Xva, va_e, Xte, te_e)


def run_lstm(lambd, seed, pack, epochs=60, patience=12):
    Xn, sc, sr, Xtr, tr_e, Xva, va_e, Xte, te_e = pack
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr).to(DEV)
    ytr_c = torch.tensor(sc.transform(scd[tr_e, None]).astype(np.float32)).to(DEV)
    ytr_r = torch.tensor(sr.transform(rh_true[tr_e, None]).astype(np.float32)).to(DEV)
    Xva_t = torch.tensor(Xva).to(DEV)
    Xte_t = torch.tensor(Xte).to(DEV)

    net = AdvLSTM(Xtr.shape[2]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    n = len(Xtr_t)
    best = (np.inf, None, 0)
    bad = 0

    def predict(X):
        net.eval()
        cs, rs, fs = [], [], []
        with torch.no_grad():
            for b in range(0, len(X), 2048):
                c, r, f = net(X[b:b + 2048], lambd)
                cs.append(c.cpu().numpy()); rs.append(r.cpu().numpy())
                fs.append(f.cpu().numpy())
        c = sc.inverse_transform(np.concatenate(cs)).ravel()
        r = sr.inverse_transform(np.concatenate(rs)).ravel()
        return c, r, np.concatenate(fs)

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for b in range(0, n, 512):
            s = perm[b:b + 512]
            c, r, _ = net(Xtr_t[s], lambd)
            loss = mse(c, ytr_c[s]) + mse(r, ytr_r[s])
            opt.zero_grad(); loss.backward(); opt.step()
        pc, _, _ = predict(Xva_t)
        mv, _, _ = protocol_eval(pc, "val")
        if mv["mae"] < best[0] - 1e-3:
            best = (mv["mae"], {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best[1])
    # va_e / te_e align 1:1 with idx['val']/idx['test'] (need <= split start)
    assert len(va_e) == len(idx["val"]) and len(te_e) == len(idx["test"])
    _, _, Ftr = predict(Xtr_t)
    pc_va, pr_va, Fva = predict(Xva_t)
    pc_te, pr_te, Fte = predict(Xte_t)
    return finalize((pc_va, pr_va), (pc_te, pr_te), lambd, seed, best[2],
                    Ftr, Fva, Fte, rh_true[tr_e], scd[tr_e])


# ---------------------------------------------------------------- finalize
def finalize(pred_va, pred_te, lambd, seed, best_ep, Ftr, Fva, Fte,
             rh_tr, co2_tr):
    pc_va, pr_va = pred_va
    pc_te, pr_te = pred_te
    rh_va = rh_true[idx["val"]]
    rh_te = rh_true[idx["test"]]
    co2_va = scd[idx["val"]]
    co2_te = scd[idx["test"]]

    val_co2, _, _ = protocol_eval(pc_va, "val")
    test_co2_recal, _, used = protocol_eval(pc_te, "test")
    test_co2_norecal = D.metrics(pc_te, co2_te)
    val_co2_norecal = D.metrics(pc_va, co2_va)

    # Canonical invariance metric: FRESH RH probe trained on the FROZEN
    # representation (train block), evaluated on test. Independent of the
    # adversarial head's degenerate scaling. Lower => more RH-invariant.
    probe = Ridge(alpha=10.0).fit(Ftr, rh_tr)
    rh_probe_te = probe.predict(Fte)
    rh_probe_va = probe.predict(Fva)
    # also: how recoverable is CO2 from the frozen features (linear probe)?
    cprobe = Ridge(alpha=10.0).fit(Ftr, co2_tr)
    co2_probe_te = cprobe.predict(Fte)

    out = {
        "lambda": lambd, "seed": seed, "best_epoch": best_ep,
        "val": {
            "co2_recal": val_co2,
            "co2_norecal": val_co2_norecal,
            "rh_head_r2": round(r2_score(pr_va, rh_va), 3),
            "rh_probe_r2": round(r2_score(rh_probe_va, rh_va), 3),
        },
        "test": {
            "co2_recal": test_co2_recal,
            "co2_norecal": test_co2_norecal,
            "rh_head_r2": round(r2_score(pr_te, rh_te), 3),
            "rh_head_r": round(float(np.corrcoef(pr_te, rh_te)[0, 1]), 3),
            "rh_probe_r2": round(r2_score(rh_probe_te, rh_te), 3),
            "rh_probe_r": round(float(np.corrcoef(rh_probe_te, rh_te)[0, 1]), 3),
            "co2_probe_r": round(float(np.corrcoef(co2_probe_te, co2_te)[0, 1]), 3),
        },
    }
    return out


# ---------------------------------------------------------------- references
def references():
    """Physics champion (B/R + measured RH) and a reflectance-only ridge, both
    no-recal (full test) and with-recal (used windows)."""
    br = pd.concat([df[f"OP{i}_blue"] / df[f"OP{i}_red"] for i in range(4)],
                   axis=1).mean(axis=1).values
    tr, te = idx["train"], idx["test"]
    # physics: B/R + measured RH (uses RH input)
    A = np.column_stack([br, rh_true, np.ones(len(df))])
    coef, *_ = np.linalg.lstsq(A[tr], scd[tr], rcond=None)
    phys = A @ coef
    pm, _, _ = protocol_eval(phys[te], "test")
    refs = {
        "physics_BR_RH": {
            "uses_RH_input": True,
            "test_norecal": D.metrics(phys[te], scd[te]),
            "test_recal": pm,
        }
    }
    # reflectance-only ridge (no RH input) — non-adversarial yardstick
    sx = StandardScaler().fit(X72.values[tr])
    best = None
    for a in (10.0, 100.0, 1000.0):
        m = Ridge(alpha=a).fit(sx.transform(X72.values[tr]), scd[tr])
        pv = m.predict(sx.transform(X72.values[idx["val"]]))
        mae = np.abs(pv - scd[idx["val"]]).mean()
        if best is None or mae < best[0]:
            best = (mae, a, m)
    _, a, m = best
    pr = m.predict(sx.transform(X72.values[te]))
    prm, _, _ = protocol_eval(pr, "test")
    refs["reflectance_ridge72"] = {
        "uses_RH_input": False, "alpha": a,
        "test_norecal": D.metrics(pr, scd[te]),
        "test_recal": prm,
    }
    log("references", refs)
    return refs


# ---------------------------------------------------------------- ortho variant
def ortho_projection():
    """Alternative to GRL: train a plain CO2 MLP (lambda=0), extract penultimate
    features, fit an RH linear probe on TRAIN features, remove the top-k
    RH-predictive directions (Gram-Schmidt on the probe weight rows after PCA on
    RH-gradient directions), refit a ridge CO2 head on projected features. Sweep
    k. Mirrors the GRL tradeoff via a linear mechanism."""
    Xn, sc, sr = prep_mlp()
    # train a plain multitask net to get a feature space, seed 0
    torch.manual_seed(0); np.random.seed(0)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    Xtr = torch.tensor(Xn[tr]).to(DEV)
    ytr_c = torch.tensor(sc.transform(scd[tr, None]).astype(np.float32)).to(DEV)
    net = AdvMLP(Xn.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    for ep in range(80):
        net.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for b in range(0, len(Xtr), 1024):
            s = perm[b:b + 1024]
            c, _, _ = net(Xtr[s], 0.0)
            loss = mse(c, ytr_c[s]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def feats(X):
        with torch.no_grad():
            _, _, f = net(torch.tensor(X).to(DEV), 0.0)
        return f.cpu().numpy()

    Ftr, Fva, Fte = feats(Xn[tr]), feats(Xn[va]), feats(Xn[te])
    # RH probe -> direction(s) most predictive of RH. Use ordinary least squares
    # on a sequence of orthogonal directions (deflation).
    out = {}
    F = {"train": Ftr.copy(), "val": Fva.copy(), "test": Fte.copy()}
    removed = []
    for k in range(0, 9):
        if k > 0:
            # find current most-RH-predictive unit direction on train, remove it
            lr = LinearRegression().fit(F["train"], rh_true[tr])
            w = lr.coef_.astype(np.float64)
            nrm = np.linalg.norm(w)
            if nrm < 1e-9:
                break
            u = w / nrm
            for key in F:
                F[key] = F[key] - np.outer(F[key] @ u, u)
            removed.append(u)
        # refit CO2 ridge on projected features
        cr = Ridge(alpha=10.0).fit(F["train"], scd[tr])
        pv = cr.predict(F["val"]); pte = cr.predict(F["test"])
        # RH residual predictability after projection (how much RH info remains)
        rhp = LinearRegression().fit(F["train"], rh_true[tr])
        rh_te_pred = rhp.predict(F["test"])
        vco2, _, _ = protocol_eval(pv, "val")
        tco2, _, _ = protocol_eval(pte, "test")
        out[f"k{k}"] = {
            "dirs_removed": k,
            "val_co2_recal": vco2,
            "test_co2_recal": tco2,
            "test_co2_norecal": D.metrics(pte, scd[te]),
            "rh_resid_r2_test": round(r2_score(rh_te_pred, rh_true[te]), 3),
        }
    log("ortho_projection_mlp", out)
    return out


def ortho_raw():
    """Cleaner alternative: orthogonal projection on the RAW 72 reflectance
    features. Iteratively remove the most RH-predictive unit direction (OLS
    deflation) from the standardized feature space, refit a CO2 ridge after
    each removal, sweep k. Directly answers: strip the RH-predictive subspace
    from reflectance -> what CO2 signal survives?"""
    tr, va, te = idx["train"], idx["val"], idx["test"]
    sx = StandardScaler().fit(X72.values[tr])
    F = {"train": sx.transform(X72.values[tr]),
         "val": sx.transform(X72.values[va]),
         "test": sx.transform(X72.values[te])}
    out = {}
    for k in range(0, 13):
        if k > 0:
            lr = LinearRegression().fit(F["train"], rh_true[tr])
            w = lr.coef_.astype(np.float64); nrm = np.linalg.norm(w)
            if nrm < 1e-9:
                break
            u = w / nrm
            for key in F:
                F[key] = F[key] - np.outer(F[key] @ u, u)
        cr = Ridge(alpha=100.0).fit(F["train"], scd[tr])
        pv = cr.predict(F["val"]); pte = cr.predict(F["test"])
        rhp = LinearRegression().fit(F["train"], rh_true[tr])
        vco2, _, _ = protocol_eval(pv, "val")
        tco2, _, _ = protocol_eval(pte, "test")
        out[f"k{k}"] = {
            "dirs_removed": k,
            "rh_resid_r2_train": round(r2_score(rhp.predict(F["train"]), rh_true[tr]), 3),
            "rh_resid_r2_test": round(r2_score(rhp.predict(F["test"]), rh_true[te]), 3),
            "val_co2_recal": vco2,
            "test_co2_recal": tco2,
            "test_co2_norecal": D.metrics(pte, scd[te]),
        }
    log("ortho_projection_raw72", out)
    return out


def forced_curve():
    """Mechanistic physics-caution curve: train FIXED epochs (no early-stop
    restore) so the GRL actually reshapes the representation, across an extended
    lambda grid. Report the FRESH-probe RH R2 (true invariance, lower=more
    invariant) and CO2 recoverability/recal at each lambda. This traces what
    happens to CO2 as REAL RH-invariance is forced -- the curve the honest
    val-selected run cannot show (it stops before the adversary acts)."""
    grid = [0.0, 0.5, 2.0, 5.0, 10.0, 20.0, 50.0]
    # ---- MLP
    Xn, sc, sr = prep_mlp()
    tr, va, te = idx["train"], idx["val"], idx["test"]
    out_mlp = {}
    for lam in grid:
        prr, cor, cre, cmae = [], [], [], []
        for sd in (0, 1):
            torch.manual_seed(sd); np.random.seed(sd)
            Xtr = torch.tensor(Xn[tr]).to(DEV)
            yc = torch.tensor(sc.transform(scd[tr, None]).astype(np.float32)).to(DEV)
            yr = torch.tensor(sr.transform(rh_true[tr, None]).astype(np.float32)).to(DEV)
            net = AdvMLP(Xn.shape[1]).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            mse = nn.MSELoss()
            for ep in range(60):
                net.train(); perm = torch.randperm(len(Xtr), device=DEV)
                for b in range(0, len(Xtr), 1024):
                    s = perm[b:b + 1024]
                    c, r, _ = net(Xtr[s], lam)
                    loss = mse(c, yc[s]) + mse(r, yr[s])
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                _, _, Ftr = net(Xtr, lam); Ftr = Ftr.cpu().numpy()
                c_te, _, Fte = net(torch.tensor(Xn[te]).to(DEV), lam)
                c_te = sc.inverse_transform(c_te.cpu().numpy()).ravel()
                Fte = Fte.cpu().numpy()
            rhp = Ridge(alpha=10.0).fit(Ftr, rh_true[tr])
            cop = Ridge(alpha=10.0).fit(Ftr, scd[tr])
            prr.append(r2_score(rhp.predict(Fte), rh_true[te]))
            cor.append(float(np.corrcoef(cop.predict(Fte), scd[te])[0, 1]))
            m, _, _ = protocol_eval(c_te, "test"); cre.append(m["pearson"]); cmae.append(m["mae"])
        out_mlp[f"lam{lam}"] = {"rh_probe_r2": round(float(np.mean(prr)), 3),
                                "co2_probe_r": round(float(np.mean(cor)), 3),
                                "co2_recal_r": round(float(np.mean(cre)), 3),
                                "co2_recal_mae": round(float(np.mean(cmae)), 1)}
    log("forced_curve_mlp", out_mlp)
    # ---- LSTM
    pack = prep_lstm(ctx_h=6)
    Xn2, sc2, sr2, Xtr, tr_e, Xva, va_e, Xte, te_e = pack
    out_lstm = {}
    for lam in grid:
        prr, cor, cre, cmae = [], [], [], []
        for sd in (0, 1):
            torch.manual_seed(sd); np.random.seed(sd)
            Xtr_t = torch.tensor(Xtr).to(DEV)
            yc = torch.tensor(sc2.transform(scd[tr_e, None]).astype(np.float32)).to(DEV)
            yr = torch.tensor(sr2.transform(rh_true[tr_e, None]).astype(np.float32)).to(DEV)
            net = AdvLSTM(Xtr.shape[2]).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            mse = nn.MSELoss()
            for ep in range(30):
                net.train(); perm = torch.randperm(len(Xtr_t), device=DEV)
                for b in range(0, len(Xtr_t), 512):
                    s = perm[b:b + 512]
                    c, r, _ = net(Xtr_t[s], lam)
                    loss = mse(c, yc[s]) + mse(r, yr[s])
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()

            def feats(X):
                fs, cs = [], []
                with torch.no_grad():
                    for b in range(0, len(X), 2048):
                        c, _, f = net(X[b:b + 2048], lam)
                        fs.append(f.cpu().numpy()); cs.append(c.cpu().numpy())
                return np.concatenate(fs), sc2.inverse_transform(np.concatenate(cs)).ravel()
            Ftr, _ = feats(Xtr_t)
            Fte, c_te = feats(torch.tensor(Xte).to(DEV))
            rhp = Ridge(alpha=10.0).fit(Ftr, rh_true[tr_e])
            cop = Ridge(alpha=10.0).fit(Ftr, scd[tr_e])
            prr.append(r2_score(rhp.predict(Fte), rh_true[te_e]))
            cor.append(float(np.corrcoef(cop.predict(Fte), scd[te_e])[0, 1]))
            m, _, _ = protocol_eval(c_te, "test"); cre.append(m["pearson"]); cmae.append(m["mae"])
        out_lstm[f"lam{lam}"] = {"rh_probe_r2": round(float(np.mean(prr)), 3),
                                 "co2_probe_r": round(float(np.mean(cor)), 3),
                                 "co2_recal_r": round(float(np.mean(cre)), 3),
                                 "co2_recal_mae": round(float(np.mean(cmae)), 1)}
    log("forced_curve_lstm", out_lstm)
    return out_mlp, out_lstm


# ---------------------------------------------------------------- main
def main(stage):
    if stage == "refs":
        references()
    elif stage == "mlp":
        Xn, sc, sr = prep_mlp()
        results = {}
        for lam in LAMBDAS:
            for sd in SEEDS:
                key = f"mlp_lam{lam}_s{sd}"
                try:
                    r = run_mlp(lam, sd, Xn, sc, sr)
                    results[key] = r
                    log(key, r)
                except Exception:
                    print(f"!! {key} FAILED:\n{traceback.format_exc()}", flush=True)
        print("MLP DONE", flush=True)
    elif stage == "lstm":
        pack = prep_lstm(ctx_h=6)
        for lam in LAMBDAS:
            for sd in SEEDS:
                key = f"lstm6h_lam{lam}_s{sd}"
                try:
                    r = run_lstm(lam, sd, pack)
                    log(key, r)
                except Exception:
                    print(f"!! {key} FAILED:\n{traceback.format_exc()}", flush=True)
        print("LSTM DONE", flush=True)
    elif stage == "ortho":
        ortho_projection()
    elif stage == "ortho_raw":
        ortho_raw()
    elif stage == "curve":
        forced_curve()
    print(f"STAGE {stage} COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "refs")
