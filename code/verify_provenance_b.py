"""Recompute every deterministic Section-IV / Table-II number from scratch."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

import data4 as D

df = D.load_unified().dropna(subset=D.INK_RAW).reset_index(drop=True)
X72, y, t_h, idx = D.load_light_fe()
Xd72, yd, th = D.split_xy(X72, y, t_h, idx)
yarr = np.asarray(y)


def rep(tag, **kw):
    print(tag, " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


M = D.metrics

# ---- 1. floors -------------------------------------------------------------
mu = yd["train"].mean(axis=0)
rep("FLOOR", RH=round(np.abs(mu[0] - yd["test"][:, 0]).mean(), 2),
    CO2=round(np.abs(mu[1] - yd["test"][:, 1]).mean(), 1))

# ---- 2. OLS16 --------------------------------------------------------------
Xraw = X72[D.INK_RAW]
Xd, _, _ = D.split_xy(Xraw, y, t_h, idx)
sx = StandardScaler().fit(Xd["train"])
m = LinearRegression().fit(sx.transform(Xd["train"]), yd["train"])
ptr = m.predict(sx.transform(Xd["train"]))
pte = m.predict(sx.transform(Xd["test"]))
rep("OLS16", tr_CO2=M(ptr[:, 1], yd["train"][:, 1])["mae"],
    te_CO2=M(pte[:, 1], yd["test"][:, 1])["mae"],
    tr_RH=M(ptr[:, 0], yd["train"][:, 0])["mae"],
    te_RH=M(pte[:, 0], yd["test"][:, 0])["mae"])


def joint_ridge(X, alphas=(10., 100., 1000.), sel_col=1):
    """Joint ridge, alpha selected on val CO2 MAE (night-ladder convention)."""
    Xd, _, _ = D.split_xy(X, y, t_h, idx)
    sx = StandardScaler().fit(Xd["train"])
    best = None
    for a in alphas:
        mm = Ridge(alpha=a).fit(sx.transform(Xd["train"]), yd["train"])
        pv = mm.predict(sx.transform(Xd["val"]))
        sc = np.abs(pv[:, sel_col] - yd["val"][:, sel_col]).mean()
        if best is None or sc < best[0]:
            best = (sc, a, mm)
    _, a, mm = best
    ptr = mm.predict(sx.transform(Xd["train"]))
    pte = mm.predict(sx.transform(Xd["test"]))
    return a, ptr, pte


# ---- 3. Ridge-72 -----------------------------------------------------------
a, ptr, pte = joint_ridge(X72)
rep("RIDGE72", alpha=a,
    tr_RH=M(ptr[:, 0], yd["train"][:, 0])["mae"], te_RH=M(pte[:, 0], yd["test"][:, 0])["mae"],
    tr_CO2=M(ptr[:, 1], yd["train"][:, 1])["mae"], te_CO2=M(pte[:, 1], yd["test"][:, 1])["mae"],
    r_te=M(pte[:, 1], yd["test"][:, 1])["pearson"])

# ---- 4/5. Ridge + HP windows ----------------------------------------------
for hours in (4.0, 8.0):
    Xhp = pd.concat([X72, D.add_highpass(X72, hours)], axis=1)
    a, ptr, pte = joint_ridge(Xhp)
    rep(f"RIDGE_HP{hours:g}", alpha=a, nfeat=Xhp.shape[1],
        params_with_intercept=(Xhp.shape[1] + 1) * 2,
        tr_RH=M(ptr[:, 0], yd["train"][:, 0])["mae"], te_RH=M(pte[:, 0], yd["test"][:, 0])["mae"],
        te_RH_r2=M(pte[:, 0], yd["test"][:, 0])["r2"],
        tr_CO2=M(ptr[:, 1], yd["train"][:, 1])["mae"], te_CO2=M(pte[:, 1], yd["test"][:, 1])["mae"],
        r_te=M(pte[:, 1], yd["test"][:, 1])["pearson"])

# ---- 6. IRN12 joint ridge ---------------------------------------------------
def irn12(dframe):
    out = {}
    for i in range(4):
        r, g, b, ir = (dframe[f"OP{i}_{c}"] for c in ("red", "green", "blue", "ir"))
        out[f"OP{i}_R_IR"] = r / ir
        out[f"OP{i}_G_IR"] = g / ir
        out[f"OP{i}_B_IR"] = b / ir
    X = pd.DataFrame(out, index=dframe.index)
    return X.replace([np.inf, -np.inf], np.nan).ffill().bfill()


XI12 = irn12(df)
a, ptr, pte = joint_ridge(XI12)
rep("RIDGE_IRN12_joint", alpha=a,
    tr_RH=M(ptr[:, 0], yd["train"][:, 0])["mae"], te_RH=M(pte[:, 0], yd["test"][:, 0])["mae"],
    tr_CO2=M(ptr[:, 1], yd["train"][:, 1])["mae"], te_CO2=M(pte[:, 1], yd["test"][:, 1])["mae"],
    te_CO2_r2=M(pte[:, 1], yd["test"][:, 1])["r2"],
    r_te=M(pte[:, 1], yd["test"][:, 1])["pearson"])
# per-target variant (idea2 style)
for col, tgt in ((0, "RH"), (1, "CO2")):
    a, ptr, pte = joint_ridge(XI12, sel_col=col)
    rep(f"RIDGE_IRN12_pertarget_{tgt}", alpha=a,
        tr=M(ptr[:, col], yd["train"][:, col])["mae"],
        te=M(pte[:, col], yd["test"][:, col])["mae"],
        te_r2=M(pte[:, col], yd["test"][:, col])["r2"],
        r_te=M(pte[:, col], yd["test"][:, col])["pearson"])

# ---- 7. random split control ------------------------------------------------
rng = np.random.RandomState(0)
n = len(X72)
perm = rng.permutation(n)
ridx = {"train": np.sort(perm[:int(0.7 * n)]),
        "val": np.sort(perm[int(0.7 * n):int(0.85 * n)]),
        "test": np.sort(perm[int(0.85 * n):])}
Xd, yd_r, _ = D.split_xy(X72, y, t_h, ridx)
sx = StandardScaler().fit(Xd["train"])
m = Ridge(alpha=100.0).fit(sx.transform(Xd["train"]), yd_r["train"])
ptr = m.predict(sx.transform(Xd["train"]))
pte = m.predict(sx.transform(Xd["test"]))
rep("RANDOMSPLIT", tr_RH=M(ptr[:, 0], yd_r["train"][:, 0])["mae"],
    te_RH=M(pte[:, 0], yd_r["test"][:, 0])["mae"],
    tr_CO2=M(ptr[:, 1], yd_r["train"][:, 1])["mae"],
    te_CO2=M(pte[:, 1], yd_r["test"][:, 1])["mae"],
    r_te=M(pte[:, 1], yd_r["test"][:, 1])["pearson"])

# ---- 8. RH ablations ---------------------------------------------------------
subsets = {
    "red4": [f"OP{i}_red" for i in range(4)],
    "OP0_4col": [f"OP0_{c}" for c in ("red", "green", "blue", "ir")],
    "raw16": list(D.INK_RAW),
    "OP0_raw_ratios": [c for c in X72.columns if c.startswith("OP0_")],
}
for nm, cols in subsets.items():
    Xs = X72[cols] if all(c in X72.columns for c in cols) else df[cols]
    # RH-selected alpha (per-target, static)
    a, ptr, pte = joint_ridge(Xs, sel_col=0)
    static_te = M(pte[:, 0], yd["test"][:, 0])["mae"]
    # with HP8h expansion, RH-selected
    Xhp = pd.concat([Xs, D.add_highpass(Xs, 8.0)], axis=1)
    a8, ptr8, pte8 = joint_ridge(Xhp, sel_col=0)
    rep(f"ABL_{nm}", static_alpha=a, static_te_RH=static_te,
        hp8_alpha=a8, hp8_te_RH=M(pte8[:, 0], yd["test"][:, 0])["mae"])

# ---- 9. drift shifts ----------------------------------------------------------
def shift(cols, src):
    out = []
    for c in cols:
        v = src[c].values
        out.append((v[idx["test"]].mean() - v[idx["train"]].mean())
                   / (v[idx["train"]].std() + 1e-12))
    return round(float(np.mean(np.abs(out))), 2)


rep("DRIFT", raw_B=shift([f"OP{i}_blue" for i in range(4)], df),
    B_over_R=shift([f"OP{i}_BR" for i in range(4)], X72),
    B_over_IR=shift([f"OP{i}_B_IR" for i in range(4)], XI12))

# ---- 10. param counts ----------------------------------------------------------
lstm_p = 4 * (16 * 64 + 64 * 64 + 2 * 64) + 64 * 2 + 2
mlp_reg_p = (72 * 128 + 128) + (128 * 64 + 64) + (64 * 2 + 2)
rep("PARAMS", irn12=(12 + 1) * 2, ridge72=(72 + 1) * 2, ridge_hp4=(288 + 1) * 2,
    mlp_reg=mlp_reg_p, lstm=lstm_p)

# ---- 11. artifact cross-checks from full_*.npz ---------------------------------
RES = D.RESULTS
for nm in ("ols16", "ridge72", "ridge_hp4", "ridge_hp8", "ridge_irn12",
           "mlp64", "mlp1024", "mlp_reg", "lstm6h"):
    d = np.load(RES / f"full_{nm}.npz")
    p = d["pred"]
    ok = ~np.isnan(p)
    mtr = M(p[idx["train"]][ok[idx["train"]]], yarr[idx["train"], 1][ok[idx["train"]]])
    mte = M(p[idx["test"]][ok[idx["test"]]], yarr[idx["test"], 1][ok[idx["test"]]])
    rep(f"FULLNPZ_{nm}", tr_CO2=mtr["mae"], te_CO2=mte["mae"],
        r_te=mte["pearson"], r2_te=mte["r2"])
print("DONE")
