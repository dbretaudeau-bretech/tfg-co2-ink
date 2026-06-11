"""TFGV5 final selection — combine stored raw predictions (results/raw5/) on
VALIDATION only: per-config seed means, seed ensembles, physics blends.
`python analyze5.py val`  -> candidate table (validation protocol only)
`python analyze5.py test <candidate>` -> single final test evaluation (once).
Candidate spec: "<cfg>:ens" (seed-ensemble) or "<cfg>:ens+phys<w>" blend.
"""
import json
import sys
import numpy as np

import run_ladder5 as L  # reuses data, protocol_eval, phys, idx

RAW = L.RAW
RES = L.RES


def load_seeds(cfg_name, split):
    preds = []
    for f in sorted(RAW.glob(f"{cfg_name}_s*.npz")):
        preds.append(np.load(f)[split])
    return np.stack(preds)


def val_table(cfg_names):
    rows = []
    for c in cfg_names:
        P = load_seeds(c, "val")
        maes = []
        for p in P:
            m, _, _ = L.protocol_eval(p, "val")
            maes.append(m["mae"])
        ens = P.mean(axis=0)
        mens, _, _ = L.protocol_eval(ens, "val")
        rows.append((c, len(P), float(np.mean(maes)), float(np.std(maes)), mens))
        # physics blends on the ensemble
        for w in (0.1, 0.2, 0.3, 0.4, 0.5):
            mb, _, _ = L.protocol_eval((1 - w) * ens + w * L.phys[L.idx["val"]], "val")
            rows.append((f"{c}:ens+phys{w}", len(P), None, None, mb))
    print(f"{'candidate':38s} {'n':>2s} {'seed mae μ±σ':>16s}  ensemble val protocol")
    for c, n, mu, sd, m in rows:
        ms = f"{mu:.1f}±{sd:.1f}" if mu is not None else "-"
        print(f"{c:38s} {n:2d} {ms:>16s}  mae={m['mae']:.2f} r2={m['r2']} "
              f"r={m['pearson']} full_r={m['full_r']}")
    return rows


def final_test(cand):
    cfg_name, _, mod = cand.partition(":")
    files = sorted(RAW.glob(f"{cfg_name}_s*.npz"))
    if mod.startswith("top"):           # top-k seeds by VAL protocol MAE
        k = int(mod[3:].split("+")[0])
        maes = [L.protocol_eval(np.load(f)["val"], "val")[0]["mae"]
                for f in files]
        files = [files[i] for i in np.argsort(maes)[:k]]
    P = np.stack([np.load(f)["test"] for f in files])
    Pv = np.stack([np.load(f)["val"] for f in files])
    pred, pred_v = P.mean(axis=0), Pv.mean(axis=0)
    w, lam = 0.0, None
    if "+phys" in mod:
        w = float(mod.split("+phys")[1])
        pred = (1 - w) * pred + w * L.phys[L.idx["test"]]
        pred_v = (1 - w) * pred_v + w * L.phys[L.idx["val"]]
    if "+leakB" in mod:    # leaky integration toward the physics baseline
        lam = float(mod.split("+leakB")[1])

        def leaky(p, base):
            d = np.diff(p, prepend=p[0]) - np.diff(base, prepend=base[0])
            out = np.empty_like(d)
            acc = 0.0
            for i, di in enumerate(d):
                acc = (1.0 - lam) * acc + di
                out[i] = acc
            return base + out

        pred = leaky(pred, L.phys[L.idx["test"]])
        pred_v = leaky(pred_v, L.phys[L.idx["val"]])
    m, pred_cal, used = L.protocol_eval(pred, "test")
    mval, _, _ = L.protocol_eval(pred_v, "val")
    te = L.idx["test"]
    np.savez_compressed(RES / "preds5_rnn_champion.npz",
                        pred=pred_cal, true=L.scd[te], t_h=L.t_h[te],
                        used_mask=used)
    info = {"candidate": cand, "n_seeds": len(P),
            "seeds": [f.name for f in files], "phys_blend_w": w,
            "leakB_lambda": lam,
            "val_CO2": mval, "TEST_CO2": m,
            "champion_ref_test": {"mae": 53.65, "r2": 0.645, "pearson": 0.822},
            "note": "exact champion protocol (30min seg recal, cal excluded)"}
    L.log(f"FINAL_{cand}", info)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "val":
        val_table(sys.argv[2:])
    elif sys.argv[1] == "test":
        final_test(sys.argv[2])
