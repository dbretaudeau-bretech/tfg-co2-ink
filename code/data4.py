"""TFGV4 — corrected-target data module.

Targets (measured references): BME_RH (%), SCD_CO2 (ppm, Sensirion SCD30 NDIR).
Inputs: ink reflectance only (OP{0..3}_{red,green,blue,ir}) + transforms thereof.
Forbidden as inputs: all schedule/setpoint channels, BME_*, SCD_*, lamp.

Splits: chronological 70/15/15 with 4-min gap (same convention as TFGV3).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/bretech/OTHERS/TFGV4")
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

TARGETS = ["BME_RH", "SCD_CO2"]
FORBIDDEN_TOKENS = ("BME_T", "BME_P", "lamp_pct", "SCD", "co2_set", "h2o_set",
                    "dry_dilution", "co2_mfc", "wet_mfc")

INK_RAW = [f"OP{i}_{c}" for i in range(4) for c in ("red", "green", "blue", "ir")]
RATIOS = ["BR", "RB", "B_RGB", "R_RGB", "G_RGB", "IR_B", "IR_R", "IR_RGB"]


def load_unified() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "unified_5s_corrected.csv")
    return df.dropna(subset=TARGETS).reset_index(drop=True)


def split_indices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    t = df["t_h"].values
    n = len(df)
    t_train_end = t[int(n * 0.70)]
    t_val_end = t[int(n * 0.85)]
    gap = 240 / 3600.0
    return {
        "train": np.where(t <= t_train_end)[0],
        "val": np.where((t >= t_train_end + gap) & (t <= t_val_end))[0],
        "test": np.where(t >= t_val_end + gap)[0],
    }


def _per_pixel_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for i in range(4):
        r, g, b, ir = (df[f"OP{i}_{c}"] for c in ("red", "green", "blue", "ir"))
        rgb = r + g + b
        out[f"OP{i}_BR"] = b / r
        out[f"OP{i}_RB"] = r / b
        out[f"OP{i}_B_RGB"] = b / rgb
        out[f"OP{i}_R_RGB"] = r / rgb
        out[f"OP{i}_G_RGB"] = g / rgb
        out[f"OP{i}_IR_B"] = ir / b
        out[f"OP{i}_IR_R"] = ir / r
        out[f"OP{i}_IR_RGB"] = ir / rgb
    return pd.DataFrame(out, index=df.index)


def _cross_pixel(ratios: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in ("red", "green", "blue", "ir"):
        block = raw[[f"OP{i}_{c}" for i in range(4)]]
        out[f"px_{c}_mean"] = block.mean(axis=1)
        out[f"px_{c}_std"] = block.std(axis=1)
    for r in RATIOS:
        block = ratios[[f"OP{i}_{r}" for i in range(4)]]
        out[f"px_{r}_mean"] = block.mean(axis=1)
        out[f"px_{r}_std"] = block.std(axis=1)
    return pd.DataFrame(out, index=ratios.index)


def load_light_fe() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """72-feature light-FE set (16 raw + 32 ratios + 24 cross-pixel)."""
    df = load_unified().dropna(subset=INK_RAW).reset_index(drop=True)
    ratios = _per_pixel_ratios(df)
    X = pd.concat([df[INK_RAW], ratios, _cross_pixel(ratios, df[INK_RAW])], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    y = df[TARGETS].copy()
    _audit(X.columns)
    return X, y, df["t_h"].values, split_indices(df)


def add_highpass(X: pd.DataFrame, hours: float = 2.0, rms=(5, 15)) -> pd.DataFrame:
    """Drift-robust expansion: high-pass each feature (x - rolling mean) at the
    given window, plus short rolling means OF the high-passed signal."""
    w = int(hours * 3600 / 5)
    hp = X - X.rolling(w, min_periods=1).mean()
    blocks = [hp.add_suffix(f"_hp{hours:g}h")]
    for mins in rms:
        blocks.append(hp.rolling(int(mins * 60 / 5), min_periods=1).mean()
                      .add_suffix(f"_hp{hours:g}h_rm{mins}"))
    return pd.concat(blocks, axis=1)


def split_xy(X, y, t_h, idx):
    Xd = {k: np.asarray(X)[v] for k, v in idx.items()}
    yd = {k: np.asarray(y)[v] for k, v in idx.items()}
    th = {k: t_h[v] for k, v in idx.items()}
    return Xd, yd, th


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    mae = float(np.abs(pred - true).mean())
    r2 = float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())
    r = float(np.corrcoef(pred, true)[0, 1])
    return {"mae": round(mae, 2), "r2": round(r2, 3), "pearson": round(r, 3)}


def _audit(names):
    for n in names:
        if n in TARGETS:
            raise ValueError(f"target leak: {n}")
        for tok in FORBIDDEN_TOKENS:
            if tok in n:
                raise ValueError(f"forbidden token '{tok}' in feature '{n}'")
