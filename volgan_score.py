"""
volgan_score.py
===============
cVolGAN relative-value score for IBOV options — the Vuletić & Cont use-case:
score today's surface against the model's conditional distribution of what
today's surface SHOULD look like given yesterday's state.

Loads the numpy generator trained by train_volgan_br_conditional.py:
  results/volgan_cond_G_weights.npy  {l1_W,l1_b,l2_W,l2_b,l3_W,l3_b}
  results/volgan_cond_norm.npy       {cond_mean,cond_std,tgt_mean,tgt_std}

Generator: [noise(32) | cond(30)] → softplus(128) → softplus(256) → linear(28)
  cond = [r_{t-1}, r_{t-2}, RV21_{t-1}, logIV_{t-1} (27)]
  out  = [r_t, ΔlogIV_t (27)]        (all z-normalised with the stored stats)

Score: per grid node, the percentile of the ACTUAL IV_t within the generated
distribution of IV_t (= exp(logIV_{t-1} + ΔlogIV sampled)). Low percentile =
today's IV is cheap vs the conditional model; high = rich. Node percentiles
are interpolated to each option's (k, T).

Public API:
  volgan_percentiles(chain_ibov_t, chain_ibov_tm1, spot_hist, n_samples=2000)
    -> callable f(k, T) giving the percentile, or None if unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

BASE = Path(__file__).parent
RES = BASE / "results"

K_GRID = np.array([-0.24, -0.18, -0.12, -0.06, 0.00, 0.06, 0.12, 0.18, 0.24])
T_GRID = np.array([1 / 12, 2 / 12, 3 / 12])
N_K, N_T = len(K_GRID), len(T_GRID)
NOISE_DIM = 32


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


class _G:
    def __init__(self, w):
        self.w = w

    def sample(self, cond_n, n):
        z = np.random.randn(n, NOISE_DIM).astype(np.float32)
        c = np.repeat(cond_n[None, :], n, axis=0).astype(np.float32)
        x = np.concatenate([z, c], axis=-1)
        h = softplus(x @ self.w["l1_W"] + self.w["l1_b"])
        h = softplus(h @ self.w["l2_W"] + self.w["l2_b"])
        return h @ self.w["l3_W"] + self.w["l3_b"]


def load_model():
    wp = RES / "volgan_cond_G_weights.npy"
    np_ = RES / "volgan_cond_norm.npy"
    if not wp.exists() or not np_.exists():
        return None, None
    w = np.load(wp, allow_pickle=True).item()
    norm = np.load(np_, allow_pickle=True).item()
    return _G(w), norm


def surface_from_chain(chain_day):
    """chain_day: rows with columns moneyness, dte, iv (percent). Returns the
    27-dim logIV grid (C-order k-major to match training) or None."""
    if chain_day is None or len(chain_day) == 0 or "iv" not in chain_day.columns:
        return None
    d = chain_day.dropna(subset=["iv"])
    d = d[(d["dte"] >= 7) & (d["dte"] <= 150)]
    if len(d) < 12:
        return None
    pts = d[["moneyness", "dte"]].values.astype(float)
    pts[:, 1] = pts[:, 1] / 365.0
    vals = d["iv"].values / 100.0
    KK, TT = np.meshgrid(K_GRID, T_GRID, indexing="ij")
    grid = griddata(pts, vals, (KK, TT), method="linear")
    # fill holes with nearest
    if np.isnan(grid).any():
        near = griddata(pts, vals, (KK, TT), method="nearest")
        grid = np.where(np.isnan(grid), near, grid)
    if np.isnan(grid).any() or (grid <= 0).any():
        return None
    return np.log(grid).reshape(-1)          # (27,) k-major


def volgan_percentiles(chain_t, chain_tm1, spot_hist, n_samples=2000, seed=7):
    """Returns f(k_arr, T_arr) -> percentile array, or None."""
    G, norm = load_model()
    if G is None:
        return None

    s_t = surface_from_chain(chain_t)
    s_tm1 = surface_from_chain(chain_tm1)
    if s_t is None or s_tm1 is None:
        return None

    # conditioning info at t-1
    sp = spot_hist.dropna().sort_index()
    lr = np.log(sp).diff().dropna()
    if len(lr) < 25:
        return None
    r1, r2 = float(lr.iloc[-1]) * 252, float(lr.iloc[-2]) * 252   # annualised
    rv21 = float(lr.iloc[-21:].std() * np.sqrt(252))

    cond = np.concatenate([[r1, r2, rv21], s_tm1]).astype(np.float32)
    cond_n = np.ravel((cond - np.ravel(norm["cond_mean"]))
                      / np.ravel(norm["cond_std"]))

    np.random.seed(seed)
    out_n = G.sample(cond_n, n_samples)                     # (n, 28) normalised
    out = out_n * np.ravel(norm["tgt_std"]) + np.ravel(norm["tgt_mean"])
    dlog = out[:, 1:]                                       # (n, 27)
    iv_samples = np.exp(s_tm1[None, :] + dlog)              # (n, 27)
    actual = np.exp(s_t)                                    # (27,)

    node_pct = (iv_samples < actual[None, :]).mean(axis=0) * 100   # (27,)

    KK, TT = np.meshgrid(K_GRID, T_GRID, indexing="ij")
    nodes = np.column_stack([KK.reshape(-1), TT.reshape(-1)])

    def f(k_arr, T_arr):
        q = griddata(nodes, node_pct,
                     (np.clip(k_arr, K_GRID[0], K_GRID[-1]),
                      np.clip(T_arr, T_GRID[0], T_GRID[-1])),
                     method="linear")
        near = griddata(nodes, node_pct,
                        (np.clip(k_arr, K_GRID[0], K_GRID[-1]),
                         np.clip(T_arr, T_GRID[0], T_GRID[-1])),
                        method="nearest")
        return np.where(np.isnan(q), near, q)

    return f
