"""
train_volgan_br_conditional.py  —  VolGAN-BR Conditional
=========================================================
Faithful port of Vuletic & Cont (2025) "VolGAN" to Brazilian IBOV options.

Architecture (from VolGAN.py, GitHub: milenavuletic/VolGAN)
-----------------------------------------------------------
  Generator  : [noise | condition] → Softplus → Softplus → output
  Discriminator: [condition | real/fake] → Softplus → Sigmoid → p
  Loss       : BCE (standard GAN) + smoothness penalty on reconstructed surface
  Penalty    : gradient-matched moneyness + maturity smoothness terms

Condition vector  (30 dims)
  [0]    r_{t-1}     annualised IBOV log-return at t-1
  [1]    r_{t-2}     annualised IBOV log-return at t-2
  [2]    RV21_{t-1}  21-day realised volatility of IBOV at t-1
  [3:30] log_IV_{t-1} yesterday's flattened log-IV surface (27 dims)

Target  (28 dims = 1 return + 27 Δlog_IV)
  [0]    r_t          today's annualised IBOV log-return
  [1:28] Δlog_IV_t    log_IV_t − log_IV_{t-1}  (stationary!)

Directional backtest
  For each test day t, generate M conditional samples.
  Sign of median predicted ΔATM IV = directional forecast.
"""

import warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    from scipy.stats import binomtest as _bt
    def binom_pval(k, n): return _bt(k, n, 0.5).pvalue
except ImportError:
    from scipy.stats import binom_test as _bt
    def binom_pval(k, n): return _bt(k, n, 0.5)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
BASE    = Path(__file__).parent
RES_DIR = BASE / "results"
RES_DIR.mkdir(exist_ok=True)

K_GRID    = np.array([-0.24, -0.18, -0.12, -0.06, 0.00, 0.06, 0.12, 0.18, 0.24])
T_GRID    = np.array([1/12, 2/12, 3/12])
N_K, N_T  = len(K_GRID), len(T_GRID)
SURF_DIM  = N_K * N_T          # 27
ATM_IDX   = (N_K // 2) * N_T   # k=0, T=1/12 → flat index 12

NOISE_DIM = 32
COND_DIM  = 3 + SURF_DIM       # r_{t-1}, r_{t-2}, RV21, log_IV_{t-1} → 30
OUT_DIM   = 1 + SURF_DIM       # r_t, Δlog_IV_t → 28

H1        = 128                 # G hidden width (×2 in second layer = 256)
H_D       = 128                 # D hidden width (single hidden layer)

N_EPOCHS   = 300
BATCH_SIZE = 64
LR_G       = 5e-5
LR_D       = 5e-5
N_CRIT     = 5                  # WGAN: 5 critic steps per generator step
CLIP_VAL   = 0.01               # WGAN weight clipping
N_SAMPLES  = 500                # conditional samples per test day

ALPHA_SM   = 0.005             # moneyness smoothness weight
BETA_SM    = 0.005             # maturity smoothness weight

# ══════════════════════════════════════════════════════════════════════════════
# Activations
# ══════════════════════════════════════════════════════════════════════════════
def softplus(x):
    return np.log1p(np.exp(np.clip(x, -500, 30)))

def softplus_grad(x):
    # d/dx softplus(x) = sigmoid(x)
    return 1.0 / (1.0 + np.exp(np.clip(-x, -500, 500)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -500, 500)))

# ══════════════════════════════════════════════════════════════════════════════
# Linear layer with Adam
# ══════════════════════════════════════════════════════════════════════════════
BETA1, BETA2, EPS = 0.5, 0.9, 1e-8

class Linear:
    def __init__(self, in_d, out_d, scale=0.02):
        self.W  = (np.random.randn(in_d, out_d) * scale).astype(np.float32)
        self.b  = np.zeros(out_d, dtype=np.float32)
        self.mW = self.vW = np.zeros_like(self.W)
        self.mb = self.vb = np.zeros_like(self.b)
        self.t  = 0

    def forward(self, x):
        self._x = x
        return (x @ self.W + self.b).astype(np.float32)

    def backward(self, g):
        B = self._x.shape[0]
        self.gW = self._x.T @ g / B
        self.gb = g.mean(axis=0)
        return g @ self.W.T

    def adam(self, lr):
        self.t += 1
        for (m, v, g, p) in [(self.mW, self.vW, self.gW, 'W'),
                              (self.mb, self.vb, self.gb, 'b')]:
            attr_m = 'm' + p; attr_v = 'v' + p
            setattr(self, attr_m, BETA1 * m + (1 - BETA1) * g)
            setattr(self, attr_v, BETA2 * v + (1 - BETA2) * g**2)
            mh = getattr(self, attr_m) / (1 - BETA1**self.t)
            vh = getattr(self, attr_v) / (1 - BETA2**self.t)
            setattr(self, p, getattr(self, p) - lr * mh / (np.sqrt(vh) + EPS))

# ══════════════════════════════════════════════════════════════════════════════
# Generator  (Vuletic architecture: noise+cond → H → 2H → out, Softplus)
# ══════════════════════════════════════════════════════════════════════════════
class Generator:
    def __init__(self):
        self.l1 = Linear(NOISE_DIM + COND_DIM, H1)
        self.l2 = Linear(H1, H1 * 2)
        self.l3 = Linear(H1 * 2, OUT_DIM)

    def forward(self, noise, cond):
        x        = np.concatenate([noise, cond], axis=-1)
        pre1     = self.l1.forward(x);   self._pre1 = pre1;  h1 = softplus(pre1)
        pre2     = self.l2.forward(h1);  self._pre2 = pre2;  h2 = softplus(pre2)
        out      = self.l3.forward(h2)   # linear output (increments, any sign)
        return out.astype(np.float32)

    def backward(self, grad_out):
        g = self.l3.backward(grad_out)
        g = g * softplus_grad(self._pre2)
        g = self.l2.backward(g)
        g = g * softplus_grad(self._pre1)
        self.l1.backward(g)

    def adam(self):
        for l in [self.l1, self.l2, self.l3]: l.adam(LR_G)

    def sample(self, cond_batch):
        """Generate fake outputs conditioned on cond_batch (n, COND_DIM)."""
        n  = cond_batch.shape[0]
        z  = np.random.randn(n, NOISE_DIM).astype(np.float32)
        return self.forward(z, cond_batch)

# ══════════════════════════════════════════════════════════════════════════════
# Critic  (WGAN: cond+output → H → H → scalar, no activation, weight clipping)
# Same conditional structure as Vuletic's discriminator but WGAN loss for stability
# ══════════════════════════════════════════════════════════════════════════════
class Critic:
    def __init__(self):
        self.l1 = Linear(COND_DIM + OUT_DIM, H_D)
        self.l2 = Linear(H_D, H_D)
        self.l3 = Linear(H_D, 1)

    def forward(self, x):
        # x: (B, COND_DIM+OUT_DIM) — pre-concatenated [cond | output]
        pre1 = self.l1.forward(x);  self._pre1 = pre1;  h1 = softplus(pre1)
        pre2 = self.l2.forward(h1); self._pre2 = pre2;  h2 = softplus(pre2)
        return self.l3.forward(h2)                        # (B, 1), linear

    def backward(self, grad_out):
        g = self.l3.backward(grad_out)
        g = g * softplus_grad(self._pre2)
        g = self.l2.backward(g)
        g = g * softplus_grad(self._pre1)
        return self.l1.backward(g)                        # (B, COND+OUT)

    def adam(self):
        for l in [self.l1, self.l2, self.l3]: l.adam(LR_D)

    def clip(self):
        for l in [self.l1, self.l2, self.l3]:
            l.W = np.clip(l.W, -CLIP_VAL, CLIP_VAL)
            l.b = np.clip(l.b, -CLIP_VAL, CLIP_VAL)

# ══════════════════════════════════════════════════════════════════════════════
# Smoothness penalty on the reconstructed log-IV surface
# Penalty = sum of squared finite differences in k and T directions
# ══════════════════════════════════════════════════════════════════════════════
# Pre-compute finite-difference weights (same as Vuletic's matrix_m, matrix_t)
# Δk and ΔT spacings
dk = np.diff(K_GRID)       # (N_K-1,) all equal = 0.06
dt = np.diff(T_GRID)       # (N_T-1,) all equal = 1/12

def smoothness_penalty_and_grad(fake_delta_iv, log_iv_tm1_batch):
    """
    fake_delta_iv : (B, SURF_DIM)  — Δlog_IV (generator output, [1:])
    log_iv_tm1_batch: (B, SURF_DIM) — log_IV_{t-1} (from condition [3:])
    Returns: (scalar penalty, grad w.r.t. fake_delta_iv (B, SURF_DIM))
    """
    B = fake_delta_iv.shape[0]
    # Reconstruct surface level: IV_t = exp(log_IV_{t-1} + Δlog_IV_t)
    log_iv_t   = log_iv_tm1_batch + fake_delta_iv          # (B, 27)
    iv_t       = np.exp(np.clip(log_iv_t, -4, 2))          # (B, 27)
    iv_3d      = iv_t.reshape(B, N_K, N_T)                 # (B, 9, 3)

    # Moneyness smoothness: squared diff along k axis
    diff_m     = np.diff(iv_3d, axis=1)                    # (B, 8, 3)
    pen_m      = float(np.mean(diff_m**2))

    # Maturity smoothness: squared diff along T axis
    diff_t     = np.diff(iv_3d, axis=2)                    # (B, 9, 2)
    pen_t      = float(np.mean(diff_t**2))

    penalty    = ALPHA_SM * pen_m + BETA_SM * pen_t

    # Gradient w.r.t. iv_3d
    grad_iv3d  = np.zeros_like(iv_3d)
    # from pen_m
    g_diff_m   = (2 * ALPHA_SM / (B * diff_m.size)) * diff_m   # (B, 8, 3)
    grad_iv3d[:, 1:,  :] += g_diff_m
    grad_iv3d[:, :-1, :] -= g_diff_m
    # from pen_t
    g_diff_t   = (2 * BETA_SM  / (B * diff_t.size)) * diff_t   # (B, 9, 2)
    grad_iv3d[:, :, 1:]  += g_diff_t
    grad_iv3d[:, :, :-1] -= g_diff_t

    # Chain rule through exp: grad_log_iv_t = grad_iv3d * iv_3d
    grad_log_iv_t = (grad_iv3d * iv_3d).reshape(B, SURF_DIM)

    # Δlog_IV and log_IV_{t-1} add identically → grad passes straight through
    return penalty, grad_log_iv_t.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(" VolGAN-BR Conditional  |  following Vuletic & Cont (2025)")
print("=" * 65)
print("\n[1/5] Loading data ...")

X_train     = np.load(RES_DIR / "surfaces_train.npy")
X_test      = np.load(RES_DIR / "surfaces_test.npy")
dates_train = pd.to_datetime(pd.read_csv(RES_DIR / "dates_train.csv").iloc[:, 0])
dates_test  = pd.to_datetime(pd.read_csv(RES_DIR / "dates_test.csv").iloc[:, 0])

# Combine and sort (so lookback can cross train/test boundary)
X_all       = np.concatenate([X_train, X_test], axis=0)
dates_all   = pd.concat([dates_train, dates_test], ignore_index=True)
is_train_all= np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))]).astype(bool)

sidx        = np.argsort(dates_all)
X_all       = X_all[sidx]
dates_all   = dates_all.iloc[sidx].reset_index(drop=True)
is_train_all= is_train_all[sidx]

log_IV_all  = np.log(np.maximum(X_all, 1e-4)).astype(np.float32)  # (N, 27)

# ── IBOV proxy returns from vxbr replication ──────────────────────────────────
vxbr_path = RES_DIR / "vxbr_replication.csv"
if vxbr_path.exists():
    vxbr_df  = pd.read_csv(vxbr_path, parse_dates=["date"])
    spot_map = dict(zip(vxbr_df["date"], vxbr_df["ibov_spot_est"]))
else:
    spot_map = {}

spots = np.array([spot_map.get(pd.Timestamp(d), np.nan) for d in dates_all])
# Forward-fill missing (and initial NaN → use ATM IV level as fallback)
atm_iv = X_all[:, ATM_IDX]
for i in range(len(spots)):
    if np.isnan(spots[i]) or spots[i] <= 0:
        spots[i] = atm_iv[i] * 10000   # scale so log-returns are comparable

# Daily annualised log-returns
log_ret = np.zeros(len(spots), dtype=np.float32)
for i in range(1, len(spots)):
    if spots[i] > 0 and spots[i-1] > 0:
        log_ret[i] = float(np.sqrt(252) * np.log(spots[i] / spots[i-1]))
log_ret = np.clip(log_ret, -5, 5)

# 21-day realised volatility
rv21 = np.zeros(len(spots), dtype=np.float32)
for i in range(21, len(spots)):
    rv21[i] = float(np.sqrt(252) * np.sqrt(np.mean(log_ret[i-21:i]**2)))

print(f"  Combined series: {len(X_all)} days  "
      f"({dates_all.iloc[0].date()} – {dates_all.iloc[-1].date()})")

# ══════════════════════════════════════════════════════════════════════════════
# Build condition / target pairs
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Building condition/target pairs ...")

conditions, targets, pair_dates, pair_train = [], [], [], []

for t in range(22, len(X_all)):
    # Only use consecutive trading days
    gap = (dates_all.iloc[t] - dates_all.iloc[t-1]).days
    if gap > 5: continue

    cond = np.concatenate([
        [log_ret[t-1]],         # r_{t-1}
        [log_ret[t-2]],         # r_{t-2}
        [rv21[t-1]],            # RV21_{t-1}
        log_IV_all[t-1]         # log_IV_{t-1}  (27 dims)
    ]).astype(np.float32)       # (30,)

    tgt = np.concatenate([
        [log_ret[t]],           # r_t
        log_IV_all[t] - log_IV_all[t-1]   # Δlog_IV_t  (27 dims)
    ]).astype(np.float32)       # (28,)

    conditions.append(cond)
    targets.append(tgt)
    pair_dates.append(dates_all.iloc[t])
    pair_train.append(bool(is_train_all[t]))

conditions  = np.array(conditions,  dtype=np.float32)
targets     = np.array(targets,     dtype=np.float32)
pair_dates  = pd.to_datetime(pair_dates)
pair_train  = np.array(pair_train,  dtype=bool)

cond_tr  = conditions[ pair_train]
tgt_tr   = targets[    pair_train]
cond_te  = conditions[~pair_train]
tgt_te   = targets[   ~pair_train]
dates_te = pair_dates[~pair_train]

n_up   = int((np.sign(tgt_te[:, 1 + ATM_IDX]) == +1).sum())
n_dn   = int((np.sign(tgt_te[:, 1 + ATM_IDX]) == -1).sum())
print(f"  Training pairs : {len(cond_tr)}")
print(f"  Test pairs     : {len(cond_te)}  (UP={n_up}, DOWN={n_dn})")

# ── Normalise ─────────────────────────────────────────────────────────────────
cond_mean = cond_tr.mean(axis=0, keepdims=True)
cond_std  = np.where(cond_tr.std(axis=0, keepdims=True) < 1e-6,
                     1.0, cond_tr.std(axis=0, keepdims=True))
tgt_mean  = tgt_tr.mean(axis=0, keepdims=True)
tgt_std   = np.where(tgt_tr.std(axis=0, keepdims=True) < 1e-6,
                     1.0, tgt_tr.std(axis=0, keepdims=True))

cond_tr_n = ((cond_tr - cond_mean) / cond_std).astype(np.float32)
tgt_tr_n  = ((tgt_tr  - tgt_mean)  / tgt_std ).astype(np.float32)
cond_te_n = ((cond_te - cond_mean) / cond_std).astype(np.float32)

# log_IV_{t-1} part of condition in normalized space (cols 3:30)
# We still need it in raw log-IV space for the smoothness penalty → keep cond_tr raw

# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[3/5] Training conditional WGAN  ({N_EPOCHS} epochs) ...")
G = Generator(); C = Critic()
g_losses, c_losses = [], []
n_batches = max(1, len(cond_tr_n) // BATCH_SIZE)
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):
    perm    = np.random.permutation(len(cond_tr_n))
    Cn, Tn  = cond_tr_n[perm], tgt_tr_n[perm]
    Cr      = cond_tr[perm]     # raw condition (for smoothness penalty)
    g_ep = c_ep = 0.0

    for b in range(n_batches):
        sl   = slice(b * BATCH_SIZE, (b+1) * BATCH_SIZE)
        c_n  = Cn[sl]; t_n = Tn[sl]; B = len(c_n)
        if B == 0: continue
        log_iv_prev = Cr[sl, 3:].astype(np.float32)   # raw log_IV_{t-1}

        # ── Critic updates (WGAN: N_CRIT steps) ─────────────────────────────
        for _ in range(N_CRIT):
            z    = np.random.randn(B, NOISE_DIM).astype(np.float32)
            fake = G.forward(z, c_n)                   # (B, 28)

            # Combined real+fake batch for a single backward pass (avoids grad overwrite)
            real_in = np.concatenate([c_n, t_n],   axis=-1)   # (B, 58)
            fake_in = np.concatenate([c_n, fake],  axis=-1)   # (B, 58)
            combined = np.concatenate([real_in, fake_in], axis=0)  # (2B, 58)

            scores = C.forward(combined)               # (2B, 1)
            w_loss = scores[B:].mean() - scores[:B].mean()
            c_ep  += float(w_loss)

            grad_c = np.concatenate([
                -np.ones((B, 1), dtype=np.float32) / B,   # real: push score UP
                 np.ones((B, 1), dtype=np.float32) / B,   # fake: push score DOWN
            ], axis=0)                                     # (2B, 1)
            C.backward(grad_c)
            C.adam()
            C.clip()

        # ── Generator update ─────────────────────────────────────────────────
        z    = np.random.randn(B, NOISE_DIM).astype(np.float32)
        fake = G.forward(z, c_n)                       # (B, 28)

        fake_in = np.concatenate([c_n, fake], axis=-1) # (B, 58)
        C.forward(fake_in)
        g_adv = -C.forward(fake_in).mean()             # maximise C(fake)

        # Smoothness penalty on Δlog_IV part (fake[:, 1:])
        pen, grad_pen = smoothness_penalty_and_grad(fake[:, 1:], log_iv_prev)
        g_ep += float(g_adv) + pen

        # Wasserstein gradient: -1/B through critic → generator
        grad_all = C.backward(-np.ones((B, 1), dtype=np.float32) / B)  # (B, 58)
        grad_out = grad_all[:, COND_DIM:].copy()       # (B, 28) — output dims only
        grad_out[:, 1:] += grad_pen                    # add smoothness gradient
        G.backward(grad_out)
        G.adam()

    g_losses.append(g_ep / n_batches)
    c_losses.append(c_ep / n_batches / N_CRIT)

    if epoch % 30 == 0 or epoch == 1:
        print(f"  Epoch {epoch:4d}/{N_EPOCHS}  "
              f"C={c_losses[-1]:+.4f}  G={g_losses[-1]:+.4f}  "
              f"[{time.time()-t0:.0f}s]")

print(f"\n  ✅ Training done in {time.time()-t0:.0f}s")

# Save weights
np.save(RES_DIR / "volgan_cond_G_weights.npy",
        {"l1_W": G.l1.W, "l1_b": G.l1.b,
         "l2_W": G.l2.W, "l2_b": G.l2.b,
         "l3_W": G.l3.W, "l3_b": G.l3.b}, allow_pickle=True)
np.save(RES_DIR / "volgan_cond_norm.npy",
        {"cond_mean": cond_mean, "cond_std": cond_std,
         "tgt_mean":  tgt_mean,  "tgt_std":  tgt_std}, allow_pickle=True)

# ══════════════════════════════════════════════════════════════════════════════
# Directional backtest
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Directional backtest on test set ...")

# ATM Δlog_IV direction: index 1+ATM_IDX = 1+12 = 13 in the 28-dim target
ATM_DELTA_IDX = 1 + ATM_IDX   # = 13

true_delta = tgt_te[:, ATM_DELTA_IDX]    # actual Δlog_ATM_IV_t
true_dirs  = np.sign(true_delta)

# Generate N_SAMPLES conditional forecasts for each test day
pred_dirs  = np.zeros(len(cond_te_n), dtype=float)
pred_delta_mean = np.zeros(len(cond_te_n), dtype=float)

for i in range(len(cond_te_n)):
    c_i  = np.tile(cond_te_n[i], (N_SAMPLES, 1)).astype(np.float32)
    z    = np.random.randn(N_SAMPLES, NOISE_DIM).astype(np.float32)
    samp = G.forward(z, c_i)                        # (N_SAMPLES, 28) normalised
    # Denormalise ATM delta
    samp_dn = samp * tgt_std + tgt_mean             # (N_SAMPLES, 28)
    atm_d   = samp_dn[:, ATM_DELTA_IDX]             # (N_SAMPLES,)
    pred_delta_mean[i] = float(atm_d.mean())
    pred_dirs[i]       = float(np.sign(atm_d.mean()))

# Exclude flat days (true=0)
valid     = true_dirs != 0
n_valid   = int(valid.sum())
n_up      = int((true_dirs[valid] ==  1).sum())
n_dn      = int((true_dirs[valid] == -1).sum())
acc       = float((pred_dirs[valid] == true_dirs[valid]).mean())
k_corr    = int((pred_dirs[valid] == true_dirs[valid]).sum())
pval      = binom_pval(k_corr, n_valid)
acc_up    = float((pred_dirs[valid & (true_dirs ==  1)] ==  1).mean())
acc_dn    = float((pred_dirs[valid & (true_dirs == -1)] == -1).mean())

print(f"\n  {'='*50}")
print(f"  cVolGAN Directional Accuracy (2024–2026 OOS)")
print(f"  {'='*50}")
print(f"  Pairs   : {n_valid}  (UP={n_up}, DOWN={n_dn})")
print(f"  Accuracy: {acc:.1%}   (UP acc={acc_up:.1%}, DOWN acc={acc_dn:.1%})")
print(f"  p-value : {pval:.4f}{'  ★ significant' if pval < 0.05 else ''}")

# Rolling accuracy
roll_correct = pd.Series((pred_dirs[valid] == true_dirs[valid]).astype(float))
roll_acc     = roll_correct.rolling(40, min_periods=5).mean()
roll_dates_valid = dates_te[valid]

# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Plotting ...")

BLUE, RED, GOLD = "#009c3b", "#e63946", "#FFDF00"

fig = plt.figure(figsize=(16, 12), facecolor="white")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35,
                         top=0.91, bottom=0.07, left=0.07, right=0.96)

# Training losses
ax1 = fig.add_subplot(gs[0, :])
ep  = np.arange(1, len(g_losses) + 1)
ax1.plot(ep, c_losses, color=RED,  lw=1.5, label="Critic Wasserstein loss")
ax1.plot(ep, g_losses, color=BLUE, lw=1.5, label="Generator loss (Wasserstein + smoothness)")
ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5, label="0 = Wasserstein equilibrium")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("cVolGAN Training Losses — Conditional GAN (Vuletic architecture)",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, ls="--"); ax1.set_facecolor("#f9f9f9")

# Rolling directional accuracy
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(roll_dates_valid, roll_acc, color=BLUE, lw=2,
         label=f"cVolGAN  acc={acc:.1%}  p={pval:.4f}"
               + ("  ★" if pval < 0.05 else ""))
ax2.axhline(0.5,    color="black", lw=0.8, ls=":", alpha=0.7, label="50% (random)")
ax2.axhline(n_up / n_valid, color=GOLD, lw=1.0, ls="--", alpha=0.8,
            label=f"Always-UP ({n_up/n_valid:.1%})")
ax2.fill_between(roll_dates_valid, 0.5, roll_acc,
                 where=(roll_acc >= 0.5), alpha=0.15, color=BLUE)
ax2.fill_between(roll_dates_valid, 0.5, roll_acc,
                 where=(roll_acc < 0.5),  alpha=0.15, color=RED)
ax2.set_ylabel("Rolling 40-day accuracy"); ax2.set_xlabel("Date")
ax2.set_title("Rolling Directional Accuracy — cVolGAN vs Unconditional baseline",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25, ls="--"); ax2.set_facecolor("#f9f9f9")
fig.autofmt_xdate(rotation=30)

# Mean predicted vs actual Δlog_ATM_IV
ax3 = fig.add_subplot(gs[2, 0])
true_delta_valid = true_delta[valid]
pred_delta_valid = pred_delta_mean[valid]
ax3.scatter(true_delta_valid, pred_delta_valid, s=8, alpha=0.4,
            color=np.where(np.sign(pred_delta_valid) == np.sign(true_delta_valid),
                           BLUE, RED))
lim = max(abs(true_delta_valid).max(), abs(pred_delta_valid).max()) * 1.1
ax3.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
ax3.axhline(0, color="black", lw=0.5); ax3.axvline(0, color="black", lw=0.5)
ax3.set_xlabel("Actual Δlog_ATM_IV"); ax3.set_ylabel("Predicted mean Δlog_ATM_IV")
ax3.set_title("Predicted vs Actual ATM IV Increment\n(blue=correct dir, red=wrong)",
              fontsize=9, fontweight="bold")
ax3.grid(True, alpha=0.3, ls="--"); ax3.set_facecolor("#f9f9f9")

# Summary stats table
ax4 = fig.add_subplot(gs[2, 1]); ax4.axis("off")
rows = [
    ["Metric", "Value"],
    ["Model", "cVolGAN (Vuletic arch.)"],
    ["Train period", "2012–2023"],
    ["Test period", "2024–2026"],
    ["Test pairs", str(n_valid)],
    ["Directional accuracy", f"{acc:.1%}"],
    ["Acc when true=UP", f"{acc_up:.1%}"],
    ["Acc when true=DOWN", f"{acc_dn:.1%}"],
    ["p-value (vs 50%)", f"{pval:.4f}" + (" ★" if pval < 0.05 else "")],
    ["Samples per day", str(N_SAMPLES)],
]
tbl = ax4.table(cellText=rows[1:], colLabels=rows[0],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.3, 1.9)
ax4.set_title("cVolGAN Summary", fontsize=10, fontweight="bold", pad=12)

fig.suptitle(
    "VolGAN-BR Conditional  |  Following Vuletic & Cont (2025)\n"
    "Condition: [r_{t-1}, r_{t-2}, RV21_{t-1}, log_IV_{t-1}]  →  Predicts Δlog_IV_t",
    fontsize=11, fontweight="bold", y=0.97)

out = RES_DIR / "volgan_conditional_backtest.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"  Saved: {out}")
print("\n✅ Done.")
