"""
backtest_directional.py  —  VolGAN-BR Directional Volatility Backtest
======================================================================
Tests four signals for predicting next-day ATM IV direction (UP / DOWN)
on the 2024-2026 out-of-sample test set.

Signals
-------
  1. kNN        k-nearest neighbours in normalised IV-surface space.
                For each test day, find the k most similar days in the
                training set and take the majority next-day direction.
  2. MR-GAN     Mean-reversion toward the GAN fair-value ATM.
                If current ATM < GAN mean → predict UP, else DOWN.
  3. Momentum   Predict the same direction as the previous day's move.
  4. Always-UP  Naïve baseline: volatility always goes up.

Outputs
-------
  results/backtest_directional.png   — accuracy plot + confusion matrices
  results/backtest_directional.csv   — day-by-day prediction table
"""

import warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(0)

# ── scipy p-value (handle old and new API) ─────────────────────────────────────
try:
    from scipy.stats import binomtest as _bt
    def binom_pval(k, n): return _bt(k, n, 0.5).pvalue
except ImportError:
    from scipy.stats import binom_test as _bt
    def binom_pval(k, n): return _bt(k, n, 0.5)

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
BASE     = Path(__file__).parent
RES_DIR  = BASE / "results"

K_GRID   = np.array([-0.24, -0.18, -0.12, -0.06, 0.00, 0.06, 0.12, 0.18, 0.24])
T_GRID   = np.array([1/12, 2/12, 3/12])
N_K, N_T = len(K_GRID), len(T_GRID)       # 9, 3
SURF_DIM = N_K * N_T                       # 27

# ATM 1M flat index in the (N_K × N_T) C-order flattened surface
ATM_IDX  = (N_K // 2) * N_T + 0           # k=0, T=1/12  →  index 12

LATENT_DIM  = 32
H1, H2      = 128, 128
KNN_VALS    = [5, 10, 20]                  # k values to test
N_GEN       = 3000                         # synthetic surfaces for fair-value
ROLLING_WIN = 40                           # days for rolling accuracy plot

BLUE, RED, GOLD, GREEN = "#009c3b", "#e63946", "#FFDF00", "#2196F3"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print(" VolGAN-BR  |  Directional Backtest (next-day ATM IV direction)")
print("=" * 65)

print("\n[1/5] Loading surfaces ...")
X_train     = np.load(RES_DIR / "surfaces_train.npy")
X_test      = np.load(RES_DIR / "surfaces_test.npy")
dates_train = pd.to_datetime(pd.read_csv(RES_DIR / "dates_train.csv").iloc[:, 0])
dates_test  = pd.to_datetime(pd.read_csv(RES_DIR / "dates_test.csv").iloc[:, 0])

print(f"  Train : {len(X_train):,} surfaces  "
      f"({dates_train.min().date()} – {dates_train.max().date()})")
print(f"  Test  : {len(X_test):,} surfaces  "
      f"({dates_test.min().date()}  – {dates_test.max().date()})")

# ── Normalise (same stats as training) ────────────────────────────────────────
iv_min   = X_train.min(axis=0, keepdims=True)
iv_max   = X_train.max(axis=0, keepdims=True)
iv_rng   = np.where((iv_max - iv_min) < 1e-6, 1.0, iv_max - iv_min)
norm     = lambda X: (2 * (X - iv_min) / iv_rng - 1).astype(np.float32)
denorm   = lambda X: ((X + 1) / 2 * iv_rng + iv_min).astype(np.float32)

X_train_n = norm(X_train)
X_test_n  = norm(X_test)

# ── Sort training by date (for next-day lookups) ───────────────────────────────
sort_idx     = np.argsort(dates_train)
X_train_s    = X_train[sort_idx]
X_train_n_s  = X_train_n[sort_idx]
dates_train_s = dates_train.iloc[sort_idx].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Reconstruct Generator and generate fair-value cloud
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Reconstructing generator and computing GAN fair-value ...")

def leaky_relu(x, a=0.2): return np.where(x >= 0, x, a * x)

w = np.load(RES_DIR / "volgan_G_weights.npy", allow_pickle=True).item()

def generate(n):
    z  = np.random.randn(n, LATENT_DIM).astype(np.float32)
    h1 = leaky_relu(z  @ w["l1_W"] + w["l1_b"])
    h2 = leaky_relu(h1 @ w["l2_W"] + w["l2_b"])
    return np.tanh(h2 @ w["l3_W"] + w["l3_b"])

gen_n        = generate(N_GEN)
gen_real     = np.clip(denorm(gen_n), 0.02, 2.0)
gan_fair_atm = float(gen_real[:, ATM_IDX].mean())
print(f"  GAN fair-value ATM 1M : {gan_fair_atm:.4f}  ({gan_fair_atm*100:.2f}%)")
print(f"  Test-set mean ATM 1M  : {X_test[:, ATM_IDX].mean():.4f}  "
      f"({X_test[:, ATM_IDX].mean()*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Build ground-truth labels
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Building prediction pairs ...")

test_atm      = X_test[:, ATM_IDX]                        # (563,)
test_changes  = np.diff(test_atm)                          # (562,)
labels        = np.sign(test_changes)                      # +1=UP, -1=DOWN, 0=FLAT

# Exclude flat days (same ATM)
valid_mask    = labels != 0                                # (562,)
pred_idx      = np.where(valid_mask)[0]                   # indices into labels/changes

n_up   = (labels[pred_idx] == +1).sum()
n_down = (labels[pred_idx] == -1).sum()
print(f"  Prediction pairs : {len(pred_idx)}   "
      f"(UP={n_up}, DOWN={n_down}, base-rate UP={n_up/len(pred_idx):.1%})")

# ── Training next-day directions ───────────────────────────────────────────────
train_atm       = X_train_s[:, ATM_IDX]
train_next_dir  = np.full(len(X_train_s) - 1, np.nan)
for i in range(len(X_train_s) - 1):
    gap = (dates_train_s.iloc[i + 1] - dates_train_s.iloc[i]).days
    if gap <= 5:                                           # consecutive trading days
        train_next_dir[i] = np.sign(train_atm[i + 1] - train_atm[i])

valid_train = ~np.isnan(train_next_dir)
print(f"  Training next-day pairs: {valid_train.sum()}")

# Features and labels for kNN (training days with known next-day direction)
knn_X = X_train_n_s[:-1][valid_train]                     # (M, 27)
knn_y = train_next_dir[valid_train]                        # (M,)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Signals
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Computing signals ...")
true_dirs = labels[pred_idx]                               # ground truth for each pair

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# Signal 1: k-NN in normalised surface space
# ─────────────────────────────────────────────────────────────────────────────
print("  Signal 1: k-Nearest Neighbours ...")
knn_best_k, knn_best_acc, knn_best_preds = None, -1, None

for K in KNN_VALS:
    preds = np.empty(len(pred_idx), dtype=float)
    for out_i, test_i in enumerate(pred_idx):
        query  = X_test_n[test_i]                         # today's surface (27,)
        dists  = np.linalg.norm(knn_X - query, axis=1)    # L2 distances to all train days
        nn_idx = np.argpartition(dists, K)[:K]            # indices of k nearest
        vote   = knn_y[nn_idx].sum()                      # positive = majority UP
        preds[out_i] = 1.0 if vote >= 0 else -1.0         # tie-break → UP

    acc     = (preds == true_dirs).mean()
    n       = len(preds)
    k_corr  = int((preds == true_dirs).sum())
    pval    = binom_pval(k_corr, n)
    acc_up  = (preds[true_dirs == +1] == +1).mean()
    acc_dn  = (preds[true_dirs == -1] == -1).mean()
    results[f"kNN-{K}"] = dict(preds=preds, acc=acc, pval=pval,
                                acc_up=acc_up, acc_dn=acc_dn, n=n)
    print(f"    k={K:2d}: acc={acc:.1%}  (up={acc_up:.1%}, dn={acc_dn:.1%})  "
          f"p={pval:.4f}{'  ✓' if pval<0.05 else ''}")
    if acc > knn_best_acc:
        knn_best_acc, knn_best_k, knn_best_preds = acc, K, preds

# ─────────────────────────────────────────────────────────────────────────────
# Signal 2: GAN Mean-Reversion
# ─────────────────────────────────────────────────────────────────────────────
print("  Signal 2: GAN Mean-Reversion ...")
mr_preds = np.where(test_atm[pred_idx] < gan_fair_atm, 1.0, -1.0)
acc_mr   = (mr_preds == true_dirs).mean()
n        = len(mr_preds)
pval_mr  = binom_pval(int((mr_preds == true_dirs).sum()), n)
acc_up_mr = (mr_preds[true_dirs == +1] == +1).mean()
acc_dn_mr = (mr_preds[true_dirs == -1] == -1).mean()
results["MR-GAN"] = dict(preds=mr_preds, acc=acc_mr, pval=pval_mr,
                          acc_up=acc_up_mr, acc_dn=acc_dn_mr, n=n)
print(f"    acc={acc_mr:.1%}  (up={acc_up_mr:.1%}, dn={acc_dn_mr:.1%})  "
      f"p={pval_mr:.4f}{'  ✓' if pval_mr<0.05 else ''}")

# ─────────────────────────────────────────────────────────────────────────────
# Signal 3: Momentum (yesterday's direction)
# ─────────────────────────────────────────────────────────────────────────────
print("  Signal 3: Momentum baseline ...")
# pred_idx[j] = i means: feature=test[i], label=sign(test[i+1]-test[i])
# Yesterday's move for test day i is sign(test[i]-test[i-1])  →  requires i>=1
mom_mask    = pred_idx >= 1
mom_sub_idx = pred_idx[mom_mask]                          # test indices ≥ 1
mom_preds   = np.sign(test_changes[mom_sub_idx - 1])     # direction of day-before
# Drop flat prior days
nz          = mom_preds != 0
mom_true    = true_dirs[mom_mask][nz]
mom_preds   = mom_preds[nz]
n_mom       = len(mom_preds)
acc_mom     = (mom_preds == mom_true).mean()
pval_mom    = binom_pval(int((mom_preds == mom_true).sum()), n_mom)
acc_up_mom  = (mom_preds[mom_true == +1] == +1).mean()
acc_dn_mom  = (mom_preds[mom_true == -1] == -1).mean()
results["Momentum"] = dict(preds=mom_preds, acc=acc_mom, pval=pval_mom,
                            acc_up=acc_up_mom, acc_dn=acc_dn_mom, n=n_mom)
print(f"    acc={acc_mom:.1%}  (up={acc_up_mom:.1%}, dn={acc_dn_mom:.1%})  "
      f"p={pval_mom:.4f}  (n={n_mom})")

# ─────────────────────────────────────────────────────────────────────────────
# Signal 4: Always-UP baseline
# ─────────────────────────────────────────────────────────────────────────────
bull_acc   = float((true_dirs == +1).mean())
bull_preds = np.ones_like(true_dirs)
results["Always-UP"] = dict(preds=bull_preds, acc=bull_acc, pval=1.0,
                             acc_up=1.0, acc_dn=0.0, n=len(true_dirs))
print(f"  Signal 4: Always-UP baseline  acc={bull_acc:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# Rolling accuracy for best kNN and MR-GAN (aligned time axis)
# ─────────────────────────────────────────────────────────────────────────────
best_key  = max(results, key=lambda k: results[k]["acc"])
roll_preds_knn = results[f"kNN-{knn_best_k}"]["preds"]
roll_preds_mr  = mr_preds
correct_knn    = (roll_preds_knn == true_dirs).astype(float)
correct_mr     = (roll_preds_mr  == true_dirs).astype(float)
roll_dates     = dates_test.iloc[pred_idx + 1].reset_index(drop=True)  # prediction target date

roll_acc_knn = pd.Series(correct_knn).rolling(ROLLING_WIN, min_periods=5).mean()
roll_acc_mr  = pd.Series(correct_mr ).rolling(ROLLING_WIN, min_periods=5).mean()

# ── Export day-by-day table ───────────────────────────────────────────────────
tbl = pd.DataFrame({
    "date_feature"  : dates_test.iloc[pred_idx].values,
    "date_target"   : dates_test.iloc[pred_idx + 1].values,
    "atm_today"     : test_atm[pred_idx],
    "atm_tomorrow"  : test_atm[pred_idx + 1],
    "atm_change"    : test_changes[pred_idx],
    "true_dir"      : true_dirs,
    f"kNN{knn_best_k}_pred" : roll_preds_knn,
    f"kNN{knn_best_k}_corr" : (roll_preds_knn == true_dirs).astype(int),
    "MR_GAN_pred"   : mr_preds,
    "MR_GAN_corr"   : (mr_preds == true_dirs).astype(int),
})
tbl.to_csv(RES_DIR / "backtest_directional.csv", index=False)
print(f"\n  Saved: results/backtest_directional.csv  ({len(tbl)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Plot
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Plotting ...")

fig = plt.figure(figsize=(18, 13), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                         top=0.91, bottom=0.07, left=0.06, right=0.97)

# ── Panel 1: Bar chart of overall accuracy ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
labels_plot  = list(results.keys())
accs_plot    = [results[k]["acc"] for k in labels_plot]
pvals_plot   = [results[k]["pval"] for k in labels_plot]
colors_plot  = [BLUE if "kNN" in k else RED if k == "MR-GAN"
                else "#888888" if k == "Momentum" else "#cccccc" for k in labels_plot]
bars = ax1.bar(labels_plot, accs_plot, color=colors_plot, width=0.55, zorder=3)
ax1.axhline(bull_acc, color=GOLD, lw=1.5, ls="--", label=f"Always-UP={bull_acc:.1%}", zorder=2)
ax1.axhline(0.5,      color="black", lw=1.0, ls=":", label="50% chance", zorder=2)
for bar, acc, pv in zip(bars, accs_plot, pvals_plot):
    star = "★" if pv < 0.05 else ("·" if pv < 0.10 else "")
    ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.004,
             f"{acc:.1%}{star}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_ylim(0.35, 0.80)
ax1.set_ylabel("Directional Accuracy", fontsize=10)
ax1.set_title("Next-Day ATM IV Direction Accuracy  (2024–2026 out-of-sample)",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=8, loc="upper right"); ax1.grid(True, axis="y", alpha=0.3, ls="--")
ax1.set_facecolor("#f9f9f9")

# ── Panel 2: UP vs DOWN breakdown ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
x   = np.arange(len(labels_plot))
w   = 0.35
ax2.bar(x - w/2, [results[k]["acc_up"] for k in labels_plot], w,
        color=BLUE, alpha=0.85, label="Acc when true=UP")
ax2.bar(x + w/2, [results[k]["acc_dn"] for k in labels_plot], w,
        color=RED,  alpha=0.85, label="Acc when true=DOWN")
ax2.axhline(0.5, color="black", lw=0.8, ls=":")
ax2.set_xticks(x); ax2.set_xticklabels(labels_plot, fontsize=7, rotation=20)
ax2.set_ylim(0, 1.05); ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy by Direction", fontsize=10, fontweight="bold")
ax2.legend(fontsize=7); ax2.grid(True, axis="y", alpha=0.3, ls="--")
ax2.set_facecolor("#f9f9f9")

# ── Panel 3: Rolling accuracy over time ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(roll_dates, roll_acc_knn, color=BLUE, lw=1.8,
         label=f"kNN-{knn_best_k}  (acc={results[f'kNN-{knn_best_k}']['acc']:.1%})")
ax3.plot(roll_dates, roll_acc_mr,  color=RED,  lw=1.8, ls="--",
         label=f"MR-GAN  (acc={results['MR-GAN']['acc']:.1%})")
ax3.axhline(0.5,      color="black", lw=0.8, ls=":", alpha=0.7)
ax3.axhline(bull_acc, color=GOLD,   lw=1.0, ls="--", alpha=0.8, label=f"Always-UP ({bull_acc:.1%})")
ax3.fill_between(roll_dates, 0.5, roll_acc_knn,
                 where=(roll_acc_knn >= 0.5), alpha=0.12, color=BLUE)
ax3.fill_between(roll_dates, 0.5, roll_acc_knn,
                 where=(roll_acc_knn < 0.5),  alpha=0.12, color=RED)
ax3.set_ylabel(f"Rolling {ROLLING_WIN}-day accuracy"); ax3.set_xlabel("Date")
ax3.set_title(f"Rolling {ROLLING_WIN}-Day Directional Accuracy Over Time", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8, loc="upper left"); ax3.grid(True, alpha=0.25, ls="--")
ax3.set_facecolor("#f9f9f9")
fig.autofmt_xdate(rotation=30, ha="right")

# ── Panel 4–5: Confusion matrices ─────────────────────────────────────────────
def confusion_mat(preds, true, title, ax):
    """2×2 confusion matrix (true DOWN / UP  vs  pred DOWN / UP)."""
    tp = ((preds==1)&(true==1)).sum()  # predicted UP, actually UP
    fp = ((preds==1)&(true==-1)).sum()
    fn = ((preds==-1)&(true==1)).sum()
    tn = ((preds==-1)&(true==-1)).sum()
    mat = np.array([[tn, fp], [fn, tp]])
    im  = ax.imshow(mat, cmap="Blues", vmin=0)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred DOWN","Pred UP"], fontsize=8)
    ax.set_yticks([0,1]); ax.set_yticklabels(["True DOWN","True UP"], fontsize=8)
    for (r,c), v in np.ndenumerate(mat):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if v > mat.max()*0.6 else "black")
    ax.set_title(title, fontsize=9, fontweight="bold"); ax.set_facecolor("#f9f9f9")

ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
confusion_mat(roll_preds_knn, true_dirs, f"kNN-{knn_best_k} Confusion", ax4)
confusion_mat(mr_preds,       true_dirs, "MR-GAN Confusion",            ax5)

# ── Panel 6: Summary stats table ──────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2]); ax6.axis("off")
rows = [["Signal", "Acc", "p-val", "n"]] + [
    [k,
     f"{results[k]['acc']:.1%}",
     f"{results[k]['pval']:.4f}" + (" ★" if results[k]['pval'] < 0.05 else ""),
     str(results[k]["n"])]
    for k in results
]
tbl_ax = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tbl_ax.auto_set_font_size(False); tbl_ax.set_fontsize(8.5); tbl_ax.scale(1.1, 1.8)
ax6.set_title("Summary  (★ = p<0.05)", fontsize=9, fontweight="bold", pad=12)

fig.suptitle(
    "VolGAN-BR — Directional Volatility Backtest\n"
    "Predicting next-day ATM IV direction: UP or DOWN  |  Out-of-sample 2024–2026",
    fontsize=12, fontweight="bold", y=0.97)

out_path = RES_DIR / "backtest_directional.png"
fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: results/backtest_directional.png")

# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(" DIRECTIONAL BACKTEST SUMMARY  (2024–2026 out-of-sample)")
print("=" * 65)
print(f"  Prediction pairs : {len(true_dirs)}  "
      f"(UP={n_up}, DOWN={n_down}, base-rate={bull_acc:.1%})\n")
print(f"  {'Signal':<15}  {'Acc':>7}  {'p-val':>8}  {'Acc-UP':>8}  {'Acc-DN':>8}  {'n':>5}")
print(f"  {'-'*58}")
for k, v in results.items():
    star = " ★" if v["pval"] < 0.05 else ("  ·" if v["pval"] < 0.10 else "")
    print(f"  {k:<15}  {v['acc']:>7.1%}  {v['pval']:>8.4f}{star:3}  "
          f"{v['acc_up']:>8.1%}  {v['acc_dn']:>8.1%}  {v['n']:>5}")
print()
print(f"  Best signal: {best_key}  (acc={results[best_key]['acc']:.1%})")
print("=" * 65)
print("\n✅ Done.")
