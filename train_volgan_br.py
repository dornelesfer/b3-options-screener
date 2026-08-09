"""
train_volgan_br.py  —  VolGAN for Brazilian IBOV Index Options
==============================================================
Three self-contained stages:

  Stage 1  Surface extraction
           Read COTAHIST parquet → Black-76 IV per option →
           interpolate onto fixed (k, T) grid → save [N × 27] array.

  Stage 2  WGAN training  (pure numpy — no PyTorch required)
           Wasserstein GAN with weight clipping (Arjovsky et al. 2017).
           Same theoretical guarantees as WGAN-GP; simpler to implement
           without autograd. Architecture: MLP G and C, Adam updates.

  Stage 3  Evaluation
           Real vs generated: IV smile shape, ATM distribution, skew,
           term-structure slope, PCA projection.

Grid
----
  k : 9 points  [-0.24, -0.18, -0.12, -0.06, 0, 0.06, 0.12, 0.18, 0.24]
  T : 3 tenors  [1/12, 2/12, 3/12]  years   → 27-dim surface vector

Training / test split
  Train : 2012–2023   (IBOV options market had consistent liquidity)
  Test  : 2024–2026   (out-of-sample; covers VXBR official launch period)
"""

import warnings, os, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.special import ndtr          # fast normal CDF
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data" / "rb3_repository" / "db" / "staging" / "b3-cotahist-yearly"
OUT_DIR  = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

K_GRID = np.array([-0.24, -0.18, -0.12, -0.06, 0.00, 0.06, 0.12, 0.18, 0.24])
T_GRID = np.array([1/12, 2/12, 3/12])
N_K, N_T = len(K_GRID), len(T_GRID)
SURF_DIM  = N_K * N_T   # 27

TRAIN_YEARS = list(range(2012, 2024))
TEST_YEARS  = list(range(2024, 2027))

R_BRAZIL    = 0.12
MIN_STRIKES = 4          # minimum strikes per expiry for a valid smile
NEEDED_COLS = ["refdate", "bdi_code", "specification_code",
               "strike_price", "close", "best_bid", "best_ask",
               "maturity_date"]

# GAN hyperparameters
LATENT_DIM  = 32
H1, H2      = 128, 128   # hidden layer widths
N_CRITIC    = 5          # critic steps per generator step
CLIP_VAL    = 0.01       # weight clipping for WGAN
LR          = 5e-5
BETA1,BETA2 = 0.5, 0.9
BATCH_SIZE  = 32
N_EPOCHS    = 300

# ══════════════════════════════════════════════════════════════════════════════
# Helpers: Black-76, put-call parity forward, smile-to-grid
# ══════════════════════════════════════════════════════════════════════════════

def b76(F, K, T, r, sigma, call):
    d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    disc = np.exp(-r*T)
    return disc*(F*ndtr(d1)-K*ndtr(d2)) if call else disc*(K*ndtr(-d2)-F*ndtr(-d1))

def iv_bisect(price, F, K, T, r, call, lo=1e-4, hi=10.0, tol=1e-5):
    if T<=0 or price<=0 or F<=0 or K<=0: return np.nan
    flo = b76(F,K,T,r,lo,call) - price
    fhi = b76(F,K,T,r,hi,call) - price
    if flo*fhi > 0: return np.nan
    for _ in range(150):
        mid = (lo+hi)/2; fm = b76(F,K,T,r,mid,call) - price
        if abs(fm)<tol or (hi-lo)/2<tol: return mid
        if flo*fm<0: hi=mid; fhi=fm
        else:        lo=mid; flo=fm
    return (lo+hi)/2

def forward_pcp(calls, puts, T, r):
    pairs = calls[["strike_price","mid"]].rename(columns={"mid":"C"}).merge(
            puts [["strike_price","mid"]].rename(columns={"mid":"P"}), on="strike_price")
    if len(pairs)==0: return np.nan
    pairs["F"] = pairs["strike_price"] + np.exp(r*T)*(pairs["C"]-pairs["P"])
    return float(pairs.loc[(pairs["C"]-pairs["P"]).abs().idxmin(),"F"])

def day_smile(day_data, r=R_BRAZIL):
    day_data = day_data.copy()
    day_data["T_exp"] = (pd.to_datetime(day_data["maturity_date"]) -
                         pd.to_datetime(day_data["refdate"].iloc[0])).dt.days / 365.0
    day_data = day_data[(day_data["close"]>0) & (day_data["T_exp"]>6/252)].copy()
    day_data["mid"] = np.where(
        (day_data["best_bid"]>0) & (day_data["best_ask"]>0),
        (day_data["best_bid"]+day_data["best_ask"])/2, day_data["close"])
    rows=[]
    for exp, grp in day_data.groupby("maturity_date"):
        T = grp["T_exp"].iloc[0]
        F = forward_pcp(grp[grp["bdi_code"]==74], grp[grp["bdi_code"]==75], T, r)
        if np.isnan(F) or F<=0: continue
        for _, opt in grp.iterrows():
            k  = np.log(opt["strike_price"]/F)
            iv = iv_bisect(opt["mid"], F, opt["strike_price"], T, r,
                           call=(opt["bdi_code"]==74))
            if np.isnan(iv) or iv<=0 or iv>4: continue
            rows.append({"k":k,"T_exp":T,"iv":iv})
    return pd.DataFrame(rows)

def smile_to_grid(smile_df):
    grid = np.full((N_K, N_T), np.nan)
    expiries = sorted(smile_df["T_exp"].unique())
    exp_smiles={}
    for T_exp in expiries:
        sub = smile_df[smile_df["T_exp"]==T_exp].sort_values("k")
        if len(sub)<MIN_STRIKES: continue
        try:
            f = interp1d(sub["k"], sub["iv"], kind="linear",
                         bounds_error=False,
                         fill_value=(sub["iv"].iloc[0], sub["iv"].iloc[-1]))
            exp_smiles[T_exp] = f(K_GRID)
        except: continue
    if not exp_smiles: return None
    exp_T   = np.array(sorted(exp_smiles)); exp_ivs = np.array([exp_smiles[t] for t in exp_T])
    for ki in range(N_K):
        iv_col = exp_ivs[:,ki]; valid = ~np.isnan(iv_col)
        if valid.sum()==0: continue
        if valid.sum()==1: grid[ki,:] = iv_col[valid][0]
        else:
            f = interp1d(exp_T[valid], iv_col[valid], kind="linear",
                         bounds_error=False, fill_value=(iv_col[valid][0],iv_col[valid][-1]))
            grid[ki,:] = f(T_GRID)
    if np.isnan(grid).mean()>0.3: return None
    for ti in range(N_T):
        col=grid[:,ti]
        if np.isnan(col).any() and not np.isnan(col).all():
            grid[np.isnan(grid[:,ti]),ti] = np.nanmean(col)
    if np.isnan(grid).any(): return None
    return grid

def extract_surfaces(years, label):
    surfaces, dates = [], []
    for year in years:
        path = DATA_DIR / f"year={year}" / "part-0.parquet"
        if not path.exists(): continue
        df = pq.read_table(str(path), columns=NEEDED_COLS).to_pandas()
        ibov = df[df["specification_code"].str.strip().str.startswith("IBO") &
                  df["bdi_code"].isin([74,75]) & (df["strike_price"]>0) & (df["close"]>0)].copy()
        del df
        if len(ibov)==0: continue
        ibov["refdate"] = pd.to_datetime(ibov["refdate"])
        ibov["maturity_date"] = pd.to_datetime(ibov["maturity_date"])
        hits=0
        for day, grp in ibov.groupby("refdate"):
            smile = day_smile(grp)
            if len(smile)<MIN_STRIKES*2: continue
            grid = smile_to_grid(smile)
            if grid is None: continue
            surfaces.append(grid.flatten()); dates.append(day); hits+=1
        print(f"   {year}: {hits:3d} surfaces")
        del ibov
    X = np.array(surfaces, dtype=np.float32)
    print(f"   [{label}] total: {len(X)} surfaces  shape {X.shape}")
    return X, pd.to_datetime(dates)

# ══════════════════════════════════════════════════════════════════════════════
# Numpy WGAN  (no autograd — all gradients by hand / chain rule)
# ══════════════════════════════════════════════════════════════════════════════

def leaky_relu(x, a=0.2): return np.where(x>=0, x, a*x)
def leaky_relu_grad(x, a=0.2): return np.where(x>=0, 1.0, a)
def tanh(x): return np.tanh(x)
def tanh_grad(x): return 1 - np.tanh(x)**2

class Linear:
    """Single fully-connected layer with Adam state."""
    def __init__(self, in_d, out_d, scale=0.02):
        self.W = np.random.randn(in_d, out_d).astype(np.float32) * scale
        self.b = np.zeros(out_d, dtype=np.float32)
        # Adam moments
        self.mW=self.vW=np.zeros_like(self.W)
        self.mb=self.vb=np.zeros_like(self.b)
        self.t = 0

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b         # [B, out]

    def backward(self, grad_out):
        B = self._x.shape[0]
        self.gW = self._x.T @ grad_out / B
        self.gb = grad_out.mean(axis=0)
        return grad_out @ self.W.T         # grad w.r.t. input

    def adam_step(self, lr, b1=BETA1, b2=BETA2, eps=1e-8):
        self.t += 1
        self.mW = b1*self.mW + (1-b1)*self.gW
        self.vW = b2*self.vW + (1-b2)*self.gW**2
        mWh = self.mW/(1-b1**self.t); vWh = self.vW/(1-b2**self.t)
        self.W -= lr * mWh/(np.sqrt(vWh)+eps)
        self.mb = b1*self.mb + (1-b1)*self.gb
        self.vb = b2*self.vb + (1-b2)*self.gb**2
        mbh = self.mb/(1-b1**self.t); vbh = self.vb/(1-b2**self.t)
        self.b -= lr * mbh/(np.sqrt(vbh)+eps)

    def clip(self, c): self.W = np.clip(self.W,-c,c); self.b = np.clip(self.b,-c,c)


class MLP_G:
    """Generator: latent → surface (tanh output → [-1,1])"""
    def __init__(self):
        self.l1=Linear(LATENT_DIM,H1); self.l2=Linear(H1,H2); self.l3=Linear(H2,SURF_DIM)

    def forward(self, z):
        # Store pre-activation outputs (outputs of linear layers BEFORE activation)
        pre1=self.l1.forward(z);  self._pre1=pre1;  h1=leaky_relu(pre1)
        pre2=self.l2.forward(h1); self._pre2=pre2;  h2=leaky_relu(pre2)
        pre3=self.l3.forward(h2); self._pre3=pre3;  out=tanh(pre3)
        return out

    def backward(self, grad_out):
        # grad_out: [B, SURF_DIM]
        g = grad_out * tanh_grad(self._pre3)      # [B, SURF_DIM]
        g = self.l3.backward(g)                   # [B, H2]
        g = g * leaky_relu_grad(self._pre2)       # [B, H2]
        g = self.l2.backward(g)                   # [B, H1]
        g = g * leaky_relu_grad(self._pre1)       # [B, H1]
        self.l1.backward(g)

    def adam_step(self, lr):
        for l in [self.l1,self.l2,self.l3]: l.adam_step(lr)

    def clip(self, c):
        for l in [self.l1,self.l2,self.l3]: l.clip(c)

    def generate(self, n):
        z = np.random.randn(n, LATENT_DIM).astype(np.float32)
        return self.forward(z)

    def layers(self): return [self.l1, self.l2, self.l3]


class MLP_C:
    """Critic: surface → scalar Wasserstein score (no sigmoid)"""
    def __init__(self):
        self.l1=Linear(SURF_DIM,H1); self.l2=Linear(H1,H2); self.l3=Linear(H2,1)

    def forward(self, x):
        # Store pre-activation outputs (outputs of linear layers BEFORE activation)
        pre1=self.l1.forward(x);  self._pre1=pre1;  h1=leaky_relu(pre1)
        pre2=self.l2.forward(h1); self._pre2=pre2;  h2=leaky_relu(pre2)
        return self.l3.forward(h2)                # [B,1], no activation

    def backward(self, grad_out):
        # grad_out: [B, 1]  →  returns grad w.r.t. input [B, SURF_DIM] (needed by G)
        g = self.l3.backward(grad_out)            # [B, H2]
        g = g * leaky_relu_grad(self._pre2)       # [B, H2]
        g = self.l2.backward(g)                   # [B, H1]
        g = g * leaky_relu_grad(self._pre1)       # [B, H1]
        return self.l1.backward(g)                # [B, SURF_DIM]

    def adam_step(self, lr):
        for l in [self.l1,self.l2,self.l3]: l.adam_step(lr)

    def clip(self, c):
        for l in [self.l1,self.l2,self.l3]: l.clip(c)

    def score(self, x): return self.forward(x).mean()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Extract surfaces
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print(" VolGAN-BR  |  Stage 1: Surface Extraction")
print("="*65+"\n")

cache_train = OUT_DIR/"surfaces_train.npy"
cache_test  = OUT_DIR/"surfaces_test.npy"

if cache_train.exists() and cache_test.exists():
    print("Loading cached surfaces ...")
    X_train = np.load(cache_train)
    X_test  = np.load(cache_test)
    dates_train = pd.to_datetime(pd.read_csv(OUT_DIR/"dates_train.csv").iloc[:,0])
    dates_test  = pd.to_datetime(pd.read_csv(OUT_DIR/"dates_test.csv").iloc[:,0])
    print(f"  Train: {len(X_train)} surfaces | Test: {len(X_test)} surfaces")
else:
    print("Extracting TRAINING surfaces (2012–2023) ...")
    X_train, dates_train = extract_surfaces(TRAIN_YEARS, "TRAIN")
    print("\nExtracting TEST surfaces (2024–2026) ...")
    X_test, dates_test   = extract_surfaces(TEST_YEARS,  "TEST")
    np.save(cache_train, X_train); np.save(cache_test, X_test)
    pd.Series(dates_train).to_csv(OUT_DIR/"dates_train.csv", index=False)
    pd.Series(dates_test ).to_csv(OUT_DIR/"dates_test.csv",  index=False)

# Normalise to [-1, 1] using training statistics
iv_min = X_train.min(axis=0, keepdims=True)
iv_max = X_train.max(axis=0, keepdims=True)
iv_rng = np.where((iv_max-iv_min)<1e-6, 1.0, iv_max-iv_min)

def norm(X):   return 2*(X-iv_min)/iv_rng - 1
def denorm(X): return (X+1)/2*iv_rng + iv_min

X_train_n = norm(X_train).astype(np.float32)
X_test_n  = norm(X_test ).astype(np.float32)

if len(X_train_n) < 50:
    raise RuntimeError("Too few training surfaces. Check MIN_STRIKES or data range.")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: WGAN training
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print(f" VolGAN-BR  |  Stage 2: WGAN Training  (pure numpy)")
print(f" Surfaces: {len(X_train_n)}  |  Dim: {SURF_DIM}  |  Epochs: {N_EPOCHS}")
print("="*65+"\n")

G = MLP_G(); C = MLP_C()
g_losses, c_losses = [], []
n_batches_per_epoch = max(1, len(X_train_n)//BATCH_SIZE)
t0 = time.time()

for epoch in range(1, N_EPOCHS+1):
    # Shuffle
    idx = np.random.permutation(len(X_train_n))
    X_shuf = X_train_n[idx]
    g_ep = c_ep = 0.0

    for b in range(n_batches_per_epoch):
        real = X_shuf[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        if len(real)==0: continue
        B = len(real)

        # ── Critic updates ────────────────────────────────────────────
        # Use combined real+fake batch so we do a single backward per critic step.
        # Loss: L_C = mean(C(fake)) - mean(C(real))
        # grad w.r.t. C output: -1/B for real half, +1/B for fake half
        for _ in range(N_CRITIC):
            z    = np.random.randn(B, LATENT_DIM).astype(np.float32)
            fake = G.forward(z)
            combined = np.concatenate([real, fake], axis=0)   # [2B, SURF_DIM]
            scores   = C.forward(combined)                     # [2B, 1]
            score_r  = scores[:B].mean()
            score_f  = scores[B:].mean()
            w_loss   = score_f - score_r
            c_ep    += w_loss

            grad_c = np.concatenate(
                [-np.ones((B, 1), dtype=np.float32) / B,
                  np.ones((B, 1), dtype=np.float32) / B], axis=0)  # [2B, 1]
            C.backward(grad_c)
            C.adam_step(LR)
            C.clip(CLIP_VAL)

        # ── Generator update ──────────────────────────────────────────
        # Generator wants to maximise C(fake) → minimise -mean(C(fake))
        z    = np.random.randn(B, LATENT_DIM).astype(np.float32)
        fake = G.forward(z)
        C.forward(fake)
        g_loss = -C.score(fake)
        g_ep  += g_loss
        grad_c_in = -np.ones((B, 1), dtype=np.float32) / B   # ∂(-mean)/∂score
        grad_fake = C.backward(grad_c_in)                     # [B, SURF_DIM]
        G.backward(grad_fake)
        G.adam_step(LR)

    g_losses.append(g_ep / n_batches_per_epoch)
    c_losses.append(c_ep / n_batches_per_epoch / N_CRITIC)

    if epoch % 30 == 0 or epoch == 1:
        elapsed = time.time()-t0
        print(f"  Epoch {epoch:4d}/{N_EPOCHS}  "
              f"C={c_losses[-1]:+.4f}  G={g_losses[-1]:+.4f}  "
              f"[{elapsed:.0f}s elapsed]")

print(f"\n  ✅ Training done in {time.time()-t0:.0f}s")

# Save weights
np.save(OUT_DIR/"volgan_G_weights.npy",
        {"l1_W":G.l1.W,"l1_b":G.l1.b,
         "l2_W":G.l2.W,"l2_b":G.l2.b,
         "l3_W":G.l3.W,"l3_b":G.l3.b}, allow_pickle=True)
print("💾 Generator weights saved.")

# Training loss plot
fig_l, ax_l = plt.subplots(figsize=(10,4), facecolor="white")
ax_l.plot(g_losses, label="Generator loss",  color="#009c3b", lw=1.2)
ax_l.plot(c_losses, label="Critic loss",     color="#e63946", lw=1.2, alpha=0.7)
ax_l.axhline(0, color="black", lw=0.7, ls="--")
ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Wasserstein loss")
ax_l.set_title("WGAN Training Losses — VolGAN-BR (numpy)")
ax_l.legend(); ax_l.grid(True, alpha=0.3, ls="--"); fig_l.tight_layout()
fig_l.savefig(OUT_DIR/"volgan_training_losses.png", dpi=150); plt.close(fig_l)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Evaluation
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*65)
print(" VolGAN-BR  |  Stage 3: Evaluation")
print("="*65+"\n")

# Generate as many samples as test set
n_gen = max(len(X_test), 200)
gen_n = G.generate(n_gen)
gen_surfaces  = denorm(gen_n).astype(np.float32)
real_surfaces = X_test

gen_3d  = gen_surfaces.reshape(-1, N_K, N_T)
real_3d = real_surfaces.reshape(-1, N_K, N_T)
train_3d= X_train.reshape(-1, N_K, N_T)

# ── Clip obviously out-of-range generated values ─────────────────────────────
gen_3d = np.clip(gen_3d, 0.02, 2.0)

# ── Build evaluation figure ───────────────────────────────────────────────────
BLUE, RED, GOLD, GRAY = "#009c3b","#e63946","#FFDF00","#888888"

fig = plt.figure(figsize=(18, 14), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                         top=0.92, bottom=0.07, left=0.07, right=0.97)

tenor_labels = ["1M (T=1/12)", "2M (T=2/12)", "3M (T=3/12)"]
for ti in range(N_T):
    ax = fig.add_subplot(gs[0, ti])
    rm=real_3d[:,:,ti].mean(0); rp10=np.percentile(real_3d[:,:,ti],10,0); rp90=np.percentile(real_3d[:,:,ti],90,0)
    gm=gen_3d[:,:,ti].mean(0);  gp10=np.percentile(gen_3d[:,:,ti],10,0);  gp90=np.percentile(gen_3d[:,:,ti],90,0)
    ax.fill_between(K_GRID,rp10,rp90,alpha=0.2,color=BLUE)
    ax.fill_between(K_GRID,gp10,gp90,alpha=0.2,color=RED)
    ax.plot(K_GRID,rm,color=BLUE,lw=2,label="Real (test)")
    ax.plot(K_GRID,gm,color=RED,lw=2,ls="--",label="Generated")
    ax.set_title(f"IV Smile — {tenor_labels[ti]}",fontsize=9,fontweight="bold")
    ax.set_xlabel("k=ln(K/F)"); ax.set_ylabel("IV"); ax.legend(fontsize=7)
    ax.grid(True,alpha=0.3,ls="--"); ax.set_facecolor("#f5f5f5")

# ATM IV distribution
ax4=fig.add_subplot(gs[1,0])
atm_r=real_3d[:,N_K//2,0]; atm_g=gen_3d[:,N_K//2,0]
bins=np.linspace(min(atm_r.min(),atm_g.min()),max(atm_r.max(),atm_g.max()),30)
ax4.hist(atm_r,bins=bins,alpha=0.6,color=BLUE,label=f"Real μ={atm_r.mean():.2f}")
ax4.hist(atm_g,bins=bins,alpha=0.6,color=RED, label=f"Gen  μ={atm_g.mean():.2f}")
ax4.set_title("ATM IV Distribution (1M)",fontsize=9,fontweight="bold")
ax4.set_xlabel("ATM IV"); ax4.legend(fontsize=7); ax4.grid(True,alpha=0.3,ls="--"); ax4.set_facecolor("#f5f5f5")

# Skew distribution
ax5=fig.add_subplot(gs[1,1])
sk_r=real_3d[:,0,0]-real_3d[:,-1,0]; sk_g=gen_3d[:,0,0]-gen_3d[:,-1,0]
bins2=np.linspace(min(sk_r.min(),sk_g.min()),max(sk_r.max(),sk_g.max()),30)
ax5.hist(sk_r,bins=bins2,alpha=0.6,color=BLUE,label=f"Real μ={sk_r.mean():.3f}")
ax5.hist(sk_g,bins=bins2,alpha=0.6,color=RED, label=f"Gen  μ={sk_g.mean():.3f}")
ax5.set_title("Skew (put−call wing, 1M)",fontsize=9,fontweight="bold")
ax5.set_xlabel("Put−Call wing IV"); ax5.legend(fontsize=7); ax5.grid(True,alpha=0.3,ls="--"); ax5.set_facecolor("#f5f5f5")

# Term-structure slope
ax6=fig.add_subplot(gs[1,2])
ts_r=real_3d[:,N_K//2,2]-real_3d[:,N_K//2,0]; ts_g=gen_3d[:,N_K//2,2]-gen_3d[:,N_K//2,0]
bins3=np.linspace(min(ts_r.min(),ts_g.min()),max(ts_r.max(),ts_g.max()),30)
ax6.hist(ts_r,bins=bins3,alpha=0.6,color=BLUE,label=f"Real μ={ts_r.mean():.3f}")
ax6.hist(ts_g,bins=bins3,alpha=0.6,color=RED, label=f"Gen  μ={ts_g.mean():.3f}")
ax6.set_title("Term-Structure Slope (3M−1M)",fontsize=9,fontweight="bold")
ax6.set_xlabel("3M−1M ATM IV"); ax6.legend(fontsize=7); ax6.grid(True,alpha=0.3,ls="--"); ax6.set_facecolor("#f5f5f5")

# PCA projection
from numpy.linalg import svd
X_all = np.vstack([X_train, real_surfaces]) - np.vstack([X_train, real_surfaces]).mean(0)
_,_,Vt = svd(X_all, full_matrices=False)
PC1,PC2 = Vt[0],Vt[1]
ax7=fig.add_subplot(gs[2,0])
ax7.scatter(X_train@PC1,X_train@PC2,s=4,alpha=0.3,color=GRAY,label="Train real")
ax7.scatter(real_surfaces@PC1,real_surfaces@PC2,s=10,alpha=0.7,color=BLUE,label="Test real")
ax7.scatter(gen_surfaces@PC1, gen_surfaces@PC2, s=10,alpha=0.7,color=RED, label="Generated",marker="x")
ax7.set_title("PCA Projection (PC1 vs PC2)",fontsize=9,fontweight="bold")
ax7.set_xlabel("PC1"); ax7.set_ylabel("PC2"); ax7.legend(fontsize=7)
ax7.grid(True,alpha=0.3,ls="--"); ax7.set_facecolor("#f5f5f5")

# 3 generated surface samples
ax8=fig.add_subplot(gs[2,1])
for i,c in enumerate([BLUE,RED,GOLD]):
    ax8.plot(K_GRID, gen_3d[i,:,0],color=c,lw=1.8,label=f"Gen {i+1}")
    ax8.plot(K_GRID, real_3d[i,:,0],color=c,lw=1.0,ls=":",alpha=0.6)
ax8.set_title("Generated vs Real smiles (1M, 3 samples)\nSolid=generated, dotted=real",fontsize=9,fontweight="bold")
ax8.set_xlabel("k=ln(K/F)"); ax8.set_ylabel("IV"); ax8.legend(fontsize=7)
ax8.grid(True,alpha=0.3,ls="--"); ax8.set_facecolor("#f5f5f5")

# Stats table
ax9=fig.add_subplot(gs[2,2]); ax9.axis("off")
rows=[["Metric","Real (test)","Generated"],
      ["ATM 1M mean",f"{atm_r.mean():.3f}",f"{atm_g.mean():.3f}"],
      ["ATM 1M std", f"{atm_r.std():.3f}", f"{atm_g.std():.3f}"],
      ["Skew mean",  f"{sk_r.mean():.3f}", f"{sk_g.mean():.3f}"],
      ["Skew std",   f"{sk_r.std():.3f}",  f"{sk_g.std():.3f}"],
      ["TS slope mean",f"{ts_r.mean():.3f}",f"{ts_g.mean():.3f}"],
      ["TS slope std", f"{ts_r.std():.3f}", f"{ts_g.std():.3f}"],
      ["N surfaces", str(len(real_3d)), str(len(gen_3d))]]
tbl=ax9.table(cellText=rows[1:],colLabels=rows[0],loc="center",cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.15,1.7)
ax9.set_title("Summary Statistics",fontsize=9,fontweight="bold",pad=15)

fig.suptitle(
    "VolGAN-BR Evaluation  |  Real (2024–2026) vs Generated IV Surfaces\n"
    "WGAN trained on Brazilian IBOV Index Options (2012–2023)",
    fontsize=12,fontweight="bold",y=0.97)
eval_path = OUT_DIR/"volgan_evaluation.png"
fig.savefig(str(eval_path),dpi=150,bbox_inches="tight"); plt.close(fig)
print(f"💾 Saved: {eval_path}")

print("\n"+"="*65)
print(" EVALUATION SUMMARY")
print("="*65)
print(f"  Train surfaces : {len(X_train)}  ({TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]})")
print(f"  Test  surfaces : {len(X_test)}  ({TEST_YEARS[0]}–{TEST_YEARS[-1]})")
print(f"  Generated      : {n_gen}")
print(f"\n  {'Metric':<20} {'Real':>10} {'Generated':>10}")
print(f"  {'-'*42}")
print(f"  {'ATM 1M mean':<20} {atm_r.mean():>10.4f} {atm_g.mean():>10.4f}")
print(f"  {'ATM 1M std':<20} {atm_r.std():>10.4f} {atm_g.std():>10.4f}")
print(f"  {'Skew mean':<20} {sk_r.mean():>10.4f} {sk_g.mean():>10.4f}")
print(f"  {'Skew std':<20} {sk_r.std():>10.4f} {sk_g.std():>10.4f}")
print(f"  {'TS slope mean':<20} {ts_r.mean():>10.4f} {ts_g.mean():>10.4f}")
print(f"  {'TS slope std':<20} {ts_r.std():>10.4f} {ts_g.std():>10.4f}")
print(f"\n✅ VolGAN-BR complete.")
print(f"   Outputs: {OUT_DIR}")
