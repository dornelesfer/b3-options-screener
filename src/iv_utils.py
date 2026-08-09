# src/iv_utils.py
import numpy as np
from math import sqrt
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, r, q, sigma, opt_type='C'):
    if T<=0 or sigma<=0:
        return max(0.0, (S*np.exp(-q*T) - K*np.exp(-r*T)) if opt_type=='C' else (K*np.exp(-r*T) - S*np.exp(-q*T)))
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if opt_type=='C':
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bs_implied_vol(price, S, K, T, r, q, opt_type='C'):
    if price<=0: return np.nan
    def f(sig): return bs_price(S,K,T,r,q,sig,opt_type) - price
    try:
        return brentq(f, 1e-4, 5.0, maxiter=200)
    except Exception:
        return np.nan

def estimate_forward_discount(cp_df):
    """
    Estima forward F e desconto D=exp(-rT) por vencimento via put–call parity:
      C - P = D*(F - K)
    cp_df: DataFrame com colunas ['K','C','P','T'] (mesmo vencimento).
    Retorna (F, D). Se falhar, retorna (np.nan, np.nan).
    """
    try:
        sub = cp_df.dropna().copy()
        y = (sub['C'] - sub['P']).values
        X = np.vstack([np.ones_like(sub['K']), -sub['K'].values]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        D = max(1e-6, min(1.0, -beta[1]))
        F = beta[0]/D if D>0 else np.nan
        return F, D
    except Exception:
        return np.nan, np.nan

def vix_like_from_strip(strip_df):
    """
    VIX/VXBR-like a partir de OTM strip (puts abaixo de K0 e calls acima).
    Requer colunas ['K','Q','T','r'].
    Retorna pontos (% a.a.).
    """
    df = strip_df.sort_values('K').copy()
    K = df['K'].values
    dK = np.zeros_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2.0
    dK[0] = K[1]-K[0]; dK[-1] = K[-1]-K[-2]
    Q = df['Q'].values
    T = df['T'].iloc[0]; r = df['r'].iloc[0]
    var = (2.0/T)*np.sum((dK/(K**2))*np.exp(r*T)*Q) - (1.0/T)*((df.get('F_over_K0', df['K']/df['K']).iloc[0]-1.0)**2)
    var = max(var, 0.0)
    return 100.0*np.sqrt(var)
