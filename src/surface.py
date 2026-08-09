# src/surface.py
import pandas as pd
import numpy as np

def build_surface_day(df_opts_day, S_or_F, r, q=0.0, iv_func=None):
    rows = []
    ref_date = df_opts_day['date'].iloc[0]
    for mat, blk in df_opts_day.groupby('maturity'):
        T = max((mat - ref_date).days/365.0, 1/365)
        F = S_or_F.get(mat, np.nan) if isinstance(S_or_F, dict) else S_or_F
        rr = r.get(mat, r) if isinstance(r, dict) else r
        if np.isnan(F): 
            continue
        for _, row in blk.iterrows():
            K = row['strike']
            if pd.isna(K) or K<=0: 
                continue
            k = np.log(K/F)
            sigma = iv_func(row['price'], F, K, T, rr, 0.0, row['type'])
            if not np.isnan(sigma) and sigma<5.0:
                rows.append([k, T, sigma])
    return pd.DataFrame(rows, columns=['k','T','iv'])
