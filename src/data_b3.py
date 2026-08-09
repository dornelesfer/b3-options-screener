# src/data_b3.py
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile, re

def read_cotahist_options(path):
    """
    Lê um arquivo COTAHIST (txt ou zip) e retorna apenas linhas de opções.
    **Ajuste as posições (fixed-width) conforme o layout oficial do seu arquivo**.
    Retorna colunas: ['date','option_symbol','type','strike','maturity','price']
    """
    p = Path(path)
    if p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p, 'r') as z:
            txt = [n for n in z.namelist() if n.lower().endswith('.txt')][0]
            lines = z.read(txt).decode('latin1').splitlines()
    else:
        lines = Path(path).read_text(encoding='latin1').splitlines()

    rows = []
    for line in lines:
        if len(line)<120: 
            continue
        # Exemplos de offsets (PLACEHOLDER). Substitua pelas colunas corretas do seu COTAHIST:
        date = line[2:10]                     # AAAAMMDD
        symbol = line[12:24].strip()          # Código do papel/derivado
        price_raw = line[109:121].strip()     # Preço *100
        price = float(price_raw)/100 if price_raw.isdigit() else np.nan
        # Heurística: série A-L -> call, M-X -> put (ajuste para seu universo)
        opt_type = 'C' if re.search(r'[A-L]\d*$', symbol) else ('P' if re.search(r'[M-X]\d*$', symbol) else None)
        # Strike/maturity exigem BVBG-086 (tabela de instrumentos). Deixe NaN e anexe depois.
        strike = np.nan
        maturity = None
        rows.append([date, symbol, opt_type, strike, maturity, price])

    df = pd.DataFrame(rows, columns=['date','option_symbol','type','strike','maturity','price'])
    df = df[df['type'].isin(['C','P'])].copy()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    return df

def attach_instrument_table(df_opts, bvbg086_df):
    """
    Enriquecer com metadados (underlying, strike, maturity) via tabela BVBG-086.
    Ajuste os nomes das colunas conforme a sua versão do arquivo.
    """
    key_left = 'option_symbol'; key_right = 'symbol'
    out = df_opts.merge(bvbg086_df, left_on=key_left, right_on=key_right, how='left')
    ren = {'underlying':'underlying', 'strike_price':'strike', 'maturity_date':'maturity'}
    for k,v in ren.items():
        if k in out.columns: out.rename(columns={k:v}, inplace=True)
    return out
