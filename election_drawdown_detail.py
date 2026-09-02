"""
election_drawdown_detail.py
===========================
Detalhamento da tabela de base histórica: para cada ciclo eleitoral,
mostra as datas e os preços exatos que produzem a queda de topo a fundo
da janela agosto-dezembro, além dos preços nas datas dos dois turnos.

O "topo" é a máxima corrente ANTES do fundo (definição de drawdown), não
necessariamente a máxima da janela inteira.

Preços: fechamento Yahoo (ajustado por grupamento/desdobramento, NÃO por
proventos). É a série correta para medir queda de preço — que é o que a
opção paga —, mas o retorno total do acionista foi melhor que o mostrado
aqui pelos dividendos recebidos no período.

Saída: results/election_drawdown_detail.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
RES = BASE / "results"
RES.mkdir(exist_ok=True)

ELECTIONS = [
    ("2002-10-06", "2002-10-27"),
    ("2006-10-01", "2006-10-29"),
    ("2010-10-03", "2010-10-31"),
    ("2014-10-05", "2014-10-26"),
    ("2018-10-07", "2018-10-28"),
    ("2022-10-02", "2022-10-30"),
]


def px_on(s, d):
    """Fechamento na data d, ou no pregão anterior mais próximo."""
    prior = s.index[s.index <= pd.Timestamp(d)]
    return (prior[-1], float(s.loc[prior[-1]])) if len(prior) else (None, np.nan)


def main():
    s = (pd.read_csv(BASE / "data" / "spot_yahoo_PETR4.csv", parse_dates=["date"])
         .set_index("date")["close"].sort_index().dropna())

    L = []
    P = L.append
    P("PETR4 — detalhamento das quedas na janela eleitoral (ago-dez)")
    P("=" * 100)
    P(f"Série: fechamento diário, {s.index.min().date()} a {s.index.max().date()}")
    P("Ajustada por grupamento/desdobramento; NÃO ajustada por proventos.")
    P("")
    P("Topo = máxima corrente antes do fundo (definição de drawdown).")
    P("")
    P(f"  {'ciclo':<7}{'TOPO':>22}{'FUNDO':>22}{'queda':>9}"
      f"{'dias':>6}{'01/ago':>10}{'31/dez':>10}")
    P(f"  {'':<7}{'data        R$':>22}{'data        R$':>22}")
    P("-" * 100)

    rows = []
    for r1, r2 in ELECTIONS:
        y = pd.Timestamp(r1).year
        win = s.loc[f"{y}-08-01":f"{y}-12-31"]
        if win.empty:
            continue
        dd_series = win / win.cummax() - 1
        d_trough = dd_series.idxmin()
        dd = float(dd_series.min())
        # topo = data da máxima corrente até o fundo
        d_peak = win.loc[:d_trough].idxmax()
        p_peak, p_trough = float(win.loc[d_peak]), float(win.loc[d_trough])

        d_ini, p_ini = win.index[0], float(win.iloc[0])
        d_fim, p_fim = win.index[-1], float(win.iloc[-1])
        dias = (d_trough - d_peak).days

        P(f"  {y:<7}{d_peak.strftime('%d/%m/%Y'):>12}{p_peak:>10.2f}"
          f"{d_trough.strftime('%d/%m/%Y'):>12}{p_trough:>10.2f}"
          f"{dd:>9.1%}{dias:>6}{p_ini:>10.2f}{p_fim:>10.2f}")

        d1, p1 = px_on(s, r1)
        d2, p2 = px_on(s, r2)
        rows.append({"y": y, "dd": dd, "peak": p_peak, "trough": p_trough,
                     "d_peak": d_peak, "d_trough": d_trough,
                     "ini": p_ini, "fim": p_fim,
                     "p_r1": p1, "p_r2": p2, "d_r1": d1, "d_r2": d2,
                     "ret_win": p_fim / p_ini - 1})

    df = pd.DataFrame(rows)
    P("-" * 100)
    P(f"  {'mediana':<7}{'':>22}{'':>22}{df['dd'].median():>9.1%}")
    P("")
    P("")

    P("Preços nas datas dos turnos e desempenho da janela")
    P("=" * 100)
    P(f"  {'ciclo':<7}{'1º turno':>14}{'R$':>9}{'2º turno':>14}{'R$':>9}"
      f"{'ago→dez':>10}{'topo→fundo':>12}")
    P("-" * 100)
    for _, x in df.iterrows():
        P(f"  {int(x['y']):<7}{x['d_r1'].strftime('%d/%m/%Y'):>14}{x['p_r1']:>9.2f}"
          f"{x['d_r2'].strftime('%d/%m/%Y'):>14}{x['p_r2']:>9.2f}"
          f"{x['ret_win']:>10.1%}{x['dd']:>12.1%}")
    P("-" * 100)
    P(f"  {'mediana':<7}{'':>14}{'':>9}{'':>14}{'':>9}"
      f"{df['ret_win'].median():>10.1%}{df['dd'].median():>12.1%}")
    P("")
    P("Observações")
    P("-" * 100)
    P("* 'ago→dez' é o retorno ponta a ponta da janela; 'topo→fundo' é a pior")
    P("  sequência dentro dela. Os dois números divergem bastante porque em")
    P("  vários ciclos a ação despencou e depois recuperou parte antes do fim")
    P(f"  do ano — em {int(df.loc[df['ret_win'].idxmax(),'y'])} a janela fechou "
      f"{df['ret_win'].max():+.0%} apesar de um fundo de "
      f"{df.loc[df['ret_win'].idxmax(),'dd']:.0%}.")
    P("* O drawdown é o número relevante para dimensionar hedge com opção:")
    P("  a proteção precisa aguentar o caminho, não só o ponto final.")
    P("* Sem ajuste por proventos. A PETR4 pagou dividendos relevantes em")
    P("  vários desses períodos, então o prejuízo total do acionista foi")
    P("  menor que a queda de preço mostrada.")

    rep = "\n".join(L)
    (RES / "election_drawdown_detail.txt").write_text(rep)
    print(rep)


if __name__ == "__main__":
    main()
