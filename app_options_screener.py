"""
B3 Options Screener — cheap/rich ranking on the metrics from the VolGAN/VRP work.

Run:  streamlit run app_options_screener.py
Data: python3 screener_metrics.py  (rebuild after updating the rb3 DB)

Metrics shown per option
  iv            Black-Scholes implied vol from EOD mid (bid/ask when quoted, else close)
  iv_minus_rv   IV minus the underlying's trailing 21d realized vol — the VRP entry
                signal from the short-vol strategy (positive = option rich vs realized)
  smile_resid   IV minus a quadratic smile fit of its own expiry — relative value
                against neighbouring strikes (positive = rich vs its peers)
  cp_gap        call IV minus put IV at the same strike — covered-call-flow gauge
                (negative = calls cheap)
  parity_resid  (C-P) - (S - K e^{-rT}) — conversion/reversal residual; large |values|
                flag borrow squeezes or corporate events (see BRAV3 case)
  below_intrinsic  price below immediate-exercise value (stale quote or event)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

BASE = Path(__file__).parent
SCR = BASE / "data" / "screener"

# dataviz reference palette — dark-surface steps (theme.base is pinned to dark)
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
INK, INK2 = "#ffffff", "#c3c2b7"
GRID = "rgba(195,194,183,0.18)"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK2),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)

st.set_page_config(page_title="B3 Options Screener", layout="wide")


@st.cache_data(show_spinner=False)
def load(mtime):
    chain = pq.read_table(str(SCR / "chain_latest.parquet")).to_pandas()
    hist = pq.read_table(str(SCR / "history_daily.parquet")).to_pandas()
    chain = chain[chain["underlying"] != "VALE5"]      # delisted 2017
    return chain, hist


try:
    chain, hist = load((SCR / "chain_latest.parquet").stat().st_mtime)
except Exception:
    st.error("Metrics not built yet — run `python3 screener_metrics.py` first.")
    st.stop()

# ── sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.title("Filters")
unds = sorted(chain["underlying"].unique())
sel_und = st.sidebar.multiselect("Underlying", unds, default=unds)

d = chain[chain["underlying"].isin(sel_und)].copy()

expiries = sorted(d["expiry"].dropna().unique())
sel_exp = st.sidebar.multiselect(
    "Expiry (empty = all)", expiries, default=[],
    format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))
if not sel_exp:
    sel_exp = expiries
dte_max = int(max(d["dte"].max(), 1))
dte_rng = st.sidebar.slider("Days to expiry", 1, dte_max, (5, min(dte_max, 120)))
otype = st.sidebar.radio("Type", ["Both", "Calls", "Puts"], horizontal=True)
min_ctr = st.sidebar.number_input("Min contracts traded", 0, value=50, step=10)
min_prem = st.sidebar.number_input("Min premium (R$ / pts)", 0.0, value=0.10,
                                   step=0.05,
                                   help="Filters out illiquid deep-OTM dust "
                                        "whose metrics are quote noise")
metric = st.sidebar.selectbox(
    "Ranking metric",
    ["smile_resid", "iv_minus_rv", "volgan_pctile", "iv", "cp_gap",
     "parity_resid"],
    format_func={"smile_resid": "Smile residual (vs own expiry)",
                 "iv_minus_rv": "IV − realized vol (VRP)",
                 "volgan_pctile": "cVolGAN percentile (IBOV, EXPERIMENTAL)",
                 "iv": "Implied vol",
                 "cp_gap": "Call−put IV gap",
                 "parity_resid": "Parity residual"}.get)
topn = st.sidebar.slider("Rows per table", 5, 40, 15)

d = d[d["expiry"].isin(sel_exp)
      & d["dte"].between(*dte_rng)
      & (d["contracts"] >= min_ctr)
      & (d["mid"] >= min_prem)]
if otype != "Both":
    d = d[d["type"] == ("C" if otype == "Calls" else "P")]

st.title("B3 Options Screener")
st.caption(f"EOD data as of "
           f"{', '.join(f'{u}: {pd.Timestamp(x).date()}' for u, x in chain.groupby('underlying')['date'].max().items())}"
           " — rebuild with `screener_metrics.py` after updating the rb3 DB.")

# ── KPI row per underlying ───────────────────────────────────────────────────
cols = st.columns(max(len(sel_und), 1))
for c, und in zip(cols, sel_und):
    g = chain[chain["underlying"] == und]
    if g.empty:
        continue
    r0 = g.iloc[0]
    with c:
        st.metric(und, f"{r0['S']:,.2f}")
        iv_atm = r0.get("iv_atm")
        rv21 = r0.get("rv21")
        spr = r0.get("spread")
        pct = r0.get("spread_pctile")
        pct_txt = f" (pctile {pct:.0f})" if pd.notna(pct) else ""
        st.caption(
            f"ATM IV **{iv_atm:.1f}%** · RV21 **{rv21:.1f}%** · "
            f"spread **{spr:+.1f}pp**{pct_txt}"
            if pd.notna(iv_atm) and pd.notna(rv21)
            else "insufficient ATM/RV history")

tab_rank, tab_smile, tab_hist, tab_flags = st.tabs(
    ["Rankings", "Smile", "IV vs RV history", "Anomalies"])

SHOW_COLS = ["underlying", "symbol", "type", "K", "expiry", "dte", "mid", "iv",
             "delta", "smile_resid", "iv_minus_rv", "volgan_pctile", "cp_gap",
             "parity_resid", "contracts"]
FMT = {"K": "{:,.2f}", "mid": "{:,.2f}", "iv": "{:.1f}", "delta": "{:+.2f}",
       "smile_resid": "{:+.2f}", "iv_minus_rv": "{:+.1f}",
       "volgan_pctile": "{:.0f}", "cp_gap": "{:+.2f}",
       "parity_resid": "{:+.2f}", "contracts": "{:,.0f}"}


def show_table(df):
    df = df[SHOW_COLS].copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%d/%m/%y")
    st.dataframe(df.style.format(FMT, na_rep="–"),
                 use_container_width=True, hide_index=True)


with tab_rank:
    if metric == "volgan_pctile":
        st.warning("cVolGAN score is EXPERIMENTAL: the current generator's "
                   "conditional dispersion is too narrow (scores cluster at "
                   "100), so rankings are not yet reliable. See HANDOFF.md.")
    dd = d.dropna(subset=[metric])
    lo, hi = st.columns(2)
    with lo:
        st.subheader("Cheapest")
        st.caption("lowest value of the chosen metric")
        show_table(dd.nsmallest(topn, metric))
    with hi:
        st.subheader("Richest")
        st.caption("highest value of the chosen metric")
        show_table(dd.nlargest(topn, metric))

with tab_smile:
    for und in sel_und:
        g = d[(d["underlying"] == und)].dropna(subset=["iv"])
        if g.empty:
            continue
        st.subheader(und)
        nearest = sorted(g["expiry"].unique())[:4]
        if len(g["expiry"].unique()) > 4:
            st.caption("showing the 4 nearest expiries — narrow with the "
                       "Expiry filter to see others")
        for exp, ge in sorted(g[g["expiry"].isin(nearest)].groupby("expiry"),
                              key=lambda t: t[0]):
            if len(ge) < 3:
                continue
            fig = go.Figure()
            for t, color, name in (("C", BLUE, "Calls"), ("P", ORANGE, "Puts")):
                gt = ge[ge["type"] == t].sort_values("K")
                if gt.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=gt["K"], y=gt["iv"], mode="markers+lines", name=name,
                    marker=dict(size=8, color=color),
                    line=dict(width=2, color=color),
                    customdata=np.stack([gt["symbol"], gt["dte"],
                                         gt["smile_resid"].fillna(0),
                                         gt["contracts"]], axis=-1),
                    hovertemplate=("%{customdata[0]} · K=%{x:,.2f} · IV %{y:.1f}%"
                                   "<br>smile resid %{customdata[2]:+.2f}pp · "
                                   "%{customdata[3]:,.0f} contracts"
                                   "<extra></extra>")))
            spot = float(ge["S"].iloc[0])
            fig.add_vline(x=spot, line_width=1, line_dash="dot", line_color=INK2,
                          annotation_text="spot", annotation_font_color=INK2)
            fig.update_layout(
                title=f"{pd.Timestamp(exp).strftime('%d/%m/%Y')} ({int(ge['dte'].iloc[0])}d)",
                height=340, margin=dict(l=40, r=20, t=48, b=40),
                legend=dict(orientation="h"), **PLOT_LAYOUT)
            fig.update_xaxes(title="Strike")
            fig.update_yaxes(title="Implied vol (%)")
            st.plotly_chart(fig, use_container_width=True)

with tab_hist:
    for und in sel_und:
        h = hist[hist["underlying"] == und].sort_values("date")
        if len(h) < 10:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["date"], y=h["iv_atm"], name="ATM IV",
                                 line=dict(width=2, color=BLUE)))
        fig.add_trace(go.Scatter(x=h["date"], y=h["rv21"], name="Realized 21d",
                                 line=dict(width=2, color=ORANGE)))
        fig.update_layout(title=f"{und} — implied vs realized (last 3y)",
                          height=320, margin=dict(l=40, r=20, t=48, b=40),
                          legend=dict(orientation="h"), **PLOT_LAYOUT)
        fig.update_yaxes(title="Vol (%)")
        st.plotly_chart(fig, use_container_width=True)

with tab_flags:
    st.subheader("Below intrinsic (check for corporate events before trading)")
    bi = d[d["below_intrinsic"]].copy()
    bi["gap"] = bi["intrinsic"] - bi["mid"]
    if bi.empty:
        st.caption("none under current filters")
    else:
        show_table(bi.sort_values("gap", ascending=False).head(topn))
    st.subheader("Largest |parity residuals| (borrow squeeze / event flag)")
    pr = d.dropna(subset=["parity_resid"]).copy()
    pr = pr[pr["type"] == "C"]      # one row per strike pair
    pr["abs_resid"] = pr["parity_resid"].abs()
    show_table(pr.nlargest(topn, "abs_resid"))
    st.caption("Reminder from the BRAV3 case: uniform one-sided residuals across "
               "strikes usually mean a tender offer, borrow squeeze, or pending "
               "corporate event — not free money.")
