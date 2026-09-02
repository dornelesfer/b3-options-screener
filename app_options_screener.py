"""
B3 Options Screener — cheap/rich ranking on the metrics from the VolGAN/VRP work.

Run:  streamlit run app_options_screener.py
Data: python3 screener_metrics.py  (rebuild after updating the rb3 DB)

Metrics shown per option
  iv            Black-Scholes implied vol from EOD mid (bid/ask when quoted, else close)
  iv_minus_rv   IV minus the underlying's trailing 21d realized vol — the VRP entry
                signal from the short-vol strategy (positive = option rich vs realized).
                LEGACY: only valid near 21 trading days to expiry; see below.
  iv_minus_rv_h IV minus realized vol measured over a window matching the option's
                OWN time to expiry (Burghardt & Lane 1990). Corrects iv_minus_rv,
                which judges a 725-day option against a 21-day window; the
                cheap/rich sign flips on ~15% of the chain.
  cone_z        the same comparison in standard deviations of the matched-horizon
                volatility cone — the ranking metric. cone_pct is the percentile
                form, but it pins at 100 for ~27% of the chain and stops ranking.
                Both compare against the underlying's ATM realized vol, so the
                smile inflates them out of the money: read near-ATM, or alongside
                smile_resid.
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

# Streamlit 1.50+ replaced use_container_width=True with width="stretch" and
# scheduled the old name for removal; Cloud installs the latest release while
# the dev Mac runs 1.32. Pick the spelling this runtime understands.
_ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ST_VER >= (1, 50) else {"use_container_width": True}


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
max_mny = st.sidebar.slider(
    "Max |log-moneyness| ln(K/F)", 0.05, 1.00, 0.30, step=0.05,
    help="Deep ITM options have almost no extrinsic value, so their implied "
         "vol is quote noise (IVs of 100-250% on a few centavos) — and that "
         "same noise drags delta back toward 0.5, so delta can't screen them "
         "out. Deep OTM is mostly smile. The cone metrics compare to ATM "
         "realised vol and are only meaningful near the money.")
min_ctr = st.sidebar.number_input("Min contracts traded", 0, value=50, step=10)
min_prem = st.sidebar.number_input("Min premium (R$ / pts)", 0.0, value=0.10,
                                   step=0.05,
                                   help="Filters out illiquid deep-OTM dust "
                                        "whose metrics are quote noise")
METRIC_LABELS = {"cone_z": "Cone z-score (horizon-matched VRP)",
                 "smile_resid": "Smile residual (vs own expiry)",
                 "iv_minus_rv_h": "IV − realized vol, horizon-matched",
                 "iv_minus_rv": "IV − RV21 (legacy, unmatched horizon)",
                 "volgan_pctile": "cVolGAN percentile (IBOV, EXPERIMENTAL)",
                 "iv": "Implied vol",
                 "cp_gap": "Call−put IV gap",
                 "parity_resid": "Parity residual"}
# only offer metrics the committed parquet actually has: a code push lands on
# Streamlit Cloud before the nightly job rebuilds the data, and a missing
# column must not take the app down in between
metric = st.sidebar.selectbox(
    "Ranking metric", [m for m in METRIC_LABELS if m in chain.columns],
    format_func=METRIC_LABELS.get)
topn = st.sidebar.slider("Rows per table", 5, 40, 15)

d = d[d["expiry"].isin(sel_exp)
      & d["dte"].between(*dte_rng)
      & (d["contracts"] >= min_ctr)
      & (d["mid"] >= min_prem)
      & (d["moneyness"].abs() <= max_mny)]
if otype != "Both":
    d = d[d["type"] == ("C" if otype == "Calls" else "P")]

st.title("B3 Options Screener")
st.caption(f"EOD data as of "
           f"{', '.join(f'{u}: {pd.Timestamp(x).date()}' for u, x in chain.groupby('underlying')['date'].max().items())}"
           " — refreshed nightly from B3 COTAHIST after the close (never intraday).")

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
        rv_m = r0.get("rv_matched_und", r0.get("rv_matched"))
        if pd.notna(iv_atm) and pd.notna(rv21):
            txt = f"ATM IV **{iv_atm:.1f}%** · RV21 **{rv21:.1f}%**"
            if pd.notna(rv_m) and pd.notna(r0.get("spread_h")):
                spr, pct = r0.get("spread_h"), r0.get("spread_h_pctile")
                txt += f" · matched RV **{rv_m:.1f}%**"
            else:                       # parquet predates the cone metrics
                spr, pct = r0.get("spread"), r0.get("spread_pctile")
            if pd.notna(spr):
                txt += f" · spread **{spr:+.1f}pp**"
                if pd.notna(pct):
                    txt += f" (pctile {pct:.0f})"
            st.caption(txt)
        else:
            st.caption("insufficient ATM/RV history")

tab_rank, tab_smile, tab_hist, tab_flags = st.tabs(
    ["Rankings", "Smile", "IV vs RV history", "Anomalies"])

SHOW_COLS = ["underlying", "symbol", "type", "K", "expiry", "dte", "mid", "iv",
             "delta", "smile_resid", "rv_horizon", "rv_matched", "iv_minus_rv_h",
             "cone_z", "iv_minus_rv", "volgan_pctile", "cp_gap",
             "parity_resid", "contracts"]
FMT = {"K": "{:,.2f}", "mid": "{:,.2f}", "iv": "{:.1f}", "delta": "{:+.2f}",
       "smile_resid": "{:+.2f}", "iv_minus_rv": "{:+.1f}",
       "rv_horizon": "{:.0f}", "rv_matched": "{:.1f}", "iv_minus_rv_h": "{:+.1f}",
       "cone_z": "{:+.2f}",
       "volgan_pctile": "{:.0f}", "cp_gap": "{:+.2f}",
       "parity_resid": "{:+.2f}", "contracts": "{:,.0f}"}


def show_table(df):
    df = df[[c for c in SHOW_COLS if c in df.columns]].copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%d/%m/%y")
    st.dataframe(df.style.format(FMT, na_rep="–"), hide_index=True, **STRETCH)


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
            st.plotly_chart(fig, **STRETCH)

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
        if "rv_matched" in h.columns and h["rv_matched"].notna().any():
            fig.add_trace(go.Scatter(x=h["date"], y=h["rv_matched"],
                                     name="Realized, ATM-horizon matched",
                                     line=dict(width=1.5, color=AQUA, dash="dot")))
        fig.update_layout(title=f"{und} — implied vs realized (last 3y)",
                          height=320, margin=dict(l=40, r=20, t=48, b=40),
                          legend=dict(orientation="h"), **PLOT_LAYOUT)
        fig.update_yaxes(title="Vol (%)")
        st.plotly_chart(fig, **STRETCH)

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
