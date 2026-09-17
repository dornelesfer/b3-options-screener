"""Historical IBOV realized-volatility cones and quote-side IV indications."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq
from scipy.special import ndtr

HORIZONS = [5, 10, 21, 42, 63, 126, 252]


def implied_vol(price, spot, strike, years, rate, call):
    if not np.isfinite([price, spot, strike, years, rate]).all() or min(price, spot, strike, years) <= 0:
        return np.nan
    def residual(vol):
        d1 = (np.log(spot / strike) + (rate + vol * vol / 2) * years) / (vol * np.sqrt(years))
        d2 = d1 - vol * np.sqrt(years)
        value = (spot * ndtr(d1) - strike * np.exp(-rate * years) * ndtr(d2) if call
                 else strike * np.exp(-rate * years) * ndtr(-d2) - spot * ndtr(-d1))
        return value - price
    try:
        return brentq(residual, .01, 3) * 100
    except ValueError:
        return np.nan


def cone(spot, day, lookback):
    """Each reference distribution excludes the selected date and future data."""
    spot = spot.loc[:pd.Timestamp(day)]
    logret = np.log(spot).diff()
    rows = []
    for h in HORIZONS:
        vol = logret.rolling(h).std(ddof=1) * np.sqrt(252) * 100
        past = vol.iloc[:-1]
        if lookback:
            past = past.tail(lookback)
        past = past.dropna()
        current = vol.iloc[-1]
        if past.empty or not np.isfinite(current):
            continue
        q = past.quantile([.1, .25, .5, .75, .9]).to_numpy()
        rows.append(dict(h=h, current=current, p10=q[0], p25=q[1], median=q[2],
                         p75=q[3], p90=q[4], rank=100 * past.le(current).mean(), samples=len(past)))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_inputs(base, signature):
    """signature includes all source mtimes so nightly updates invalidate cache."""
    base = Path(base)
    s = pd.read_csv(base / 'ibov_daily.csv', parse_dates=['date'])
    s = s[s.date.dt.dayofweek < 5].drop_duplicates('date', keep='last').sort_values('date')
    spot = s.set_index('date').ibov_close
    spot = spot[spot.gt(0) & np.isfinite(spot)]
    rates = pd.read_csv(base / 'rates_cdi.csv', parse_dates=['date']).dropna(subset=['r_cc'])
    rates = rates.drop_duplicates('date', keep='last').set_index('date').r_cc.sort_index()
    rate = rates.reindex(spot.index, method='ffill')
    files = sorted(base.glob('ibov_options_all*.parquet'))
    opts = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    opts = opts.drop_duplicates(['refdate', 'symbol'], keep='last')
    opts['spot'] = opts.refdate.map(spot)
    opts['rate'] = opts.refdate.map(rate)
    opts['days'] = (opts.maturity_date - opts.refdate).dt.days
    opts = opts[opts.days.between(8, 365) & opts.strike_price.gt(0) & opts.spot.gt(0)
                & opts.best_bid.gt(0) & opts.best_ask.ge(opts.best_bid) & opts.bdi_code.isin([74, 75])].copy()
    opts['distance'] = abs(np.log(opts.strike_price / (opts.spot * np.exp(opts.rate * opts.days / 365))))
    opts = opts[opts.distance.le(np.log(1.05))]
    # One nearest quoted contract per expiry and type; do not pick by attractive IV.
    opts = opts.sort_values(['refdate', 'maturity_date', 'bdi_code', 'distance', 'volume'],
                            ascending=[True, True, True, True, False])
    opts = opts.drop_duplicates(['refdate', 'maturity_date', 'bdi_code'])
    quotes = []
    for q in opts.itertuples():
        lo = implied_vol(q.best_bid, q.spot, q.strike_price, q.days / 365, q.rate, q.bdi_code == 74)
        hi = implied_vol(q.best_ask, q.spot, q.strike_price, q.days / 365, q.rate, q.bdi_code == 74)
        if not np.isfinite([lo, hi]).all():
            continue
        quotes.append(dict(date=q.refdate, symbol=q.symbol, expiry=q.maturity_date,
                           type='Calls' if q.bdi_code == 74 else 'Puts', days=q.days,
                           horizon=q.days / 365 * 252, bid_iv=lo, ask_iv=hi,
                           bid=q.best_bid, ask=q.best_ask,
                           spread_pct=100 * (q.best_ask-q.best_bid)/((q.best_ask+q.best_bid)/2),
                           last_outside=bool(q.close < q.best_bid or q.close > q.best_ask)))
    return spot, pd.DataFrame(quotes, columns=['date', 'symbol', 'expiry', 'type', 'days', 'horizon',
                                              'bid_iv', 'ask_iv', 'bid', 'ask', 'spread_pct', 'last_outside'])


def figure(data, quotes):
    f = go.Figure()
    for low, high, label, color in [('p10', 'p90', 'Historical 10th–90th percentile', 'rgba(60,150,170,.18)'),
                                    ('p25', 'p75', 'Historical 25th–75th percentile', 'rgba(60,150,170,.30)')]:
        f.add_trace(go.Scatter(x=data.h, y=data[low], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        f.add_trace(go.Scatter(x=data.h, y=data[high], mode='lines', fill='tonexty', fillcolor=color,
                              line=dict(width=0), name=label, hoverinfo='skip'))
    f.add_trace(go.Scatter(x=data.h, y=data['median'], name='Historical median', line=dict(dash='dash', color='#a1a8b0')))
    f.add_trace(go.Scatter(x=data.h, y=data.current, name='As-of realized volatility', mode='lines+markers', line=dict(color='#42c8bc')))
    for j, q in enumerate(quotes.itertuples()):
        f.add_trace(go.Scatter(x=[q.horizon, q.horizon], y=[q.bid_iv, q.ask_iv],
                              mode='lines+markers', marker=dict(symbol='line-ew', size=14, line=dict(width=2, color='#f3b35c')),
                              line=dict(color='#f3b35c', width=3), name='Bid–ask IV', legendgroup='iv', showlegend=j == 0,
                              text=[f'{q.symbol} · {q.expiry.date()} · bid (sell)', f'{q.symbol} · {q.expiry.date()} · ask (buy)'],
                              hovertemplate='%{text}<br>IV %{y:.1f}%<br>Horizon %{x:.1f} trading-day equivalents<extra></extra>'))
    f.update_layout(xaxis=dict(type='log', title='Horizon (trading days; IV uses 252 × calendar days / 365)',
                               tickvals=HORIZONS, ticktext=list(map(str, HORIZONS))),
                    yaxis=dict(title='Annualized volatility (%)', rangemode='tozero'),
                    height=470, margin=dict(l=40, r=20, t=30, b=40), legend=dict(orientation='h', y=1.2),
                    hovermode='closest')
    return f


def render(base):
    st.subheader('IBOV volatility cone with quote-side IV')
    st.caption('This view uses its own controls; screener sidebar filters do not apply. IV bars are quoted expiries, not additional historical cone bands.')
    sources = [base/'ibov_daily.csv', base/'rates_cdi.csv', *sorted(base.glob('ibov_options_all*.parquet'))]
    try:
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sources)
        spot, quotes = load_inputs(str(base), signature)
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f'Cone data unavailable: {exc}')
        return
    if len(spot) < 505:
        st.info('At least 505 price observations are needed for this view.')
        return
    a, b, c = st.columns(3)
    kind = a.selectbox('Option type', ['Calls', 'Puts'], key='cone_type')
    history = b.selectbox('Historical reference window', ['5 trading years', '2 trading years', 'All history'], key='cone_window')
    available = quotes[quotes.type.eq(kind)]
    latest_iv = available.date.max() if not available.empty else spot.index[-1]
    if 'cone_date' not in st.session_state:
        st.session_state.cone_date = latest_iv.date()
    def set_date(value):
        st.session_state.cone_date = value.date()
    buttons = st.columns(4)
    buttons[0].button('Latest quoted IV', on_click=set_date, args=(latest_iv,), disabled=available.empty)
    buttons[1].button('Latest price', on_click=set_date, args=(spot.index[-1],))
    buttons[2].button('March 2020', on_click=set_date, args=(pd.Timestamp('2020-03-23'),))
    buttons[3].button('October 2008', on_click=set_date, args=(pd.Timestamp('2008-10-27'),))
    day = c.date_input('As-of date', min_value=spot.index[504].date(), max_value=spot.index[-1].date(), key='cone_date')
    actual = spot.loc[:pd.Timestamp(day)].index[-1]
    n = {'5 trading years': 1260, '2 trading years': 504, 'All history': 0}[history]
    data = cone(spot, actual, n)
    q = available[available.date.eq(actual)].sort_values('expiry')
    stretch = {'width': 'stretch'} if tuple(map(int, st.__version__.split('.')[:2])) >= (1, 50) else {'use_container_width': True}
    st.plotly_chart(figure(data, q), **stretch)
    st.caption(f'Price history: {spot.index[0].date()}–{spot.index[-1].date()}. Selected trading date: {actual.date()}. Historical distributions exclude this date.')
    if q.empty:
        st.info(f'No usable two-sided near-ATM {kind.lower()} on {actual.date()}. No older quotes or midpoint prices substituted; try Latest quoted IV.')
    else:
        st.write(f'**{len(q)} IV bars = {len(q)} qualifying expiries.** Lower endpoint: bid IV (selling). Upper endpoint: ask IV (buying).')
        display = q[['symbol', 'expiry', 'days', 'horizon', 'bid_iv', 'ask_iv', 'bid', 'ask', 'spread_pct', 'last_outside']].copy()
        display.expiry = display.expiry.dt.strftime('%Y-%m-%d')
        st.dataframe(display.round(2), hide_index=True, **stretch)
    with st.expander('Realized values and methodology'):
        st.dataframe(data.round(2), hide_index=True, **stretch)
        st.markdown('Realized volatility is the sample standard deviation of daily log IBOV returns × √252. '
                    'Reference samples are overlapping rolling windows, not independent observations; shaded percentiles are not confidence intervals. '
                    'IBOV is a total-return index, so dividends are not added again. Weekend rows are removed; missing sessions are not interpolated. '
                    'The reference window counts past endpoints; underlying return windows can begin earlier.\n\n'
                    'IV is a **European Black–Scholes model proxy** using cash IBOV, zero dividend yield and a flat, as-of CDI rate. '
                    'One nearest quoted strike per expiry/type is selected within approximately 5% of estimated forward S exp(rT), with 8–365 calendar days remaining. '
                    'Positive uncrossed bid/ask and successful 1%–300% IV inversions are required. Wide spreads remain visible. '
                    'Daily quotes and spot are not synchronized; size, freshness, the actual forward curve and executable fills are unknown. '
                    'IV horizons use calendar-to-trading-day equivalents. This is neither an alpha estimate nor a trading recommendation.')
