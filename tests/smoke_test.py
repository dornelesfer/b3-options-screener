"""
Smoke test for the deployed surface: metrics build + app render.

Runs in CI on every code push (.github/workflows/ci.yml) against the same
fresh-install stack Streamlit Cloud uses, so a library break (pandas 3 dropping
groupby columns, Streamlit removing use_container_width, ...) fails here
instead of on the deployed URL. Uses the committed data caches — no network.

    python tests/smoke_test.py
"""

import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


# 1. metrics build ------------------------------------------------------------
print("[1/3] screener_metrics.py")
t0 = time.time()
r = subprocess.run([sys.executable, "screener_metrics.py"], cwd=ROOT,
                   capture_output=True, text=True)
print(r.stdout[-1500:])
check(r.returncode == 0, f"screener_metrics exits 0 ({time.time() - t0:.0f}s)")
if r.returncode:
    print(r.stderr[-3000:])

import pandas as pd
import pyarrow.parquet as pq

chain = pq.read_table(str(ROOT / "data/screener/chain_latest.parquet")).to_pandas()
hist = pq.read_table(str(ROOT / "data/screener/history_daily.parquet")).to_pandas()
need = ["underlying", "symbol", "type", "K", "expiry", "dte", "mid", "iv", "delta",
        "moneyness", "smile_resid", "iv_minus_rv", "rv_matched", "iv_minus_rv_h",
        "cone_pct", "cone_z", "cp_gap", "parity_resid", "below_intrinsic",
        "volgan_pctile"]
missing = [c for c in need if c not in chain.columns]
check(not missing, f"chain_latest has expected columns (missing: {missing})")
check(len(chain) > 500, f"chain_latest has {len(chain):,} rows")
check(chain["iv"].notna().mean() > 0.5, "most options have an IV")
check("VALE5" not in set(chain["underlying"]), "stale underlying VALE5 dropped")
for und in ("IBOV", "PETR4", "VALE3"):
    check(und in set(chain["underlying"]), f"{und} present")
check(hist["spread_h_pctile"].notna().any(), "history has horizon-matched pctiles")
ib = chain[chain["underlying"] == "IBOV"]
check(ib["volgan_pctile"].notna().sum() > 0, "cVolGAN score attached for IBOV")

# 2. cVolGAN surface is order-independent --------------------------------------
print("[2/3] volgan_score.surface_from_chain")
import numpy as np
from volgan_score import surface_from_chain
s1 = surface_from_chain(ib)
s2 = surface_from_chain(ib.sample(frac=1, random_state=3))
check(s1 is not None and np.allclose(s1, s2), "surface identical under row shuffle")

# 3. app renders ---------------------------------------------------------------
print("[3/3] app_options_screener.py via AppTest")
import streamlit as st
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / "app_options_screener.py"), default_timeout=240).run()
check(not at.exception, "no exception: " + "; ".join(str(e.value)[:200] for e in at.exception))
check(not at.error, "no st.error")
check([t.label for t in at.tabs] == ["Rankings", "Smile", "IV vs RV history", "Anomalies"],
      "four tabs")
check(len(at.dataframe) >= 4, f"{len(at.dataframe)} tables rendered")
check(len(at.get("plotly_chart")) >= 5, f"{len(at.get('plotly_chart'))} charts rendered")
rich = at.dataframe[1].value
check(len(rich) > 0 and rich["iv"].max() < 200,
      f"moneyness filter keeps deep-ITM junk out of Richest (max iv {rich['iv'].max():.0f})")
print(f"  streamlit {st.__version__}, pandas {pd.__version__}, numpy {np.__version__}")

print()
if failures:
    print(f"{len(failures)} FAILED:", *failures, sep="\n  - ")
    sys.exit(1)
print("all smoke checks passed")
