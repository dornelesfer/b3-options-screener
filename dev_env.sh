#!/usr/bin/env bash
# One-command local environment matching what CI and Streamlit Cloud run
# (Python 3.12, latest libraries within requirements.txt bounds). The Mac's
# anaconda is Python 3.9 / pandas 2 / Streamlit 1.32 and has already hidden
# one production break (pandas 3 groupby). Use this instead.
#
#   ./dev_env.sh                  # create/refresh .venv
#   .venv/bin/python tests/smoke_test.py
#   .venv/bin/python -m streamlit run app_options_screener.py --server.port 8601
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv (https://astral.sh/uv) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv .venv --python 3.12 --allow-existing
uv pip install --python .venv/bin/python -r requirements.txt pyflakes
.venv/bin/python - <<'EOF'
import sys, streamlit, pandas, numpy, pyarrow
print(f"ready: python {sys.version.split()[0]}, streamlit {streamlit.__version__}, "
      f"pandas {pandas.__version__}, numpy {numpy.__version__}, pyarrow {pyarrow.__version__}")
EOF
echo "next: .venv/bin/python tests/smoke_test.py"
