#!/usr/bin/env bash
# Create a local virtual environment with the pinned dependencies.
#
# Uses `uv` if available (fast, and works on systems without python3-venv),
# otherwise falls back to `python -m venv`.
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}

if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON" "$VENV"
    uv pip install --python "$VENV/bin/python" -r requirements.txt
    uv pip install --python "$VENV/bin/python" -e .
else
    "$PYTHON" -m venv "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
fi

echo
echo "Environment ready. Activate with:  source $VENV/bin/activate"
