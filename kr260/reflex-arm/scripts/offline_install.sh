#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then /usr/bin/python3 -m venv .venv; fi
.venv/bin/python -m pip install --no-index --find-links wheelhouse \
 'torch==2.8.0+cpu' 'stable-baselines3==2.7.1' 'gymnasium==1.2.0' 'numpy==2.2.6' \
 'pytest==8.4.2' 'PyYAML==6.0.3' 'opencv-python-headless==4.12.0.88' 'setuptools==80.9.0'
.venv/bin/python -m pip install --no-index --no-build-isolation --no-deps -e .
.venv/bin/python -m pip freeze > requirements.lock
