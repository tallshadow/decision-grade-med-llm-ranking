#!/usr/bin/env bash
set -euo pipefail

streamlit run src/medrank/label/app_streamlit.py \
  --server.address 0.0.0.0 \
  --server.port ${PORT:-8501} \
  --server.headless true
