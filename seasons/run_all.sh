#!/bin/bash
# variabili per ogni stagione -> apprendimento -> grande backtest (senza attese su pgrep)
cd "$(dirname "$0")/.."
for y in 2023 2024 2025; do
  (cd seasons/$y && F1_SEASON=$y ../../.venv/bin/python ../../features.py 2>&1 | tail -1) &
done
.venv/bin/python features.py 2>&1 | tail -1
wait
.venv/bin/python learn.py 2>&1 | grep -v "fatto$" > data/learn_output.txt
echo "learn finito $(date)"
./seasons/big_backtest.sh > data/big_backtest_output.txt 2>&1
echo "backtest finito $(date)"
