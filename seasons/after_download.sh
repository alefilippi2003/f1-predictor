#!/bin/bash
# Dopo lo scaricamento: tabelle variabili per ogni stagione + apprendimento su tutte
cd "$(dirname "$0")/.."
while pgrep -f "download_all.sh" > /dev/null; do sleep 30; done
for y in 2023 2024 2025; do
  (cd seasons/$y && F1_SEASON=$y ../../.venv/bin/python ../../features.py 2>&1 | tail -1)
done
.venv/bin/python features.py 2>&1 | tail -1
.venv/bin/python learn.py 2>&1 | grep -v "fatto$" > data/learn_output.txt
cat data/learn_output.txt
