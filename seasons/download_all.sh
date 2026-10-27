#!/bin/bash
# Scarica i dati delle stagioni storiche (una cartella per stagione, cache condivisa)
cd "$(dirname "$0")"
for y in 2023 2024 2025; do
  cd $y
  echo "=== $y ===" 
  F1_SEASON=$y ../../.venv/bin/python ../../data.py 2>&1 | grep -v "^R[0-9]" | tail -3
  F1_SEASON=$y ../../.venv/bin/python ../../laps.py 2>&1 | tail -9
  cd ..
done
