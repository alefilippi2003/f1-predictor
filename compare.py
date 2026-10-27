"""Tabella di confronto fra modelli sul backtest (gare 6-17): ordine d'arrivo previsto vs reale.

Uso: python compare.py     (legge data/backtest_sim*.csv; calcola griglia e modello semplice)
"""
import glob
import warnings

import numpy as np
import pandas as pd

from evaluate import race_metrics, summarize
from scipy.optimize import minimize_scalar

from model import add_features, pl_loglik, simulate, walk_forward

warnings.filterwarnings("ignore")
NAMES = {"": "Simulatore A (qualifica + libere + forma)", "_B": "Simulatore B (FP3/sprint, senza qualifica)",
         "_B2": "Simulatore B2 (FP2/sprint, senza qualifica)", "_C": "Simulatore C (60% FP3/sprint + 40% qualifica)",
         "_C2": "Simulatore C2 (60% FP2/sprint + 40% qualifica)",
         "_D": "Simulatore D (griglia + libere + forma)",
         "_ML_lasso": "Simulatore + Lasso (tutte le stagioni)", "_ML_lasso_stag": "Simulatore + Lasso (solo stagione)",
         "_ML_lgbm": "Simulatore + LightGBM (tutte le stagioni)",
         "_ML_lgbm_rel": "Simulatore + LightGBM relativo", "_ML_lgbm_w3": "Simulatore + LightGBM, stagione x3",
         "_ML_lgbm_rel_w3": "Simulatore + LightGBM relativo, stagione x3", "_ML_lgbm_stag": "Simulatore + LightGBM (solo stagione)",
         "_ML_lgbm_t": "Simulatore + LightGBM regolato", "_ML_lgbm_t_w3": "Simulatore + LightGBM regolato, stagione x3"}

w = add_features(pd.read_csv("data/weekends.csv"))
rows_grid, rows_rank = [], []
LAST = int(w.loc[w["finish_pos"].notna(), "round"].max())
oos, races = walk_forward(w, LAST)
beta = minimize_scalar(lambda b: -pl_loglik(b, races), bounds=(0.01, 20), method="bounded").x
for rnd in range(6, LAST + 1):
    r = w[w["round"] == rnd]
    rows_grid.append(race_metrics(r.set_index("driver")["grid"].astype(float), r))
    o = oos[oos["round"] == rnd]
    pos, dnf = simulate(o["score"].values, o["p_dnf"].values, beta, n=4000)
    exp = np.where(dnf, len(o) + 1, pos).mean(0)
    rows_rank.append(race_metrics(pd.Series(exp, index=o["driver"].values), o))

out = [summarize(rows_grid, "Ordine della griglia"), summarize(rows_rank, "Modello semplice (classifica)")]
for path in sorted(glob.glob("data/backtest_sim*.csv")):
    tag = path[len("data/backtest_sim"):-4]
    d = pd.read_csv(path)
    if "m_errore_pos" not in d:
        continue
    d = d.rename(columns=lambda c: c[2:] if c.startswith("m_") else c)
    out.append(summarize(d.to_dict("records"), NAMES.get(tag, tag)))
t = pd.DataFrame(out)
pd.set_option("display.width", 220)
print(t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
t.to_csv("data/confronto_modelli.csv", index=False)
