"""Combina le matrici pilota x posizione dei due modelli (pesi scelti sul backtest).

Uso: python combine.py [round] [peso_simulatore]
Output: data/matrix_r{round}.csv  (righe = posizione 1..22 + Ritirato, colonne = piloti, cella = probabilita')
"""
import sys

import numpy as np
import pandas as pd

rnd = int(sys.argv[1]) if len(sys.argv) > 1 else 18
w_sim = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4

sim = pd.read_csv(f"data/matrix_sim_r{rnd}.csv", index_col="posizione", dtype={"posizione": str})
rank = pd.read_csv(f"data/matrix_rank_r{rnd}.csv", index_col="posizione", dtype={"posizione": str})
assert set(sim.columns) == set(rank.columns), set(sim.columns) ^ set(rank.columns)
m = w_sim * sim + (1 - w_sim) * rank[sim.columns]

# controlli: ogni pilota ha esattamente un esito (una posizione o il ritiro); ogni posizione
# ha al massimo un pilota; i ritiri attesi = posizioni rimaste vuote in fondo
D = m.shape[1]
for name, x in [("simulatore", sim), ("modello semplice", rank), ("combinato", m)]:
    assert np.allclose(x.sum(0), 1, atol=1e-6), f"{name}: colonne non sommano a 1"
    rows = x.drop("Ritirato").sum(1)
    assert (rows <= 1 + 1e-6).all(), f"{name}: una posizione ha piu' di un pilota"
    assert np.isclose(D - rows.sum(), x.loc["Ritirato"].sum(), atol=1e-6), f"{name}: ritiri incoerenti"
    assert np.allclose(rows.iloc[:10], 1, atol=1e-3), f"{name}: le prime posizioni devono essere sempre occupate"

pos_val = pd.Series(list(range(1, D + 1)) + [D + 1], index=m.index)  # ritiro = in fondo
exp_pos = m.mul(pos_val, axis=0).sum()
m = m[exp_pos.sort_values().index]
m.to_csv(f"data/matrix_r{rnd}.csv", float_format="%.6f")

summary = pd.DataFrame({
    "P(vittoria)": m.loc["1"], "P(podio)": m.iloc[:3].sum(), "P(punti)": m.iloc[:10].sum(),
    "P(ritiro)": m.loc["Ritirato"], "pos_media": exp_pos, "esito_piu_probabile": m.idxmax(),
})
pd.set_option("display.width", 200)
print(f"Combinazione: {w_sim:.0%} simulatore giro per giro + {1 - w_sim:.0%} modello semplice")
print(summary.loc[m.columns].to_string(formatters={
    "P(vittoria)": "{:.1%}".format, "P(podio)": "{:.1%}".format, "P(punti)": "{:.1%}".format,
    "P(ritiro)": "{:.1%}".format, "pos_media": "{:.1f}".format}))
print(f"\nSalvato data/matrix_r{rnd}.csv ({D} posizioni + 'Ritirato' x {D} piloti)")
