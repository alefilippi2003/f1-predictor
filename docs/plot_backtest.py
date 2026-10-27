"""Grafico del README: errore medio di posizione per stagione (backtest walk-forward).

Uso: python docs/plot_backtest.py   (dalla cartella principale, dopo compare.py per ogni stagione)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEASONS = {2023: "seasons/2023", 2024: "seasons/2024", 2025: "seasons/2025", 2026: "."}
SERIES = [("Ordine della griglia", "Starting grid order", "#a3a29d"),
          ("Simulatore A (qualifica + libere + forma)", "Simulator + linear pace model", "#eb6834"),
          ("Simulatore + LightGBM regolato, stagione x3", "Simulator + tuned LightGBM (final)", "#2a78d6")]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

rows = {y: pd.read_csv(f"{d}/data/confronto_modelli.csv").set_index("modello") for y, d in SEASONS.items()}
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
x = np.arange(len(SEASONS))
w = 0.26
for k, (key, label, color) in enumerate(SERIES):
    vals = [rows[y].loc[key, "errore_pos"] for y in SEASONS]
    bars = ax.bar(x + (k - 1) * w, vals, w - 0.02, color=color, label=label, zorder=2)
    if k == len(SERIES) - 1:  # etichette solo sul modello finale
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
n = [int(rows[y].loc[SERIES[0][0], "gare"]) for y in SEASONS]
ax.set_xticks(x, [f"{y}\n{k} races" for y, k in zip(SEASONS, n)], color=MUTED)
ax.set_ylim(0, 3.2)
ax.set_ylabel("Mean |predicted - actual| finishing position", color=MUTED, fontsize=9)
ax.set_title("Walk-forward backtest: lower is better", loc="left", color=INK, fontsize=12, pad=12)
ax.grid(axis="y", color="#e4e3df", zorder=0)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c9c8c3")
ax.tick_params(axis="y", colors=MUTED, length=0)
ax.tick_params(axis="x", length=0)
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.02), labelcolor=INK)
fig.tight_layout()
fig.savefig("docs/backtest.png", facecolor=SURFACE)
print("docs/backtest.png")
