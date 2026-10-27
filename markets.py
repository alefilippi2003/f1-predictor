"""Mercati di scommessa calcolati dalle gare simulate (data/sims_r{round}.npz).

Le quote sono "eque": 1 / probabilita', senza il margine del bookmaker.

Uso: python markets.py [round] [giri_gara] [modello: ML_lgbm_t_w3 (default) | A | ML_lasso ...]
Output: data/mercati_r{round}.xlsx (un foglio per mercato) + CSV in data/mercati_r{round}/
"""
import itertools
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

RND = int(sys.argv[1]) if len(sys.argv) > 1 else 18
N_LAPS = int(sys.argv[2]) if len(sys.argv) > 2 else 56
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "ML_lgbm_t_w3"
TAG = "" if VARIANT == "A" else f"_{VARIANT}"   # stesso suffisso dei file di racesim.py
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 12)
TOP_COMBOS = 30

# testa a testa richiesti
H2H_DRIVERS = [("SAI", "ALB"), ("LAW", "LIN"), ("LEC", "HAM"), ("BEA", "OCO"), ("ALO", "STR"), ("GAS", "COL"),
               ("BOR", "HUL"), ("RUS", "ANT"), ("VER", "HAD"), ("NOR", "PIA"), ("VER", "ANT"), ("LEC", "HAD"),
               ("HAM", "NOR"), ("RUS", "NOR"), ("LAW", "GAS")]
H2H_TEAMS = [("Alpine", "Audi"), ("Aston Martin", "Williams"), ("Racing Bulls", "Alpine"),
             ("Red Bull Racing", "Mercedes")]

s = np.load(f"data/sims_r{RND}{TAG}.npz", allow_pickle=False)
drivers, teams = list(s["drivers"]), list(s["teams"])
order, dnf, dnf_lap, fastest = s["order"].astype(int), s["dnf"], s["dnf_lap"], s["fastest"].astype(int)
N, D = order.shape
idx = {d: i for i, d in enumerate(drivers)}
team_of = dict(zip(drivers, teams))
team_list = list(dict.fromkeys(teams))

pos = np.empty_like(order)
pos[np.arange(N)[:, None], order] = np.arange(1, D + 1)[None, :]  # posizione nell'ordine finale
fin = ~dnf
# classificato: arrivato oppure ritirato dopo aver percorso almeno il 90% della gara
classified = fin | (dnf_lap > 0.9 * N_LAPS)


def odds(p):
    p = np.asarray(p, float)
    return np.where(p > 0, 1 / np.maximum(p, 1e-12), np.nan)


def table(probs, label="probabilita'"):
    d = pd.DataFrame({"selezione": list(probs.keys()), label: list(probs.values())})
    d = d.sort_values(label, ascending=False).reset_index(drop=True)
    d["quota"] = odds(d[label])
    return d


def top_in(k):
    return {d: float(((pos[:, i] <= k) & fin[:, i]).mean()) for d, i in idx.items()}


def exact(k):
    return {d: float(((pos[:, i] == k) & fin[:, i]).mean()) for d, i in idx.items()}


def dead_heat(winners_per_sim, keys):
    """Probabilita' con regola dead-heat: in caso di parita' fra k selezioni, ognuna vale 1/k."""
    c = Counter()
    for ws in winners_per_sim:
        for w in ws:
            c[w] += 1 / len(ws)
    return {k: c[k] / N for k in keys}


sheets = {}

# ---------------------------------------------------------------- piloti
sheets["Vincente"] = table(exact(1))
sheets["Podio"] = table(top_in(3))
sheets["Primi 2"] = table(top_in(2))
sheets["2o posto"] = table(exact(2))
sheets["3o posto"] = table(exact(3))
sheets["Primi 6"] = table(top_in(6))
sheets["Zona punti"] = table(top_in(10))

# piazzamento finale: matrice pilota x posizione (+ ritirato)
pf = pd.DataFrame(index=[str(k) for k in range(1, D + 1)] + ["Ritirato"], columns=drivers, dtype=float)
for d, i in idx.items():
    counts = np.bincount(pos[fin[:, i], i] - 1, minlength=D) / N
    pf[d] = list(counts) + [dnf[:, i].mean()]
pf = pf[pf.drop("Ritirato").mul(np.arange(1, D + 1), axis=0).sum().add(pf.loc["Ritirato"] * (D + 1))
        .sort_values().index]
sheets["Piazzamento finale"] = pf.reset_index().rename(columns={"index": "posizione"})

rows = []
for a, b in H2H_DRIVERS:
    pa = float((pos[:, idx[a]] < pos[:, idx[b]]).mean())
    rows.append({"testa a testa": f"{a} - {b}", f"P(1)": pa, "P(2)": 1 - pa,
                 "quota 1": odds(pa), "quota 2": odds(1 - pa)})
sheets["Testa a testa"] = pd.DataFrame(rows)

sheets["Si classifica"] = table({d: float(classified[:, i].mean()) for d, i in idx.items()})

# ultimo classificato: fra i classificati, quello con la posizione peggiore
last = []
for r in range(N):
    c = np.where(classified[r])[0]
    last.append((drivers[c[np.argmax(pos[r, c])]],) if len(c) else ())
sheets["Ultimo classificato"] = table(dead_heat(last, drivers))

sheets["Giro piu' veloce"] = table({d: float((fastest == i).mean()) for d, i in idx.items()})
sheets["Giro veloce + vittoria"] = table({d: float(((fastest == i) & (order[:, 0] == i)).mean())
                                         for d, i in idx.items()})

first_out = []
for r in range(N):
    laps_r = np.where(dnf[r], dnf_lap[r], 10 ** 4)
    m = laps_r.min()
    first_out.append(tuple(drivers[i] for i in np.where(laps_r == m)[0]) if m < 10 ** 4 else ("Nessun ritiro",))
sheets["Primo a ritirarsi"] = table(dead_heat(first_out, drivers + ["Nessun ritiro"]))

# combinazioni
names = np.array(drivers)
top2 = names[order[:, :2]]
top3 = names[order[:, :3]]
fin_top3 = fin[np.arange(N)[:, None], order[:, :3]].all(1)


def combos(keys, label):
    c = Counter(map(tuple, keys))
    d = pd.DataFrame([{"combinazione": " - ".join(k), "probabilita'": v / N} for k, v in c.most_common(TOP_COMBOS)])
    d["quota"] = odds(d["probabilita'"])
    return d


sheets["Accoppiata"] = combos(np.sort(top2, axis=1), "accoppiata")
sheets["Accoppiata in ordine"] = combos(top2, "in ordine")
sheets["Trio in ordine"] = combos(top3[fin_top3], "trio")
sheets["Trio a girare"] = combos(np.sort(top3[fin_top3], axis=1), "trio")
# combinazione podio: coppie di piloti entrambi sul podio (in qualsiasi posizione)
c = Counter()
for row in np.sort(top3[fin_top3], axis=1):
    for pair in itertools.combinations(row, 2):
        c[pair] += 1
d = pd.DataFrame([{"combinazione": " + ".join(k), "probabilita'": v / N} for k, v in c.most_common(TOP_COMBOS)])
d["quota"] = odds(d["probabilita'"])
sheets["Coppie sul podio"] = d

# ---------------------------------------------------------------- scuderie
pairs = {tm: [i for i, t in enumerate(teams) if t == tm] for tm in team_list}
rows = []
for tm, (a, b) in pairs.items():
    both = fin[:, a] & fin[:, b]
    worst = np.maximum(pos[:, a], pos[:, b])
    r = {"scuderia": tm,
         "entrambe a podio": float((both & (worst <= 3)).mean()),
         "entrambe prime 6": float((both & (worst <= 6)).mean()),
         "entrambe a punti": float((both & (worst <= 10)).mean()),
         "entrambe classificate": float((classified[:, a] & classified[:, b]).mean())}
    for k in list(r)[1:]:
        r[f"quota {k}"] = float(odds(r[k]))
    rows.append(r)
sheets["Scuderie - entrambe"] = pd.DataFrame(rows).sort_values("entrambe a punti", ascending=False)

team_arr = np.array(teams)
sheets["Monoposto vincente"] = table({tm: float((team_arr[order[:, 0]] == tm).mean()) for tm in team_list})
sheets["Scuderia giro veloce"] = table({tm: float((team_arr[fastest] == tm).mean()) for tm in team_list})

first_team = [tuple(dict.fromkeys(team_of[d] for d in w)) if w != ("Nessun ritiro",) else w for w in first_out]
sheets["Primo costruttore ritirato"] = table(dead_heat(first_team, team_list + ["Nessun ritiro"]))

# punti di scuderia in ogni gara simulata (i ritirati non prendono punti)
pts = np.where(fin, POINTS[pos - 1], 0)
team_pts = {tm: pts[:, ii].sum(1) for tm, ii in pairs.items()}
rows = []
for a, b in H2H_TEAMS:
    pa, pb = float((team_pts[a] > team_pts[b]).mean()), float((team_pts[b] > team_pts[a]).mean())
    rows.append({"testa a testa": f"{a} - {b}", "P(1)": pa, "P(pari)": 1 - pa - pb, "P(2)": pb,
                 "quota 1": float(odds(pa)), "quota 2": float(odds(pb)),
                 "quota 1 (pari = rimborso)": float(odds(pa / (pa + pb))) if pa + pb else np.nan,
                 "quota 2 (pari = rimborso)": float(odds(pb / (pa + pb))) if pa + pb else np.nan})
sheets["Testa a testa scuderie"] = pd.DataFrame(rows)

tp = np.column_stack([team_pts[tm] for tm in team_list])
best_team = [tuple(np.array(team_list)[row == row.max()]) for row in tp]
sheets["Scuderia piu' punti"] = table(dead_heat(best_team, team_list))

# ---------------------------------------------------------------- salvataggio
out_dir = f"data/mercati_r{RND}{TAG}"
os.makedirs(out_dir, exist_ok=True)
with pd.ExcelWriter(f"data/mercati_r{RND}{TAG}.xlsx") as xw:
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name[:31], index=False)
        df.to_csv(f"{out_dir}/{name.replace(' ', '_').replace(chr(39), '')}.csv", index=False, float_format="%.6f")
print(f"{N} gare simulate -> {len(sheets)} mercati in data/mercati_r{RND}{TAG}.xlsx e {out_dir}/")
