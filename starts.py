"""Caos della partenza per pista: di quante posizioni cambia in media l'ordine
dalla griglia alla fine del giro 1 (2023-2026).

Uso: python starts.py   -> data/starts.csv
"""
import numpy as np
import pandas as pd

from data import get
from data import SEASON
from overtakes import HIST_YEARS, race_sessions

OUT = "data/starts.csv"
GRID_GAP_S = 0.25


def lap1_shuffle(sk):
    pos = get("position", session_key=sk)
    laps = get("laps", session_key=sk)
    if pos.empty or laps.empty:
        return None
    grid = pos.sort_values("date").groupby("driver_number")["position"].first()
    l2 = laps[laps["lap_number"] == 2].dropna(subset=["date_start"])
    l2 = l2.assign(t=pd.to_datetime(l2["date_start"], format="ISO8601")).sort_values("t")
    after = pd.Series(np.arange(1, len(l2) + 1), index=l2["driver_number"].values)
    d = pd.DataFrame({"grid": grid}).join(after.rename("lap1"), how="inner")
    d = d[d["grid"] <= 20]
    if len(d) < 15:
        return None
    # ordine relativo fra chi ha completato il giro 1
    return float((d["grid"].rank() - d["lap1"].rank()).abs().mean())


def lap1_gains_2026():
    """Posizioni guadagnate al giro 1 da ogni pilota in ogni gara 2026, al netto di quanto
    si guadagna in media partendo da quella casella (chi parte dietro tende a guadagnare)."""
    from racesim_rounds import round_circuits  # evita import circolari
    sess = race_sessions(SEASON)
    rows = []
    for rnd, circ in round_circuits().items():
        s = sess[sess["circuit_short_name"] == circ]
        if s.empty:
            continue
        sk = s["session_key"].iat[0]
        pos, laps = get("position", session_key=sk), get("laps", session_key=sk)
        if pos.empty or laps.empty:
            continue
        drv = get("drivers", session_key=sk).drop_duplicates("driver_number").set_index("driver_number")
        grid = pos.sort_values("date").groupby("driver_number")["position"].first()
        l2 = laps[laps["lap_number"] == 2].dropna(subset=["date_start"])
        l2 = l2.assign(t=pd.to_datetime(l2["date_start"], format="ISO8601")).sort_values("t")
        after = pd.Series(np.arange(1, len(l2) + 1), index=l2["driver_number"].values)
        # controllo incrociato: i primi 3 dopo il giro 1 secondo i tempi sul giro devono
        # coincidere con le posizioni ufficiali in quel momento, altrimenti scarto la gara
        pos_t = pos.assign(t=pd.to_datetime(pos["date"], format="ISO8601"))
        snap = pos_t[pos_t["t"] <= l2["t"].min()].sort_values("t").groupby("driver_number")["position"].last()
        if set(snap.nsmallest(3).index) != set(after.index[:3]):
            print(f"  R{rnd} {circ}: ordine al giro 1 incoerente fra le fonti, gara esclusa")
            continue
        d = pd.DataFrame({"grid": grid}).join(after.rename("lap1"), how="inner")
        d["gain"] = d["grid"].rank() - d["lap1"].rank()
        d["round"] = rnd
        d["driver"] = d.index.map(drv["name_acronym"])
        rows.append(d.reset_index(drop=True))
    g = pd.concat(rows, ignore_index=True)
    g["grid_bin"] = pd.cut(g["grid"], [0, 1, 3, 6, 10, 15, 22], labels=False)
    g["resid"] = g["gain"] - g.groupby("grid_bin")["gain"].transform("mean")
    return g


def driver_start_form(gains, before_round, prior=15):
    """Guadagno medio residuo al via per pilota, solo gare precedenti, 'tirato' verso 0.
    prior=15: l'affidabilita' fra meta' stagione e l'altra e' ~0.3, quindi con ~17 gare
    il valore osservato va circa dimezzato."""
    g = gains[gains["round"] < before_round]
    s = g.groupby("driver")["resid"].agg(["sum", "size"])
    return (s["sum"] / (s["size"] + prior)).to_dict()


def start_stats(a, b, n=4000, cars=20, seed=2):
    """(P(poleman primo dopo il via), posizioni cambiate in media) con rumore a + b*(casella-1)."""
    rng = np.random.default_rng(seed)
    sd = a + b * np.arange(cars)
    t = np.arange(cars)[None, :] * GRID_GAP_S + rng.normal(0, 1, (n, cars)) * sd[None, :]
    rank = t.argsort(1).argsort(1)
    return (rank[:, 0] == 0).mean(), np.abs(rank - np.arange(cars)[None, :]).mean()


def calibrate_start(gains, before_round):
    """Rumore al via crescente con la casella, calibrato sulle partenze 2026 precedenti:
    quante volte il poleman e' ancora primo dopo il giro 1 e quante posizioni cambiano in media."""
    g = gains[(gains["round"] < before_round) & (gains["grid"] <= 20)]
    lap1_rank = g.groupby("round")["lap1"].rank()
    grid_rank = g.groupby("round")["grid"].rank()
    pole = g[grid_rank == 1]
    p_pole = float((lap1_rank[pole.index] == 1).mean())
    shuffle = float((grid_rank - lap1_rank).abs().mean())
    best = None
    for a in np.linspace(0.05, 1.5, 30):
        for b in np.linspace(0.0, 0.2, 21):
            pp, sh = start_stats(a, b)
            err = ((pp - p_pole) / 0.1) ** 2 + ((sh - shuffle) / 0.3) ** 2
            if best is None or err < best[0]:
                best = (err, a, b)
    return best[1], best[2], (p_pole, shuffle)


def positions_per_second(sd, n=20000, cars=20, seed=1):
    """Quante posizioni guadagna in media al via un pilota di centro gruppo con 1 s di vantaggio."""
    rng = np.random.default_rng(seed)
    base = np.arange(cars) * GRID_GAP_S
    gains = []
    for bias in (0.0, 0.5):
        t = base[None, :] + rng.normal(0, sd, (n, cars))
        t[:, 10] -= bias
        gains.append(10 - t.argsort(1).argsort(1)[:, 10].mean())
    return (gains[1] - gains[0]) / 0.5


def build():
    rows = []
    for y in HIST_YEARS + [2026]:
        for _, s in race_sessions(y).iterrows():
            m = lap1_shuffle(s["session_key"])
            if m is not None:
                rows.append({"year": y, "circuit": s["circuit_short_name"], "date": s["date_start"],
                             "shuffle": m})
                print(f"{y} {s['circuit_short_name']:20s} {m:.2f}", flush=True)
    return pd.DataFrame(rows)


def start_sd_table(n=20000, cars=20, seed=0):
    """Rimescolamento medio prodotto da un rumore N(0, sd) al via, per vari sd."""
    rng = np.random.default_rng(seed)
    sds = np.linspace(0.05, 4.0, 60)
    out = []
    for sd in sds:
        t = np.arange(cars)[None, :] * GRID_GAP_S + rng.normal(0, sd, (n, cars))
        rank = t.argsort(1).argsort(1)
        out.append(np.abs(rank - np.arange(cars)[None, :]).mean())
    return sds, np.array(out)


def target_shuffle(starts, circuit, past_circuits=None, use_track=False):
    """Livello 2026 (gare precedenti), opzionalmente x difficolta' relativa della pista.
    Di default la pista NON si usa: il rimescolamento al via non e' stabile negli anni
    (correlazione pista per pista 2024 vs 2025 = 0.04), e' dominato dagli episodi."""
    h = starts[starts["year"].isin(HIST_YEARS)]
    n26 = starts[starts["year"] == SEASON]
    if past_circuits is not None:
        n26 = n26[n26["circuit"].isin(past_circuits)]
    level = n26["shuffle"].mean() if len(n26) else h["shuffle"].mean()
    hc = h[h["circuit"] == circuit]["shuffle"]
    rel = hc.mean() / h["shuffle"].mean() if len(hc) else 1.0
    rel = (len(hc) * rel + 1.0) / (len(hc) + 1)  # pochi dati -> verso la media
    return level * rel if use_track else level


if __name__ == "__main__":
    s = build()
    s.to_csv(OUT, index=False)
    h = s[s["year"].isin(HIST_YEARS)].groupby("circuit")["shuffle"].mean()
    n = s[s["year"] == 2026].set_index("circuit")["shuffle"]
    print(pd.DataFrame({"storico": h, "2026": n}).sort_values("storico").round(2).to_string())
    print(f"media storico {s[s['year'] < 2026]['shuffle'].mean():.2f}, 2026 {n.mean():.2f}")
