"""Difficolta' di sorpasso per pista, dai duelli giro per giro (2023-2026).

Un "duello" = a fine giro L l'inseguitore B e' a meno di 1 s dalla macchina A davanti.
Esito: a fine giro L+1 B e' davanti ad A? Il vantaggio di passo di B su A (in secondi,
al netto di mescola ed eta' gomma) viene dal modello del passo della stessa gara.

  P(sorpasso) = sigmoide(a_pista + b * vantaggio_s)

a_pista dallo storico 2023-2025 (difficolta' relativa della pista); il livello e la
pendenza del 2026 si stimano sulle gare 2026 (vedi fit_2026).

Uso: python overtakes.py     scarica lo storico, costruisce data/duels.csv e stampa le soglie
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from data import SEASON, get
from laps import session_laps
from pace import clean_race_laps, fit_pace

warnings.filterwarnings("ignore")
# storico = solo le stagioni DRS precedenti a quella che si prevede (niente dati dal futuro)
HIST_YEARS = [y for y in (2023, 2024, 2025) if y < SEASON]
DUEL_GAP_S = 1.0
OUT = "data/duels.csv"
# piste con layout cambiato nel periodo: lo storico vale solo dall'anno indicato
LAYOUT_FROM = {"Catalunya": 2023}


def race_sessions(year):
    s = get("sessions", year=year)
    s = s[(s["session_name"] == "Race") & ~s["is_cancelled"].fillna(False)]
    return s.sort_values("date_start")


def duels_for_race(year, sk, circuit):
    drivers = get("drivers", session_key=sk).drop_duplicates("driver_number").set_index("driver_number")
    l, _ = session_laps(0, circuit, "R", sk, drivers)
    if l is None or l.empty:
        return None
    l = l.dropna(subset=["driver"])

    # passo atteso di ogni giro (pilota + mescola + degrado), dal modello della stessa gara
    c = clean_race_laps(l)
    ok = c.groupby("driver").size()
    c = c[c["driver"].isin(ok[ok >= 10].index)]
    if c["driver"].nunique() < 10:
        return None
    pace, deg, _, _, ref, coef = fit_pace(c, "lap_number")
    off = {"SOFT": 0.0, "MEDIUM": coef.get("off_MEDIUM", 0.0), "HARD": coef.get("off_HARD", 0.0)}
    l["exp_pct"] = l["driver"].map(coef) + l["compound"].map(off) + \
        l["compound"].map(deg).fillna(0) * l["tyre_age"]
    l["exp_s"] = l["exp_pct"] / 100 * ref

    l["t_end"] = pd.to_datetime(l["date_start"], format="ISO8601") + pd.to_timedelta(l["lap_duration"], unit="s")
    w = l.pivot_table(index="lap_number", columns="driver", values="t_end", aggfunc="first")
    exp = l.pivot_table(index="lap_number", columns="driver", values="exp_s", aggfunc="first")
    bad = l[l["neutralized"] | l["is_pit_in_lap"] | l["is_pit_out_lap"].fillna(False)]
    bad = set(zip(bad["lap_number"], bad["driver"]))
    neutral = set(l.loc[l["neutralized"], "lap_number"])

    rows = []
    for L in range(3, int(l["lap_number"].max())):
        if L in neutral or L + 1 in neutral or L not in w.index or L + 1 not in w.index:
            continue
        t0 = w.loc[L].dropna().sort_values()
        names, times = t0.index.tolist(), t0.values
        for i in range(1, len(names)):
            a, b = names[i - 1], names[i]
            gap = (times[i] - times[i - 1]) / np.timedelta64(1, "s")
            if gap >= DUEL_GAP_S:
                continue
            if {(L, a), (L, b), (L + 1, a), (L + 1, b)} & bad:
                continue
            ta, tb = w.at[L + 1, a], w.at[L + 1, b]
            ea, eb = exp.at[L + 1, a] if a in exp else np.nan, exp.at[L + 1, b] if b in exp else np.nan
            if pd.isna(ta) or pd.isna(tb) or pd.isna(ea) or pd.isna(eb):
                continue
            rows.append({"year": year, "circuit": circuit, "lap": L, "ahead": a, "behind": b,
                         "gap_s": gap, "delta_s": ea - eb, "passed": int(tb < ta)})
    return pd.DataFrame(rows)


def build(years):
    frames = []
    for y in years:
        for _, s in race_sessions(y).iterrows():
            if y < LAYOUT_FROM.get(s["circuit_short_name"], 0):
                continue
            try:
                d = duels_for_race(y, s["session_key"], s["circuit_short_name"])
            except Exception as e:
                print(f"  ! {y} {s['circuit_short_name']}: {e}")
                continue
            if d is not None and len(d):
                frames.append(d)
                print(f"{y} {s['circuit_short_name']:20s} duelli {len(d):4d}  sorpassi {d['passed'].sum():3d}",
                      flush=True)
    return pd.concat(frames, ignore_index=True)


def _nll(params, X, y, offset, l2):
    z = offset + X @ params
    return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(params[1:-1] ** 2)


def fit_history(duels, years=HIST_YEARS):
    """a_pista + pendenza comune sugli anni dati. Le piste con pochi dati vengono 'tirate'
    verso la media (regolarizzazione)."""
    h = duels[duels["year"].isin(years)]
    if h.empty:
        return {"base": 0.0, "track": {}, "slope": 2.0}
    circuits = sorted(h["circuit"].unique())
    X = np.column_stack([np.ones(len(h))] + [(h["circuit"] == c).astype(float) for c in circuits]
                        + [h["delta_s"].values])
    res = minimize(_nll, np.zeros(X.shape[1]), args=(X, h["passed"].values, 0.0, 2.0), method="L-BFGS-B")
    base, eff, slope = res.x[0], res.x[1:-1], res.x[-1]
    return {"base": base, "track": dict(zip(circuits, eff)), "slope": slope}


def fit_2026(duels, hist, before_round_circuits=None):
    """Livello e pendenza 2026, tenendo la difficolta' relativa delle piste dallo storico."""
    d = duels[duels["year"] == SEASON]
    if before_round_circuits is not None:
        d = d[d["circuit"].isin(before_round_circuits)]
    if len(d) < 50:
        return {"shift": 0.0, "slope": hist["slope"]}
    off = hist["base"] + d["circuit"].map(hist["track"]).fillna(0.0).values
    X = np.column_stack([np.ones(len(d)), d["delta_s"].values])
    nll = lambda p: np.sum(np.logaddexp(0, off + X @ p) - d["passed"].values * (off + X @ p))
    p = minimize(nll, [0.0, hist["slope"]], method="L-BFGS-B").x
    return {"shift": p[0], "slope": p[1]}


AGE_EDGES = [1, 2, 3, 5, 10]  # fasce: 1, 2, 3, 4-5, 6-10, >10 giri consecutivi in scia
AGE_LABELS = ["1", "2", "3", "4-5", "6-10", ">10"]


def add_duel_age(duels):
    """Da quanti giri consecutivi l'inseguitore e' in scia (< 1 s) alla stessa macchina."""
    d = duels.sort_values(["year", "circuit", "ahead", "behind", "lap"]).copy()
    new = d.groupby(["year", "circuit", "ahead", "behind"])["lap"].diff() != 1
    d["age"] = d.groupby(new.cumsum()).cumcount() + 1
    d["age_bin"] = np.searchsorted(AGE_EDGES, d["age"], side="left")
    return d


def similar_tracks(circuit, hist, k=8):
    thr = {c: -(hist["base"] + v) / hist["slope"] for c, v in hist["track"].items()}
    if circuit not in thr:
        return list(thr)
    return sorted(thr, key=lambda c: abs(thr[c] - thr[circuit]))[:k]


def age_curve(duels, hist, circuit, k=8):
    """Effetto (logit) dei giri passati in scia, stimato sulle k piste storiche piu' simili,
    a parita' di pista e di vantaggio di passo. Centrato: media pesata sui duelli = 0,
    cosi' il livello medio di sorpasso della pista resta quello stimato prima."""
    d = add_duel_age(duels)
    d = d[d["year"].isin(HIST_YEARS) & d["circuit"].isin(similar_tracks(circuit, hist, k))]
    if d.empty:
        return np.zeros(len(AGE_LABELS))
    off = hist["base"] + d["circuit"].map(hist["track"]).fillna(0.0).values + hist["slope"] * d["delta_s"].values
    X = np.column_stack([(d["age_bin"] == i).astype(float) for i in range(len(AGE_LABELS))])
    y = d["passed"].values

    def nll(g):
        z = off + X @ g
        return np.sum(np.logaddexp(0, z) - y * z) + 0.05 * np.sum(g ** 2)

    g = minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B").x
    w = X.sum(0)
    return g - (g * w).sum() / w.sum()


def pass_logit(circuit, hist, y26):
    return hist["base"] + hist["track"].get(circuit, 0.0) + y26["shift"]


def threshold_s(circuit, hist, y26, p=0.5):
    """Vantaggio di passo (s/giro) che serve per avere probabilita' p di passare in un giro."""
    return (np.log(p / (1 - p)) - pass_logit(circuit, hist, y26)) / y26["slope"]


def compare_eras(duels):
    """Lo storico (DRS) dice qualcosa sul 2026 (overtake mode)? Confronto la difficolta'
    relativa di ogni pista stimata separatamente nelle due ere."""
    from scipy.stats import pearsonr, spearmanr
    h, n = fit_history(duels, HIST_YEARS), fit_history(duels, [2026])
    common = sorted(set(h["track"]) & set(n["track"]))
    d = pd.DataFrame({"pista": common,
                      "storico": [h["base"] + h["track"][c] for c in common],
                      "2026": [n["base"] + n["track"][c] for c in common]})
    d["P storico"] = expit(d["storico"])
    d["P 2026"] = expit(d["2026"])
    print("\n=== Confronto ere: P(sorpasso in un giro) a pari passo, inseguitore entro 1 s ===")
    print(d.sort_values("storico")[["pista", "P storico", "P 2026"]].to_string(
        index=False, float_format=lambda x: f"{x:.1%}"))
    print(f"\nPearson {pearsonr(d['storico'], d['2026']).statistic:.2f}   "
          f"Spearman {spearmanr(d['storico'], d['2026']).statistic:.2f}   su {len(d)} piste")
    print(f"pendenza (quanto conta 1 s di vantaggio): storico {h['slope']:.2f}, 2026 {n['slope']:.2f}")
    h26 = duels[duels["year"] == 2026]
    hh = duels[duels["year"].isin(HIST_YEARS)]
    print(f"sorpassi per duello: storico {hh['passed'].mean():.1%}, 2026 {h26['passed'].mean():.1%}")
    return d


def main():
    if os.path.exists(OUT):
        duels = pd.read_csv(OUT)
    else:
        duels = build(HIST_YEARS + [2026])
        duels.to_csv(OUT, index=False)
    print(f"\n{len(duels)} duelli, {duels['passed'].sum()} sorpassi")
    hist = fit_history(duels)
    y26 = fit_2026(duels, hist)
    print(f"pendenza storico {hist['slope']:.2f}/s, 2026 {y26['slope']:.2f}/s, spostamento 2026 {y26['shift']:+.2f}")
    rows = []
    for c in sorted(set(duels["circuit"])):
        h = duels[(duels["circuit"] == c) & duels["year"].isin(HIST_YEARS)]
        rows.append({"pista": c, "duelli_storico": len(h), "sorpassi_storico": int(h["passed"].sum()),
                     "P(passa) a pari passo": expit(pass_logit(c, hist, y26)),
                     "soglia 50% (s/giro)": threshold_s(c, hist, y26)})
    t = pd.DataFrame(rows).sort_values("soglia 50% (s/giro)", ascending=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    t.to_csv("data/overtaking_by_track.csv", index=False)
    compare_eras(duels).to_csv("data/overtaking_eras.csv", index=False)


if __name__ == "__main__":
    main()
