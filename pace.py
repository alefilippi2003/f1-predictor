"""Livelli 1-2: dal dataset giro per giro al passo gara di ogni pilota.

race_pace(laps)      passo "vero" in gara per pilota/weekend (al netto di carburante e gomme)
practice_pace(laps)  stesso modello sui long run delle libere (+ sprint)
pace_table()         tabella pilota x weekend con tutti i passi e le feature
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

warnings.filterwarnings("ignore")
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def clean_race_laps(l):
    """Giri di gara rappresentativi del passo: niente giro 1, box, SC/VSC, giri anomali."""
    l = l[l["lap_duration"].notna() & (l["lap_number"] > 1) & ~l["is_pit_out_lap"].fillna(False)
          & ~l["is_pit_in_lap"].fillna(False) & ~l["neutralized"].fillna(False)
          & l["compound"].isin(COMPOUNDS) & l["tyre_age"].notna()]
    if l.empty:
        return l
    # via giri lenti non segnalati (bandiere gialle locali, errori, danni): > 5% sopra il
    # giro "tipico" del pilota
    med = l.groupby("driver")["lap_duration"].transform("median")
    return l[(l["lap_duration"] < med * 1.05) & (l["lap_duration"] > med * 0.95)]


def fit_pace(l, lap_term):
    """Regressione robusta: tempo = driver + mescola + degrado_mescola*eta' + lap_term*k.
    Ritorna (passo pilota in % rispetto al migliore, degrado %/giro per mescola, residuo %)."""
    ref = l["lap_duration"].median()
    y = (l["lap_duration"] / ref - 1) * 100
    X = pd.get_dummies(l["driver"], dtype=float)
    drivers = list(X.columns)
    for c in COMPOUNDS[1:]:
        X[f"off_{c}"] = (l["compound"] == c).astype(float)
    for c in COMPOUNDS:
        X[f"deg_{c}"] = np.where(l["compound"] == c, l["tyre_age"], 0.0)
    X["k"] = l[lap_term].astype(float)
    X = X.loc[:, X.abs().sum() > 0]
    m = HuberRegressor(alpha=1e-3, max_iter=500, fit_intercept=True).fit(X.values, y.values)
    coef = pd.Series(m.coef_, index=X.columns)
    pace = coef.reindex(drivers)
    pace = pace - pace.min()
    deg = {c: coef.get(f"deg_{c}", np.nan) for c in COMPOUNDS}
    resid = float(np.median(np.abs(y.values - m.predict(X.values))) * 1.4826)
    return pace, deg, resid, coef.get("k", np.nan), ref, coef


def race_pace(laps):
    rows, degs = [], []
    for (rnd, sess), l in laps[laps["session"] == "R"].groupby(["round", "session"]):
        c = clean_race_laps(l)
        if c["driver"].nunique() < 10:
            continue
        # i piloti con troppo pochi giri puliti (ritiri precoci) non sono stimabili
        ok = c.groupby("driver").size()
        c = c[c["driver"].isin(ok[ok >= 10].index)]
        pace, deg, resid, fuel, ref, _ = fit_pace(c, "lap_number")
        rows.append(pd.DataFrame({"round": rnd, "driver": pace.index, "race_pace": pace.values}))
        degs.append({"round": rnd, **{f"deg_{k}": v for k, v in deg.items()},
                     "fuel_pct_per_lap": fuel, "lap_noise_pct": resid, "race_ref_lap": ref})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(degs)


def longrun_laps(l):
    """Giri dei long run in libere/sprint: stint >= 4 giri puliti e costanti."""
    l = l[l["lap_duration"].notna() & ~l["is_pit_out_lap"].fillna(False) & ~l["is_pit_in_lap"].fillna(False)
          & l["compound"].isin(COMPOUNDS) & l["tyre_age"].notna() & ~l["neutralized"].fillna(False)]
    stint_med = l.groupby(["session", "driver", "stint"])["lap_duration"].transform("median")
    l = l[(l["lap_duration"] < stint_med * 1.02) & (l["lap_duration"] > stint_med * 0.98)]
    g = l.groupby(["session", "driver", "stint"])
    l = l[g["lap_duration"].transform("size") >= 4]
    l = l.assign(stint_lap=l.groupby(["session", "driver", "stint"]).cumcount())
    return l


def practice_pace(laps):
    rows = []
    sess = laps[laps["session"].isin(["FP1", "FP2", "FP3", "S"])]
    for rnd, l in sess.groupby("round"):
        lr = longrun_laps(l)
        if lr["driver"].nunique() < 8:
            continue
        # ogni sessione ha la sua pista: normalizzo per sessione prima di unire
        lr = lr.assign(lap_duration=lr["lap_duration"] / lr.groupby("session")["lap_duration"].transform("median")
                       * lr["lap_duration"].median())
        # nelle libere il carburante e' ignoto: stint_lap assorbe carburante+degrado medio
        pace, deg, resid, *_ = fit_pace(lr, "stint_lap")
        n = lr.groupby("driver").size()
        rows.append(pd.DataFrame({"round": rnd, "driver": pace.index, "fp_pace": pace.values,
                                  "fp_laps": n.reindex(pace.index).values}))
    return pd.concat(rows, ignore_index=True)


def session_pace(laps, code):
    """Passo per pilota/weekend da UNA sessione: long run per le libere, giri di gara per la sprint."""
    rows = []
    for rnd, l in laps[laps["session"] == code].groupby("round"):
        if code == "S":
            c = clean_race_laps(l)
            ok = c.groupby("driver").size()
            c = c[c["driver"].isin(ok[ok >= 5].index)]
            if c["driver"].nunique() < 10:
                continue
            pace, *_ = fit_pace(c, "lap_number")
        else:
            lr = longrun_laps(l)
            if lr["driver"].nunique() < 8:
                continue
            pace, *_ = fit_pace(lr, "stint_lap")
        rows.append(pd.DataFrame({"round": rnd, "driver": pace.index, f"pace_{code}": pace.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["round", "driver", f"pace_{code}"])


def pace_table(laps=None):
    laps = pd.read_csv("data/laps.csv") if laps is None else laps
    rp, track = race_pace(laps)
    # il distacco in qualifica viene dai risultati ufficiali (i giri grezzi hanno anomalie)
    t = pd.read_csv("data/weekends.csv")[["round", "event", "driver", "team", "grid", "quali_pos",
                                          "quali_gap_pct", "finish_pos", "dnf", "points"]]
    t = t.rename(columns={"quali_gap_pct": "quali_pace"})
    t = t.merge(practice_pace(laps), on=["round", "driver"], how="left") \
         .merge(rp, on=["round", "driver"], how="left")
    for code in ["FP2", "FP3", "S"]:
        t = t.merge(session_pace(laps, code), on=["round", "driver"], how="left")
    return t.sort_values(["round", "driver"]).reset_index(drop=True), track


if __name__ == "__main__":
    t, track = pace_table()
    pd.set_option("display.width", 220)
    print(track.round(3).to_string(index=False))
    from scipy.stats import spearmanr
    c = t.dropna(subset=["race_pace"])
    print("\nCorrelazione (Spearman) col passo gara vero, per weekend (media):")
    for f in ["quali_pace", "fp_pace"]:
        r = c.groupby("round").apply(lambda g: spearmanr(g[f], g["race_pace"], nan_policy="omit").statistic)
        print(f"  {f:11s} {r.mean():.3f}")
    print(t[t["round"] == 17].sort_values("race_pace")[["driver", "team", "quali_pace", "fp_pace", "race_pace",
                                                         "finish_pos"]].round(3).head(10).to_string(index=False))
