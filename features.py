"""Tabella delle variabili candidate: una riga per pilota per gara, tutte separate.

Ogni variabile usa solo informazioni disponibili PRIMA della partenza (le "forme" usano
solo le gare precedenti). L'obiettivo e' il passo gara reale (race_pace) e, come controllo,
la posizione d'arrivo.

Uso (dentro la cartella di una stagione):  F1_SEASON=2024 python ../../features.py
Output: data/features.csv
"""
import os
import warnings

import numpy as np
import pandas as pd

from data import SEASON, get
from model import add_features
from overtakes import HIST_YEARS, fit_history
from pace import longrun_laps, fit_pace, practice_pace, race_pace, session_pace
from racesim_rounds import round_circuits
from starts import driver_start_form, lap1_gains_2026

warnings.filterwarnings("ignore")
SEASONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seasons")


def teammate_delta(df, col):
    """Valore del pilota meno quello del compagno di squadra nello stesso weekend."""
    other = df.groupby(["round", "team"])[col].transform("sum") - df[col]
    n = df.groupby(["round", "team"])[col].transform("count")
    return np.where(n == 2, df[col] - other, np.nan)


def best_practice_lap(laps):
    fp = laps[laps["session"].isin(["FP1", "FP2", "FP3"]) & laps["lap_duration"].notna()]
    b = fp.groupby(["round", "driver"])["lap_duration"].min().rename("b").reset_index()
    b["fp_best_lap_gap"] = (b["b"] / b.groupby("round")["b"].transform("min") - 1) * 100
    return b[["round", "driver", "fp_best_lap_gap"]]


def weekend_degradation(laps):
    """Degrado medio (%/giro) stimato dai long run del weekend (libere + sprint)."""
    rows = []
    for rnd, l in laps[laps["session"].isin(["FP1", "FP2", "FP3", "S"])].groupby("round"):
        lr = longrun_laps(l)
        if lr["driver"].nunique() < 8:
            continue
        _, deg, *_ = fit_pace(lr, "stint_lap")
        v = [x for x in deg.values() if pd.notna(x)]
        rows.append({"round": rnd, "weekend_deg": float(np.mean(v)) if v else np.nan})
    return pd.DataFrame(rows)


def wet_before_race():
    """Pioggia in libere o qualifica (nota prima della gara): quota delle rilevazioni meteo
    con pioggia nella sessione piu' bagnata (0 = asciutto, 1 = pioggia per tutta la sessione)."""
    s = get("sessions", year=SEASON)
    s = s[~s["is_cancelled"].fillna(False)]
    circ = round_circuits()
    rows = []
    for rnd, c in circ.items():
        ss = s[(s["circuit_short_name"] == c) & s["session_name"].isin(
            ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Sprint Qualifying", "Sprint Shootout"])]
        wet = 0.0
        for sk in ss["session_key"]:
            w = get("weather", session_key=sk)
            if not w.empty:
                wet = max(wet, float((w["rainfall"] > 0).mean()))
        rows.append({"round": rnd, "wet_before_race": wet})
    return pd.DataFrame(rows)


def track_features(circuits):
    """Difficolta' di sorpasso (storico precedente) e safety car medie per pista negli anni precedenti."""
    duels = pd.read_csv("data/duels.csv")
    hist = fit_history(duels, HIST_YEARS)
    sc = []
    for y in HIST_YEARS:
        p = os.path.join(SEASONS_DIR, str(y), "data", "races.csv")
        if os.path.exists(p):
            r = pd.read_csv(p)
            r = r[r["session"] == "R"].copy()
            rc = {k: v for k, v in round_circuits_for(y).items()}
            r["circuit"] = r["round"].map(rc)
            sc.append(r[["circuit", "n_sc", "n_vsc"]])
    sc = pd.concat(sc) if sc else pd.DataFrame(columns=["circuit", "n_sc", "n_vsc"])
    sc_mean = sc.groupby("circuit")["n_sc"].mean()
    out = pd.DataFrame({"round": list(circuits), "circuit": list(circuits.values())})
    out["track_overtake"] = out["circuit"].map(hist["track"]).astype(float)  # NaN se pista senza storico
    out["track_sc_rate"] = out["circuit"].map(sc_mean)
    return out


def round_circuits_for(year):
    s = get("sessions", year=year)
    s = s[~s["is_cancelled"].fillna(False)]
    ms = s.groupby("meeting_key").agg(d=("date_start", "min"), c=("circuit_short_name", "first"),
                                      q=("session_name", lambda x: "Qualifying" in set(x)))
    ms = ms[ms["q"]].sort_values("d")
    return {i + 1: c for i, c in enumerate(ms["c"])}


def build():
    w = add_features(pd.read_csv("data/weekends.csv"))
    laps = pd.read_csv("data/laps.csv")
    rp, _ = race_pace(laps)
    df = w.merge(rp, on=["round", "driver"], how="left")

    # --- qualifica e partenza
    df["quali_gap"] = df["quali_gap_pct"]
    df["grid_penalty"] = df["grid"] - df["quali_pos"]
    df["quali_tm_gap"] = teammate_delta(df, "quali_gap")

    # --- libere e sprint, ogni sessione separata
    for code in ["FP1", "FP2", "FP3", "S"]:
        sp = session_pace(laps, code).rename(columns={f"pace_{code}": f"pace_{code}"})
        df = df.merge(sp, on=["round", "driver"], how="left")
    pp = practice_pace(laps).rename(columns={"fp_pace": "fp_pooled", "fp_laps": "longrun_laps"})
    df = df.merge(pp, on=["round", "driver"], how="left")
    df["longrun_laps"] = df["longrun_laps"].fillna(0)
    df["fp_tm_gap"] = teammate_delta(df, "fp_pooled")
    df = df.merge(best_practice_lap(laps), on=["round", "driver"], how="left")

    # --- forma del pilota (solo gare precedenti)
    df = df.sort_values(["driver", "round"])
    prev = lambda s, k=3: s.shift(1).rolling(k, min_periods=1).mean()
    df["drv_race_pace3"] = df.groupby("driver")["race_pace"].transform(prev)
    df["drv_points3"] = df["form_points3"]
    df["drv_avg_gain"] = df["avg_gain"]
    gains = lap1_gains_2026()
    df["drv_start_form"] = [driver_start_form(gains, r).get(d, 0.0) for r, d in zip(df["round"], df["driver"])]
    df["drv_dnf_rate"] = df["dnf_rate_drv"]

    # --- forma della scuderia (solo gare precedenti)
    team = df.groupby(["round", "team"])["race_pace"].mean().rename("trp").reset_index().sort_values("round")
    team["team_race_pace3"] = team.groupby("team")["trp"].transform(prev)
    df = df.merge(team[["round", "team", "team_race_pace3"]], on=["round", "team"], how="left")
    df["team_quali_gap3"] = df["team_quali_gap3"]
    df["team_points3"] = df["team_points3"]
    df["team_dnf_rate"] = df["dnf_rate_team"]

    # --- pista e condizioni
    circ = round_circuits()
    df = df.merge(track_features(circ), on="round", how="left")
    df = df.merge(weekend_degradation(laps), on="round", how="left")
    df = df.merge(wet_before_race(), on="round", how="left")
    df["sprint_weekend"] = df["sprint_weekend"]

    df["season"] = SEASON
    cols = ["season", "round", "event", "circuit", "driver", "team",
            # qualifica e partenza
            "quali_gap", "quali_pos", "grid", "grid_penalty", "quali_tm_gap",
            # libere e sprint
            "pace_FP1", "pace_FP2", "pace_FP3", "pace_S", "sprint_pos", "longrun_laps", "fp_tm_gap",
            "fp_best_lap_gap", "fp_pooled",
            # forma pilota
            "drv_race_pace3", "drv_points3", "drv_avg_gain", "drv_start_form", "drv_dnf_rate",
            # forma scuderia
            "team_race_pace3", "team_quali_gap3", "team_points3", "team_dnf_rate",
            # pista e condizioni
            "track_overtake", "track_sc_rate", "weekend_deg", "sprint_weekend", "wet_before_race",
            # obiettivi e info a posteriori (NON usabili come variabili)
            "race_pace", "finish_pos", "dnf", "points", "wet_race"]
    out = df[cols].sort_values(["round", "grid"])
    out.to_csv("data/features.csv", index=False)
    print(f"{SEASON}: {len(out)} righe, {out['round'].nunique()} gare, "
          f"passo gara disponibile per {out['race_pace'].notna().sum()} righe")
    return out


if __name__ == "__main__":
    build()
