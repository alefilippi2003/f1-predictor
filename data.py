"""Scarica i dati 2026 da OpenF1 e costruisce una tabella pilota x gara."""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

API = "https://api.openf1.org/v1/"
CACHE = "cache/openf1"
OUT = "data/weekends.csv"
# stagione: 2026 di default; F1_SEASON=2024 per il backtest storico
SEASON = int(os.environ.get("F1_SEASON", 2026))

# Griglie di partenza ufficiali (con penalita') per gare non ancora corse
GRID_OVERRIDE = {
    # Austin: griglia effettiva in gara (Stroll partito 22°)
    18: ["ANT", "VER", "HAD", "LEC", "HAM", "RUS", "PIA", "NOR", "LAW", "GAS", "HUL", "ALO", "BEA",
         "LIN", "OCO", "SAI", "ALB", "PER", "BOT", "COL", "BOR", "STR"],
}


def get(endpoint, **params):
    """GET con cache su disco (i dati storici non cambiano) e rispetto del rate limit."""
    qs = urllib.parse.urlencode(params)
    path = os.path.join(CACHE, f"{endpoint}_{qs}.json".replace("&", "_").replace("=", "-"))
    if os.path.exists(path):
        with open(path) as f:
            return _clean(pd.DataFrame(json.load(f)))
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}{endpoint}?{qs}", timeout=60) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = []
                break
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    if isinstance(data, dict):  # {"detail": "No results found."}
        data = []
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    time.sleep(0.4)
    return _clean(pd.DataFrame(data))


def _clean(df):
    # alcune posizioni arrivano come stringhe ("RT", "DQ"...): diventano NaN
    if "position" in df:
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
    return df


def long_run_pace(session_keys):
    """Gap % del passo nei long run (stint >= 5 giri puliti), confrontando ogni pilota
    con il migliore sulla STESSA mescola; media fra sessioni e mescole."""
    gaps = []
    for sk in session_keys:
        laps, stints = get("laps", session_key=sk), get("stints", session_key=sk)
        if laps.empty or stints.empty:
            continue
        laps = laps[laps["lap_duration"].notna() & ~laps["is_pit_out_lap"].fillna(False)]
        rows = []
        for _, st in stints.iterrows():
            if pd.isna(st["lap_start"]) or pd.isna(st["lap_end"]):
                continue
            # escludo il giro di rientro (ultimo dello stint)
            l = laps[(laps["driver_number"] == st["driver_number"]) &
                     laps["lap_number"].between(st["lap_start"], st["lap_end"] - 1)]
            rows.append(l.assign(compound=st["compound"], stint=st["stint_number"]))
        if not rows:
            continue
        l = pd.concat(rows)
        # via giri lenti (traffico, bandiere, cool-down) rispetto alla mediana dello stint
        stint_med = l.groupby(["driver_number", "stint"])["lap_duration"].transform("median")
        l = l[l["lap_duration"] < stint_med * 1.02]
        # via i tentativi di giro secco: lo stint deve essere "lento" e costante
        runs = l.groupby(["driver_number", "compound", "stint"]).filter(
            lambda g: len(g) >= 4 and g["lap_duration"].std() < 1.5)
        if runs.empty:
            continue
        med = runs.groupby(["driver_number", "compound", "stint"])["lap_duration"].median() \
                  .groupby(["driver_number", "compound"]).min()
        best = med.groupby("compound").transform("min")
        g = ((med / best - 1) * 100)
        # mescola valida solo se almeno 3 piloti ci hanno fatto un long run
        n = g.groupby("compound").transform("size")
        gaps.append(g[n >= 3].groupby("driver_number").mean())
    if not gaps:
        return pd.Series(dtype=float)
    return pd.concat(gaps, axis=1).mean(axis=1)


def best_time(d):
    if isinstance(d, list):
        d = [x for x in d if x is not None]
        return min(d) if d else np.nan
    return d if d is not None else np.nan


def weekend_table(rnd, meeting_name, sessions):
    by_name = {s["session_name"]: s["session_key"] for _, s in sessions.iterrows()}
    if "Qualifying" not in by_name:
        return None
    sprint = "Sprint" in by_name
    print(f"R{rnd:02d} {meeting_name} ({'sprint' if sprint else 'classico'})", flush=True)

    q = get("session_result", session_key=by_name["Qualifying"])
    if q.empty:
        return None
    race_key = by_name.get("Race")
    drv = get("drivers", session_key=race_key or by_name["Qualifying"])
    drv = drv.drop_duplicates("driver_number").set_index("driver_number")

    # piloti senza tempo in qualifica (o assenti) partono in fondo
    r = get("session_result", session_key=race_key) if race_key else pd.DataFrame()
    extra = set(r["driver_number"]) - set(q["driver_number"]) if not r.empty else set()
    q = pd.concat([q, pd.DataFrame({"driver_number": sorted(extra)})], ignore_index=True)
    missing = q["position"].isna()
    q.loc[missing, "position"] = q["position"].max() + 1 + np.arange(missing.sum())
    q["t"] = q["duration"].map(best_time)
    df = pd.DataFrame({
        "round": rnd, "event": meeting_name,
        "driver_number": q["driver_number"].values,
        "quali_pos": q["position"].values,
        "quali_gap_pct": ((q["t"] / q["t"].min() - 1) * 100).values,
        "sprint_weekend": int(sprint),
    })
    df["driver"] = df["driver_number"].map(drv["name_acronym"])
    df["team"] = df["driver_number"].map(drv["team_name"])

    pace_keys = [by_name[n] for n in ("Practice 1", "Practice 2", "Practice 3", "Sprint") if n in by_name]
    df["longrun_gap_pct"] = df["driver_number"].map(long_run_pace(pace_keys))

    df["sprint_pos"] = np.nan
    if sprint:
        sp = get("session_result", session_key=by_name["Sprint"])
        if not sp.empty:
            df["sprint_pos"] = df["driver_number"].map(sp.set_index("driver_number")["position"])

    if not r.empty and r["position"].notna().any():
        r = r.set_index("driver_number")
        # griglia reale (con penalita') = prima posizione registrata in gara
        pos = get("position", session_key=race_key)
        grid = pos.sort_values("date").groupby("driver_number")["position"].first() if not pos.empty \
            else pd.Series(dtype=float)
        df["grid"] = df["driver_number"].map(grid).fillna(df["quali_pos"])
        df["dnf"] = df["driver_number"].map(r["dnf"] | r["dns"] | r["dsq"]).astype(float)
        # i ritirati hanno position nulla: li metto in fondo nell'ordine dei giri completati
        order = r.assign(p=r["position"].fillna(100 - r["number_of_laps"].fillna(0))) \
                 .sort_values("p").reset_index()
        order["finish_pos"] = np.arange(1, len(order) + 1)
        df["finish_pos"] = df["driver_number"].map(order.set_index("driver_number")["finish_pos"])
        df["points"] = df["driver_number"].map(r["points"]).fillna(0)
        w = get("weather", session_key=race_key)
        df["wet_race"] = int((w["rainfall"] > 0).any()) if not w.empty else np.nan
    else:
        # gara non ancora corsa: griglia ufficiale con penalita' se nota, altrimenti la qualifica
        order = GRID_OVERRIDE.get(rnd) if SEASON == 2026 else None
        df["grid"] = df["driver"].map({d: i + 1 for i, d in enumerate(order)}) if order else df["quali_pos"]
        df["finish_pos"] = df["dnf"] = df["points"] = df["wet_race"] = np.nan
    return df


def main(max_round):
    sessions = get("sessions", year=SEASON)
    sessions = sessions[~sessions["is_cancelled"].fillna(False)]
    sessions = sessions[sessions["session_name"] != "Day 1"]  # niente test
    meetings = sessions.groupby("meeting_key")["date_start"].min().sort_values()
    meetings = [m for m in meetings.index
                if "Qualifying" in set(sessions.loc[sessions["meeting_key"] == m, "session_name"])]
    names = get("meetings", year=SEASON).set_index("meeting_key")["meeting_name"]
    frames = []
    for rnd, mk in enumerate(meetings[:max_round], start=1):
        t = weekend_table(rnd, names.get(mk, str(mk)), sessions[sessions["meeting_key"] == mk])
        if t is not None:
            frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nSalvato {OUT}: {len(out)} righe, {out['round'].nunique()} weekend")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
