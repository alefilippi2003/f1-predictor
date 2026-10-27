"""Costruisce il dataset giro per giro 2026 (libere, qualifiche, sprint, gara) da OpenF1.

Output:
  data/laps.csv    una riga per giro
  data/races.csv   una riga per gara/sprint: giri, safety car, pit loss
"""
import os
import sys

import numpy as np
import pandas as pd

from data import SEASON, get

SESSION_CODE = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Sprint Qualifying": "SQ", "Sprint": "S", "Qualifying": "Q", "Race": "R",
}


def neutralized_laps(sk):
    """Giri sotto safety car / VSC / bandiera rossa secondo la direzione gara."""
    rc = get("race_control", session_key=sk)
    if rc.empty:
        return set(), 0, 0
    rc = rc.sort_values("date")
    out, start, n_sc, n_vsc = set(), None, 0, 0
    for _, m in rc.iterrows():
        msg = str(m["message"]).upper()
        lap = m["lap_number"]
        if pd.isna(lap):
            continue
        lap = int(lap)
        if m["category"] == "SafetyCar" and "DEPLOYED" in msg:
            start = lap
            n_vsc += "VSC" in msg or "VIRTUAL" in msg
            n_sc += not ("VSC" in msg or "VIRTUAL" in msg)
        elif m["category"] == "SafetyCar" and ("ENDING" in msg or "IN THIS LAP" in msg) and start:
            out.update(range(start, lap + 2))  # +1: giro di ripartenza ancora lento
            start = None
        elif m["flag"] == "RED":
            out.update(range(lap, lap + 2))
    if start:
        out.update(range(start, start + 4))
    return out, n_sc, n_vsc


def session_laps(rnd, event, code, sk, drivers):
    laps, stints = get("laps", session_key=sk), get("stints", session_key=sk)
    if laps.empty:
        return None, None
    laps = laps[["driver_number", "lap_number", "date_start", "lap_duration", "is_pit_out_lap",
                 "duration_sector_1", "duration_sector_2", "duration_sector_3", "st_speed"]].copy()

    # mescola, stint ed eta' gomma per ogni giro
    laps["compound"], laps["stint"], laps["tyre_age"] = None, np.nan, np.nan
    for _, st in stints.iterrows():
        if pd.isna(st["lap_start"]) or pd.isna(st["lap_end"]):
            continue
        m = (laps["driver_number"] == st["driver_number"]) & \
            laps["lap_number"].between(st["lap_start"], st["lap_end"])
        laps.loc[m, "compound"] = st["compound"]
        laps.loc[m, "stint"] = st["stint_number"]
        laps.loc[m, "tyre_age"] = (st["tyre_age_at_start"] or 0) + laps.loc[m, "lap_number"] - st["lap_start"]

    # giro di rientro ai box
    laps["is_pit_in_lap"] = False
    if code in ("R", "S"):
        pit = get("pit", session_key=sk)
        if not pit.empty:
            key = set(zip(pit["driver_number"], pit["lap_number"]))
            laps["is_pit_in_lap"] = [(d, l) in key for d, l in zip(laps["driver_number"], laps["lap_number"])]
    else:
        last = laps.groupby(["driver_number", "stint"])["lap_number"].transform("max")
        laps["is_pit_in_lap"] = laps["lap_number"] == last

    neutral, n_sc, n_vsc = neutralized_laps(sk) if code in ("R", "S") else (set(), 0, 0)
    laps["neutralized"] = laps["lap_number"].isin(neutral)

    laps.insert(0, "session", code)
    laps.insert(0, "event", event)
    laps.insert(0, "round", rnd)
    laps["driver"] = laps["driver_number"].map(drivers["name_acronym"])
    laps["team"] = laps["driver_number"].map(drivers["team_name"])

    info = None
    if code in ("R", "S"):
        pit = get("pit", session_key=sk)
        info = {"round": rnd, "event": event, "session": code,
                "total_laps": int(laps["lap_number"].max()),
                "n_sc": n_sc, "n_vsc": n_vsc, "neutralized_laps": len(neutral),
                "pit_lane_s": float(pit["lane_duration"].median()) if not pit.empty and "lane_duration" in pit
                else np.nan}
    return laps, info


def main(max_round):
    sessions = get("sessions", year=SEASON)
    sessions = sessions[~sessions["is_cancelled"].fillna(False)]
    names = get("meetings", year=SEASON).set_index("meeting_key")["meeting_name"]
    meetings = sessions.groupby("meeting_key")["date_start"].min().sort_values()
    meetings = [m for m in meetings.index
                if "Qualifying" in set(sessions.loc[sessions["meeting_key"] == m, "session_name"])]

    all_laps, all_info = [], []
    for rnd, mk in enumerate(meetings[:max_round], start=1):
        ss = sessions[sessions["meeting_key"] == mk]
        keys = dict(zip(ss["session_name"], ss["session_key"]))
        ref = keys.get("Race") or keys["Qualifying"]
        drivers = get("drivers", session_key=ref).drop_duplicates("driver_number").set_index("driver_number")
        print(f"R{rnd:02d} {names.get(mk, mk)}", flush=True)
        for name, code in SESSION_CODE.items():
            if name not in keys:
                continue
            laps, info = session_laps(rnd, names.get(mk, str(mk)), code, keys[name], drivers)
            if laps is not None:
                all_laps.append(laps)
            if info:
                all_info.append(info)

    laps = pd.concat(all_laps, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    laps.to_csv("data/laps.csv", index=False)
    pd.DataFrame(all_info).to_csv("data/races.csv", index=False)
    print(f"\n{len(laps)} giri salvati in data/laps.csv")
    print(laps.groupby("session").size().to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
