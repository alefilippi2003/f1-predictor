"""Round 2026 -> pista OpenF1 (stesso ordinamento di data.py)."""
from data import SEASON, get


def round_circuits():
    s = get("sessions", year=SEASON)
    s = s[~s["is_cancelled"].fillna(False)]
    ms = s.groupby("meeting_key").agg(d=("date_start", "min"), c=("circuit_short_name", "first"),
                                      q=("session_name", lambda x: "Qualifying" in set(x)))
    ms = ms[ms["q"]].sort_values("d")
    return {i + 1: c for i, c in enumerate(ms["c"])}
