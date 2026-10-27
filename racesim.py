"""Livello 3: simulatore di gara giro per giro + backtest walk-forward.

Uso: python racesim.py [round_da_prevedere]
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import HuberRegressor

from data import SEASON, get
from evaluate import expected_position, race_metrics
from model import add_features
from overtakes import AGE_EDGES, DUEL_GAP_S, age_curve, fit_2026, fit_history, pass_logit
from pace import pace_table
from racesim_rounds import round_circuits
from starts import (calibrate_start, driver_start_form, lap1_gains_2026, positions_per_second,
                    start_sd_table, target_shuffle)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(7)

# Variante A: il passo gara si stima soprattutto dalla qualifica.
# Variante B: dalla FP3 (weekend classico) o dalla sprint (weekend sprint); qualifica esclusa.
PACE_FEATURES_A = ["quali_pace", "fp_pace_f", "has_fp", "team_rp3", "drv_rp3"]
PACE_FEATURES_B = ["main_pace_f", "has_main", "fp_pace_f", "has_fp", "team_rp3", "drv_rp3"]
# Variante C: come B ma la qualifica resta, con peso minore (MAIN_WEIGHT alla sessione principale)
PACE_FEATURES_C = ["blend_pace", "has_main", "team_rp3", "drv_rp3"]
# Variante D: come A, ma la griglia effettiva (penalita' incluse) al posto del distacco in qualifica
PACE_FEATURES_D = ["grid", "fp_pace_f", "has_fp", "team_rp3", "drv_rp3"]
MAIN_WEIGHT = 0.6
PACE_FEATURES = PACE_FEATURES_A
MAIN_FP = "FP3"
GRID_GAP_S = 0.25      # distacco fra due caselle in griglia (s)
MIN_GAP_S = 0.3        # distanza minima dietro a un'altra macchina (aria sporca)
SC_GAP_S = 0.6         # distacco fra le macchine alla ripartenza dopo una safety car
LAP_NOISE_PCT = 0.25   # variabilita' giro-giro del singolo pilota
DEFAULT_LAPS = 56
BACKTEST_SIMS = 10000
CHUNK = 50000
TEAM_CUTS = (3, 6, 10)  # entrambe a podio, entrambe nelle prime 6, entrambe a punti
FP_MAX_DEV = 2.0       # % massima credibile fra passo nelle libere e qualifica

# Strategie previste (anteprima Pirelli), in frazione di gara. Solo per la previsione,
# non per il backtest: per le gare passate non le abbiamo.
STRATEGY = {
    18: {  # Austin, 56 giri: 1 sosta M-H con pit giri 18-24; alternativa 2 soste M-H-M
        "windows": ((18 / 56, 24 / 56), (14 / 56, 20 / 56), (34 / 56, 42 / 56)),
        "p_two_stop": 0.25,
    },
}


# ---------------------------------------------------------------- livello 2
def add_pace_features(t):
    t = t.sort_values("round").copy()
    # senza tempo in qualifica: come il piu' lento del weekend
    t["quali_pace"] = t["quali_pace"].fillna(t.groupby("round")["quali_pace"].transform("max"))
    prev3 = lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    t["drv_rp3"] = t.groupby("driver")["race_pace"].transform(prev3)
    team = t.groupby(["round", "team"])["race_pace"].mean().rename("trp").reset_index()
    team["team_rp3"] = team.groupby("team")["trp"].transform(prev3)
    t = t.merge(team[["round", "team", "team_rp3"]], on=["round", "team"], how="left")
    for c in ["drv_rp3", "team_rp3"]:
        t[c] = t[c].fillna(t["team_rp3"]).fillna(t.groupby("round")[c].transform("median")).fillna(1.0)

    # riferimento per scartare i passi fuori scala: A la qualifica, B il passo gara recente
    # della scuderia (cosi' in B la qualifica non entra affatto nella stima del passo)
    ref = t["quali_pace"] if PACE_FEATURES in (PACE_FEATURES_A, PACE_FEATURES_D) else t["team_rp3"]
    # > 2% dal riferimento = errore di misura (bandiere rosse, traffico, programmi di prova)
    for c in ["fp_pace", "pace_FP2", "pace_FP3", "pace_S"]:
        t.loc[(t[c] - ref).abs() > FP_MAX_DEV, c] = np.nan
    # passo mancante: uso il riferimento come proxy, e lo segnalo
    t["has_fp"] = t["fp_pace"].notna().astype(float)
    t["fp_pace_f"] = t["fp_pace"].fillna(ref)
    # sessione principale: sprint nei weekend sprint, FP3 (variante B) o FP2 (B2) in quelli classici
    t["main_pace"] = t["pace_S"].where(t["pace_S"].notna(), t[f"pace_{MAIN_FP}"])
    t["has_main"] = t["main_pace"].notna().astype(float)
    t["main_pace_f"] = t["main_pace"].fillna(t["fp_pace_f"])
    t["blend_pace"] = MAIN_WEIGHT * t["main_pace_f"] + (1 - MAIN_WEIGHT) * t["quali_pace"]
    return t.sort_values(["round", "driver"]).reset_index(drop=True)


# Modello del passo appreso su tutte le variabili (learn.py). None = modello A classico.
PACE_MODEL = None          # "Lasso (tutte)" o "LightGBM (tutte)"
PACE_POOL = "tutte"        # "tutte" = stagioni precedenti + gare precedenti, "stagione" = solo stagione
_FEATS = None


def ml_pace(rnd, drivers):
    """Passo previsto dal modello appreso, allenato solo su gare precedenti a (SEASON, rnd)."""
    global _FEATS
    import learn
    if _FEATS is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        os.chdir(here)
        try:
            _FEATS = learn.load()
        finally:
            os.chdir(cwd)
    df = _FEATS
    key = SEASON * 100 + rnd
    pool = df[(df["key"] < key) & df["race_pace"].notna()]
    if PACE_POOL == "stagione":
        pool = pool[pool["season"] == SEASON]
    feats = learn.MODELS[PACE_MODEL][0]
    m = learn.fit(PACE_MODEL, pool, SEASON)
    y = pool[learn.TARGET.get(PACE_MODEL, "race_pace")]
    resid = y - m.predict(pool[feats])
    sigma = float(np.median(np.abs(resid - resid.median())) * 1.4826)
    cur = df[df["key"] == key].set_index("driver")
    pace = m.predict(cur.reindex(drivers.values)[feats])
    return m, sigma, pace


def fit_pace_model(train):
    tr = train.dropna(subset=["race_pace"])
    m = HuberRegressor(alpha=1e-2, max_iter=1000).fit(tr[PACE_FEATURES], tr["race_pace"])
    resid = tr["race_pace"] - m.predict(tr[PACE_FEATURES])
    sigma = float(np.median(np.abs(resid - resid.median())) * 1.4826)
    return m, sigma


# ---------------------------------------------------------------- livello 3
def season_params(track, races, stints, upto):
    """Parametri di gara stimati SOLO dalle gare precedenti a `upto`."""
    tr = track[track["round"] < upto]
    deg = {c: float(np.clip(tr[f"deg_{c}"].median(), 0.0, 0.3)) for c in ["SOFT", "MEDIUM", "HARD"]}
    rc = races[(races["round"] < upto) & (races["session"] == "R")]
    st = stints[stints["round"] < upto]
    # dati mancanti (es. tempi pit lane assenti a inizio 2023): valori tipici, mai NaN
    or_default = lambda x, d: d if pd.isna(x) else float(x)
    return {
        "deg": {c: or_default(v, 0.05) for c, v in deg.items()},
        "sc_rate": or_default(rc["n_sc"].mean(), 0.8),
        "pit_loss": or_default(rc["pit_lane_s"].clip(15, 35).median(), 22.0),
        "p_two_stop": or_default((st["stops"] >= 2).mean(), 0.3),
    }


def race_strategies(laps):
    """Numero di soste dei piloti arrivati al traguardo in ogni gara passata."""
    r = laps[(laps["session"] == "R") & laps["stint"].notna()]
    s = r.groupby(["round", "driver"])["stint"].max().rename("stints").reset_index()
    s["stops"] = s["stints"] - 1
    return s


def simulate_race(field, ref_lap, n_laps, sp, sigma, p_pass, n=4000):
    """field: DataFrame con grid, pace (% vs migliore), p_dnf. Ritorna posizioni finali (n x D)."""
    D = len(field)
    ar = np.arange(n)
    pace = field["pace"].values[None, :] + RNG.normal(0, sigma, (n, D))  # giornata buona/cattiva

    # strategia: 1 o 2 soste. Finestre (frazione di gara) generiche, o quelle passate in sp
    # (es. anteprima Pirelli). Partenza su media, poi dura; con 2 soste ultimo stint su media.
    w1, w2a, w2b = sp.get("windows", ((0.30, 0.60), (0.22, 0.40), (0.55, 0.75)))
    two = RNG.random((n, D)) < sp["p_two_stop"]
    stop1 = np.where(two, RNG.uniform(*w2a, (n, D)), RNG.uniform(*w1, (n, D)))
    stop2 = np.where(two, RNG.uniform(*w2b, (n, D)), 2.0)
    stop1, stop2 = (stop1 * n_laps).astype(int), (stop2 * n_laps).astype(int)
    deg_m, deg_h = sp["deg"]["MEDIUM"], sp["deg"]["HARD"]

    # ritiri: probabilita' per pilota, giro casuale
    dnf = RNG.random((n, D)) < field["p_dnf"].values[None, :]
    dnf_lap = np.where(dnf, RNG.integers(1, n_laps + 1, (n, D)), n_laps + 10)

    # safety car: numero ~ Poisson, giro casuale
    n_sc = RNG.poisson(sp["sc_rate"], n)
    sc_laps = [set(RNG.integers(2, n_laps - 2, k)) for k in n_sc]
    sc_mask = np.zeros((n, n_laps + 1), bool)
    for i, s in enumerate(sc_laps):
        sc_mask[i, list(s)] = True

    grid = field["grid"].values
    cum = np.tile((grid - 1) * GRID_GAP_S, (n, 1)).astype(float)
    # caos della partenza: davanti si cambia meno posizione che in mezzo al gruppo
    sd_start = sp["start_sd_a"] + sp["start_sd_b"] * (grid - 1) if "start_sd_a" in sp \
        else np.full(D, sp.get("start_sd", 0.8))
    cum += RNG.normal(0, 1, (n, D)) * sd_start[None, :]
    cum -= sp.get("start_bias_s", np.zeros(D))[None, :]     # chi parte bene guadagna terreno
    age = np.zeros((n, D))
    comp_deg = np.full((n, D), deg_m)
    duel_age = np.zeros((n, D), int)
    best = np.full((n, D), np.inf)
    prev_ahead = np.full((n, D), -1)
    if "age_curve" not in sp:
        sp = {**sp, "age_curve": np.zeros(len(AGE_EDGES) + 1)}

    for lap in range(1, n_laps + 1):
        out = dnf_lap <= lap
        pit = (lap == stop1) | (lap == stop2)
        t_exp = ref_lap * (1 + (pace + comp_deg * age) / 100)  # passo atteso, senza rumore
        t = t_exp + ref_lap * RNG.normal(0, LAP_NOISE_PCT, (n, D)) / 100
        t = t + np.where(pit, sp["pit_loss"], 0.0)
        new = cum + t
        new[out] = np.inf

        # sorpassi: elaboro le macchine nell'ordine di inizio giro. Chi e' a meno di 1 s
        # da quello davanti tenta il sorpasso con probabilita' che dipende dalla pista e dal
        # vantaggio di passo atteso; se non passa (o arriva da piu' lontano) resta dietro.
        order = np.argsort(cum + np.where(out, np.inf, 0), axis=1)
        for r in range(1, D):
            idx, ahead = order[:, r], order[:, r - 1]
            cand, a = new[ar, idx], new[ar, ahead]
            # chi si ferma ai box non lotta: niente blocco se uno dei due entra in pit
            fight = ~pit[ar, idx] & ~pit[ar, ahead] & np.isfinite(cand)
            if "pass_logit" in sp:
                close = (cum[ar, idx] - cum[ar, ahead]) < DUEL_GAP_S
                # giri consecutivi in scia alla stessa macchina: piu' sono, piu' e' difficile
                same = close & (prev_ahead[ar, idx] == ahead)
                duel_age[ar, idx] = np.where(same, duel_age[ar, idx] + 1, np.where(close, 1, 0))
                prev_ahead[ar, idx] = np.where(close, ahead, -1)
                g = sp["age_curve"][np.searchsorted(AGE_EDGES, duel_age[ar, idx], side="left")]
                delta = t_exp[ar, ahead] - t_exp[ar, idx]
                p = expit(sp["pass_logit"] + sp["pass_slope"] * delta + g)
                passes = close & (RNG.random(n) < p)
                new[ar, idx] = np.where(fight & passes, np.minimum(cand, a - 0.1), cand)
                blocked = fight & ~passes & (new[ar, idx] < a + MIN_GAP_S)
            else:
                blocked = fight & (cand < a + MIN_GAP_S) & (RNG.random(n) >= p_pass)
            new[ar, idx] = np.where(blocked, a + MIN_GAP_S, new[ar, idx])

        # giro piu' veloce: tempo reale del giro (dopo eventuale blocco), esclusi giro 1,
        # giri ai box, safety car e piloti ritirati
        valid = ~out & ~pit & ~sc_mask[:, lap][:, None] & (lap > 1)
        best = np.where(valid, np.minimum(best, new - cum), best)

        # safety car: il gruppo si ricompatta nell'ordine attuale
        sc = sc_mask[:, lap]
        if sc.any():
            o = np.argsort(new[sc], axis=1)
            lead = np.take_along_axis(new[sc], o[:, :1], axis=1)
            packed = lead + np.arange(D)[None, :] * SC_GAP_S
            fin = np.isfinite(np.take_along_axis(new[sc], o, axis=1))
            vals = np.where(fin, packed, np.inf)
            tmp = np.empty_like(vals)
            np.put_along_axis(tmp, o, vals, axis=1)
            new[sc] = tmp

        cum = new
        if "trace" in sp:  # diagnostica: chi e' in testa a fine giro
            sp["trace"].append(np.argmin(cum, axis=1))
        age += 1
        age[pit] = 0
        comp_deg = np.where(lap == stop1, deg_h, np.where(lap == stop2, deg_m, comp_deg))

    # classifica finale: chi e' arrivato per tempo, i ritirati dietro (ritiro piu' tardi = davanti)
    key = np.where(np.isfinite(cum), cum, 1e9 - dnf_lap)
    order = np.argsort(key, axis=1)
    pos = np.empty_like(order)
    pos[ar[:, None], order] = np.arange(1, D + 1)[None, :]
    sp["_extra"] = (dnf_lap, best)
    return pos, dnf


# ---------------------------------------------------------------- orchestrazione
def overtaking_model():
    duels = pd.read_csv("data/duels.csv")
    ot = {"duels": duels, "hist": fit_history(duels), "circuits": round_circuits(), "age_cache": {}}
    if os.path.exists("data/starts.csv"):
        ot["starts"] = pd.read_csv("data/starts.csv")
        ot["sd_table"] = start_sd_table()
        ot["gains"] = lap1_gains_2026()
        ot["start_cal"] = {}
    return ot


def load():
    laps = pd.read_csv("data/laps.csv")
    t, track = pace_table(laps)
    w = add_features(pd.read_csv("data/weekends.csv"))[["round", "driver", "p_dnf"]]
    t = add_pace_features(t).merge(w, on=["round", "driver"], how="left")
    races = pd.read_csv("data/races.csv")
    return t, track, races, race_strategies(laps), laps


def best_quali_lap(laps, rnd):
    """Giro piu' veloce in qualifica; se OpenF1 non ha i giri (es. Baku 2025) quello delle libere,
    scalato di un ~1% medio fra libere e qualifica."""
    l = laps[laps["round"] == rnd]
    q = l.loc[l["session"] == "Q", "lap_duration"].min()
    if pd.isna(q):
        q = l.loc[l["session"].isin(["FP1", "FP2", "FP3", "SQ"]), "lap_duration"].min() * 0.99
    return q


def predict_round(rnd, t, track, races, stints, laps, p_pass, n=4000, strategy=None, sigma_scale=1.0,
                  ot=None, save_sims=None):
    # stessi numeri casuali per ogni gara: le calibrazioni si confrontano a parita' di "fortuna"
    global RNG
    RNG = np.random.default_rng(1000 + rnd)
    train, field = t[t["round"] < rnd], t[t["round"] == rnd].copy()
    if PACE_MODEL:
        m, sigma, pace = ml_pace(rnd, field["driver"])
        field["pace"] = pace
    else:
        m, sigma = fit_pace_model(train)
        field["pace"] = m.predict(field[PACE_FEATURES])
    # il residuo del modello include l'errore di stima del passo "vero": quanta di questa
    # incertezza si traduce in gara lo calibra il backtest
    sigma *= sigma_scale
    field["pace"] -= field["pace"].min()
    sp = season_params(track, races, stints, rnd)
    rr = races[(races["round"] == rnd) & (races["session"] == "R")]
    n_laps = int(rr["total_laps"].iat[0]) if len(rr) else DEFAULT_LAPS
    if strategy:
        sp.update(strategy)
    if ot is not None:
        # difficolta' della pista dallo storico; livello e pendenza 2026 dalle sole gare precedenti
        circ = ot["circuits"][rnd]
        past = [ot["circuits"][r] for r in range(1, rnd)]
        y26 = fit_2026(ot["duels"], ot["hist"], past)
        sp["pass_logit"] = pass_logit(circ, ot["hist"], y26)
        sp["pass_slope"] = y26["slope"]
        sp["circuit"] = circ
        if "starts" in ot:
            # caos di partenza: rimescolamento atteso al giro 1 su questa pista -> rumore al via
            target = target_shuffle(ot["starts"], circ, past)
            sp["start_sd"] = float(np.interp(target, ot["sd_table"][1], ot["sd_table"][0]))
            sp["start_shuffle"] = target
            if rnd not in ot["start_cal"]:
                ot["start_cal"][rnd] = calibrate_start(ot["gains"], rnd)
            sp["start_sd_a"], sp["start_sd_b"], sp["start_targets"] = ot["start_cal"][rnd]
            # forma in partenza del pilota: posizioni guadagnate in piu' della media -> vantaggio al via
            form = driver_start_form(ot["gains"], rnd)
            pps = positions_per_second(sp["start_sd_a"] + sp["start_sd_b"] * 10)
            sp["start_bias_s"] = field["driver"].map(form).fillna(0.0).values / pps
            sp["start_form"] = field["driver"].map(form).fillna(0.0).values
        if ot.get("use_age", True):
            if circ not in ot["age_cache"]:
                ot["age_cache"][circ] = age_curve(ot["duels"], ot["hist"], circ)
            sp["age_curve"] = ot["age_cache"][circ]
    # giro di riferimento: pole in qualifica x rapporto medio gara/qualifica delle gare passate
    q = best_quali_lap(laps, rnd)
    ratios = []
    for r0 in track[track["round"] < rnd]["round"]:
        q0 = best_quali_lap(laps, r0)
        ratios.append(track.loc[track["round"] == r0, "race_ref_lap"].iat[0] / q0)
    ratios = [r for r in ratios if np.isfinite(r)]
    ref = q * (np.median(ratios) if ratios else 1.08)
    # a blocchi, per non esaurire la memoria con milioni di simulazioni
    D = len(field)
    counts, dnf_count, done = np.zeros((D, D)), np.zeros(D), 0
    # eventi di scuderia (entrambe le macchine nelle prime k), contati nella STESSA gara simulata:
    # i compagni di squadra sono correlati, il prodotto delle probabilita' singole sarebbe sbagliato
    teams = field["team"].values
    pairs = {tm: np.where(teams == tm)[0] for tm in pd.unique(teams) if (teams == tm).sum() == 2}
    team_counts = {tm: np.zeros(len(TEAM_CUTS)) for tm in pairs}
    sims = {"order": [], "dnf": [], "dnf_lap": [], "fastest": []}
    while done < n:
        k = min(CHUNK, n - done)
        pos, dnf = simulate_race(field, ref, n_laps, sp, sigma, p_pass, n=k)
        # posizione d'arrivo solo per chi finisce (i ritirati sono gia' in fondo all'ordine,
        # quindi chi arriva occupa le posizioni 1..arrivati); i ritiri a parte
        for j in range(D):
            counts[j] += np.bincount(pos[~dnf[:, j], j] - 1, minlength=D)
        dnf_count += dnf.sum(0)
        if save_sims:
            dnf_lap, best = sp.pop("_extra")
            sims["order"].append(np.argsort(pos, axis=1).astype(np.int8))  # indice pilota per posizione
            sims["dnf"].append(dnf)
            sims["dnf_lap"].append(np.where(dnf, dnf_lap, 0).astype(np.int16))
            sims["fastest"].append(np.argmin(best, axis=1).astype(np.int8))
        for tm, (a, b) in pairs.items():
            ok = ~dnf[:, a] & ~dnf[:, b]
            worst = np.maximum(pos[:, a], pos[:, b])
            team_counts[tm] += [(ok & (worst <= c)).sum() for c in TEAM_CUTS]
        done += k
    sp["team_events"] = {tm: c / n for tm, c in team_counts.items()}
    if save_sims:
        np.savez_compressed(save_sims, drivers=field["driver"].values.astype(str),
                            teams=field["team"].values.astype(str), grid=field["grid"].values,
                            **{k: np.concatenate(v) for k, v in sims.items()})
    # prob[pilota, posizione-1], ultima colonna = ritirato
    prob = np.column_stack([counts / n, dnf_count / n])
    field["P(vittoria)"] = prob[:, 0]
    field["P(podio)"] = prob[:, :3].sum(1)
    field["P(punti)"] = prob[:, :10].sum(1)
    field["P(ritiro)"] = prob[:, -1]
    field["pos_mediana"] = (prob.cumsum(1) >= 0.5).argmax(1) + 1
    sp["pos_matrix"] = prob
    return field, sigma, sp, m


def backtest(t, track, races, stints, laps, p_pass, sigma_scale, ot=None, first=6, last=None):
    # fino all'ultima gara con risultato disponibile
    last = last or int(t.loc[t["finish_pos"].notna(), "round"].max())
    rows = []
    for rnd in range(first, last + 1):
        f, _, sp, _ = predict_round(rnd, t, track, races, stints, laps, p_pass, n=BACKTEST_SIMS,
                                    sigma_scale=sigma_scale, ot=ot)
        win = f.loc[f["finish_pos"].idxmin(), "driver"]
        pod = set(f.nsmallest(3, "finish_pos")["driver"])
        top = f.sort_values("P(vittoria)", ascending=False)["driver"].tolist()
        exp_pod = set(f.nlargest(3, "P(podio)")["driver"])
        p = f.set_index("driver")["P(vittoria)"].clip(lower=1 / BACKTEST_SIMS)
        mt = race_metrics(expected_position(sp["pos_matrix"], f["driver"].tolist()), f)
        rows.append({"round": rnd, "event": f["event"].iat[0], "vincitore": win, "favorito": top[0],
                     "p_vincitore": p[win], "hit": top[0] == win, "podio": len(pod & exp_pod),
                     **{f"m_{k}": v for k, v in mt.items()}})
    return pd.DataFrame(rows)


def position_matrix_csv(prob, drivers, path):
    """Righe = posizione d'arrivo 1..D + 'Ritirato', colonne = piloti, cella = probabilita'."""
    D = len(drivers)
    idx = pd.Index([str(i) for i in range(1, D + 1)] + ["Ritirato"], name="posizione")
    m = pd.DataFrame(prob.T, index=idx, columns=drivers)
    m = m[m.columns[np.argsort((prob * np.arange(1, D + 2)).sum(1))]]  # per posizione media (ritiro = D+1)
    m.to_csv(path, float_format="%.6f")
    return m


def fair_odds(p):
    """Quota 'equa' (senza margine del bookmaker): 1 / probabilita'."""
    return np.where(p > 0, 1 / np.maximum(p, 1e-12), np.inf)


def export_events(sp, f, target, tag=""):
    prob = sp["pos_matrix"]
    drv = pd.DataFrame({
        "pilota": f["driver"].values, "scuderia": f["team"].values, "griglia": f["grid"].values,
        "vincente": prob[:, 0], "primi 2": prob[:, :2].sum(1), "podio": prob[:, :3].sum(1),
        "primi 6": prob[:, :6].sum(1), "punti": prob[:, :10].sum(1), "si ritira": prob[:, -1],
    }).sort_values("vincente", ascending=False)
    ev = ["vincente", "primi 2", "podio", "primi 6", "punti", "si ritira"]
    for e in ev:
        drv[f"quota {e}"] = fair_odds(drv[e].values)
    drv.to_csv(f"data/eventi_piloti_r{target}{tag}.csv", index=False, float_format="%.4f")

    team = pd.DataFrame([{"scuderia": tm, "entrambe a podio": c[0], "entrambe prime 6": c[1],
                          "entrambe a punti": c[2]} for tm, c in sp["team_events"].items()])
    team = team.sort_values("entrambe a punti", ascending=False)
    for e in ["entrambe a podio", "entrambe prime 6", "entrambe a punti"]:
        team[f"quota {e}"] = fair_odds(team[e].values)
    team.to_csv(f"data/eventi_scuderie_r{target}{tag}.csv", index=False, float_format="%.4f")


def main():
    """python racesim.py [round] [--sims N] [--skip-backtest] [--variant ML_lgbm_t_w3|A|...]"""
    args = sys.argv[1:]
    n_sims = int(args[args.index("--sims") + 1]) if "--sims" in args else 100000
    skip = "--skip-backtest" in args
    global PACE_FEATURES, MAIN_FP, PACE_MODEL, PACE_POOL
    # modello scelto dopo il grande backtest (68 gare 2023-26): LightGBM regolato su tutte le stagioni, stagione in corso x3
    variant = args[args.index("--variant") + 1] if "--variant" in args else "ML_lgbm_t_w3"
    # varianti col modello appreso: ML_lasso, ML_lgbm (+ suffisso _stag per allenare solo sulla stagione)
    if variant.startswith("ML_"):
        PACE_MODEL = "Lasso (tutte)" if "lasso" in variant else {
            "ML_lgbm_rel": "LightGBM (relativo)", "ML_lgbm_w3": "LightGBM (tutte, stagione x3)",
            "ML_lgbm_rel_w3": "LightGBM (relativo, stagione x3)", "ML_lgbm_t": "LightGBM regolato",
            "ML_lgbm_t_w3": "LightGBM regolato (stagione x3)"}.get(variant, "LightGBM (tutte)")
        PACE_POOL = "stagione" if variant.endswith("_stag") else "tutte"
        variant_feats = "A"
    else:
        variant_feats = variant
    PACE_FEATURES = {"A": PACE_FEATURES_A, "B": PACE_FEATURES_B, "B2": PACE_FEATURES_B,
                     "C": PACE_FEATURES_C, "C2": PACE_FEATURES_C, "D": PACE_FEATURES_D}[variant_feats]
    MAIN_FP = "FP2" if variant in ("B2", "C2") else "FP3"
    tag = "" if variant == "A" else f"_{variant}"
    sigmas = [float(x) for x in args[args.index("--sigmas") + 1].split(",")] if "--sigmas" in args \
        else [0.15, 0.3, 0.5, 0.75, 1.0]
    pos_args = [a for i, a in enumerate(args)
                if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--sims", "--variant", "--sigmas"))]
    t, track, races, stints, laps = load()
    target = int(pos_args[0]) if pos_args else int(t[t["finish_pos"].isna()]["round"].min())
    pd.set_option("display.width", 220)

    # sorpassi e partenze per pista (storico + livello 2026); si calibra solo l'incertezza del passo
    ot = overtaking_model()
    if skip:
        ss = json.load(open(f"data/sim_params{tag}.json"))["sigma_scale"]
    else:
        print("=== BACKTEST simulatore giro per giro (gare 6-17, solo dati precedenti) ===")
        best = None
        for ss in sigmas:
            ev = backtest(t, track, races, stints, laps, None, ss, ot=ot)
            ll = -np.log(ev["p_vincitore"]).mean()
            print(f"incertezza x{ss:.2f}  vincitore {ev['hit'].sum()}/{len(ev)}  podio {ev['podio'].mean():.2f}"
                  f"  P media vincitore {ev['p_vincitore'].mean():.1%}  log-loss {ll:.3f}"
                  f"  | errore pos {ev['m_errore_pos'].mean():.2f}  spearman {ev['m_spearman'].mean():.3f}"
                  f"  punti {ev['m_punti'].mean():.2f}/10", flush=True)
            if best is None or ll < best[0]:
                best = (ll, ss, ev)
        ll, ss, ev = best
        print(f"\nScelto incertezza x{ss}")
        print(ev.to_string(index=False))
        ev.to_csv(f"data/backtest_sim{tag}.csv", index=False)
        json.dump({"sigma_scale": ss, "log_loss": ll}, open(f"data/sim_params{tag}.json", "w"))
        if "--backtest-only" in args:
            return

    f, sigma, sp, m = predict_round(target, t, track, races, stints, laps, None, n=n_sims,
                                    strategy=STRATEGY.get(target) if SEASON == 2026 else None,
                                    sigma_scale=ss, ot=ot,
                                    save_sims=f"data/sims_r{target}{tag}.npz")
    print(f"\n=== PREVISIONE R{target} {f['event'].iat[0]} ({n_sims} gare simulate giro per giro) ===")
    print(f"parametri: degrado %/giro {sp['deg']}, safety car/gara {sp['sc_rate']:.2f}, "
          f"pit {sp['pit_loss']:.1f}s, 2 soste {sp['p_two_stop']:.0%}, incertezza passo {sigma:.2f}%")
    print(f"sorpassi a {sp['circuit']}: P(passa) a pari passo {expit(sp['pass_logit']):.1%}, "
          f"soglia 50% = {-sp['pass_logit'] / sp['pass_slope']:.2f} s/giro")
    print("effetto giri in scia (logit, fasce 1/2/3/4-5/6-10/>10):", np.round(sp["age_curve"], 2))
    if "start_sd" in sp:
        tg = sp["start_targets"]
        print(f"partenza: rumore al via {sp['start_sd_a']:.2f}s + {sp['start_sd_b']:.3f}s per casella "
              f"(obiettivi 2026: poleman primo dopo giro 1 {tg[0]:.0%}, posizioni cambiate {tg[1]:.2f})")
        f["start_form"] = sp["start_form"]
        print("forma in partenza (posizioni guadagnate in piu' della media, gare 2026):")
        print(f.sort_values("start_form", ascending=False)[["driver", "start_form"]].round(2)
              .to_string(index=False))
    if not PACE_MODEL:
        print("pesi modello passo:", dict(zip(PACE_FEATURES, np.round(m.coef_, 3))))
    show = f.sort_values("P(vittoria)", ascending=False)[
        ["driver", "team", "grid", "quali_pace", "fp_pace", "pace", "P(vittoria)", "P(podio)", "P(punti)",
         "P(ritiro)", "pos_mediana"]]
    fmt = {c: "{:.1%}".format for c in ["P(vittoria)", "P(podio)", "P(punti)", "P(ritiro)"]}
    fmt.update({c: "{:.2f}".format for c in ["quali_pace", "fp_pace", "pace"]})
    print(show.to_string(index=False, formatters=fmt))
    show.to_csv(f"data/prediction_sim_r{target}{tag}.csv", index=False)
    position_matrix_csv(sp["pos_matrix"], f["driver"].tolist(), f"data/matrix_sim_r{target}{tag}.csv")
    export_events(sp, f, target, tag)


if __name__ == "__main__":
    main()
