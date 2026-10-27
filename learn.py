"""Quali variabili contano davvero per il passo gara? Confronto di modelli su tutte le stagioni.

Ogni gara di test viene prevista con modelli allenati SOLO su gare precedenti, in due modalita':
  stagione   solo le gare precedenti della stessa stagione
  tutte      tutte le stagioni precedenti + le gare precedenti della stagione

Modelli: solo qualifica, modello A attuale (5 variabili), Lasso, LightGBM, rete neurale.
Metriche per gara (passo previsto vs passo gara reale): correlazione d'ordine, errore medio (%).
Poi l'importanza delle variabili (permutation importance + coefficienti Lasso).

Uso: python learn.py            (dalla cartella principale)
Output: data/learn_results.csv, data/learn_importance.csv
"""
import glob
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
MIN_TRAIN_ROWS = 100
ONLY = None  # lista di modelli da valutare (None = tutti)
FIRST_TEST_ROUND = 6

FEATURES = [
    # qualifica e partenza
    "quali_gap", "quali_pos", "grid", "grid_penalty", "quali_tm_gap",
    # libere e sprint
    "pace_FP1", "pace_FP2", "pace_FP3", "pace_S", "sprint_pos", "longrun_laps", "fp_tm_gap", "fp_best_lap_gap",
    # forma pilota
    "drv_race_pace3", "drv_points3", "drv_avg_gain", "drv_start_form", "drv_dnf_rate",
    # forma scuderia
    "team_race_pace3", "team_quali_gap3", "team_points3", "team_dnf_rate",
    # pista e condizioni
    "track_overtake", "track_sc_rate", "weekend_deg", "sprint_weekend", "wet_before_race",
]
# i passi delle libere hanno valori assurdi (bandiere rosse, programmi): li limito per i modelli lineari
CLIP = {c: (-5, 5) for c in ["pace_FP1", "pace_FP2", "pace_FP3", "pace_S", "fp_tm_gap", "fp_best_lap_gap",
                             "quali_gap", "quali_tm_gap"]}


# variabili di passo/distacco: nella versione "relativa" diventano distacco dalla mediana del gruppo
# in quella gara, cosi' il modello impara chi e' piu' veloce di chi e non quanto sono larghi i
# distacchi su quella pista
PACE_LIKE = ["quali_gap", "pace_FP1", "pace_FP2", "pace_FP3", "pace_S", "fp_best_lap_gap", "fp_pooled",
             "drv_race_pace3", "team_race_pace3", "team_quali_gap3"]
# uguali per tutti i piloti della stessa gara: con obiettivo relativo non servono a ordinare
RACE_CONST = ["track_overtake", "track_sc_rate", "weekend_deg", "sprint_weekend", "wet_before_race"]
REL_FEATURES = [f"rel_{c}" if c in PACE_LIKE else c for c in FEATURES if c not in RACE_CONST]


def load():
    paths = sorted(glob.glob("seasons/*/data/features.csv")) + ["data/features.csv"]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c, (lo, hi) in CLIP.items():
        df[c] = df[c].clip(lo, hi)
    df["fp_pooled"] = df["fp_pooled"].clip(-5, 5)
    df["key"] = df["season"] * 100 + df["round"]
    # replica esatta del modello A: libere (tutte le sessioni) scartate se oltre il 2% dalla qualifica
    ok = (df["fp_pooled"] - df["quali_gap"]).abs() <= 2.0
    df["A_has_fp"] = ok.astype(float)
    df["A_fp"] = df["fp_pooled"].where(ok, df["quali_gap"])
    for c in PACE_LIKE + ["race_pace"]:
        df[f"rel_{c}"] = df[c] - df.groupby("key")[c].transform("median")
    return df


def lgbm():
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)


def lgbm_tuned():
    # scelte da tune.py (2023-25 + 2026 R6-R11, verificate sul 2026 da R12)
    return lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.01, num_leaves=7, min_child_samples=10, subsample=0.6,
        subsample_freq=1, colsample_bytree=1.0, reg_lambda=0.0, verbose=-1, random_state=0)


# obiettivo di ogni modello (default race_pace) e peso delle gare della stagione in corso
TARGET = {"LightGBM (relativo)": "rel_race_pace", "LightGBM (relativo, stagione x3)": "rel_race_pace"}
SEASON_WEIGHT = {"LightGBM (tutte, stagione x3)": 3.0, "LightGBM (relativo, stagione x3)": 3.0,
                 "LightGBM regolato (stagione x3)": 3.0}


def fit(name, pool, season):
    """Allena il modello `name` sul pool; le gare della stagione `season` pesano SEASON_WEIGHT volte."""
    feats, make = MODELS[name]
    y = pool[TARGET.get(name, "race_pace")]
    w = np.where(pool["season"] == season, SEASON_WEIGHT.get(name, 1.0), 1.0)
    m = make()
    if name in SEASON_WEIGHT:
        m.fit(pool[feats], y, sample_weight=w)
    else:
        m.fit(pool[feats], y)
    return m


def linear(model):
    # dati mancanti: mediana + colonna "dato presente sì/no"; poi standardizzo
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), model)


MODELS = {
    "solo qualifica": (["quali_gap"], lambda: linear(HuberRegressor(alpha=1e-3, max_iter=1000))),
    "modello A (5 variabili)": (["quali_gap", "A_fp", "A_has_fp", "team_race_pace3", "drv_race_pace3"],
                                lambda: linear(HuberRegressor(alpha=1e-2, max_iter=1000))),
    "Lasso (tutte)": (FEATURES, lambda: linear(LassoCV(cv=5, alphas=30, max_iter=5000, random_state=0))),
    "LightGBM (tutte)": (FEATURES, lambda: lgbm()),
    "LightGBM (tutte, stagione x3)": (FEATURES, lambda: lgbm()),
    "LightGBM regolato": (FEATURES, lambda: lgbm_tuned()),
    "LightGBM regolato (stagione x3)": (FEATURES, lambda: lgbm_tuned()),
    "LightGBM (relativo)": (REL_FEATURES, lambda: lgbm()),
    "LightGBM (relativo, stagione x3)": (REL_FEATURES, lambda: lgbm()),
    "Rete neurale (tutte)": (FEATURES, lambda: linear(MLPRegressor(
        hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, max_iter=2000,
        early_stopping=True, random_state=0))),
}


def evaluate(df):
    rows = []
    tests = df[df["round"] >= FIRST_TEST_ROUND][["season", "round", "key"]].drop_duplicates().sort_values("key")
    for _, tr in tests.iterrows():
        test = df[(df["key"] == tr["key"]) & df["race_pace"].notna()]
        if len(test) < 10:
            continue
        for mode in ["stagione", "tutte"]:
            pool = df[(df["key"] < tr["key"]) & df["race_pace"].notna()]
            if mode == "stagione":
                pool = pool[pool["season"] == tr["season"]]
            if len(pool) < MIN_TRAIN_ROWS:
                continue
            for name in ONLY or MODELS:
                feats = MODELS[name][0]
                pred = fit(name, pool, tr["season"]).predict(test[feats])
                # errore dopo aver allineato le mediane: confrontabile fra obiettivi assoluti e relativi
                err = (pred - np.median(pred)) - (test["race_pace"] - test["race_pace"].median())
                rows.append({"season": tr["season"], "round": tr["round"], "modalita": mode, "modello": name,
                             "spearman": spearmanr(pred, test["race_pace"]).statistic,
                             "errore_pct": float(np.mean(np.abs(err)))})
        print(f"  {tr['season']} R{tr['round']:02d} fatto", flush=True)
    return pd.DataFrame(rows)


def importance(df):
    """Allena su tutto tranne il 2026, misura l'importanza sul 2026 (dati mai visti)."""
    train = df[(df["season"] < 2026) & df["race_pace"].notna()]
    test = df[(df["season"] == 2026) & df["race_pace"].notna()]
    out = {}
    for name in ["Lasso (tutte)", "LightGBM (tutte)"]:
        feats, make = MODELS[name]
        m = make().fit(train[feats], train["race_pace"])
        pi = permutation_importance(m, test[feats], test["race_pace"], n_repeats=20, random_state=0,
                                    scoring="neg_mean_absolute_error")
        out[f"{name}: peggioramento errore se rimescolata"] = pd.Series(pi.importances_mean, index=feats)
    # coefficienti Lasso standardizzati: segno e peso (0 = variabile scartata)
    feats, make = MODELS["Lasso (tutte)"]
    m = make().fit(train[feats], train["race_pace"])
    coef = m[-1].coef_[:len(feats)]
    out["Lasso: coefficiente (standardizzato)"] = pd.Series(coef, index=feats)
    return pd.DataFrame(out)


def main():
    global ONLY
    import sys
    if "--only" in sys.argv:
        ONLY = sys.argv[sys.argv.index("--only") + 1].split(";")
    df = load()
    print(f"{len(df)} righe, stagioni {sorted(df['season'].unique())}")
    res = evaluate(df)
    res.to_csv("data/learn_results.csv", index=False)
    pd.set_option("display.width", 220)
    summary = res.groupby(["modalita", "modello"]).agg(
        gare=("spearman", "size"), spearman=("spearman", "mean"), errore_pct=("errore_pct", "mean")).round(3)
    print("\n=== Tutte le stagioni ===")
    print(summary.to_string())
    print("\n=== Per stagione (modalita' 'tutte') ===")
    print(res[res["modalita"] == "tutte"].pivot_table(index="modello", columns="season", values="spearman",
                                                     aggfunc="mean").round(3).to_string())
    if ONLY:
        return
    imp = importance(df)
    imp.to_csv("data/learn_importance.csv")
    print("\n=== Importanza delle variabili (allenato 2023-25, misurato sul 2026) ===")
    print(imp.sort_values(imp.columns[1], ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
