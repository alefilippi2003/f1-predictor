"""Modello di ranking + simulazione Monte Carlo per prevedere un GP 2026.

Uso: python model.py [round_da_prevedere]   (default: primo round senza risultati)
"""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
N_SIMS = 20000
MIN_TRAIN_ROUNDS = 5

# Guadagno per posizione = punti F1 (P1=25 ... P10=1, oltre = 0)
F1_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
LABEL_GAIN = [0] + F1_POINTS[::-1]

FEATURES = [
    "grid", "quali_gap_pct", "longrun_gap_pct", "longrun_rank", "sprint_pos",
    "form_finish3", "form_points3", "team_points3", "season_points_avg",
    "avg_gain", "team_quali_gap3",
]


def add_features(df):
    """Feature di forma calcolate SOLO con le gare precedenti (niente leakage)."""
    df = df.sort_values(["round", "driver"]).reset_index(drop=True)
    df["longrun_rank"] = df.groupby("round")["longrun_gap_pct"].rank()
    df["gain"] = df["grid"] - df["finish_pos"]

    by_drv = df.groupby("driver")
    prev = lambda s, w=None: s.shift(1).rolling(w or 99, min_periods=1).mean()
    df["form_finish3"] = by_drv["finish_pos"].transform(lambda s: prev(s, 3))
    df["form_points3"] = by_drv["points"].transform(lambda s: prev(s, 3))
    df["season_points_avg"] = by_drv["points"].transform(prev)
    df["avg_gain"] = by_drv["gain"].transform(prev)

    # affidabilita': DNF con prior bayesiano (circa 8% di ritiri a gara)
    n_prev = by_drv.cumcount()
    dnf_prev = by_drv["dnf"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    df["dnf_rate_drv"] = (dnf_prev + 0.4) / (n_prev + 5)

    team = df.groupby(["round", "team"]).agg(
        tp=("points", "mean"), tq=("quali_gap_pct", "mean"),
        td=("dnf", "sum"), tn=("dnf", "count")).reset_index()
    team = team.sort_values("round")
    g = team.groupby("team")
    team["team_points3"] = g["tp"].transform(lambda s: prev(s, 3))
    team["team_quali_gap3"] = g["tq"].transform(lambda s: prev(s, 3))
    team["dnf_rate_team"] = (g["td"].transform(lambda s: s.shift(1).fillna(0).cumsum()) + 0.8) / \
                            (g["tn"].transform(lambda s: s.shift(1).fillna(0).cumsum()) + 10)
    df = df.merge(team[["round", "team", "team_points3", "team_quali_gap3", "dnf_rate_team"]],
                  on=["round", "team"], how="left")
    df["p_dnf"] = 0.5 * df["dnf_rate_drv"] + 0.5 * df["dnf_rate_team"]
    return df


def relevance(finish_pos):
    return np.clip(11 - finish_pos, 0, 10).astype(int)


def fit(train):
    # i ritiri non dicono nulla sul passo: li toglie dal ranking (gestiti a parte con p_dnf)
    train = train[(train["dnf"] == 0) & train["finish_pos"].notna()].sort_values("round")
    model = lgb.LGBMRanker(
        objective="lambdarank", label_gain=LABEL_GAIN, n_estimators=300,
        learning_rate=0.03, num_leaves=7, min_child_samples=8,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
        random_state=0)
    model.fit(train[FEATURES], relevance(train["finish_pos"]),
              group=train.groupby("round").size().values)
    return model


def simulate(scores, p_dnf, beta, n=N_SIMS):
    """Plackett-Luce via trucco di Gumbel + ritiri casuali. Ritorna matrice posizioni."""
    k = len(scores)
    util = beta * scores[None, :] + RNG.gumbel(size=(n, k))
    dnf = RNG.random((n, k)) < p_dnf[None, :]
    util[dnf] = -np.inf
    order = np.argsort(-util, axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n)[:, None], order] = np.arange(1, k + 1)[None, :]
    return pos, dnf


def pl_loglik(beta, races, top_k=3):
    """Log-verosimiglianza Plackett-Luce delle prime top_k posizioni reali (solo classificati).
    Calibrare su tutto l'ordine renderebbe il modello troppo prudente in testa:
    il centro gruppo e' molto piu' caotico del podio."""
    ll = 0.0
    for scores, finish in races:
        s = beta * scores[np.argsort(finish)]
        for i in range(min(top_k, len(s))):
            ll += s[i] - np.logaddexp.reduce(s[i:])
    return ll


def walk_forward(df, last_round):
    rows, races = [], []
    for rnd in range(MIN_TRAIN_ROUNDS + 1, last_round + 1):
        train, test = df[df["round"] < rnd], df[df["round"] == rnd].copy()
        if test["finish_pos"].isna().all():
            continue
        test["score"] = fit(train).predict(test[FEATURES])
        fin = test[test["dnf"] == 0]
        races.append((fin["score"].values, fin["finish_pos"].values))
        rows.append(test)
    return pd.concat(rows), races


def fit_beta(races):
    return minimize_scalar(lambda b: -pl_loglik(b, races), bounds=(0.01, 20), method="bounded").x


def evaluate(oos, beta, beta_grid):
    out = []
    for rnd, r in oos.groupby("round"):
        fin = r[r["dnf"] == 0]
        actual_win = r.loc[r["finish_pos"].idxmin(), "driver"]
        actual_pod = set(r.nsmallest(3, "finish_pos")["driver"])
        model_order = r.sort_values("score", ascending=False)["driver"].tolist()
        grid_order = r.sort_values("grid")["driver"].tolist()
        pos, _ = simulate(r["score"].values, r["p_dnf"].values, beta, n=4000)
        p_win = (pos == 1).mean(axis=0)
        pos_g, _ = simulate(-r["grid"].values, r["p_dnf"].values, beta_grid, n=4000)
        p_win_g = (pos_g == 1).mean(axis=0)
        i_win = r["driver"].tolist().index(actual_win)
        out.append({
            "round": rnd, "event": r["event"].iat[0], "vincitore": actual_win,
            "modello": model_order[0], "pole": grid_order[0],
            "win_modello": model_order[0] == actual_win,
            "win_pole": grid_order[0] == actual_win,
            "podio_modello": len(actual_pod & set(model_order[:3])),
            "podio_griglia": len(actual_pod & set(grid_order[:3])),
            "spearman_modello": spearmanr(-fin["score"], fin["finish_pos"]).statistic,
            "spearman_griglia": spearmanr(fin["grid"], fin["finish_pos"]).statistic,
            "p_vincitore": p_win[i_win],
            "p_vincitore_griglia": p_win_g[i_win],
        })
    return pd.DataFrame(out)


def main():
    df = add_features(pd.read_csv("data/weekends.csv"))
    done = df.groupby("round")["finish_pos"].apply(lambda s: s.notna().any())
    global N_SIMS
    args = sys.argv[1:]
    if "--sims" in args:
        N_SIMS = int(args.pop(args.index("--sims") + 1))
        args.remove("--sims")
    target = int(args[0]) if args else int(done[~done].index.min())
    last_done = int(done[done].index.max())
    pd.set_option("display.width", 200)

    # 1) backtest onesto: per ogni gara alleno solo sulle precedenti
    oos, races = walk_forward(df, min(last_done, target - 1))
    beta = fit_beta(races)
    fin = oos[oos["dnf"] == 0]
    beta_grid = fit_beta([(-g["grid"].values, g["finish_pos"].values) for _, g in fin.groupby("round")])
    ev = evaluate(oos, beta, beta_grid)
    print("=== BACKTEST walk-forward (ogni gara prevista con i soli dati precedenti) ===")
    print(ev[["round", "event", "vincitore", "modello", "pole", "p_vincitore",
              "p_vincitore_griglia"]].to_string(index=False))
    print(f"\nVincitore azzeccato  modello: {ev['win_modello'].sum()}/{len(ev)}   "
          f"'vince chi parte primo': {ev['win_pole'].sum()}/{len(ev)}")
    print(f"Podio (su 3)         modello: {ev['podio_modello'].mean():.2f}       "
          f"griglia: {ev['podio_griglia'].mean():.2f}")
    print(f"Spearman ordine      modello: {ev['spearman_modello'].mean():.3f}      "
          f"griglia: {ev['spearman_griglia'].mean():.3f}")
    print(f"Prob. media data al vero vincitore  modello: {ev['p_vincitore'].mean():.1%}   "
          f"solo griglia: {ev['p_vincitore_griglia'].mean():.1%}")
    print(f"Log-loss vincitore (piu' basso = meglio)  modello: {-np.log(ev['p_vincitore']).mean():.3f}   "
          f"solo griglia: {-np.log(ev['p_vincitore_griglia']).mean():.3f}")
    ev.to_csv("data/backtest.csv", index=False)

    # 2) previsione
    train, test = df[df["round"] < target], df[df["round"] == target].copy()
    model = fit(train)
    test["score"] = model.predict(test[FEATURES])
    # a blocchi: con milioni di simulazioni la memoria non basterebbe
    D, counts, dnf_count, done = len(test), np.zeros((len(test), len(test))), np.zeros(len(test)), 0
    while done < N_SIMS:
        k = min(50000, N_SIMS - done)
        pos, dnf = simulate(test["score"].values, test["p_dnf"].values, beta, n=k)
        # posizione d'arrivo solo per chi finisce (i ritirati sono gia' in fondo all'ordine,
        # quindi chi arriva occupa le posizioni 1..arrivati); i ritiri a parte
        for j in range(D):
            counts[j] += np.bincount(pos[~dnf[:, j], j] - 1, minlength=D)
        dnf_count += dnf.sum(0)
        done += k
    # prob[pilota, posizione-1], ultima colonna = ritirato
    prob = np.column_stack([counts / N_SIMS, dnf_count / N_SIMS])
    test["P(vittoria)"] = prob[:, 0]
    test["P(podio)"] = prob[:, :3].sum(1)
    test["P(punti)"] = prob[:, :10].sum(1)
    test["P(ritiro)"] = prob[:, -1]
    test["pos_attesa"] = (prob.cumsum(1) >= 0.5).argmax(1) + 1
    idx = pd.Index([str(i) for i in range(1, D + 1)] + ["Ritirato"], name="posizione")
    m = pd.DataFrame(prob.T, index=idx, columns=test["driver"].tolist())
    m[m.columns[np.argsort((prob * np.arange(1, D + 2)).sum(1))]].to_csv(
        f"data/matrix_rank_r{target}.csv", float_format="%.6f")

    print(f"\n=== PREVISIONE R{target} {test['event'].iat[0]} ({N_SIMS} simulazioni) ===")
    show = test.sort_values("P(vittoria)", ascending=False)[
        ["driver", "team", "grid", "P(vittoria)", "P(podio)", "P(punti)", "P(ritiro)", "pos_attesa"]]
    fmt = {c: "{:.1%}".format for c in ["P(vittoria)", "P(podio)", "P(punti)", "P(ritiro)"]}
    print(show.to_string(index=False, formatters=fmt))
    show.to_csv(f"data/prediction_r{target}.csv", index=False)

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    print("\nImportanza feature (gain):")
    print((imp / imp.sum()).sort_values(ascending=False).map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
