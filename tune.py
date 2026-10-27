"""Regolazione delle impostazioni di LightGBM per il passo gara (modalita' 'tutte', walk-forward).

Le impostazioni si scelgono sulle gare 2023-25 + la prima parte del 2026 (nuovo regolamento: deve
funzionare anche quando i rapporti di forza ripartono da zero); la seconda parte del 2026 resta come
verifica su gare mai usate per la scelta.

Uso: python tune.py [n_prove]      Output: data/tune_results.csv
"""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from learn import FEATURES, FIRST_TEST_ROUND, MIN_TRAIN_ROWS, load

warnings.filterwarnings("ignore")
VERIFY_FROM = 12  # 2026: dalla R12 in poi solo verifica
BASE = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0)
GRID = dict(n_estimators=[150, 300, 600], learning_rate=[0.01, 0.03, 0.06], num_leaves=[4, 7, 15, 31],
            min_child_samples=[10, 20, 40, 80], subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.4, 0.6, 0.8, 1.0],
            reg_lambda=[0.0, 1.0, 5.0, 20.0], objective=["regression", "huber"])


def evaluate(df, params):
    rows = []
    tests = df[df["round"] >= FIRST_TEST_ROUND][["season", "round", "key"]].drop_duplicates().sort_values("key")
    for _, tr in tests.iterrows():
        test = df[(df["key"] == tr["key"]) & df["race_pace"].notna()]
        pool = df[(df["key"] < tr["key"]) & df["race_pace"].notna()]
        if len(test) < 10 or len(pool) < MIN_TRAIN_ROWS:
            continue
        m = lgb.LGBMRegressor(subsample_freq=1, verbose=-1, random_state=0, n_jobs=2, **params)
        pred = m.fit(pool[FEATURES], pool["race_pace"]).predict(test[FEATURES])
        err = (pred - np.median(pred)) - (test["race_pace"] - test["race_pace"].median())
        rows.append({"season": tr["season"], "round": tr["round"], "spearman": spearmanr(pred, test["race_pace"]).statistic,
                     "errore": float(np.mean(np.abs(err)))})
    r = pd.DataFrame(rows)
    ver = (r["season"] == 2026) & (r["round"] >= VERIFY_FROM)
    sel, chk, y26 = r[~ver], r[ver], r[r["season"] == 2026]
    return {"sp_scelta": sel["spearman"].mean(), "err_scelta": sel["errore"].mean(),
            "sp_verifica": chk["spearman"].mean(), "err_verifica": chk["errore"].mean(),
            "sp_26": y26["spearman"].mean(), "err_26": y26["errore"].mean()}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    df = load()
    rng = np.random.default_rng(0)
    trials = [dict(BASE, objective="regression")]
    trials += [{k: v[rng.integers(len(v))] for k, v in GRID.items()} for _ in range(n)]
    rows = []
    for i, p in enumerate(trials):
        res = evaluate(df, p)
        rows.append({**p, **res})
        print(f"{i:3d} scelta sp {res['sp_scelta']:.4f} err {res['err_scelta']:.4f} | verifica sp "
              f"{res['sp_verifica']:.4f} err {res['err_verifica']:.4f} | 2026 sp {res['sp_26']:.4f}  {p}", flush=True)
        pd.DataFrame(rows).to_csv("data/tune_results.csv", index=False)
    out = pd.DataFrame(rows)
    out["rank"] = out["sp_scelta"].rank(ascending=False) + out["err_scelta"].rank()
    pd.set_option("display.width", 250)
    print("\n=== Migliori (scelti su 2023-25 + 2026 fino a R11; riga 0 = impostazioni attuali) ===")
    print(out.sort_values("rank").head(10).round(4).to_string())


if __name__ == "__main__":
    main()
