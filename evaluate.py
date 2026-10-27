"""Metriche di confronto fra modelli, gara per gara, sull'ordine d'arrivo previsto.

Ogni modello fornisce per ogni pilota una "posizione attesa" (ritiro = in fondo).
L'ordine previsto e' quello delle posizioni attese.

  errore_pos    errore medio |posizione prevista - reale| sui piloti arrivati al traguardo
                (la posizione prevista e' ricalcolata fra i soli arrivati, per confronto equo)
  spearman      correlazione fra ordine previsto e ordine reale (arrivati)
  punti         quanti dei 10 previsti a punti sono arrivati davvero nei primi 10 (su 10)
  podio         quanti dei 3 previsti sul podio ci sono arrivati (su 3)
  vincitore     1 se il primo previsto ha vinto
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def race_metrics(pred, actual):
    """pred: Series pilota -> posizione attesa. actual: DataFrame con driver, finish_pos, dnf."""
    a = actual.set_index("driver")
    a = a[a.index.isin(pred.index)]
    order = pred.reindex(a.index).sort_values()
    fin = a[a["dnf"] == 0]
    pred_fin = pred.reindex(fin.index).rank(method="first")
    real_fin = fin["finish_pos"].rank(method="first")
    top10_real = set(a.nsmallest(10, "finish_pos").index)
    top3_real = set(a.nsmallest(3, "finish_pos").index)
    return {
        "errore_pos": float((pred_fin - real_fin).abs().mean()),
        "spearman": float(spearmanr(pred_fin, real_fin).statistic),
        "punti": len(set(order.index[:10]) & top10_real),
        "podio": len(set(order.index[:3]) & top3_real),
        "vincitore": int(order.index[0] == a["finish_pos"].idxmin()),
    }


def expected_position(prob_matrix, drivers):
    """prob_matrix[pilota, posizione-1], ultima colonna = ritiro (vale D+1)."""
    D = len(drivers)
    vals = np.arange(1, D + 2)
    return pd.Series((prob_matrix * vals[None, :]).sum(1), index=drivers)


def summarize(rows, name):
    d = pd.DataFrame(rows)
    return {"modello": name, "gare": len(d),
            "errore_pos": d["errore_pos"].mean(), "spearman": d["spearman"].mean(),
            "punti (su 10)": d["punti"].mean(), "podio (su 3)": d["podio"].mean(),
            "vincitori": int(d["vincitore"].sum())}
