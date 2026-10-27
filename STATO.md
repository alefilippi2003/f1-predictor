# Stato del progetto (27 ottobre 2026)

Previsione dei GP di F1 2026: stima del passo gara, poi simulatore Monte Carlo giro per giro, poi mercati con quote eque (1/p).

## Pipeline
- `data.py` (OpenF1, cache in `cache/openf1`): `data/weekends.csv`. `GRID_OVERRIDE` per inserire a mano la griglia con le penalità di una gara non ancora corsa.
- `laps.py`: `data/laps.csv`, `data/races.csv`. `features.py`: `data/features.csv` (27 variabili, una colonna ciascuna).
- `overtakes.py` (sorpassi per pista, storico 2023–25 + livello 2026), `starts.py` (caos della partenza + forma in partenza dei piloti).
- `racesim.py [round] --sims N [--variant ...] [--skip-backtest]`: simulazioni, matrice delle posizioni, `data/sims_r{round}_ML_lgbm_t_w3.npz`.
- `markets.py [round] [giri] [variante]`: `data/mercati_r{round}_ML_lgbm_t_w3.xlsx` (25 mercati, testa a testa, scuderie; niente scommesse sui gruppi).
- `tune.py [n]`: regolazione delle impostazioni di LightGBM.
- `learn.py [--only "nome;nome"]`: confronto dei modelli del passo + importanza delle variabili. `compare.py`: tabella del backtest.
- Altre stagioni in `seasons/{2023,2024,2025}`, da lanciare con `F1_SEASON=y`. `seasons/big_backtest.sh` (le stagioni si scelgono con `SEASONS="2023 2025"`).

## Decisione presa
Modello del passo: **LightGBM regolato, allenato su tutte le stagioni, con le gare della stagione in corso che pesano 3 volte** (`ML_lgbm_t_w3`, predefinito dal 27 ottobre). Impostazioni in `learn.lgbm_tuned()`.

Backtest 27 ottobre (68 gare): errore di posizione 2.21 (era 2.28), penalità sul vincitore 1.13 (era 1.17), 39 vincitori; 2026: errore 1.88, penalità 1.12. Migliora in tutte e 4 le stagioni.

Backtest precedente (prima della regolazione):

Backtest su 68 gare 2023–26, errore medio di posizione / log-loss sul vincitore / vincitori azzeccati:

| Modello | Errore di posizione | Log-loss vincitore | Vincitori |
|---|---|---|---|
| Ordine della griglia | 2.42 | – | – |
| Modello A | 2.32 | 1.41 | 40 |
| Lasso | 2.21 | 1.39 | 33 |
| LightGBM | 2.28 | 1.20 | 40 |

## Cosa conta (allenato 2023–25, misurato sul 2026)
- Contano: passo gara della scuderia nelle ultime 3 gare, distacco e posizione in qualifica, miglior giro nelle libere, passo gara del pilota nelle ultime 3 gare.
- Contano poco: griglia effettiva, FP2.
- Inutili: FP1, FP3, posizione nella sprint, penalità in griglia, pioggia prima della gara, degrado.

## Prove del 27 ottobre
- Backtest del simulatore delle varianti (68 gare): le "relative" sono scartate perché danno meno probabilità al vincitore vero (penalità 1.38–1.42 contro 1.17). "Stagione x3" (`ML_lgbm_w3`) fa appena meglio dell'attuale (penalità 1.12, 42 vincitori contro 40) ma il margine è piccolo.
- `tune.py`: 40 combinazioni di impostazioni di LightGBM, scelte sulle gare 2023–25 + 2026 R6–R11 e verificate sul 2026 da R12. Migliore: 600 alberi, velocità di apprendimento 0.01, 7 foglie, almeno 10 righe per foglia, 60% delle righe per albero, tutte le variabili, nessuna regolarizzazione. Sulla verifica l'errore scende da 0.388 a 0.359 (−7%), la correlazione d'ordine passa da 0.918 a 0.920. Applicata (vedi Decisione presa).

## Idee per dopo
- Togliere le variabili inutili (FP1, FP3, posizione nella sprint, penalità in griglia, pioggia, degrado) e vedere se il modello migliora.

## Problemi noti nei dati
- In OpenF1 mancano i giri della qualifica di Baku 2025 e i tempi della pit lane di 2023 R1–R6. `racesim.py` ora usa valori sostitutivi.
- OpenF1 risponde 401 durante le sessioni in diretta.
- Austin 2026 (R18), risultato reale: ANT-LEC-VER-HAM-NOR-HAD-PIA-RUS-LAW-GAS. Previsione (modello A): podio esatto, 10 su 10 dei piloti a punti.
