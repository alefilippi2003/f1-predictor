#!/bin/bash
# Grande backtest: simulatore con diversi modelli del passo, su ogni stagione 2023-2026
cd "$(dirname "$0")/.."
ROOT=$(pwd)
run() {  # stagione variante
  y=$1; v=$2
  if [ "$y" = 2026 ]; then dir=$ROOT; else dir=$ROOT/seasons/$y; fi
  last=$(cd $dir && $ROOT/.venv/bin/python -c "import pandas as pd; w=pd.read_csv('data/weekends.csv'); print(int(w.loc[w.finish_pos.notna(),'round'].max())+1)")
  (cd $dir && F1_SEASON=$y $ROOT/.venv/bin/python $ROOT/racesim.py $last --variant $v --sigmas 0.05,0.3 --backtest-only \
      > $dir/data/bigbt_$v.log 2>&1)
  echo "$y $v: $(grep '^Scelto' $dir/data/bigbt_$v.log) $(grep -c Traceback $dir/data/bigbt_$v.log) errori"
}
jobs_list=()
for y in ${SEASONS:-2023 2024 2025 2026}; do for v in ${VARIANTS:-A ML_lasso ML_lgbm ML_lasso_stag ML_lgbm_stag}; do
  [ "$y" = 2023 ] && [[ "$v" == *_stag ]] && continue   # 2023: nessuna stagione precedente, coincide
  jobs_list+=("$y $v")
done; done
i=0
for j in "${jobs_list[@]}"; do
  run $j &
  i=$((i+1)); if [ $((i % 2)) = 0 ]; then wait; fi
done
wait
for y in ${SEASONS:-2023 2024 2025 2026}; do
  if [ "$y" = 2026 ]; then dir=$ROOT; else dir=$ROOT/seasons/$y; fi
  echo "=== $y ==="
  (cd $dir && F1_SEASON=$y timeout 900 $ROOT/.venv/bin/python $ROOT/compare.py 2>&1 | tail -12)
done
