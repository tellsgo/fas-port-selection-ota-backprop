#!/usr/bin/env bash
# One-command reproduction. Modes:
#   bash run_all.sh quick   (default) recompute Tables I-IV, the text intervals, Figs. 1-5, the certificate constants
#                           and the training surrogate at 0.9 lambda from the shipped results; about 2 minutes on a laptop CPU
#   bash run_all.sh cpu     quick + every CPU experiment that needs no ray tracing: surrogate at 0.7 lambda, 3GPP CDL check,
#                           finite-power link training (extended version); see README for times
#   bash run_all.sh full    regenerate the ray-traced data and every result (needs Sionna RT and the Chicago scene, see README)
# Outputs go to out/ (override with OUT=...). PY selects the python interpreter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE="${1:-quick}"; OUT="${OUT:-out}"; PY="${PY:-python3}"; mkdir -p "$OUT"

quick() {
  echo "== certificate constants (Fig. 1)";       $PY scripts/certificate_example.py "$OUT/certificate_example.json"
  echo "== tables and intervals";                 $PY scripts/tables.py "$OUT"
  echo "== figures";                              $PY scripts/make_figures.py "$OUT"
  echo "== training surrogate, 0.9 lambda";       $PY scripts/e2e_surrogate.py "$OUT/e2e_d09.json" 0.9 > "$OUT/e2e_d09.log"
  $PY scripts/check_json.py e2e "$OUT/e2e_d09.json" results/e2e/results_E2.json
}
cpu() {
  quick
  echo "== training surrogate, 0.9 and 0.7 lambda"; $PY scripts/e2e_surrogate.py "$OUT/e2e.json" 0.9,0.7 > "$OUT/e2e.log"
  $PY scripts/check_json.py e2e "$OUT/e2e.json" results/e2e/results_E2.json
  echo "== finite-power link training";          $PY scripts/d1_finite_power.py "$OUT/d1.json" > "$OUT/d1.log"
  $PY scripts/check_json.py d1 "$OUT/d1.json" results/d1/d1_v2_results.json
  echo "== 3GPP CDL second-order check";         $PY scripts/cdl_check.py "$OUT/cdl.json" > "$OUT/cdl.log"
  $PY scripts/check_json.py cdl "$OUT/cdl.json" results/cdl/results.json
}
full() {
  : "${CHICAGO_DIR:?set CHICAGO_DIR to the Chicago scene folder (01chicago.xml, scene_config.json, xyz_paths.json)}"
  D="$OUT/data"; mkdir -p "$D"
  HERE="$(pwd)"; (cd "$OUT" && for b in 2 1; do $PY "$HERE/full/gen_real4.py" $b; done && $PY "$HERE/full/gen_real2_repro.py" 2)
  $PY full/gen_real3_data.py 2 9600 0.0 "$D/bs2_scat0.npz"; $PY full/gen_real3_data.py 2 9600 0.3 "$D/bs2_scat03.npz"; $PY full/gen_real3_data.py 1 9600 0.0 "$D/bs1_scat0.npz"
  $PY full/lm_chicago.py "$D/bs2_scat0.npz" "$D/bs1_scat0.npz"
  for sc in munich etoile florence san_francisco; do for s in high low; do $PY full/gen_xscene.py $sc $s 3000 "$D/${sc}_${s}.npz"; done; done
  for n in bs2_scat0 bs2_scat03 bs1_scat0; do $PY full/real3_model.py "$D/$n.npz" test "$OUT/test_$n"; done
  mkdir -p "$D/xscene"; (cd "$D/xscene" && for f in ../*.npz ../*_LM.npy; do ln -sf "$f" .; done; rm -f bs2_scat03.npz)
  $PY full/xtest.py "$D/xscene" test "$OUT/xtest"
  $PY full/d3_joint_error.py "$D/bs2_scat0.npz" "$OUT/d3_results.json"
  $PY full/extract_small.py "$D" "$OUT/extract"
  echo "Full run done. The ray tracer is not bit-reproducible across runs; compare $OUT with results/ by rates, not digits."
}
case "$MODE" in quick) quick;; cpu) cpu;; full) full;; *) echo "usage: bash run_all.sh [quick|cpu|full]"; exit 2;; esac
echo "done: $MODE (outputs in $OUT/)"
