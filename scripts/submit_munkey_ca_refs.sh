#!/usr/bin/env bash
# PKM paper, 2026-11-24: three additions to the poisoned grid (all training on H200, as the Toys grid).
#   munkey  MUNKEY baseline (mk11, picked on validation) on the 14 remaining beauty cells:
#           user keys (CPU) -> training -> chain: pre, zs (key deletion), zsfull and full x F+R / NAU
#   ca      second attack, clone-append (beauty, mid target, data seeds 2 3 5 7 11):
#           plain im imc ims imsc -> chain: pre, full x F+R / NAU (+ memory-off evals)
#   refs    retrained references (clean model, zero-step eval on each cell's spam data, CPU):
#           beauty: plain/imsc for the 3 missing mid cells, im (new imclean model) for all 15;
#           clone-append: plain/im/imsc x 5 seeds; toys: cleanref_{plain,im,imsc} models x 15 cells
# Usage (from GRID/): bash scripts/submit_munkey_ca_refs.sh [munkey] [ca] [refs]
set -euo pipefail
unset ${!SLURM_*}
cd "$(dirname "$0")/.."
export TRAIN_GRES=gpu:nvidia_h200:2
LOG=logs/munkey_ca_refs_submit_$(date +%Y%m%d).txt
jid() { grep -oE "Submitted batch job [0-9]+" | awk '{print $4}' | paste -sd: ; }
chain() {  # $1 dependency (a:b:c), $2 job name, $3 command run from GRID/
  sbatch --parsable --partition=compute --gres=none -c 1 --mem=2G --time=00:30:00 \
    --dependency=afterok:$1 --job-name="$2" --output=logs/%j.out \
    --wrap "cd $PWD && unset \${!SLURM_*} && export TRAIN_GRES=$TRAIN_GRES && $3"
}
tok() { case $1 in unpopular) echo _up ;; popular) echo _pop ;; *) echo "" ;; esac; }
WHAT="${*:-munkey ca refs}"

if [[ " $WHAT " == *" munkey "* ]]; then
  for st in unpopular mid popular; do
    for d in 2 3 5 7 11; do
      [ $st = mid ] && [ $d = 2 ] && continue   # the original cell is complete
      dt=""; [ $d != 2 ] && dt="_d$d"; m="mk11$(tok $st)$dt"
      kid=$(GRID_KEYS=1 STAGE=keys STRATUM=$st DSEED=$d ./run_munkey_beauty.sh | jid)
      tid=$(DEP=$kid STAGE=train ARMS=mk11 STRATUM=$st DSEED=$d ./run_munkey_beauty.sh | jid)
      cid=$(chain $tid "chain_mk_$m" "STAGE=unlearn SPECS='$m:pre $m:zs $m:zsfull $m:full' ./run_munkey_beauty.sh")
      echo "munkey $st $d $m keys=$kid train=$tid chain=$cid" | tee -a "$LOG"
    done
  done
fi

if [[ " $WHAT " == *" ca "* ]]; then
  for d in 2 3 5 7 11; do
    dt=""; [ $d != 2 ] && dt="_d$d"
    for a in plain im imc ims imsc; do
      m="${a}_ca$dt"
      tid=$(ATTACK=clone_append STAGE=train ARMS=$a DSEED=$d ./run_item_memory_beauty.sh | jid)
      cid=$(chain $tid "chain_ca_$m" "STAGE=unlearn SPECS='$m:pre $m:full' ./run_item_memory_beauty.sh")
      echo "ca mid $d $m train=$tid chain=$cid" | tee -a "$LOG"
    done
  done
fi

if [[ " $WHAT " == *" refs "* ]]; then
  # beauty plain / imsc: models exist, the 3 missing mid cells now
  for d in 2 3 5; do
    STAGE=ref STRATUM=mid DSEED=$d REF_MODELS="plain imsc" ./run_item_memory_beauty.sh | jid \
      | sed "s/^/ref beauty mid $d plain+imsc jobs=/" | tee -a "$LOG"
  done
  # beauty im: train the clean im model, then im refs on all 15 cells + clone-append refs
  tim=$(STAGE=train ARMS=imclean ./run_item_memory_beauty.sh | jid)
  cmd=""
  for st in unpopular mid popular; do for d in 2 3 5 7 11; do
    cmd+="STAGE=ref STRATUM=$st DSEED=$d REF_MODELS=im ./run_item_memory_beauty.sh; "
  done; done
  for d in 2 3 5 7 11; do
    cmd+="ATTACK=clone_append STAGE=ref DSEED=$d REF_MODELS='plain im imsc' ./run_item_memory_beauty.sh; "
  done
  cid=$(chain $tim chain_ref_beauty_im "$cmd")
  echo "ref beauty imclean train=$tim chain=$cid (im x15 cells, clone-append x5 plain/im/imsc)" | tee -a "$LOG"
  # toys: three clean models, then refs on all 15 toys cells
  tt=$(DS=toys STAGE=train ARMS="cleanref_plain cleanref_im cleanref_imsc" ./run_item_memory_beauty.sh | jid)
  cmd=""
  for st in unpopular mid popular; do for d in 2 3 5 7 11; do
    cmd+="DS=toys STAGE=ref STRATUM=$st DSEED=$d REF_MODELS='plain im imsc' ./run_item_memory_beauty.sh; "
  done; done
  cid=$(chain $tt chain_ref_toys "$cmd")
  echo "ref toys cleanref train=$tt chain=$cid (15 cells x plain/im/imsc)" | tee -a "$LOG"
fi
