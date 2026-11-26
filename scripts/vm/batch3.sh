#!/usr/bin/env bash
# Batch 3: the shortcut-wins grid (results doc "RLVR study: results since the brainstorm", section 11), on a 1x A100
# 80GB VM with the data volume at /data, a checkpoint volume at /ckpt and local disk at /ephemeral. Ends by deleting
# the VM (unless NO_DELETE=1).
#
#   tmux new -d -s batch3 'bash scripts/vm/batch3.sh'
#
# Runs (Qwen3-4B SFT start, GRPO, correctness only, 300 steps; seeds 1 and 2 as the existing q = 0 / 0.6 runs):
#   cap128_q00  arith_cap128_q00.yaml   q = 0,   128-token training completion cap   (section 11 run 2: gives s)
#   cap128_q09  arith_cap128_q09.yaml   q = 0.9, cap 128                              (run 4)
#   cap128_q06  arith_cap128_q06.yaml   q = 0.6, cap 128                              (run 3)
#   q09         arith_planted_q09.yaml  q = 0.9, no cap (2048)                        (run 1)
#   zebra_plain zebra_plain.yaml, seed 2 only                                         (run 5)
# Order as section 11 asks: run 2 (both seeds) first, then 4, 3, 1, 5. cap128_q00 also writes $R/s_ready when
# both seeds' step-300 evals exist, for the local watcher (s is computed and written to the doc before any eval of
# runs 3 or 4 is opened).
# Evals (arithmetic): standard eval (500 puzzles, seed 9999), 8 samples at T=1.0 at steps 25 50 100 150 200 300;
# 16 samples + 1 at T=0.1 at step 300. Capped runs: at 2,048 tokens ($E/<run>/step<N>/std, the usual layout) and
# at the 128 cap ($E/<run>/step<N>/std128). Zebra: as zebra_plain seed 1 (200 puzzles, 8 + T=0.1, 2,048 tokens).
# Checkpoints: step 300 of every run is copied to /ckpt/<run>/checkpoint-300 (durable volume); the others are
# deleted after the run's evals. trainer_state, effective_config and training generations go to $R/<run>/.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend.
set -uo pipefail
DRY=${DRY:-0}
RUNS=${RUNS:-"cap128_q00_seed1 cap128_q00_seed2 cap128_q09_seed1 cap128_q09_seed2 cap128_q06_seed1 cap128_q06_seed2 q09_seed1 q09_seed2 zebra_plain_seed2"}
STEPS="25 50 100 150 200 300"
PY="python -m experiments.atom_decomposition"
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}; CK=$R/ephemeral; KEEP=$R/ckpt; SFT=none
    BACKEND="--backend fake --fake_p_correct 0.6"; N=20; NZ=10
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/batch3; CK=/ephemeral/batch3; KEEP=/ckpt; SFT=/data/rlvr/sft_qwen3_4b
    sudo mkdir -p $CK && sudo chown "$USER" $CK
    BACKEND="--backend vllm"; N=500; NZ=200
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval; EO=$R/eval_other
STD="full --tasks basic_arithmetic --n $N --seed 9999 --k 8 $BACKEND"
STD_LAST="full --tasks basic_arithmetic --n $N --seed 9999 --k 16 --t01 $BACKEND"
CAP="--max_new_tokens_full 128 --max_model_len 1024"
ZEBRA="full --tasks zebra_puzzles --n $NZ --seed 9999 --k 8 --t01 --max_new_tokens_full 2048 --max_model_len 4096 $BACKEND"
mkdir -p "$R" "$E" "$EO" "$CK" "$KEEP"
DONE=$R/batch3.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_batch3.txt 2>&1 || true
    sync
    [ "${NO_DELETE:-0}" = 1 ] && return
    sleep "${HOLD:-300}"
    cd "$HOME"   # leave /data so it can be unmounted
    bash "$HOME/self_delete.sh"
}
trap finish EXIT
rm -f "$DONE"

ev() {   # ev <name> <limit> <out_dir> <args...>: run an eval unless its manifest exists
    local name=$1 limit=$2 out=$3; shift 3
    [ -f $out/manifest.json ] || J $name $limit "$@" --out_dir $out || failed+=($name)
}

# --- SFT start at the 128 cap (the 2,048 eval exists in /data/rlvr/planted/eval/sft) ---
ev eval_sft_std128 30m $E/sft/step000/std128 $PY.eval_atoms $STD_LAST $CAP --model $SFT

for run in $RUNS; do
    name=${run%_seed*}; seed=${run#*_seed}
    case $name in
        cap128_*) cfg=experiments/atom_decomposition/configs/arith_$name.yaml; steps=$STEPS; limit=2h30m ;;
        q09) cfg=experiments/atom_decomposition/configs/arith_planted_q09.yaml; steps=$STEPS; limit=4h ;;
        zebra_plain) cfg=experiments/atom_decomposition/configs/zebra_plain.yaml; steps="100 200 300"; limit=5h ;;
    esac
    last=${steps##* }
    if [ ! -f $CK/$run/checkpoint-$last/config.json ] && [ ! -f $KEEP/$run/checkpoint-$last/config.json ]; then
        J train_$run $limit python train_grpo.py $cfg --seed $seed --output_dir $CK/$run || failed+=(train_$run)
    fi
    mkdir -p $R/$run
    cp $CK/$run/trainer_state.json $CK/$run/effective_config.json $R/$run/ 2>/dev/null || true
    cp -r $CK/$run/generations $R/$run/ 2>/dev/null || true
    for n in $steps; do
        ck=$CK/$run/checkpoint-$n
        [ -d $ck ] || ck=$KEEP/$run/checkpoint-$n
        [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$run-$n); continue; }
        case $name in
            zebra_plain)
                ev eval_${run}_zebra_$n 60m $EO/$run/zebra_step$n $PY.eval_atoms $ZEBRA --model $ck ;;
            *)
                args=$STD; [ $n = $last ] && args=$STD_LAST
                ev eval_${run}_std_$n 40m $E/$run/step$n/std $PY.eval_atoms $args --model $ck
                case $name in cap128_*)
                    ev eval_${run}_std128_$n 30m $E/$run/step$n/std128 $PY.eval_atoms $args $CAP --model $ck ;;
                esac ;;
        esac
    done
    # keep the last checkpoint on the durable volume, drop the rest
    if [ "$DRY" != 1 ] && [ -d $CK/$run/checkpoint-$last ]; then
        mkdir -p $KEEP/$run && cp -r $CK/$run/checkpoint-$last $KEEP/$run/ && sync \
            && [ -f $KEEP/$run/checkpoint-$last/config.json ] && rm -rf $CK/$run/checkpoint-* \
            || failed+=(keep_$run)
    fi
    if [ $name = cap128_q00 ] && [ -f $E/cap128_q00_seed1/step300/std128/manifest.json ] \
        && [ -f $E/cap128_q00_seed2/step300/std128/manifest.json ]; then
        date -u +%FT%TZ > $R/s_ready
    fi
done
echo "BATCH3 DONE (failed: ${failed[*]:-none})"
