#!/usr/bin/env bash
# Batch 4: the hint grid (results doc section 14), on a 1x A100 80GB VM with the data volume at /data, the
# checkpoint volume at /ckpt and local disk at /ephemeral. Ends by deleting the VM (unless NO_DELETE=1).
#
#   tmux new -d -s batch4 'bash scripts/vm/batch4.sh'
#
# 1. Probe (eval only, SFT start): each candidate hint in experiments/atom_decomposition/hint.py on 1000 probe
#    puzzles (seed 7777), 8 samples at T=1.0. hint.py select picks the weak and strong hints by its fixed rule and
#    writes $R/chosen.json; if it aborts (no candidate reaches 1%), the batch stops here.
# 2. Configs: arith_planted_q{00,06,09}.yaml plus environment.hint (and checkpoints at the eval steps only),
#    written to $R/configs.
# 3. SFT start with each chosen hint on the standard eval (500 puzzles, seed 9999), 16 samples + 1 at T=0.1:
#    the starting rate.
# 4. Runs (Qwen3-4B SFT start, GRPO, correctness only, 300 steps, no cap beyond 2,048): hint {weak, strong} x
#    q {0, 0.6, 0.9}, seeds 1 and 2; all of seed 1 first, cells in the order of CELLS.
#    Evals on the standard eval: with the run's hint at steps 25 50 100 150 200 (8 samples at T=1.0) and 300
#    (16 + T=0.1), $E/<run>/step<N>/hint; without it at steps 100 (8) and 300 (16 + T=0.1), $E/<run>/step<N>/std.
#    Step 300 is copied to /ckpt/<run>/checkpoint-300 while /ckpt has 9 GB free; the other checkpoints are deleted.
#
# DRY=1 R=<dir> prints the training commands and runs the evals with the fake backend (which never answers -1, so
# the probe aborts; a dry run then continues with c2/c4 as stand-ins to test the rest).
set -uo pipefail
DRY=${DRY:-0}
SEEDS=${SEEDS:-"1 2"}
CELLS=${CELLS:-"strong_q09 strong_q00 weak_q09 weak_q00 strong_q06 weak_q06"}
STEPS="25 50 100 150 200 300"
PY="python -m experiments.atom_decomposition"
HM="python -m experiments.atom_decomposition.hint"
if [ "$DRY" = 1 ]; then
    R=${R:?set R=<scratch dir> for a dry run}; CK=$R/ephemeral; KEEP=$R/ckpt; SFT=none
    BACKEND="--backend fake --fake_p_correct 0.6"; N=20; NP=30
    J() { local name=$1; shift 2; echo "[$name] $*"; case "$1" in python) [ "$2" = train_grpo.py ] || "$@";; *) "$@";; esac; }
else
    source /data/env.sh
    R=/data/rlvr/batch4; CK=/ephemeral/batch4; KEEP=/ckpt; SFT=/data/rlvr/sft_qwen3_4b
    sudo mkdir -p $CK && sudo chown "$USER" $CK
    BACKEND="--backend vllm"; N=500; NP=1000
    J() { bash scripts/vm/run_job.sh "$@"; }
fi
E=$R/eval; P=$R/probe; CFG=$R/configs
STD="full --tasks basic_arithmetic --n $N --seed 9999 --k 8 $BACKEND"
STD_LAST="full --tasks basic_arithmetic --n $N --seed 9999 --k 16 --t01 $BACKEND"
PROBE="full --tasks basic_arithmetic --n $NP --seed 7777 --k 8 $BACKEND"
mkdir -p "$R" "$E" "$P" "$CFG" "$CK" "$KEEP"
DONE=$R/batch4.done
failed=()

finish() {
    printf 'failed=%s\nend=%s\n' "${failed[*]:-none}" "$(date -u +%FT%TZ)" > "$DONE"
    [ "$DRY" = 1 ] && return
    nvidia-smi > /data/logs/nvidia_smi_end_batch4.txt 2>&1 || true
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
cand() { python -c "from experiments.atom_decomposition.hint import CANDIDATES; print(CANDIDATES['$1'])"; }
chosen() { python -c "import json; print(json.load(open('$R/chosen.json'))['$1']['text'])"; }

# --- 1. probe ---
for key in c1 c2 c3 c4 c5 c6; do
    ev probe_$key 20m $P/$key $PY.eval_atoms $PROBE --model $SFT --hint "$(cand $key)"
done
if ! J hint_select 10m $HM select $P $R/chosen.json; then
    if [ "$DRY" = 1 ]; then
        echo "[dry] probe aborted as expected with the fake backend; using c2/c4 as stand-ins"
        python -c "import json; from experiments.atom_decomposition.hint import CANDIDATES as C
json.dump({'weak': {'key': 'c2', 'text': C['c2'], 'rate_big': 0.03}, 'strong': {'key': 'c4', 'text': C['c4'], 'rate_big': 0.15}}, open('$R/chosen.json', 'w'))"
    else
        failed+=(probe_abort); echo "probe aborted: $(cat $R/chosen.json 2>/dev/null | grep -m1 abort)"; exit 1
    fi
fi

# --- 2. configs ---
for cell in $CELLS; do
    h=${cell%_q*}; q=${cell#*_q}
    $HM config experiments/atom_decomposition/configs/arith_planted_q$q.yaml $R/chosen.json $h $CFG/hint_$cell.yaml \
        || { failed+=(config_$cell); exit 1; }
done

# --- 3. starting rate: SFT with each chosen hint ---
for h in weak strong; do
    ev eval_sft_hint_$h 30m $E/sft/step000/hint_$h $PY.eval_atoms $STD_LAST --model $SFT --hint "$(chosen $h)"
done

# --- 4. runs ---
for seed in $SEEDS; do
    for cell in $CELLS; do
        run=hint_${cell}_seed$seed; h=${cell%_q*}; text=$(chosen $h)
        # a rerun (e.g. after a lost VM) skips runs whose step-300 evals are done
        [ -f $E/$run/step300/hint/manifest.json ] && [ -f $E/$run/step300/std/manifest.json ] && continue
        if [ ! -f $CK/$run/checkpoint-300/config.json ] && [ ! -f $KEEP/$run/checkpoint-300/config.json ]; then
            J train_$run 4h python train_grpo.py $CFG/hint_$cell.yaml --seed $seed --output_dir $CK/$run \
                || failed+=(train_$run)
        fi
        mkdir -p $R/$run
        cp $CK/$run/trainer_state.json $CK/$run/effective_config.json $R/$run/ 2>/dev/null || true
        cp -r $CK/$run/generations $R/$run/ 2>/dev/null || true
        [ -f $R/$run/effective_config.json ] && { python scripts/vm/check_effective_config.py \
            $R/$run/effective_config.json > $R/$run/effective_config_check.txt 2>&1 || failed+=(cfgcheck_$run); }
        for n in $STEPS; do
            ck=$CK/$run/checkpoint-$n
            [ -d $ck ] || ck=$KEEP/$run/checkpoint-$n
            [ "$DRY" = 1 ] || [ -d $ck ] || { failed+=(missing_$run-$n); continue; }
            args=$STD; [ $n = 300 ] && args=$STD_LAST
            ev eval_${run}_hint_$n 40m $E/$run/step$n/hint $PY.eval_atoms $args --model $ck --hint "$text"
            case $n in 100|300)
                ev eval_${run}_std_$n 40m $E/$run/step$n/std $PY.eval_atoms $args --model $ck ;;
            esac
        done
        # keep step 300 on the durable volume while it has room, drop the rest
        if [ "$DRY" != 1 ] && [ -d $CK/$run/checkpoint-300 ]; then
            rm -rf $CK/$run/checkpoint-{25,50,100,150,200}
            avail=$(df -BG --output=avail $KEEP | tail -1 | tr -dc 0-9)
            if [ "${avail:-0}" -lt 9 ]; then
                echo "$run: /ckpt has ${avail}G free; step 300 not kept" | tee -a $R/not_kept.txt
                rm -rf $CK/$run/checkpoint-300
            elif mkdir -p $KEEP/$run && cp -r $CK/$run/checkpoint-300 $KEEP/$run/ && sync \
                    && [ -f $KEEP/$run/checkpoint-300/config.json ]; then
                rm -rf $CK/$run/checkpoint-300
            else
                failed+=(keep_$run)   # step 300 stays on /ephemeral
            fi
        fi
    done
done
echo "BATCH4 DONE (failed: ${failed[*]:-none})"
