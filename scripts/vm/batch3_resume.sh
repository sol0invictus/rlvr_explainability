#!/usr/bin/env bash
# Batch 3, second pass. The first pass skipped all six capped runs: their time limit was written "2h30m", which
# timeout rejects (exit 125 at 0 s), so the batch ran q09 and zebra seed 2 only. This waits for that pass to end
# (batch3.done), stops its finish trap before it deletes the VM, pulls the fixed batch3.sh and runs the capped runs
# in the section 11 order (run 2, then 4, then 3). That pass ends by deleting the VM as usual.
#
# Copy outside the repo (so git pull is clean) and run on the VM:
#   tmux new -d -s resume 'bash ~/batch3_resume.sh <pid of the first pass>'
set -uo pipefail
P=${1:?pid of the running bash scripts/vm/batch3.sh}
R=/data/rlvr/batch3
cd /data/rlvr_experiments
grep -q 'batch3.sh' /proc/$P/cmdline || { echo "pid $P is not batch3.sh"; exit 1; }
echo "$(date -u +%FT%TZ) waiting for $R/batch3.done (first pass pid $P)"
until [ -f $R/batch3.done ]; do sleep 5; done
# kill the first pass's bash before its children, so its finish trap cannot go on to self_delete.sh
C=$(pgrep -P $P)
kill -KILL $P; [ -n "$C" ] && kill -KILL $C; sleep 2
if kill -0 $P 2>/dev/null; then echo "could not stop pid $P; leaving it to delete the VM"; exit 1; fi
tmux kill-session -t batch3 2>/dev/null || true
cp $R/batch3.done $R/batch3_part1.done
echo "$(date -u +%FT%TZ) first pass stopped: $(tr '\n' ' ' < $R/batch3_part1.done)"
git pull --ff-only || true
grep -q 'limit=150m' scripts/vm/batch3.sh || sed -i 's/limit=2h30m/limit=150m/' scripts/vm/batch3.sh
if ! grep -q 'limit=150m' scripts/vm/batch3.sh; then echo "fix not in place; deleting the VM"; cd ~; bash ~/self_delete.sh; exit 1; fi
git log --oneline -1
RUNS="cap128_q00_seed1 cap128_q00_seed2 cap128_q09_seed1 cap128_q09_seed2 cap128_q06_seed1 cap128_q06_seed2" \
    bash scripts/vm/batch3.sh
