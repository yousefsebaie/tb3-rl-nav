#!/usr/bin/env bash
# Long headless MAPPO run (2 robots, warm-started from V4) with live progress.
#   ./train_mappo_long.sh                     # baseline eval once, then 24 h of MAPPO
#   HOURS=48 ./train_mappo_long.sh            # longer; resumes from the latest checkpoint
#   CRITIC=local ./train_mappo_long.sh        # IPPO baseline (models/ippo_v4)
#   TASK=coop START=models/mappo_v4/checkpoint_160000_steps.pt ./train_mappo_long.sh
#                                             # MAPPO v2 (models/mappo_v4_coop, best_model.pt kept)
#   TASK=coop3 HOURS=12 START=models/mappo_v4_coop/best_model.pt ./train_mappo_long.sh
#                                             # MAPPO v3 (models/mappo_v4_coop3)
# Training runs detached (survives closing the terminal). Ctrl-C closes only the progress view.
# Stop early (saves a checkpoint): pkill -INT -f "kit/python/bin/python3 train_mappo.py"
cd "$(dirname "$0")"
CRITIC="${CRITIC:-central}"
TASK="${TASK:-base}"
SUFFIX=$([ "$TASK" != base ] && echo "_$TASK")
NAME="${NAME:-$([ "$CRITIC" = central ] && echo mappo_v4 || echo ippo_v4)$SUFFIX}"
HOURS="${HOURS:-24}"
TIMESTEPS="${TIMESTEPS:-1000000}"      # upper bound; the timer normally ends the run
LR="${LR:-$([ "$TASK" != base ] && echo 5e-5 || echo 1e-4)}"
EVAL_EVERY="${EVAL_EVERY:-$([ "$TASK" != base ] && echo 25000 || echo 0)}"
export TB3_SCENE_USD="${TB3_SCENE_USD:-$HOME/Desktop/FInalSceneDemo_realsize.usd}"
# coop and coop3 differ only in reward, so an untrained policy has the same outcomes: one baseline.
BASELINE_LOG="results/mappo_baseline_v4_$([ "$TASK" != base ] && echo coop || echo encounters).log"
LOG="results/train_${NAME}.log"
PATTERN="kit/python/bin/python3 train_mapp[o].py"
mkdir -p results "models/$NAME"

if pgrep -f "$PATTERN" > /dev/null; then
    echo "MAPPO is already running; showing its progress."
else
    LATEST=$(ls -v models/"$NAME"/checkpoint_*_steps.pt 2>/dev/null | tail -1)
    RESUME=()
    [ -n "$LATEST" ] && RESUME=(--resume "$LATEST") && echo "Resuming from $LATEST"
    [ -z "$LATEST" ] && [ -n "$START" ] && RESUME=(--resume "$START") && echo "Starting from $START"
    [ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d_%H%M%S)"
    setsid nohup bash -c "
        if ! grep -q 'EVAL 100 episodes' '$BASELINE_LOG' 2>/dev/null; then
            ./isaac_run.sh train_mappo.py --task $TASK --eval-episodes 100 --seed 1000 > '$BASELINE_LOG' 2>&1
        fi
        ( sleep $((HOURS * 3600)); pgrep -f '$PATTERN' | xargs -r kill -INT ) &
        ./isaac_run.sh train_mappo.py --critic $CRITIC --task $TASK --output-name $NAME --timesteps $TIMESTEPS \
            --learning-rate $LR --eval-every $EVAL_EVERY ${RESUME[*]} $TRAIN_ARGS
    " > "$LOG" 2>&1 < /dev/null &
    disown
    echo "Started: baseline eval (if not done, ~1.5 h, log $BASELINE_LOG), then $HOURS h of $CRITIC-critic training -> models/$NAME"
fi
echo "Waiting for training to start ..."
until grep -q "TB03_PROGRESS" "$LOG" 2>/dev/null; do
    [ -f "$BASELINE_LOG" ] && grep "TB03_MAPPO\] episode" "$BASELINE_LOG" | tail -1 | sed 's/^/baseline: /'
    sleep 30
done
exec python3 training_progress.py "$LOG" --models "models/$NAME"
