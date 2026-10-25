#!/usr/bin/env bash
# Start (or continue) a long headless PPO run on the real-size scene and show live progress.
#   ./train_long.sh                  # 1,000,000 steps, resumes from the latest checkpoint if one exists
#   TIMESTEPS=2000000 ./train_long.sh
# Training runs detached (survives closing the terminal / VS Code).
# Ctrl-C closes only the progress view. Stop training: pkill -f "kit/python/bin/python3 train_isaac"
cd "$(dirname "$0")"
TIMESTEPS="${TIMESTEPS:-1000000}"
NAME="${NAME:-isaac_point_nav_lidar_ppo_v3_realsize}"
SCENE="${TB3_SCENE_USD:-$HOME/Desktop/FInalSceneDemo_realsize.usd}"
LOG="results/train_${NAME}.log"
mkdir -p results "models/$NAME"

if pgrep -f "kit/python/bin/python3 train_isaac_point_nav_lidar_ppo_v2.py" > /dev/null; then
    echo "Training is already running; showing its progress."
else
    LATEST=$(ls -v models/"$NAME"/checkpoint_*_steps.zip 2>/dev/null | tail -1)
    RESUME=()
    if [ -n "$LATEST" ]; then
        RESUME=(--resume "$LATEST")
        echo "Resuming from $LATEST"
    fi
    [ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d_%H%M%S)"
    TB3_SCENE_USD="$SCENE" setsid nohup ./isaac_run.sh train_isaac_point_nav_lidar_ppo_v2.py \
        --timesteps "$TIMESTEPS" --output-name "$NAME" "${RESUME[@]}" > "$LOG" 2>&1 < /dev/null &
    disown
    echo "Started: $TIMESTEPS steps -> models/$NAME   (log: $LOG)"
    sleep 2
fi
exec python3 training_progress.py "$LOG" --models "models/$NAME"
