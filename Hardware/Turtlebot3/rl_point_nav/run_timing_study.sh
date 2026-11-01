#!/usr/bin/env bash
# Timing study (Mission-Time & RL Performance Evaluation Framework), single-robot layer:
#   1) mission level: the same frozen CyberScape plan per scenario, executed headless by
#      B1 (rule-based) and PPO behind the same goal interface and safety layer
#   2) episode level: B1 vs PPO on EPISODES seeded scenarios (eval_timing.py)
# then timing_report.py prints the tables.
#
#   ./run_timing_study.sh                 # EPISODES=30 by default
#   EPISODES=50 ./run_timing_study.sh
set -euo pipefail
cd "$(dirname "$0")"
export TB3_SCENE_USD="${TB3_SCENE_USD:-$HOME/Desktop/FInalSceneDemo_realsize.usd}"
EPISODES="${EPISODES:-30}"
RUN="study_$(date +%Y%m%d_%H%M%S)"
PLANS=(patrol:../../../Plans/rl_demo_burger_plan.json disaster:../../../Plans/rl_disaster_burger_plan.json)

for entry in "${PLANS[@]}"; do
    scenario="${entry%%:*}"; plan="${entry#*:}"
    for controller in rule ppo; do
        echo "== mission: $scenario / $controller"
        ./isaac_run.sh cyberscape_rl_bridge.py --headless --plan "$plan" --scenario "$scenario" \
            --controller "$controller" --spin-guard 2>&1 | grep --line-buffered "MISSION\|Traceback" || true
    done
done

echo "== episodes: $EPISODES seeded scenarios, rule vs ppo"
./isaac_run.sh eval_timing.py --episodes "$EPISODES" --run-name "$RUN" 2>&1 | grep --line-buffered "TB03_TIMING\] scenario\|Traceback" || true

python3 timing_report.py --run "results/timing/$RUN"
