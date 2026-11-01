#!/usr/bin/env bash
# Experimental design comparison (slide "Experimental Design & Quantitative Comparison"):
#   B0 existing system (LLM plan + original move_to_point, no RL) vs B1 heuristic vs PPO.
# Episode conditions (nominal / generalization / robustness) run in parallel, then the
# CyberScape missions (patrol, disaster) per controller. Results: results/timing/cmp_<stamp>_*,
# results/missions/missions.csv. Report: make_comparison_report.py.
set -uo pipefail
cd "$(dirname "$0")"
export TB3_SCENE_USD="${TB3_SCENE_USD:-$HOME/Desktop/FInalSceneDemo_realsize.usd}"
STAMP="cmp_$(date +%Y%m%d_%H%M%S)"
echo "$STAMP" > results/latest_comparison.txt
./isaac_run.sh eval_timing.py --condition nominal --episodes "${NOMINAL:-30}" --run-name "${STAMP}_nominal" > "results/${STAMP}_nominal.log" 2>&1 &
sleep 60
./isaac_run.sh eval_timing.py --condition generalization --episodes "${GENERAL:-20}" --run-name "${STAMP}_generalization" > "results/${STAMP}_generalization.log" 2>&1 &
sleep 60
./isaac_run.sh eval_timing.py --condition robustness --episodes "${ROBUST:-20}" --run-name "${STAMP}_robustness" > "results/${STAMP}_robustness.log" 2>&1 &
sleep 60
for scenario in patrol disaster; do
    plan=../../../Plans/rl_demo_burger_plan.json
    [ "$scenario" = disaster ] && plan=../../../Plans/rl_disaster_burger_plan.json
    for controller in b0 rule ppo; do
        ./isaac_run.sh cyberscape_rl_bridge.py --headless --plan "$plan" --scenario "$scenario" \
            --controller "$controller" --spin-guard 2>&1 | grep "MISSION" >> "results/${STAMP}_missions.log"
    done
done
wait
echo "done $STAMP"
