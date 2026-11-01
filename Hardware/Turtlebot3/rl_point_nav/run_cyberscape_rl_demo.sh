#!/usr/bin/env bash
# Full chain demo: CyberScape plan -> HITL approval -> plan sender -> RL policy in Isaac Sim.
#
#   ./run_cyberscape_rl_demo.sh [PLAN_JSON]      (default: Plans/rl_demo_burger_plan.json)
#
# 1) starts Isaac Sim with cyberscape_rl_bridge.py (waits for commands)
# 2) starts the CyberScape HITL server (port 5001) with the plan pre-loaded for review
# 3) open http://localhost:5001/dashboard/index.html and press Approve:
#    the server launches the plan sender and the RL policy executes each command.
#
# The server runs from ~/cyberscape_rl_demo with a copy of config.json that only lists
# TURTLEBOT3, so the project's config.json is not changed. Ctrl-C stops the server.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PLAN="$(realpath "${1:-$REPO/Plans/rl_demo_burger_plan.json}")"
DEMO_DIR="$HOME/cyberscape_rl_demo"
mkdir -p "$DEMO_DIR" "$HERE/results"

python3 - "$REPO/config.json" "$DEMO_DIR/config.json" "$PLAN" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
config["robots_in_curr_mission"] = ["TURTLEBOT3"]
config["robots_config"]["TURTLEBOT3"]["final_low"] = sys.argv[3]
json.dump(config, open(sys.argv[2], "w"), indent=4)
EOF
rm -f "$HOME/cyberscape_command.txt"

# Fresh bridge and server every time (an old one would hold the command file / port).
pgrep -f "^$HOME/isaac-sim/kit/python/bin/python3 cyberscape_rl_bridge.py" | xargs -r kill -9 || true
pgrep -f "^python3 $REPO/hitl-comm-server.py" | xargs -r kill || true
sleep 2
rm -f "$HERE/results/bridge.log"
echo "Starting Isaac Sim + RL bridge (window opens in 1-2 min) ..."
(cd "$HERE" && TB3_SCENE_USD="${TB3_SCENE_USD:-$HOME/Desktop/FInalSceneDemo_realsize.usd}" \
    setsid nohup ./isaac_run.sh cyberscape_rl_bridge.py > results/bridge.log 2>&1 < /dev/null &)
echo "Waiting for the bridge to be ready ..."
until grep -q "waiting for CyberScape commands" "$HERE/results/bridge.log" 2>/dev/null; do
    if grep -q Traceback "$HERE/results/bridge.log" 2>/dev/null; then
        echo "Bridge failed, see $HERE/results/bridge.log"; exit 1
    fi
    sleep 3
done

echo
echo "Plan: $PLAN"
echo "Open  http://localhost:5001/dashboard/index.html  and approve the plan."
echo "Bridge log (what the robot does):  tail -f $HERE/results/bridge.log | grep CYBERSCAPE_RL"
echo
cd "$DEMO_DIR"
# The server launches the plan sender after approval; RL legs in the GUI take minutes.
export CYBERSCAPE_ISAAC_COMMAND_TIMEOUT="${CYBERSCAPE_ISAAC_COMMAND_TIMEOUT:-900}"
exec python3 "$REPO/hitl-comm-server.py"
