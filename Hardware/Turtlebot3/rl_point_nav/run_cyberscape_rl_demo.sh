#!/usr/bin/env bash
# Full chain demo: CyberScape plan -> HITL approval -> plan sender -> RL policy in Isaac Sim.
#
#   ./run_cyberscape_rl_demo.sh [PLAN_JSON]      (default: Plans/rl_demo_burger_plan.json)
#   ./run_cyberscape_rl_demo.sh --live           (no plan: generate one with the LLM pipeline,
#                                                 see run_cyberscape_full_cycle.sh)
#
# 1) starts Isaac Sim with cyberscape_rl_bridge.py (waits for commands)
# 2) starts the CyberScape HITL server (port 5001) with the plan pre-loaded for review
# 3) open http://localhost:5001/dashboard/index.html and press Approve:
#    the server launches the plan sender and the RL policy executes each command.
#
# The server runs from ~/cyberscape_rl_demo: links to every project file plus its own
# config.json (TURTLEBOT3 only, plan parser on Cohere), so the project's config.json and
# tracked plan files are not changed. Ctrl-C stops the server.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
DEMO_DIR="$HOME/cyberscape_rl_demo"
BRIDGE_ARGS="${BRIDGE_ARGS:---spin-guard}"   # BRIDGE_ARGS="" for pure RL
if [ "${1:-}" = "--live" ]; then
    PLAN="Plans/rl_live_burger_plan.json"     # written by the LLM pipeline
    rm -f "$REPO/$PLAN"                          # nothing pre-loaded: generate first
else
    PLAN="$(realpath "${1:-$REPO/Plans/rl_demo_burger_plan.json}")"
fi
mkdir -p "$DEMO_DIR" "$HERE/results"
for entry in "$REPO"/* "$REPO"/.[!.]*; do
    name="$(basename "$entry")"
    [ "$name" = config.json ] || [ "$name" = .git ] || ln -sfn "$entry" "$DEMO_DIR/$name"
done

python3 - "$REPO/config.json" "$DEMO_DIR/config.json" "$PLAN" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
config["robots_in_curr_mission"] = ["TURTLEBOT3"]
config["robots_config"]["TURTLEBOT3"]["final_low"] = sys.argv[3]
config["mission_plan_file"] = "Plans/intermediate/rl_live_mission_plan.json"
config["plan_parser_llm"] = "cohere"
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
    setsid nohup ./isaac_run.sh cyberscape_rl_bridge.py $BRIDGE_ARGS > results/bridge.log 2>&1 < /dev/null &)
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
