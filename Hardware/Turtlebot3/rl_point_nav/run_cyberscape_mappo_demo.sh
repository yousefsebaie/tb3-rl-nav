#!/usr/bin/env bash
# Full chain, two robots: CyberScape MULTI_TURTLEBOT plan -> HITL approval -> MAPPO in Isaac Sim.
#
#   ./run_cyberscape_mappo_demo.sh                      disaster: earthquake search and rescue
#   ./run_cyberscape_mappo_demo.sh disaster_simple      shorter disaster: one doorway, 3 survivors
#   ./run_cyberscape_mappo_demo.sh warehouse            station swap + storage-room doorway
#   ./run_cyberscape_mappo_demo.sh disaster PLAN_JSON   another plan on that layout
#   VIEW=angled ./run_cyberscape_mappo_demo.sh          angled camera instead of overhead
#   ./run_cyberscape_mappo_demo.sh disaster --live      nothing pre-loaded: CyberScape's LLM pipeline
#       plans TURTLEBOT3_1 and TURTLEBOT3_2 from a mission text (run_cyberscape_mappo_full_cycle.sh)
#
# 1) starts Isaac Sim with train_mappo.py --mission-listen (V5 MAPPO + gap controller, GUI)
# 2) starts the CyberScape HITL server (port 5001) with the plan pre-loaded for review
# 3) open http://localhost:5001/dashboard/index.html and press Approve: the server launches
#    cyberscape_mappo_plan_sender.py, which sends one phase at a time; both robots drive together.
#
# Like run_cyberscape_rl_demo.sh, the server runs from its own folder (~/cyberscape_mappo_demo)
# with its own config.json (MULTI_TURTLEBOT only), so the project's config.json is not changed.
# Without the dashboard:  ./isaac_run.sh train_mappo.py --resume models/mappo_v5_kl05/best_model.pt \
#     --gap-assist --scenario disaster --mission ../../../Plans/rl_mappo_disaster_plan.json --gui --show-robots
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
DEMO_DIR="$HOME/cyberscape_mappo_demo"
SCENARIO="${1:-disaster}"
case "$SCENARIO" in
    disaster)  DEFAULT_PLAN="$REPO/Plans/rl_mappo_disaster_plan.json" ;;
    disaster_simple) DEFAULT_PLAN="$REPO/Plans/rl_mappo_disaster_simple_plan.json" ;;
    warehouse) DEFAULT_PLAN="$REPO/Plans/rl_mappo_demo_plan.json" ;;
    *) echo "unknown scenario '$SCENARIO' (disaster, disaster_simple or warehouse)"; exit 1 ;;
esac
if [ "${2:-}" = "--live" ]; then
    PLAN=--live
else
    PLAN="$(realpath "${2:-$DEFAULT_PLAN}")"
fi
mkdir -p "$DEMO_DIR" "$HERE/results"
for entry in "$REPO"/* "$REPO"/.[!.]*; do
    name="$(basename "$entry")"
    [ "$name" = config.json ] || [ "$name" = .git ] || ln -sfn "$entry" "$DEMO_DIR/$name"
done

python3 - "$REPO/config.json" "$DEMO_DIR/config.json" "$PLAN" <<'EOF'
import json, sys
import os
config = json.load(open(sys.argv[1]))
if sys.argv[3] == "--live":
    # Two CyberScape robots, each planned by the normal pipeline (high level, low level,
    # verification, parsing) with its own files; nothing pre-loaded until /generate_plan runs.
    config["robots_in_curr_mission"] = ["TURTLEBOT3_1", "TURTLEBOT3_2"]
    for k in (1, 2):
        robot = dict(config["robots_config"]["TURTLEBOT3"])
        robot.update(high_plan_key=f"turtlebot3_{k}_plan", instruction_format=f"TurtleBot3_{k}.instruction()",
                     temp_low=f"Plans/intermediate/temp_tb3_{k}_low_level_plan.txt",
                     verified=f"Plans/intermediate/verified_tb3_{k}_plan.txt",
                     parsed=f"Plans/intermediate/parsed_tb3_{k}_verified_plan.txt",
                     final_low=f"Plans/rl_mappo_live_tb3_{k}_plan.json")
        config["robots_config"][f"TURTLEBOT3_{k}"] = robot
        path = os.path.join(os.path.dirname(sys.argv[2]), robot["final_low"])
        if os.path.exists(path):
            os.remove(path)
    config["mission_plan_file"] = "Plans/intermediate/rl_mappo_live_mission_plan.json"
    config["plan_parser_llm"] = "cohere"
else:
    config["robots_in_curr_mission"] = ["MULTI_TURTLEBOT"]
    config["robots_config"].setdefault("MULTI_TURTLEBOT", {})["final_low"] = sys.argv[3]
json.dump(config, open(sys.argv[2], "w"), indent=4)
EOF
rm -f "$HOME/cyberscape_command.txt"

# Fresh bridge and server every time (an old one would hold the command file / port).
pgrep -f "^$HOME/isaac-sim/kit/python/bin/python3 train_mappo.py .*--mission-listen" | xargs -r kill -9 || true
pgrep -f "^python3 $REPO/hitl-comm-server.py" | xargs -r kill || true
sleep 2
rm -f "$HERE/results/mappo_bridge.log"
echo "Starting Isaac Sim + MAPPO bridge (window opens in 1-2 min) ..."
(cd "$HERE" && TB3_SCENE_USD="${TB3_SCENE_USD:-$HOME/Desktop/FInalSceneDemo_realsize.usd}" \
    setsid nohup ./isaac_run.sh train_mappo.py --resume models/mappo_v5_kl05/best_model.pt --gap-assist \
    --scenario "$SCENARIO" --mission-listen --gui --show-robots > results/mappo_bridge.log 2>&1 < /dev/null &)
echo "Waiting for the bridge to be ready ..."
until grep -q "waiting for CyberScape commands" "$HERE/results/mappo_bridge.log" 2>/dev/null; do
    if grep -q "^Traceback" "$HERE/results/mappo_bridge.log" 2>/dev/null \
            && ! grep -q "omni.kit.menu" "$HERE/results/mappo_bridge.log"; then
        echo "Bridge failed, see $HERE/results/mappo_bridge.log"; exit 1
    fi
    sleep 3
done

echo
echo "Scenario: $SCENARIO   Plan: $PLAN"
echo "Open  http://localhost:5001/dashboard/index.html  and approve the plan."
echo "Bridge log (what the robots do):  tail -f $HERE/results/mappo_bridge.log | grep CYBERSCAPE_MAPPO"
echo
cd "$DEMO_DIR"
export CYBERSCAPE_ISAAC_SENDER="$REPO/Hardware/Isaac/cyberscape_mappo_plan_sender.py"
export CYBERSCAPE_ISAAC_COMMAND_TIMEOUT="${CYBERSCAPE_ISAAC_COMMAND_TIMEOUT:-900}"
exec python3 "$REPO/hitl-comm-server.py"
