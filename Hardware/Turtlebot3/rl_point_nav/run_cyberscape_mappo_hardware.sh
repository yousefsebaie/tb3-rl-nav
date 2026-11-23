#!/usr/bin/env bash
# Full chain on the two real Burgers: CyberScape two-robot plan -> HITL approval -> MAPPO V5 + gap controller.
# The hardware twin of run_cyberscape_mappo_demo.sh (run on the Mac, both robots on and on the hotspot).
#
#   ./run_cyberscape_mappo_hardware.sh BURGER1_IP BURGER2_IP                 hand-written plan (Plans/rl_mappo_hardware_plan.json)
#   ./run_cyberscape_mappo_hardware.sh BURGER1_IP BURGER2_IP PLAN_JSON       another MULTI_TURTLEBOT plan
#   ./run_cyberscape_mappo_hardware.sh BURGER1_IP BURGER2_IP --live          CyberScape's LLM pipeline plans
#       TURTLEBOT3_1 and TURTLEBOT3_2 (generate from mission_files/rl_mappo_hardware_mission.txt in the dashboard)
#
# Place the robots facing each other (caster sides towards each other) about 2.0 m apart, nothing
# else within ~1 m between them. The script:
#   1) copies the MAPPO code and model to both Pis, starts bringup where it is not running;
#   2) measures where the robots stand from what each LiDAR sees of the other (no tape measure):
#      Burger 1 = mission origin facing +x, Burger 2 = wherever its LiDAR says;
#   3) starts mappo_agent.py on both Pis (pose exchange, MAPPO, gap controller);
#   4) starts the CyberScape HITL server (port 5001) with the plan, and the hardware sender as the
#      post-approval hook: open http://localhost:5001/dashboard/index.html and press Approve.
# Survivor stand-ins (camera stand-in, as in simulation): SURVIVORS="1.6,-1.1 0.4,-1.1" by default.
# Emergency stop: curl -X POST http://BURGER_IP:8765/stop   (or switch the robot off)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
DEMO_DIR="$HOME/cyberscape_mappo_hw"
B1="${1:?usage: $0 BURGER1_IP BURGER2_IP [PLAN_JSON|--live]}"
B2="${2:?usage: $0 BURGER1_IP BURGER2_IP [PLAN_JSON|--live]}"
PLAN="${3:-$REPO/Plans/rl_mappo_hardware_plan.json}"
[ "$PLAN" = "--live" ] || PLAN="$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")"
SURVIVORS="${SURVIVORS:-1.6,-1.1 0.4,-1.1}"
ROS='source /opt/ros/humble/setup.bash; source ~/turtlebot3_ws/install/setup.bash; export TURTLEBOT3_MODEL=burger LDS_MODEL=LDS-02;'
FILES="config.py policy_numpy.py lidar_sectors.py route_planner.py ros_point_nav_env.py ros_point_nav_lidar_env.py
       hardware_rl_bridge.py cyberscape_mappo_mission.py mappo_hardware.py ros_mappo_env.py mappo_drive.py
       mappo_agent.py mappo_dry_run.py mappo_find_teammate.py"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
domain() { [ "$1" = "$B1" ] && echo 30 || echo 31; }

say "1) Code, model and bringup on both Burgers"
for ip in "$B1" "$B2"; do
    ssh -o ConnectTimeout=8 "ubuntu@$ip" "mkdir -p ~/rl_point_nav/models/mappo_v5_kl05"
    (cd "$HERE" && scp -q $FILES "ubuntu@$ip:rl_point_nav/")
    scp -q "$HERE/models/mappo_v5_kl05/best_model.npz" "ubuntu@$ip:rl_point_nav/models/mappo_v5_kl05/"
    ssh "ubuntu@$ip" "pkill -f mappo_agent.py || true; $ROS export ROS_DOMAIN_ID=$(domain "$ip");
        pgrep -x turtlebot3_ros >/dev/null || { nohup ros2 launch turtlebot3_bringup robot.launch.py > /tmp/bringup.log 2>&1 < /dev/null & sleep 20; };
        ros2 service call /motor_power std_srvs/srv/SetBool '{data: true}' > /dev/null 2>&1 || true;
        echo \"\$(hostname): \$(timeout 10 ros2 topic echo --once /sensor_state 2>/dev/null | grep -E 'torque|battery' | tr '\n' ' ')\""
done

say "2) Where the robots stand (each LiDAR looks for the other)"
SEEN1="$(ssh "ubuntu@$B1" "$ROS export ROS_DOMAIN_ID=30; cd ~/rl_point_nav && python3 mappo_find_teammate.py")"
SEEN2="$(ssh "ubuntu@$B2" "$ROS export ROS_DOMAIN_ID=31; cd ~/rl_point_nav && python3 mappo_find_teammate.py")"
echo "Burger 1 sees: $SEEN1"
echo "Burger 2 sees: $SEEN2"
read -r START1 START2 < <(cd "$HERE" && python3 - "$SEEN1" "$SEEN2" <<'EOF'
import json, math, sys
from mappo_hardware import head_on_starts
a, b = (json.loads(s) for s in sys.argv[1:3])
s1, s2 = head_on_starts((a["range_m"], a["bearing_deg"]), (b["range_m"], b["bearing_deg"]))
f = lambda s: f"{s[0]:.3f},{s[1]:.3f},{math.degrees(s[2]):.1f}"
print(f(s1), f(s2))
EOF
)
echo "Burger 1 (TurtleBot3_1) base A: ${START1//,/ }   Burger 2 (TurtleBot3_2) base B: ${START2//,/ }   (x m, y m, heading deg)"
if [ -t 0 ]; then read -r -p "Does that match the floor (Burger 2 about 2 m straight ahead of Burger 1)? Enter = yes, Ctrl+C = no " _; fi

say "3) MAPPO agents on both Burgers"
ssh "ubuntu@$B1" "$ROS export ROS_DOMAIN_ID=30; cd ~/rl_point_nav && nohup python3 mappo_agent.py --robot TURTLEBOT3_1 \
    --start ${START1//,/ } --peer $B2 --survivors $SURVIVORS > /tmp/mappo_agent.log 2>&1 < /dev/null &"
ssh "ubuntu@$B2" "$ROS export ROS_DOMAIN_ID=31; cd ~/rl_point_nav && nohup python3 mappo_agent.py --robot TURTLEBOT3_2 \
    --start ${START2//,/ } --peer $B1 --survivors $SURVIVORS > /tmp/mappo_agent.log 2>&1 < /dev/null &"
for ip in "$B1" "$B2"; do
    for _ in $(seq 1 60); do curl -s -m 2 "http://$ip:8765/status" > /dev/null && break; sleep 1; done
    echo "$ip: $(curl -s -m 2 "http://$ip:8765/status" || echo 'NOT ANSWERING, see /tmp/mappo_agent.log on the robot')"
done

say "4) CyberScape server"
mkdir -p "$DEMO_DIR"
for entry in "$REPO"/* "$REPO"/.[!.]*; do
    name="$(basename "$entry")"
    [ "$name" = config.json ] || [ "$name" = .git ] || ln -sfn "$entry" "$DEMO_DIR/$name"
done
python3 - "$REPO/config.json" "$DEMO_DIR/config.json" "$PLAN" <<'EOF'
import json, os, sys
config = json.load(open(sys.argv[1]))
if sys.argv[3] == "--live":
    # Same as run_cyberscape_mappo_demo.sh --live: two robots, each planned by the normal pipeline.
    config["robots_in_curr_mission"] = ["TURTLEBOT3_1", "TURTLEBOT3_2"]
    for k in (1, 2):
        robot = dict(config["robots_config"]["TURTLEBOT3"])
        robot.update(high_plan_key=f"turtlebot3_{k}_plan", instruction_format=f"TurtleBot3_{k}.instruction()",
                     temp_low=f"Plans/intermediate/temp_tb3_{k}_low_level_plan.txt",
                     verified=f"Plans/intermediate/verified_tb3_{k}_plan.txt",
                     parsed=f"Plans/intermediate/parsed_tb3_{k}_verified_plan.txt",
                     final_low=f"Plans/rl_mappo_hw_live_tb3_{k}_plan.json")
        config["robots_config"][f"TURTLEBOT3_{k}"] = robot
        path = os.path.join(os.path.dirname(sys.argv[2]), robot["final_low"])
        if os.path.exists(path):
            os.remove(path)
    config["mission_plan_file"] = "Plans/intermediate/rl_mappo_hw_live_mission_plan.json"
    config["plan_parser_llm"] = "cohere"
else:
    config["robots_in_curr_mission"] = ["MULTI_TURTLEBOT"]
    config["robots_config"].setdefault("MULTI_TURTLEBOT", {})["final_low"] = sys.argv[3]
json.dump(config, open(sys.argv[2], "w"), indent=4)
EOF
pgrep -f "hitl-comm-server.py" | xargs kill 2>/dev/null || true
sleep 1
echo
echo "Plan: $PLAN"
echo "Open  http://localhost:5001/dashboard/index.html  and approve. Robots' logs: ssh ubuntu@IP tail -f /tmp/mappo_agent.log"
echo "Sender log (phases): tail -f $DEMO_DIR/cyberscape_sender.log"
echo "EMERGENCY STOP: curl -X POST http://$B1:8765/stop; curl -X POST http://$B2:8765/stop   (or switch the robots off)"
cd "$DEMO_DIR"
source "$REPO/.venv/bin/activate"
export CYBERSCAPE_ISAAC_SENDER="$REPO/Hardware/Turtlebot3/cyberscape_mappo_hw_sender.py"
export CYBERSCAPE_ISAAC_SENDER_LOG="$DEMO_DIR/cyberscape_sender.log"
export CYBERSCAPE_ISAAC_COMMAND_TIMEOUT="${CYBERSCAPE_ISAAC_COMMAND_TIMEOUT:-900}"
export MAPPO_AGENTS="http://$B1:8765,http://$B2:8765"
exec python3 "$REPO/hitl-comm-server.py"
