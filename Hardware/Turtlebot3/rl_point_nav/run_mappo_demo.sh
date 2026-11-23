#!/usr/bin/env bash
# GUI demo: V5 MAPPO + gap controller over DEMO_PLAYLIST (crossing, head-on, random, doorway, corridor).
#   ./run_mappo_demo.sh                 # 2 rounds (10 episodes), overhead camera per episode
#   VIEW=angled ./run_mappo_demo.sh     # fixed angled view of the whole area
#   ROUNDS=1 SEEDS=5001,5000,5002,5003,5000 ./run_mappo_demo.sh   # one seed per playlist entry
cd "$(dirname "$0")"
export TB3_SCENE_USD="${TB3_SCENE_USD:-$HOME/Desktop/FInalSceneDemo_realsize.usd}"
ROUNDS="${ROUNDS:-2}"
exec ./isaac_run.sh train_mappo.py --resume models/mappo_v5_kl05/best_model.pt --gap-assist --demo \
    --eval-episodes $((ROUNDS * 5)) --seed "${SEED:-5000}" ${SEEDS:+--demo-seeds "$SEEDS"} --gui --show-robots
