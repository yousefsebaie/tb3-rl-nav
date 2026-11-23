#!/usr/bin/env python3
"""Print where this Burger's LiDAR sees the other Burger: the nearest object near a bearing.

Robots standing still, nothing else nearby. Median over --scans scans, printed as one JSON
line {"range_m", "bearing_deg", "scans"} for run_cyberscape_mappo_hardware.sh, which turns
both robots' readings into their start poses (mappo_hardware.head_on_starts).

  python3 mappo_find_teammate.py              # the other robot is straight ahead
  python3 mappo_find_teammate.py --toward 90  # ... on the left
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from mappo_hardware import nearest_ahead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--toward", type=float, default=0.0, help="expected bearing of the other robot (deg)")
    parser.add_argument("--cone", type=float, default=30.0, help="search +- this many degrees around it")
    parser.add_argument("--scans", type=int, default=15)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("mappo_find_teammate")
    seen = []

    def on_scan(msg: LaserScan) -> None:
        shift = args.toward * 3.141592653589793 / 180.0
        hit = nearest_ahead(msg.ranges, msg.angle_min - shift, msg.angle_increment, msg.range_min,
                            msg.range_max, args.cone)
        if hit is not None:
            seen.append((hit[0], hit[1] + args.toward))

    node.create_subscription(LaserScan, "/scan", on_scan, qos_profile_sensor_data)
    deadline = time.monotonic() + 15.0
    while len(seen) < args.scans and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if not seen:
        raise SystemExit(json.dumps({"error": "nothing seen in the cone"}))
    print(json.dumps({"range_m": round(statistics.median(r for r, _ in seen), 4),
                      "bearing_deg": round(statistics.median(b for _, b in seen), 2), "scans": len(seen)}))


if __name__ == "__main__":
    main()
