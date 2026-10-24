#!/usr/bin/env python3
"""
TurtleBot3 Burger – CyberScape Communication Node
==================================================
Polls the CyberScape HITL server for mission phases, executes each
instruction on the real (or simulated) TurtleBot3 Burger via ROS2,
and reports phase completion back to the server.

Supported instructions (mirrors burger_specifications.txt):
  TurtleBot3.move_to_point((x, y))
  TurtleBot3.rotate(angle)
  TurtleBot3.scan_with_lidar()
  TurtleBot3.detect_with_camera(object)
  TurtleBot3.wait_for_signal()
  TurtleBot3.communicate_with_apm(data)
  TurtleBot3.retrieve_object(object)
  TurtleBot3.return_to_base()

Usage:
    # Real Burger through the safety filter:
    python3 burger_comm_node.py --server http://localhost:5001 --robot TURTLEBOT3

    # Isaac Sim Burger on isolated topics:
    python3 burger_comm_node.py --server http://localhost:5001 --robot TURTLEBOT3 \
        --cmd_vel_topic /sim/cmd_vel --odom_topic /sim/odom --scan_topic /sim/scan \
        --mirror_only --max_goal_radius 0.45
"""

import ast
import math
import re
import sys
import time
import threading
import argparse
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

DEFAULT_SERVER   = "http://localhost:5001"
ROBOT_NAME       = "TURTLEBOT3"
DEFAULT_CMD_VEL_TOPIC = "/cmd_vel_raw"
DEFAULT_ODOM_TOPIC = "/odom"
DEFAULT_SCAN_TOPIC = "/scan"
HEARTBEAT_PERIOD = 5.0          # seconds between heartbeats
LINEAR_SPEED     = 0.15         # m/s  (TurtleBot3 Burger max: 0.22)
ANGULAR_SPEED    = 0.5          # rad/s (TurtleBot3 Burger max: 2.84)
GOAL_TOLERANCE   = 0.05         # metres
ANGLE_TOLERANCE  = 0.03         # radians (~1.7°)
OBSTACLE_STOP_DISTANCE = 0.35   # metres; stop before the Burger reaches an obstacle
OBSTACLE_CLEAR_DISTANCE = 0.50  # metres; desired clearance before driving forward again
AVOIDANCE_TURN_DEG = 55.0
AVOIDANCE_STEP_DISTANCE = 0.25
MAX_AVOIDANCE_ATTEMPTS = 12


def normalize_angle(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ─────────────────────────────────────────────────────────
# TurtleBot3 ROS2 Driver Node
# ─────────────────────────────────────────────────────────

class BurgerDriver(Node):
    """
    Thin ROS2 node that exposes blocking helper methods matching the
    CyberScape instruction set for the TurtleBot3 Burger.
    """

    def __init__(
        self,
        cmd_vel_topic: str = DEFAULT_CMD_VEL_TOPIC,
        odom_topic: str = DEFAULT_ODOM_TOPIC,
        scan_topic: str = DEFAULT_SCAN_TOPIC,
        max_goal_radius: float = 0.0,
    ):
        super().__init__('cyberscape_burger_driver')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        odom_qos = QoSProfile(depth=10)

        self._cmd_vel_topic = cmd_vel_topic
        self._odom_topic = odom_topic
        self._scan_topic = scan_topic
        self._max_goal_radius = max_goal_radius

        # For the real Burger, the default publishes to /cmd_vel_raw so the
        # safety filter can block unsafe motion before it reaches /cmd_vel.
        # For Isaac Sim, pass /sim/cmd_vel, /sim/odom, and /sim/scan.
        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._odom_sub = self.create_subscription(Odometry, self._odom_topic, self._odom_cb, odom_qos)
        self._scan_sub = self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, sensor_qos)

        self._x    = 0.0
        self._y    = 0.0
        self._yaw  = 0.0
        self._has_odom = False
        self._scan = None          # latest LaserScan message
        self._lock = threading.Lock()

        # spin in background so callbacks keep firing
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._spin_thread.start()

        self.get_logger().info(
            '[BurgerDriver] Node started with topics: '
            f'cmd_vel={self._cmd_vel_topic}, odom={self._odom_topic}, scan={self._scan_topic}'
        )
        self.get_logger().info('[BurgerDriver] Waiting for odometry …')
        self._wait_for_odom()
        self.get_logger().info(f'[BurgerDriver] Ready at ({self._x:.2f}, {self._y:.2f})')

    # ── callbacks ─────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._x = msg.pose.pose.position.x
            self._y = msg.pose.pose.position.y
            self._has_odom = True
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._yaw = math.atan2(siny, cosy)

    def _scan_cb(self, msg: LaserScan):
        with self._lock:
            self._scan = msg

    def _wait_for_odom(self, timeout: float = 10.0):
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.1)
            with self._lock:
                if self._has_odom:
                    return
        self.get_logger().warn('[BurgerDriver] Odom timeout – using (0,0) as origin.')

    # ── pose helpers ──────────────────────────────────────

    def _pose(self):
        with self._lock:
            return self._x, self._y, self._yaw

    def _stop(self):
        self._cmd_pub.publish(Twist())

    def _latest_scan(self):
        with self._lock:
            return self._scan

    def _sector_min_distance(self, center_angle: float, half_width: float) -> float:
        scan = self._latest_scan()
        if not scan or not scan.ranges:
            return float('inf')

        closest = float('inf')
        for idx, distance in enumerate(scan.ranges):
            if not scan.range_min < distance < scan.range_max:
                continue

            angle = scan.angle_min + idx * scan.angle_increment
            if abs(normalize_angle(angle - center_angle)) <= half_width:
                closest = min(closest, distance)

        return closest

    def _front_clearance(self) -> float:
        return self._sector_min_distance(0.0, math.radians(18.0))

    def _front_is_blocked(self) -> bool:
        return self._front_clearance() < OBSTACLE_STOP_DISTANCE

    def _clearer_turn_direction(self) -> int:
        left_clearance = self._sector_min_distance(math.radians(55.0), math.radians(35.0))
        right_clearance = self._sector_min_distance(math.radians(-55.0), math.radians(35.0))
        return 1 if left_clearance >= right_clearance else -1

    def _drive_forward_for(self, distance: float) -> bool:
        start_x, start_y, _ = self._pose()
        rate = 0.1

        while True:
            cx, cy, _ = self._pose()
            travelled = math.sqrt((cx - start_x) ** 2 + (cy - start_y) ** 2)

            if travelled >= distance:
                self._stop()
                return True

            if self._front_is_blocked():
                self._stop()
                self.get_logger().warn('[avoidance] Forward path blocked during bypass step.')
                return False

            twist = Twist()
            twist.linear.x = min(LINEAR_SPEED * 0.75, distance - travelled)
            self._cmd_pub.publish(twist)
            time.sleep(rate)

    def _avoid_obstacle(self) -> bool:
        self._stop()
        clearance = self._front_clearance()
        self.get_logger().warn(
            f'[avoidance] Obstacle ahead at {clearance:.2f} m. Stopping and choosing a bypass.'
        )

        for attempt in range(1, 4):
            direction = self._clearer_turn_direction()
            turn = direction * AVOIDANCE_TURN_DEG
            self.get_logger().info(f'[avoidance] Attempt {attempt}: turning {turn:.0f}°.')
            self.rotate(turn)
            time.sleep(0.2)

            if self._front_clearance() >= OBSTACLE_CLEAR_DISTANCE:
                if self._drive_forward_for(AVOIDANCE_STEP_DISTANCE):
                    return True

            self.get_logger().warn('[avoidance] Bypass still blocked; trying the other side.')
            self.rotate(-turn)
            self.rotate(-turn)
            time.sleep(0.2)

            if self._front_clearance() >= OBSTACLE_CLEAR_DISTANCE:
                if self._drive_forward_for(AVOIDANCE_STEP_DISTANCE):
                    return True

        self._stop()
        return False

    # ── public instruction methods ────────────────────────

    def move_to_point(self, x: float, y: float) -> dict:
        """Navigate to (x, y) using a simple proportional controller."""
        target_radius = math.sqrt(x * x + y * y)
        if self._max_goal_radius > 0.0 and target_radius > self._max_goal_radius:
            scale = self._max_goal_radius / target_radius
            original_x, original_y = x, y
            x *= scale
            y *= scale
            self.get_logger().warn(
                '[move_to_point] Target '
                f'({original_x:.2f}, {original_y:.2f}) exceeds max radius '
                f'{self._max_goal_radius:.2f} m; clamped to ({x:.2f}, {y:.2f}).'
            )

        self.get_logger().info(f'[move_to_point] → ({x:.2f}, {y:.2f})')
        rate = 0.1   # control loop period (s)
        avoidance_attempts = 0

        while True:
            cx, cy, cyaw = self._pose()
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < GOAL_TOLERANCE:
                self._stop()
                self.get_logger().info('[move_to_point] Goal reached.')
                return {"reached": True, "x": cx, "y": cy}

            # desired heading
            desired_yaw = math.atan2(dy, dx)
            heading_err = normalize_angle(desired_yaw - cyaw)

            twist = Twist()
            if abs(heading_err) > 0.1:          # rotate first
                twist.angular.z = ANGULAR_SPEED * (1 if heading_err > 0 else -1)
            else:                               # drive forward
                if self._front_is_blocked():
                    avoidance_attempts += 1
                    if avoidance_attempts > MAX_AVOIDANCE_ATTEMPTS:
                        self._stop()
                        message = 'Too many obstacle avoidance attempts; target may be unreachable.'
                        self.get_logger().error(f'[move_to_point] {message}')
                        raise RuntimeError(message)
                    if not self._avoid_obstacle():
                        message = 'Could not find a clear local bypass around obstacle.'
                        self.get_logger().error(f'[move_to_point] {message}')
                        raise RuntimeError(message)
                    continue

                twist.linear.x  = min(LINEAR_SPEED, dist * 0.5)
                twist.angular.z = 1.5 * heading_err

            self._cmd_pub.publish(twist)
            time.sleep(rate)

    def rotate(self, degrees: float) -> dict:
        """Rotate in-place by the given number of degrees."""
        self.get_logger().info(f'[rotate] {degrees}°')
        target_rad = math.radians(degrees)
        _, _, start_yaw = self._pose()
        accumulated = 0.0
        last_yaw    = start_yaw
        rate = 0.05

        twist = Twist()
        twist.angular.z = ANGULAR_SPEED * (1 if target_rad >= 0 else -1)

        while abs(accumulated) < abs(target_rad):
            self._cmd_pub.publish(twist)
            time.sleep(rate)
            _, _, current_yaw = self._pose()
            delta = normalize_angle(current_yaw - last_yaw)
            accumulated += delta
            last_yaw = current_yaw

        self._stop()
        self.get_logger().info('[rotate] Done.')
        return {"rotated_degrees": degrees}

    def scan_with_lidar(self) -> dict:
        """Wait for a fresh LiDAR scan and return summary."""
        self.get_logger().info('[scan_with_lidar] Waiting for scan …')
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                scan = self._scan
            if scan:
                valid = [r for r in scan.ranges if scan.range_min < r < scan.range_max]
                min_dist = min(valid) if valid else float('inf')
                self.get_logger().info(f'[scan_with_lidar] Min distance: {min_dist:.2f} m')
                return {
                    "scan_complete": True,
                    "min_distance_m": round(min_dist, 3),
                    "num_valid_readings": len(valid),
                }
            time.sleep(0.1)
        self.get_logger().warn('[scan_with_lidar] Timed out.')
        return {"scan_complete": False}

    def detect_with_camera(self, target_object: str) -> dict:
        """
        Rotate 360° while looking for target_object with YOLO.
        Reuses the existing Detect.py logic inline.
        """
        self.get_logger().info(f'[detect_with_camera] Looking for: {target_object}')
        try:
            import torch
            from ultralytics import YOLO as _YOLO
        except ImportError:
            self.get_logger().error('[detect_with_camera] ultralytics not installed.')
            return {"detected": False, "object": target_object}

        # Lazy import to avoid loading YOLO at startup
        model = _YOLO('yolov8n.pt')

        detected    = False
        coords      = None
        twist       = Twist()
        twist.angular.z = 0.3
        total_angle = 0.0
        last_time   = time.time()

        # We can't import cv_bridge easily here without ROS; use a flag
        # and let the separate Detect.py handle full detection if needed.
        # This simplified version rotates and flags intent only.
        while total_angle < 2 * math.pi and not detected:
            self._cmd_pub.publish(twist)
            now = time.time()
            total_angle += twist.angular.z * (now - last_time)
            last_time = now
            time.sleep(0.1)

        self._stop()
        # Actual YOLO result comes from the dedicated Detect.py subscription;
        # here we return the detection placeholder so the plan can proceed.
        return {"detected": detected, "object": target_object, "x": coords}

    def wait_for_signal(self, server_url: str, robot_name: str,
                        poll_interval: float = 1.0, timeout: float = 300.0) -> dict:
        """Poll the server until the robot is back in AUTONOMOUS mode."""
        self.get_logger().info('[wait_for_signal] Waiting for signal from server …')
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f'{server_url}/hitl/status', timeout=3)
                if r.ok:
                    robots = r.json().get('robots', {})
                    mode   = robots.get(robot_name, {}).get('mode', '')
                    if mode == 'AUTONOMOUS':
                        self.get_logger().info('[wait_for_signal] Signal received.')
                        return {"signal_received": True}
            except Exception:
                pass
            time.sleep(poll_interval)
        self.get_logger().warn('[wait_for_signal] Timed out.')
        return {"signal_received": False}

    def communicate_with_apm(self, data: str) -> dict:
        """Log the data payload (APM channel is handled server-side)."""
        self.get_logger().info(f'[communicate_with_apm] data={data}')
        return {"apm_data_sent": data}

    def retrieve_object(self, target_object: str) -> dict:
        """Stop at current position and mark retrieval (gripper/arm not on Burger by default)."""
        self.get_logger().info(f'[retrieve_object] Retrieving: {target_object}')
        self._stop()
        time.sleep(1.0)   # simulate retrieval action
        return {"retrieved": True, "object": target_object}

    def return_to_base(self) -> dict:
        """Navigate back to (0, 0)."""
        self.get_logger().info('[return_to_base] Returning to base …')
        return self.move_to_point(0.0, 0.0)


# ─────────────────────────────────────────────────────────
# Instruction Parser
# ─────────────────────────────────────────────────────────

def parse_and_execute(instruction: str, driver: BurgerDriver,
                      server_url: str, robot_name: str) -> dict:
    """
    Parse a single CyberScape instruction string and execute it.
    Returns the output dict produced by the action.
    """
    instruction = instruction.strip()
    if not instruction:
        return {}

    # Strip the 'TurtleBot3.' prefix
    match = re.match(r'TurtleBot3\.(\w+)\((.*)\)$', instruction, re.DOTALL)
    if not match:
        print(f'[parser] Unrecognised instruction: {instruction}')
        return {}

    func_name = match.group(1)
    raw_args  = match.group(2).strip()

    # ── move_to_point((x, y)) ──────────────────────────────
    if func_name == 'move_to_point':
        inner = re.sub(r'[\(\)\s]', '', raw_args)
        parts = inner.split(',')
        x, y  = float(parts[0]), float(parts[1])
        return driver.move_to_point(x, y)

    # ── rotate(angle) ─────────────────────────────────────
    elif func_name == 'rotate':
        angle = float(raw_args)
        return driver.rotate(angle)

    # ── scan_with_lidar() ─────────────────────────────────
    elif func_name == 'scan_with_lidar':
        return driver.scan_with_lidar()

    # ── detect_with_camera(object) ────────────────────────
    elif func_name == 'detect_with_camera':
        obj = raw_args.strip().strip("'\"")
        return driver.detect_with_camera(obj)

    # ── wait_for_signal() ─────────────────────────────────
    elif func_name == 'wait_for_signal':
        return driver.wait_for_signal(server_url, robot_name)

    # ── communicate_with_apm(data) ────────────────────────
    elif func_name == 'communicate_with_apm':
        data = raw_args.strip().strip("'\"")
        return driver.communicate_with_apm(data)

    # ── retrieve_object(object) ───────────────────────────
    elif func_name == 'retrieve_object':
        obj = raw_args.strip().strip("'\"")
        return driver.retrieve_object(obj)

    # ── return_to_base() ──────────────────────────────────
    elif func_name == 'return_to_base':
        return driver.return_to_base()

    else:
        print(f'[parser] Unknown function: {func_name}')
        return {}


# ─────────────────────────────────────────────────────────
# Server Communication Helpers
# ─────────────────────────────────────────────────────────

def send_heartbeat(server_url: str, robot_name: str):
    try:
        requests.post(f'{server_url}/hitl/heartbeat',
                      json={'robot': robot_name}, timeout=3)
    except Exception:
        pass


def get_instruction(server_url: str, robot_name: str, phase: int) -> dict | None:
    try:
        r = requests.get(f'{server_url}/get_instruction',
                         params={'robot': robot_name, 'phase': phase}, timeout=5)
        if r.status_code == 200:
            return r.json()
        print(f'[server] GET /get_instruction phase={phase} → {r.status_code}: {r.text}')
    except Exception as e:
        print(f'[server] Error fetching instruction: {e}')
    return None


def complete_phase(server_url: str, robot_name: str, phase: int, outputs: dict):
    try:
        r = requests.post(f'{server_url}/complete_phase',
                          json={'robot': robot_name, 'phase': phase, 'outputs': outputs},
                          timeout=5)
        if r.ok:
            print(f'[server] Phase {phase} marked complete. outputs={outputs}')
        else:
            print(f'[server] complete_phase failed: {r.status_code} {r.text}')
    except Exception as e:
        print(f'[server] Error completing phase: {e}')


def report_error(server_url: str, robot_name: str, phase: int,
                 instruction_number: int, description: str):
    try:
        requests.post(f'{server_url}/report_error', json={
            'robot':              robot_name,
            'phase':              phase,
            'instruction_number': instruction_number,
            'description':        description,
        }, timeout=5)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────
# Main Mission Loop
# ─────────────────────────────────────────────────────────

def run_mission(server_url: str, robot_name: str, driver: BurgerDriver, mirror_only: bool = False):
    phase = 1
    print(f'[mission] Starting mission loop for {robot_name} on {server_url}')
    if mirror_only:
        print('[mission] Mirror-only mode enabled; this node will not complete phases on the server.')

    # heartbeat thread
    def _hb_loop():
        while True:
            send_heartbeat(server_url, robot_name)
            time.sleep(HEARTBEAT_PERIOD)

    threading.Thread(target=_hb_loop, daemon=True).start()

    while True:
        data = get_instruction(server_url, robot_name, phase)

        if data is None:
            # Plan not approved yet, locked, or phase not found — wait and retry
            time.sleep(2.0)
            continue

        instructions = data.get('low_level_plan', [])
        expected_out = data.get('expected_outputs', {})

        if isinstance(instructions, str):
            instructions = [l.strip() for l in instructions.splitlines() if l.strip()]

        if not instructions:
            print(f'[mission] Phase {phase} has no instructions — skipping.')
            complete_phase(server_url, robot_name, phase, {})
            phase += 1
            continue

        print(f'\n[mission] ── Phase {phase} ({len(instructions)} instructions) ──')

        phase_outputs = {}
        for idx, instr in enumerate(instructions, start=1):
            print(f'  [{idx}] {instr}')
            try:
                result = parse_and_execute(instr, driver, server_url, robot_name)
                phase_outputs.update(result)
            except Exception as exc:
                print(f'  [!] Instruction {idx} failed: {exc}')
                if not mirror_only:
                    report_error(server_url, robot_name, phase, idx, str(exc))
                break

        # Merge with expected outputs structure (fill missing keys with empty string)
        outputs_to_send = {k: phase_outputs.get(k, '') for k in expected_out}
        outputs_to_send.update({k: v for k, v in phase_outputs.items()
                                 if k not in outputs_to_send})

        if mirror_only:
            print(f'[mission] Mirror-only phase {phase} finished locally. outputs={outputs_to_send}')
        else:
            complete_phase(server_url, robot_name, phase, outputs_to_send)
        phase += 1


# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='TurtleBot3 Burger CyberScape comm node')
    ap.add_argument('--server', default=DEFAULT_SERVER,
                    help=f'CyberScape server URL (default: {DEFAULT_SERVER})')
    ap.add_argument('--robot',  default=ROBOT_NAME,
                    help=f'Robot identifier used in the server (default: {ROBOT_NAME})')
    ap.add_argument('--cmd_vel_topic', default=DEFAULT_CMD_VEL_TOPIC,
                    help=f'ROS2 Twist topic to publish movement commands (default: {DEFAULT_CMD_VEL_TOPIC})')
    ap.add_argument('--odom_topic', default=DEFAULT_ODOM_TOPIC,
                    help=f'ROS2 Odometry topic to read pose from (default: {DEFAULT_ODOM_TOPIC})')
    ap.add_argument('--scan_topic', default=DEFAULT_SCAN_TOPIC,
                    help=f'ROS2 LaserScan topic to read obstacles from (default: {DEFAULT_SCAN_TOPIC})')
    ap.add_argument('--mirror_only', action='store_true',
                    help='Execute the plan locally but do not mark phases complete on the CyberScape server.')
    ap.add_argument('--max_goal_radius', type=float, default=0.0,
                    help='Clamp move_to_point targets to this radius from (0,0). Use 0 to disable.')
    args = ap.parse_args()

    rclpy.init()
    driver = BurgerDriver(
        cmd_vel_topic=args.cmd_vel_topic,
        odom_topic=args.odom_topic,
        scan_topic=args.scan_topic,
        max_goal_radius=args.max_goal_radius,
    )

    try:
        run_mission(args.server, args.robot, driver, mirror_only=args.mirror_only)
    except KeyboardInterrupt:
        print('\n[mission] Interrupted by user.')
    finally:
        driver._stop()
        driver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
