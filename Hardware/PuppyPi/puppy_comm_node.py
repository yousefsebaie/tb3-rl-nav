#!/usr/bin/env python3
"""
PuppyPi CyberScape Communication Node
=====================================
Polls the CyberScape HITL server for RobotDog/PuppyPi phases and executes
them on a Hiwonder PuppyPi through ROS 1 rosbridge.

This adapter intentionally does not reuse the TurtleBot3 ROS 2 driver:

  - TurtleBot3 Burger uses ROS 2 /cmd_vel, /odom, and /scan.
  - PuppyPi uses ROS 1 /puppy_control/pose, /puppy_control/gait,
    /puppy_control/velocity_move, and /scan through rosbridge.

PuppyPi currently has no /odom topic, so coordinate movement is only an
estimated dead-reckoning fallback. Prefer timed velocity commands for early
hardware tests.

Usage:
    python3 Hardware/PuppyPi/puppy_comm_node.py \
        --server http://localhost:5001 \
        --robot ROBOT_DOG \
        --puppy_ip 192.168.137.136
"""

import argparse
import atexit
import math
import re
import signal
import sys
import threading
import time
from typing import Optional

import requests

try:
    import roslibpy
except ImportError:  # pragma: no cover - runtime dependency on deployment PC
    roslibpy = None


DEFAULT_SERVER = "http://localhost:5001"
DEFAULT_ROBOT_NAME = "ROBOT_DOG"
DEFAULT_PUPPY_IP = "192.168.137.136"
DEFAULT_ROSBRIDGE_PORT = 9090

HEARTBEAT_PERIOD = 5.0
COMMAND_TIMEOUT = 2.0
CONTROL_RATE_HZ = 20.0

FRONT_SECTOR_DEG = 25.0
OBSTACLE_STOP_DISTANCE = 0.40

DEFAULT_FORWARD_SPEED = 5.0
DEFAULT_TURN_RATE = 0.18
DEFAULT_LINEAR_UNITS_PER_METER = 10.0
DEFAULT_YAW_RATE_RAD_PER_SEC = 0.18


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PuppyPiDriver:
    """ROS 1 rosbridge driver for PuppyPi pose, gait, velocity, and scan."""

    def __init__(
        self,
        puppy_ip: str = DEFAULT_PUPPY_IP,
        rosbridge_port: int = DEFAULT_ROSBRIDGE_PORT,
        forward_speed: float = DEFAULT_FORWARD_SPEED,
        turn_rate: float = DEFAULT_TURN_RATE,
        obstacle_stop_distance: float = OBSTACLE_STOP_DISTANCE,
        linear_units_per_meter: float = DEFAULT_LINEAR_UNITS_PER_METER,
        yaw_rate_rad_per_sec: float = DEFAULT_YAW_RATE_RAD_PER_SEC,
    ):
        if roslibpy is None:
            raise RuntimeError(
                "roslibpy is not installed. Install it on the CyberScape PC with "
                "`python3 -m pip install roslibpy`, or run a rospy-based adapter "
                "inside the PuppyPi Docker container."
            )

        self._puppy_ip = puppy_ip
        self._rosbridge_port = rosbridge_port
        self._forward_speed = forward_speed
        self._turn_rate = turn_rate
        self._obstacle_stop_distance = obstacle_stop_distance
        self._linear_units_per_meter = linear_units_per_meter
        self._yaw_rate_rad_per_sec = yaw_rate_rad_per_sec

        self._scan = None
        self._scan_lock = threading.Lock()
        self._last_velocity_time = 0.0
        self._last_velocity = {"x": 0.0, "y": 0.0, "yaw_rate": 0.0}
        self._closed = False

        # Dead-reckoned local estimate used only when CyberScape emits
        # RobotDog.move_to_point. PuppyPi does not currently publish odom.
        self._est_x = 0.0
        self._est_y = 0.0
        self._est_yaw = 0.0

        print(f"[puppy] Connecting to rosbridge ws://{puppy_ip}:{rosbridge_port}")
        self._ros = roslibpy.Ros(host=puppy_ip, port=rosbridge_port)
        self._ros.run(timeout=15)
        if not self._ros.is_connected:
            raise RuntimeError(f"Could not connect to PuppyPi rosbridge at {puppy_ip}:{rosbridge_port}")

        self._pose_pub = roslibpy.Topic(self._ros, "/puppy_control/pose", "puppy_control/Pose")
        self._gait_pub = roslibpy.Topic(self._ros, "/puppy_control/gait", "puppy_control/Gait")
        self._vel_pub = roslibpy.Topic(self._ros, "/puppy_control/velocity_move", "puppy_control/Velocity")
        self._scan_sub = roslibpy.Topic(self._ros, "/scan", "sensor_msgs/LaserScan")
        self._scan_sub.subscribe(self._scan_cb)

        self.initialize_body()

        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

        print("[puppy] Ready. Pose/gait initialized and /scan subscribed.")

    def _scan_cb(self, msg):
        with self._scan_lock:
            self._scan = msg

    def initialize_body(self):
        self.publish_pose()
        time.sleep(0.2)
        self.publish_gait()
        time.sleep(0.2)
        self.stop()

    def publish_pose(
        self,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        height: float = -10.0,
        x_shift: float = -0.5,
        stance_x: float = 0.0,
        stance_y: float = 0.0,
        run_time: int = 500,
    ):
        self._pose_pub.publish(roslibpy.Message({
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "height": height,
            "x_shift": x_shift,
            "stance_x": stance_x,
            "stance_y": stance_y,
            "run_time": run_time,
        }))
        print("[puppy] Standing pose published.")

    def publish_gait(
        self,
        overlap_time: float = 0.35,
        swing_time: float = 0.4,
        clearance_time: float = 0.0,
        z_clearance: float = 7.0,
    ):
        self._gait_pub.publish(roslibpy.Message({
            "overlap_time": overlap_time,
            "swing_time": swing_time,
            "clearance_time": clearance_time,
            "z_clearance": z_clearance,
        }))
        print("[puppy] Gait published.")

    def _front_min_distance(self) -> float:
        return self._sector_min_distance(0.0, FRONT_SECTOR_DEG)

    def _sector_min_distance(self, center_angle_deg: float, half_width_deg: float) -> float:
        with self._scan_lock:
            scan = self._scan

        if not scan:
            return float("inf")

        ranges = scan.get("ranges") or []
        if not ranges:
            return float("inf")

        angle_min = float(scan.get("angle_min", -math.pi))
        angle_inc = float(scan.get("angle_increment", 0.0))
        range_min = float(scan.get("range_min", 0.0))
        range_max = float(scan.get("range_max", float("inf")))
        center_angle = math.radians(center_angle_deg)
        half_width = math.radians(half_width_deg)

        closest = float("inf")
        for idx, raw_distance in enumerate(ranges):
            try:
                distance = float(raw_distance)
            except (TypeError, ValueError):
                continue
            if math.isnan(distance) or math.isinf(distance):
                continue
            if not range_min < distance < range_max:
                continue

            angle = angle_min + idx * angle_inc
            if abs(normalize_angle(angle - center_angle)) <= half_width:
                closest = min(closest, distance)

        return closest

    def obstacle_too_close(self) -> bool:
        return self._front_min_distance() < self._obstacle_stop_distance

    def avoid_obstacle(self) -> dict:
        front = self._front_min_distance()
        if front >= self._obstacle_stop_distance:
            return {
                "avoidance_needed": False,
                "front_distance_m": round(front, 3),
            }

        self.stop()
        left = self._sector_min_distance(55.0, 35.0)
        right = self._sector_min_distance(-55.0, 35.0)
        turn_left = left >= right
        direction = "left" if turn_left else "right"

        print(
            f"[puppy][avoidance] Front blocked at {front:.2f} m; "
            f"turning {direction} toward clearer side."
        )
        if turn_left:
            turn_result = self.turn_left(duration=1.2, yaw_rate=self._turn_rate)
        else:
            turn_result = self.turn_right(duration=1.2, yaw_rate=self._turn_rate)

        new_front = self._front_min_distance()
        return {
            "avoidance_needed": True,
            "avoidance_direction": direction,
            "front_distance_m": round(new_front, 3),
            **turn_result,
        }

    def publish_velocity(self, x: float, y: float = 0.0, yaw_rate: float = 0.0) -> bool:
        x = float(x)
        y = float(y)
        yaw_rate = float(yaw_rate)

        if x > 0.0 and self.obstacle_too_close():
            front = self._front_min_distance()
            print(f"[puppy][safety] Obstacle at {front:.2f} m; blocking forward command.")
            x = 0.0
            y = 0.0
            yaw_rate = 0.0

        msg = {"x": x, "y": y, "yaw_rate": yaw_rate}
        self._vel_pub.publish(roslibpy.Message(msg))
        self._last_velocity = msg
        self._last_velocity_time = time.time()
        return x != 0.0 or y != 0.0 or yaw_rate != 0.0

    def stop(self):
        self._vel_pub.publish(roslibpy.Message({"x": 0.0, "y": 0.0, "yaw_rate": 0.0}))
        self._last_velocity = {"x": 0.0, "y": 0.0, "yaw_rate": 0.0}
        self._last_velocity_time = time.time()

    def _watchdog_loop(self):
        while not self._closed:
            time.sleep(0.1)
            moving = any(abs(v) > 1e-6 for v in self._last_velocity.values())
            if moving and time.time() - self._last_velocity_time > COMMAND_TIMEOUT:
                print("[puppy][watchdog] No recent command; sending stop.")
                self.stop()

    def move_for(self, x: float, yaw_rate: float, duration: float) -> dict:
        duration = max(0.0, float(duration))
        period = 1.0 / CONTROL_RATE_HZ
        deadline = time.time() + duration
        blocked = False

        while time.time() < deadline:
            active = self.publish_velocity(x=x, y=0.0, yaw_rate=yaw_rate)
            if x > 0.0 and not active:
                blocked = True
                break
            time.sleep(period)

        self.stop()
        return {
            "duration_s": round(duration, 3),
            "blocked_by_obstacle": blocked,
            "front_distance_m": round(self._front_min_distance(), 3),
        }

    def move_forward(self, duration: float = 1.0, speed: Optional[float] = None) -> dict:
        speed = self._forward_speed if speed is None else float(speed)
        return self.move_for(x=abs(speed), yaw_rate=0.0, duration=duration)

    def move_backward(self, duration: float = 1.0, speed: Optional[float] = None) -> dict:
        speed = self._forward_speed if speed is None else float(speed)
        return self.move_for(x=-abs(speed), yaw_rate=0.0, duration=duration)

    def turn_left(self, duration: float = 1.0, yaw_rate: Optional[float] = None) -> dict:
        yaw_rate = self._turn_rate if yaw_rate is None else float(yaw_rate)
        return self.move_for(x=0.0, yaw_rate=abs(yaw_rate), duration=duration)

    def turn_right(self, duration: float = 1.0, yaw_rate: Optional[float] = None) -> dict:
        yaw_rate = self._turn_rate if yaw_rate is None else float(yaw_rate)
        return self.move_for(x=0.0, yaw_rate=-abs(yaw_rate), duration=duration)

    def rotate(self, degrees: float) -> dict:
        degrees = float(degrees)
        yaw_rate = self._turn_rate if degrees >= 0.0 else -self._turn_rate
        duration = abs(math.radians(degrees)) / max(self._yaw_rate_rad_per_sec, 1e-6)
        # Switch to pivot-friendly gait for rotation
        self.publish_gait(overlap_time=0.1, swing_time=0.3, clearance_time=0.0, z_clearance=5.0)
        time.sleep(0.2)
        result = self.move_for(x=0.0, yaw_rate=yaw_rate, duration=duration)
        # Restore walking gait
        self.publish_gait()
        time.sleep(0.2)
        self._est_yaw = normalize_angle(self._est_yaw + math.radians(degrees))
        result.update({"rotated_degrees": degrees, "estimated": True})
        return result

    def move_to_point(self, x: float, y: float) -> dict:
        """Estimated coordinate move using commanded velocity/time, not odom."""
        x = float(x)
        y = float(y)
        dx = x - self._est_x
        dy = y - self._est_y
        distance = math.hypot(dx, dy)

        if distance < 0.01:
            self.stop()
            return {"reached": True, "x": self._est_x, "y": self._est_y, "estimated": True}

        target_yaw = math.atan2(dy, dx)
        turn_deg = math.degrees(normalize_angle(target_yaw - self._est_yaw))
        self.rotate(turn_deg)

        duration = (distance * self._linear_units_per_meter) / max(abs(self._forward_speed), 1e-6)
        result = self.move_forward(duration=duration, speed=self._forward_speed)
        if not result.get("blocked_by_obstacle"):
            self._est_x = x
            self._est_y = y
            self._est_yaw = target_yaw

        result.update({
            "reached": not result.get("blocked_by_obstacle"),
            "x": round(self._est_x, 3),
            "y": round(self._est_y, 3),
            "estimated": True,
            "note": "PuppyPi has no /odom; move_to_point used dead-reckoning timing.",
        })
        return result

    def scan_with_lidar(self) -> dict:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._scan_lock:
                scan = self._scan
            if scan:
                ranges = scan.get("ranges") or []
                valid = []
                range_min = float(scan.get("range_min", 0.0))
                range_max = float(scan.get("range_max", float("inf")))
                for raw_distance in ranges:
                    try:
                        distance = float(raw_distance)
                    except (TypeError, ValueError):
                        continue
                    if not math.isnan(distance) and not math.isinf(distance) and range_min < distance < range_max:
                        valid.append(distance)
                return {
                    "scan_complete": True,
                    "min_distance_m": round(min(valid), 3) if valid else "",
                    "front_distance_m": round(self._front_min_distance(), 3),
                    "num_valid_readings": len(valid),
                }
            time.sleep(0.1)
        return {"scan_complete": False}

    def detect_with_camera(self, target_object: str) -> dict:
        print(f"[puppy] Camera detection requested for {target_object}; detector is not wired yet.")
        return {"detected": False, "object": target_object}

    def retrieve_object(self, target_object: str) -> dict:
        self.stop()
        print(f"[puppy] Retrieval requested for {target_object}; actuator is not wired yet.")
        return {"retrieved": False, "object": target_object}

    def return_to_base(self) -> dict:
        return self.move_to_point(0.0, 0.0)

    def wait_for_signal(
        self,
        server_url: str,
        robot_name: str,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = requests.get(f"{server_url}/hitl/status", timeout=3)
                if response.ok:
                    robots = response.json().get("robots", {})
                    mode = robots.get(robot_name, {}).get("mode", "")
                    if mode == "AUTONOMOUS":
                        return {"signal_received": True}
            except Exception:
                pass
            self.stop()
            time.sleep(poll_interval)
        return {"signal_received": False}

    def communicate_with_apm(self, data: str) -> dict:
        print(f"[puppy] APM data: {data}")
        return {"apm_data_sent": data}

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
            time.sleep(0.1)
        except Exception:
            pass
        try:
            self._scan_sub.unsubscribe()
            self._pose_pub.unadvertise()
            self._gait_pub.unadvertise()
            self._vel_pub.unadvertise()
            self._ros.terminate()
        except Exception:
            pass


def _parse_args(raw_args: str):
    raw_args = raw_args.strip()
    if not raw_args:
        return []

    # Keep parsing deliberately small and predictable for generated plans.
    if raw_args.startswith("((") or raw_args.startswith("("):
        inner = re.sub(r"[\(\)\s]", "", raw_args)
        return [float(part) for part in inner.split(",") if part]

    parts = [part.strip() for part in raw_args.split(",") if part.strip()]
    parsed = []
    for part in parts:
        if "=" in part:
            _, part = part.split("=", 1)
        part = part.strip().strip("'\"")
        try:
            parsed.append(float(part))
        except ValueError:
            parsed.append(part)
    return parsed


def parse_and_execute(instruction: str, driver: PuppyPiDriver, server_url: str, robot_name: str) -> dict:
    instruction = instruction.strip()
    if not instruction:
        return {}

    match = re.match(r"(RobotDog|PuppyPi)\.(\w+)\((.*)\)$", instruction, re.DOTALL)
    if not match:
        print(f"[parser] Unrecognised PuppyPi instruction: {instruction}")
        return {}

    func_name = match.group(2)
    args = _parse_args(match.group(3))

    if func_name == "move_to_point":
        if len(args) < 2:
            raise ValueError("move_to_point requires x and y")
        return driver.move_to_point(args[0], args[1])
    if func_name == "rotate":
        return driver.rotate(args[0] if args else 0.0)
    if func_name == "move_forward":
        duration = args[0] if args else 1.0
        speed = args[1] if len(args) > 1 else None
        return driver.move_forward(duration=duration, speed=speed)
    if func_name == "move_backward":
        duration = args[0] if args else 1.0
        speed = args[1] if len(args) > 1 else None
        return driver.move_backward(duration=duration, speed=speed)
    if func_name == "turn_left":
        duration = args[0] if args else 1.0
        yaw_rate = args[1] if len(args) > 1 else None
        return driver.turn_left(duration=duration, yaw_rate=yaw_rate)
    if func_name == "turn_right":
        duration = args[0] if args else 1.0
        yaw_rate = args[1] if len(args) > 1 else None
        return driver.turn_right(duration=duration, yaw_rate=yaw_rate)
    if func_name == "stop":
        driver.stop()
        return {"stopped": True}
    if func_name == "avoid_obstacle":
        return driver.avoid_obstacle()
    if func_name == "scan_with_lidar":
        return driver.scan_with_lidar()
    if func_name == "detect_with_camera":
        target = str(args[0]) if args else ""
        return driver.detect_with_camera(target)
    if func_name == "wait_for_signal":
        return driver.wait_for_signal(server_url, robot_name)
    if func_name == "communicate_with_apm":
        data = str(args[0]) if args else ""
        return driver.communicate_with_apm(data)
    if func_name in ("retrieve_object", "retrieve_the_object"):
        target = str(args[0]) if args else ""
        return driver.retrieve_object(target)
    if func_name == "return_to_base":
        return driver.return_to_base()
    if func_name == "jump":
        raise RuntimeError("PuppyPi jump is not implemented in the current ROS 1 control interface.")

    print(f"[parser] Unknown PuppyPi function: {func_name}")
    return {}


def send_heartbeat(server_url: str, robot_name: str):
    try:
        requests.post(f"{server_url}/hitl/heartbeat", json={"robot": robot_name}, timeout=3)
    except Exception:
        pass


def get_instruction(server_url: str, robot_name: str, phase: int) -> Optional[dict]:
    try:
        response = requests.get(
            f"{server_url}/get_instruction",
            params={"robot": robot_name, "phase": phase},
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
        print(f"[server] GET /get_instruction phase={phase} -> {response.status_code}: {response.text}")
    except Exception as exc:
        print(f"[server] Error fetching instruction: {exc}")
    return None


def complete_phase(server_url: str, robot_name: str, phase: int, outputs: dict):
    try:
        response = requests.post(
            f"{server_url}/complete_phase",
            json={"robot": robot_name, "phase": phase, "outputs": outputs},
            timeout=5,
        )
        if response.ok:
            print(f"[server] Phase {phase} marked complete. outputs={outputs}")
        else:
            print(f"[server] complete_phase failed: {response.status_code} {response.text}")
    except Exception as exc:
        print(f"[server] Error completing phase: {exc}")


def report_error(server_url: str, robot_name: str, phase: int, instruction_number: int, description: str):
    try:
        requests.post(f"{server_url}/report_error", json={
            "robot": robot_name,
            "phase": phase,
            "instruction_number": instruction_number,
            "description": description,
        }, timeout=5)
    except Exception:
        pass


def run_mission(server_url: str, robot_name: str, driver: PuppyPiDriver, mirror_only: bool = False):
    phase = 1
    print(f"[mission] Starting PuppyPi mission loop for {robot_name} on {server_url}")
    if mirror_only:
        print("[mission] Mirror-only mode enabled; server phases will not be completed.")

    def heartbeat_loop():
        while True:
            send_heartbeat(server_url, robot_name)
            time.sleep(HEARTBEAT_PERIOD)

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    while True:
        data = get_instruction(server_url, robot_name, phase)
        if data is None:
            driver.stop()
            time.sleep(2.0)
            continue

        instructions = data.get("low_level_plan", [])
        expected_outputs = data.get("expected_outputs", {})
        if isinstance(instructions, str):
            instructions = [line.strip() for line in instructions.splitlines() if line.strip()]

        if not instructions:
            print(f"[mission] Phase {phase} has no instructions; skipping.")
            if not mirror_only:
                complete_phase(server_url, robot_name, phase, {})
            phase += 1
            continue

        print(f"\n[mission] -- PuppyPi phase {phase} ({len(instructions)} instructions) --")
        phase_outputs = {}
        phase_failed = False

        for idx, instruction in enumerate(instructions, start=1):
            print(f"  [{idx}] {instruction}")
            try:
                result = parse_and_execute(instruction, driver, server_url, robot_name)
                phase_outputs.update(result)
            except Exception as exc:
                phase_failed = True
                driver.stop()
                print(f"  [!] Instruction {idx} failed: {exc}")
                if not mirror_only:
                    report_error(server_url, robot_name, phase, idx, str(exc))
                break

        if phase_failed:
            time.sleep(2.0)
            continue

        outputs_to_send = {key: phase_outputs.get(key, "") for key in expected_outputs}
        outputs_to_send.update({key: value for key, value in phase_outputs.items() if key not in outputs_to_send})

        if mirror_only:
            print(f"[mission] Mirror-only phase {phase} finished. outputs={outputs_to_send}")
        else:
            complete_phase(server_url, robot_name, phase, outputs_to_send)
        phase += 1


def main():
    parser = argparse.ArgumentParser(description="PuppyPi CyberScape rosbridge comm node")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"CyberScape server URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--robot", default=DEFAULT_ROBOT_NAME, help=f"Server robot name (default: {DEFAULT_ROBOT_NAME})")
    parser.add_argument("--puppy_ip", default=DEFAULT_PUPPY_IP, help=f"PuppyPi IP address (default: {DEFAULT_PUPPY_IP})")
    parser.add_argument("--rosbridge_port", type=int, default=DEFAULT_ROSBRIDGE_PORT, help="PuppyPi rosbridge port")
    parser.add_argument("--forward_speed", type=float, default=DEFAULT_FORWARD_SPEED, help="PuppyPi forward x command")
    parser.add_argument("--turn_rate", type=float, default=DEFAULT_TURN_RATE, help="PuppyPi yaw_rate command")
    parser.add_argument("--obstacle_stop_distance", type=float, default=OBSTACLE_STOP_DISTANCE, help="Front obstacle stop distance in meters")
    parser.add_argument("--linear_units_per_meter", type=float, default=DEFAULT_LINEAR_UNITS_PER_METER, help="Dead-reckoning calibration for move_to_point")
    parser.add_argument("--yaw_rate_rad_per_sec", type=float, default=DEFAULT_YAW_RATE_RAD_PER_SEC, help="Dead-reckoning turn calibration")
    parser.add_argument("--mirror_only", action="store_true", help="Execute locally without completing phases on the server")
    args = parser.parse_args()

    driver = PuppyPiDriver(
        puppy_ip=args.puppy_ip,
        rosbridge_port=args.rosbridge_port,
        forward_speed=args.forward_speed,
        turn_rate=args.turn_rate,
        obstacle_stop_distance=args.obstacle_stop_distance,
        linear_units_per_meter=args.linear_units_per_meter,
        yaw_rate_rad_per_sec=args.yaw_rate_rad_per_sec,
    )

    def shutdown_handler(*_):
        print("\n[puppy] Shutdown requested; stopping PuppyPi.")
        driver.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    atexit.register(driver.close)

    run_mission(args.server, args.robot, driver, mirror_only=args.mirror_only)


if __name__ == "__main__":
    main()
