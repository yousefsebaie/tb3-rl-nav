#!/usr/bin/env python3
"""Safety filter node for TurtleBot3 raw velocity commands.

This node protects the robot by filtering raw motion commands before they
reach /cmd_vel. It subscribes to /cmd_vel_raw and /scan, then publishes
safe twist messages to /cmd_vel.
"""

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class SafetyFilter(Node):
    def __init__(self):
        super().__init__('burger_safety_filter')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._raw_sub = self.create_subscription(Twist, '/cmd_vel_raw', self._raw_cmd_cb, qos)
        self._scan_sub = self.create_subscription(LaserScan, '/scan', self._scan_cb, qos)

        self._latest_raw = Twist()
        self._has_raw = False
        self._scan = None

        self.get_logger().info('[SafetyFilter] Started. Filtering /cmd_vel_raw → /cmd_vel.')

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg
        self._publish_safe_command()

    def _raw_cmd_cb(self, msg: Twist):
        self._latest_raw = msg
        self._has_raw = True
        self._publish_safe_command()

    def _obstacle_in_front(self) -> bool:
        if not self._scan:
            return False

        if not self._scan.ranges:
            return False

        front_half_width = math.radians(18.0)
        for idx, r in enumerate(self._scan.ranges):
            if not self._scan.range_min < r < self._scan.range_max:
                continue
            angle = self._scan.angle_min + idx * self._scan.angle_increment
            if abs(normalize_angle(angle)) <= front_half_width and r < 0.30:
                return True
        return False

    def _publish_safe_command(self):
        if not self._has_raw:
            return

        safe_cmd = Twist()
        safe_cmd.linear.x = self._latest_raw.linear.x
        safe_cmd.linear.y = self._latest_raw.linear.y
        safe_cmd.linear.z = self._latest_raw.linear.z
        safe_cmd.angular.x = self._latest_raw.angular.x
        safe_cmd.angular.y = self._latest_raw.angular.y
        safe_cmd.angular.z = self._latest_raw.angular.z

        if self._obstacle_in_front() and safe_cmd.linear.x > 0.0:
            self.get_logger().warn('[SafetyFilter] Obstacle detected < 0.30 m; blocking forward motion.')
            safe_cmd.linear.x = 0.0

        self._cmd_pub.publish(safe_cmd)


def main():
    rclpy.init()
    node = SafetyFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
