#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3


class LaneControlNode(Node):
    """
    Stanley controller cho xe Ackermann bám làn đường.

    Subscribe /status_err (Vector3: x=e_y m, y=e_psi rad)
    Publish   /cmd_vel   (Twist)  ở 20 Hz

    Stanley:  delta     = e_psi + arctan(k * e_y / v)
              angular_z = v * tan(delta) / L
    """

    _WHEELBASE_M = 0.21   # m — khoảng cách trục trước–sau

    def __init__(self):
        super().__init__('lane_control_node')

        self.declare_parameter('speed',      0.15)   # m/s
        self.declare_parameter('k',          1.0)    # Stanley gain
        self.declare_parameter('max_steer',  0.52)   # rad — giới hạn lái vật lý
        self.declare_parameter('timeout',    0.5)    # s  — safety stop nếu mất tin hiệu

        self._speed     = self.get_parameter('speed').value
        self._k         = self.get_parameter('k').value
        self._max_steer = self.get_parameter('max_steer').value
        self._timeout   = self.get_parameter('timeout').value

        self._e_y:   float = 0.0
        self._e_psi: float = 0.0
        self._last_err_time: float = 0.0   # ROS time (s)

        self._sub = self.create_subscription(
            Vector3, '/status_err', self._err_callback, 10
        )
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / 20.0, self._control_loop)

        self.get_logger().info(
            f'Stanley lane control: speed={self._speed} m/s  k={self._k}  '
            f'max_steer={math.degrees(self._max_steer):.1f}deg  '
            f'timeout={self._timeout}s'
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _err_callback(self, msg: Vector3):
        self._e_y   = msg.x
        self._e_psi = msg.y
        self._last_err_time = self.get_clock().now().nanoseconds * 1e-9

    # ──────────────────────────────────────────────────────────────────────────
    def _control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        twist = Twist()

        # Safety stop: mất tín hiệu làn quá lâu
        if self._last_err_time == 0.0 or (now - self._last_err_time) > self._timeout:
            self._pub.publish(twist)   # (0, 0)
            return

        v = self._speed
        L = self._WHEELBASE_M

        # Stanley formula
        delta = self._e_psi + math.atan2(self._k * self._e_y, max(v, 0.1))
        delta = max(-self._max_steer, min(self._max_steer, delta))

        angular_z = v * math.tan(delta) / L

        twist.linear.x  = v
        twist.angular.z = angular_z
        self._pub.publish(twist)


# ──────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = LaneControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
