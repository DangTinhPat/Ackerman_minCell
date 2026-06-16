#!/usr/bin/env python3
"""
Relay /ackermann_steering_controller/tf_odometry → /tf
dùng tf2_ros.TransformBroadcaster để đảm bảo đúng QoS cho tf2 listener.
"""
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
import tf2_ros


class OdomTfRelay(Node):
    def __init__(self):
        super().__init__('odom_tf_relay')
        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        self.create_subscription(
            TFMessage,
            '/ackermann_steering_controller/tf_odometry',
            self._cb,
            10,
        )

    def _cb(self, msg: TFMessage):
        for t in msg.transforms:
            self.broadcaster.sendTransform(t)


def main():
    rclpy.init()
    rclpy.spin(OdomTfRelay())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
