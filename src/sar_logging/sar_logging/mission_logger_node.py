#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sar_interfaces.msg import MissionEvent


class MissionLoggerNode(Node):
    def __init__(self):
        super().__init__('mission_logger')


        self.sub = self.create_subscription(
            MissionEvent,
            '/mission/events',
            self.event_callback,
            10
        )

        self.get_logger().info('mission_logger started. Listening on /mission/events')

    def event_callback(self, msg: MissionEvent):
        self.get_logger().info(
            f"[EVENT] source={msg.source_node} type={msg.event_type} "
            f"entity={msg.entity_id} detail={msg.detail} "
            f"stamp=({msg.stamp.sec}.{msg.stamp.nanosec:09d})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MissionLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
