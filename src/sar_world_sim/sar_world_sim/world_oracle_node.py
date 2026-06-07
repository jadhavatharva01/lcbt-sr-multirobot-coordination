#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sar_interfaces.msg import MissionEvent


class WorldOracleNode(Node):
    def __init__(self):
        super().__init__('world_oracle')

        # Read Phase 1A params (flat ROS 2-safe keys)
        self.declare_parameter('mission_true_object_poi_id', 'poi_unknown')
        self.true_object_poi_id = self.get_parameter(
            'mission_true_object_poi_id'
        ).get_parameter_value().string_value

        self.pub = self.create_publisher(MissionEvent, '/mission/events', 10)

        self.get_logger().info(
            f'world_oracle started (true_object_poi_id={self.true_object_poi_id})'
        )

        # Publish immediately (works with or without /clock)
        self.publish_startup_event()

    def publish_startup_event(self):
        msg = MissionEvent()
        msg.source_node = 'world_oracle'
        msg.event_type = 'STARTUP'
        msg.entity_id = 'oracle_1'
        msg.detail = f'phase1a oracle online, true_object_poi_id={self.true_object_poi_id}'
        msg.stamp = self.get_clock().now().to_msg()

        self.pub.publish(msg)
        self.get_logger().info('Published STARTUP MissionEvent on /mission/events')


def main(args=None):
    rclpy.init(args=args)
    node = WorldOracleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
