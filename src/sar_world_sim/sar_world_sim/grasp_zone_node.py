import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GraspZoneNode(Node):
    def __init__(self):
        super().__init__('grasp_zone_node')

        self.current_owner = None
        self.current_poi = None

        self.request_sub = self.create_subscription(
            String,
            '/grasp_request',
            self.request_callback,
            10
        )

        self.release_sub = self.create_subscription(
            String,
            '/grasp_release',
            self.release_callback,
            10
        )

        self.reply_pub = self.create_publisher(
            String,
            '/grasp_reply',
            10
        )

        self.get_logger().info("Grasp zone node ready")

    def request_callback(self, msg: String):
        data = msg.data.strip()

        try:
            robot_id, poi_id = data.split(',')
        except Exception:
            self.get_logger().warn(f"Could not parse grasp request: {data}")
            return

        if self.current_owner is None:
            self.current_owner = robot_id
            self.current_poi = poi_id

            reply = String()
            reply.data = f"{robot_id},{poi_id},GRANTED"
            self.reply_pub.publish(reply)

            self.get_logger().info(
                f"Granted grasp zone to {robot_id} for {poi_id}"
            )
        elif self.current_owner == robot_id and self.current_poi == poi_id:
            reply = String()
            reply.data = f"{robot_id},{poi_id},GRANTED"
            self.reply_pub.publish(reply)
        else:
            reply = String()
            reply.data = f"{robot_id},{poi_id},DENIED"
            self.reply_pub.publish(reply)

            self.get_logger().info(
                f"Denied grasp zone to {robot_id} for {poi_id}; "
                f"currently owned by {self.current_owner} for {self.current_poi}"
            )

    def release_callback(self, msg: String):
        data = msg.data.strip()

        try:
            robot_id, poi_id = data.split(',')
        except Exception:
            self.get_logger().warn(f"Could not parse grasp release: {data}")
            return

        if self.current_owner == robot_id and self.current_poi == poi_id:
            self.get_logger().info(
                f"Released grasp zone from {robot_id} for {poi_id}"
            )
            self.current_owner = None
            self.current_poi = None


def main(args=None):
    rclpy.init(args=args)
    node = GraspZoneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
