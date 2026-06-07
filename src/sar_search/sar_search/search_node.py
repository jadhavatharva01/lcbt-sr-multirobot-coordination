import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random


class SearchNode(Node):
    def __init__(self):
        super().__init__('search_node')

        self.declare_parameter('robot_id', 'S1')
        self.declare_parameter('num_pois', 10)
        self.declare_parameter('true_poi', -1)
        self.declare_parameter('seed', 42)
        self.declare_parameter('visit_period', 2.0)

        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.num_pois = self.get_parameter('num_pois').get_parameter_value().integer_value
        self.true_poi_param = self.get_parameter('true_poi').get_parameter_value().integer_value
        self.seed = self.get_parameter('seed').get_parameter_value().integer_value
        self.visit_period = self.get_parameter('visit_period').get_parameter_value().double_value

        self.rng = random.Random(self.seed)

        self.epoch = 0
        self.pois = list(range(1, self.num_pois + 1))

        if self.true_poi_param in self.pois:
            self.true_poi = self.true_poi_param
        else:
            self.true_poi = self.rng.choice(self.pois)

        self.visited = set()
        self.found = False
        self.mission_started = False
        self.start_time = self.now_sec()

        self.detection_pub = self.create_publisher(String, '/detection', 10)
        self.event_pub = self.create_publisher(String, '/mission_event', 10)

        self.timer = self.create_timer(self.visit_period, self.search_callback)

        self.get_logger().info(
            f"{self.robot_id}: search node ready with {self.num_pois} POIs. "
            f"True object at POI {self.true_poi}"
        )

    def publish_event(self, event_type, poi_id=-1, epoch=0, value=""):
        msg = String()
        t = self.now_sec()

        msg.data = (
            f"time={t:.3f},"
            f"event={event_type},"
            f"robot={self.robot_id},"
            f"poi={poi_id},"
            f"epoch={epoch},"
            f"value={value}"
        )

        self.event_pub.publish(msg)

    def search_callback(self):
        if not self.mission_started:
            self.mission_started = True
            self.start_time = self.now_sec()
            self.publish_event("MISSION_START")
            self.get_logger().info(f"{self.robot_id}: mission started")

        if self.found:
            return

        if len(self.visited) == len(self.pois):
            self.get_logger().info("Search complete: object not found")
            self.publish_event("SEARCH_COMPLETE_NOT_FOUND")
            self.found = True
            self.timer.cancel()
            return

        next_poi = self.rng.choice([p for p in self.pois if p not in self.visited])
        self.visited.add(next_poi)

        self.get_logger().info(f"{self.robot_id}: visiting POI {next_poi}")
        self.publish_event("POI_VISIT", poi_id=next_poi)

        if next_poi == self.true_poi:
            self.epoch += 1

            detection_msg = String()
            detection_msg.data = f"OBJECT_FOUND_AT_{next_poi}_EPOCH_{self.epoch}"
            self.detection_pub.publish(detection_msg)

            self.publish_event("DETECTION", poi_id=next_poi, epoch=self.epoch)

            self.get_logger().info(f"{self.robot_id}: published detection with epoch {self.epoch}")
            self.get_logger().info(f"{self.robot_id}: object FOUND at POI {next_poi}")

            self.found = True
            self.timer.cancel()

            self.publish_event("SEARCH_STOPPED", poi_id=next_poi, epoch=self.epoch)
            self.get_logger().info("Search stopped. Waiting for retrieval phase.")

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()