import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random

class SearchNode(Node):
    def __init__(self):
        super().__init__('search_node')

        self.epoch = 0
        self.publisher = self.create_publisher(String, '/detection', 10)
        self.timer = self.create_timer(2.0, self.search_callback)

        self.pois = [1, 2, 3, 4, 5]
        self.true_poi = random.choice(self.pois)
        self.visited = set()
        self.found = False

        self.get_logger().info(f"True object at POI {self.true_poi}")

    def search_callback(self):
        if self.found:
            return

        if len(self.visited) == len(self.pois):
            self.get_logger().info("Search complete: object not found")
            self.found = True
            return

        next_poi = random.choice([p for p in self.pois if p not in self.visited])
        self.visited.add(next_poi)

        self.get_logger().info(f"Visiting POI {next_poi}")

        if next_poi == self.true_poi:
            self.epoch += 1

            msg = String()
            msg.data = f"OBJECT_FOUND_AT_{next_poi}_EPOCH_{self.epoch}"
            self.publisher.publish(msg)

            self.get_logger().info(f"Published detection with epoch {self.epoch}")

            self.get_logger().info(f"Object FOUND at POI {next_poi}")
            self.found = True
            self.timer.cancel()
            self.get_logger().info("Search stopped. Waiting for next phase.")


def main(args=None):
    rclpy.init(args=args)
    node = SearchNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()