import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LeaderNode(Node):
    def __init__(self):
        super().__init__('leader_node')

        self.bid_sub = self.create_subscription(
            String,
            '/bids',
            self.bid_callback,
            10
        )

        self.winner_pub = self.create_publisher(
            String,
            '/winner',
            10
        )

        self.current_bids = {}
        self.bid_timer = None
        self.bid_window = 3.0
        self.current_poi = None

        self.get_logger().info("Leader node ready")

    def bid_callback(self, msg: String):
        data = msg.data.strip()

        try:
            robot_id, poi_id, bid_value = data.split(',')
            poi_id = int(poi_id)
            bid_value = float(bid_value)
        except Exception:
            self.get_logger().warn(f"Could not parse bid message: {data}")
            return

        if self.current_poi is None:
            self.current_poi = poi_id
            self.current_bids = {}

            self.get_logger().info(f"Starting auction for POI {poi_id}")

            self.bid_timer = self.create_timer(self.bid_window, self.finish_auction)

        if poi_id != self.current_poi:
            self.get_logger().warn(
                f"Ignoring bid for POI {poi_id}, current auction is for POI {self.current_poi}"
            )
            return

        self.current_bids[robot_id] = bid_value
        self.get_logger().info(f"Received bid from {robot_id}: {bid_value:.3f}")

    def finish_auction(self):
        if not self.current_bids:
            self.get_logger().warn("No bids received")
            self.current_poi = None
            self.bid_timer.cancel()
            return

        winner = min(self.current_bids, key=self.current_bids.get)
        winning_bid = self.current_bids[winner]

        msg = String()
        msg.data = f"{winner},{self.current_poi},{winning_bid:.3f}"
        self.winner_pub.publish(msg)

        self.get_logger().info(
            f"Auction complete: winner={winner}, POI={self.current_poi}, bid={winning_bid:.3f}"
        )

        self.bid_timer.cancel()
        self.bid_timer = None
        self.current_bids = {}
        self.current_poi = None


def main(args=None):
    rclpy.init(args=args)
    node = LeaderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
