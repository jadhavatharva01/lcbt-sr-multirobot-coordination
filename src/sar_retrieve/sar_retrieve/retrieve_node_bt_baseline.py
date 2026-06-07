import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import math


class RetrieveNode(Node):
    def __init__(self):
        super().__init__('retrieve_node')

        self.declare_parameter('robot_id', 'R1')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('battery', 0.9)
        self.declare_parameter('health', 0.95)

        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.x = self.get_parameter('x').get_parameter_value().double_value
        self.y = self.get_parameter('y').get_parameter_value().double_value
        self.battery = self.get_parameter('battery').get_parameter_value().double_value
        self.health = self.get_parameter('health').get_parameter_value().double_value

        self.poi_map = {
            1: (10.0, 10.0),
            2: (20.0, 5.0),
            3: (5.0, 25.0),
            4: (30.0, 30.0),
            5: (15.0, 18.0),
        }

        # Task and auction belief
        self.current_task_poi = None
        self.local_bids = {}
        self.auction_active = False
        self.auction_timer = None
        self.auction_window = 3.0
        self.local_winner = None

        # BT execution phase
        self.phase = "IDLE"
        self.phase_start_time = None

        self.detection_sub = self.create_subscription(
            String,
            '/detection',
            self.detection_callback,
            10
        )

        self.bid_sub = self.create_subscription(
            String,
            '/bids',
            self.bid_callback,
            10
        )

        self.bid_pub = self.create_publisher(
            String,
            '/bids',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/retrieval_status',
            10
        )

        # BT tick timer
        self.bt_timer = self.create_timer(1.0, self.tick_bt)

        self.get_logger().info(
            f"{self.robot_id} ready at ({self.x:.1f}, {self.y:.1f}), "
            f"battery={self.battery:.2f}, health={self.health:.2f}"
        )

    # -----------------------------
    # Communication callbacks
    # -----------------------------
    def detection_callback(self, msg: String):
        data = msg.data.strip()

        if not data.startswith("OBJECT_FOUND_AT_"):
            return

        try:
            poi_id = int(data.split("_")[-1])
        except ValueError:
            self.get_logger().warn(f"{self.robot_id}: could not parse detection message: {data}")
            return

        if poi_id not in self.poi_map:
            self.get_logger().warn(f"{self.robot_id}: unknown POI {poi_id}")
            return

        self.current_task_poi = poi_id
        self.local_bids = {}
        self.local_winner = None
        self.auction_active = True

        goal_x, goal_y = self.poi_map[poi_id]
        distance = math.sqrt((goal_x - self.x) ** 2 + (goal_y - self.y) ** 2)

        bid_value = distance + 10.0 * (1.0 - self.battery) + 10.0 * (1.0 - self.health)

        self.local_bids[self.robot_id] = bid_value

        bid_msg = String()
        bid_msg.data = f"{self.robot_id},{poi_id},{bid_value:.3f}"
        self.bid_pub.publish(bid_msg)

        self.get_logger().info(
            f"{self.robot_id}: detected task at POI {poi_id}, "
            f"distance={distance:.2f}, bid={bid_value:.3f}"
        )

        self.get_logger().info(f"{self.robot_id}: starting local auction window for POI {poi_id}")

        if self.auction_timer is not None:
            self.auction_timer.cancel()

        self.auction_timer = self.create_timer(self.auction_window, self.finish_local_auction)

    def bid_callback(self, msg: String):
        data = msg.data.strip()

        try:
            robot_id, poi_id, bid_value = data.split(',')
            poi_id = int(poi_id)
            bid_value = float(bid_value)
        except Exception:
            self.get_logger().warn(f"{self.robot_id}: could not parse bid: {data}")
            return

        if not self.auction_active:
            return

        if poi_id != self.current_task_poi:
            return

        self.local_bids[robot_id] = bid_value
        self.get_logger().info(f"{self.robot_id}: stored bid from {robot_id}: {bid_value:.3f}")

    def finish_local_auction(self):
        if not self.auction_active or self.current_task_poi is None:
            return

        if self.auction_timer is not None:
            self.auction_timer.cancel()
            self.auction_timer = None

        if not self.local_bids:
            self.get_logger().warn(f"{self.robot_id}: no bids available")
            self.auction_active = False
            return

        self.local_winner = min(self.local_bids, key=self.local_bids.get)
        winning_bid = self.local_bids[self.local_winner]

        self.get_logger().info(
            f"{self.robot_id}: local auction complete. "
            f"winner={self.local_winner}, POI={self.current_task_poi}, bid={winning_bid:.3f}"
        )

        self.auction_active = False

    # -----------------------------
    # Local Behavior Tree
    # -----------------------------
    def tick_bt(self):
        # Root Selector:
        # 1) Try retrieve sequence
        # 2) Else standby
        if self.retrieve_sequence():
            return
        self.standby_action()

    def retrieve_sequence(self):
        if not self.condition_task_detected():
            return False

        if not self.condition_i_am_winner():
            return False

        if self.phase == "IDLE":
            self.start_pre()

        elif self.phase == "PRE":
            self.run_pre()

        elif self.phase == "CRIT":
            self.run_crit()

        elif self.phase == "DONE":
            self.run_done()

        return True

    # -----------------------------
    # BT Conditions
    # -----------------------------
    def condition_task_detected(self):
        return self.current_task_poi is not None

    def condition_i_am_winner(self):
        return self.local_winner == self.robot_id

    # -----------------------------
    # BT Actions
    # -----------------------------
    def standby_action(self):
        # keep quiet for now
        pass

    def start_pre(self):
        self.phase = "PRE"
        self.phase_start_time = self.get_clock().now().nanoseconds / 1e9
        self.publish_status(f"{self.robot_id},POI_{self.current_task_poi},PRE")
        self.get_logger().info(
            f"{self.robot_id}: BT -> PRE started for POI {self.current_task_poi}"
        )

    def run_pre(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.phase_start_time >= 3.0:
            self.phase = "CRIT"
            self.phase_start_time = now
            self.publish_status(f"{self.robot_id},POI_{self.current_task_poi},CRIT")
            self.get_logger().info(
                f"{self.robot_id}: BT -> CRIT started for POI {self.current_task_poi}"
            )

    def run_crit(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.phase_start_time >= 2.0:
            self.phase = "DONE"
            self.publish_status(f"{self.robot_id},POI_{self.current_task_poi},DONE")
            self.get_logger().info(
                f"{self.robot_id}: BT -> DONE for POI {self.current_task_poi}"
            )

    def run_done(self):
        # Task completed; hold state for now
        pass

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RetrieveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()