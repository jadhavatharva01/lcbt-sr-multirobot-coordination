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
        self.declare_parameter('bid_delay', 0.0)

        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.x = self.get_parameter('x').get_parameter_value().double_value
        self.y = self.get_parameter('y').get_parameter_value().double_value
        self.battery = self.get_parameter('battery').get_parameter_value().double_value
        self.health = self.get_parameter('health').get_parameter_value().double_value
        self.bid_delay = self.get_parameter('bid_delay').get_parameter_value().double_value

        self.poi_map = {
            1: (10.0, 10.0),
            2: (20.0, 5.0),
            3: (5.0, 25.0),
            4: (30.0, 30.0),
            5: (15.0, 18.0),
            6: (40.0, 12.0),
            7: (8.0, 40.0),
            8: (35.0, 42.0),
            9: (50.0, 25.0),
            10: (22.0, 38.0),
        }

        self.current_task_poi = None
        self.current_epoch = 0
        self.local_bids = {}
        self.auction_active = False
        self.auction_timer = None
        self.auction_window = 3.0
        self.local_winner = None
        self.my_bid = None
        self.winning_bid = None

        self.lease_owner = None
        self.lease_epoch = 0
        self.lease_expiry = 0.0

        self.nominal_speed = 5.0
        self.crit_duration = 2.0
        self.comm_margin = 1.5
        self.safety_margin = 1.0

        # Intent / reservation layer
        self.intent_wait_duration = 1.0
        self.intent_wait_start_time = None
        self.intent_published = False
        self.intent_conflict_detected = False

        # Heartbeat layer
        self.heartbeat_period = 0.75
        self.heartbeat_timer = self.create_timer(self.heartbeat_period, self.publish_heartbeat)

        self.phase = "IDLE"
        self.phase_start_time = None
        self.task_blocked = False

        self.waiting_for_grasp = False
        self.grasp_granted = False

        self.detection_sub = self.create_subscription(
            String, '/detection', self.detection_callback, 10
        )

        self.bid_sub = self.create_subscription(
            String, '/bids', self.bid_callback, 10
        )

        self.intent_sub = self.create_subscription(
            String, '/retrieval_intent', self.intent_callback, 10
        )

        self.heartbeat_sub = self.create_subscription(
            String, '/retrieval_heartbeat', self.heartbeat_callback, 10
        )

        self.grasp_reply_sub = self.create_subscription(
            String, '/grasp_reply', self.grasp_reply_callback, 10
        )

        self.bid_pub = self.create_publisher(String, '/bids', 10)
        self.intent_pub = self.create_publisher(String, '/retrieval_intent', 10)
        self.heartbeat_pub = self.create_publisher(String, '/retrieval_heartbeat', 10)
        self.status_pub = self.create_publisher(String, '/retrieval_status', 10)
        self.event_pub = self.create_publisher(String, '/mission_event', 10)
        self.grasp_request_pub = self.create_publisher(String, '/grasp_request', 10)
        self.grasp_release_pub = self.create_publisher(String, '/grasp_release', 10)

        self.get_logger().info(
            f"{self.robot_id} ready at ({self.x:.1f}, {self.y:.1f}), "
            f"battery={self.battery:.2f}, health={self.health:.2f}, "
            f"bid_delay={self.bid_delay:.2f}s"
        )

    # -----------------------------
    # Event logging
    # -----------------------------
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

    # -----------------------------
    # Detection and bid callbacks
    # -----------------------------
    def detection_callback(self, msg: String):
        data = msg.data.strip()

        if not data.startswith("OBJECT_FOUND_AT_"):
            return

        try:
            parts = data.split("_")
            poi_id = int(parts[3])
            epoch = int(parts[5])
        except Exception:
            self.get_logger().warn(f"{self.robot_id}: could not parse detection message: {data}")
            return

        if poi_id not in self.poi_map:
            self.get_logger().warn(f"{self.robot_id}: unknown POI {poi_id}")
            return

        if epoch < self.current_epoch:
            self.get_logger().warn(
                f"{self.robot_id}: ignoring stale detection for POI {poi_id} with epoch {epoch}, "
                f"current epoch is {self.current_epoch}"
            )
            self.publish_event(
                "STALE_DETECTION_IGNORED",
                poi_id=poi_id,
                epoch=epoch,
                value=f"current_epoch={self.current_epoch}"
            )
            return

        self.current_epoch = epoch
        self.lease_epoch = epoch

        self.current_task_poi = poi_id
        self.local_bids = {}
        self.local_winner = None
        self.my_bid = None
        self.winning_bid = None
        self.auction_active = True
        self.phase = "IDLE"
        self.task_blocked = False

        self.intent_wait_start_time = None
        self.intent_published = False
        self.intent_conflict_detected = False

        self.waiting_for_grasp = False
        self.grasp_granted = False

        goal_x, goal_y = self.poi_map[poi_id]
        distance = math.sqrt((goal_x - self.x) ** 2 + (goal_y - self.y) ** 2)

        bid_value = distance + 10.0 * (1.0 - self.battery) + 10.0 * (1.0 - self.health)

        self.my_bid = bid_value
        self.local_bids[self.robot_id] = bid_value

        bid_msg = String()
        bid_msg.data = f"{self.robot_id},{poi_id},{bid_value:.3f}"
        self.bid_pub.publish(bid_msg)

        self.publish_event("BID", poi_id=poi_id, epoch=epoch, value=f"bid={bid_value:.3f}")

        self.get_logger().info(
            f"{self.robot_id}: accepted detection at POI {poi_id} with epoch {epoch}, "
            f"distance={distance:.2f}, bid={bid_value:.3f}"
        )

        if self.auction_timer is not None:
            self.auction_timer.cancel()

        self.auction_timer = self.create_timer(self.auction_window, self.finish_local_auction)

    def bid_callback(self, msg: String):
        data = msg.data.strip()

        if self.bid_delay > 0.0:
            self.create_timer(self.bid_delay, lambda d=data: self.process_bid(d))
        else:
            self.process_bid(data)

    def process_bid(self, data: str):
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
        self.winning_bid = self.local_bids[self.local_winner]

        now = self.now_sec()
        self.lease_owner = self.local_winner
        self.lease_epoch = self.current_epoch

        winner_x, winner_y = self.get_robot_position(self.local_winner)
        goal_x, goal_y = self.poi_map[self.current_task_poi]
        distance = math.sqrt((goal_x - winner_x) ** 2 + (goal_y - winner_y) ** 2)

        lease_duration = (
            distance / self.nominal_speed
            + self.crit_duration
            + self.comm_margin
            + self.safety_margin
        )

        self.lease_expiry = now + lease_duration

        self.publish_event(
            "LOCAL_AUCTION_RESULT",
            poi_id=self.current_task_poi,
            epoch=self.current_epoch,
            value=(
                f"winner={self.local_winner};"
                f"winning_bid={self.winning_bid:.3f};"
                f"lease_owner={self.lease_owner};"
                f"lease_epoch={self.lease_epoch};"
                f"lease_expiry={self.lease_expiry:.3f}"
            )
        )

        self.get_logger().info(
            f"{self.robot_id}: local auction complete. "
            f"winner={self.local_winner}, POI={self.current_task_poi}, "
            f"bid={self.winning_bid:.3f}, lease_owner={self.lease_owner}, "
            f"lease_epoch={self.lease_epoch}, "
            f"lease_expiry={self.lease_expiry:.2f}"
        )

        self.auction_active = False

    # -----------------------------
    # Intent layer
    # -----------------------------
    def publish_intent(self):
        if self.current_task_poi is None or self.my_bid is None:
            return

        expiry = self.lease_expiry

        msg = String()
        msg.data = (
            f"{self.robot_id},"
            f"{self.current_task_poi},"
            f"{self.current_epoch},"
            f"{self.my_bid:.3f},"
            f"{expiry:.3f}"
        )
        self.intent_pub.publish(msg)

        self.publish_event(
            "INTENT",
            poi_id=self.current_task_poi,
            epoch=self.current_epoch,
            value=f"bid={self.my_bid:.3f};expiry={expiry:.3f}"
        )

        self.get_logger().info(
            f"{self.robot_id}: broadcast INTENT for POI {self.current_task_poi}, "
            f"epoch={self.current_epoch}, bid={self.my_bid:.3f}"
        )

    def intent_callback(self, msg: String):
        data = msg.data.strip()

        try:
            other_robot, poi_id, epoch, other_bid, expiry = data.split(',')
            poi_id = int(poi_id)
            epoch = int(epoch)
            other_bid = float(other_bid)
            expiry = float(expiry)
        except Exception:
            self.get_logger().warn(f"{self.robot_id}: could not parse intent: {data}")
            return

        if other_robot == self.robot_id:
            return

        if self.current_task_poi is None:
            return

        if poi_id != self.current_task_poi or epoch != self.current_epoch:
            return

        if self.my_bid is None:
            return

        other_is_better = self.is_other_better(other_robot, other_bid)

        if other_is_better and self.phase in ["INTENT_WAIT", "IDLE"]:
            self.intent_conflict_detected = True
            self.task_blocked = True
            self.phase = "IDLE"

            self.publish_event(
                "INTENT_CONFLICT_BLOCKED",
                poi_id=self.current_task_poi,
                epoch=self.current_epoch,
                value=f"blocked_by={other_robot};other_bid={other_bid:.3f};my_bid={self.my_bid:.3f}"
            )

            self.get_logger().warn(
                f"{self.robot_id}: blocked before PRE by better INTENT from {other_robot} "
                f"(other_bid={other_bid:.3f}, my_bid={self.my_bid:.3f})"
            )

    # -----------------------------
    # Heartbeat layer
    # -----------------------------
    def publish_heartbeat(self):
        if self.current_task_poi is None:
            return

        if self.phase not in ["PRE", "CRIT"]:
            return

        if self.my_bid is None:
            return

        msg = String()
        msg.data = (
            f"{self.robot_id},"
            f"{self.current_task_poi},"
            f"{self.current_epoch},"
            f"{self.phase},"
            f"{self.my_bid:.3f},"
            f"{self.lease_expiry:.3f}"
        )
        self.heartbeat_pub.publish(msg)

        self.publish_event(
            "HEARTBEAT",
            poi_id=self.current_task_poi,
            epoch=self.current_epoch,
            value=(
                f"phase={self.phase};"
                f"bid={self.my_bid:.3f};"
                f"lease_expiry={self.lease_expiry:.3f}"
            )
        )

    def heartbeat_callback(self, msg: String):
        data = msg.data.strip()

        try:
            other_robot, poi_id, epoch, other_phase, other_bid, expiry = data.split(',')
            poi_id = int(poi_id)
            epoch = int(epoch)
            other_bid = float(other_bid)
            expiry = float(expiry)
        except Exception:
            self.get_logger().warn(f"{self.robot_id}: could not parse heartbeat: {data}")
            return

        if other_robot == self.robot_id:
            return

        if self.current_task_poi is None:
            return

        if poi_id != self.current_task_poi or epoch != self.current_epoch:
            return

        if self.my_bid is None:
            return

        if other_phase not in ["PRE", "CRIT"]:
            return

        other_is_better = self.is_other_better(other_robot, other_bid)

        # If I am still preparing or moving and I hear a better active robot,
        # I back off before getting near the object.
        if other_is_better and self.phase in ["INTENT_WAIT", "PRE"]:
            self.task_blocked = True
            self.phase = "IDLE"

            self.publish_event(
                "HEARTBEAT_CONFLICT_BLOCKED",
                poi_id=self.current_task_poi,
                epoch=self.current_epoch,
                value=(
                    f"blocked_by={other_robot};"
                    f"other_phase={other_phase};"
                    f"other_bid={other_bid:.3f};"
                    f"my_bid={self.my_bid:.3f}"
                )
            )

            self.get_logger().warn(
                f"{self.robot_id}: backed off due to HEARTBEAT from better active robot "
                f"{other_robot} phase={other_phase}, other_bid={other_bid:.3f}, my_bid={self.my_bid:.3f}"
            )

    def is_other_better(self, other_robot, other_bid):
        if self.my_bid is None:
            return False

        if other_bid < self.my_bid:
            return True

        if other_bid == self.my_bid and other_robot < self.robot_id:
            return True

        return False

    # -----------------------------
    # Grasp-zone callbacks
    # -----------------------------
    def grasp_reply_callback(self, msg: String):
        data = msg.data.strip()

        try:
            robot_id, poi_id_text, decision = data.split(',')
        except Exception:
            self.get_logger().warn(f"{self.robot_id}: could not parse grasp reply: {data}")
            return

        if robot_id != self.robot_id:
            return

        expected_poi = f"POI_{self.current_task_poi}"
        if poi_id_text != expected_poi:
            return

        self.waiting_for_grasp = False

        if decision == "GRANTED":
            self.grasp_granted = True
            self.publish_event("GRASP_GRANTED", poi_id=self.current_task_poi, epoch=self.current_epoch)
            self.get_logger().info(f"{self.robot_id}: grasp-zone access GRANTED for {poi_id_text}")
        else:
            self.grasp_granted = False
            self.task_blocked = True
            self.phase = "IDLE"
            self.publish_event("GRASP_DENIED", poi_id=self.current_task_poi, epoch=self.current_epoch)
            self.get_logger().warn(
                f"{self.robot_id}: grasp-zone access DENIED for {poi_id_text}. Standing down."
            )

    # -----------------------------
    # Helper methods
    # -----------------------------
    def get_robot_position(self, robot_id):
        if robot_id == self.robot_id:
            return self.x, self.y

        teammate_positions = {
            'R1': (0.0, 0.0),
            'R2': (25.0, 10.0),
            'R3': (45.0, 5.0),
            'R4': (5.0, 45.0),
            'R5': (50.0, 50.0),
        }

        return teammate_positions.get(robot_id, (0.0, 0.0))

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def request_grasp_zone(self):
        if self.current_task_poi is None:
            return

        msg = String()
        msg.data = f"{self.robot_id},POI_{self.current_task_poi}"
        self.grasp_request_pub.publish(msg)
        self.waiting_for_grasp = True

        self.publish_event("GRASP_REQUEST", poi_id=self.current_task_poi, epoch=self.current_epoch)

        self.get_logger().info(
            f"{self.robot_id}: requested grasp-zone access for POI_{self.current_task_poi}"
        )

    def release_grasp_zone(self):
        if self.current_task_poi is None:
            return

        msg = String()
        msg.data = f"{self.robot_id},POI_{self.current_task_poi}"
        self.grasp_release_pub.publish(msg)

        self.publish_event("GRASP_RELEASE", poi_id=self.current_task_poi, epoch=self.current_epoch)

        self.get_logger().info(
            f"{self.robot_id}: released grasp-zone access for POI_{self.current_task_poi}"
        )

    # -----------------------------
    # Local Behavior Tree
    # -----------------------------
    def tick_bt(self):
        if self.retrieve_sequence():
            return
        self.standby_action()

    def retrieve_sequence(self):
        if not self.condition_task_detected():
            return False

        if not self.condition_i_am_winner():
            return False

        if self.phase == "IDLE":
            self.start_intent_wait()

        elif self.phase == "INTENT_WAIT":
            self.run_intent_wait()

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
        return self.current_task_poi is not None and not self.task_blocked

    def condition_i_am_winner(self):
        return self.local_winner == self.robot_id

    def condition_valid_lease(self):
        return (self.lease_owner == self.robot_id) and (self.now_sec() <= self.lease_expiry)

    # -----------------------------
    # BT Actions
    # -----------------------------
    def standby_action(self):
        pass

    def start_intent_wait(self):
        self.phase = "INTENT_WAIT"
        self.intent_wait_start_time = self.now_sec()
        self.intent_published = True
        self.intent_conflict_detected = False

        self.publish_intent()

        self.publish_event(
            "INTENT_WAIT_START",
            poi_id=self.current_task_poi,
            epoch=self.current_epoch,
            value=f"wait={self.intent_wait_duration:.3f}"
        )

        self.get_logger().info(
            f"{self.robot_id}: INTENT_WAIT started before PRE for POI {self.current_task_poi}"
        )

    def run_intent_wait(self):
        if self.intent_conflict_detected or self.task_blocked:
            return

        now = self.now_sec()

        if now - self.intent_wait_start_time >= self.intent_wait_duration:
            self.start_pre()

    def start_pre(self):
        self.phase = "PRE"
        self.phase_start_time = self.now_sec()
        self.publish_status(f"{self.robot_id},POI_{self.current_task_poi},PRE")
        self.publish_event("PRE", poi_id=self.current_task_poi, epoch=self.current_epoch)

        self.get_logger().info(
            f"{self.robot_id}: BT -> PRE started for POI {self.current_task_poi}"
        )

    def run_pre(self):
        now = self.now_sec()

        if now - self.phase_start_time < 3.0:
            return

        if not self.condition_valid_lease():
            self.phase = "IDLE"
            self.task_blocked = True
            self.publish_event(
                "LEASE_EXPIRED_BEFORE_CRIT",
                poi_id=self.current_task_poi,
                epoch=self.current_epoch,
                value=f"lease_owner={self.lease_owner};lease_expiry={self.lease_expiry:.3f}"
            )
            self.get_logger().warn(
                f"{self.robot_id}: CRIT blocked, lease invalid or expired. Standing down for this task."
            )
            return

        if self.grasp_granted:
            self.phase = "CRIT"
            self.phase_start_time = now
            self.publish_status(f"{self.robot_id},POI_{self.current_task_poi},CRIT")
            self.publish_event("CRIT", poi_id=self.current_task_poi, epoch=self.current_epoch)

            self.get_logger().info(
                f"{self.robot_id}: BT -> CRIT started for POI {self.current_task_poi}"
            )
            return

        if not self.waiting_for_grasp:
            self.request_grasp_zone()

    def run_crit(self):
        now = self.now_sec()

        if now - self.phase_start_time >= 2.0:
            self.phase = "DONE"
            self.publish_status(f"{self.robot_id},POI_{self.current_task_poi},DONE")
            self.publish_event("DONE", poi_id=self.current_task_poi, epoch=self.current_epoch)

            self.get_logger().info(
                f"{self.robot_id}: BT -> DONE for POI {self.current_task_poi}"
            )

            self.release_grasp_zone()

    def run_done(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = RetrieveNode()
    node.bt_timer = node.create_timer(1.0, node.tick_bt)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()