from pathlib import Path
import re

path = Path.home() / "phase1a_ws/src/sar_retrieve/sar_retrieve/retrieve_node.py"
text = path.read_text()

# 1. Add robot_states after local_bids initialization if not already present
if "self.robot_states" not in text:
    text = text.replace(
        "        self.local_bids = {}\n",
        "        self.local_bids = {}\n"
        "        self.robot_states = {}\n",
        1
    )

# 2. Reset robot_states whenever a new detection starts
text = text.replace(
    "        self.local_bids = {}\n        self.local_winner = None",
    "        self.local_bids = {}\n"
    "        self.robot_states = {}\n"
    "        self.local_winner = None"
)

# 3. Store own robot state after own bid is saved
text = text.replace(
    "        self.local_bids[self.robot_id] = bid_value\n\n        bid_msg = String()",
    "        self.local_bids[self.robot_id] = bid_value\n"
    "        self.robot_states[self.robot_id] = {\n"
    "            'battery': self.battery,\n"
    "            'health': self.health,\n"
    "            'obstacle_density': self.obstacle_density,\n"
    "            'effective_speed': self.effective_speed,\n"
    "        }\n\n"
    "        bid_msg = String()"
)

# 4. Replace bid message with stateful bid message
text = text.replace(
    '        bid_msg.data = f"{self.robot_id},{poi_id},{bid_value:.3f}"',
    '        bid_msg.data = ('
    'f"{self.robot_id},{poi_id},{bid_value:.3f},"'
    'f"{self.battery:.3f},{self.health:.3f},"'
    'f"{self.obstacle_density:.3f},{self.effective_speed:.3f}")'
)

# 5. Replace process_bid with stateful version
process_bid_new = r'''    def process_bid(self, data: str):
        try:
            parts = data.split(',')

            # Backward compatible:
            # old format: robot_id,poi_id,bid
            # new format: robot_id,poi_id,bid,battery,health,obstacle_density,effective_speed
            if len(parts) == 3:
                robot_id, poi_id, bid_value = parts
                poi_id = int(poi_id)
                bid_value = float(bid_value)

                battery = None
                health = None
                obstacle_density = None
                effective_speed = None

            elif len(parts) == 7:
                robot_id, poi_id, bid_value, battery, health, obstacle_density, effective_speed = parts
                poi_id = int(poi_id)
                bid_value = float(bid_value)
                battery = float(battery)
                health = float(health)
                obstacle_density = float(obstacle_density)
                effective_speed = float(effective_speed)

            else:
                raise ValueError(f"unexpected bid format with {len(parts)} fields")

        except Exception:
            self.get_logger().warn(f"{self.robot_id}: could not parse bid: {data}")
            return

        if not self.auction_active:
            return

        if poi_id != self.current_task_poi:
            return

        self.local_bids[robot_id] = bid_value

        if effective_speed is not None:
            self.robot_states[robot_id] = {
                'battery': battery,
                'health': health,
                'obstacle_density': obstacle_density,
                'effective_speed': effective_speed,
            }

        self.get_logger().info(
            f"{self.robot_id}: stored bid from {robot_id}: {bid_value:.3f}"
        )

'''

text = re.sub(
    r"    def process_bid\(self, data: str\):.*?(?=    def finish_local_auction\(self\):)",
    process_bid_new,
    text,
    flags=re.DOTALL
)

# 6. Replace finish_local_auction so lease uses winner's effective speed
finish_new = r'''    def finish_local_auction(self):
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

        winner_state = self.robot_states.get(self.local_winner, {})
        winner_effective_speed = winner_state.get(
            'effective_speed',
            self.compute_effective_speed(self.local_winner)
        )

        winner_effective_speed = max(0.01, winner_effective_speed)
        travel_time = distance / winner_effective_speed

        lease_duration = (
            travel_time
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
                f"distance={distance:.3f};"
                f"effective_speed={winner_effective_speed:.3f};"
                f"travel_time={travel_time:.3f};"
                f"lease_duration={lease_duration:.3f};"
                f"lease_expiry={self.lease_expiry:.3f}"
            )
        )

        self.get_logger().info(
            f"{self.robot_id}: local auction complete. "
            f"winner={self.local_winner}, POI={self.current_task_poi}, "
            f"bid={self.winning_bid:.3f}, lease_owner={self.lease_owner}, "
            f"lease_epoch={self.lease_epoch}, "
            f"winner_effective_speed={winner_effective_speed:.3f}, "
            f"travel_time={travel_time:.3f}, "
            f"lease_duration={lease_duration:.3f}, "
            f"lease_expiry={self.lease_expiry:.2f}"
        )

        self.auction_active = False

'''

text = re.sub(
    r"    def finish_local_auction\(self\):.*?(?=    # -----------------------------\n    # Intent layer)",
    finish_new,
    text,
    flags=re.DOTALL
)

path.write_text(text)
print(f"Patched: {path}")
