#!/usr/bin/env python3
"""
Dynamic BT / lease-certified scoped re-auction RViz simulator.

Visualizes two methods side-by-side in RViz:
  1) CBBA-like consensus with marginal route bids
  2) Our lease-certified Behavior Tree with scoped re-auction approximation

This is a visualization/demo node. It publishes MarkerArray messages only;
no Gazebo, Nav2, or custom messages are required.

Run after sourcing ROS 2:
  python3 dynamic_bt_rviz_sim_node.py

RViz fixed frame: world
MarkerArray topic: /dynamic_bt_sim/markers
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, TransformStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import StaticTransformBroadcaster


Vec2 = Tuple[float, float]


@dataclass
class Task:
    task_id: str
    poi_id: int
    release_time: float
    status: str = "unreleased"  # unreleased, pending, assigned, done
    owner: Optional[str] = None
    assigned_time: Optional[float] = None
    done_time: Optional[float] = None
    allocation_due_time: Optional[float] = None
    allocation_started_time: Optional[float] = None
    planned_winner: Optional[str] = None
    planned_route: Optional[List[int]] = None
    bid_margin: float = 0.0


@dataclass
class Robot:
    robot_id: str
    start_pos: Vec2
    current_pos: Vec2
    battery: float
    health: float
    remaining_route: List[int] = field(default_factory=list)
    completed_route: List[int] = field(default_factory=list)
    distance: float = 0.0
    phase: str = "IDLE"  # IDLE, PRE, CRIT
    active_task: Optional[str] = None


@dataclass
class MethodWorld:
    name: str
    label: str
    x_offset: float
    robots: Dict[str, Robot]
    tasks: List[Task]
    pois: Dict[int, Vec2]
    total_messages: int = 0
    total_allocation_delay: float = 0.0
    allocation_count: int = 0
    completed_count: int = 0


def euclidean(a: Vec2, b: Vec2) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def interpolate(a: Vec2, b: Vec2, alpha: float) -> Vec2:
    alpha = max(0.0, min(1.0, alpha))
    return (a[0] + alpha * (b[0] - a[0]), a[1] + alpha * (b[1] - a[1]))


def route_distance_from(start_pos: Vec2, route: List[int], pois: Dict[int, Vec2]) -> float:
    total = 0.0
    cur = start_pos
    for poi_id in route:
        nxt = pois[poi_id]
        total += euclidean(cur, nxt)
        cur = nxt
    return total


def best_insert_route(current_pos: Vec2, remaining_route: List[int], new_poi_id: int, pois: Dict[int, Vec2]) -> Tuple[float, List[int]]:
    current_cost = route_distance_from(current_pos, remaining_route, pois)
    best_extra = None
    best_route: List[int] = []
    for insert_idx in range(len(remaining_route) + 1):
        candidate = remaining_route[:insert_idx] + [new_poi_id] + remaining_route[insert_idx:]
        candidate_cost = route_distance_from(current_pos, candidate, pois)
        extra = candidate_cost - current_cost
        if best_extra is None or extra < best_extra:
            best_extra = extra
            best_route = candidate
    return float(best_extra if best_extra is not None else 0.0), best_route


def robot_state_penalty(robot: Robot) -> float:
    return 5.0 * (1.0 - robot.battery) + 5.0 * (1.0 - robot.health)


def make_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


class DynamicBtRvizSim(Node):
    def __init__(self) -> None:
        super().__init__("dynamic_bt_rviz_sim")

        # Parameters are intentionally declared so you can override them from command line.
        self.declare_parameter("num_pois", 36)
        self.declare_parameter("num_robots", 4)
        self.declare_parameter("num_tasks", 10)
        self.declare_parameter("world_size", 60.0)
        self.declare_parameter("release_gap", 1.5)
        self.declare_parameter("robot_speed", 5.0)
        self.declare_parameter("sim_rate_hz", 10.0)
        self.declare_parameter("time_scale", 1.0)
        self.declare_parameter("seed", 2606)
        self.declare_parameter("cbba_rounds", 3)
        self.declare_parameter("cbba_round_delay", 1.5)
        self.declare_parameter("ours_base_alloc_delay", 0.5)
        self.declare_parameter("ours_min_wait", 0.3)
        self.declare_parameter("ours_max_wait", 1.5)
        self.declare_parameter("ours_margin_scale", 1.0)
        self.declare_parameter("ours_heartbeat_msgs", 4)
        self.declare_parameter("auto_reset", True)
        self.declare_parameter("reset_after_done_sec", 8.0)
        self.declare_parameter("enable_csv_logging", True)
        self.declare_parameter("log_dir", str(Path.home() / "phase1a_ws/results/dynamic_bt_rviz"))

        self.num_pois = int(self.get_parameter("num_pois").value)
        self.num_robots = int(self.get_parameter("num_robots").value)
        self.num_tasks = int(self.get_parameter("num_tasks").value)
        self.world_size = float(self.get_parameter("world_size").value)
        self.release_gap = float(self.get_parameter("release_gap").value)
        self.speed = float(self.get_parameter("robot_speed").value)
        self.time_scale = float(self.get_parameter("time_scale").value)
        self.seed = int(self.get_parameter("seed").value)
        self.cbba_rounds = int(self.get_parameter("cbba_rounds").value)
        self.cbba_round_delay = float(self.get_parameter("cbba_round_delay").value)
        self.ours_base_alloc_delay = float(self.get_parameter("ours_base_alloc_delay").value)
        self.ours_min_wait = float(self.get_parameter("ours_min_wait").value)
        self.ours_max_wait = float(self.get_parameter("ours_max_wait").value)
        self.ours_margin_scale = float(self.get_parameter("ours_margin_scale").value)
        self.ours_heartbeat_msgs = int(self.get_parameter("ours_heartbeat_msgs").value)
        self.auto_reset = bool(self.get_parameter("auto_reset").value)
        self.reset_after_done_sec = float(self.get_parameter("reset_after_done_sec").value)
        self.enable_csv_logging = bool(self.get_parameter("enable_csv_logging").value)
        self.log_dir = Path(str(self.get_parameter("log_dir").value))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "dynamic_bt_rviz_events.csv"
        self.summary_path = self.log_dir / "dynamic_bt_rviz_summary.csv"
        self._logged_done_tasks = set()
        self._init_csv_logs()

        self.pub = self.create_publisher(MarkerArray, "/dynamic_bt_sim/markers", 10)

        # RViz needs a TF frame to exist for MarkerArray displays.
        # Publish an identity world -> map transform so the demo works
        # without requiring a separate static_transform_publisher terminal.
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_map_frame()

        self.sim_time = 0.0
        self.done_since: Optional[float] = None
        self.rng = random.Random(self.seed)
        self.worlds = self._create_worlds()
        self._logged_done_tasks = set()

        rate = float(self.get_parameter("sim_rate_hz").value)
        self.dt_wall = 1.0 / max(rate, 0.1)
        self.timer = self.create_timer(self.dt_wall, self._on_timer)

        self.get_logger().info("Dynamic BT RViz simulator ready.")
        self.get_logger().info("RViz fixed frame: world")
        self.get_logger().info("Published static TF: world -> map")
        self.get_logger().info("MarkerArray topic: /dynamic_bt_sim/markers")
        if self.enable_csv_logging:
            self.get_logger().info(f"CSV event log: {self.log_path}")
            self.get_logger().info(f"CSV summary log: {self.summary_path}")



    def _init_csv_logs(self) -> None:
        if not self.enable_csv_logging:
            return
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sim_time", "method", "event", "task_id", "poi_id", "owner",
                "release_time", "assigned_time", "done_time", "messages",
                "allocation_delay", "bid_margin", "distance", "completed",
            ])
            writer.writeheader()
        with open(self.summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sim_time", "method", "completed_tasks", "total_tasks", "messages",
                "avg_allocation_delay", "total_distance",
            ])
            writer.writeheader()

    def _log_event(self, world: MethodWorld, event: str, task: Optional[Task] = None) -> None:
        if not self.enable_csv_logging:
            return
        total_dist = sum(r.distance for r in world.robots.values())
        row = {
            "sim_time": round(self.sim_time, 3),
            "method": world.name,
            "event": event,
            "task_id": task.task_id if task else "",
            "poi_id": task.poi_id if task else "",
            "owner": task.owner or task.planned_winner or "" if task else "",
            "release_time": task.release_time if task else "",
            "assigned_time": task.assigned_time if task else "",
            "done_time": task.done_time if task else "",
            "messages": world.total_messages,
            "allocation_delay": (task.allocation_due_time - task.release_time) if task and task.allocation_due_time else "",
            "bid_margin": task.bid_margin if task else "",
            "distance": round(total_dist, 3),
            "completed": world.completed_count,
        }
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _log_summary(self) -> None:
        if not self.enable_csv_logging:
            return
        with open(self.summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "sim_time", "method", "completed_tasks", "total_tasks", "messages",
                "avg_allocation_delay", "total_distance",
            ])
            for world in self.worlds:
                avg_delay = world.total_allocation_delay / max(1, world.allocation_count)
                total_dist = sum(r.distance for r in world.robots.values())
                writer.writerow({
                    "sim_time": round(self.sim_time, 3),
                    "method": world.name,
                    "completed_tasks": world.completed_count,
                    "total_tasks": self.num_tasks,
                    "messages": world.total_messages,
                    "avg_allocation_delay": round(avg_delay, 3),
                    "total_distance": round(total_dist, 3),
                })

    def _publish_static_map_frame(self) -> None:
        """Publish an identity static transform from world to map for RViz."""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = "map"
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(transform)

    def _generate_pois(self, rng: random.Random) -> Dict[int, Vec2]:
        pois: Dict[int, Vec2] = {}
        grid_n = int(math.ceil(math.sqrt(self.num_pois)))
        spacing = self.world_size / max(1, grid_n - 1)
        idx = 1
        for i in range(grid_n):
            for j in range(grid_n):
                if idx > self.num_pois:
                    break
                x = i * spacing + rng.uniform(-1.0, 1.0)
                y = j * spacing + rng.uniform(-1.0, 1.0)
                x = max(0.0, min(self.world_size, x))
                y = max(0.0, min(self.world_size, y))
                pois[idx] = (x, y)
                idx += 1
        return pois

    def _generate_robots(self, rng: random.Random) -> Dict[str, Robot]:
        robots: Dict[str, Robot] = {}
        # Fixed-ish starts near corners/sides for repeatable visual clarity.
        starts = [
            (3.0, 3.0),
            (self.world_size - 3.0, 3.0),
            (3.0, self.world_size - 3.0),
            (self.world_size - 3.0, self.world_size - 3.0),
            (self.world_size / 2.0, 3.0),
            (self.world_size / 2.0, self.world_size - 3.0),
        ]
        for i in range(self.num_robots):
            rid = f"R{i+1}"
            pos = starts[i % len(starts)]
            robots[rid] = Robot(
                robot_id=rid,
                start_pos=pos,
                current_pos=pos,
                battery=rng.uniform(0.75, 1.0),
                health=rng.uniform(0.80, 1.0),
            )
        return robots

    def _generate_tasks(self, rng: random.Random, pois: Dict[int, Vec2]) -> List[Task]:
        # Mixed task releases: mostly random, but biased to create route dependencies.
        poi_ids = list(pois.keys())
        rng.shuffle(poi_ids)
        tasks: List[Task] = []
        for k in range(self.num_tasks):
            tasks.append(Task(task_id=f"T{k+1}", poi_id=poi_ids[k], release_time=(k + 1) * self.release_gap))
        return tasks

    def _clone_robots(self, robots: Dict[str, Robot]) -> Dict[str, Robot]:
        return {
            rid: Robot(
                robot_id=r.robot_id,
                start_pos=r.start_pos,
                current_pos=r.current_pos,
                battery=r.battery,
                health=r.health,
            )
            for rid, r in robots.items()
        }

    def _clone_tasks(self, tasks: List[Task]) -> List[Task]:
        return [Task(task_id=t.task_id, poi_id=t.poi_id, release_time=t.release_time) for t in tasks]

    def _create_worlds(self) -> List[MethodWorld]:
        rng = random.Random(self.seed)
        pois = self._generate_pois(rng)
        base_robots = self._generate_robots(rng)
        base_tasks = self._generate_tasks(rng, pois)
        return [
            MethodWorld(
                name="cbba_like_consensus",
                label="CBBA-like consensus",
                x_offset=-(self.world_size + 20.0),
                robots=self._clone_robots(base_robots),
                tasks=self._clone_tasks(base_tasks),
                pois=pois,
            ),
            MethodWorld(
                name="ours_lease_certified_reauction",
                label="Our lease-certified BT-SR",
                x_offset=20.0,
                robots=self._clone_robots(base_robots),
                tasks=self._clone_tasks(base_tasks),
                pois=pois,
            ),
        ]

    def _reset(self) -> None:
        self.sim_time = 0.0
        self.done_since = None
        self.worlds = self._create_worlds()
        self._logged_done_tasks = set()

    def _on_timer(self) -> None:
        self.sim_time += self.dt_wall * self.time_scale
        for world in self.worlds:
            self._update_world(world, self.sim_time)
        self._publish_markers()
        if int(self.sim_time * 10) % 10 == 0:
            self._log_summary()

        all_done = all(all(t.status == "done" for t in w.tasks) for w in self.worlds)
        if all_done:
            if self.done_since is None:
                self.done_since = self.sim_time
            elif self.auto_reset and self.sim_time - self.done_since > self.reset_after_done_sec:
                self._reset()

    def _update_world(self, world: MethodWorld, t: float) -> None:
        # Release tasks and plan assignment events.
        for task in world.tasks:
            if task.status == "unreleased" and t >= task.release_time:
                task.status = "pending"
                task.allocation_started_time = task.release_time
                self._start_allocation(world, task)

            if task.status == "pending" and task.allocation_due_time is not None and t >= task.allocation_due_time:
                self._complete_allocation(world, task)

        # Move robots along routes.
        for robot in world.robots.values():
            self._move_robot_one_step(world, robot, self.dt_wall * self.time_scale)

        world.completed_count = sum(1 for task in world.tasks if task.status == "done")

    def _start_allocation(self, world: MethodWorld, task: Task) -> None:
        bids: Dict[str, float] = {}
        best_routes: Dict[str, List[int]] = {}
        for rid, robot in world.robots.items():
            extra, route = best_insert_route(robot.current_pos, robot.remaining_route, task.poi_id, world.pois)
            bids[rid] = extra + robot_state_penalty(robot)
            best_routes[rid] = route

        sorted_pairs = sorted(bids.items(), key=lambda kv: kv[1])
        winner = sorted_pairs[0][0]
        margin = sorted_pairs[1][1] - sorted_pairs[0][1] if len(sorted_pairs) >= 2 else 0.0

        if world.name == "cbba_like_consensus":
            delay = self.cbba_rounds * self.cbba_round_delay
            msgs = self.num_robots * (self.num_robots - 1) * self.cbba_rounds
        else:
            risk_wait = self.ours_min_wait + self.ours_margin_scale / (1.0 + margin)
            risk_wait = min(self.ours_max_wait, risk_wait)
            delay = self.ours_base_alloc_delay + risk_wait
            msgs = self.num_robots + 1 + self.ours_heartbeat_msgs

        task.planned_winner = winner
        task.planned_route = best_routes[winner]
        task.bid_margin = margin
        task.allocation_due_time = task.release_time + delay
        world.total_messages += msgs
        world.total_allocation_delay += delay
        world.allocation_count += 1
        self._log_event(world, "TASK_RELEASE_PENDING", task)

    def _complete_allocation(self, world: MethodWorld, task: Task) -> None:
        if task.planned_winner is None or task.planned_route is None:
            return
        robot = world.robots[task.planned_winner]
        robot.remaining_route = task.planned_route
        task.status = "assigned"
        task.owner = task.planned_winner
        task.assigned_time = task.allocation_due_time
        self._log_event(world, "TASK_ASSIGNED", task)

    def _move_robot_one_step(self, world: MethodWorld, robot: Robot, dt: float) -> None:
        remaining_time = dt
        robot.phase = "IDLE" if not robot.remaining_route else "PRE"
        robot.active_task = None

        while remaining_time > 1e-9 and robot.remaining_route:
            next_poi = robot.remaining_route[0]
            next_pos = world.pois[next_poi]
            dist_to_next = euclidean(robot.current_pos, next_pos)

            if dist_to_next < 1e-9:
                self._arrive_at_poi(world, robot, next_poi)
                continue

            time_to_next = dist_to_next / self.speed
            if time_to_next <= remaining_time:
                robot.distance += dist_to_next
                robot.current_pos = next_pos
                remaining_time -= time_to_next
                self._arrive_at_poi(world, robot, next_poi)
            else:
                travel_dist = self.speed * remaining_time
                alpha = travel_dist / dist_to_next
                robot.current_pos = interpolate(robot.current_pos, next_pos, alpha)
                robot.distance += travel_dist
                remaining_time = 0.0

        if robot.remaining_route:
            robot.phase = "PRE"
            next_poi = robot.remaining_route[0]
            for task in world.tasks:
                if task.poi_id == next_poi and task.owner == robot.robot_id and task.status == "assigned":
                    robot.active_task = task.task_id
                    # Treat being close to target as CRIT for visualization.
                    if euclidean(robot.current_pos, world.pois[next_poi]) < 2.0:
                        robot.phase = "CRIT"
                    break
        else:
            robot.phase = "IDLE"

    def _arrive_at_poi(self, world: MethodWorld, robot: Robot, poi_id: int) -> None:
        if robot.remaining_route and robot.remaining_route[0] == poi_id:
            robot.remaining_route.pop(0)
        robot.completed_route.append(poi_id)
        for task in world.tasks:
            if task.poi_id == poi_id and task.owner == robot.robot_id and task.status == "assigned":
                task.status = "done"
                task.done_time = self.sim_time
                key = (world.name, task.task_id)
                if key not in self._logged_done_tasks:
                    self._logged_done_tasks.add(key)
                    self._log_event(world, "TASK_DONE", task)
                break

    def _map_point(self, world: MethodWorld, p: Vec2, z: float = 0.0) -> Point:
        msg = Point()
        msg.x = float(p[0] + world.x_offset)
        msg.y = float(p[1])
        msg.z = float(z)
        return msg

    def _new_marker(self, ns: str, marker_id: int, marker_type: int, world: Optional[MethodWorld] = None) -> Marker:
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.lifetime = Duration(sec=0, nanosec=0)
        return m

    def _publish_markers(self) -> None:
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        marker_id = 1
        marker_id = self._add_legend_marker(arr, marker_id)

        for world in self.worlds:
            marker_id = self._add_world_markers(arr, marker_id, world)

        self.pub.publish(arr)


    def _add_legend_marker(self, arr: MarkerArray, marker_id: int) -> int:
        legend = self._new_marker("legend", marker_id, Marker.TEXT_VIEW_FACING)
        marker_id += 1
        legend.pose.position.x = -95.0
        legend.pose.position.y = -9.0
        legend.pose.position.z = 3.0
        legend.scale.z = 2.6
        legend.color = make_color(1.0, 1.0, 1.0, 1.0)
        legend.text = (
            "Legend\n"
            "Gray dots: POIs | Yellow: assigned task | Green: done task\n"
            "Blue robot: PRE / moving | Red robot: CRIT | Gray robot: idle\n"
            "Blue ring: certified lease owner | Line: planned route\n"
            "CBBA: multi-round consensus delay | Ours: certified lease + scoped re-auction signals"
        )
        arr.markers.append(legend)
        return marker_id

    def _add_world_markers(self, arr: MarkerArray, marker_id: int, world: MethodWorld) -> int:
        # World title and metrics.
        title = self._new_marker("titles", marker_id, Marker.TEXT_VIEW_FACING)
        marker_id += 1
        title.pose.position = self._map_point(world, (self.world_size / 2.0, self.world_size + 13.0), 3.0)
        title.scale.z = 5.0
        title.color = make_color(1.0, 1.0, 1.0, 1.0)
        avg_delay = world.total_allocation_delay / max(1, world.allocation_count)
        total_dist = sum(r.distance for r in world.robots.values())
        title.text = (
            f"{world.label}\n"
            f"t={self.sim_time:.1f}s | done={world.completed_count}/{self.num_tasks} | "
            f"msgs={world.total_messages} | avg alloc={avg_delay:.2f}s | dist={total_dist:.1f}"
        )
        arr.markers.append(title)

        # Boundary rectangle.
        rect = self._new_marker("world_bounds", marker_id, Marker.LINE_STRIP)
        marker_id += 1
        rect.scale.x = 0.12
        rect.color = make_color(0.6, 0.6, 0.6, 0.8)
        corners = [(0, 0), (self.world_size, 0), (self.world_size, self.world_size), (0, self.world_size), (0, 0)]
        rect.points = [self._map_point(world, p, 0.02) for p in corners]
        arr.markers.append(rect)

        # POIs.
        for poi_id, pos in world.pois.items():
            poi = self._new_marker("pois", marker_id, Marker.SPHERE)
            marker_id += 1
            poi.pose.position = self._map_point(world, pos, 0.05)
            poi.scale.x = 0.8
            poi.scale.y = 0.8
            poi.scale.z = 0.12
            poi.color = make_color(0.45, 0.45, 0.45, 0.45)
            arr.markers.append(poi)

        # Tasks.
        for task in world.tasks:
            if task.status == "unreleased":
                continue
            pos = world.pois[task.poi_id]
            task_m = self._new_marker("tasks", marker_id, Marker.CUBE)
            marker_id += 1
            task_m.pose.position = self._map_point(world, pos, 0.35)
            task_m.scale.x = 1.8
            task_m.scale.y = 1.8
            task_m.scale.z = 0.35
            if task.status == "pending":
                task_m.color = make_color(1.0, 0.45, 0.0, 0.9)  # orange
            elif task.status == "assigned":
                task_m.color = make_color(1.0, 0.9, 0.0, 0.9)  # yellow
            elif task.status == "done":
                task_m.color = make_color(0.0, 0.9, 0.2, 0.9)  # green
            else:
                task_m.color = make_color(0.8, 0.8, 0.8, 0.6)
            arr.markers.append(task_m)

            label = self._new_marker("task_labels", marker_id, Marker.TEXT_VIEW_FACING)
            marker_id += 1
            label.pose.position = self._map_point(world, (pos[0], pos[1] + 2.0), 2.5)
            label.scale.z = 2.2
            label.color = make_color(1.0, 1.0, 1.0, 1.0)
            owner = task.owner if task.owner else (task.planned_winner if task.planned_winner else "-")
            if task.status == "pending":
                wait_left = max(0.0, (task.allocation_due_time or self.sim_time) - self.sim_time)
                label.text = f"{task.task_id}\n{task.status}\n{owner}\nwait {wait_left:.1f}s"
            else:
                label.text = f"{task.task_id}\n{task.status}\n{owner}"
            arr.markers.append(label)

            # Lease/ownership ring for assigned tasks in our method.
            if world.name == "ours_lease_certified_reauction" and task.status in ["assigned", "pending"]:
                ring = self._circle_marker("lease_rings", marker_id, world, pos, radius=2.0, z=0.1)
                marker_id += 1
                ring.color = make_color(0.2, 0.8, 1.0, 0.8)
                arr.markers.append(ring)

        # Robot routes.
        for robot in world.robots.values():
            if robot.remaining_route:
                route = self._new_marker("routes", marker_id, Marker.LINE_STRIP)
                marker_id += 1
                route.scale.x = 0.18
                route.color = make_color(0.2, 0.7, 1.0, 0.8) if world.name == "ours_lease_certified_reauction" else make_color(1.0, 0.7, 0.2, 0.8)
                pts = [self._map_point(world, robot.current_pos, 0.15)]
                pts += [self._map_point(world, world.pois[poi_id], 0.15) for poi_id in robot.remaining_route]
                route.points = pts
                arr.markers.append(route)

        # Robots.
        for robot in world.robots.values():
            rb = self._new_marker("robots", marker_id, Marker.CYLINDER)
            marker_id += 1
            rb.pose.position = self._map_point(world, robot.current_pos, 0.5)
            rb.scale.x = 2.0
            rb.scale.y = 2.0
            rb.scale.z = 1.0
            if robot.phase == "IDLE":
                rb.color = make_color(0.7, 0.7, 0.7, 0.95)
            elif robot.phase == "PRE":
                rb.color = make_color(0.1, 0.35, 1.0, 0.95)
            else:
                rb.color = make_color(1.0, 0.0, 0.0, 0.95)
            arr.markers.append(rb)

            rlabel = self._new_marker("robot_labels", marker_id, Marker.TEXT_VIEW_FACING)
            marker_id += 1
            rlabel.pose.position = self._map_point(world, (robot.current_pos[0] + 2.0, robot.current_pos[1] + 2.0), 3.0)
            rlabel.scale.z = 2.4
            rlabel.color = make_color(1.0, 1.0, 1.0, 1.0)
            extra = ""
            if world.name == "ours_lease_certified_reauction" and robot.phase in ["PRE", "CRIT"]:
                extra = "\nHB/cert"
            rlabel.text = f"{robot.robot_id}\n{robot.phase}{extra}"
            arr.markers.append(rlabel)

        return marker_id

    def _circle_marker(self, ns: str, marker_id: int, world: MethodWorld, center: Vec2, radius: float, z: float) -> Marker:
        m = self._new_marker(ns, marker_id, Marker.LINE_STRIP)
        m.scale.x = 0.12
        pts: List[Point] = []
        for k in range(41):
            theta = 2.0 * math.pi * k / 40.0
            p = (center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta))
            pts.append(self._map_point(world, p, z))
        m.points = pts
        return m


def main() -> None:
    rclpy.init()
    node = DynamicBtRvizSim()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
