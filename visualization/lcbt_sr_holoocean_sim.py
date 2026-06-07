#!/usr/bin/env python3
"""
LCBT-SR HoloOcean USV surface retrieval visualization (mission-scale, multi-USV version).

Research algorithm visualized:
  Lease-Certified Behavior Tree Coordination with Scoped Re-auction (LCBT-SR)

This version uses SurfaceVessel agents instead of HoveringAUVs.
It keeps retrieval visualization simple:
  - USVs move on the water surface toward task objects.
  - Each released task is a retrieval object/POI.
  - When a USV reaches an object, it pauses briefly to represent arm pickup.
  - Algorithm events are logged to CSV.

Designed for HoloOcean 2.3.x with the Ocean package installed.
"""

import argparse
import csv
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import holoocean
    from holoocean import packagemanager
except Exception as exc:
    raise SystemExit(
        "Could not import holoocean. Activate your venv first:\n"
        "  source ~/holoocean_env/bin/activate\n"
        f"Original error: {exc}"
    )


# ---------------- Geometry helpers ----------------

def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def yaw_to(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def resolve_scenario_name(world_name: str) -> str:
    aliases = {
        "OpenWater": "OpenWater-Hovering",
        "Dam": "Dam-Hovering",
        "OpenWater-Hovering": "OpenWater-Hovering",
        "Dam-Hovering": "Dam-Hovering",
    }
    return aliases.get(world_name, world_name)


def get_template_scenario(world_name: str) -> dict:
    scenario_name = resolve_scenario_name(world_name)
    try:
        cfg = packagemanager.get_scenario(scenario_name)
    except Exception as exc:
        raise SystemExit(
            f"Could not load installed HoloOcean scenario '{scenario_name}'.\n"
            "Try --world OpenWater or --world Dam.\n"
            f"Original error: {exc}"
        )
    cfg = deepcopy(cfg)
    cfg["package_name"] = "Ocean"
    cfg["name"] = "LCBT-SR-USV-Surface"
    return cfg


def get_template_location(cfg: dict, surface_z: float):
    agents = cfg.get("agents", [])
    if not agents:
        return [0.0, 0.0, surface_z]
    loc = agents[0].get("location", [0.0, 0.0, surface_z])
    if isinstance(loc, dict):
        return [float(loc.get("x", 0.0)), float(loc.get("y", 0.0)), surface_z]
    return [float(loc[0]), float(loc[1]), surface_z]


def make_usv_agent(name, loc, yaw=0.0, main=False):
    return {
        "agent_name": name,
        "agent_type": "SurfaceVessel",
        "sensors": [],
        "location": [float(loc[0]), float(loc[1]), float(loc[2])],
        "rotation": [0.0, 0.0, float(yaw)],
        # SurfaceVessel control scheme 1 is PD target [des_x, des_y, des_yaw]
        "control_scheme": 1,
        "is_main_agent": bool(main),
    }


def make_object_agent(name, loc):
    # A simple visible marker object. If SphereAgent is unavailable in the world,
    # the script will still run if HoloOcean ignores it; otherwise user can pass --no_objects.
    return {
        "agent_name": name,
        "agent_type": "SphereAgent",
        "sensors": [],
        "location": [float(loc[0]), float(loc[1]), float(loc[2])],
        "rotation": [0.0, 0.0, 0.0],
        "control_scheme": 0,
        "is_main_agent": False,
    }


def make_poi_agent(name, loc):
    # Static marker for every POI. Task objects are spawned higher than POIs,
    # so POIs remain visible even when an object is located there.
    return make_object_agent(name, loc)


def generate_start_positions(base, num_robots, spacing=8.0):
    starts = []
    # compact row near the camera, all on surface
    for i in range(num_robots):
        starts.append([base[0], base[1] + (i - (num_robots - 1) / 2.0) * spacing, base[2]])
    return starts


def generate_pois(num_pois, rng, base, surface_z, world_extent, poi_start_offset=25.0):
    # Mission-scale POI field. For the HoloOcean demo this can be 200 m+.
    # POIs are placed in a grid forward of the USV start line.
    pois = {}
    grid_n = int(math.ceil(math.sqrt(num_pois)))
    spacing = world_extent / max(1, grid_n - 1)
    start_x = base[0] + poi_start_offset
    start_y = base[1] - world_extent / 2.0
    jitter = min(3.0, max(0.4, spacing * 0.08))
    idx = 1
    for i in range(grid_n):
        for j in range(grid_n):
            if idx > num_pois:
                break
            x = start_x + i * spacing + rng.uniform(-jitter, jitter)
            y = start_y + j * spacing + rng.uniform(-jitter, jitter)
            pois[idx] = [x, y, surface_z]
            idx += 1
    return pois


def generate_tasks(num_tasks, pois, rng, release_gap):
    ids = list(pois.keys())
    rng.shuffle(ids)
    tasks = []
    for k in range(num_tasks):
        tasks.append({
            "task_id": f"T{k+1}",
            "poi_id": ids[k],
            "object_name": f"obj{k+1}",
            "release_time": (k + 1) * release_gap,
            "owner": None,
            "status": "unreleased",
            "done_time": None,
            "pickup_start": None,
        })
    return tasks


def build_scenario(args, pois, tasks):
    cfg = get_template_scenario(args.world)
    base = get_template_location(cfg, args.surface_z)
    starts = generate_start_positions(base, args.num_robots)

    usv_names = []
    agents = []
    for i, loc in enumerate(starts):
        name = f"usv{i+1}"
        usv_names.append(name)
        agents.append(make_usv_agent(name, loc, yaw=0.0, main=(i == 0)))

    if args.show_pois:
        for poi_id, poi_loc in pois.items():
            loc = list(poi_loc)
            loc[2] = args.poi_z
            agents.append(make_poi_agent(f"poi{poi_id}", loc))

    if not args.no_objects:
        for t in tasks:
            loc = list(pois[t["poi_id"]])
            loc[2] = args.object_z
            agents.append(make_object_agent(t["object_name"], loc))

    cfg["agents"] = agents
    cfg["main_agent"] = usv_names[0]
    cfg["ticks_per_sec"] = int(args.ticks_per_sec)
    cfg["frames_per_sec"] = int(args.ticks_per_sec)
    cfg["window_width"] = 1280
    cfg["window_height"] = 720
    return cfg, usv_names, starts, base


# ---------------- LCBT-SR coordination helpers ----------------

def route_distance(start, route, pois):
    total = 0.0
    cur = start
    for poi in route:
        total += dist2(cur, pois[poi])
        cur = pois[poi]
    return total


def best_insert(current_pos, route, poi_id, pois):
    current_cost = route_distance(current_pos, route, pois)
    best_extra = None
    best_route = None
    for k in range(len(route) + 1):
        cand = route[:k] + [poi_id] + route[k:]
        cost = route_distance(current_pos, cand, pois)
        extra = cost - current_cost
        if best_extra is None or extra < best_extra:
            best_extra = extra
            best_route = cand
    return best_extra, best_route


def make_robot_state(names, start_positions):
    robots = {}
    for n, p in zip(names, start_positions):
        robots[n] = {
            "pos": [float(p[0]), float(p[1]), float(p[2])],
            "yaw": 0.0,
            "route": [],
            "distance": 0.0,
            "phase": "IDLE",
            "active_task": None,
            "pickup_until": 0.0,
        }
    return robots


def scored_bid(robot, poi_id, pois, args):
    extra, _ = best_insert(robot["pos"], robot["route"], poi_id, pois)
    load_cost = args.load_penalty * len(robot["route"])
    busy_cost = args.busy_penalty if robot.get("phase") in ["PRE", "CRIT_PICKUP"] else 0.0
    return extra + load_cost + busy_cost


def choose_winner(task, robots, pois, args):
    # For a real multi-USV retrieval mission, we want parallelism.
    # Marginal insertion remains the base bid, but a load penalty prevents one
    # route-efficient USV from hoarding all released objects.
    candidates = []
    for rid, r in robots.items():
        if len(r["route"]) < args.max_route_len:
            candidates.append(rid)
    if not candidates:
        candidates = list(robots.keys())

    bids = {}
    for rid in candidates:
        bids[rid] = scored_bid(robots[rid], task["poi_id"], pois, args)
    return min(bids, key=bids.get), bids


def move_robot_toward(robot, target, speed, dt):
    d = dist2(robot["pos"], target)
    if d < 1e-6:
        return True
    step = min(d, speed * dt)
    alpha = step / d
    old = robot["pos"][:]
    robot["pos"][0] += alpha * (target[0] - robot["pos"][0])
    robot["pos"][1] += alpha * (target[1] - robot["pos"][1])
    robot["yaw"] = yaw_to(old, robot["pos"])
    robot["distance"] += step
    return step >= d - 1e-6


def usv_action(robot):
    # SurfaceVessel PD controller command: [desired_x, desired_y, desired_yaw]
    return np.array([robot["pos"][0], robot["pos"][1], robot["yaw"]], dtype=np.float32)


def try_teleport_object(env, name, loc):
    # Best-effort object visualization update; failures are harmless.
    try:
        if hasattr(env, "agents") and name in env.agents:
            env.agents[name].teleport(location=np.array(loc, dtype=np.float32))
            return True
    except Exception:
        pass
    return False


# ---------------- Camera helpers ----------------

def _candidate_command_centers(env):
    """Return possible command centers for HoloOcean versions that expose them differently."""
    centers = []
    for attr in ["_command_center", "command_center"]:
        if hasattr(env, attr):
            centers.append(getattr(env, attr))
    client = getattr(env, "_client", None)
    if client is not None:
        for attr in ["command_center", "_command_center"]:
            if hasattr(client, attr):
                centers.append(getattr(client, attr))
    client = getattr(env, "client", None)
    if client is not None:
        for attr in ["command_center", "_command_center"]:
            if hasattr(client, attr):
                centers.append(getattr(client, attr))
    return centers


def send_camera_pose(env, location, rotation, verbose=False):
    """
    Best-effort viewport camera teleport.
    HoloOcean exposes TeleportCameraCommand in the command module, but the
    command-center location differs across versions. This helper tries the
    common paths and never stops the simulation if camera control is unavailable.
    """
    location = [float(x) for x in location]
    rotation = [float(x) for x in rotation]

    # First try the formal command class.
    try:
        from holoocean.command import TeleportCameraCommand
        cmd = TeleportCameraCommand(location, rotation)
        for center in _candidate_command_centers(env):
            if hasattr(center, "enqueue_command"):
                center.enqueue_command(cmd)
                return True
            if hasattr(center, "add_command"):
                center.add_command(cmd)
                return True
    except Exception as exc:
        if verbose:
            print("TeleportCameraCommand path failed:", exc)

    # Some versions expose command methods directly on env.
    for method_name in ["send_command", "send_client_command"]:
        try:
            if hasattr(env, method_name):
                from holoocean.command import TeleportCameraCommand
                getattr(env, method_name)(TeleportCameraCommand(location, rotation))
                return True
        except Exception as exc:
            if verbose:
                print(method_name, "camera path failed:", exc)

    # Avoid send_world_command by default because unsupported world commands
    # can terminate some HoloOcean worlds.
    return False


def compute_scene_center(pois, starts):
    xs = [p[0] for p in pois.values()] + [s[0] for s in starts]
    ys = [p[1] for p in pois.values()] + [s[1] for s in starts]
    z = 0.0
    return [sum(xs) / len(xs), sum(ys) / len(ys), z]


def camera_pose_for_mode(mode, center, robots, args):
    """
    Returns camera location and rotation for top-down/side/follow.
    Rotation ordering follows the HoloOcean/Unreal convention used by the
    viewport camera command in most versions: [pitch, yaw, roll]. If a version
    interprets it differently, topdown still usually gives the most visible view.
    """
    mode = str(mode).lower()
    if mode == "none":
        return None, None

    if mode == "follow" and robots:
        r0 = next(iter(robots.values()))
        cx, cy, cz = r0["pos"]
        loc = [cx - args.camera_distance, cy, args.camera_height * 0.6]
        rot = [-35.0, 0.0, 0.0]
        return loc, rot

    if mode == "side":
        cx, cy, cz = center
        loc = [cx - args.camera_distance, cy - args.camera_distance * 0.35, args.camera_height * 0.65]
        rot = [-35.0, 25.0, 0.0]
        return loc, rot

    # Default: top-down oblique view over the whole mission area.
    cx, cy, cz = center
    loc = [cx, cy, args.camera_height]
    rot = [-88.0, 0.0, 0.0]
    return loc, rot


def update_camera(env, mode, center, robots, args, verbose=False):
    loc, rot = camera_pose_for_mode(mode, center, robots, args)
    if loc is None:
        return False
    return send_camera_pose(env, loc, rot, verbose=verbose)


# ---------------- Main simulation ----------------

def run(args):
    rng = random.Random(args.seed)
    template_cfg = get_template_scenario(args.world)
    base = get_template_location(template_cfg, args.surface_z)
    pois = generate_pois(args.num_pois, rng, base, args.surface_z, args.world_extent, args.poi_start_offset)
    tasks = generate_tasks(args.num_tasks, pois, rng, args.release_gap)
    cfg, names, start_positions, base = build_scenario(args, pois, tasks)
    robots = make_robot_state(names, start_positions)
    scene_center = compute_scene_center(pois, start_positions)

    if args.debug_scenario:
        print("Scenario world:", cfg.get("world"), "package:", cfg.get("package_name"))
        print("Main agent:", cfg.get("main_agent"))
        for a in cfg["agents"][: min(12, len(cfg["agents"]))]:
            print("Agent:", a.get("agent_name"), "type=", a.get("agent_type"), "loc=", a.get("location"), "control=", a.get("control_scheme"))
        print("Base:", base, "surface_z:", args.surface_z)
        print("Camera mode:", args.camera_mode, "height:", args.camera_height, "distance:", args.camera_distance)

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    event_path = out_dir / "holoocean_lcbt_sr_usv_events.csv"
    summary_path = out_dir / "holoocean_lcbt_sr_usv_summary.csv"

    events = []
    total_messages = 0
    reauction_count = 0
    cert_blocks = 0
    pickups = 0

    env = holoocean.make(scenario_cfg=cfg, show_viewport=True, verbose=args.verbose)
    try:
        for _ in range(15):
            for rid, r in robots.items():
                try:
                    env.act(rid, usv_action(r))
                except Exception:
                    pass
            env.tick()

        if args.camera_mode != "none" and not args.no_initial_camera:
            camera_ok = update_camera(env, args.camera_mode, scene_center, robots, args, verbose=args.verbose)
            print("Initial camera command sent. Manual camera control is now free." if camera_ok else "Camera command not accepted by this HoloOcean build; using default viewport.")
            if args.lock_camera:
                print("Camera lock is ON: the script will keep resetting the camera.")
            else:
                print("Camera lock is OFF: the script will not reset the camera again.")

        print("USV surface retrieval visualization running.")
        print("Robots:", names)
        print("If the camera is awkward, use the main USV as reference; all tasks are placed forward of the spawn area.")

        dt = 1.0 / args.ticks_per_sec
        sim_time = 0.0
        next_print = 0.0

        while sim_time <= args.max_time:
            # Release tasks and allocate with marginal route bids.
            for task in tasks:
                if task["status"] == "unreleased" and sim_time >= task["release_time"]:
                    winner, bids = choose_winner(task, robots, pois, args)
                    task["owner"] = winner
                    task["status"] = "assigned"
                    robots[winner]["active_task"] = task["task_id"]
                    robots[winner]["phase"] = "PRE"
                    _, new_route = best_insert(robots[winner]["pos"], robots[winner]["route"], task["poi_id"], pois)
                    robots[winner]["route"] = new_route
                    total_messages += len(robots) + 1
                    events.append([sim_time, "release_assign", task["task_id"], task["poi_id"], winner, len(robots) + 1, 0, 0])

            for rid, robot in robots.items():
                # If currently simulating pickup arm action, hold still.
                if robot["pickup_until"] > sim_time:
                    robot["phase"] = "CRIT_PICKUP"
                elif robot["route"]:
                    target_poi = robot["route"][0]
                    target = pois[target_poi]
                    robot["phase"] = "PRE"
                    arrived = move_robot_toward(robot, target, args.robot_speed, dt)

                    if arrived:
                        # Lease certification before critical pickup. No guard.
                        stress_fail = rng.random() < max(0.0, args.drop_rate - args.certification_resilience)
                        if stress_fail:
                            cert_blocks += 1
                            reauction_count += 1
                            ranked = []
                            for other_id, other in robots.items():
                                # Exclude robots that are busy picking another object.
                                if other["phase"] == "CRIT_PICKUP" and other_id != rid:
                                    continue
                                score = scored_bid(other, target_poi, pois, args)
                                ranked.append((score, other_id))
                            ranked.sort()
                            eligible = [x[1] for x in ranked[: max(1, min(args.reauction_top_k, len(ranked)))]]
                            total_messages += len(eligible) + 1
                            winner = eligible[0]
                            if winner != rid:
                                if target_poi in robot["route"]:
                                    robot["route"].remove(target_poi)
                                _, nr = best_insert(robots[winner]["pos"], robots[winner]["route"], target_poi, pois)
                                robots[winner]["route"] = nr
                                robots[winner]["phase"] = "PRE"
                                for task_obj in tasks:
                                    if task_obj["poi_id"] == target_poi and task_obj["status"] != "done":
                                        task_obj["owner"] = winner
                                        break
                            events.append([sim_time, "scoped_reauction", "", target_poi, winner, len(eligible) + 1, len(eligible), 1])
                        else:
                            robot["phase"] = "CRIT_PICKUP"
                            robot["pickup_until"] = sim_time + args.pickup_time
                            pickups += 1
                            # Mark task done at pickup start for coordination/logging.
                            for task in tasks:
                                if task["poi_id"] == target_poi and task["owner"] == rid and task["done_time"] is None:
                                    task["done_time"] = sim_time
                                    task["status"] = "done"
                                    # Move visual object below surface / away if possible.
                                    try_teleport_object(env, task["object_name"], [target[0], target[1], args.object_hide_z])
                                    break
                            robot["route"].pop(0)
                            events.append([sim_time, "pickup_done", "", target_poi, rid, 0, 0, 0])
                else:
                    robot["phase"] = "IDLE"
                    robot["active_task"] = None

                try:
                    env.act(rid, usv_action(robot))
                except Exception as exc:
                    if args.verbose:
                        print("act failed", rid, exc)

            if args.lock_camera and args.camera_mode != "none" and (sim_time - getattr(args, "_last_camera_update", -999.0)) >= args.camera_update_period:
                update_camera(env, args.camera_mode, scene_center, robots, args, verbose=False)
                args._last_camera_update = sim_time

            env.tick()
            if args.realtime:
                time.sleep(dt)

            if sim_time >= next_print:
                done = sum(1 for t in tasks if t["status"] == "done")
                print(
                    f"t={sim_time:5.1f}s done={done}/{len(tasks)} "
                    f"msgs={total_messages} reauc={reauction_count} cert_blocks={cert_blocks} pickups={pickups}"
                )
                next_print += 2.0

            if all(t["status"] == "done" for t in tasks):
                break
            sim_time += dt

    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    with event_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "event", "task_id", "poi_id", "robot", "messages", "eligible", "cert_block"])
        writer.writerows(events)

    done = sum(1 for t in tasks if t["status"] == "done")
    total_distance = sum(r["distance"] for r in robots.values())
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["done_tasks", "total_tasks", "mission_time", "messages", "reauction_count", "certification_blocks", "pickups", "total_distance"])
        writer.writerow([done, len(tasks), sim_time, total_messages, reauction_count, cert_blocks, pickups, total_distance])

    print("\nSimulation complete.")
    print("Events:", event_path)
    print("Summary:", summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="OpenWater", help="OpenWater, Dam, OpenWater-Hovering, or Dam-Hovering")
    parser.add_argument("--num_pois", type=int, default=36)
    parser.add_argument("--num_robots", type=int, default=4)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--release_gap", type=float, default=1.5)
    parser.add_argument("--delay_mean", type=float, default=4.0)
    parser.add_argument("--drop_rate", type=float, default=0.30)
    parser.add_argument("--certification_resilience", type=float, default=0.22,
                        help="Higher means LCBT-SR tolerates more drop stress before re-auction.")
    parser.add_argument("--robot_speed", type=float, default=2.2)
    parser.add_argument("--ticks_per_sec", type=float, default=30.0)
    parser.add_argument("--max_time", type=float, default=160.0)
    parser.add_argument("--world_extent", type=float, default=200.0)
    parser.add_argument("--poi_start_offset", type=float, default=25.0)
    parser.add_argument("--surface_z", type=float, default=0.0)
    parser.add_argument("--poi_z", type=float, default=0.25)
    parser.add_argument("--object_z", type=float, default=1.2)
    parser.add_argument("--object_hide_z", type=float, default=-30.0)
    parser.add_argument("--pickup_time", type=float, default=1.2)
    parser.add_argument("--camera_mode", default="topdown", choices=["topdown", "side", "follow", "none"],
                        help="Best-effort HoloOcean viewport camera mode.")
    parser.add_argument("--camera_height", type=float, default=65.0)
    parser.add_argument("--camera_distance", type=float, default=45.0)
    parser.add_argument("--camera_update_period", type=float, default=2.0,
                        help="How often to resend the camera pose command when --lock_camera is enabled.")
    parser.add_argument("--lock_camera", action="store_true",
                        help="Continuously resend the camera pose. Leave off for manual camera control after the first setup command.")
    parser.add_argument("--no_initial_camera", action="store_true",
                        help="Do not send even the initial camera pose; use HoloOcean's default viewport camera.")
    parser.add_argument("--reauction_top_k", type=int, default=2)
    parser.add_argument("--max_route_len", type=int, default=2,
                        help="Soft cap on queued tasks per USV before assigning to another USV for parallel retrieval visualization.")
    parser.add_argument("--load_penalty", type=float, default=70.0,
                        help="Penalty per queued task, encourages multiple USVs to move instead of one route-efficient USV taking everything.")
    parser.add_argument("--busy_penalty", type=float, default=20.0,
                        help="Penalty for assigning new tasks to a USV that is already moving/picking up.")
    parser.add_argument("--show_pois", action=argparse.BooleanOptionalAction, default=True,
                        help="Show all POI markers in addition to active retrieval objects.")
    parser.add_argument("--seed", type=int, default=202606)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no_objects", action="store_true")
    parser.add_argument("--debug_scenario", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_dir", default="~/phase1a_ws/results/holoocean_lcbt_sr")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
