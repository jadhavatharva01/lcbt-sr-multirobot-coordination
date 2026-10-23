#!/usr/bin/env python3
"""
LCBT-SR HoloOcean SurfaceVessel simulation of the CURRENT paper contribution:
CHANNEL-AWARE FAILURE RECOVERY for decentralized MRTA under bursty communication.

This is NOT the old lease-certification / scoped-reauction demo. It embodies, on
real SurfaceVessels in the Ocean world, the mechanism in paper/main.tex:

  * Heartbeats over a Gilbert-Elliott BURST channel  -> correlated silence.
  * A monitor USV (usv1) watches all teammates and must decide "failed vs
    partitioned-but-alive" from silence alone.
  * Two detectors run every tick:
        - adaptive_pernode  : phi-accrual baseline (per-node inter-arrival timeout).
                              Runs as a SHADOW counterfactual (logged, not acted on).
        - ours_correlated   : failure belief  beta = rho * eta  (channel-health
                              scaled), fired at the cost-optimal threshold tau.
                              This is the LIVE controller.
  * Locality-targeted relay (Alg 3): when OURS fires on a silent robot, query the
    k USVs predicted nearest the silent robot. If any relay hears it -> CANCEL the
    takeover (it was a partition, not a failure). Else -> re-allocate its task.

Scripted ground-truth events make the gap visible and auditable:
   PHASE A  normal retrieval, everyone hears everyone.
   PHASE B  BURST OUTAGE on the monitor's channel (correlated silence). phi-accrual
            false-takes-over alive robots; OURS holds because eta -> 0.
   PHASE C  GENUINE FAILURE: one USV stops + goes silent while the channel is good.
            Both detectors fire; OURS re-allocates correctly.
   PHASE D  PARTITION (Gap-2): one alive USV's link to the monitor goes bad while
            the team channel is good. OURS' belief fires (false alarm) -> the relay
            asks the nearest USVs, they hear it, and the takeover is CANCELLED.

Formulas (rho, eta, beta, tau, phi-accrual timeout) are copied from
failure_recovery_sim.py so this is numerically consistent with the paper.

Designed for HoloOcean 2.3.x with the Ocean package installed.
Run:
  source ~/holoocean_env/bin/activate
  python lcbt_sr_channel_aware_holoocean.py            # with viewport
  python lcbt_sr_channel_aware_holoocean.py --headless # no window (CI/over-ssh)
"""

import argparse
import csv
import math
import random
import select
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import holoocean
    from holoocean import packagemanager
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import holoocean. Activate your venv first:\n"
        "  source ~/holoocean_env/bin/activate\n"
        f"Original error: {exc}"
    )


# ============================================================ geometry helpers

def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def yaw_to(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


# ============================================================ scenario build

def resolve_scenario_name(world_name):
    aliases = {
        "OpenWater": "OpenWater-Hovering",
        "Dam": "Dam-Hovering",
        "OpenWater-Hovering": "OpenWater-Hovering",
        "Dam-Hovering": "Dam-Hovering",
    }
    return aliases.get(world_name, world_name)


def get_template_scenario(world_name):
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
    cfg["name"] = "LCBT-SR-ChannelAware"
    return cfg


def get_template_location(cfg, surface_z):
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
        "control_scheme": 1,  # PD target [des_x, des_y, des_yaw]
        "is_main_agent": bool(main),
    }


def make_object_agent(name, loc):
    return {
        "agent_name": name,
        "agent_type": "SphereAgent",
        "sensors": [],
        "location": [float(loc[0]), float(loc[1]), float(loc[2])],
        "rotation": [0.0, 0.0, 0.0],
        "control_scheme": 0,
        "is_main_agent": False,
    }


def generate_start_positions(base, num_robots, spacing=8.0):
    return [[base[0], base[1] + (i - (num_robots - 1) / 2.0) * spacing, base[2]]
            for i in range(num_robots)]


def generate_pois(num_pois, rng, base, surface_z, world_extent, poi_start_offset=25.0):
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


def generate_tasks(pois, rng, release_gap, num_items):
    # Every POI defined at the start is a SEARCH target. A random subset actually
    # contain an item to retrieve; the rest are searched-and-found-empty.
    ids = list(pois.keys())
    rng.shuffle(ids)
    item_ids = set(ids[:num_items])
    tasks = []
    for k, pid in enumerate(ids):
        tasks.append({
            "task_id": f"S{k+1}", "poi_id": pid, "object_name": f"obj{pid}",
            "release_time": k * release_gap, "owner": None,
            "status": "unreleased", "done_time": None,
            "has_item": pid in item_ids,
        })
    return tasks


def build_scenario(args, pois, tasks):
    cfg = get_template_scenario(args.world)
    base = get_template_location(cfg, args.surface_z)
    starts = generate_start_positions(base, args.num_robots)
    names, agents = [], []
    for i, loc in enumerate(starts):
        name = f"usv{i+1}"
        names.append(name)
        agents.append(make_usv_agent(name, loc, yaw=0.0, main=(i == 0)))
    if not args.no_objects:
        for t in tasks:
            if not t["has_item"]:
                continue
            loc = list(pois[t["poi_id"]])
            loc[2] = args.object_z
            agents.append(make_object_agent(t["object_name"], loc))
    cfg["agents"] = agents
    cfg["main_agent"] = names[0]
    cfg["ticks_per_sec"] = int(args.ticks_per_sec)
    cfg["frames_per_sec"] = int(args.ticks_per_sec)
    cfg["window_width"] = 1280
    cfg["window_height"] = 720
    return cfg, names, starts, base


# ============================================================ allocation core

def route_distance(start, route, pois):
    total, cur = 0.0, start
    for poi in route:
        total += dist2(cur, pois[poi])
        cur = pois[poi]
    return total


def best_insert(pos, route, poi_id, pois):
    base = route_distance(pos, route, pois)
    best_extra, best_route = None, None
    for k in range(len(route) + 1):
        cand = route[:k] + [poi_id] + route[k:]
        extra = route_distance(pos, cand, pois) - base
        if best_extra is None or extra < best_extra:
            best_extra, best_route = extra, cand
    return best_extra, best_route


def scored_bid(robot, poi_id, pois, args):
    extra, _ = best_insert(robot["pos"], robot["route"], poi_id, pois)
    load = args.load_penalty * len(robot["route"])
    busy = args.busy_penalty if robot["phase"] in ("PICKUP",) else 0.0
    dead = 1e9 if not robot["alive"] else 0.0
    return extra + load + busy + dead


def choose_winner(poi_id, robots, pois, args, exclude=None):
    cand = [r for r in robots if r != exclude and robots[r]["alive"]
            and len(robots[r]["route"]) < args.max_route_len]
    if not cand:
        cand = [r for r in robots if r != exclude and robots[r]["alive"]]
    bids = {r: scored_bid(robots[r], poi_id, pois, args) for r in cand}
    return min(bids, key=bids.get), bids


def move_robot_toward(robot, target, speed, dt):
    d = dist2(robot["pos"], target)
    if d < 1e-6:
        return True
    step = min(d, speed * dt)
    a = step / d
    old = robot["pos"][:]
    robot["pos"][0] += a * (target[0] - robot["pos"][0])
    robot["pos"][1] += a * (target[1] - robot["pos"][1])
    robot["yaw"] = yaw_to(old, robot["pos"])
    robot["distance"] += step
    return step >= d - 1e-6


def usv_action(robot):
    return np.array([robot["pos"][0], robot["pos"][1], robot["yaw"]], dtype=np.float32)


# ============================================================ comms + detection
# Faithful to failure_recovery_sim.py:
#   rho  = clip((silence - 2hb)/(4hb), 0, 1)       silence severity
#   eta  = fraction of OTHER teammates heard within window W   channel health
#   beta = rho * eta                                failure belief
#   tau  = c_conflict / (c_miss + c_conflict)       cost-optimal fire threshold
#   phi-accrual timeout = mean + 3*std of received inter-arrival gaps + hb

class Channel:
    """Gilbert-Elliott burst channel from the MONITOR's perspective.

    A single team-wide good/bad state drives CORRELATED silence: in the bad
    state nearly every incoming heartbeat is dropped, so the monitor hears
    almost no one (eta -> 0). Per-link overrides model an isolated partition
    (one sender silent while the team channel stays good) -> the Gap-2 case.
    """

    def __init__(self, rng, p_g2b=0.15, p_b2g=0.30, good_drop=0.2, burst_drop=0.9):
        self.rng = rng
        self.p_g2b, self.p_b2g = p_g2b, p_b2g
        self.good_drop, self.burst_drop = good_drop, burst_drop
        self.bad = False
        self.forced = None          # None | "good" | "bad"  (scripted outage)
        self.dead_links = set()     # senders whose link to monitor is forced down

    def step(self):
        if self.forced == "bad":
            self.bad = True
        elif self.forced == "good":
            self.bad = False
        else:
            if self.bad:
                self.bad = self.rng.random() > self.p_b2g
            else:
                self.bad = self.rng.random() < self.p_g2b

    def delivers(self, sender):
        if sender in self.dead_links:
            return False
        drop = self.burst_drop if self.bad else self.good_drop
        return self.rng.random() > drop


def channel_health(delivered, ids, monitor, exclude, t, window):
    others = [j for j in ids if j != monitor and j != exclude]
    if not others:
        return 1.0
    heard = 0
    for j in others:
        lst = delivered[j]
        if lst and lst[-1] >= t - window:
            heard += 1
    return heard / len(others)


def last_hb(delivered, j, t):
    lst = delivered[j]
    return lst[-1] if lst else -1e9


def adaptive_timeout(delivered, j, hb, kstd=3.0):
    times = delivered[j]
    if len(times) < 2:
        return kstd * hb
    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    mean = sum(gaps) / len(gaps)
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return mean + kstd * math.sqrt(var) + hb


# ============================================================ viewport camera
# HoloOcean's viewport camera is an "agent follower" moved only via
# env.move_viewport(location, rotation=[roll,pitch,yaw]) -- the mouse/scroll do
# NOTHING. So we drive it ourselves: an orbit camera around a target point that
# the user can pan/zoom/rotate live from the terminal (no extra deps).

class OrbitCamera:
    def __init__(self, target, back=55.0, height=70.0, yaw=0.0):
        self.tx, self.ty = float(target[0]), float(target[1])
        self.back = back        # horizontal distance behind target (zoom)
        self.height = height    # camera altitude (zoom/tilt)
        self.yaw = yaw          # orbit angle around target, degrees
        self.follow = True      # auto-recenter target on the scene/action

    def set_target(self, x, y, alpha=0.1):
        # smooth follow so the view doesn't jitter
        self.tx += alpha * (x - self.tx)
        self.ty += alpha * (y - self.ty)

    def pose(self):
        rad = math.radians(self.yaw)
        cx = self.tx - self.back * math.cos(rad)
        cy = self.ty - self.back * math.sin(rad)
        cz = self.height
        view_yaw = math.degrees(math.atan2(self.ty - cy, self.tx - cx))
        view_pitch = -math.degrees(math.atan2(self.height, max(self.back, 1e-3)))
        return [cx, cy, cz], [0.0, view_pitch, view_yaw]

    def handle_key(self, k):
        # zoom
        if k in ("+", "="):
            self.back = max(8.0, self.back * 0.88); self.height = max(8.0, self.height * 0.88)
        elif k in ("-", "_"):
            self.back *= 1.14; self.height *= 1.14
        # altitude / tilt
        elif k == "r":
            self.height += 6.0
        elif k == "f":
            self.height = max(6.0, self.height - 6.0)
        # orbit
        elif k in ("a",):
            self.yaw -= 8.0
        elif k in ("d",):
            self.yaw += 8.0
        # pan target
        elif k in ("i",):
            self.ty += 6.0; self.follow = False
        elif k in ("k",):
            self.ty -= 6.0; self.follow = False
        elif k in ("j",):
            self.tx -= 6.0; self.follow = False
        elif k in ("l",):
            self.tx += 6.0; self.follow = False
        # presets
        elif k == "t":           # top-down
            self.yaw = 0.0; self.back = 1.0; self.height = 130.0
        elif k == "o":           # oblique default
            self.back = 55.0; self.height = 70.0; self.yaw = 0.0
        elif k == "c":           # re-enable auto follow
            self.follow = True
        else:
            return False
        return True


class KeyPoller:
    """Non-blocking single-key reader for the launching terminal (POSIX)."""
    def __init__(self, enabled):
        self.enabled = enabled and sys.stdin.isatty()
        self._fd = None

    def __enter__(self):
        if self.enabled:
            import termios, tty
            self._termios = termios
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def poll(self):
        if not self.enabled:
            return None
        keys = None
        while select.select([sys.stdin], [], [], 0)[0]:
            keys = sys.stdin.read(1)
        return keys

    def __exit__(self, *a):
        if self._fd is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)


# ============================================================ world overlays
# HoloOcean has no in-world TEXT, but it can draw colored boxes/points/lines.
# We repaint every tick (short lifetime) so USVs read RED and POIs read YELLOW
# even fully zoomed out, with state colors so the user SEES what is happening.

RED      = [255, 30, 30]      # USV, alive / en route
GREEN    = [40, 220, 60]      # USV picking up a target
ORANGE   = [255, 150, 0]      # USV partitioned-but-alive (Gap-2)
GREY      = [130, 130, 130]   # USV failed (dead)
YELLOW   = [255, 230, 0]      # POI: unsearched
BLUE     = [70, 140, 255]     # POI: visited / searched (no item, or already cleared)
LIME     = [0, 255, 120]      # POI: item FOUND / retrieved here
CYAN     = [0, 220, 255]      # relay query
WHITE    = [240, 240, 240]    # route line
MAGENTA  = [255, 0, 200]      # POI just EMERGED (flash)


def usv_color(robot, partitioned):
    if not robot["alive"]:
        return GREY
    if robot["phase"] == "PICKUP":
        return GREEN
    if partitioned:
        return ORANGE
    return RED


def draw_overlays(env, robots, pois, tasks, partitioned_set,
                  visited_pois, found_pois, released_pois, emerge_flash, t, life):
    # Only POIs that have EMERGED so far are known to the team and get drawn.
    # yellow = emerged+unsearched, blue = visited/searched, lime = item found.
    for pid in released_pois:
        loc = pois[pid]
        if pid in found_pois:
            col, th = LIME, 26.0
        elif pid in visited_pois:
            col, th = BLUE, 16.0
        else:
            col, th = YELLOW, 18.0
        env.draw_point([loc[0], loc[1], 0.5], color=col, thickness=th, lifetime=life)
        # bright magenta expanding marker the moment a POI emerges
        if emerge_flash.get(pid, 0.0) > t:
            env.draw_point([loc[0], loc[1], 0.5], color=MAGENTA, thickness=40.0, lifetime=life)
            env.draw_line([loc[0], loc[1], 0.5], [loc[0], loc[1], 18.0],
                          color=MAGENTA, thickness=8.0, lifetime=life)
    # item-found POIs get a bright lime box marker; still-active emerged task POIs a yellow box
    for tk in tasks:
        if tk["poi_id"] not in released_pois:
            continue
        p = pois[tk["poi_id"]]
        if tk["poi_id"] in found_pois:
            env.draw_box([p[0], p[1], 1.5], [1.8, 1.8, 1.8], color=LIME,
                         thickness=8.0, lifetime=life)
        elif tk["status"] != "done":
            env.draw_box([p[0], p[1], 1.5], [1.6, 1.6, 1.6], color=YELLOW,
                         thickness=6.0, lifetime=life)
    # USVs (red, with state tint) + a tall beacon so they read when zoomed out
    for rid, r in robots.items():
        c = usv_color(r, rid in partitioned_set)
        x, y, _ = r["pos"]
        env.draw_box([x, y, 1.5], [2.6, 2.6, 2.6], color=c, thickness=8.0, lifetime=life)
        env.draw_line([x, y, 0.5], [x, y, 12.0], color=c, thickness=10.0, lifetime=life)
        env.draw_point([x, y, 12.5], color=c, thickness=22.0, lifetime=life)
        # route line so assignments are visible
        cur = [x, y, 1.0]
        for poi in r["route"]:
            nxt = [pois[poi][0], pois[poi][1], 1.0]
            env.draw_line(cur, nxt, color=WHITE, thickness=4.0, lifetime=life)
            cur = nxt


# ============================================================ main simulation

def run(args):
    rng = random.Random(args.seed)
    tmpl = get_template_scenario(args.world)
    base = get_template_location(tmpl, args.surface_z)
    pois = generate_pois(args.num_pois, rng, base, args.surface_z,
                         args.world_extent, args.poi_start_offset)
    tasks = generate_tasks(pois, rng, args.release_gap, args.num_items)
    cfg, names, starts, base = build_scenario(args, pois, tasks)

    robots = {}
    for n, p in zip(names, starts):
        robots[n] = {"pos": [float(p[0]), float(p[1]), float(p[2])], "yaw": 0.0,
                     "route": [], "distance": 0.0, "phase": "IDLE",
                     "pickup_until": 0.0, "alive": True}
    monitor = names[0]
    teammates = [n for n in names if n != monitor]
    visited_pois = set()   # POIs a USV has physically reached (searched)
    found_pois = set()     # POIs where a task item was retrieved
    released_pois = set()  # POIs that have actually EMERGED (become known) so far
    emerge_flash = {}      # poi_id -> sim-time until which to draw the emergence flash

    # comms / detection state, from the monitor's perspective
    hb = args.hb_period
    window = 3.0 * hb
    tau = args.c_conflict / (args.c_miss + args.c_conflict)
    chan = Channel(rng, good_drop=args.good_drop, burst_drop=args.burst_drop)
    delivered = {j: [] for j in names}          # emission times the monitor received
    next_hb_emit = 0.0

    # detector book-keeping
    ours_state = {j: {"fired": False} for j in teammates}
    base_state = {j: {"fired": False} for j in teammates}
    ours_false_takeovers = 0
    base_false_takeovers = 0
    ours_missed = 0
    correct_recoveries = 0
    relay_cancels = 0
    relay_queries = 0
    messages = 0
    events = []

    # ground-truth scripted events (see module docstring)
    fail_robot = teammates[-1] if len(teammates) >= 1 else None       # PHASE C
    part_robot = teammates[0] if len(teammates) >= 2 else None        # PHASE D
    # ensure the two roles are distinct
    if part_robot == fail_robot and len(teammates) >= 2:
        part_robot = teammates[1]

    T_outage = (args.outage_start, args.outage_start + args.outage_len)   # PHASE B
    T_fail = args.fail_time                                               # PHASE C
    T_part = (args.partition_start, args.partition_start + args.partition_len)  # PHASE D

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_path = out_dir / "channel_aware_events.csv"
    sum_path = out_dir / "channel_aware_summary.csv"

    def log(t, ev, who="", note=""):
        events.append([round(t, 2), ev, who, note])
        # always surface events to the terminal so nothing is hidden
        print(f"  >> [t={t:6.2f}] {ev:24s} {who:6s} {note}")

    def predicted_pos(j, t):
        # last-known position + heading toward current route head (Alg 3 prediction)
        r = robots[j]
        if r["route"]:
            tgt = pois[r["route"][0]]
            d = dist2(r["pos"], tgt)
            if d > 1e-6:
                step = min(d, args.robot_speed * 1.5)
                a = step / d
                return [r["pos"][0] + a * (tgt[0] - r["pos"][0]),
                        r["pos"][1] + a * (tgt[1] - r["pos"][1])]
        return r["pos"][:2]

    def locality_relay(silent, t):
        # Alg 3: ask the k USVs predicted NEAREST the silent robot whether they
        # can hear it. We resolve "can hear" by ground truth (the relay is close
        # and on a healthy link to the silent robot).
        nonlocal relay_queries, messages
        pp = predicted_pos(silent, t)
        cand = sorted((dist2(robots[r]["pos"], pp), r) for r in teammates
                      if r != silent and robots[r]["alive"])
        relays = [r for _, r in cand[: args.relay_k]]
        relay_queries += 1
        messages += len(relays) + 1
        # a relay hears the silent robot iff the silent robot is actually alive
        # (a partition is a monitor-local link problem, not a global one)
        heard = robots[silent]["alive"] and len(relays) > 0
        # visualize the relay query (cyan arrows from each relay to the silent robot)
        if (not args.headless) and (not args.no_overlays):
            sp = robots[silent]["pos"]
            for r in relays:
                rp = robots[r]["pos"]
                try:
                    env.draw_arrow([rp[0], rp[1], 4.0], [sp[0], sp[1], 4.0],
                                   color=CYAN, thickness=12.0, lifetime=1.5)
                except Exception:
                    pass
        return heard, relays

    show_viewport = not args.headless
    env = holoocean.make(scenario_cfg=cfg, show_viewport=show_viewport,
                         verbose=args.verbose)

    def scene_centroid():
        xs = [r["pos"][0] for r in robots.values()]
        ys = [r["pos"][1] for r in robots.values()]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    cx0, cy0 = scene_centroid()
    cam = OrbitCamera([cx0, cy0], back=args.cam_back, height=args.cam_height)
    interactive_cam = show_viewport and not args.no_interactive_cam

    def push_camera():
        if not show_viewport:
            return
        loc, rot = cam.pose()
        try:
            env.move_viewport(loc, rot)
        except Exception:
            pass

    keypoller = KeyPoller(interactive_cam)
    keypoller.__enter__()
    try:
        # settle
        for _ in range(15):
            for r, st in robots.items():
                try:
                    env.act(r, usv_action(st))
                except Exception:
                    pass
            env.tick()
        push_camera()
        if interactive_cam:
            print("\nCAMERA CONTROLS (focus THIS terminal): "
                  "+/- zoom | a/d orbit | r/f raise/lower | i j k l pan | "
                  "t top-down | o oblique | c auto-follow | x quit\n")

        print("=" * 70)
        print("CHANNEL-AWARE FAILURE RECOVERY  (live: ours_correlated, shadow: phi-accrual)")
        print(f"monitor={monitor}  teammates={teammates}")
        print(f"scripted:  outage {T_outage}s | fail {fail_robot}@{T_fail}s | "
              f"partition {part_robot}@{T_part}s")
        print(f"tau={tau:.2f}  hb={hb}s  window={window}s")
        print("=" * 70)

        dt = 1.0 / args.ticks_per_sec
        t = 0.0
        next_print = 0.0

        while t <= args.max_time:
            # ---- scripted ground-truth events -------------------------------
            chan.forced = "bad" if (T_outage[0] <= t < T_outage[1]) else None
            if fail_robot and t >= T_fail and robots[fail_robot]["alive"]:
                robots[fail_robot]["alive"] = False
                log(t, "GROUND_TRUTH_FAIL", fail_robot, "robot stopped + silent")
            if part_robot:
                if T_part[0] <= t < T_part[1]:
                    chan.dead_links.add(part_robot)   # link to monitor down, robot alive
                else:
                    chan.dead_links.discard(part_robot)

            chan.step()

            # ---- heartbeat emission + (lossy) reception at the monitor ------
            if t >= next_hb_emit:
                for j in teammates:
                    if robots[j]["alive"] and chan.delivers(j):
                        delivered[j].append(t)
                next_hb_emit += hb

            # ---- task release + allocation ----------------------------------
            for task in tasks:
                if task["status"] == "unreleased" and t >= task["release_time"]:
                    pid = task["poi_id"]
                    # the POI just became known to the team ("emerges")
                    released_pois.add(pid)
                    emerge_flash[pid] = t + 1.5
                    log(t, "POI_EMERGE", "-", f"{task['task_id']} @poi{pid}")
                    w, _ = choose_winner(pid, robots, pois, args)
                    task["owner"], task["status"] = w, "assigned"
                    _, nr = best_insert(robots[w]["pos"], robots[w]["route"],
                                        pid, pois)
                    robots[w]["route"] = nr
                    messages += len(robots) + 1
                    log(t, "release_assign", w, task["task_id"])

            # ---- detection: run BOTH detectors on each teammate -------------
            for j in teammates:
                # bootstrap guard: a detector cannot suspect a node it has never
                # heard from (avoids a spurious t=0 "infinite silence" fire that
                # would unfairly pad either detector's count).
                if len(delivered[j]) < 2:
                    continue
                silence = t - last_hb(delivered, j, t)
                eta = channel_health(delivered, names, monitor, j, t, window)

                # phi-accrual baseline (SHADOW — counted, not acted on)
                if not base_state[j]["fired"]:
                    if silence > adaptive_timeout(delivered, j, hb):
                        base_state[j]["fired"] = True
                        if robots[j]["alive"]:
                            base_false_takeovers += 1
                            log(t, "phiaccrual_FALSE_TKO", j,
                                f"silence={silence:.1f} (robot alive)")
                        else:
                            log(t, "phiaccrual_detect", j, "(correct)")
                # heard again -> reset shadow
                if silence < hb:
                    base_state[j]["fired"] = False

                # ours_correlated (LIVE controller)
                rho = max(0.0, min(1.0, (silence - 2.0 * hb) / (4.0 * hb)))
                beta = rho * eta
                if not ours_state[j]["fired"] and beta >= tau:
                    ours_state[j]["fired"] = True
                    # OURS fired -> locality-targeted relay before any takeover
                    heard, relays = locality_relay(j, t)
                    if heard:
                        relay_cancels += 1
                        log(t, "ours_fire->RELAY_CANCEL", j,
                            f"beta={beta:.2f} relays={relays} heard->alive, hold")
                    else:
                        # nobody hears it -> treat as a real failure, re-allocate
                        if robots[j]["alive"]:
                            ours_false_takeovers += 1
                            log(t, "ours_FALSE_TKO", j, f"beta={beta:.2f}")
                        else:
                            correct_recoveries += 1
                            log(t, "ours_RECOVER", j, f"beta={beta:.2f} re-allocate")
                        # re-allocate the silent robot's queued tasks
                        for task in tasks:
                            if task["owner"] == j and task["status"] != "done":
                                w, _ = choose_winner(task["poi_id"], robots, pois,
                                                     args, exclude=j)
                                if w != j:
                                    task["owner"] = w
                                    _, nr = best_insert(robots[w]["pos"],
                                                        robots[w]["route"],
                                                        task["poi_id"], pois)
                                    robots[w]["route"] = nr
                                    messages += args.relay_k + 1
                if silence < hb:
                    ours_state[j]["fired"] = False
                    # if a missed real failure is never detected -> coverage hole
                # missed-failure accounting at end

            # ---- motion + pickup --------------------------------------------
            for rid, robot in robots.items():
                if not robot["alive"]:
                    robot["phase"] = "DEAD"
                elif robot["pickup_until"] > t:
                    robot["phase"] = "PICKUP"
                elif robot["route"]:
                    tp = robot["route"][0]
                    robot["phase"] = "ENROUTE"
                    if move_robot_toward(robot, pois[tp], args.robot_speed, dt):
                        robot["phase"] = "PICKUP"
                        robot["pickup_until"] = t + args.pickup_time
                        visited_pois.add(tp)   # this POI has now been searched
                        for task in tasks:
                            if task["poi_id"] == tp and task["status"] != "done":
                                task["status"] = "done"
                                task["done_time"] = t
                                if task["has_item"]:
                                    found_pois.add(tp)   # item retrieved here
                                    try:
                                        if hasattr(env, "agents") and task["object_name"] in env.agents:
                                            env.agents[task["object_name"]].teleport(
                                                location=np.array(
                                                    [pois[tp][0], pois[tp][1], args.object_hide_z],
                                                    dtype=np.float32))
                                    except Exception:
                                        pass
                                break
                        robot["route"].pop(0)
                        log(t, "item_FOUND" if tp in found_pois else "searched",
                            rid, f"poi{tp}")
                else:
                    robot["phase"] = "IDLE"
                try:
                    env.act(rid, usv_action(robot))
                except Exception:
                    pass

            # ---- mark POIs a USV has searched (reached within radius) --------
            for pid, ploc in pois.items():
                if pid in visited_pois or pid not in released_pois:
                    continue
                for r in robots.values():
                    if dist2(r["pos"], ploc) <= args.search_radius:
                        visited_pois.add(pid)
                        break

            # ---- world overlays: red USVs, yellow/blue/lime POIs -------------
            if show_viewport and not args.no_overlays:
                draw_overlays(env, robots, pois, tasks, set(chan.dead_links),
                              visited_pois, found_pois, released_pois, emerge_flash,
                              t, life=max(2.0 * dt, 0.12))

            # ---- viewport camera: follow the active phase, accept live keys ---
            if interactive_cam:
                k = keypoller.poll()
                if k == "x":
                    log(t, "user_quit", "", "camera key x")
                    break
                if k:
                    cam.handle_key(k)
            if cam.follow:
                # frame whatever is most interesting right now
                if fail_robot and abs(t - T_fail) < 12.0:
                    cam.set_target(*robots[fail_robot]["pos"][:2])
                elif part_robot and T_part[0] - 2 <= t < T_part[1] + 4:
                    cam.set_target(*robots[part_robot]["pos"][:2])
                else:
                    cam.set_target(*scene_centroid())
            push_camera()

            env.tick()
            if args.realtime:
                time.sleep(dt)

            if t >= next_print:
                done = sum(1 for x in tasks if x["status"] == "done")
                ch = "BURST-OUTAGE" if chan.bad else "good"
                n_items = sum(1 for x in tasks if x["has_item"])
                phase = ("A:normal" if t < T_outage[0] else
                         "B:OUTAGE" if T_outage[0] <= t < T_outage[1] else
                         "C:FAILURE" if abs(t - T_fail) < 6 else
                         "D:PARTITION" if T_part[0] <= t < T_part[1] else "normal")
                print("\n" + "-" * 78)
                print(f" t={t:6.1f}s  phase={phase:12s} channel={ch:12s} "
                      f"POIs emerged={len(released_pois)}/{len(tasks)}  "
                      f"searched={done}/{len(tasks)}  items found={len(found_pois)}/{n_items}")
                print(f"   OURS: falseTKO={ours_false_takeovers}  relayCancel={relay_cancels}"
                      f"  recover={correct_recoveries}  missed(running)"
                      f"   ||  phi-accrual falseTKO={base_false_takeovers}")
                for j in teammates:
                    tag = ("DEAD" if not robots[j]["alive"] else
                           "PARTITIONED" if j in chan.dead_links else
                           robots[j]["phase"])
                    if len(delivered[j]) < 2:        # bootstrap (never heard yet)
                        print(f"     {j:5s} {tag:11s} silence=  --  eta= --  beta= --"
                              f"  (warmup)  route={robots[j]['route']}")
                        continue
                    sil = t - last_hb(delivered, j, t)
                    eta = channel_health(delivered, names, monitor, j, t, window)
                    rho = max(0.0, min(1.0, (sil - 2.0 * hb) / (4.0 * hb)))
                    beta = rho * eta
                    print(f"     {j:5s} {tag:11s} silence={sil:5.1f}s eta={eta:.2f}"
                          f" beta={beta:.2f}{'  <FIRE>' if beta >= tau else ''}"
                          f"  route={robots[j]['route']}")
                print("-" * 78)
                next_print += 2.0

            if all(x["status"] == "done" for x in tasks):
                log(t, "MISSION_COMPLETE", "",
                    f"all {len(tasks)} POIs searched, {len(found_pois)} items found")
                break
            t += dt

        # any failed robot never recovered = missed
        for j in teammates:
            if not robots[j]["alive"] and not ours_state[j]["fired"]:
                ours_missed += 1

    finally:
        try:
            keypoller.__exit__()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass

    # ---- write logs ------------------------------------------------------
    with ev_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "event", "robot", "note"])
        w.writerows(events)
    done = sum(1 for x in tasks if x["status"] == "done")
    n_items = sum(1 for x in tasks if x["has_item"])
    total_dist = sum(r["distance"] for r in robots.values())
    with sum_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pois_searched", "total_pois", "items_found", "total_items",
                    "mission_time", "ours_false_takeovers", "phiaccrual_false_takeovers",
                    "relay_cancels", "relay_queries", "correct_recoveries",
                    "missed_failures", "messages", "total_distance"])
        w.writerow([done, len(tasks), len(found_pois), n_items, round(t, 2),
                    ours_false_takeovers, base_false_takeovers, relay_cancels,
                    relay_queries, correct_recoveries, ours_missed, messages,
                    round(total_dist, 1)])

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"  POIs searched              : {done}/{len(tasks)}")
    print(f"  items found                : {len(found_pois)}/{n_items}")
    print(f"  OURS false takeovers       : {ours_false_takeovers}")
    print(f"  phi-accrual false takeovers: {base_false_takeovers}   "
          f"(baseline; would have wrongly seized {base_false_takeovers} alive robots)")
    print(f"  relay cancels (Gap-2 saves): {relay_cancels} / {relay_queries} queries")
    print(f"  correct recoveries         : {correct_recoveries}")
    print(f"  missed real failures       : {ours_missed}")
    print(f"  messages / total distance  : {messages} / {total_dist:.0f} m")
    gap = base_false_takeovers / max(ours_false_takeovers, 1)
    print(f"  => false-takeover gap (phi/ours): {gap:.1f}x fewer with channel-awareness")
    print(f"  events : {ev_path}")
    print(f"  summary: {sum_path}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="OpenWater")
    p.add_argument("--num_pois", type=int, default=16,
                   help="POIs to search. ALL of them must be visited for mission complete.")
    p.add_argument("--num_robots", type=int, default=5)
    p.add_argument("--num_items", type=int, default=6,
                   help="How many POIs actually contain a retrievable item (turn lime).")
    p.add_argument("--release_gap", type=float, default=0.6)
    p.add_argument("--robot_speed", type=float, default=3.0)
    p.add_argument("--ticks_per_sec", type=float, default=30.0)
    p.add_argument("--max_time", type=float, default=160.0)
    p.add_argument("--world_extent", type=float, default=110.0)
    p.add_argument("--poi_start_offset", type=float, default=25.0)
    p.add_argument("--surface_z", type=float, default=0.0)
    p.add_argument("--object_z", type=float, default=1.2)
    p.add_argument("--object_hide_z", type=float, default=-30.0)
    p.add_argument("--pickup_time", type=float, default=1.0)
    p.add_argument("--search_radius", type=float, default=4.0,
                   help="A POI counts as 'visited/searched' when a USV gets this close (m).")
    # comms / detection
    p.add_argument("--hb_period", type=float, default=0.75)
    p.add_argument("--good_drop", type=float, default=0.20)
    p.add_argument("--burst_drop", type=float, default=0.90)
    p.add_argument("--c_miss", type=float, default=1.0)
    p.add_argument("--c_conflict", type=float, default=1.0)
    p.add_argument("--relay_k", type=int, default=2)
    # scripted ground-truth events
    p.add_argument("--outage_start", type=float, default=18.0)
    p.add_argument("--outage_len", type=float, default=10.0)
    p.add_argument("--fail_time", type=float, default=45.0)
    p.add_argument("--partition_start", type=float, default=70.0)
    p.add_argument("--partition_len", type=float, default=12.0)
    # allocation
    p.add_argument("--max_route_len", type=int, default=4)
    p.add_argument("--load_penalty", type=float, default=45.0)
    p.add_argument("--busy_penalty", type=float, default=20.0)
    # run control
    p.add_argument("--seed", type=int, default=202606)
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--headless", action="store_true",
                   help="Run without the Unreal viewport window.")
    # camera
    p.add_argument("--cam_height", type=float, default=70.0)
    p.add_argument("--cam_back", type=float, default=55.0)
    p.add_argument("--no_interactive_cam", action="store_true",
                   help="Disable live keyboard camera control (auto-follow only).")
    p.add_argument("--no_overlays", action="store_true",
                   help="Disable colored 3D overlays (red USVs / yellow POIs / state colors).")
    p.add_argument("--no_objects", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output_dir", default="~/phase1a_ws/results/holoocean_channel_aware")
    run(p.parse_args())


if __name__ == "__main__":
    main()
