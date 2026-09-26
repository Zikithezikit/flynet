#!/usr/bin/env python3
"""Interactive GUI: a fly forages for food while its brain keeps learning.

The fly walks a 2-D arena driven by the **real** fly brain loaded from
neuPrint (the full ~66k-neuron male CNS by default).  It has two senses:

* **sight** -- a forward-facing visual field; food inside the field is
  detected by contrast, strongest in the eye that faces it;
* **smell** -- an odor plume around the food, sampled at the two antennae.

Both senses are mixed into the two sensor channels the brain exposes, the
brain settles, and the motor difference becomes a turn.  Reward-gated
Hebbian plasticity (:class:`flynet.learning.MotorRewardHebbian`) reinforces the
co-active synapses of every successful hunt, so the fly visibly gets better
while the window is open.

Run it::

    python examples/fly_foraging_gui.py                # full real brain
    python examples/fly_foraging_gui.py --synthetic    # no token needed
    python examples/fly_foraging_gui.py --smoke        # headless CI check

Controls
--------
``[`` / ``-``   slower          ``]`` / ``+``   faster
``space``       pause / resume  ``r``           restart the current hunt
``b``           reset the brain weights to the connectome's own
``q`` / ``esc`` quit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from flynet.connectome import ConnectomeLoader, ConnectomeUnavailableError
from flynet.learning import MotorRewardHebbian, TrainingLogger
from flynet.network import SpikingNetwork

# ---------------------------------------------------------------------------
# Arena / sensing constants
# ---------------------------------------------------------------------------
ARENA = 100.0
FOOD = (80.0, 80.0)
FOUND_RADIUS = 4.0
MAX_STEPS = 300
MIN_START_DIST = 25.0
CURVE_HORIZON = 50       # initial episode axis; doubles as the run goes on

ANTENNA_OFFSET = 2.7      # antenna tip distance from the body centre
STEP_LEN = 2.0            # walking distance per step while tracking
SEARCH_STEP = 2.5         # walking distance per step while searching
WANDER_TURN = 0.25        # random heading jitter while searching (rad)
SCAN_RATE = 1.1           # head-scan frequency (rad/step)
SCAN_SWEEP = 0.55         # head-scan amplitude (rad)
SEARCH_SWEEP = 0.10       # body turn while searching: wide arcs, not a spin

FOV_DEG = 110.0           # total width of the visual field
SIGHT_RANGE = 35.0        # how far the fly can see food
SIGHT_GAIN = 1.8          # weight of sight vs smell in the sensor channels
SMELL_GAIN = 1.0
SIGHT_NOISE = 0.06        # absolute sensory noise (contrast saturates at 1.0)
SMELL_NOISE = 0.012
ODOR_THRESHOLD = 0.005    # odor strength that marks the plume as usable

STEER_GAIN = 6.0          # motor difference -> turn angle
MAX_TURN = 0.45           # turn-angle clip per step (rad)

SETTLE_ITERS_FULL = 6     # one settle iteration is a 228k-nnz mat-vec (~1ms)
SETTLE_ITERS_SMALL = 12
ACTIVITY_TOPK = 256       # neurons kept per learning event
HEBBIAN_EVENTS = 24       # events per plasticity update (bounds the cost)
ROLLING_WINDOW = 20       # episodes in the rolling success/step average

PROGRESS_GAIN = 4.0       # reward per unit of distance closed on the food
SUCCESS_BONUS = 3.0       # terminal reward for reaching the food
FAILURE_PENALTY = 0.0     # terminal reward for running out of steps
PROBE_SEED = 4242         # fixed seed so the steering probe is repeatable

MIN_SPEED = 1
MAX_SPEED = 16
FLASH_FRAMES = 10       # how long the "FOUND FOOD!" banner stays up


# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------
def wrap_angle(angle: float) -> float:
    """Wrap an angle into ``(-pi, pi]``.

    Args:
        angle (float): Angle in radians.

    Returns:
        float: The equivalent angle in ``(-pi, pi]``.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def smell_at(pos: np.ndarray, food: tuple[float, float]) -> float:
    """Return the odor concentration at *pos*.

    Args:
        pos (np.ndarray): ``(x, y)`` position in the arena.
        food (tuple[float, float]): ``(x, y)`` position of the food.

    Returns:
        float: Odor strength, ``1.0`` at the food and decaying with the
        squared distance.
    """
    d = float(np.linalg.norm(np.asarray(food, dtype=float) - pos))
    return 2.0 / (1.0 + 0.3 * d * d)


def bearing_error(
    pos: np.ndarray, heading: float, food: tuple[float, float]
) -> float:
    """Signed angle from *heading* to the food.

    The fly's left is its forward direction rotated by +90 degrees, so a
    positive result means the food lies off the fly's left side and a
    negative one off its right.

    Args:
        pos (np.ndarray): ``(x, y)`` position in the arena.
        heading (float): Direction the fly faces, in radians.
        food (tuple[float, float]): ``(x, y)`` position of the food.

    Returns:
        float: Signed bearing error in ``(-pi, pi]``, positive to the left.
    """
    to_food = np.asarray(food, dtype=float) - pos
    return wrap_angle(float(np.arctan2(to_food[1], to_food[0])) - heading)


def antenna_readings(
    pos: np.ndarray, heading: float, food: tuple[float, float]
) -> tuple[float, float]:
    """Sample the odor plume at the two antenna tips.

    The antennae stick out at +/-45 degrees from the heading, so a plume
    coming from the right biases the right antenna.

    Args:
        pos (np.ndarray): ``(x, y)`` position in the arena.
        heading (float): Direction the fly faces, in radians.
        food (tuple[float, float]): ``(x, y)`` position of the food.

    Returns:
        tuple[float, float]: ``(right, left)`` odor readings.
    """
    d = np.array([np.cos(heading), np.sin(heading)])
    s = np.array([-d[1], d[0]])  # left-hand perpendicular
    right_tip = pos + ANTENNA_OFFSET * (d - s)
    left_tip = pos + ANTENNA_OFFSET * (d + s)
    return smell_at(right_tip, food), smell_at(left_tip, food)


def visual_readings(
    pos: np.ndarray,
    head: float,
    food: tuple[float, float],
    fov_deg: float = FOV_DEG,
    view_range: float = SIGHT_RANGE,
) -> tuple[float, float]:
    """Sample the two eyes for the food inside the visual field.

    Food outside the field of view or beyond *view_range* is invisible.
    Inside it, contrast falls off with distance and is shared between the
    eyes according to which side of the midline the food sits on; food
    straight ahead lights both eyes equally, which drives the fly forward.
    A food off the fly's left brightens the left eye, and vice versa.

    Args:
        pos (np.ndarray): ``(x, y)`` position in the arena.
        head (float): Direction the head faces, in radians.
        food (tuple[float, float]): ``(x, y)`` position of the food.
        fov_deg (float): Total width of the visual field, in degrees.
        view_range (float): Maximum distance at which food is visible.

    Returns:
        tuple[float, float]: ``(right, left)`` contrast readings.
    """
    err = bearing_error(pos, head, food)
    dist = float(np.linalg.norm(np.asarray(food, dtype=float) - pos))
    half_fov = np.radians(fov_deg) / 2.0
    if dist > view_range or abs(err) > half_fov:
        return 0.0, 0.0
    contrast = 1.0 / (1.0 + 0.01 * dist * dist)
    # bearing_error is positive to the fly's left, so the right eye's
    # share of the image shrinks as the food moves left.
    right_share = 0.5 - 0.5 * err / half_fov
    return contrast * right_share, contrast * (1.0 - right_share)


def topk_activity(trace: list[np.ndarray], k: int = ACTIVITY_TOPK) -> tuple:
    """Compress a settling trace into a sparse top-k activity vector.

    The full brain settles sparsely, so keeping only the ``k`` strongest
    neurons turns a 66k-element vector into a few hundred values.  The
    result is the ``(indices, values)`` form that
    :meth:`flynet.learning.MotorRewardHebbian.update` understands.

    Args:
        trace (list[np.ndarray]): Activity vectors, one per iteration.
        k (int): Number of neurons to keep.

    Returns:
        tuple: ``(indices, values)`` numpy arrays of equal length.
    """
    if not trace:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, np.zeros(0, dtype=np.float64)
    act = np.mean(trace[-2:], axis=0)
    keep = int(min(k, act.size))
    idx = np.argpartition(np.abs(act), -keep)[-keep:]
    return idx, act[idx]


def encode_readings(
    smell: tuple[float, float], sight: tuple[float, float]
) -> tuple[float, float]:
    """Turn the four sensory channels into the brain's two sensor channels.

    The brain's motor pair is far more sensitive to a left/right
    *contrast* than to a symmetric drive: probing the real brain shows a
    symmetric input produces a motor difference about eight times weaker
    than an antisymmetric one of the same size.  Feeding the raw levels
    would therefore let a constant offset dominate the steering, so the
    contrast between the two sides is what gets encoded, split so that
    only the stronger side drives the network.  Food straight ahead
    encodes as no input at all, which walks the fly forward.

    Args:
        smell (tuple[float, float]): ``(right, left)`` odor readings.
        sight (tuple[float, float]): ``(right, left)`` contrast readings.

    Returns:
        tuple[float, float]: ``(rho_right, rho_left)`` sensor channels.
    """
    delta = SIGHT_GAIN * (sight[0] - sight[1]) + SMELL_GAIN * (
        smell[0] - smell[1]
    )
    return max(delta, 0.0), max(-delta, 0.0)


# ---------------------------------------------------------------------------
# Episode / forager
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    """State of a single hunt for food.

    Attributes:
        pos (np.ndarray): Current ``(x, y)`` position.
        heading (float): Body direction in radians.
        scan_phase (float): Head-scan oscillator phase in radians.
        step (int): Steps taken so far.
        found (bool): Whether the food was reached.
        events (list): ``(rho_right, rho_left, activity, reward)`` learning
            events, one per guided step.
        path (list[np.ndarray]): Visited positions, oldest first.
        mode (str): Sensory mode of the last step.
        start_dist (float): Initial distance to the food.
        progress (float): Reward accumulated by the guided steps.
    """

    pos: np.ndarray
    heading: float
    scan_phase: float = 0.0
    step: int = 0
    found: bool = False
    events: list = field(default_factory=list)
    path: list = field(default_factory=list)
    mode: str = "search"
    start_dist: float = 0.0
    progress: float = 0.0


class FlyForager:
    """Headless foraging simulation driven by a :class:`SpikingNetwork`.

    Owns the sensory mixing, the three behaviour modes (visual search,
    odor tracking, and plume following) and the steering law.  It never
    touches matplotlib, so it can be unit-tested without a display.
    """

    def __init__(
        self,
        net: SpikingNetwork,
        food: tuple[float, float] = FOOD,
        arena: float = ARENA,
        settle_iters: int = SETTLE_ITERS_FULL,
        rng: np.random.Generator | None = None,
        invert_steer: bool = False,
        Minv: np.ndarray | None = None,
    ) -> None:
        """Build a forager on top of *net* and calibrate it.

        Args:
            net (SpikingNetwork): Brain providing the steering command.
            food (tuple[float, float]): ``(x, y)`` position of the food.
            arena (float): Side length of the square arena.
            settle_iters (int): Settling iterations per steering call.
            rng (np.random.Generator | None): Random generator; a fresh one
                is created when ``None``.
            invert_steer (bool): Flip the turn direction, for brains whose
                motor pair is wired the other way round.
            Minv (np.ndarray | None): Frozen inverse calibration matrix.
                Pass the initial one to keep steering comparable while the
                weights are still moving; ``None`` recalibrates from *net*.
        """
        self.net = net
        self.food = food
        self.arena = arena
        self.settle_iters = settle_iters
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.invert_steer = invert_steer

        # One calibration, held for the whole run: turn() documents that a
        # cached Minv keeps steering consistent while weights are updated.
        # Recalibrating every frame would cancel out learning, because a
        # 2x2 pinv always inverts whatever the current matrix is.
        if Minv is None:
            matrix, self.Minv = net.calibrate()
        else:
            matrix, _ = net.calibrate()
            self.Minv = Minv
        self.cond = float(np.linalg.cond(matrix))

        self.episode = Episode(
            pos=np.zeros(2), heading=0.0, path=[np.zeros(2)]
        )
        self.distance = 0.0
        self.readings = (0.0, 0.0)
        self.sight = (0.0, 0.0)
        self.smell = (0.0, 0.0)
        self.drv = 0.0
        self.step_count = 0
        self.reset()

    # -- episode control ------------------------------------------------
    def reset(
        self, start: np.ndarray | None = None, heading: float | None = None
    ) -> Episode:
        """Start a new hunt from a random spot far enough from the food.

        Args:
            start (np.ndarray | None): Explicit ``(x, y)`` start; a random
                one is drawn when ``None``.
            heading (float | None): Explicit body direction in radians; a
                random one is drawn when ``None``.

        Returns:
            Episode: The freshly reset episode.
        """
        if start is None:
            for _ in range(64):
                cand = self.rng.uniform(4.0, self.arena - 4.0, size=2)
                if np.linalg.norm(cand - np.asarray(self.food)) >= MIN_START_DIST:
                    start = cand
                    break
            else:  # pragma: no cover - arena too small to satisfy the rule
                start = self.rng.uniform(4.0, self.arena - 4.0, size=2)
        if heading is None:
            heading = float(self.rng.uniform(-np.pi, np.pi))
        self.episode = Episode(
            pos=np.asarray(start, dtype=float).copy(),
            heading=float(heading),
            scan_phase=0.0,
            path=[np.asarray(start, dtype=float).copy()],
            start_dist=float(
                np.linalg.norm(np.asarray(start) - np.asarray(self.food))
            ),
        )
        self.distance = self.episode.start_dist
        self.drv = 0.0
        self.readings = (0.0, 0.0)
        self.sight = (0.0, 0.0)
        self.smell = (0.0, 0.0)
        return self.episode

    # -- sensing --------------------------------------------------------
    def sense(self, pos: np.ndarray, heading: float) -> tuple[float, float, str]:
        """Mix sight and smell into the brain's two sensor channels.

        Three regimes, chosen by what the fly can actually perceive:

        ``sight``
            the food is inside the visual field -- steer on the image;
        ``smell``
            the plume is strong enough to follow -- steer on the odor
            difference between the antennae;
        ``search``
            nothing usable yet -- sweep the head and body while walking.

        Args:
            pos (np.ndarray): ``(x, y)`` position in the arena.
            heading (float): Body direction in radians.

        Returns:
            tuple: ``(rho_right, rho_left, mode)``.
        """
        ep = self.episode
        ep.scan_phase += SCAN_RATE
        head = heading + SCAN_SWEEP * float(np.sin(ep.scan_phase))

        smell_r, smell_l = antenna_readings(pos, heading, self.food)
        sight_r, sight_l = visual_readings(pos, head, self.food)
        # Real senses are noisy, and the noise is what gives the learner
        # something to improve on: early on the fly steers noisily, and
        # reward-gated plasticity sharpens the motor projection.  The
        # noise floor is absolute, so uncertainty never vanishes.
        if SIGHT_NOISE or SMELL_NOISE:
            sight_r = max(0.0, sight_r + self.rng.normal(0.0, SIGHT_NOISE))
            sight_l = max(0.0, sight_l + self.rng.normal(0.0, SIGHT_NOISE))
            smell_r = max(0.0, smell_r + self.rng.normal(0.0, SMELL_NOISE))
            smell_l = max(0.0, smell_l + self.rng.normal(0.0, SMELL_NOISE))
        self.smell = (smell_r, smell_l)
        self.sight = (sight_r, sight_l)

        rho_right, rho_left = encode_readings(
            (smell_r, smell_l), (sight_r, sight_l)
        )

        if sight_r + sight_l > 1e-9:
            mode = "sight"
        elif max(smell_r, smell_l) >= ODOR_THRESHOLD:
            mode = "smell"
        else:
            mode = "search"
        return rho_right, rho_left, mode

    # -- one simulation step --------------------------------------------
    def step(self) -> Episode:
        """Advance the hunt by one step.

        Guided steps (``sight`` / ``smell``) ask the brain for a steering
        command and record a learning event; ``search`` steps sweep the head
        and body so the fly scans the horizon instead.

        Returns:
            Episode: The current episode state.
        """
        ep = self.episode
        if ep.found:
            return ep

        pending: list = []
        rho_right, rho_left, mode = self.sense(ep.pos, ep.heading)
        self.readings = (rho_right, rho_left)
        ep.mode = mode
        before = self.distance

        if mode == "search":
            ep.heading = wrap_angle(
                ep.heading
                + SEARCH_SWEEP
                + self.rng.uniform(-WANDER_TURN, WANDER_TURN)
            )
            advance = SEARCH_STEP
        else:
            self.drv, trace = self.net.turn(
                rho_right, rho_left, Minv=self.Minv,
                iterations=self.settle_iters,
            )
            sign = -1.0 if self.invert_steer else 1.0
            turn = float(
                np.clip(sign * STEER_GAIN * self.drv, -MAX_TURN, MAX_TURN)
            )
            ep.heading = wrap_angle(ep.heading + turn)
            advance = STEP_LEN
            pending.append((rho_right, rho_left, topk_activity(trace)))

        direction = np.array([np.cos(ep.heading), np.sin(ep.heading)])
        ep.pos = np.clip(ep.pos + direction * advance, 0.0, self.arena)
        ep.path.append(ep.pos.copy())
        ep.step += 1
        self.step_count += 1
        self.distance = float(np.linalg.norm(ep.pos - np.asarray(self.food)))

        # Third factor: a guided step is rewarded by the progress it made
        # towards the food and punished when it wandered further away.
        if mode != "search":
            reward = (before - self.distance) * PROGRESS_GAIN
            ep.progress += reward
            rho_r, rho_l, activity = pending.pop()
            ep.events.append((rho_r, rho_l, activity, reward))

        if self.distance < FOUND_RADIUS:
            ep.found = True
        return ep

    def auto_steer_sign(self, steps: int = 40) -> int:
        """Find the turn direction that actually closes on the food.

        Which way the motor difference has to be driven to turn towards
        the food depends on how the brain's motor pair happens to be
        wired, so it is measured instead of assumed: two short probes
        start the fly short of the food, one with the food off its left
        and one off its right, and the sign that ends up closer wins.

        The probes do not learn, so *net* is used as-is.

        Args:
            steps (int): Probe length in steps.

        Returns:
            int: ``-1`` when the steering must be inverted, ``1``
            otherwise.  ``1`` is returned when neither sign is clearly
            better, so a degenerate brain falls back to the default.
        """
        food = np.asarray(self.food, dtype=float)
        probes = (
            np.array([food[0] - 20.0, food[1] - 15.0]),   # food off the left
            np.array([food[0] - 20.0, food[1] + 15.0]),   # food off the right
        )
        totals: dict[bool, float] = {}
        for invert in (False, True):
            probe = FlyForager(
                self.net,
                food=self.food,
                arena=self.arena,
                settle_iters=self.settle_iters,
                rng=np.random.default_rng(PROBE_SEED),
                invert_steer=invert,
                Minv=self.Minv,
            )
            total = 0.0
            for start in probes:
                probe.reset(start=start, heading=0.0)
                for _ in range(steps):
                    probe.step()
                    if probe.episode.found:
                        break
                total += probe.distance
            totals[invert] = total
        if abs(totals[False] - totals[True]) < 1e-6:
            return 1
        return -1 if totals[True] < totals[False] else 1

    # -- learning -------------------------------------------------------
    def learning_batch(
        self, limit: int = HEBBIAN_EVENTS, found: bool | None = None
    ) -> list:
        """Return recent guided steps as signed-reward learning events.

        The last *limit* events are returned: the plasticity update costs
        one sparse pass over every synapse per event, so a long hunt would
        otherwise freeze the window.  When *found* is given, a terminal
        bonus or penalty is added to the last event so the whole hunt --
        not just its individual steps -- carries the outcome.

        Args:
            limit (int): Maximum number of events to return.
            found (bool | None): Outcome of the hunt; ``None`` scores the
                steps on progress alone.

        Returns:
            list: ``(rho_right, rho_left, activity, reward)`` tuples.
        """
        if limit <= 0:
            return []
        batch = self.episode.events[-limit:]
        if found is None or not batch:
            return list(batch)
        rho_r, rho_l, activity, reward = batch[-1]
        terminal = SUCCESS_BONUS if found else FAILURE_PENALTY
        batch = list(batch[:-1]) + [(rho_r, rho_l, activity,
                                     reward + terminal)]
        return batch


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------
class TrainingStats:
    """Rolling success/step statistics plus a full learning curve.

    Wraps :class:`flynet.learning.TrainingLogger` so the plotted curve is
    the same history the library's learning plots consume.
    """

    def __init__(self, window: int = ROLLING_WINDOW) -> None:
        """Create empty statistics.

        Args:
            window (int): Number of recent episodes used for the rolling
                averages.
        """
        self.window = window
        self.logger = TrainingLogger()
        self.episodes = 0
        self.successes = 0
        self.best_steps: int | None = None
        self.history: list[tuple[bool, int]] = []
        self.recent: list[tuple[bool, int]] = []
        self.steps_done = 0

    def record(self, found: bool, steps: int) -> None:
        """Record the outcome of one hunt.

        Args:
            found (bool): Whether the fly reached the food.
            steps (int): Steps the hunt took.

        Returns:
            None: Appends to the history and the rolling window.
        """
        self.episodes += 1
        self.steps_done += steps
        if found:
            self.successes += 1
            if self.best_steps is None or steps < self.best_steps:
                self.best_steps = steps
        self.history.append((found, steps))
        self.recent.append((found, steps))
        if len(self.recent) > self.window:
            self.recent.pop(0)
        self.logger.log_episode(1.0 if found else 0.0, steps)

    def success_rate(self) -> float:
        """Return the success rate over the rolling window.

        Returns:
            float: Fraction of recent hunts that found the food, ``0.0``
            before the first hunt finishes.
        """
        if not self.recent:
            return 0.0
        return sum(1 for found, _ in self.recent if found) / len(self.recent)

    def avg_steps(self) -> float:
        """Return the mean hunt length over the rolling window.

        Returns:
            float: Mean steps per hunt, ``0.0`` before the first hunt.
        """
        if not self.recent:
            return 0.0
        return float(np.mean([steps for _, steps in self.recent]))

    def smoothed_steps(self, window: int = 5) -> tuple[list[int], list[float]]:
        """Return a moving average of hunt length for plotting.

        Args:
            window (int): Width of the moving average in episodes.

        Returns:
            tuple[list[int], list[float]]: Episode numbers and smoothed
            steps, both shorter than the full history.
        """
        episodes, steps = self.logger.get_learning_curve()
        if not steps:
            return [], []
        arr = np.asarray(steps, dtype=float)
        if len(arr) < window:
            return episodes, list(arr)
        kernel = np.ones(window) / window
        smoothed = np.convolve(arr, kernel, mode="valid")
        return list(episodes[window - 1:]), list(smoothed)

    def success_series(self) -> tuple[list[int], list[float]]:
        """Return the rolling success rate after each episode.

        Returns:
            tuple[list[int], list[float]]: Episode numbers and the success
            rate over the trailing window, ready to plot.
        """
        xs: list[int] = []
        ys: list[float] = []
        for i, (found, _) in enumerate(self.history, start=1):
            chunk = self.history[max(0, i - self.window):i]
            xs.append(i)
            ys.append(sum(1 for f, _ in chunk if f) / len(chunk))
        return xs, ys


def apply_learning(
    net: SpikingNetwork,
    hebbian: MotorRewardHebbian,
    events: list,
) -> bool:
    """Apply one reward-gated Hebbian update in place.

    Args:
        net (SpikingNetwork): Brain whose weights are updated.
        hebbian (MotorRewardHebbian): Plasticity rule.
        events (list): Learning events from a successful hunt.

    Returns:
        bool: ``True`` when at least one event was applied.
    """
    if not events:
        return False
    hebbian.update(net, events)
    return True


# ---------------------------------------------------------------------------
# Brain loading
# ---------------------------------------------------------------------------
def load_brain(args: argparse.Namespace) -> tuple[SpikingNetwork, str, str]:
    """Load the brain requested on the command line.

    Real data is the default and the only silent path.  ``--offline`` reads
    the cache only, and ``--synthetic`` must be asked for by name and is
    reported loudly.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: ``(net, banner, description)`` where *banner* is the
        provenance line to print and *description* names the dataset.

    Raises:
        SystemExit: If the requested real brain is unavailable.
    """
    if args.synthetic:
        ids, edges, motors = ConnectomeLoader.synthetic_brain(
            max_edges=40, seed=args.seed
        )
        return (
            SpikingNetwork(ids, edges, motor_ids=motors),
            "[connectome] --synthetic: using SYNTHETIC brain",
            "synthetic 41-neuron test brain",
        )

    loader = ConnectomeLoader(dataset=args.dataset)
    try:
        if args.offline:
            if args.mini:
                cached = loader.load_cached_mini("DNge104")
                if cached is None:
                    raise SystemExit(_cold_cache_message(loader, "DNge104"))
                ids, edges, motors = cached
            else:
                cached = loader.load_cached_full()
                if cached is None:
                    raise SystemExit(_cold_cache_message(loader, args.dataset))
                ids, edges = cached
                motors = []
        elif args.mini:
            ids, edges, motors = loader.load_or_fetch_mini(
                neuron_type="DNge104", max_edges=40
            )
        else:
            ids, edges, motors = loader.load_or_fetch_full(
                min_weight=args.min_weight
            )
    except ConnectomeUnavailableError as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1) from exc

    source = loader.last_source
    net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
    scope = "DNge104 mini brain" if args.mini else "full connectome"
    return (
        net,
        f"[connectome] data source: {source} ({scope}, {args.dataset})",
        f"{scope} from {args.dataset}",
    )


def _cold_cache_message(loader: ConnectomeLoader, what: str) -> str:
    """Build the error shown when ``--offline` finds no cached brain.

    Args:
        loader (ConnectomeLoader): Loader whose cache directory is empty.
        what (str): The dataset or neuron type that was requested.

    Returns:
        str: The error message.
    """
    return (
        f"offline: no cached real brain for {what!r} in {loader.cache_dir}. "
        "Run once with neuPrint access to populate the cache. flynet never "
        "substitutes synthetic data here; pass --synthetic to ask for the "
        "test brain explicitly."
    )


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ForagingApp:
    """Matplotlib front end: draws the arena and drives the forager.

    The figure is built once and only artist data is updated per frame, so
    the window stays responsive while the brain settles.
    """

    def __init__(
        self,
        forager: FlyForager,
        stats: TrainingStats,
        hebbian: MotorRewardHebbian,
        banner: str,
        description: str,
        speed: int = 1,
    ) -> None:
        """Create the figure and its artists.

        Args:
            forager (FlyForager): Simulation to drive.
            stats (TrainingStats): Metrics to display.
            hebbian (MotorRewardHebbian): Plasticity rule applied to each hunt.
            banner (str): Provenance line for the title.
            description (str): Human-readable brain description.
            speed (int): Simulation steps per animation frame.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge

        self.forager = forager
        self.stats = stats
        self.hebbian = hebbian
        self.banner = banner
        self.description = description
        self.speed = int(np.clip(speed, MIN_SPEED, MAX_SPEED))
        self.paused = False
        self.running = True
        self.started = time.time()
        self.last_learn = "start"
        self.slider = None
        self.anim = None
        self.flash_frames = 0

        self.fig = plt.figure(figsize=(13.0, 7.2))
        self.ax = self.fig.add_axes([0.035, 0.07, 0.55, 0.86])
        self.ax_curve = self.fig.add_axes([0.655, 0.70, 0.32, 0.23])
        self.ax_hud = self.fig.add_axes([0.655, 0.06, 0.32, 0.50])
        self.ax_hud.set_axis_off()

        # -- arena ------------------------------------------------------
        arena = forager.arena
        grid = np.linspace(0.0, arena, 96)
        gx, gy = np.meshgrid(grid, grid)
        field = 2.0 / (1.0 + 0.3 * ((gx - forager.food[0]) ** 2
                                    + (gy - forager.food[1]) ** 2))
        # The plume decays as 1/d^2, so plot a compressed field: the
        # physics still uses smell_at(), this is display only.
        shown = (field / field.max()) ** 0.18
        self.ax.imshow(
            shown, origin="lower", extent=(0, arena, 0, arena),
            cmap="YlGn", alpha=0.45, zorder=0,
        )
        self.ax.set_xlim(0, arena)
        self.ax.set_ylim(0, arena)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title(self.banner, fontsize=9)

        self.ax.plot(
            forager.food[0], forager.food[1], "*", color="gold",
            markersize=22, markeredgecolor="black", zorder=6,
            label="food",
        )
        (self.trail,) = self.ax.plot([], [], "-", color="black", alpha=0.35,
                                     linewidth=1.0, zorder=2)
        self.fov = Wedge(
            (0, 0), SIGHT_RANGE, 0, 1, facecolor="dodgerblue", alpha=0.13,
            edgecolor="dodgerblue", zorder=1,
        )
        self.ax.add_patch(self.fov)
        (self.body,) = self.ax.plot([], [], "-", color="black",
                                    linewidth=2.5, zorder=4)
        self.ax.plot([], [], "o", color="black", markersize=4, zorder=4)
        self.fly_marker = self.ax.lines[-1]
        (self.antenna,) = self.ax.plot([], [], "-", color="darkgreen",
                                       linewidth=1.4, zorder=4)
        self.flash = self.ax.text(
            0.5, 0.94, "", transform=self.ax.transAxes, ha="center",
            fontsize=15, fontweight="bold", color="crimson", zorder=8,
        )

        # -- learning curve ---------------------------------------------
        # Fixed limits keep the tick labels static, which is what makes
        # blitting safe: nothing outside the returned artists ever moves.
        (self.curve_raw,) = self.ax_curve.plot(
            [], [], "-", color="steelblue", alpha=0.35, label="steps"
        )
        (self.curve_avg,) = self.ax_curve.plot(
            [], [], "-", color="crimson", linewidth=2, label="smoothed"
        )
        self.ax_curve.set_xlim(0, CURVE_HORIZON)
        self.ax_curve.set_ylim(0, float(MAX_STEPS))
        self.horizon = CURVE_HORIZON
        self.full_redraw = False
        self.ax_curve.set_xlabel("episode", fontsize=8)
        self.ax_curve.set_ylabel("steps to food", fontsize=8)
        self.ax_curve.tick_params(labelsize=7)
        self.ax_curve.set_title("Learning curve", fontsize=10)
        self.ax_rate = self.ax_curve.twinx()
        (self.curve_rate,) = self.ax_rate.plot(
            [], [], "-", color="seagreen", linewidth=1.5, alpha=0.9
        )
        self.ax_rate.set_ylabel("success rate", fontsize=8, color="seagreen")
        self.ax_rate.tick_params(labelsize=7, colors="seagreen")
        self.ax_rate.set_ylim(-0.05, 1.05)

        self.hud = self.ax_hud.text(
            0.0, 1.0, "", va="top", family="monospace", fontsize=9.5,
            transform=self.ax_hud.transAxes,
        )

    # -- interaction ----------------------------------------------------
    def add_controls(self) -> None:
        """Attach the speed slider and keyboard bindings to the figure.

        Safe to call without a display: the slider is just artists, so
        the documentation figures reuse this to show it.

        Returns:
            None: Registers matplotlib callbacks on the figure canvas.
        """
        from matplotlib.widgets import Slider

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        slider_ax = self.fig.add_axes([0.70, 0.605, 0.20, 0.022])
        self.slider = Slider(
            slider_ax, "steps/frame", MIN_SPEED, MAX_SPEED,
            valinit=self.speed, valstep=1,
        )
        self.slider.on_changed(self.on_slide)
        self.fig.canvas.draw_idle()

    def on_slide(self, value: float) -> None:
        """Apply a new speed from the slider.

        Args:
            value (float): Requested steps per frame.

        Returns:
            None: Updates the simulation rate.
        """
        self.set_speed(int(value))

    def set_speed(self, speed: int) -> None:
        """Clamp and apply a new steps-per-frame value.

        Args:
            speed (int): Requested steps per frame.

        Returns:
            None: Updates the simulation rate.
        """
        self.speed = int(np.clip(speed, MIN_SPEED, MAX_SPEED))
        if self.slider is not None and int(self.slider.val) != self.speed:
            self.slider.set_val(self.speed)

    def on_key(self, event) -> None:
        """Handle a keypress in the figure window.

        Args:
            event: The matplotlib key event.

        Returns:
            None: Changes speed, pauses, restarts or quits.
        """
        key = (event.key or "").lower()
        if key in ("]", "+", "="):
            self.set_speed(self.speed + 1)
        elif key in ("[", "-"):
            self.set_speed(self.speed - 1)
        elif key == " ":
            self.paused = not self.paused
        elif key == "r":
            self.forager.reset()
        elif key == "b":
            self.reset_brain()
        elif key in ("q", "escape"):
            self.running = False

    def on_close(self, _event) -> None:
        """Stop the animation when the window is closed.

        Args:
            _event: The matplotlib close event.

        Returns:
            None: Stops the animation loop.
        """
        self.running = False

    def reset_brain(self) -> None:
        """Restore the connectome's own weights, discarding learning.

        Returns:
            None: Replaces the brain with an untrained copy.
        """
        self.forager.net = self.forager.net.copy()
        self.forager = FlyForager(
            self.forager.net,
            food=self.forager.food,
            arena=self.forager.arena,
            settle_iters=self.forager.settle_iters,
            rng=self.forager.rng,
            invert_steer=self.forager.invert_steer,
        )
        self.last_learn = "brain reset to connectome weights"

    # -- simulation loop -------------------------------------------------
    def finish_episode(self) -> None:
        """Record the finished hunt and reinforce it when it succeeded.

        Returns:
            None: Applies plasticity and starts the next hunt.
        """
        ep = self.forager.episode
        self.stats.record(ep.found, ep.step)
        batch = self.forager.learning_batch(found=ep.found)
        if apply_learning(self.forager.net, self.hebbian, batch):
            self.last_learn = (
                f"{len(batch)} events, "
                f"reward {ep.progress:+.1f}"
                f"{' +bonus' if ep.found else ' -penalty'}"
            )
        self.flash.set_text("FOUND FOOD!" if ep.found else "")
        self.flash_frames = FLASH_FRAMES if ep.found else 0
        self.forager.reset()

    def tick(self) -> None:
        """Advance the simulation by ``speed`` steps.

        Returns:
            None: Runs the forager and closes out finished episodes.
        """
        if self.paused:
            return
        for _ in range(self.speed):
            if self.forager.episode.found or \
                    self.forager.episode.step >= MAX_STEPS:
                self.finish_episode()
            self.forager.step()
        if self.forager.episode.found or \
                self.forager.episode.step >= MAX_STEPS:
            self.finish_episode()

    def on_frame(self, _frame) -> list:
        """Animation callback: tick, update artists, and stop when asked.

        Returns the list of artists that changed so that
        :class:`matplotlib.animation.FuncAnimation` can blit just those,
        which is far cheaper than redrawing the whole figure.
        """
        if not self.running:
            return []
        self.tick()
        self.draw()
        if self.full_redraw:
            # The axis limits just moved, so the cached blit background is
            # stale: repaint everything once, then go back to blitting.
            self.full_redraw = False
            self.fig.canvas.draw()
        return self.dirty_artists()

    def dirty_artists(self) -> list:
        """Return every artist that :meth:`draw` updates.

        Returns:
            list: Artists to re-blit, in draw order.
        """
        return [
            self.trail, self.fov, self.body, self.fly_marker, self.antenna,
            self.flash, self.curve_raw, self.curve_avg, self.curve_rate,
            self.hud,
        ]

    def draw(self) -> None:
        """Refresh every artist from the current simulation state.

        Returns:
            None: Updates arena, learning curve and HUD.  Deliberately
            does not draw the canvas: the animation callback blits.
        """
        forager = self.forager
        ep = forager.episode
        pos = ep.pos
        heading = ep.heading
        head = heading + SCAN_SWEEP * float(np.sin(ep.scan_phase))

        d = np.array([np.cos(heading), np.sin(heading)])
        s = np.array([-d[1], d[0]])
        self.body.set_data([pos[0] - 3 * d[0], pos[0] + 3 * d[0]],
                           [pos[1] - 3 * d[1], pos[1] + 3 * d[1]])
        self.fly_marker.set_data([pos[0] + 3 * d[0]], [pos[1] + 3 * d[1]])
        self.antenna.set_data(
            [pos[0] + 2 * d[0] - 3 * s[0], pos[0] + 2 * d[0] + 3 * s[0]],
            [pos[1] + 2 * d[1] - 3 * s[1], pos[1] + 2 * d[1] + 3 * s[1]],
        )
        self.fov.set_center((float(pos[0]), float(pos[1])))
        self.fov.set_theta1(np.degrees(head) - FOV_DEG / 2.0)
        self.fov.set_theta2(np.degrees(head) + FOV_DEG / 2.0)
        path = ep.path
        self.trail.set_data([p[0] for p in path], [p[1] for p in path])

        episodes, steps = self.stats.logger.get_learning_curve()
        self.curve_raw.set_data(episodes, steps)
        sm_x, sm_y = self.stats.smoothed_steps()
        self.curve_avg.set_data(sm_x, sm_y)
        rate_x, rate_y = self.stats.success_series()
        self.curve_rate.set_data(rate_x, rate_y)

        # Grow the episode axis in coarse steps so the curve stays
        # readable without moving tick labels every frame (which would
        # invalidate blitting).
        if episodes and episodes[-1] > self.horizon * 0.9:
            self.horizon *= 2
            self.ax_curve.set_xlim(0, self.horizon)
            self.full_redraw = True

        elapsed = max(time.time() - self.started, 1e-9)
        rate = self.forager.step_count / elapsed * 60.0
        n_active = len(self.forager.net.get_activity(0.05))
        mean_w = float(self.forager.net.W.data.mean())
        lines = [
            f"brain      {len(self.forager.net.ids):,} neurons",
            f"           {self.forager.net.W.nnz:,} synapses",
            f"dataset    {self.description}",
            "",
            f"episode    {self.stats.episodes}",
            f"mode       {ep.mode}",
            f"distance   {forager.distance:5.1f} (start {ep.start_dist:.0f})",
            f"motor drv  {forager.drv:+.3f}",
            f"sight R/L  {forager.sight[0]:.3f} / {forager.sight[1]:.3f}",
            f"smell R/L  {forager.smell[0]:.3f} / {forager.smell[1]:.3f}",
            f"neurons on {n_active:,}",
            f"mean syn.  {mean_w:.5f}",
            "",
            f"success    {self.stats.success_rate():.0%} (last "
            f"{len(self.stats.recent)})",
            f"avg steps  {self.stats.avg_steps():.1f}",
            f"best steps {self.stats.best_steps if self.stats.best_steps else '-'}",
            f"hunts      {self.stats.successes}/{self.stats.episodes}",
            f"learn      {self.last_learn}",
            "",
            f"speed      {self.speed} steps/frame"
            f"{'  [PAUSED]' if self.paused else ''}",
            f"steps sim  {self.forager.step_count:,}",
            f"rate       {rate:,.0f} steps/min",
            "",
            "[ / ]  speed      space  pause",
            "r  restart hunt     b  reset brain",
            "q  quit",
        ]
        if self.flash_frames > 0:
            self.flash_frames -= 1
            if self.flash_frames == 0:
                self.flash.set_text("")
        self.hud.set_text("\n".join(lines))

    def summary(self) -> str:
        """Return a one-shot text summary of the run.

        Returns:
            str: Episode counts, success rate, steps simulated and best
            hunt length.
        """
        best = self.stats.best_steps
        return (
            f"steps={self.forager.step_count} "
            f"episodes={self.stats.episodes} "
            f"success={self.stats.successes} "
            f"({self.stats.success_rate():.0%} rolling) "
            f"best_steps={best if best else '-'}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser for the foraging GUI.
    """
    p = argparse.ArgumentParser(
        description="Watch a fly learn to find food on the real fly brain."
    )
    p.add_argument(
        "--dataset", default="male-cns:v0.9",
        help="neuPrint dataset (default: %(default)s)",
    )
    p.add_argument(
        "--min-weight", type=int, default=50,
        help="minimum synapse weight for the full connectome (default: %(default)s)",
    )
    p.add_argument(
        "--mini", action="store_true",
        help="use the 41-neuron DNge104 mini brain instead of the full connectome",
    )
    p.add_argument(
        "--offline", action="store_true",
        help="use the cached real brain only (error if the cache is cold)",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="use the explicit synthetic test brain (no token needed)",
    )
    p.add_argument(
        "--speed", type=int, default=1,
        help="simulation steps per frame at startup (default: %(default)s)",
    )
    p.add_argument(
        "--eta", type=float, default=0.05,
        help="Hebbian learning rate (default: %(default)s)",
    )
    p.add_argument(
        "--invert-steer", action="store_true",
        help="flip the turn direction instead of detecting it",
    )
    p.add_argument(
        "--steer-sign", choices=["auto", "default"], default="auto",
        help="detect the turn direction (default: %(default)s)",
    )
    p.add_argument(
        "--seed", type=int, default=3, help="random seed (default: %(default)s)"
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="headless run of --smoke-frames frames, then exit (for CI)",
    )
    p.add_argument(
        "--smoke-frames", type=int, default=240,
        help="frames to run in --smoke mode (default: %(default)s)",
    )
    p.add_argument(
        "--save", default=None,
        help="save the final frame to this PNG path",
    )
    p.add_argument(
        "--duration", type=float, default=0.0,
        help="close the window automatically after N seconds (0 = never)",
    )
    return p


def main() -> None:
    """Run the foraging GUI.

    Raises:
        SystemExit: If the requested real brain cannot be loaded.
    """
    import matplotlib

    args = build_parser().parse_args()
    if args.smoke:
        matplotlib.use("Agg")

    net, banner, description = load_brain(args)
    print(banner)
    print(f"[gui] {description}: {len(net.ids):,} neurons, "
          f"{net.W.nnz:,} synapses")

    settle = SETTLE_ITERS_SMALL if (args.mini or args.synthetic) \
        else SETTLE_ITERS_FULL
    forager = FlyForager(
        net,
        settle_iters=settle,
        rng=np.random.default_rng(args.seed),
        invert_steer=args.invert_steer,
    )
    print(f"[gui] sensors={net.sensors} motors={net.motors} "
          f"calibration condition number={forager.cond:.2f}")
    if forager.cond > 50:
        print("[gui] warning: the sensor->motor matrix is poorly conditioned; "
              "steering may be weak. Try --mini, or a different dataset.")

    if args.steer_sign == "auto" and not args.invert_steer:
        invert = forager.auto_steer_sign() == -1
        forager.invert_steer = invert
        print(f"[gui] steering probe: turn direction "
              f"{'inverted' if invert else 'as wired'}")

    stats = TrainingStats()
    hebbian = MotorRewardHebbian(eta=args.eta)
    app = ForagingApp(
        forager, stats, hebbian, banner, description, speed=args.speed
    )

    if args.smoke:
        for _ in range(args.smoke_frames):
            app.tick()
        app.draw()
        _save_frame(app, args.save)
        print(f"[gui] {app.summary()}")
        return

    app.add_controls()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # The animation must stay referenced, or it is garbage collected before
    # it ever renders a frame.
    app.anim = FuncAnimation(
        app.fig, app.on_frame, interval=33, blit=True,
        cache_frame_data=False, repeat=False,
    )
    if args.duration > 0:
        _close_after(app.fig, args.duration)
    print("[gui] running -- press q or close the window to stop")
    plt.show()
    _save_frame(app, args.save)
    print(f"[gui] {app.summary()}")


def _save_frame(app: "ForagingApp", path: str | None) -> None:
    """Write the current figure to *path* when one was requested.

    Args:
        app (ForagingApp): Application whose figure is saved.
        path (str | None): Destination PNG path; ignored when ``None``.

    Returns:
        None: Saves the figure and reports the path.
    """
    if not path:
        return
    app.fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"[gui] saved {path}")


def _close_after(fig, seconds: float) -> None:
    """Close *fig* automatically once *seconds* have elapsed.

    Uses the canvas timer API, falling back to a repeating callback on
    older matplotlib where ``single_shot_add_callback`` does not exist.

    Args:
        fig (matplotlib.figure.Figure): Figure to close.
        seconds (float): Delay before closing.

    Returns:
        None: Starts a one-shot timer bound to the figure's canvas.
    """
    import matplotlib.pyplot as plt

    timer = fig.canvas.new_timer(interval=int(seconds * 1000))
    once = getattr(timer, "single_shot_add_callback", None)
    if once is not None:
        once(plt.close, fig)
    else:  # matplotlib < 3.9
        def _close() -> None:
            timer.stop()
            plt.close(fig)

        timer.add_callback(_close)
    fig._flynet_timer = timer  # keep a reference alive
    timer.start()


if __name__ == "__main__":
    main()
