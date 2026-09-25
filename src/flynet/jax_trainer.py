"""Gradient-based training for JAX-differentiable recurrent networks.

Implements DAgger-style training for navigation: a student network learns
to reproduce expert steering commands via backpropagation through time
(BPTT) with an Adam optimizer.

The FLYNN approach (arxiv 2607.00025) trains a differentiable recurrent
network against a teacher signal.  This module provides the training loop,
expert teacher, and evaluation utilities for that pipeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

__all__ = [
    "JaxNetwork",
    "ExpertTeacher",
    "GradientTrainingLog",
    "GradientTrainer",
    "RLTrainer",
]


# ---------------------------------------------------------------------------
# Protocol for the JAX network the trainer operates on
# ---------------------------------------------------------------------------

class JaxNetwork(Protocol):
    """Minimal interface a JAX network must satisfy for GradientTrainer.

    Any network implementing these methods can be trained with
    :class:`GradientTrainer`.  The network holds its own learnable
    parameters (e.g. JAX arrays) and exposes a differentiable forward
    pass.
    """

    # Structural members of the concrete flynet.jax_network.JaxNetwork
    # that the trainers rely on.
    n: int
    _ix: dict[int, int]
    sensors: list[int]
    motors: list[int]
    _sparsity_indices: Any
    _sparsity_shape: tuple[int, int]

    def forward(
        self,
        z_right: float | Any,
        z_left: float | Any,
        steps: int = 12,
    ) -> tuple[Any, Any]:
        """Run a differentiable forward pass (settle under sensory injection).

        Args:
            z_right: Right antenna injection value.
            z_left: Left antenna injection value.
            steps: Number of settling iterations.

        Returns:
            tuple: ``(motor_output, trace)`` where *motor_output* is a
            scalar (or 2-element vector) representing the motor command,
            and *trace* is the full activity trace through time.
        """

    def get_params(self) -> Any:
        """Return the current learnable parameters.

        Returns:
            Any: The parameter pytree (e.g. a parameter dict) currently
            held by the trainer.
        """

    def set_params(self, params: Any) -> None:
        """Replace the learnable parameters.

        Args:
            params (Any): New parameter pytree; must match the structure
                expected by the trainer's optimizer.
        """


# ---------------------------------------------------------------------------
# Expert teacher
# ---------------------------------------------------------------------------

class ExpertTeacher:
    """Compute expert steering commands for navigation.

    Uses a simple VFH-like controller: steer toward the food along the
    smell gradient.  The expert is *not* differentiable; it produces
    scalar labels that the student network learns to mimic.

    Attributes:
        step_len: Forward step length per time step.
        lookahead: Distance ahead used to decide forward vs. wander mode.
        antenna_offset: Lateral offset of each antenna from the body axis.
        steer_gain: Gain applied to the motor-difference signal.
        max_turn: Maximum turn angle per step.
        wander_step: Step length in wander mode.
        wander_turn: Maximum random turn in wander mode.
    """

    def __init__(
        self,
        step_len: float = 2.0,
        lookahead: float = 9.0,
        antenna_offset: float = 2.7,
        steer_gain: float = 6.0,
        max_turn: float = 0.45,
        wander_step: float = 0.5,
        wander_turn: float = 1.5,
    ) -> None:
        """Initialize the expert teacher.

        Args:
            step_len (float): Forward step length per time step.
            lookahead (float): Distance ahead used to decide forward vs.
                wander mode.
            antenna_offset (float): Lateral offset of each antenna from
                the body axis.
            steer_gain (float): Gain applied to the motor-difference
                signal.
            max_turn (float): Maximum turn angle per step.
            wander_step (float): Step length in wander mode.
            wander_turn (float): Maximum random turn in wander mode.
        """
        self.step_len = step_len
        self.lookahead = lookahead
        self.antenna_offset = antenna_offset
        self.steer_gain = steer_gain
        self.max_turn = max_turn
        self.wander_step = wander_step
        self.wander_turn = wander_turn

    @staticmethod
    def smell(pos: np.ndarray, food: np.ndarray) -> float:
        """Compute the smell intensity at *pos* given food at *food*.

        Uses the inverse-square model ``rho = 2 / (1 + 0.3 * d^2)``.

        Args:
            pos (np.ndarray): Position ``(x, y)``.
            food (np.ndarray): Food location ``(x, y)``.

        Returns:
            float: Smell intensity in ``[0, 2]``.
        """
        d = float(np.linalg.norm(food - pos))
        return 2.0 / (1.0 + 0.3 * d ** 2)

    def compute_steering(
        self,
        pos: np.ndarray,
        heading: float,
        food: np.ndarray,
        arena: float = 100.0,
    ) -> tuple[float, float, float, bool]:
        """Compute expert steering for the current state.

        Returns motor commands and the resulting sensor readings so the
        student network can be supervised.

        Args:
            pos (np.ndarray): Current position ``(x, y)``.
            heading (float): Current heading angle in radians.
            food (np.ndarray): Food location ``(x, y)``.
            arena (float): Arena boundary.

        Returns:
            tuple: ``(rho_right, rho_left, expert_steering, is_forward)``
            where *rho_right* and *rho_left* are the antenna smell
            readings, *expert_steering* is the ideal motor difference
            in ``[-1, 1]`` (negative = left, positive = right), and
            *is_forward* indicates whether the expert walks forward
            (True) or wanders (False).
        """
        d = np.array([np.cos(heading), np.sin(heading)])
        s = np.array([-d[1], d[0]])

        forward_smell = self.smell(pos + d * self.lookahead, food)
        backward_smell = self.smell(pos - d * self.lookahead, food)

        if forward_smell > backward_smell:
            rho_right = float(self.smell(
                pos + d * self.antenna_offset - s * self.antenna_offset, food
            ))
            rho_left = float(self.smell(
                pos + d * self.antenna_offset + s * self.antenna_offset, food
            ))
            expert_steering = float(np.clip(
                -self.steer_gain * (rho_right - rho_left),
                -self.max_turn,
                self.max_turn,
            ))
            return rho_right, rho_left, expert_steering, True
        else:
            return 0.0, 0.0, 0.0, False

    def step(
        self,
        pos: np.ndarray,
        heading: float,
        food: np.ndarray,
        rng: np.random.Generator,
        arena: float = 100.0,
    ) -> tuple[np.ndarray, float, float, float, float, bool]:
        """Advance the expert by one time step.

        Args:
            pos (np.ndarray): Current position ``(x, y)``.
            heading (float): Current heading angle in radians.
            food (np.ndarray): Food location ``(x, y)``.
            rng (np.random.Generator): RNG for wander noise.
            arena (float): Arena boundary.

        Returns:
            tuple: ``(new_pos, new_heading, rho_right, rho_left,
            expert_steering, is_forward)``.
        """
        rho_right, rho_left, expert_steering, is_forward = (
            self.compute_steering(pos, heading, food, arena)
        )

        if is_forward:
            new_heading = heading + expert_steering
            new_pos = pos + np.array([np.cos(heading), np.sin(heading)]) * self.step_len
        else:
            new_heading = heading + rng.uniform(-self.wander_turn, self.wander_turn)
            new_pos = pos + np.array([np.cos(heading), np.sin(heading)]) * self.wander_step

        new_pos = np.clip(new_pos, 0.0, arena)
        return new_pos, new_heading, rho_right, rho_left, expert_steering, is_forward


# ---------------------------------------------------------------------------
# Training logger (reuses the existing TrainingLogger pattern)
# ---------------------------------------------------------------------------

@dataclass
class GradientTrainingLog:
    """Log for gradient-based training runs.

    Attributes:
        episode_rewards: Total reward per episode (1.0 if food found, 0.0 otherwise).
        episode_steps: Number of steps taken per episode.
        episode_losses: Mean squared error per episode.
        episode_success_rate: Running success rate over a sliding window.
        wall_time: Cumulative wall-clock seconds per episode.
    """

    episode_rewards: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    episode_losses: list[float] = field(default_factory=list)
    episode_success_rate: list[float] = field(default_factory=list)
    wall_time: list[float] = field(default_factory=list)

    def log_episode(
        self,
        reward: float,
        steps: int,
        loss: float,
        window: int = 10,
    ) -> None:
        """Record one training episode.

        Args:
            reward (float): 1.0 if food found, 0.0 otherwise.
            steps (int): Number of steps taken.
            loss (float): Mean squared error for this episode.
            window (int): Sliding window for success-rate computation.
        """
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.episode_losses.append(loss)
        window = min(window, len(self.episode_rewards))
        self.episode_success_rate.append(
            sum(self.episode_rewards[-window:]) / window
        )

    def get_learning_curve(self) -> tuple[list[int], list[float]]:
        """Return episode numbers and success rate for plotting.

        Returns:
            tuple: ``(episode_numbers, success_rate)``.
        """
        n = len(self.episode_rewards)
        return list(range(1, n + 1)), list(self.episode_success_rate)

    def get_reward_curve(self) -> tuple[list[int], list[float]]:
        """Return episode numbers and rewards for plotting.

        Returns:
            tuple: ``(episode_numbers, rewards)``.
        """
        n = len(self.episode_rewards)
        return list(range(1, n + 1)), list(self.episode_rewards)

    def get_loss_curve(self) -> tuple[list[int], list[float]]:
        """Return episode numbers and mean loss for plotting.

        Returns:
            tuple: ``(episode_numbers, losses)``.
        """
        n = len(self.episode_losses)
        return list(range(1, n + 1)), list(self.episode_losses)


# ---------------------------------------------------------------------------
# Gradient trainer
# ---------------------------------------------------------------------------

class GradientTrainer:
    """Train a JaxNetwork on navigation tasks using gradient descent.

    Uses the FLYNN-style approach: a differentiable recurrent network
    trained with backpropagation through time against a teacher signal.
    The teacher (ExpertTeacher) computes ideal steering, and the student
    (the JaxNetwork) learns to reproduce it via MSE loss and Adam
    optimization.

    Attributes:
        network: The JaxNetwork being trained.
        teacher: The ExpertTeacher producing supervision.
        lr: Learning rate for the Adam optimizer.
        steps_per_turn: Settling iterations per forward pass.
    """

    def __init__(
        self,
        network: JaxNetwork,
        lr: float = 1e-3,
        steps_per_turn: int = 12,
        teacher: ExpertTeacher | None = None,
    ) -> None:
        """Initialize the gradient trainer.

        Args:
            network (JaxNetwork): A differentiable network instance.
            lr (float): Learning rate (Adam optimizer).
            steps_per_turn (int): Settling iterations per forward pass.
            teacher (ExpertTeacher | None): Expert teacher instance.
                A default :class:`ExpertTeacher` is created when ``None``.
        """
        self.network = network
        self.teacher = teacher or ExpertTeacher()
        self.lr = lr
        self.steps_per_turn = steps_per_turn

        # Lazily imported JAX objects (avoid hard import at module level)
        self._jax: Any = None
        self._optax: Any = None
        self._optimizer: Any = None
        self._opt_state: Any = None
        self._params: Any = None

        self._init_jax()

    # ------------------------------------------------------------------
    # JAX / optax bootstrap
    # ------------------------------------------------------------------

    def _init_jax(self) -> None:
        """Import JAX and optax, then build the Adam optimizer.

        Stashes references to the network's learnable parameters.

        Raises:
            ImportError: If ``jax`` or ``optax`` is not installed.
        """
        try:
            import jax
            import optax
        except ImportError as exc:
            raise ImportError(
                "jax and optax are required for gradient training.  "
                "Install them with: pip install flynet[gradient]"
            ) from exc

        self._jax = jax
        self._optax = optax

        self._params = self.network.get_params()
        self._optimizer = optax.adam(self.lr)
        self._opt_state = self._optimizer.init(self._params)

    # ------------------------------------------------------------------
    # Forward pass (differentiable, functional)
    # ------------------------------------------------------------------

    def _make_forward_fn(self, params: Any) -> Any:
        """Build a JIT-compiled forward function bound to *params*.

        Returns a callable ``(rho_right, rho_left) -> motor_output``
        that closes over the parameter arrays and can be traced by JAX.

        Args:
            params: Parameter dict (JAX arrays).

        Returns:
            Callable: Differentiable forward function.
        """
        jax = self._jax
        jnp = jax.numpy
        from jax.experimental.sparse import BCOO

        net = self.network
        n = net.n
        sL = net._ix[net.sensors[0]]
        sR = net._ix[net.sensors[1]]
        m0 = net._ix[net.motors[0]]
        m1 = net._ix[net.motors[1]]
        indices = net._sparsity_indices
        shape = net._sparsity_shape  # concrete Python tuple (n, n)
        steps = self.steps_per_turn
        W = params["W"]
        alpha = params["alpha"]
        b = params["b"]

        def forward(z_right: Any, z_left: Any) -> Any:
            """Run one differentiable steering episode step function.

            Args:
                z_right (Any): Right-antenna odor input (scalar or JAX
                    tracer).
                z_left (Any): Left-antenna odor input (scalar or JAX
                    tracer).

            Returns:
                Any: Steer command ``h[m0] - h[m1]`` for this input.
            """
            x = jnp.zeros(n, dtype=jnp.float32)
            x = x.at[sL].set(jnp.float32(z_left))
            x = x.at[sR].set(jnp.float32(z_right))
            h = jnp.zeros(n, dtype=jnp.float32)
            W_sparse = BCOO((W, indices), shape=shape)
            for _ in range(steps):
                h = alpha * h + (1.0 - alpha) * jnp.tanh(W_sparse @ h + x + b)
            return h[m0] - h[m1]

        return jax.jit(forward)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_episode(
        self,
        episode_idx: int,
        max_steps: int = 400,
        food_pos: tuple[float, float] = (80.0, 80.0),
        start_pos: tuple[float, float] = (15.0, 20.0),
        found_radius: float = 4.0,
        arena: float = 100.0,
        seed: int = 0,
    ) -> tuple[float, int, float]:
        """Run one training episode and update the network.

        Steps:
            1. Roll out the episode to get expert labels (numpy, no grad).
            2. Compute the total MSE loss weighted by a speed reward.
            3. Differentiate and update parameters with Adam.

        The loss is weighted by ``reward_weight = max_steps / steps_taken``
        so that fast episodes (few steps to reach food) contribute more to
        learning than slow ones.  Failed episodes (no food found) get the
        minimum weight of ``1.0``.

        Args:
            episode_idx (int): Episode number.
            max_steps (int): Maximum steps per episode.
            food_pos (tuple[float, float]): Food location ``(x, y)``.
            start_pos (tuple[float, float]): Starting location ``(x, y)``.
            found_radius (float): Distance at which food is "found".
            arena (float): Arena boundary.
            seed (int): Random seed for this episode.

        Returns:
            tuple: ``(loss, steps_taken, reward)``.
        """
        jax = self._jax
        rng = np.random.default_rng(seed)
        food = np.array(food_pos, dtype=np.float64)
        start = np.array(start_pos, dtype=np.float64)

        # Phase 1: Roll out the expert trajectory (no grad needed)
        pos = start.copy()
        heading = float(rng.uniform(-np.pi, np.pi))
        labels: list[tuple[float, float, float]] = []  # (rhoR, rhoL, expert_steer)
        found = False
        for step in range(max_steps):
            dist_to_food = float(np.linalg.norm(pos - food))
            if dist_to_food < found_radius:
                found = True
                break
            new_pos, new_heading, rho_right, rho_left, expert_steering, is_forward = (
                self.teacher.step(pos, heading, food, rng, arena)
            )
            if is_forward:
                labels.append((rho_right, rho_left, expert_steering))
            pos = new_pos
            heading = new_heading
        steps_taken = step + 1
        reward = 1.0 if found else 0.0

        if not labels:
            return 0.0, steps_taken, reward

        # Speed reward: faster episodes contribute more to the gradient
        reward_weight = max_steps / max(steps_taken, 1)

        # Phase 2: Define loss as a function of params (differentiable)
        rhoR_arr = jax.numpy.array([l[0] for l in labels], dtype=jax.numpy.float32)
        rhoL_arr = jax.numpy.array([l[1] for l in labels], dtype=jax.numpy.float32)
        steer_arr = jax.numpy.array([l[2] for l in labels], dtype=jax.numpy.float32)

        def loss_fn(params: Any) -> Any:
            """Compute the reward-weighted steering loss for *params*.

            Args:
                params (Any): Candidate network parameters consumed by
                    :meth:`_make_forward_fn`.

            Returns:
                Any: Scalar weighted mean squared steering error.
            """
            fwd = self._make_forward_fn(params)
            preds = jax.vmap(fwd)(rhoR_arr, rhoL_arr)
            return reward_weight * jax.numpy.mean((preds - steer_arr) ** 2)

        loss_val, grads = jax.value_and_grad(loss_fn)(self._params)
        updates, self._opt_state = self._optimizer.update(
            grads, self._opt_state, self._params,
        )
        self._params = self._optax.apply_updates(self._params, updates)
        self.network.set_params(self._params)

        return float(loss_val), steps_taken, reward

    def train_navigation(
        self,
        episodes: int = 50,
        max_steps: int = 400,
        food_pos: tuple[float, float] = (80.0, 80.0),
        start_pos: tuple[float, float] = (15.0, 20.0),
        found_radius: float = 4.0,
        arena: float = 100.0,
        print_every: int = 5,
        base_seed: int = 0,
    ) -> GradientTrainingLog:
        """Train the network on a navigation task.

        The task: navigate from *start_pos* toward *food_pos* using the
        smell gradient.  The expert teacher computes ideal steering at each
        step; the student network learns to reproduce it.

        Training signal: MSE between the network's motor output and the
        expert's steering command, accumulated over the full episode and
        differentiated via BPTT.

        Args:
            episodes (int): Number of training episodes.
            max_steps (int): Maximum steps per episode.
            food_pos (tuple[float, float]): Food location ``(x, y)``.
            start_pos (tuple[float, float]): Starting location ``(x, y)``.
            found_radius (float): Distance to consider food "found".
            arena (float): Arena size.
            print_every (int): Print stats every *N* episodes.
            base_seed (int): Base random seed.

        Returns:
            GradientTrainingLog: Training log with rewards, losses, and
            success rates.
        """
        log = GradientTrainingLog()
        t0 = time.time()

        for ep in range(1, episodes + 1):
            ep_t0 = time.time()
            loss, steps, reward = self.train_episode(
                episode_idx=ep,
                max_steps=max_steps,
                food_pos=food_pos,
                start_pos=start_pos,
                found_radius=found_radius,
                arena=arena,
                seed=base_seed + ep,
            )
            dt = time.time() - ep_t0
            log.log_episode(reward, steps, loss)
            log.wall_time.append(dt)

            if print_every and ep % print_every == 0:
                window = min(ep, 10)
                sr = log.episode_success_rate[-1]
                avg_loss = np.mean(log.episode_losses[-window:])
                avg_dt = np.mean(log.wall_time[-window:])
                print(
                    f"  Episode {ep:>4d}/{episodes} | "
                    f"loss {avg_loss:.4f} | "
                    f"reward {reward:.0f} | "
                    f"steps {steps:>4d} | "
                    f"success {sr:.0%} | "
                    f"{avg_dt:.2f}s/ep"
                )

        total = time.time() - t0
        print(
            f"\nTraining complete: {episodes} episodes in {total:.1f}s "
            f"({total / episodes:.2f}s/ep)"
        )
        final_sr = log.episode_success_rate[-1] if log.episode_success_rate else 0.0
        print(f"Final success rate (10-ep window): {final_sr:.0%}")

        return log

    # ------------------------------------------------------------------
    # Evaluation (no gradients)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        episodes: int = 20,
        max_steps: int = 400,
        food_pos: tuple[float, float] = (80.0, 80.0),
        start_pos: tuple[float, float] = (15.0, 20.0),
        found_radius: float = 4.0,
        arena: float = 100.0,
        base_seed: int = 1000,
    ) -> dict[str, float]:
        """Evaluate the trained network without gradient updates.

        The student's own motor output drives the heading; the teacher
        supplies only the antenna readings, the forward/wander mode, and
        the expert label used to compute the imitation loss.  Success
        rate therefore reflects the trained network, not the expert.

        Args:
            episodes (int): Number of evaluation episodes.
            max_steps (int): Maximum steps per episode.
            food_pos (tuple[float, float]): Food location ``(x, y)``.
            start_pos (tuple[float, float]): Starting location ``(x, y)``.
            found_radius (float): Distance to consider food "found".
            arena (float): Arena boundary.
            base_seed (int): Base random seed for evaluation.

        Returns:
            dict: Summary statistics with keys ``"success_rate"``,
            ``"avg_steps_to_food"``, ``"avg_loss"``.
        """
        food = np.array(food_pos, dtype=np.float64)
        start = np.array(start_pos, dtype=np.float64)

        # Build a forward function from current params (no grad needed)
        forward_fn = self._make_forward_fn(self._params)

        successes = 0
        found_steps: list[int] = []
        losses: list[float] = []

        for ep in range(episodes):
            rng = np.random.default_rng(base_seed + ep)
            pos = start.copy()
            heading = float(rng.uniform(-np.pi, np.pi))
            ep_loss = 0.0
            n_forward = 0

            for step in range(max_steps):
                dist = float(np.linalg.norm(pos - food))
                if dist < found_radius:
                    successes += 1
                    found_steps.append(step + 1)
                    break

                rho_right, rho_left, expert_steering, is_forward = (
                    self.teacher.compute_steering(pos, heading, food, arena)
                )

                if is_forward:
                    # Student's own action drives the environment
                    mu = float(forward_fn(rho_right, rho_left))
                    ep_loss += (mu - expert_steering) ** 2
                    n_forward += 1
                    new_heading = heading + mu
                    new_pos = pos + np.array([
                        np.cos(heading), np.sin(heading),
                    ]) * self.teacher.step_len
                else:
                    new_heading = heading + rng.uniform(
                        -self.teacher.wander_turn, self.teacher.wander_turn,
                    )
                    new_pos = pos + np.array([
                        np.cos(heading), np.sin(heading),
                    ]) * self.teacher.wander_step

                new_pos = np.clip(new_pos, 0.0, arena)
                pos = new_pos
                heading = new_heading
            else:
                losses.append(ep_loss / max(n_forward, 1))
                continue

            losses.append(ep_loss / max(n_forward, 1))

        return {
            "success_rate": successes / episodes,
            "avg_steps_to_food": (
                float(np.mean(found_steps)) if found_steps else float(max_steps)
            ),
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save trained parameters and training config to disk.

        Saves parameters as a NumPy ``.npz`` archive with a JSON
        sidecar containing metadata.

        Args:
            path (str | Path): Destination path (directory or file).
                If a directory, ``params.npz`` and ``config.json`` are
                created inside it.
        """
        path = Path(path)
        if path.is_dir():
            param_path = path / "params.npz"
            config_path = path / "config.json"
        else:
            path.mkdir(parents=True, exist_ok=True)
            param_path = path / "params.npz"
            config_path = path / "config.json"

        # Flatten params to a dict of numpy arrays
        params = self._params
        np_params: dict[str, Any]
        if isinstance(params, dict):
            np_params = {
                k: np.asarray(v) for k, v in params.items()
            }
        elif isinstance(params, (list, tuple)):
            np_params = {
                f"param_{i}": np.asarray(p) for i, p in enumerate(params)
            }
        else:
            np_params = {"param_0": np.asarray(params)}

        np.savez_compressed(str(param_path), **np_params)

        config = {
            "lr": self.lr,
            "steps_per_turn": self.steps_per_turn,
            "teacher": {
                "step_len": self.teacher.step_len,
                "lookahead": self.teacher.lookahead,
                "antenna_offset": self.teacher.antenna_offset,
                "steer_gain": self.teacher.steer_gain,
                "max_turn": self.teacher.max_turn,
                "wander_step": self.teacher.wander_step,
                "wander_turn": self.teacher.wander_turn,
            },
        }
        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)

        print(f"Saved parameters to {param_path}")

    def load(self, path: str | Path) -> None:
        """Load trained parameters and config from disk.

        Args:
            path (str | Path): Source path (directory or file).
                Expects ``params.npz`` and optionally ``config.json``.
        """
        path = Path(path)
        if path.is_dir():
            param_path = path / "params.npz"
            config_path = path / "config.json"
        else:
            param_path = path / "params.npz"
            config_path = path.parent / "config.json"

        data = np.load(str(param_path))
        loaded = dict(data.files)

        # Reconstruct params in the same structure the network expects
        current = self._params
        restored: Any
        if isinstance(current, dict):
            restored = {
                k: self._jax.numpy.array(loaded[k])
                for k in current
                if k in loaded
            }
            # Fill in any missing keys from current params
            for k in current:
                if k not in restored:
                    restored[k] = current[k]
        elif isinstance(current, (list, tuple)):
            restored = type(current)(
                self._jax.numpy.array(loaded[f"param_{i}"])
                if f"param_{i}" in loaded
                else p
                for i, p in enumerate(current)
            )
        else:
            key = "param_0" if "param_0" in loaded else list(loaded.keys())[0]
            restored = self._jax.numpy.array(loaded[key])

        self._params = restored
        self.network.set_params(restored)

        if config_path.exists():
            with open(config_path) as fh:
                config = json.load(fh)
            self.lr = config.get("lr", self.lr)
            self.steps_per_turn = config.get("steps_per_turn", self.steps_per_turn)
            teacher_cfg = config.get("teacher", {})
            if teacher_cfg:
                self.teacher.step_len = teacher_cfg.get(
                    "step_len", self.teacher.step_len
                )
                self.teacher.lookahead = teacher_cfg.get(
                    "lookahead", self.teacher.lookahead
                )
                self.teacher.antenna_offset = teacher_cfg.get(
                    "antenna_offset", self.teacher.antenna_offset
                )
                self.teacher.steer_gain = teacher_cfg.get(
                    "steer_gain", self.teacher.steer_gain
                )
                self.teacher.max_turn = teacher_cfg.get(
                    "max_turn", self.teacher.max_turn
                )
                self.teacher.wander_step = teacher_cfg.get(
                    "wander_step", self.teacher.wander_step
                )
                self.teacher.wander_turn = teacher_cfg.get(
                    "wander_turn", self.teacher.wander_turn
                )

        print(f"Loaded parameters from {param_path}")


# ---------------------------------------------------------------------------
# RL trainer (REINFORCE policy gradient)
# ---------------------------------------------------------------------------

class RLTrainer:
    """Train a JaxNetwork using REINFORCE policy gradient.

    Unlike GradientTrainer (DAgger), the agent explores on its own
    without an expert teacher driving the environment.  It learns from
    its own successes and failures via policy gradient: actions that
    lead to faster food-finding are reinforced, slow ones are suppressed.

    The policy is a Gaussian over the network's deterministic output::

        mu    = network(rhoR, rhoL)        # deterministic motor diff
        sigma = exp(log_sigma)             # learnable noise std
        a     = mu + sigma * epsilon       # stochastic action

    The REINFORCE update maximises ``E[R * sum(log_prob)]`` where ``R``
    is a speed reward (``max_steps / steps_taken``).

    Attributes:
        network: The JaxNetwork being trained.
        teacher: ExpertTeacher for environment interaction.
        lr: Learning rate for Adam optimizer.
        steps_per_turn: Settling iterations per forward pass.
        baseline: Running average of returns for variance reduction.
    """

    def __init__(
        self,
        network: JaxNetwork,
        lr: float = 1e-5,
        sigma_init: float = 0.1,
        steps_per_turn: int = 12,
        teacher: ExpertTeacher | None = None,
    ) -> None:
        """Initialize the RL trainer.

        Args:
            network (JaxNetwork): A differentiable network instance.
            lr (float): Learning rate (Adam optimizer).
            sigma_init (float): Initial action noise std (learnable).
            steps_per_turn (int): Settling iterations per forward pass.
            teacher (ExpertTeacher | None): Expert teacher for environment
                interaction (smell, position dynamics).  A default
                :class:`ExpertTeacher` is created when ``None``.
        """
        self.network = network
        self.teacher = teacher or ExpertTeacher()
        self.lr = lr
        self.steps_per_turn = steps_per_turn

        # Lazily imported JAX objects (avoid hard import at module level)
        self._jax: Any = None
        self._optax: Any = None
        self._optimizer: Any = None
        self._opt_state: Any = None
        self._params: Any = None
        self._log_sigma: Any = None
        self._full_params: Any = None
        self._baseline: float = 1.0  # Running mean of returns

        self._init_jax(sigma_init)

    # ------------------------------------------------------------------
    # JAX / optax bootstrap
    # ------------------------------------------------------------------

    def _init_jax(self, sigma_init: float) -> None:
        """Import JAX and optax, then build the Adam optimizer.

        Stashes references to the network's learnable parameters and the
        learnable log-sigma.

        Args:
            sigma_init (float): Initial action noise standard deviation.

        Raises:
            ImportError: If ``jax`` or ``optax`` is not installed.
        """
        try:
            import jax
            import optax
        except ImportError as exc:
            raise ImportError(
                "jax and optax are required for RL training.  "
                "Install them with: pip install flynet[gradient]"
            ) from exc

        self._jax = jax
        self._optax = optax

        self._params = self.network.get_params()
        self._log_sigma = jax.numpy.log(jax.numpy.float32(sigma_init))

        # Combine network params and log_sigma for joint optimisation
        self._full_params = {
            "net": self._params,
            "log_sigma": self._log_sigma,
        }
        self._optimizer = optax.adam(self.lr)
        self._opt_state = self._optimizer.init(self._full_params)

    # ------------------------------------------------------------------
    # Forward pass (differentiable, functional) -- same as GradientTrainer
    # ------------------------------------------------------------------

    def _make_forward_fn(self, params: Any) -> Any:
        """Build a JIT-compiled forward function bound to *params*.

        Returns a callable ``(rho_right, rho_left) -> motor_output``
        that closes over the parameter arrays and can be traced by JAX.

        Args:
            params: Parameter dict (JAX arrays).

        Returns:
            Callable: Differentiable forward function.
        """
        jax = self._jax
        jnp = jax.numpy
        from jax.experimental.sparse import BCOO

        net = self.network
        n = net.n
        sL = net._ix[net.sensors[0]]
        sR = net._ix[net.sensors[1]]
        m0 = net._ix[net.motors[0]]
        m1 = net._ix[net.motors[1]]
        indices = net._sparsity_indices
        shape = net._sparsity_shape
        steps = self.steps_per_turn
        W = params["W"]
        alpha = params["alpha"]
        b = params["b"]

        def forward(z_right: Any, z_left: Any) -> Any:
            """Run one differentiable steering episode step function.

            Args:
                z_right (Any): Right-antenna odor input (scalar or JAX
                    tracer).
                z_left (Any): Left-antenna odor input (scalar or JAX
                    tracer).

            Returns:
                Any: Steer command ``h[m0] - h[m1]`` for this input.
            """
            x = jnp.zeros(n, dtype=jnp.float32)
            x = x.at[sL].set(jnp.float32(z_left))
            x = x.at[sR].set(jnp.float32(z_right))
            h = jnp.zeros(n, dtype=jnp.float32)
            W_sparse = BCOO((W, indices), shape=shape)
            for _ in range(steps):
                h = alpha * h + (1.0 - alpha) * jnp.tanh(W_sparse @ h + x + b)
            return h[m0] - h[m1]

        return jax.jit(forward)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_episode(
        self,
        episode_idx: int,
        max_steps: int = 400,
        food_pos: tuple[float, float] = (80.0, 80.0),
        start_pos: tuple[float, float] = (15.0, 20.0),
        found_radius: float = 4.0,
        arena: float = 100.0,
        seed: int = 0,
    ) -> tuple[float, int, float]:
        """Run one RL training episode and update via REINFORCE.

        Steps:
            1. Roll out the episode with Gaussian exploration noise.
               The agent's own actions drive the environment (not the
               expert).  Record ``(rhoR, rhoL, action)`` at each step.
            2. Compute speed reward ``R = max_steps / steps_taken``.
            3. Define loss ``= -(R - baseline) * sum(log_prob)`` as a
               differentiable function of network params and log_sigma.
            4. Differentiate and update with Adam.

        Args:
            episode_idx (int): Episode number.
            max_steps (int): Maximum steps per episode.
            food_pos (tuple[float, float]): Food location ``(x, y)``.
            start_pos (tuple[float, float]): Starting location ``(x, y)``.
            found_radius (float): Distance at which food is "found".
            arena (float): Arena boundary.
            seed (int): Random seed for this episode.

        Returns:
            tuple: ``(loss, steps_taken, reward)``.
        """
        jax = self._jax
        rng = np.random.default_rng(seed)
        food = np.array(food_pos, dtype=np.float64)
        start = np.array(start_pos, dtype=np.float64)

        # Build forward function and read current sigma
        forward_fn = self._make_forward_fn(self._params)
        sigma = float(jax.numpy.exp(self._log_sigma))

        # JAX random key for reproducible exploration noise
        key = jax.random.PRNGKey(seed)

        # Phase 1: Roll out episode with exploration noise (no grad)
        pos = start.copy()
        heading = float(rng.uniform(-np.pi, np.pi))
        steps_data: list[tuple[float, float, float]] = []  # (rhoR, rhoL, action)
        found = False

        for step in range(max_steps):
            dist_to_food = float(np.linalg.norm(pos - food))
            if dist_to_food < found_radius:
                found = True
                break

            # Get sensor readings and navigation mode from teacher
            rho_right, rho_left, _, is_forward = self.teacher.compute_steering(
                pos, heading, food, arena,
            )

            if is_forward:
                # Deterministic network output
                mu = float(forward_fn(rho_right, rho_left))

                # Sample stochastic action
                key, subkey = jax.random.split(key)
                epsilon = float(jax.random.normal(subkey))
                action = mu + sigma * epsilon

                steps_data.append((rho_right, rho_left, action))

                # Agent's action drives the environment
                new_heading = heading + action
                new_pos = pos + np.array([
                    np.cos(heading), np.sin(heading),
                ]) * self.teacher.step_len
            else:
                # Wander mode (same as expert)
                new_heading = heading + rng.uniform(
                    -self.teacher.wander_turn, self.teacher.wander_turn,
                )
                new_pos = pos + np.array([
                    np.cos(heading), np.sin(heading),
                ]) * self.teacher.wander_step

            new_pos = np.clip(new_pos, 0.0, arena)
            pos = new_pos
            heading = new_heading

        steps_taken = step + 1
        reward = 1.0 if found else 0.0

        # No forward steps -- nothing to learn from
        if not steps_data:
            return 0.0, steps_taken, reward

        # Speed reward: faster episodes get higher weight
        R = max_steps / max(steps_taken, 1)

        # Update baseline (running mean of returns)
        self._baseline = 0.9 * self._baseline + 0.1 * R
        advantage = R - self._baseline

        # Phase 2: REINFORCE loss (differentiable w.r.t. params + log_sigma)
        rhoR_arr = jax.numpy.array(
            [d[0] for d in steps_data], dtype=jax.numpy.float32,
        )
        rhoL_arr = jax.numpy.array(
            [d[1] for d in steps_data], dtype=jax.numpy.float32,
        )
        action_arr = jax.numpy.array(
            [d[2] for d in steps_data], dtype=jax.numpy.float32,
        )
        advantage_val = jax.numpy.float32(advantage)

        def loss_fn(full_params: dict[str, Any]) -> Any:
            """Compute the REINFORCE loss for the full parameter set.

            Args:
                full_params (dict[str, Any]): Dict with ``'net'``
                    (network parameters) and ``'log_sigma'`` (action
                    noise, log standard deviation).

            Returns:
                Any: Scalar loss ``-advantage * sum(log_prob)``;
                differentiable w.r.t. ``full_params``.
            """
            net_params = full_params["net"]
            log_sigma = full_params["log_sigma"]
            sigma = jax.numpy.exp(log_sigma)

            fwd = self._make_forward_fn(net_params)

            # Recompute mu from params (differentiable)
            mus = jax.vmap(fwd)(rhoR_arr, rhoL_arr)

            # Log probability of each recorded action
            log_probs = (
                -0.5 * ((action_arr - mus) / sigma) ** 2
                - log_sigma
                - 0.5 * jax.numpy.log(2.0 * jax.numpy.pi)
            )

            # REINFORCE: maximise advantage * sum(log_prob)
            return -advantage_val * jax.numpy.sum(log_probs)

        loss_val, grads = jax.value_and_grad(loss_fn)(self._full_params)

        # Clip gradients to prevent large updates on high-dim params
        def clip_grad(g: Any) -> Any:
            """Clip a gradient tensor element-wise to ``[-1, 1]``.

            Args:
                g (Any): A gradient leaf; arrays are clipped, other
                    pytree leaves are returned unchanged.

            Returns:
                Any: Clipped gradient leaf.
            """
            if isinstance(g, (jax.numpy.ndarray, jax.Array)):
                return jax.numpy.clip(g, -1.0, 1.0)
            return g
        grads = jax.tree.map(clip_grad, grads)

        updates, self._opt_state = self._optimizer.update(
            grads, self._opt_state, self._full_params,
        )
        self._full_params = self._optax.apply_updates(self._full_params, updates)
        self._params = self._full_params["net"]
        self._log_sigma = self._full_params["log_sigma"]
        self.network.set_params(self._params)

        return float(loss_val), steps_taken, reward

    def train_navigation(
        self,
        episodes: int = 50,
        max_steps: int = 400,
        food_pos: tuple[float, float] = (80.0, 80.0),
        start_pos: tuple[float, float] = (15.0, 20.0),
        found_radius: float = 4.0,
        arena: float = 100.0,
        print_every: int = 5,
        base_seed: int = 0,
    ) -> GradientTrainingLog:
        """Train the network on a navigation task using REINFORCE.

        The task: navigate from *start_pos* toward *food_pos* using the
        smell gradient.  The agent explores on its own and learns from
        its own successes and failures via policy gradient.

        Args:
            episodes (int): Number of training episodes.
            max_steps (int): Maximum steps per episode.
            food_pos (tuple[float, float]): Food location ``(x, y)``.
            start_pos (tuple[float, float]): Starting location ``(x, y)``.
            found_radius (float): Distance to consider food "found".
            arena (float): Arena size.
            print_every (int): Print stats every *N* episodes.
            base_seed (int): Base random seed.

        Returns:
            GradientTrainingLog: Training log with rewards, losses, and
            success rates.
        """
        log = GradientTrainingLog()
        t0 = time.time()

        for ep in range(1, episodes + 1):
            ep_t0 = time.time()
            loss, steps, reward = self.train_episode(
                episode_idx=ep,
                max_steps=max_steps,
                food_pos=food_pos,
                start_pos=start_pos,
                found_radius=found_radius,
                arena=arena,
                seed=base_seed + ep,
            )
            dt = time.time() - ep_t0
            log.log_episode(reward, steps, loss)
            log.wall_time.append(dt)

            if print_every and ep % print_every == 0:
                window = min(ep, 10)
                sr = log.episode_success_rate[-1]
                avg_loss = np.mean(log.episode_losses[-window:])
                avg_dt = np.mean(log.wall_time[-window:])
                sigma = float(self._jax.numpy.exp(self._log_sigma))
                print(
                    f"  Episode {ep:>4d}/{episodes} | "
                    f"loss {avg_loss:.4f} | "
                    f"reward {reward:.0f} | "
                    f"steps {steps:>4d} | "
                    f"success {sr:.0%} | "
                    f"sigma {sigma:.3f} | "
                    f"{avg_dt:.2f}s/ep"
                )

        total = time.time() - t0
        print(
            f"\nTraining complete: {episodes} episodes in {total:.1f}s "
            f"({total / episodes:.2f}s/ep)"
        )
        final_sr = log.episode_success_rate[-1] if log.episode_success_rate else 0.0
        print(f"Final success rate (10-ep window): {final_sr:.0%}")

        return log

    # ------------------------------------------------------------------
    # Evaluation (no gradients, no noise)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        episodes: int = 20,
        max_steps: int = 400,
        food_pos: tuple[float, float] = (80.0, 80.0),
        start_pos: tuple[float, float] = (15.0, 20.0),
        found_radius: float = 4.0,
        arena: float = 100.0,
        base_seed: int = 1000,
    ) -> dict[str, float]:
        """Evaluate the trained network without gradient updates or noise.

        Uses the agent's deterministic policy (mu only, no exploration
        noise) and lets the agent drive the environment.

        Args:
            episodes (int): Number of evaluation episodes.
            max_steps (int): Maximum steps per episode.
            food_pos (tuple[float, float]): Food location ``(x, y)``.
            start_pos (tuple[float, float]): Starting location ``(x, y)``.
            found_radius (float): Distance to consider food "found".
            arena (float): Arena size.
            base_seed (int): Base random seed for evaluation.

        Returns:
            dict: Summary statistics with keys ``"success_rate"``,
            ``"avg_steps_to_food"``.
        """
        rng_np = np.random.default_rng(base_seed)
        food = np.array(food_pos, dtype=np.float64)
        start = np.array(start_pos, dtype=np.float64)

        # Build a forward function from current params (no grad needed)
        forward_fn = self._make_forward_fn(self._params)

        successes = 0
        found_steps: list[int] = []

        for ep in range(episodes):
            rng = np.random.default_rng(base_seed + ep)
            pos = start.copy()
            heading = float(rng_np.uniform(-np.pi, np.pi))

            for step in range(max_steps):
                dist = float(np.linalg.norm(pos - food))
                if dist < found_radius:
                    successes += 1
                    found_steps.append(step + 1)
                    break

                rho_right, rho_left, _, is_forward = (
                    self.teacher.compute_steering(pos, heading, food, arena)
                )

                if is_forward:
                    # Deterministic action (no noise)
                    mu = float(forward_fn(rho_right, rho_left))
                    new_heading = heading + mu
                    new_pos = pos + np.array([
                        np.cos(heading), np.sin(heading),
                    ]) * self.teacher.step_len
                else:
                    new_heading = heading + rng.uniform(
                        -self.teacher.wander_turn, self.teacher.wander_turn,
                    )
                    new_pos = pos + np.array([
                        np.cos(heading), np.sin(heading),
                    ]) * self.teacher.wander_step

                new_pos = np.clip(new_pos, 0.0, arena)
                pos = new_pos
                heading = new_heading

        return {
            "success_rate": successes / episodes,
            "avg_steps_to_food": (
                float(np.mean(found_steps)) if found_steps else float(max_steps)
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save trained parameters (network + log_sigma) to disk.

        Saves parameters as a NumPy ``.npz`` archive with a JSON
        sidecar containing metadata.

        Args:
            path (str | Path): Destination path (directory or file).
                If a directory, ``params.npz`` and ``config.json`` are
                created inside it.
        """
        path = Path(path)
        if path.is_dir():
            param_path = path / "params.npz"
            config_path = path / "config.json"
        else:
            path.mkdir(parents=True, exist_ok=True)
            param_path = path / "params.npz"
            config_path = path / "config.json"

        # Flatten params to a dict of numpy arrays
        params = self._params
        np_params: dict[str, Any]
        if isinstance(params, dict):
            np_params = {k: np.asarray(v) for k, v in params.items()}
        elif isinstance(params, (list, tuple)):
            np_params = {
                f"param_{i}": np.asarray(p) for i, p in enumerate(params)
            }
        else:
            np_params = {"param_0": np.asarray(params)}

        # Include log_sigma
        np_params["log_sigma"] = np.asarray(self._log_sigma)

        np.savez_compressed(str(param_path), **np_params)

        config = {
            "lr": self.lr,
            "steps_per_turn": self.steps_per_turn,
            "sigma_init": float(self._jax.numpy.exp(self._log_sigma)),
            "baseline": self._baseline,
            "teacher": {
                "step_len": self.teacher.step_len,
                "lookahead": self.teacher.lookahead,
                "antenna_offset": self.teacher.antenna_offset,
                "steer_gain": self.teacher.steer_gain,
                "max_turn": self.teacher.max_turn,
                "wander_step": self.teacher.wander_step,
                "wander_turn": self.teacher.wander_turn,
            },
        }
        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)

        print(f"Saved parameters to {param_path}")

    def load(self, path: str | Path) -> None:
        """Load trained parameters and config from disk.

        Args:
            path (str | Path): Source path (directory or file).
                Expects ``params.npz`` and optionally ``config.json``.
        """
        path = Path(path)
        if path.is_dir():
            param_path = path / "params.npz"
            config_path = path / "config.json"
        else:
            param_path = path / "params.npz"
            config_path = path.parent / "config.json"

        data = np.load(str(param_path))
        loaded = dict(data.files)

        # Reconstruct params in the same structure the network expects
        current = self._params
        restored: Any
        if isinstance(current, dict):
            restored = {
                k: self._jax.numpy.array(loaded[k])
                for k in current
                if k in loaded
            }
            for k in current:
                if k not in restored:
                    restored[k] = current[k]
        elif isinstance(current, (list, tuple)):
            restored = type(current)(
                self._jax.numpy.array(loaded[f"param_{i}"])
                if f"param_{i}" in loaded
                else p
                for i, p in enumerate(current)
            )
        else:
            key = "param_0" if "param_0" in loaded else list(loaded.keys())[0]
            restored = self._jax.numpy.array(loaded[key])

        self._params = restored
        self.network.set_params(self._params)

        # Restore log_sigma
        if "log_sigma" in loaded:
            self._log_sigma = self._jax.numpy.array(loaded["log_sigma"])

        # Rebuild combined params for the optimizer
        self._full_params = {"net": self._params, "log_sigma": self._log_sigma}

        if config_path.exists():
            with open(config_path) as fh:
                config = json.load(fh)
            self.lr = config.get("lr", self.lr)
            self.steps_per_turn = config.get("steps_per_turn", self.steps_per_turn)
            self._baseline = config.get("baseline", self._baseline)
            teacher_cfg = config.get("teacher", {})
            if teacher_cfg:
                self.teacher.step_len = teacher_cfg.get(
                    "step_len", self.teacher.step_len
                )
                self.teacher.lookahead = teacher_cfg.get(
                    "lookahead", self.teacher.lookahead
                )
                self.teacher.antenna_offset = teacher_cfg.get(
                    "antenna_offset", self.teacher.antenna_offset
                )
                self.teacher.steer_gain = teacher_cfg.get(
                    "steer_gain", self.teacher.steer_gain
                )
                self.teacher.max_turn = teacher_cfg.get(
                    "max_turn", self.teacher.max_turn
                )
                self.teacher.wander_step = teacher_cfg.get(
                    "wander_step", self.teacher.wander_step
                )
                self.teacher.wander_turn = teacher_cfg.get(
                    "wander_turn", self.teacher.wander_turn
                )

        print(f"Loaded parameters from {param_path}")
