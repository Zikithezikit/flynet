#!/usr/bin/env python3
"""Navigation task: fly navigates to food using the real brain wiring.

Simplified version of working-brain/main.py using the flynet library.
"""

import numpy as np
from flynet.connectome import ConnectomeLoader
from flynet.network import SpikingNetwork
from flynet.learning import RewardHebbian, TrainingLogger

# Arena parameters
ARENA = 100.0
FOOD = np.array([80.0, 80.0])
START = np.array([15.0, 20.0])
FOUND_RADIUS = 4.0
MAX_STEPS = 400
LOOKAHEAD = 9.0
ANTENNA_OFFSET = 2.7
STEER_GAIN = 6.0
MAX_TURN = 0.45
STEP_LEN = 2.0
WANDER_STEP = 0.5
WANDER_TURN = 1.5


def smell(pos: np.ndarray) -> float:
    """Compute the odor concentration at a position.

    Args:
        pos (np.ndarray): Current ``(x, y)`` position in the arena.

    Returns:
        float: Odor strength; 1.0 at the food source, decaying with
        squared distance.
    """
    d = float(np.linalg.norm(FOOD - pos))
    return 2.0 / (1.0 + 0.3 * d**2)


def play_game(
    net: SpikingNetwork,
    rng: np.random.Generator,
    max_steps: int = MAX_STEPS,
) -> tuple[int | None, list[np.ndarray], list[tuple[float, float, np.ndarray]], bool]:
    """Run one navigation episode in the arena.

    Args:
        net (SpikingNetwork): Network used for steering decisions.
        rng (np.random.Generator): Random generator for wander noise.
        max_steps (int): Episode step limit.

    Returns:
        tuple: ``(steps, path, events, found)`` where *steps* is the step
        count when food was reached (``None`` otherwise), *path* is the
        list of visited positions, *events* holds per-step
        ``(rho_right, rho_left, activity)`` tuples, and *found* reports
        whether the food was reached.
    """
    pos = START.astype(float).copy()
    geo = float(rng.uniform(-np.pi, np.pi))
    path = [pos.copy()]
    events = []

    for step in range(max_steps):
        if np.linalg.norm(pos - FOOD) < FOUND_RADIUS:
            return step, path, events, True

        d = np.array([np.cos(geo), np.sin(geo)])
        s = np.array([-d[1], d[0]])

        if smell(pos + d * LOOKAHEAD) > smell(pos - d * LOOKAHEAD):
            rhoL = smell(pos + d * ANTENNA_OFFSET + s * ANTENNA_OFFSET)
            rhoR = smell(pos + d * ANTENNA_OFFSET - s * ANTENNA_OFFSET)
            drv, tr = net.turn(rhoR, rhoL)
            steering = np.clip(-STEER_GAIN * drv, -MAX_TURN, MAX_TURN)
            geo += steering
            nxt = pos + d * STEP_LEN
            a = np.mean(np.array(tr[4:]), axis=0)
            events.append((rhoR, rhoL, a))
        else:
            geo += rng.uniform(-WANDER_TURN, WANDER_TURN)
            nxt = pos + d * WANDER_STEP

        nxt = np.clip(nxt, 0.0, ARENA)
        pos = nxt
        path.append(pos.copy())

    return None, path, events, False


def main() -> None:
    """Train the fly to navigate to food (hebbian, gradient, or RL).

    Raises:
        SystemExit: If ``--offline`` is set and no cached real brain
            exists (never falls back to synthetic data).
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--method", choices=["hebbian", "gradient", "rl"], default="hebbian")
    p.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for gradient method"
    )
    p.add_argument("--eta", type=float, default=0.6)
    p.add_argument(
        "--offline",
        action="store_true",
        help="Load the cached real brain only (no network; error if not cached)",
    )
    p.add_argument(
        "--full", action="store_true", help="Use whole connectome (~176k neurons)"
    )
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--eval-seeds", type=int, default=20)
    args = p.parse_args()

    loader = ConnectomeLoader()
    if args.offline:
        if args.full:
            cached = loader.load_cached_full()
            if cached is None:
                raise SystemExit(
                    f"offline: no cached real connectome in {loader.cache_dir}. "
                    "Run once with neuPrint access to populate the cache. "
                    "flynet never substitutes synthetic data here; synthetic "
                    "test data exists only behind the flynet CLI --offline flag."
                )
            ids, edges = cached
            motors = []
        else:
            cached = loader.load_cached_mini("DNge104")
            if cached is None:
                raise SystemExit(
                    "offline: no cached real brain for 'DNge104' in "
                    f"{loader.cache_dir}. Run once with neuPrint access to "
                    "populate the cache. flynet never substitutes synthetic "
                    "data here; synthetic test data exists only behind the "
                    "flynet CLI --offline flag."
                )
            ids, edges, motors = cached
        print(f"[connectome] data source: {loader.last_source}")
    elif args.full:
        ids, edges, motors = loader.load_or_fetch_full()
        print(f"[connectome] data source: {loader.last_source} (full)")
    else:
        ids, edges, motors = loader.load_or_fetch_mini(neuron_type="DNge104")
        print(f"[connectome] data source: {loader.last_source}")
    net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)

    print(f"Brain: {len(net.ids)} neurons, {net.W.nnz} synapses")

    if args.method == "gradient":
        from flynet.jax_network import JaxNetwork
        from flynet.jax_trainer import GradientTrainer

        jax_net = JaxNetwork.from_spiking_network(net)
        trainer = GradientTrainer(jax_net, lr=args.lr)
        log = trainer.train_navigation(
            episodes=args.episodes,
            base_seed=args.seed,
            print_every=1,
        )
        stats = trainer.evaluate(episodes=args.eval_seeds, base_seed=args.seed + 9999)
        print(f"\nEvaluation: {stats['success_rate']:.0%} success")
        print(f"Average steps: {stats['avg_steps_to_food']:.1f}")
    elif args.method == "rl":
        from flynet.jax_network import JaxNetwork
        from flynet.jax_trainer import RLTrainer

        jax_net = JaxNetwork.from_spiking_network(net)
        trainer = RLTrainer(jax_net, lr=args.lr)
        log = trainer.train_navigation(
            episodes=args.episodes,
            base_seed=args.seed,
            print_every=1,
        )
        stats = trainer.evaluate(episodes=args.eval_seeds, base_seed=args.seed + 9999)
        print(f"\nEvaluation: {stats['success_rate']:.0%} success")
        print(f"Average steps: {stats['avg_steps_to_food']:.1f}")
    else:
        hebbian = RewardHebbian(eta=args.eta)
        logger = TrainingLogger()

        for ep in range(1, args.episodes + 1):
            rng = np.random.default_rng(args.seed + ep)
            steps, path, events, found = play_game(net, rng)
            logger.log_episode(1.0 if found else 0.0, steps or MAX_STEPS)

            if found and events:
                hebbian.update(net, events)

            status = f"FOUND in {steps}" if found else f"FAILED ({MAX_STEPS} steps)"
            print(f"  Episode {ep}: {status}")

        # Evaluate
        success = 0
        total_steps = []
        for s in range(args.eval_seeds):
            rng = np.random.default_rng(1000 + s)
            steps, _, _, found = play_game(net, rng)
            if found:
                success += 1
                total_steps.append(steps)

        print(f"\nEvaluation: {success}/{args.eval_seeds} success")
        if total_steps:
            print(f"Average steps: {np.mean(total_steps):.1f}")


if __name__ == "__main__":
    main()
