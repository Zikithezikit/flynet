"""Command-line interface for flynet."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``flynet`` CLI.

    Parses command-line arguments and dispatches to the appropriate
    subcommand handler.

    Args:
        argv (list[str] | None): Command-line arguments. Uses ``sys.argv`` if ``None``.
    """
    p = argparse.ArgumentParser(
        prog="flynet",
        description="Spiking neural network library based on the Drosophila brain connectome.",
    )
    sub = p.add_subparsers(dest="command")

    # -- flynet simulate ----------------------------------------------------
    sim = sub.add_parser("simulate", help="Run a simulation with the real or synthetic connectome")
    sim.add_argument("--dataset", default="male-cns:v1.0",
                     help="Connectome dataset (default: %(default)s)")
    sim.add_argument("--neuron-type", default="DNge104",
                     help="Cell type for mini brain (default: %(default)s)")
    sim.add_argument("--full", action="store_true",
                     help="Use the whole connectome (~176k neurons)")
    sim.add_argument("--max-edges", type=int, default=40,
                     help="Strongest synapses kept for mini brain (default: %(default)s)")
    sim.add_argument("--min-weight", type=int, default=50,
                     help="Min synapse weight for full connectome (default: %(default)s)")
    sim.add_argument("--settle", type=int, default=12,
                     help="Settling iterations (default: %(default)s)")
    sim.add_argument("--outdir", default="output",
                     help="Output directory (default: %(default)s)")
    sim.add_argument("--offline", action="store_true",
                     help="Use cached or synthetic data only")
    sim.add_argument("--seed", type=int, default=3,
                     help="Random seed (default: %(default)s)")
    sim.add_argument("--plot", action="store_true",
                     help="Show brain circuit plot")

    # -- flynet train -------------------------------------------------------
    tr = sub.add_parser("train", help="Train the brain with reward-gated Hebbian plasticity")
    tr.add_argument("--dataset", default="male-cns:v1.0")
    tr.add_argument("--neuron-type", default="DNge104")
    tr.add_argument("--full", action="store_true")
    tr.add_argument("--episodes", type=int, default=2, help="Training episodes (default: %(default)s)")
    tr.add_argument("--eta", type=float, default=0.6, help="Learning rate (default: %(default)s)")
    tr.add_argument("--reward-bonus", type=float, default=6.0, help="Reward bonus (default: %(default)s)")
    tr.add_argument("--max-steps", type=int, default=400, help="Max steps per game (default: %(default)s)")
    tr.add_argument("--eval-seeds", type=int, default=50, help="Eval games per checkpoint (default: %(default)s)")
    tr.add_argument("--seed", type=int, default=3)
    tr.add_argument("--outdir", default="output")
    tr.add_argument("--offline", action="store_true")
    tr.add_argument("--resume", nargs="?", const="__auto__", default=None)
    tr.add_argument("--animate", action="store_true")

    # -- flynet list-datasets -----------------------------------------------
    sub.add_parser("list-datasets", help="List available connectome datasets")

    args = p.parse_args(argv)

    if args.command is None:
        p.print_help()
        return

    if args.command == "list-datasets":
        _list_datasets()
    elif args.command == "simulate":
        _run_simulate(args)
    elif args.command == "train":
        _run_train(args)


def _list_datasets() -> None:
    """Print available connectome datasets.

    Loads the dataset registry from ``ConnectomeLoader`` and prints each
    dataset name and its description.
    """
    from flynet.connectome import ConnectomeLoader
    datasets = ConnectomeLoader.DATASETS
    print(f"{'Dataset':<30} Description")
    print("-" * 80)
    for name, desc in datasets.items():
        print(f"{name:<30} {desc}")


def _run_simulate(args: argparse.Namespace) -> None:
    """Run a single simulation.

    Builds a spiking network from the selected connectome dataset,
    settles it with a random odor input, and optionally saves a brain
    circuit plot.

    Args:
        args (argparse.Namespace): Parsed command-line arguments for the ``simulate`` subcommand.
    """
    import numpy as np
    from flynet.connectome import ConnectomeLoader
    from flynet.network import SpikingNetwork

    loader = ConnectomeLoader(dataset=args.dataset)

    if args.full:
        ids, edges, motors = loader.load_or_fetch_full(min_weight=args.min_weight)
    else:
        ids, edges, motors = loader.load_or_fetch_mini(
            neuron_type=args.neuron_type, max_edges=args.max_edges)

    net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
    print(f"Network: {len(net.ids)} neurons, {net.W.nnz} synapses")
    print(f"Sensors: {net.sensors}")
    print(f"Motors:  {net.motors}")

    # Quick test: settle with random smell
    rng = np.random.default_rng(args.seed)
    zr = rng.uniform(0, 2)
    zl = rng.uniform(0, 2)
    trace = net.settle(zr, zl, iterations=args.settle)
    m_activity = trace[-1][[net._ix[b] for b in net.motors]]
    print(f"Motor activity: {m_activity}")

    if args.plot:
        from flynet.visualize import plot_brain_circuit
        import matplotlib
        matplotlib.use("Agg")
        plot_brain_circuit(net, title=f"Brain Circuit ({args.dataset})")
        import os
        os.makedirs(args.outdir, exist_ok=True)
        import matplotlib.pyplot as plt
        plt.savefig(f"{args.outdir}/brain_circuit.png", dpi=130, bbox_inches="tight")
        print(f"Saved {args.outdir}/brain_circuit.png")


def _run_train(args: argparse.Namespace) -> None:
    """Run the training loop.

    Trains the spiking network using reward-gated Hebbian plasticity
    across multiple food-search episodes. The fly navigates toward
    an odor source and synapses are updated after successful foraging.

    Args:
        args (argparse.Namespace): Parsed command-line arguments for the ``train`` subcommand.
    """
    import numpy as np
    from flynet.connectome import ConnectomeLoader
    from flynet.network import SpikingNetwork
    from flynet.learning import RewardHebbian, TrainingLogger
    from flynet.visualize import plot_learning_curve, plot_plasticity_heatmap

    loader = ConnectomeLoader(dataset=args.dataset)

    if args.full:
        ids, edges, motors = loader.load_or_fetch_full()
    else:
        ids, edges, motors = loader.load_or_fetch_mini(
            neuron_type=args.neuron_type)

    net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
    print(f"Network: {len(net.ids)} neurons, {net.W.nnz} synapses")

    # Simplified training loop (for the full game loop, use the examples)
    hebbian = RewardHebbian(eta=args.eta, reward_bonus=args.reward_bonus)
    logger = TrainingLogger()
    W_start = net.W.copy()

    FOOD = np.array([80.0, 80.0])
    START = np.array([15.0, 20.0])
    LOOKAHEAD = 9.0
    ANTENNA_OFFSET = 2.7
    STEER_GAIN = 6.0
    MAX_TURN = 0.45
    STEP_LEN = 2.0
    FOUND_RADIUS = 4.0
    ARENA = 100.0

    def smell(pos: np.ndarray) -> float:
        d = float(np.linalg.norm(FOOD - pos))
        return 2.0 / (1.0 + 0.3 * d**2)

    rng = np.random.default_rng(args.seed)

    for ep in range(1, args.episodes + 1):
        pos = START.astype(float).copy()
        geo = float(rng.uniform(-np.pi, np.pi))
        path = [pos.copy()]
        events = []

        for step in range(args.max_steps):
            if np.linalg.norm(pos - FOOD) < FOUND_RADIUS:
                break
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
                geo += rng.uniform(-1.5, 1.5)
                nxt = pos + d * 0.5
            nxt = np.clip(nxt, 0.0, ARENA)
            pos = nxt
            path.append(pos.copy())

        found = np.linalg.norm(pos - FOOD) < FOUND_RADIUS
        logger.log_episode(1.0 if found else 0.0, len(path))

        if found and events:
            hebbian.update(net, events)

        print(f"Episode {ep}/{args.episodes}: {'FOUND' if found else 'FAILED'} "
              f"in {len(path)} steps")

    print(f"\nTraining complete. Success rate: {logger.episode_rewards.count(1.0)}/{args.episodes}")


if __name__ == "__main__":
    main()
