"""Command-line interface for flynet."""

from __future__ import annotations

import argparse
import sys

from flynet.connectome import ConnectomeUnavailableError

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``flynet`` CLI.

    Parses command-line arguments and dispatches to the appropriate
    subcommand handler.

    Args:
        argv (list[str] | None): Command-line arguments. Uses ``sys.argv`` if ``None``.

    Raises:
        SystemExit: With status 1 when real connectome data cannot be
            loaded (``ConnectomeUnavailableError``); never falls back to
            synthetic data silently.
    """
    p = argparse.ArgumentParser(
        prog="flynet",
        description="Spiking neural network library based on the Drosophila brain connectome.",
    )
    sub = p.add_subparsers(dest="command")

    # -- flynet simulate ----------------------------------------------------
    sim = sub.add_parser("simulate", help="Run a simulation with the real or synthetic connectome")
    sim.add_argument("--dataset", default="male-cns:v0.9",
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
    tr = sub.add_parser("train", help="Train the brain (Hebbian or gradient-based)")
    tr.add_argument("--dataset", default="male-cns:v0.9")
    tr.add_argument("--neuron-type", default="DNge104")
    tr.add_argument("--full", action="store_true")
    tr.add_argument("--episodes", type=int, default=2, help="Training episodes (default: %(default)s)")
    tr.add_argument("--method", choices=["hebbian", "gradient", "rl"], default="hebbian",
                     help="Training method (default: hebbian)")
    tr.add_argument("--lr", type=float, default=1e-3,
                     help="Learning rate for gradient method (default: %(default)s)")
    tr.add_argument("--rl-lr", type=float, default=1e-5,
                     help="Learning rate for the RL method (default: %(default)s)")
    tr.add_argument("--eta", type=float, default=0.6, help="Hebbian learning rate (default: %(default)s)")
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

    try:
        if args.command == "list-datasets":
            _list_datasets()
        elif args.command == "simulate":
            _run_simulate(args)
        elif args.command == "train":
            _run_train(args)
    except ConnectomeUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


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

    if args.offline:
        print("[connectome] --offline: using SYNTHETIC brain (not real data)")
        ids, edges, motors = ConnectomeLoader.synthetic_brain(
            max_edges=args.max_edges)
    elif args.full:
        ids, edges, motors = loader.load_or_fetch_full(min_weight=args.min_weight)
        print(f"[connectome] data source: {loader.last_source} "
              f"(full {args.dataset})")
    else:
        ids, edges, motors = loader.load_or_fetch_mini(
            neuron_type=args.neuron_type, max_edges=args.max_edges)
        print(f"[connectome] data source: {loader.last_source} "
              f"({args.dataset})")

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

    Dispatches to Hebbian or gradient-based training depending on
    ``args.method``.

    Args:
        args (argparse.Namespace): Parsed command-line arguments for the ``train`` subcommand.
    """
    if args.method == "gradient":
        _run_train_gradient(args)
    elif args.method == "rl":
        _run_train_rl(args)
    else:
        _run_train_hebbian(args)


def _run_train_gradient(args: argparse.Namespace) -> None:
    """Train using gradient-based (FLYNN-style) backpropagation.

    Builds a JaxNetwork from the connectome, wraps it in a
    GradientTrainer, and runs DAgger-style training with an expert
    teacher.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    import numpy as np
    from flynet.connectome import ConnectomeLoader
    from flynet.network import SpikingNetwork
    from flynet.jax_network import JaxNetwork
    from flynet.jax_trainer import GradientTrainer

    loader = ConnectomeLoader(dataset=args.dataset)

    if args.offline:
        print("[connectome] --offline: using SYNTHETIC brain (not real data)")
        ids, edges, motors = ConnectomeLoader.synthetic_brain()
    elif args.full:
        ids, edges, motors = loader.load_or_fetch_full()
        print(f"[connectome] data source: {loader.last_source} "
              f"(full {args.dataset})")
    else:
        ids, edges, motors = loader.load_or_fetch_mini(
            neuron_type=args.neuron_type)
        print(f"[connectome] data source: {loader.last_source} "
              f"({args.dataset})")

    print("Building connectome network...")
    spike_net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
    print(f"  {len(spike_net.ids)} neurons, {spike_net.W.nnz} synapses")
    print(f"  Sensors: {spike_net.sensors}")
    print(f"  Motors:  {spike_net.motors}")

    print("Converting to differentiable JAX network...")
    jax_net = JaxNetwork.from_spiking_network(spike_net)
    print(f"  {jax_net.n} neurons, {jax_net._sparsity_indices.shape[0]} trainable edges")

    trainer = GradientTrainer(jax_net, lr=args.lr, steps_per_turn=12)

    print(f"\nGradient training ({args.method}): {args.episodes} episodes, lr={args.lr}")
    print("-" * 60)

    log = trainer.train_navigation(
        episodes=args.episodes,
        max_steps=args.max_steps,
        print_every=1,
        base_seed=args.seed,
    )

    # Final evaluation
    print("\nEvaluating...")
    stats = trainer.evaluate(
        episodes=args.eval_seeds, max_steps=args.max_steps,
        base_seed=args.seed + 9999,
    )
    print(f"  Success rate: {stats['success_rate']:.0%}")
    print(f"  Avg steps to food: {stats['avg_steps_to_food']:.1f}")

    # Save
    import os
    os.makedirs(args.outdir, exist_ok=True)
    trainer.save(args.outdir)


def _run_train_rl(args: argparse.Namespace) -> None:
    """Train using reinforcement learning.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    from flynet.connectome import ConnectomeLoader
    from flynet.network import SpikingNetwork
    from flynet.jax_network import JaxNetwork
    from flynet.jax_trainer import RLTrainer

    loader = ConnectomeLoader(dataset=args.dataset)

    if args.offline:
        print("[connectome] --offline: using SYNTHETIC brain (not real data)")
        ids, edges, motors = ConnectomeLoader.synthetic_brain()
    elif args.full:
        ids, edges, motors = loader.load_or_fetch_full()
        print(f"[connectome] data source: {loader.last_source} "
              f"(full {args.dataset})")
    else:
        ids, edges, motors = loader.load_or_fetch_mini(neuron_type=args.neuron_type)
        print(f"[connectome] data source: {loader.last_source} "
              f"({args.dataset})")

    spike_net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
    jax_net = JaxNetwork.from_spiking_network(spike_net)

    trainer = RLTrainer(jax_net, lr=args.rl_lr)

    # Resume from pre-trained checkpoint (e.g. after --method gradient)
    if args.resume:
        import os
        resume_path = args.resume if args.resume != "__auto__" else args.outdir
        if os.path.exists(os.path.join(resume_path, "params.npz")):
            print(f"Loading pre-trained weights from {resume_path}...")
            # Load network params from GradientTrainer checkpoint
            data = np.load(os.path.join(resume_path, "params.npz"))
            net_params = {k: data[k] for k in ("W", "alpha", "b") if k in data}
            if net_params:
                trainer.network.set_params({k: jnp.array(v) for k, v in net_params.items()})
                trainer._params = trainer.network.get_params()
                trainer._full_params = {"net": trainer._params, "log_sigma": trainer._log_sigma}
                trainer._opt_state = trainer._optimizer.init(trainer._full_params)
                print(f"  Loaded W, alpha, b from checkpoint")

    log = trainer.train_navigation(episodes=args.episodes, max_steps=args.max_steps,
                                    print_every=1, base_seed=args.seed)

    stats = trainer.evaluate(
        episodes=args.eval_seeds, max_steps=args.max_steps,
        base_seed=args.seed + 9999,
    )
    print(f"  Success rate: {stats['success_rate']:.0%}")
    print(f"  Avg steps to food: {stats['avg_steps_to_food']:.1f}")

    import os
    os.makedirs(args.outdir, exist_ok=True)
    trainer.save(args.outdir)


def _run_train_hebbian(args: argparse.Namespace) -> None:
    """Train using reward-gated Hebbian plasticity.

    This is the original training method, kept for backward compatibility.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    import numpy as np
    from flynet.connectome import ConnectomeLoader
    from flynet.network import SpikingNetwork
    from flynet.learning import RewardHebbian, TrainingLogger
    from flynet.visualize import plot_learning_curve, plot_plasticity_heatmap

    loader = ConnectomeLoader(dataset=args.dataset)

    if args.offline:
        print("[connectome] --offline: using SYNTHETIC brain (not real data)")
        ids, edges, motors = ConnectomeLoader.synthetic_brain()
    elif args.full:
        ids, edges, motors = loader.load_or_fetch_full()
        print(f"[connectome] data source: {loader.last_source} "
              f"(full {args.dataset})")
    else:
        ids, edges, motors = loader.load_or_fetch_mini(
            neuron_type=args.neuron_type)
        print(f"[connectome] data source: {loader.last_source} "
              f"({args.dataset})")

    net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
    print(f"Network: {len(net.ids)} neurons, {net.W.nnz} synapses")

    # Simplified training loop (for the full game loop, use the examples)
    hebbian = RewardHebbian(eta=args.eta, reward_bonus=args.reward_bonus)
    logger = TrainingLogger()
    W_start = net.W.copy()

    # Calibrate ONCE before training so steering stays consistent
    # as Hebbian updates modify the weights.
    _, Minv = net.calibrate()

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
        """Compute the odor concentration at a position.

        Args:
            pos (np.ndarray): Current ``(x, y)`` position in the arena.

        Returns:
            float: Odor strength; 1.0 at the food source, decaying with
            squared distance.
        """
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
                drv, tr = net.turn(rhoR, rhoL, Minv=Minv)
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
