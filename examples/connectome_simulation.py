#!/usr/bin/env python3
"""Connectome example: load the real fly brain and simulate.

Usage:
    # With neuPrint access (first fetch needs NEUPRINT_APPLICATION_CREDENTIALS):
    NEUPRINT_APPLICATION_CREDENTIALS=<token> python connectome_simulation.py

    # Offline: load the cached REAL brain only (never synthetic, never a
    # network request); exits with an error if the cache is empty:
    python connectome_simulation.py --offline
"""

import argparse

import numpy as np


def main() -> None:
    """Load a fly connectome (live, cached, or explicit offline) and simulate it.

    Raises:
        SystemExit: If ``--offline`` is set and no cached real brain
            exists (never falls back to synthetic data).
    """
    p = argparse.ArgumentParser(description="Load and simulate a fly connectome")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Load the cached real brain only (no network; error if not cached)",
    )
    p.add_argument("--full", action="store_true", help="Use whole connectome")
    p.add_argument("--dataset", default="male-cns:v1.0")
    p.add_argument("--neuron-type", default="DNge104")
    p.add_argument("--max-edges", type=int, default=40)
    p.add_argument("--settle", type=int, default=12)
    args = p.parse_args()

    from flynet.connectome import ConnectomeLoader
    from flynet.network import SpikingNetwork

    loader = ConnectomeLoader(dataset=args.dataset)

    if args.offline:
        if args.full:
            cached = loader.load_cached_full()
            if cached is None:
                raise SystemExit(
                    f"offline: no cached real connectome in "
                    f"{loader.cache_dir} (dataset {args.dataset!r}). "
                    "Run once with neuPrint access to populate the cache. "
                    "flynet never substitutes synthetic data here; synthetic "
                    "test data exists only behind the flynet CLI --offline flag."
                )
            ids, edges = cached
            motors = []
        else:
            cached = loader.load_cached_mini(args.neuron_type)
            if cached is None:
                raise SystemExit(
                    f"offline: no cached real brain for {args.neuron_type!r} "
                    f"in {loader.cache_dir} (dataset {args.dataset!r}). "
                    "Run once with neuPrint access to populate the cache. "
                    "flynet never substitutes synthetic data here; synthetic "
                    "test data exists only behind the flynet CLI --offline flag."
                )
            ids, edges, motors = cached
        print(f"[connectome] data source: {loader.last_source} ({args.dataset})")
    elif args.full:
        ids, edges, motors = loader.load_or_fetch_full()
        print(f"[connectome] data source: {loader.last_source} (full {args.dataset})")
    else:
        ids, edges, motors = loader.load_or_fetch_mini(
            neuron_type=args.neuron_type, max_edges=args.max_edges
        )
        print(f"[connectome] data source: {loader.last_source} ({args.dataset})")

    net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
    print(f"Loaded: {len(net.ids)} neurons, {net.W.nnz} synapses")
    print(f"Sensors: {net.sensors}")
    print(f"Motors:  {net.motors}")

    # Settle with symmetric input
    trace = net.settle(z_right=1.0, z_left=1.0, iterations=args.settle)
    print(
        f"\nSymmetric input - motor activities: "
        f"{trace[-1][net._ix[net.motors[0]]]:.4f}, "
        f"{trace[-1][net._ix[net.motors[1]]]:.4f}"
    )

    # Settle with asymmetric input (more on right)
    trace = net.settle(z_right=2.0, z_left=0.5, iterations=args.settle)
    print(
        f"Asymmetric input - motor activities: "
        f"{trace[-1][net._ix[net.motors[0]]]:.4f}, "
        f"{trace[-1][net._ix[net.motors[1]]]:.4f}"
    )

    # Show wiring queries
    if len(net.ids) <= 50:
        print("\n--- Wiring ---")
        for body in net.ids[:5]:
            pres = net.presynaptic(body)
            posts = net.postsynaptic(body)
            print(f"  {body}: {len(pres)} inputs, {len(posts)} outputs")


if __name__ == "__main__":
    main()
