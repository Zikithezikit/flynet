#!/usr/bin/env python3
"""Connectome example: load the real fly brain and simulate.

Usage:
    # With neuPrint access:
    NEUPRINT_APPLICATION_CREDENTIALS=<token> python connectome_simulation.py

    # Offline (uses cache or synthetic fallback):
    python connectome_simulation.py --offline
"""

import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Load and simulate a fly connectome")
    p.add_argument("--offline", action="store_true", help="Use cache/synthetic only")
    p.add_argument("--full", action="store_true", help="Use whole connectome")
    p.add_argument("--dataset", default="male-cns:v1.0")
    p.add_argument("--neuron-type", default="DNge104")
    p.add_argument("--max-edges", type=int, default=40)
    p.add_argument("--settle", type=int, default=12)
    args = p.parse_args()

    from flynet.connectome import ConnectomeLoader
    from flynet.network import SpikingNetwork

    loader = ConnectomeLoader(dataset=args.dataset)

    if args.full:
        ids, edges, motors = loader.load_or_fetch_full()
    else:
        ids, edges, motors = loader.load_or_fetch_mini(
            neuron_type=args.neuron_type, max_edges=args.max_edges)

    net = SpikingNetwork(ids, edges, motor_ids=motors if motors else None)
    print(f"Loaded: {len(net.ids)} neurons, {net.W.nnz} synapses")
    print(f"Sensors: {net.sensors}")
    print(f"Motors:  {net.motors}")

    # Settle with symmetric input
    trace = net.settle(z_right=1.0, z_left=1.0, iterations=args.settle)
    print(f"\nSymmetric input - motor activities: "
          f"{trace[-1][net._ix[net.motors[0]]]:.4f}, "
          f"{trace[-1][net._ix[net.motors[1]]]:.4f}")

    # Settle with asymmetric input (more on right)
    trace = net.settle(z_right=2.0, z_left=0.5, iterations=args.settle)
    print(f"Asymmetric input - motor activities: "
          f"{trace[-1][net._ix[net.motors[0]]]:.4f}, "
          f"{trace[-1][net._ix[net.motors[1]]]:.4f}")

    # Show wiring queries
    if len(net.ids) <= 50:
        print("\n--- Wiring ---")
        for body in net.ids[:5]:
            pres = net.presynaptic(body)
            posts = net.postsynaptic(body)
            print(f"  {body}: {len(pres)} inputs, {len(posts)} outputs")


if __name__ == "__main__":
    main()
