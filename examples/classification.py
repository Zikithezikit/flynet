#!/usr/bin/env python3
"""Classification example: unsupervised STDP learning on simple patterns.

Demonstrates STDP training without neuPrint -- builds a small spiking
network that learns to recognize hand-coded binary patterns.
"""

import numpy as np
from flynet import SpikingNeuron, SynapseList, STDPRule, LateralInhibition


def make_pattern(size, density=0.3, rng=None):
    """Generate a random binary pattern."""
    rng = rng or np.random.default_rng()
    return (rng.random(size) < density).astype(float)


def pattern_to_spikes(pattern, t_steps=150):
    """Convert a binary pattern to a spike train via rate coding."""
    n = len(pattern)
    train = np.zeros((n, t_steps + 1), dtype=int)
    for i in range(n):
        if pattern[i] > 0:
            rate = pattern[i]
            for t in range(t_steps + 1):
                if np.random.random() < rate * 0.3:
                    train[i, t] = 1
    return train


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-inputs", type=int, default=16)
    p.add_argument("--n-neurons", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n-patterns", type=int, default=3)
    p.add_argument("--t-steps", type=int, default=150)
    args = p.parse_args()

    rng = np.random.default_rng(42)

    # Create patterns (16x16 binary images)
    patterns = []
    for i in range(args.n_patterns):
        p = make_pattern(args.n_inputs, density=0.3 + 0.1 * i, rng=rng)
        patterns.append(p)
        print(f"Pattern {i}: {p.sum():.0f} active pixels")

    # Initialize synapses
    synapses = SynapseList(args.n_neurons, args.n_inputs)
    for i in range(args.n_neurons):
        for j in range(args.n_inputs):
            synapses.set_weight(i, j, rng.uniform(0, 0.75))

    stdp = STDPRule()
    inhibition = LateralInhibition(mode="wta")

    # Training
    print(f"\nTraining: {args.epochs} epochs, {args.n_neurons} neurons, "
          f"{args.n_inputs} inputs")
    for epoch in range(args.epochs):
        winners = []
        for pat_idx, pat in enumerate(patterns):
            train = pattern_to_spikes(pat, t_steps=args.t_steps)
            neurons = [SpikingNeuron(threshold=5.0, rest=0.0, min_pot=-500,
                                     leak=0.15, refrac_time=30)
                       for _ in range(args.n_neurons)]
            potentials = np.zeros(args.n_neurons)
            fired = np.zeros(args.n_neurons, dtype=bool)
            winner = -1

            for t in range(args.t_steps + 1):
                for j in range(args.n_neurons):
                    if neurons[j]._refrac_counter <= 0:
                        potentials[j] += np.dot(synapses.get_row(j), train[:, t])
                        if potentials[j] > 0:
                            potentials[j] -= 0.15

                # WTA inhibition
                mask = inhibition.apply(potentials, fired, threshold=5.0)
                for j in range(args.n_neurons):
                    if not mask[j]:
                        potentials[j] = -500

                # Check spikes
                for j in range(args.n_neurons):
                    if neurons[j]._refrac_counter <= 0 and potentials[j] >= 5.0:
                        fired[j] = True
                        winner = j
                        neurons[j]._refrac_counter = 30
                        neurons[j]._potential = 0

                        # STDP update
                        for h in range(args.n_inputs):
                            if train[h, t] == 1:
                                w = synapses.get_row(j)[h]
                                for dt in range(-2, -21, -1):
                                    if 0 <= t + dt < args.t_steps + 1:
                                        if train[h, t + dt] == 1:
                                            w = stdp.update_weight(w, dt)
                                for dt in range(2, 21):
                                    if 0 <= t + dt < args.t_steps + 1:
                                        if train[h, t + dt] == 1:
                                            w = stdp.update_weight(w, dt)
                                synapses.set_weight(j, h, w)

                # Reset for next timestep
                fired[:] = False

        winners.append(winner)

    # Show learned weights
    print("\nLearned weight patterns:")
    for i in range(args.n_neurons):
        w = synapses.get_row(i)
        active = (w > 0.3).sum()
        print(f"  Neuron {i}: {active} strong connections, "
              f"max={w.max():.3f}, mean={w.mean():.3f}")


if __name__ == "__main__":
    main()
