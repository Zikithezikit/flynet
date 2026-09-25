#!/usr/bin/env python3
"""Classification example: unsupervised STDP learning on simple patterns.

Demonstrates STDP training without neuPrint -- builds a small spiking
network that learns to recognize hand-coded binary patterns.
"""

import numpy as np
from flynet import SpikingNeuron, SynapseList, STDPRule, LateralInhibition


def make_pattern(
    size: int,
    density: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a random binary pattern.

    Args:
        size (int): Number of bits in the pattern.
        density (float): Probability that any given bit is on.
        rng (np.random.Generator | None): Random generator; a fresh one
            is created when ``None``.

    Returns:
        np.ndarray: Float array of 0.0/1.0 values of shape ``(size,)``.
    """
    rng = rng or np.random.default_rng()
    return (rng.random(size) < density).astype(float)


def pattern_to_spikes(pattern: np.ndarray, t_steps: int = 150) -> np.ndarray:
    """Convert a binary pattern to a spike train via rate coding.

    Args:
        pattern (np.ndarray): Float 0/1 pattern of shape ``(n,)``.
        t_steps (int): Number of encoding timesteps.

    Returns:
        np.ndarray: Integer spike train of shape ``(n, t_steps + 1)``.
    """
    n = len(pattern)
    train = np.zeros((n, t_steps + 1), dtype=int)
    for i in range(n):
        if pattern[i] > 0:
            rate = pattern[i]
            for t in range(t_steps + 1):
                if np.random.random() < rate * 0.3:
                    train[i, t] = 1
    return train


def main() -> None:
    """Run the STDP classification demo (flags via ``--help``)."""
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
    print(
        f"\nTraining: {args.epochs} epochs, {args.n_neurons} neurons, "
        f"{args.n_inputs} inputs"
    )
    for epoch in range(args.epochs):
        winners = []
        for pat_idx, pat in enumerate(patterns):
            train = pattern_to_spikes(pat, t_steps=args.t_steps)
            neurons = [
                SpikingNeuron(
                    threshold=5.0, rest=0.0, min_pot=-500, leak=0.15, refrac_time=30
                )
                for _ in range(args.n_neurons)
            ]
            potentials = np.zeros(args.n_neurons)
            fired = np.zeros(args.n_neurons, dtype=bool)
            winner = -1

            for t in range(args.t_steps + 1):
                # 1. Integrate input (count down refractory timers)
                for j in range(args.n_neurons):
                    if neurons[j]._refrac_counter > 0:
                        neurons[j]._refrac_counter -= 1
                        continue
                    potentials[j] += np.dot(synapses.get_row(j), train[:, t])
                    if potentials[j] > 0:
                        potentials[j] -= 0.15

                # 2. Detect spikes (before inhibition, so `fired`
                #    reflects this timestep)
                fired[:] = False
                for j in range(args.n_neurons):
                    if neurons[j]._refrac_counter <= 0 and potentials[j] >= 5.0:
                        fired[j] = True

                # 3. WTA inhibition: if anyone fired, keep only the
                #    strongest and suppress the rest for this step
                if fired.any():
                    mask = inhibition.apply(potentials, fired, threshold=5.0)

                    for j in range(args.n_neurons):
                        if not mask[j]:
                            potentials[j] = -500

                    # 4. Reset the winner, start its refractory period,
                    #    and apply STDP
                    for j in np.flatnonzero(mask):
                        winner = j
                        potentials[j] = 0
                        neurons[j]._refrac_counter = 30
                        neurons[j]._potential = 0

                        for h in range(args.n_inputs):
                            if train[h, t] == 1:
                                w = synapses.get_row(j)[h]
                                # dt = t_post - t_pre: inputs fired before
                                # this spike (offset < 0) -> potentiation
                                for offset in range(-2, -21, -1):
                                    t_in = t + offset
                                    if (
                                        0 <= t_in < args.t_steps + 1
                                        and train[h, t_in] == 1
                                    ):
                                        w = stdp.update_weight(w, -offset)
                                # inputs fired after this spike -> depression
                                for offset in range(2, 21):
                                    t_in = t + offset
                                    if (
                                        0 <= t_in < args.t_steps + 1
                                        and train[h, t_in] == 1
                                    ):
                                        w = stdp.update_weight(w, -offset)
                                synapses.set_weight(j, h, w)

        winners.append(winner)

    # Show learned weights
    print("\nLearned weight patterns:")
    for i in range(args.n_neurons):
        w = synapses.get_row(i)
        active = (w > 0.3).sum()
        print(
            f"  Neuron {i}: {active} strong connections, "
            f"max={w.max():.3f}, mean={w.mean():.3f}"
        )


if __name__ == "__main__":
    main()
