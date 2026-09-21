#!/usr/bin/env python3
"""Basic example: create a small spiking network and simulate it."""

import numpy as np
from flynet import SpikingNetwork, LIFNeuron, rate_encode, STDPRule


def main():
    # -- 1. Build a small network from edges --------------------------------
    ids = [100, 101, 102, 103, 104]
    edges = [
        (100, 102, 3.0),  # sensor 0 -> neuron A
        (101, 103, 3.0),  # sensor 1 -> neuron B
        (102, 104, 2.0),  # neuron A -> motor
        (103, 104, 2.0),  # neuron B -> motor
        (102, 103, 0.5),  # recurrent A -> B
        (103, 102, 0.5),  # recurrent B -> A
    ]

    net = SpikingNetwork(ids, edges, sensor_ids=[100, 101], motor_ids=[104])
    print(f"Network: {len(net.ids)} neurons, {net.W.nnz} synapses")
    print(f"Sensors: {net.sensors}")
    print(f"Motors:  {net.motors}")

    # -- 2. Settle under sensory injection ----------------------------------
    trace = net.settle(z_right=1.5, z_left=0.5, iterations=10)
    final = trace[-1]
    for i, body in enumerate(net.ids):
        print(f"  Neuron {body}: activity = {final[i]:.4f}")

    # -- 3. Calibrate and steer ---------------------------------------------
    M, Minv = net.calibrate()
    print(f"\nSensor->Motor response matrix:\n{M}")
    drv, _ = net.turn(rho_right=2.0, rho_left=1.0)
    print(f"Steering command: {drv:.4f}")

    # -- 4. LIF neuron demo ------------------------------------------------
    print("\n--- LIF Neuron Demo ---")
    neuron = LIFNeuron()
    spikes = 0
    for t in range(100):
        I = 2.0 if 20 <= t <= 60 else 0.0  # pulse of current
        if neuron.step(I, dt=1.0):
            spikes += 1
            print(f"  Spike at t={t}, V={neuron.get_potential():.2f}")
    print(f"Total spikes: {spikes}")

    # -- 5. STDP demo ------------------------------------------------------
    print("\n--- STDP Demo ---")
    stdp = STDPRule()
    w = 0.5
    print(f"Initial weight: {w}")
    w = stdp.update_weight(w, dt=-3)  # pre before post -> potentiate
    print(f"After potentiation (dt=-3): {w:.4f}")
    w = stdp.update_weight(w, dt=5)   # post before pre -> depress
    print(f"After depression (dt=+5): {w:.4f}")


if __name__ == "__main__":
    main()
