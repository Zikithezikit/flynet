"""Tests for flynet.neurons module."""

import numpy as np
import pytest
from flynet.neurons import LIFNeuron, NeuronModel, SpikingNeuron


class TestLIFNeuron:
    def test_rest_potential(self):
        n = LIFNeuron()
        assert n.get_potential() == -70.0

    def test_no_spike_at_rest(self):
        n = LIFNeuron()
        assert n.step(0.0, dt=1.0) is False
        assert n.get_potential() == -70.0

    def test_spike_with_sufficient_current(self):
        n = LIFNeuron(v_thresh=-55.0)
        # Apply strong current for enough time to reach threshold
        for _ in range(20):
            n.step(5.0, dt=1.0)
        # Should have spiked by now
        assert n.get_potential() == -75.0  # reset to v_reset

    def test_refractory_period(self):
        n = LIFNeuron(v_thresh=-55.0, t_refrac=3.0)
        # Force a spike
        for _ in range(20):
            if n.step(5.0, dt=1.0):
                break
        # During refractory period, should not spike
        spikes = [n.step(10.0, dt=1.0) for _ in range(2)]
        assert not any(spikes)

    def test_reset(self):
        n = LIFNeuron()
        n.step(5.0, dt=1.0)
        n.reset()
        assert n.get_potential() == -70.0

    def test_is_neuron_model(self):
        assert issubclass(LIFNeuron, NeuronModel)


class TestSpikingNeuron:
    def test_rest_potential(self):
        n = SpikingNeuron()
        assert n.get_potential() == 0.0

    def test_spike_at_threshold(self):
        n = SpikingNeuron(threshold=5.0)
        for _ in range(10):
            n.step(3.0, dt=1.0)
        # Should spike eventually

    def test_refractory_counter(self):
        n = SpikingNeuron(threshold=2.0, refrac_time=3)
        # Force spike by setting potential well above threshold
        n._potential = 5.0
        n.step(0.0, dt=1.0)
        assert n._refrac_counter == 3

    def test_inhibit(self):
        n = SpikingNeuron(min_pot=-500)
        n.step(10.0, dt=1.0)
        n.inhibit()
        assert n.get_potential() == -500

    def test_leak(self):
        n = SpikingNeuron(leak=0.15)
        n.step(1.0, dt=1.0)  # P = 0 + 1 = 1, then 1 > 0 so P -= 0.15
        assert abs(n.get_potential() - 0.85) < 0.01
