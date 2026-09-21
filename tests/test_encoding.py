"""Tests for flynet.encoding module."""

import numpy as np
import pytest
from flynet.encoding import rate_encode, SpikeTrain, poisson_encode


class TestRateEncode:
    def test_shape(self):
        potentials = np.array([0.5, 1.0, 0.0])
        train = rate_encode(potentials, t_steps=100)
        assert train.shape == (3, 101)

    def test_binary_output(self):
        potentials = np.array([0.5, 1.0])
        train = rate_encode(potentials, t_steps=200)
        assert set(np.unique(train)).issubset({0, 1})

    def test_higher_potential_more_spikes(self):
        potentials = np.array([0.1, 0.9])
        train = rate_encode(potentials, t_steps=1000)
        rates = train.mean(axis=1)
        assert rates[1] > rates[0]

    def test_zero_potential(self):
        potentials = np.array([0.0, 0.0])
        train = rate_encode(potentials, t_steps=100)
        assert train.sum() == 0

    def test_2d_input(self):
        potentials = np.ones((4, 4))
        train = rate_encode(potentials, t_steps=50)
        assert train.shape == (16, 51)


class TestSpikeTrain:
    def test_properties(self):
        data = np.random.randint(0, 2, (5, 100))
        st = SpikeTrain(data)
        assert st.n_neurons == 5
        assert st.n_timesteps == 99

    def test_spike_times(self):
        data = np.zeros((2, 5), dtype=int)
        data[0, 2] = 1
        data[0, 4] = 1
        st = SpikeTrain(data)
        assert st.spike_times(0) == [2, 4]
        assert st.spike_times(1) == []

    def test_rates(self):
        data = np.ones((3, 10), dtype=int)
        st = SpikeTrain(data)
        np.testing.assert_array_equal(st.rates(), np.ones(3))


class TestPoissonEncode:
    def test_shape(self):
        rates = np.array([10.0, 20.0])
        train = poisson_encode(rates, t_steps=100)
        assert train.shape == (2, 101)

    def test_higher_rate_more_spikes(self):
        rates = np.array([5.0, 50.0])
        train = poisson_encode(rates, t_steps=1000)
        counts = train.sum(axis=1)
        assert counts[1] > counts[0]
