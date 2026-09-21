"""Tests for flynet.stdp module."""

import numpy as np
import pytest
from flynet.stdp import STDPRule, RewardModulatedSTDP


class TestSTDPRule:
    def test_potentiation(self):
        stdp = STDPRule()
        # Pre before post (dt < 0) -> potentiation (positive delta)
        curve = stdp.curve(-5)
        assert curve > 0

    def test_depression(self):
        stdp = STDPRule()
        # Post before pre (dt > 0) -> depression (negative delta)
        curve = stdp.curve(5)
        assert curve < 0

    def test_weight_bounds(self):
        stdp = STDPRule(w_min=-1.2, w_max=1.5)
        w = 1.4
        w_new = stdp.update_weight(w, dt=-1)
        assert w_new <= 1.5
        w_new = stdp.update_weight(-1.1, dt=1)
        assert w_new >= -1.2

    def test_spike_pair(self):
        stdp = STDPRule()
        w = stdp.apply_spike_pair(0.5, pre_spike_time=10, post_spike_time=13)
        assert isinstance(w, float)

    def test_decay_with_distance(self):
        stdp = STDPRule()
        d1 = abs(stdp.curve(-2))
        d2 = abs(stdp.curve(-10))
        assert d1 > d2


class TestRewardModulatedSTDP:
    def test_eligibility_trace(self):
        rm = RewardModulatedSTDP(n_synapses=10)
        rm.update_eligibility(0, dt=-3)
        rm.update_eligibility(0, dt=-3)
        assert rm.eligibility_traces[0] != 0

    def test_apply_reward(self):
        rm = RewardModulatedSTDP(n_synapses=10)
        rm.update_eligibility(0, dt=-3)
        deltas = rm.apply_reward(reward=1.0, learning_rate=0.5)
        assert deltas.shape[0] == 10

    def test_eligibility_resets_after_reward(self):
        rm = RewardModulatedSTDP(n_synapses=10)
        rm.update_eligibility(0, dt=-3)
        rm.apply_reward(reward=1.0)
        assert rm.eligibility_traces[0] == 0
