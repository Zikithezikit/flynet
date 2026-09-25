"""Tests for flynet.stdp module."""

import numpy as np
import pytest
from flynet.stdp import STDPRule, RewardModulatedSTDP


class TestSTDPRule:
    def test_potentiation(self):
        stdp = STDPRule()
        # Pre before post (dt = t_post - t_pre > 0) -> potentiation
        curve = stdp.curve(5)
        assert curve > 0

    def test_depression(self):
        stdp = STDPRule()
        # Post before pre (dt = t_post - t_pre < 0) -> depression
        curve = stdp.curve(-5)
        assert curve < 0

    def test_ltd_decreases_weight(self):
        """Regression: depression must lower weights across the whole range."""
        stdp = STDPRule(w_min=-1.2, w_max=1.5)
        for w in (-1.1, 0.0, 0.5, 1.0, 1.4):
            new_w = stdp.update_weight(w, dt=-5)
            assert new_w < w, f"LTD increased weight at w={w}: {w} -> {new_w}"

    def test_ltp_increases_weight(self):
        stdp = STDPRule(w_min=-1.2, w_max=1.5)
        for w in (-1.1, 0.0, 0.5, 1.0, 1.4):
            new_w = stdp.update_weight(w, dt=5)
            assert new_w > w, f"LTP decreased weight at w={w}: {w} -> {new_w}"

    def test_weight_bounds(self):
        stdp = STDPRule(w_min=-1.2, w_max=1.5)
        w_new = stdp.update_weight(1.45, dt=100)
        assert w_new <= 1.5
        w_new = stdp.update_weight(-1.15, dt=-100)
        assert w_new >= -1.2

    def test_spike_pair(self):
        stdp = STDPRule()
        w = stdp.apply_spike_pair(0.5, pre_spike_time=10, post_spike_time=13)
        assert isinstance(w, float)
        # Pre (10) before post (13) -> potentiation
        assert w > 0.5
        w = stdp.apply_spike_pair(0.5, pre_spike_time=13, post_spike_time=10)
        # Post (10) before pre (13) -> depression
        assert w < 0.5

    def test_decay_with_distance(self):
        stdp = STDPRule()
        d1 = abs(stdp.curve(2))
        d2 = abs(stdp.curve(10))
        assert d1 > d2


class TestRewardModulatedSTDP:
    def test_eligibility_trace(self):
        rm = RewardModulatedSTDP(n_synapses=10)
        rm.update_eligibility(0, dt=3)
        rm.decay_eligibility()
        rm.update_eligibility(0, dt=3)
        assert rm.eligibility_traces[0] != 0

    def test_apply_reward(self):
        rm = RewardModulatedSTDP(n_synapses=10)
        rm.update_eligibility(0, dt=3)
        deltas = rm.apply_reward(reward=1.0, learning_rate=0.5)
        assert deltas.shape[0] == 10

    def test_eligibility_resets_after_reward(self):
        rm = RewardModulatedSTDP(n_synapses=10)
        rm.update_eligibility(0, dt=3)
        rm.apply_reward(reward=1.0)
        assert rm.eligibility_traces[0] == 0

    def test_decay_applies_once_per_call(self):
        """Regression: decay must be an explicit per-timestep operation."""
        rm = RewardModulatedSTDP(
            n_synapses=10, eligibility_trace_decay=0.5
        )
        rm.update_eligibility(0, dt=3)
        expected = rm.eligibility_traces[0]
        rm.update_eligibility(1, dt=3)  # must NOT decay synapse 0's trace
        assert rm.eligibility_traces[0] == expected
        rm.decay_eligibility()
        assert rm.eligibility_traces[0] == pytest.approx(expected * 0.5)
